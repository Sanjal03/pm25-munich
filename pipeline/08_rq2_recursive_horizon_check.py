"""
08_rq2_recursive_horizon_check.py — RQ2 Robustness Check: Recursive Rollout
Error Accumulation Over a Matched Horizon
=============================================================================
Motivation (supervisor feedback, Sep 2026): the Validity Gate only certifies
ONE-STEP-AHEAD accuracy. RQ2's actual lockdown counterfactual is a 44-day
(1056h) RECURSIVE rollout, where every hour's prediction feeds the next
hour's PM2.5 lag/rolling features. A model can pass the one-step gate and
still drift badly once its own errors start compounding — one-step accuracy
says nothing directly about 44-day recursive reliability.

This script answers that directly: it takes the SAME trained model used for
each station's real RQ2 lockdown rollout (identical training data — split
=="train" i.e. 2019-2023, minus the gate+lockdown exclusion window) and runs
the SAME recursive_counterfactual_forecast mechanism over three 1056-hour
windows inside the 2024 test period (genuinely unseen in training, confirmed
via split_tabular/split_sequence, which always filter to split=="train"
first). Because 2024 is a normal year, actual observations ARE the correct
"ground truth" for these windows (unlike the real lockdown window, where a
gap is the whole point) — so any divergence between the recursive rollout
and reality here is pure model/rollout error, not a real external effect.
This isolates how much of the real RQ2 gap could plausibly be explained by
accumulated rollout drift alone, vs a genuine lockdown effect.

Windows (each exactly matches the real lockdown's 1056-hour length, 00:00
on the first day to 23:00 on the last):
  - Spring: 2024-03-22 -> 2024-05-04  (same calendar dates as the real
    lockdown, four years later — controls for season)
  - Summer: 2024-07-01 -> 2024-08-13
  - Autumn: 2024-10-01 -> 2024-11-13

For each (station, model, window), computes hourly absolute error, then
aggregates into daily MAE across the 44-day horizon — this is the error
accumulation curve. Output:
  - results/rq2_horizon_{station}_{model}.csv   (per-hour pred/actual/error,
    all 3 windows, one file per station x model)
  - results/rq2_horizon_daily_summary.csv        (day-by-day MAE, aggregated
    across all stations/models/windows — the headline curve)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from model_factory import (
    GATE_START, LOCKDOWN_START, LOCKDOWN_END,
    load_station_data, tabular_feature_cols, train_and_eval,
    recursive_counterfactual_forecast,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

STATIONS = [
    "landshuter_allee", "stachus", "lothstrasse", "johanneskirchen",
    "augsburg_bourges", "augsburg_lfu", "andechs",
]
MODELS = ["catboost", "xgboost", "lightgbm", "dlinear", "patchtst"]

# Must match 06_phase3_rq2_counterfactual.py exactly — same excluded window,
# so the model trained here IS the same model (up to random seed, which is
# fixed) as the one that produced the real lockdown gap.
EXCLUDE_RANGE = (GATE_START, LOCKDOWN_END)

N_HOURS = len(pd.date_range(LOCKDOWN_START, LOCKDOWN_END, freq="h"))  # 1056
WINDOWS = {
    "spring_2024": "2024-03-22",
    "summer_2024": "2024-07-01",
    "autumn_2024": "2024-10-01",
}


def run_one(station: str, model_type: str) -> pd.DataFrame:
    df = load_station_data(station)
    feature_cols = tabular_feature_cols(df)

    # Reuses the identical training procedure as the real RQ2 gate step
    # (same exclude_range), so this is the same model the real lockdown
    # gap was computed with — apples-to-apples, not a different fit.
    fit = train_and_eval(
        model_type, df, feature_cols=feature_cols,
        exclude_range=EXCLUDE_RANGE, return_model=True,
    )
    model = fit["_model"]

    rows = []
    for label, start in WINDOWS.items():
        end = pd.date_range(start, periods=N_HOURS, freq="h")[-1]
        y_pred = recursive_counterfactual_forecast(
            model, df, feature_cols, start, str(end), model_type=model_type)
        y_true = df.loc[start:end, "target"].reindex(y_pred.index)

        window_df = pd.DataFrame({
            "station": station, "model": model_type, "window": label,
            "hour_index": range(len(y_pred)),
            "day_index": [h // 24 for h in range(len(y_pred))],
            "pred": y_pred.values, "actual": y_true.values,
        })
        window_df["abs_error"] = (window_df["pred"] - window_df["actual"]).abs()
        rows.append(window_df)

    return pd.concat(rows, ignore_index=True)


def run_all():
    all_dfs = []
    for model_type in MODELS:
        for station in STATIONS:
            print(f"[{model_type}] {station} ...", end=" ", flush=True)
            try:
                out = run_one(station, model_type)
                out_path = RESULTS_DIR / f"rq2_horizon_{station}_{model_type}.csv"
                out.to_csv(out_path, index=False)
                all_dfs.append(out)
                end_day_mae = out[out["day_index"] == out["day_index"].max()]["abs_error"].mean()
                print(f"ok (final-day MAE={end_day_mae:.2f})")
            except Exception as e:
                print(f"ERROR: {e}")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(RESULTS_DIR / "rq2_horizon_all_raw.csv", index=False)

    # Daily MAE aggregated across every station/model/window at each day
    # of the rollout — this is the headline error-accumulation curve.
    daily = (combined.groupby("day_index")["abs_error"]
             .agg(["mean", "std", "median"]).reset_index()
             .rename(columns={"mean": "MAE", "std": "MAE_std", "median": "MAE_median"}))
    # Cumulative MAE: mean absolute error over [0, day] — shows whether
    # error is stabilising or still climbing by the end of the horizon.
    cum = []
    for d in daily["day_index"]:
        cum.append(combined.loc[combined["day_index"] <= d, "abs_error"].mean())
    daily["cumulative_MAE"] = cum
    daily.to_csv(RESULTS_DIR / "rq2_horizon_daily_summary.csv", index=False)

    print(f"\n✓ Day-0 MAE: {daily['MAE'].iloc[0]:.3f} µg/m³")
    print(f"✓ Day-{daily['day_index'].max()} (final) MAE: {daily['MAE'].iloc[-1]:.3f} µg/m³")
    print(f"✓ Full-horizon cumulative MAE: {daily['cumulative_MAE'].iloc[-1]:.3f} µg/m³")
    print(f"✓ Saved: results/rq2_horizon_daily_summary.csv, "
          f"results/rq2_horizon_all_raw.csv, "
          f"and {len(STATIONS)*len(MODELS)} per-combo CSVs")
    return daily


if __name__ == "__main__":
    run_all()
