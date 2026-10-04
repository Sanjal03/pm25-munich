"""
04_phase1_benchmark.py — Phase 1: Model Comparison (Benchmark)
===============================================================
Runs all 5 models (LightGBM, XGBoost, CatBoost, DLinear, PatchTST) + the
Persistence baseline, on all 7 stations, evaluated on
the Validity Gate window (15 Jan 00:00 – 21 Mar 23:00 2020, 1608 h).

WHY THE VALIDITY GATE WINDOW, AND NOT THE STANDARD 2024 TEST SET
------------------------------------------------------------------
Each model is trained on the training split (2019-2023) with the gate
window removed, and evaluated ONLY on the gate window — a held-out,
ordinary (non-lockdown) period. 2024 is never used for training here.
Two things fall out of one run:

  1. A fair Phase-1 leaderboard (R², MAE, Bias) for selecting the
     "Champion" model(s) to carry into RQ1 and RQ2.
  2. The exact same numbers ARE the RQ2 Validity Gate check — a model
     that cannot reproduce ordinary, pre-lockdown PM2.5 behaviour on
     data it has not seen has no business being trusted to estimate
     what PM2.5 "would have been" during lockdown. Reusing this result
     in Phase 3 avoids training every model twice.

  A model "passes the gate" if R² >= 0.80 AND |Bias| < 1.5 µg/m³ on
  this window (the supervisor's methodological requirement).

Output: results/phase1_benchmark.csv (station, model, R2, MAE, RMSE,
Bias, n_test, passes_gate)
"""

import warnings
from pathlib import Path

import pandas as pd

from model_factory import (
    ALL_MODELS, GATE_START, GATE_END, load_station_data, tabular_feature_cols,
    train_and_eval,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

STATIONS = [
    "landshuter_allee", "stachus", "lothstrasse", "johanneskirchen",
    "augsburg_bourges", "augsburg_lfu", "andechs",
]

GATE_R2_MIN  = 0.80
GATE_BIAS_MAX = 1.5

MODELS_TO_RUN = ALL_MODELS + ["persistence"]


def run_phase1(stations=STATIONS, models=MODELS_TO_RUN) -> pd.DataFrame:
    rows = []
    for station in stations:
        print(f"\n{'='*70}\n  Phase 1 | Station: {station}\n{'='*70}")
        df = load_station_data(station)
        feature_cols = tabular_feature_cols(df)

        for model_type in models:
            try:
                m = train_and_eval(
                    model_type, df, feature_cols=feature_cols,
                    date_range=(GATE_START, GATE_END),
                    exclude_range=(GATE_START, GATE_END),
                )
                passes = (m["R2"] >= GATE_R2_MIN) and (abs(m["Bias"]) < GATE_BIAS_MAX)
                rows.append({
                    "station": station, "model": model_type,
                    "R2": m["R2"], "MAE": m["MAE"], "RMSE": m["RMSE"],
                    "Bias": m["Bias"], "n_test": m["n_test"],
                    "passes_gate": passes,
                })
                flag = "PASS" if passes else "FAIL"
                print(f"  {model_type:<12} R2={m['R2']:.4f}  MAE={m['MAE']:.3f}  "
                      f"Bias={m['Bias']:+.3f}  [{flag}]")
            except Exception as e:
                print(f"  {model_type:<12} ✗ ERROR: {e}")
                rows.append({
                    "station": station, "model": model_type,
                    "R2": None, "MAE": None, "RMSE": None, "Bias": None,
                    "n_test": None, "passes_gate": False, "error": str(e),
                })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "phase1_benchmark.csv", index=False)
    print(f"\n✓ Saved: {RESULTS_DIR / 'phase1_benchmark.csv'}")
    return out


def rank_models(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Ranks models by mean R2 across stations (gate-passing only) — used
    to pick the Champion for Phase 2 and the top-3 for Phase 3."""
    valid = benchmark_df.dropna(subset=["R2"])
    ranking = (valid.groupby("model")
               .agg(mean_R2=("R2", "mean"), mean_MAE=("MAE", "mean"),
                    n_stations_passed=("passes_gate", "sum"))
               .sort_values("mean_R2", ascending=False))
    ranking.to_csv(RESULTS_DIR / "phase1_ranking.csv")
    print("\n── Overall ranking (mean R² across stations, gate window) ──")
    print(ranking.to_string())
    return ranking


if __name__ == "__main__":
    bench = run_phase1()
    rank_models(bench)
