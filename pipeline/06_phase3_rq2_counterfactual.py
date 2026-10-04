"""
06_phase3_rq2_counterfactual.py — Phase 3: RQ2 COVID-19 Counterfactual
========================================================================
RQ2: Can a counterfactual model — gated by a mandatory validity check —
quantify the COVID-19 lockdown effect on Munich PM2.5?

For the Champion model and the next two best models from Phase 1
(by mean gate-window R²), for each of the 7 stations:

  Step 1 — VALIDITY GATE (mandatory)
    The model is trained on 2019-2023 EXCLUDING 15 Jan - 4 May 2020 (gate
    + lockdown), then evaluated on the gate window (15 Jan 00:00 - 21 Mar
    23:00 2020) — an ordinary, non-lockdown period
    it has never seen. If R² < 0.80 or |Bias| >= 1.5 µg/m³, the model
    is NOT trusted to extrapolate into the lockdown window and the gap
    is reported as unavailable (never silently reported anyway).
    This gate exists because a counterfactual is only as credible as
    the model's demonstrated ability to reproduce known-normal
    behaviour on unseen data — without it, any gap you find could just
    be model error, not a real lockdown effect.

  Step 2 — LOCKDOWN COUNTERFACTUAL (only if gate passed) — RECURSIVE ROLLOUT
    Using the SAME trained model, roll forward hour-by-hour through the
    lockdown window (22 Mar 00:00 - 4 May 23:00 2020, 1056 h). Only the
    history up to the lockdown start is real. From the first lockdown hour
    onward, every PM2.5 lag/rolling feature is rebuilt from the model's
    OWN previous predictions, not from the real (already lockdown-affected)
    observed series — feeding real values here would anchor predictions
    to the actual outcome via the dominant lag features (RQ1: removing
    them raises MAE far more than any other group) and mechanically
    shrink any real counterfactual gap toward zero.
    Weather, co-pollutants (PM10, NO2, O3, pm25_background) and temporal
    features stay at their observed values. The co-pollutants were
    themselves affected by lockdown, so the counterfactual is partly
    anchored to the lockdown outcome: the reported gap is a conservative
    (lower-bound) estimate. Errors compound over the rollout; 08 and 10
    quantify and correct for that drift.

Output: results/rq2_{station}_{model}.json — one file per
station x model, consumed by 07_make_figures.py and 10_rq2_drift_correction.py.
"""

import json
import warnings
from pathlib import Path

import pandas as pd

from model_factory import (
    GATE_START, GATE_END, LOCKDOWN_START, LOCKDOWN_END,
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

GATE_R2_MIN, GATE_BIAS_MAX   = 0.80, 1.5

# Excluding BOTH the gate window and the lockdown window from training
# keeps the gate check honest (model never saw either period) while
# still letting one trained model serve both checks.
EXCLUDE_RANGE = (GATE_START, LOCKDOWN_END)


def pick_rq2_models(ranking_csv: Path = RESULTS_DIR / "phase1_ranking.csv",
                     n=3) -> list[str]:
    if not ranking_csv.exists():
        print("  ⚠ phase1_ranking.csv not found — defaulting to "
              "[lightgbm, xgboost, catboost]. Run 04_phase1_benchmark.py first.")
        return ["lightgbm", "xgboost", "catboost"]
    ranking = pd.read_csv(ranking_csv, index_col=0)
    top = list(ranking.index[:n])
    print(f"  Using top {n} models for RQ2: {top}")
    return top


def run_rq2_experiment(station: str, model_type: str) -> dict:
    df = load_station_data(station)
    feature_cols = tabular_feature_cols(df)

    # ── Step 1: validity gate (one-step-ahead, real data — this is correct
    #    as-is: we're testing forecast accuracy on unseen NORMAL data, not
    #    simulating a counterfactual, so real lag values are appropriate
    #    here and nowhere else in this function) ─────────────────────────
    gate = train_and_eval(
        model_type, df, feature_cols=feature_cols,
        date_range=(GATE_START, GATE_END), exclude_range=EXCLUDE_RANGE,
        return_model=True,
    )
    passes_gate = (gate["R2"] >= GATE_R2_MIN) and (abs(gate["Bias"]) < GATE_BIAS_MAX)
    model = gate["_model"]  # reuse the exact same trained model for the rollout below —
                            # do not retrain, or the gate check no longer certifies
                            # the model actually used for the counterfactual

    print(f"\n  [{model_type}] Validity gate: R2={gate['R2']:.4f} "
          f"Bias={gate['Bias']:+.3f} → {'VALID' if passes_gate else 'NOT VALID'}")

    result = {
        "station": station, "model": model_type,
        "sanity_valid": bool(passes_gate),
        "sanity_R2": gate["R2"], "sanity_gap": gate["Bias"],
    }

    if not passes_gate:
        print(f"  ⚠ {model_type} fails the validity gate at {station} — "
              f"lockdown gap NOT reported.")
        result.update({"gap": None, "gap_pct": None,
                        "mean_counterfactual": None, "mean_actual": None})
        out_path = RESULTS_DIR / f"rq2_{station}_{model_type}.json"
        out_path.write_text(json.dumps(result, indent=2))
        return result

    # ── Step 2: lockdown counterfactual — RECURSIVE rollout ───────────────
    # Only pre-lockdown PM2.5 history is real. From 22 Mar 2020 onward,
    # every PM2.5 lag/rolling feature is rebuilt from the model's own
    # previous predictions (see recursive_counterfactual_forecast
    # docstring for why this matters). Weather, co-pollutants, and
    # temporal features stay at their true observed values.
    y_pred = recursive_counterfactual_forecast(
        model, df, feature_cols, LOCKDOWN_START, LOCKDOWN_END, model_type=model_type)
    y_true = df.loc[LOCKDOWN_START:LOCKDOWN_END, "target"]
    y_true = y_true.reindex(y_pred.index)

    mean_pred = float(y_pred.mean())
    mean_actual = float(y_true.mean())
    gap = mean_pred - mean_actual
    gap_pct = gap / mean_actual * 100 if mean_actual else None

    print(f"  Mean predicted (counterfactual, recursive rollout): {mean_pred:.2f} µg/m³")
    print(f"  Mean actual (observed):                             {mean_actual:.2f} µg/m³")
    print(f"  Gap: {gap:+.2f} µg/m³ ({gap_pct:+.1f}%)")

    result.update({
        "mean_counterfactual": round(mean_pred, 2),
        "mean_actual": round(mean_actual, 2),
        "gap": round(gap, 2),
        "gap_pct": round(gap_pct, 1) if gap_pct is not None else None,
        "method": "recursive_rollout",
    })
    out_path = RESULTS_DIR / f"rq2_{station}_{model_type}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  ✓ Saved: {out_path}")
    return result


def run_all(stations=STATIONS, models=None):
    models = models or pick_rq2_models()
    all_results = []
    for model_type in models:
        for station in stations:
            print(f"\n{'='*70}\n  Phase 3 (RQ2) | Station: {station} | Model: {model_type}\n{'='*70}")
            try:
                all_results.append(run_rq2_experiment(station, model_type))
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                all_results.append({"station": station, "model": model_type,
                                     "sanity_valid": False, "error": str(e)})
    summary = pd.DataFrame(all_results)
    summary.to_csv(RESULTS_DIR / "phase3_rq2_summary.csv", index=False)
    print(f"\n✓ Phase 3 complete — {len(summary)} runs — "
          f"results/phase3_rq2_summary.csv")
    return summary


if __name__ == "__main__":
    run_all()