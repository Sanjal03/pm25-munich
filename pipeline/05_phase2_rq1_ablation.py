"""
05_phase2_rq1_ablation.py — Phase 2: RQ1 Feature Group Ablation
=================================================================
RQ1: Which feature groups (PM2.5_lags, PM2.5_rolling, Co-pollutants,
Meteorology, Temporal) contribute most to PM2.5 prediction?

Method: Leave-One-Group-Out. Train once with the full feature set
(baseline), then once per feature group with that group's columns
removed. The ΔMAE% when a group is removed is the group's importance —
a bigger jump means the model relied on it more.

Uses the standard chronological split (train 2019-2023, test 2024) —
NOT the validity gate window — because RQ1 is about everyday predictive
performance, not lockdown-period trustworthiness.

MODEL CHOICE
------------
This script runs the Phase 1 Champion; _rq1_remaining_models.py runs the
other four models with the same function. If the Champion were a
sequence model, the best-ranked tabular model is used here instead (the
fallback never triggered: CatBoost is the Champion). Sequence models are
ablated through their exogenous array and a shuffled lookback window —
see run_rq1_experiment().

Output: results/rq1_{station}_{model}.csv (Configuration, R2, MAE, RMSE,
Bias, delta_MAE_pct) — one row per feature-group configuration.
"""

import warnings
from pathlib import Path

import pandas as pd

from model_factory import (
    FEATURE_GROUPS, TABULAR_MODELS, SEQUENCE_MODELS, SEQUENCE_EXO_COLS,
    load_station_data, tabular_feature_cols, train_and_eval,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

STATIONS = [
    "landshuter_allee", "stachus", "lothstrasse", "johanneskirchen",
    "augsburg_bourges", "augsburg_lfu", "andechs",
]


def pick_rq1_model(ranking_csv: Path = RESULTS_DIR / "phase1_ranking.csv") -> str:
    if not ranking_csv.exists():
        print("  ⚠ phase1_ranking.csv not found — defaulting to lightgbm. "
              "Run 04_phase1_benchmark.py first for a data-driven choice.")
        return "lightgbm"
    ranking = pd.read_csv(ranking_csv, index_col=0)
    top_model = ranking.index[0]
    if top_model not in TABULAR_MODELS:
        tabular_ranking = ranking.loc[ranking.index.intersection(TABULAR_MODELS)]
        fallback = tabular_ranking["mean_R2"].idxmax()
        print(f"  Champion '{top_model}' is a sequence model — RQ1 ablation "
              f"requires tabular feature groups, using best tabular model "
              f"instead: '{fallback}'")
        return fallback
    print(f"  Using Champion model for RQ1: '{top_model}'")
    return top_model


def run_rq1_experiment(station: str, model_type: str) -> pd.DataFrame:
    """
    TABULAR models (lightgbm/xgboost/catboost): ablation removes columns
    from the engineered feature table directly, via `feature_cols`.

    SEQUENCE models (dlinear/patchtst) don't consume that table at all —
    they take a raw 168h PM2.5 lookback window + a separate exogenous
    array (see train_and_eval / build_sequence_frame). So the same 5
    FEATURE_GROUPS are ablated differently for this family:
      - Co-pollutants / Meteorology / Temporal: restrict the exogenous
        array via `sequence_exo_cols` (the direct sequence-family
        equivalent of dropping those tabular columns).
      - PM2.5_lags: the lookback window IS the model's PM2.5-history
        signal, so this is where that gets removed — via
        `shuffle_lookback` (see split_sequence docstring for why
        shuffling, not zeroing).
      - PM2.5_rolling: sequence models have no separate rolling-stat
        representation to remove (the raw window already covers whatever
        the model derives from it) — reported as not-applicable (NaN)
        rather than faking a number, matching the earlier discovery that
        running this configuration unmodified silently no-ops.
    """
    is_sequence = model_type in SEQUENCE_MODELS
    df = load_station_data(station)
    all_cols = tabular_feature_cols(df)

    rows = []
    baseline = train_and_eval(model_type, df, feature_cols=all_cols)
    rows.append({"Configuration": "All features", "delta_MAE_pct": 0.0, **baseline})
    print(f"  All features        MAE={baseline['MAE']:.3f}  R2={baseline['R2']:.4f}")

    for group_name, group_cols in FEATURE_GROUPS.items():
        if is_sequence:
            if group_name == "PM2.5_rolling":
                rows.append({"Configuration": f"Without {group_name}",
                              "delta_MAE_pct": float("nan"),
                              "note": "not applicable to sequence models"})
                print(f"  Without {group_name:<15} N/A (no separate rolling-stat "
                      f"representation for this model family)")
                continue
            elif group_name == "PM2.5_lags":
                m = train_and_eval(model_type, df, shuffle_lookback=True)
            else:
                restricted_exo = [c for c in SEQUENCE_EXO_COLS if c not in group_cols]
                m = train_and_eval(model_type, df, sequence_exo_cols=restricted_exo)
        else:
            reduced_cols = [c for c in all_cols if c not in group_cols]
            m = train_and_eval(model_type, df, feature_cols=reduced_cols)

        delta_pct = (m["MAE"] - baseline["MAE"]) / baseline["MAE"] * 100
        rows.append({"Configuration": f"Without {group_name}",
                      "delta_MAE_pct": round(delta_pct, 2), **m})
        print(f"  Without {group_name:<15} MAE={m['MAE']:.3f}  "
              f"ΔMAE={delta_pct:+.1f}%")

    out = pd.DataFrame(rows)
    for c in ["_model", "_y_pred", "_y_test", "_test_index"]:
        if c in out.columns:
            out = out.drop(columns=c)
    out.to_csv(RESULTS_DIR / f"rq1_{station}_{model_type}.csv", index=False)
    return out


def run_all(stations=STATIONS):
    model_type = pick_rq1_model()
    for station in stations:
        print(f"\n{'='*70}\n  Phase 2 (RQ1 ablation) | Station: {station} | Model: {model_type}\n{'='*70}")
        run_rq1_experiment(station, model_type)
    print(f"\n✓ RQ1 ablation complete — results/rq1_*_{model_type}.csv")
    return model_type


if __name__ == "__main__":
    run_all()