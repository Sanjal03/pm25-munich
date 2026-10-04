"""
patchtst_epoch_test.py
================================================================================
Tests whether PatchTST-lite is undertrained at its current n_epochs=30
default -- checks n_epochs=100 against the existing Phase 1 baseline, same
protocol (validity gate window, 7 stations), without touching
model_factory.py's shared get_model() default (so this can't affect any
other result in the repo).
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from model_factory import (
    PatchTSTLite, LOOKBACK, GATE_START, GATE_END, build_sequence_frame,
    split_sequence, compute_metrics, load_station_data,
)

warnings.filterwarnings("ignore")

STATIONS = [
    "landshuter_allee", "stachus", "lothstrasse", "johanneskirchen",
    "augsburg_bourges", "augsburg_lfu", "andechs",
]
N_EPOCHS_TEST = 100

rows = []
for station in STATIONS:
    print(f"\n{'='*70}\n  PatchTST epoch test | {station} | n_epochs={N_EPOCHS_TEST}\n{'='*70}")
    df = load_station_data(station)
    seq_df = build_sequence_frame(df, lookback=LOOKBACK)
    Xtr, Etr, ytr, Xte, Ete, yte = split_sequence(
        seq_df, date_range=(GATE_START, GATE_END), exclude_range=(GATE_START, GATE_END))

    model = PatchTSTLite(lookback=LOOKBACK, n_epochs=N_EPOCHS_TEST)
    model.fit(Xtr.values, ytr.values, exo=Etr.values)
    y_pred = model.predict(Xte.values, exo=Ete.values)

    m = compute_metrics(yte, y_pred)
    m["n_test"] = len(yte)
    rows.append({"station": station, **m})
    print(f"  R2={m['R2']:.4f}  MAE={m['MAE']:.3f}  Bias={m['Bias']:+.3f}")

out = pd.DataFrame(rows)
out.to_csv("results/patchtst_epoch100_test.csv", index=False)

baseline = pd.read_csv("results/phase1_benchmark.csv")
baseline_patchtst = baseline[baseline["model"] == "patchtst"][["station", "R2", "MAE", "Bias"]]

print("\n" + "=" * 70)
print(" COMPARISON: n_epochs=30 (existing) vs n_epochs=100 (this test)")
print("=" * 70)
merged = out[["station", "R2", "MAE"]].merge(
    baseline_patchtst, on="station", suffixes=("_100ep", "_30ep"))
merged["R2_delta"] = (merged["R2_100ep"] - merged["R2_30ep"]).round(4)
print(merged.to_string(index=False))
print(f"\nMean R2 @ 30 epochs:  {baseline_patchtst['R2'].mean():.4f}")
print(f"Mean R2 @ 100 epochs: {out['R2'].mean():.4f}")
