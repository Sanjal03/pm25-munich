"""
10_rq2_drift_correction.py — RQ2 Drift Correction of the Lockdown Gap
=====================================================================
Input:  results/rq2_horizon_all_raw.csv   (from 08_rq2_recursive_horizon_check.py)
        results/rq2_{station}_{model}.json (from 06 + _rq2_remaining_models.py)
Output: results/rq2_horizon_pseudo_gaps.csv
        results/rq2_bias_corrected.csv

08 runs the exact RQ2 models + recursive rollout over three 1056-hour
windows in 2024 (the lockdown's length), where no intervention happened. Any mean offset between
the rollout and the observed series there is pure rollout drift — a
"pseudo-gap". This script subtracts each (station, model)'s own average
pseudo-gap from its real lockdown gap:

    pseudo_gap[w]   = mean(pred - actual) over window w (paired hours only)
    baseline_bias   = mean of pseudo_gap over the 2024 windows
    corrected_gap   = gap - baseline_bias

A lockdown effect "survives" correction if corrected_gap > 0.

Paired hours: DLinear/PatchTST return NaN predictions when a real >24h
target gap falls inside their 168h lookback (tabular models handle NaN
natively). pred and actual are averaged over the same hours, otherwise
mean(pred) and mean(actual) would cover different periods. A window that
is entirely NaN (Lothstraße spring 2024 for the sequence models) is
dropped from that combo's baseline_bias average. The lockdown window
itself has no target gaps at any station, so the real gaps from 06 are
unaffected.
"""

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")

STATION_LABELS = {
    "augsburg_lfu": "Augsburg/LfU", "andechs": "Andechs/Rothenfeld",
    "augsburg_bourges": "Augsburg/Bourges", "lothstrasse": "Lothstraße",
    "landshuter_allee": "Landshuter Allee", "stachus": "Stachus",
    "johanneskirchen": "Johanneskirchen",
}


def load_lockdown_gaps() -> pd.DataFrame:
    rows = []
    for f in sorted(RESULTS_DIR.glob("rq2_*.json")):
        r = json.loads(f.read_text())
        if r.get("gap") is not None:
            rows.append({k: r[k] for k in ("station", "model", "gap", "gap_pct")})
    return pd.DataFrame(rows)


def run():
    raw = pd.read_csv(RESULTS_DIR / "rq2_horizon_all_raw.csv")
    paired = raw.dropna(subset=["pred", "actual"])
    pseudo = (paired.assign(err=paired["pred"] - paired["actual"])
              .groupby(["station", "model", "window"])["err"].mean()
              .rename("pseudo_gap").reset_index())
    pseudo.to_csv(RESULTS_DIR / "rq2_horizon_pseudo_gaps.csv", index=False)

    baseline = (pseudo.groupby(["station", "model"])["pseudo_gap"].mean()
                .rename("baseline_bias").reset_index())
    out = load_lockdown_gaps().merge(baseline, on=["station", "model"], how="left")
    out["corrected_gap"] = out["gap"] - out["baseline_bias"]
    out.to_csv(RESULTS_DIR / "rq2_bias_corrected.csv", index=False)

    n = len(out)
    n_raw, n_corr = int((out["gap"] > 0).sum()), int((out["corrected_gap"] > 0).sum())
    print(f"Positive lockdown gap: raw {n_raw}/{n} ({n_raw/n:.0%}) -> "
          f"drift-corrected {n_corr}/{n} ({n_corr/n:.0%})")

    per_station = (out.assign(raw=out["gap"] > 0, corrected=out["corrected_gap"] > 0)
                   .groupby("station")[["raw", "corrected"]].sum()
                   .sort_values(["corrected", "raw"], ascending=False))
    per_station.index = [STATION_LABELS.get(s, s) for s in per_station.index]
    print("\nModels (of 5) with a positive gap, per station:")
    print(per_station.to_string())
    print(f"\nMean |raw lockdown gap|: {out['gap'].abs().mean():.3f} µg/m³")
    print("✓ Saved: results/rq2_horizon_pseudo_gaps.csv, results/rq2_bias_corrected.csv")
    return out


if __name__ == "__main__":
    run()
