"""
leakage_sensitivity_check.py — Outlier thresholds: train-only vs full data
==========================================================================
Both outlier caps learn their thresholds from the training period only
(2019-2023) and then apply them to every year:

  Stage 1  02_preprocess.py          99.5th-percentile cap per column
  Stage 2  03_feature_engineering.py Q3 + 3*IQR fence on the PM2.5 target

Earlier versions of the pipeline computed both on 2019-2024, letting test
data shape the training thresholds. This script documents how much that
mattered: for each PM2.5 threshold it compares the train-only value (used
now) with the full-data value (used before). Stage 1 is evaluated
twice: on the raw measurements, and on the imputed series exactly as 02
caps it (col_name/hybrid_impute are loaded
from 02_preprocess.py itself, so the logic can't drift). Stage 2 is
evaluated on Munich_AQ_clean.csv, as 03 sees it.

Output: results/leakage_sensitivity.csv
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AQ_RAW = Path("munich_aq_final") / "Munich_AQ_2019-01-01_2024-12-31.csv"
CLEAN = Path("Munich_AQ_clean.csv")
TRAIN_END = "2023-12-31 23:00"
FULL = pd.date_range("2019-01-01", "2024-12-31 23:00", freq="h")

STATIONS = {
    "landshuter_allee": "mun_landshuter_allee", "stachus": "mun_stachus",
    "lothstrasse": "mun_lothstrasse", "johanneskirchen": "mun_johanneskirchen",
    "augsburg_bourges": "aug_bourges-platz", "augsburg_lfu": "aug_lfu",
    "andechs": "andechs_rothenfeld",
}


def load_preprocess_functions() -> dict:
    """Exec only the function defs from 02_preprocess.py (the module itself
    runs the whole pipeline at import time)."""
    src = (ROOT / "pipeline" / "02_preprocess.py").read_text()
    tree = ast.parse(src)
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("col_name", "hybrid_impute")]
    ns = {"pd": pd, "np": np}
    exec(compile(ast.Module(body=defs, type_ignores=[]), "02_preprocess.py", "exec"), ns)
    return ns


def raw_hourly(fns: dict) -> pd.DataFrame:
    df = pd.read_csv(AQ_RAW)
    df["datetime"] = pd.to_datetime(df["Datetime"])
    wide = df.pivot_table(index="datetime", columns=["Station Name", "Pollutant"],
                          values="Value (µg/m³)", aggfunc="mean")
    wide.columns = [fns["col_name"](s, p) for s, p in wide.columns]
    return wide.reindex(FULL)


def iqr_fence(s: pd.Series) -> float:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return q3 + 3 * (q3 - q1)


def run():
    fns = load_preprocess_functions()
    raw = raw_hourly(fns)
    wide = fns["hybrid_impute"](raw)
    clean = pd.read_csv(CLEAN, index_col="datetime", parse_dates=True)

    rows = []
    for station, prefix in STATIONS.items():
        col = f"{prefix}_pm25"
        for stage, series, fn in [
            ("1a: 99.5th pct cap, raw", raw[col], lambda s: s.quantile(0.995)),
            ("1b: 99.5th pct cap, imputed (02)", wide[col], lambda s: s.quantile(0.995)),
            ("2: Q3+3*IQR fence (03)", clean[col], iqr_fence),
        ]:
            full, train = fn(series), fn(series.loc[:TRAIN_END])
            rows.append({"stage": stage, "station": station,
                         "threshold_full": round(full, 3),
                         "threshold_train_only": round(train, 3),
                         "abs_diff": round(abs(full - train), 3),
                         "pct_diff": round(abs(full - train) / train * 100, 2)})

    out = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/leakage_sensitivity.csv", index=False)
    print(out.to_string(index=False))
    for stage, g in out.groupby("stage"):
        print(f"\nStage {stage}: max diff {g['pct_diff'].max():.2f}%, "
              f"{(g['abs_diff'] == 0).sum()} of {len(g)} stations identical")
    print("\n✓ Saved: results/leakage_sensitivity.csv")


if __name__ == "__main__":
    run()
