"""
03_feature_engineering.py
==========================
Input:  Munich_AQ_clean.csv
Output: One features CSV per station  →  features/{station_code}_features.csv
        Also saves combined: Munich_AQ_features_all.csv

Predicts PM2.5 for ALL 7 stations independently.
Each station gets its own feature set built from:
  - Its own PM2.5 history (lags + rolling)
  - Its own co-pollutants (PM10, NO2, O3 where available)
  - Shared DWD meteorology (same for all stations)
  - Shared temporal + event flags

PM2.5 stations:
  mun_landshuter_allee   mun_stachus        mun_lothstrasse
  mun_johanneskirchen    aug_bourges-platz  aug_lfu
  andechs_rothenfeld
"""

import pandas as pd
import numpy as np
from pathlib import Path

from model_factory import LOCKDOWN_START, LOCKDOWN_END, TRAIN_END

DATA = Path(".")  # Changed from "data" to "." (current directory)
IN   = DATA / "Munich_AQ_clean.csv"
OUT_DIR = DATA / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)  # Added parents=True
OUT_ALL = DATA / "Munich_AQ_features_all.csv"

# ── PM2.5 stations ────────────────────────────────────────────────────────────
# Maps station_code → column prefix in clean CSV
PM25_STATIONS = {
    "landshuter_allee": "mun_landshuter_allee",
    "stachus":          "mun_stachus",
    "lothstrasse":      "mun_lothstrasse",
    "johanneskirchen":  "mun_johanneskirchen",
    "augsburg_bourges": "aug_bourges-platz",
    "augsburg_lfu":     "aug_lfu",
    "andechs":          "andechs_rothenfeld",
}

# O3 not measured at Landshuter Allee (traffic station — EU design)
# Use Lothstraße O3 as proxy for stations without O3
O3_PROXY = "mun_lothstrasse_o3"

# ── Met + shared feature columns ──────────────────────────────────────────────
MET_COLS = [
    "temp_c", "humidity_pct",
    "wind_speed", "wind_u", "wind_v",
    "pressure", "pressure_diff_24h", "temp_diff_1h",
]

# ── Load clean data ───────────────────────────────────────────────────────────
print("Loading clean data...")
df = pd.read_csv(IN, index_col="datetime", parse_dates=True)
print(f"  {len(df):,} rows | {df.shape[1]} columns")

# ── Shared temporal features (same for every station) ────────────────────────
print("Building temporal features...")
df["hour_sin"]  = np.sin(2 * np.pi * df.index.hour  / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df.index.hour  / 24)
df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
df["dow_sin"]   = np.sin(2 * np.pi * df.index.dayofweek / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df.index.dayofweek / 7)
df["is_weekend"]     = (df.index.dayofweek >= 5).astype(int)
df["is_heating"]     = df.index.month.isin([10,11,12,1,2,3]).astype(int)
df["covid_lockdown"] = (
    (df.index >= LOCKDOWN_START) & (df.index <= LOCKDOWN_END)
).astype(int)
df["year"] = df.index.year
if "is_nye" not in df.columns:
    df["is_nye"] = (
        ((df.index.month==12)&(df.index.day==31)&(df.index.hour>=20)) |
        ((df.index.month==1) &(df.index.day==1) &(df.index.hour<=3))
    ).astype(int)

TEMPORAL_COLS = [
    "hour_sin","hour_cos","month_sin","month_cos","dow_sin","dow_cos",
    "is_weekend","is_heating","covid_lockdown","year","is_nye","gap_flag",
]

# ── Build features per station ────────────────────────────────────────────────
all_frames = []

for station_code, col_prefix in PM25_STATIONS.items():
    print(f"\nBuilding features for: {station_code}")

    target_col = f"{col_prefix}_pm25"
    if target_col not in df.columns:
        print(f"  ✗ {target_col} not found — skipping")
        continue

    feat = pd.DataFrame(index=df.index)
    feat["station"] = station_code
    feat["target"]  = df[target_col]      # PM2.5 = prediction target

    # ── Outlier removal (IQR 3× fence) ───────────────────────────────────────
    # Applied to target only — co-pollutants were already capped in preprocessing.
    # Quartiles come from the training period only (2019-2023).
    train_target = feat["target"].loc[:TRAIN_END]
    Q1   = train_target.quantile(0.25)
    Q3   = train_target.quantile(0.75)
    IQR  = Q3 - Q1
    fence = Q3 + 3 * IQR
    n_removed = (feat["target"] > fence).sum()
    feat["target"] = feat["target"].clip(upper=fence)
    print(f"  Outlier cap: {fence:.1f} µg/m³  ({n_removed} rows capped)")

    # ── Lag features ──────────────────────────────────────────────────────────
    # PM2.5 autocorrelation is the strongest predictor at 1h horizon
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        feat[f"pm25_lag{lag}h"] = feat["target"].shift(lag)

    # ── Rolling features ──────────────────────────────────────────────────────
    shifted = feat["target"].shift(1)
    feat["pm25_roll6h"]    = shifted.rolling(6,   min_periods=1).mean()
    feat["pm25_roll24h"]   = shifted.rolling(24,  min_periods=6).mean()
    feat["pm25_roll168h"]  = shifted.rolling(168, min_periods=24).mean()
    feat["pm25_roll24h_std"] = shifted.rolling(24, min_periods=6).std()

    # ── Co-pollutant features ─────────────────────────────────────────────────
    feat["pm10"] = df.get(f"{col_prefix}_pm10",  np.nan)
    feat["no2"]  = df.get(f"{col_prefix}_no2",   np.nan)

    # O3: use station's own if available, else Lothstraße proxy
    o3_col = f"{col_prefix}_o3"
    feat["o3"] = df[o3_col] if o3_col in df.columns else df.get(O3_PROXY, np.nan)
    if o3_col not in df.columns:
        print(f"  O3: using proxy ({O3_PROXY})")

    # Background PM2.5: mean of all OTHER stations (spatial context)
    other_pm25 = [f"{pfx}_pm25" for code, pfx in PM25_STATIONS.items()
                  if code != station_code and f"{pfx}_pm25" in df.columns]
    feat["pm25_background"] = df[other_pm25].mean(axis=1)

    # ── Met features ──────────────────────────────────────────────────────────
    for col in MET_COLS:
        if col in df.columns:
            feat[col] = df[col]

    # ── Temporal + event flags ────────────────────────────────────────────────
    for col in TEMPORAL_COLS:
        if col in df.columns:
            feat[col] = df[col]

    # ── Drop rows where critical features are NaN ─────────────────────────────
    # First 168h loses lag168h; first few hours lose rolling means
    required = ["target", "pm25_lag1h", "pm25_lag24h", "temp_c", "humidity_pct"]
    feat = feat.dropna(subset=[c for c in required if c in feat.columns])

    # ── Train / test split ────────────────────────────────────────────────────
    # Chronological: train 2019-2023, test 2024
    # No random shuffle — mandatory for time series to prevent data leakage
    feat["split"] = "train"
    feat.loc[feat.index.year == 2024, "split"] = "test"

    n_train = (feat["split"]=="train").sum()
    n_test  = (feat["split"]=="test").sum()
    print(f"  Train: {n_train:,} rows | Test: {n_test:,} rows")
    print(f"  Features: {feat.shape[1]-3} ({feat.shape[1]} cols - station/target/split)")
    print(f"  Missing: {feat.drop(columns=['station','split']).isna().mean().mean()*100:.3f}%")

    # Save per-station file
    out_path = OUT_DIR / f"{station_code}_features.csv"
    feat.to_csv(out_path)
    print(f"  ✓ Saved: {out_path.name}")

    all_frames.append(feat)

# ── Save combined file ────────────────────────────────────────────────────────
combined = pd.concat(all_frames)
combined.to_csv(OUT_ALL)
print(f"\n✓ Combined file saved: {OUT_ALL}")
print(f"  {len(combined):,} rows | {combined['station'].nunique()} stations")

# ── Feature summary ───────────────────────────────────────────────────────────
sample = all_frames[0]
feature_cols = [c for c in sample.columns if c not in ["station","target","split"]]
groups = {
    "PM2.5 lags (8)":    [c for c in feature_cols if "pm25_lag" in c],
    "PM2.5 rolling (4)": [c for c in feature_cols if "roll" in c],
    "Co-pollutants (4)": ["pm10","no2","o3","pm25_background"],
    "Meteorology (8)":   MET_COLS,
    "Temporal (9)":      [c for c in feature_cols if any(x in c for x in
                          ["sin","cos","weekend","heating","year"])],
    "Event flags (3)":   ["is_nye","covid_lockdown","gap_flag"],
}
print("\n── Feature Groups ──────────────────────────────────")
total = 0
for group, feats in groups.items():
    available = [f for f in feats if f in feature_cols]
    print(f"  {group:<25}: {available}")
    total += len(available)
print(f"  {'TOTAL':<25}: {total} features")
