"""
02_preprocess.py — Data Preparation Pipeline
=============================================
Input:  Munich_AQ_2019-01-01_2024-12-31.csv
        DWD met files (tu, ff, p0)
Output: Munich_AQ_clean.csv

PM2.5 stations (7):
  München/Landshuter Allee, München/Stachus, München/Lothstraße,
  München/Johanneskirchen, Augsburg/Bourges-Platz, Augsburg/LfU,
  Andechs/Rothenfeld

Steps:
  1. Load AQ data
  2. Pivot to wide hourly format
  3. Impute AQ gaps (linear ≤6h, spatial 7-24h, flag >24h)
  4. Cap outliers at the 2019-2023 99.5th percentile + NYE flag
  5. Load & merge DWD met (with correct interpolation limits)
  6. Fix PM10 at Landshuter Allee (264h gap → fill from Stachus)
  7. Save
"""

import pandas as pd
import numpy as np
from pathlib import Path

from model_factory import TRAIN_END

# The CSV is inside 'munich_aq_final'. A fresh run of 01_fetch_data.py
# (7 modeled stations + Allach) reproduces this exact filename.
AQ_RAW = Path("munich_aq_final") / "Munich_AQ_2019-01-01_2024-12-31.csv"

# The text files and the output will go in the root directory
ROOT_DIR = Path(".")
DWD_TU = ROOT_DIR / "Temperature.txt"
DWD_FF = ROOT_DIR / "Wind.txt"
DWD_P0 = ROOT_DIR / "Pressure.txt"
OUT    = ROOT_DIR / "Munich_AQ_clean.csv"

FULL = pd.date_range("2019-01-01", "2024-12-31 23:00", freq="h")

# clean column name helper
def col_name(station, pollutant):
    s = (station.replace("München/","mun_")
                .replace("Augsburg/","aug_")
                .replace("/","_").replace(" ","_")
                .replace("ü","u").replace("ß","ss")
                .lower())
    p = pollutant.lower().replace("2.","2").replace(".","")
    return f"{s}_{p}"

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading air quality data...")
df = pd.read_csv(AQ_RAW)
df["datetime"] = pd.to_datetime(df["Datetime"])
# No unit correction needed — measures API returns real µg/m³

# ── 2. Pivot ──────────────────────────────────────────────────────────────────
print("Pivoting...")
wide = df.pivot_table(
    index="datetime",
    columns=["Station Name","Pollutant"],
    values="Value (µg/m³)",
    aggfunc="mean",
)
wide.columns = [col_name(s,p) for s,p in wide.columns]
wide = wide.reindex(FULL)
wide.index.name = "datetime"
print(f"  {wide.shape[1]} columns | {len(wide):,} rows")
print(f"  Columns: {list(wide.columns)}")

# ── 3. Impute AQ gaps ─────────────────────────────────────────────────────────
print("Imputing AQ gaps...")

def hybrid_impute(df_wide, max_short=6, max_medium=24):
    result   = df_wide.copy()
    gap_flag = pd.Series(False, index=df_wide.index)

    def get_pol(col): return col.split("_")[-1]

    for col in result.columns:
        s       = result[col].copy()
        nb_cols = [c for c in result.columns
                   if c != col and get_pol(c) == get_pol(col)]
        # Find gaps
        gaps=[]; count=0; start=None
        for i, v in enumerate(s.isna()):
            if v:
                if count==0: start=i
                count+=1
            else:
                if count>0: gaps.append((start, count))
                count=0
        if count>0: gaps.append((start, count))

        for start_i, length in gaps:
            end_i = start_i + length
            if length > max_medium:
                gap_flag.iloc[start_i:end_i] = True
            elif length > max_short and nb_cols:
                nb_mean = df_wide.iloc[start_i:end_i][nb_cols].mean(axis=1)
                if nb_mean.notna().any():
                    s.iloc[start_i:end_i] = nb_mean.values
        result[col] = s.interpolate(method="linear", limit=max_short)

    result["gap_flag"] = gap_flag.astype(int)
    return result

wide = hybrid_impute(wide)
print(f"  Long gaps flagged: {wide['gap_flag'].sum():,} hours")

# ── 4. Cap outliers + flags ───────────────────────────────────────────────────
# Thresholds are learned on the training period only, then applied to all
# years, so 2024 (test) values never influence preprocessing.
print("Capping outliers (99.5th percentile of 2019-2023)...")
for col in [c for c in wide.columns if c != "gap_flag"]:
    wide[col] = wide[col].clip(upper=wide[col].loc[:TRAIN_END].quantile(0.995))

wide["is_nye"] = (
    ((wide.index.month==12)&(wide.index.day==31)&(wide.index.hour>=20)) |
    ((wide.index.month==1) &(wide.index.day==1) &(wide.index.hour<=3))
).astype(int)

# ── 5. Load DWD met ───────────────────────────────────────────────────────────
print("Loading DWD met...")

def load_dwd(path, cols):
    d = pd.read_csv(path, sep=";", low_memory=False)
    d.columns = d.columns.str.strip()
    d["datetime"] = pd.to_datetime(
        d["MESS_DATUM"].astype(str).str.zfill(10), format="%Y%m%d%H")
    return d.set_index("datetime")[cols].replace(-999, np.nan).reindex(FULL)

tu = load_dwd(DWD_TU, ["TT_TU","RF_TU"])
ff = load_dwd(DWD_FF, ["F","D"])
p0 = load_dwd(DWD_P0, ["P0"])

met = pd.DataFrame({
    "temp_c":       tu["TT_TU"],
    "humidity_pct": tu["RF_TU"],
    "wind_speed":   ff["F"],
    "wind_dir":     ff["D"],
    "pressure":     p0["P0"],
}, index=FULL)

# Interpolate with limits matched to actual gap sizes
# temp: 4h gap, humidity: 25h gap, wind: 9h gap, pressure: 0 gaps
met["temp_c"]       = met["temp_c"].interpolate(limit=6)
met["humidity_pct"] = met["humidity_pct"].interpolate(limit=26)
met["wind_speed"]   = met["wind_speed"].interpolate(limit=12)
met["wind_dir"]     = met["wind_dir"].interpolate(limit=6)
met["pressure"]     = met["pressure"].interpolate(limit=3)

# Derived features
rad = np.deg2rad(met["wind_dir"])
met["wind_u"]            = (-met["wind_speed"] * np.sin(rad)).round(3)
met["wind_v"]            = (-met["wind_speed"] * np.cos(rad)).round(3)
met["pressure_diff_24h"] = met["pressure"].diff(24).round(3)
met["temp_diff_1h"]      = met["temp_c"].diff(1).round(3)

for col in met.columns:
    n = met[col].isna().sum()
    status = "✓" if n==0 else f"✗ {n} remaining"
    print(f"  {col}: {status}")

# ── 6. Fix PM10 Landshuter Allee (264h gap) ───────────────────────────────────
la_pm10 = col_name("München/Landshuter Allee", "PM10")
st_pm10 = col_name("München/Stachus", "PM10")
if la_pm10 in wide.columns and st_pm10 in wide.columns:
    still_missing = wide[la_pm10].isna()
    wide.loc[still_missing, la_pm10] = wide.loc[still_missing, st_pm10]
    wide[la_pm10] = wide[la_pm10].interpolate(limit=24)
    print(f"  {la_pm10}: {wide[la_pm10].isna().sum()} remaining after Stachus fill")

# ── 7. Merge & save ───────────────────────────────────────────────────────────
clean = wide.join(met)
clean.index.name = "datetime"
clean.to_csv(OUT)

print(f"\n✓ Saved: {OUT}")
print(f"  {len(clean):,} rows | {clean.shape[1]} columns")
print(f"  Overall missing: {clean.isna().mean().mean()*100:.3f}%")
remaining = [(c, clean[c].isna().sum()) for c in clean.columns if clean[c].isna().sum()>0]
if remaining:
    print("  Remaining NaN (intentional long gaps, handled by model natively):")
    for c, n in remaining:
        print(f"    {c}: {n:,} ({n/len(clean)*100:.2f}%)")
