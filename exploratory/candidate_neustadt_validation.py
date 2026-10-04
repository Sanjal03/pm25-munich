"""
candidate_neustadt_validation.py
================================================================================
Full 2019-2024 validation of Neustadt a.d. Donau/Eining (DEBY049), the one
candidate station flagged as worth a real look when scanning for additional
Munich-region PM2.5 stations (north of Munich -- a direction none of the
current 9 stations cover; Trostberg and Mehring were also found but sit in
the same easterly direction as the already-added Burghausen, so were judged
lower priority).

Earlier probing (munich_aq_final/_candidate_probe2/station_validation.csv)
confirmed DEBY049 has a working PM2.5 sensor via a 1-week sample. This script
does the real validation: fetches the FULL 2019-01-01 to 2024-12-31 hourly
record (matching what Burghausen/Oberaudorf went through before being added)
and computes:
  1. PM2.5 completeness (% missing) over the full 6-year period
  2. Daily-mean PM2.5 correlation against the München/Landshuter Allee
     reference station, same method as station_metadata_spatial.py

Reuses fetch_one() from pipeline/01_fetch_data.py so results are directly
comparable -- same API, same parsing, same units. Output does NOT touch
the existing modeling pipeline or tracked results -- this is a standalone
validation probe, same as the earlier _candidate_probe* runs.
"""

import importlib.util
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Import fetch_one() from pipeline/01_fetch_data.py without renaming the file ──
_spec = importlib.util.spec_from_file_location(
    "fetch_data", Path(__file__).resolve().parent.parent / "pipeline" / "01_fetch_data.py"
)
fetch_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_data)

DEBY_CODE = "DEBY049"
NUMERIC_ID = "483"  # confirmed in munich_aq_final/_candidate_probe2/station_validation.csv
STATION_NAME = "Neustadt a.d. Donau/Eining"
DISTANCE_KM = 81

OUT_DIR = Path("munich_aq_final/_candidate_probe_neustadt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_CSV = Path("munich_aq_final/Munich_AQ_2019-01-01_2024-12-31.csv")
REFERENCE_STATION_NAME = "München/Landshuter Allee"


def _months(start: date, end: date):
    cur = start.replace(day=1)
    while cur <= end:
        nxt = date(cur.year + 1, 1, 1) if cur.month == 12 else cur.replace(month=cur.month + 1)
        yield cur, min(nxt - timedelta(days=1), end)
        cur = nxt


def fetch_full_record(pause: float = 0.5) -> pd.DataFrame:
    fetch_data.STATIONS_MAP[DEBY_CODE] = (STATION_NAME, "rural-background", DISTANCE_KM)

    months = list(_months(date(2019, 1, 1), date(2024, 12, 31)))
    log.info("Fetching %s (DEBY049): %d months x 4 pollutants", STATION_NAME, len(months))

    rows = []
    for i, (m0, m1) in enumerate(months, 1):
        for comp_id in fetch_data.COMP_MAP:
            recs = fetch_data.fetch_one(NUMERIC_ID, DEBY_CODE, comp_id,
                                         m0.strftime("%Y-%m-%d"), m1.strftime("%Y-%m-%d"))
            rows.extend(recs)
            time.sleep(pause)
        if i % 12 == 0 or i == len(months):
            log.info("  %3d/%d months done (%d rows so far)", i, len(months), len(rows))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "neustadt_full_2019_2024.csv", index=False)
    log.info("Saved %d rows -> %s", len(df), OUT_DIR / "neustadt_full_2019_2024.csv")
    return df


def compute_completeness(df: pd.DataFrame) -> pd.DataFrame:
    n_expected = 52608  # 6 years x 8768h... matches project convention (2019-01-01 00:00 to 2024-12-31 23:00)
    rows = []
    for pollutant in ["PM2.5", "PM10", "NO2", "O3"]:
        n_present = (df["Pollutant"] == pollutant).sum()
        rows.append({
            "station_code": DEBY_CODE, "display_name": STATION_NAME, "pollutant": pollutant,
            "n_expected": n_expected, "n_present": n_present,
            "n_missing": n_expected - n_present,
            "pct_missing": round(100 * (n_expected - n_present) / n_expected, 3),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "neustadt_completeness.csv", index=False)
    return out


def compute_correlation(df: pd.DataFrame) -> float | None:
    if not REFERENCE_CSV.exists():
        log.warning("Reference CSV not found at %s -- skipping correlation", REFERENCE_CSV)
        return None

    ref_raw = pd.read_csv(REFERENCE_CSV)
    ref = ref_raw[(ref_raw["Station Name"] == REFERENCE_STATION_NAME) &
                  (ref_raw["Pollutant"] == "PM2.5")].copy()
    ref["Datetime"] = pd.to_datetime(ref["Datetime"])
    ref_daily = ref.set_index("Datetime")["Value (µg/m³)"].resample("D").mean()

    cand = df[df["Pollutant"] == "PM2.5"].copy()
    if cand.empty:
        log.warning("No PM2.5 rows fetched for %s -- cannot compute correlation", STATION_NAME)
        return None
    cand["Datetime"] = pd.to_datetime(cand["Datetime"])
    cand_daily = cand.set_index("Datetime")["Value (µg/m³)"].resample("D").mean()

    joined = pd.concat([ref_daily, cand_daily], axis=1, join="inner").dropna()
    joined.columns = ["reference", "candidate"]
    r = joined["reference"].corr(joined["candidate"])
    n_days = len(joined)

    pd.DataFrame([{
        "station": DEBY_CODE, "display_name": STATION_NAME, "distance_km": DISTANCE_KM,
        "pearson_r": round(r, 3), "n_days": n_days,
    }]).to_csv(OUT_DIR / "neustadt_spatial_correlation.csv", index=False)

    return r


def run():
    print("=" * 90)
    print(f" CANDIDATE STATION VALIDATION: {STATION_NAME} ({DEBY_CODE}, {DISTANCE_KM} km, N of Munich)")
    print("=" * 90)

    df = fetch_full_record()

    print("\n--- Completeness ---")
    comp = compute_completeness(df)
    print(comp.to_string(index=False))

    print("\n--- Spatial correlation vs München/Landshuter Allee ---")
    r = compute_correlation(df)
    if r is not None:
        print(f"Pearson r = {r:.3f} (daily-mean PM2.5, vs {REFERENCE_STATION_NAME})")

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    run()
