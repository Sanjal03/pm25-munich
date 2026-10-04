"""
candidate_batch_validation.py
================================================================================
Full 2019-2024 validation of 5 more candidate stations near Munich, found via
station_discovery.py but never fully validated (only 1-week probed before):

  Trostberg/Schwimmbadstraße   (DEBY088, 73 km E)
  Mehring/Sportplatz           (DEBY013, 90 km E)
  Kempten (Allgäu)/Westendstr. (DEBY031, 105 km SW -- new direction)
  Oettingen/Goethestraße       (DEBY187, 116 km NW -- new direction)
  Neu-Ulm/Gabelsbergerstraße   (DEBY052, 120 km W)

Same method as candidate_neustadt_validation.py: reuses fetch_one() from
pipeline/01_fetch_data.py, computes PM2.5 completeness over the full 6-year
period, and daily-mean correlation against München/Landshuter Allee.
Standalone probe -- does not touch the modeling pipeline or tracked results.
"""

import importlib.util
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_spec = importlib.util.spec_from_file_location(
    "fetch_data", Path(__file__).resolve().parent.parent / "pipeline" / "01_fetch_data.py"
)
fetch_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_data)

CANDIDATES = {
    "DEBY088": {"id": "522", "name": "Trostberg/Schwimmbadstraße",   "env": "suburban-background", "dist": 73},
    "DEBY013": {"id": "447", "name": "Mehring/Sportplatz",           "env": "rural-background",    "dist": 90},
    "DEBY031": {"id": "465", "name": "Kempten (Allgäu)/Westendstraße","env": "suburban-background", "dist": 105},
    "DEBY187": {"id": "607", "name": "Oettingen/Goethestraße",       "env": "suburban-background",  "dist": 116},
    "DEBY052": {"id": "486", "name": "Neu-Ulm/Gabelsbergerstraße",   "env": "urban-background",     "dist": 120},
}

OUT_DIR = Path("munich_aq_final/_candidate_batch2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_CSV = Path("munich_aq_final/Munich_AQ_2019-01-01_2024-12-31.csv")
REFERENCE_STATION_NAME = "München/Landshuter Allee"


def _months(start: date, end: date):
    cur = start.replace(day=1)
    while cur <= end:
        nxt = date(cur.year + 1, 1, 1) if cur.month == 12 else cur.replace(month=cur.month + 1)
        yield cur, min(nxt - timedelta(days=1), end)
        cur = nxt


def fetch_full_record(deby_code: str, info: dict, pause: float = 0.5) -> pd.DataFrame:
    fetch_data.STATIONS_MAP[deby_code] = (info["name"], info["env"], info["dist"])
    months = list(_months(date(2019, 1, 1), date(2024, 12, 31)))
    log.info("Fetching %s (%s): %d months x 4 pollutants", info["name"], deby_code, len(months))

    rows = []
    for i, (m0, m1) in enumerate(months, 1):
        for comp_id in fetch_data.COMP_MAP:
            recs = fetch_data.fetch_one(info["id"], deby_code, comp_id,
                                         m0.strftime("%Y-%m-%d"), m1.strftime("%Y-%m-%d"))
            rows.extend(recs)
            time.sleep(pause)
        if i % 24 == 0 or i == len(months):
            log.info("  %-30s %3d/%d months (%d rows so far)", info["name"], i, len(months), len(rows))

    df = pd.DataFrame(rows)
    safe_name = info["name"].split("/")[0].lower().replace(" ", "_").replace("(", "").replace(")", "")
    df.to_csv(OUT_DIR / f"{safe_name}_full_2019_2024.csv", index=False)
    return df


def compute_completeness(deby_code: str, info: dict, df: pd.DataFrame) -> list[dict]:
    n_expected = 52608
    rows = []
    for pollutant in ["PM2.5", "PM10", "NO2", "O3"]:
        n_present = (df["Pollutant"] == pollutant).sum() if not df.empty else 0
        rows.append({
            "station_code": deby_code, "display_name": info["name"], "pollutant": pollutant,
            "n_expected": n_expected, "n_present": n_present,
            "n_missing": n_expected - n_present,
            "pct_missing": round(100 * (n_expected - n_present) / n_expected, 3),
        })
    return rows


def compute_correlation(info: dict, df: pd.DataFrame, ref_daily: pd.Series) -> tuple[float, int] | tuple[None, None]:
    cand = df[df["Pollutant"] == "PM2.5"].copy()
    if cand.empty:
        return None, None
    cand["Datetime"] = pd.to_datetime(cand["Datetime"])
    cand_daily = cand.set_index("Datetime")["Value (µg/m³)"].resample("D").mean()
    joined = pd.concat([ref_daily, cand_daily], axis=1, join="inner").dropna()
    joined.columns = ["reference", "candidate"]
    if len(joined) < 30:
        return None, None
    return round(joined["reference"].corr(joined["candidate"]), 3), len(joined)


def run():
    print("=" * 90)
    print(" BATCH CANDIDATE VALIDATION: 5 stations")
    print("=" * 90)

    ref_raw = pd.read_csv(REFERENCE_CSV)
    ref = ref_raw[(ref_raw["Station Name"] == REFERENCE_STATION_NAME) &
                  (ref_raw["Pollutant"] == "PM2.5")].copy()
    ref["Datetime"] = pd.to_datetime(ref["Datetime"])
    ref_daily = ref.set_index("Datetime")["Value (µg/m³)"].resample("D").mean()

    all_completeness = []
    all_correlation = []

    for deby_code, info in CANDIDATES.items():
        df = fetch_full_record(deby_code, info)
        all_completeness.extend(compute_completeness(deby_code, info, df))
        r, n_days = compute_correlation(info, df, ref_daily)
        all_correlation.append({
            "station": deby_code, "display_name": info["name"], "distance_km": info["dist"],
            "pearson_r": r, "n_days": n_days,
        })
        print(f"\n--- {info['name']} ({info['dist']} km) ---")
        pm25_row = [r for r in all_completeness if r["station_code"] == deby_code and r["pollutant"] == "PM2.5"][0]
        print(f"  PM2.5 completeness: {100 - pm25_row['pct_missing']:.1f}% ({pm25_row['pct_missing']}% missing)")
        pm10_row = [r for r in all_completeness if r["station_code"] == deby_code and r["pollutant"] == "PM10"][0]
        print(f"  PM10 sensor present: {'yes' if pm10_row['pct_missing'] < 100 else 'NO -- 100% missing'}")
        print(f"  Correlation vs {REFERENCE_STATION_NAME}: r={r}" if r is not None else "  Correlation: could not compute")

    comp_df = pd.DataFrame(all_completeness)
    corr_df = pd.DataFrame(all_correlation)
    comp_df.to_csv(OUT_DIR / "batch_completeness.csv", index=False)
    corr_df.to_csv(OUT_DIR / "batch_spatial_correlation.csv", index=False)

    print("\n" + "=" * 90)
    print(" SUMMARY")
    print("=" * 90)
    print(corr_df.to_string(index=False))
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    run()
