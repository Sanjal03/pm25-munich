"""
DEPRECATED — DO NOT RUN. Kept only for project history.
=========================================================
This script hits `luftdaten.umweltbundesamt.de/api-proxy/measures/json`,
which silently redirects and returns an HTML error page instead of real
data (see pipeline/01_fetch_data.py's module docstring for the full
story of which endpoint versions actually work). It predates the
correct fetcher and produces no usable output. Superseded entirely by
pipeline/01_fetch_data.py.

Original docstring, preserved for context:
------------------------------------------
check_uba_availability.py
Quick diagnostic — runs in ~2 minutes.
Fetches ONE week per year per station to check availability and missing %.
Run this BEFORE the full download to know what you're dealing with.

    pip install requests pandas
    python check_uba_availability.py
"""

import time
import requests
import pandas as pd
from datetime import timedelta

BASE_URL = "https://luftdaten.umweltbundesamt.de/api-proxy/measures/json"

STATIONS = {
    "DEBY115": "München/Landshuter Allee",
    "DEBY037": "München/Stachus",
    "DEBY089": "München/Johanneskirchen",
    "DEBY109": "Andechs/Rothenfeld",
    "DEBY121": "Oberaudorf/Inntal-Autobahn",
    "DEBY099": "Augsburg/LfU",
    "DEBY110": "Augsburg/Karlstraße",
}

# Sample one week per year — middle of each year avoids NYE/summer edge cases
SAMPLE_WEEKS = {
    2019: ("2019-06-10", "2019-06-16"),
    2020: ("2020-06-10", "2020-06-16"),
    2021: ("2021-06-10", "2021-06-16"),
    2022: ("2022-06-10", "2022-06-16"),
    2023: ("2023-06-10", "2023-06-16"),
    2024: ("2024-06-10", "2024-06-16"),
}

EXPECTED_HOURS = 7 * 24  # 168 hours per week sample

def fetch_week(station_code, date_from, date_to):
    params = {
        "date_from": date_from, "date_to": date_to,
        "time_from": 1, "time_to": 24,
        "station": station_code, "component": 9, "scope": 2,
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        return None, str(e)

    scope_data = (js.get("data", {})
                    .get(station_code, {})
                    .get("9", {})
                    .get("2", {}))
    values = [v for v in scope_data.values() if v is not None]
    return len(scope_data), len(values)

print("=" * 72)
print("  UBA PM2.5 Availability Check — sample week per year per station")
print("=" * 72)
print(f"\n{'Station':<32}", end="")
for y in SAMPLE_WEEKS:
    print(f"  {y}", end="")
print(f"  {'Notes'}")
print("-" * 72)

results = {}
for code, name in STATIONS.items():
    row_str = f"{name:<32}"
    year_results = {}
    for year, (d_from, d_to) in SAMPLE_WEEKS.items():
        total, valid = fetch_week(code, d_from, d_to)
        time.sleep(0.4)

        if total is None:
            symbol = " ERR"
            year_results[year] = {"total": 0, "valid": 0, "miss_pct": 100, "error": True}
        else:
            miss_pct = 100 * (1 - valid / EXPECTED_HOURS)
            year_results[year] = {"total": total, "valid": valid, "miss_pct": miss_pct, "error": False}
            if miss_pct == 0:
                symbol = "  ✓ "
            elif miss_pct < 10:
                symbol = f"{miss_pct:4.0f}%"
            elif miss_pct < 50:
                symbol = f"⚠{miss_pct:3.0f}%"
            else:
                symbol = f"✗{miss_pct:3.0f}%"
        row_str += f"  {symbol}"

    results[code] = year_results

    # Notes
    bad_years = [y for y, r in year_results.items() if r["miss_pct"] > 20]
    note = f"  DROP {bad_years}" if bad_years else "  OK"
    row_str += note
    print(row_str)

print("\n" + "=" * 72)
print("  VERDICT")
print("=" * 72)

keep, warn, drop = [], [], []
for code, year_results in results.items():
    avg_miss = sum(r["miss_pct"] for r in year_results.values()) / len(year_results)
    name = STATIONS[code]
    if avg_miss < 5:
        keep.append((code, name, avg_miss))
    elif avg_miss < 20:
        warn.append((code, name, avg_miss))
    else:
        drop.append((code, name, avg_miss))

print("\n✓ KEEP (< 5% missing avg):")
for code, name, pct in keep:
    print(f"    {code}  {name}  ({pct:.1f}% avg missing)")

print("\n⚠ INTERPOLATE (5–20% missing avg):")
for code, name, pct in warn:
    print(f"    {code}  {name}  ({pct:.1f}% avg missing)")

print("\n✗ CONSIDER DROPPING (> 20% missing avg):")
for code, name, pct in drop:
    print(f"    {code}  {name}  ({pct:.1f}% avg missing)")

print("\n[INFO] Now run download_uba_munich.py to get the full dataset.")
print("       Add any DROP stations to the SKIP list in that script.")