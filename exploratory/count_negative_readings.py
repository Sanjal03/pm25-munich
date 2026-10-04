"""
count_negative_readings.py
================================================================================
Counts negative pollutant readings across the full 2019-2024 record for the
7 final stations -- these are silently discarded in pipeline/01_fetch_data.py
(fetch_one(), "negative = sensor error") and were never logged, so this
re-queries the UBA API directly to get the real count for the Data chapter's
Physical Consistency note.
"""
import importlib.util
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "fetch_data", Path(__file__).resolve().parent.parent / "pipeline" / "01_fetch_data.py"
)
fetch_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_data)

STATIONS = {
    "DEBY037": "München/Stachus", "DEBY115": "München/Landshuter Allee",
    "DEBY039": "München/Lothstraße", "DEBY089": "München/Johanneskirchen",
    "DEBY007": "Augsburg/Bourges-Platz", "DEBY099": "Augsburg/LfU",
    "DEBY109": "Andechs/Rothenfeld",
}


def fetch_one_counting_negatives(numeric_id, deby_code, comp_id, date_from, date_to):
    """Same request as fetch_one(), but counts negatives instead of discarding silently."""
    try:
        resp = fetch_data.SESSION.get(
            fetch_data.MEASURES_URL,
            params={"station": numeric_id, "component": comp_id, "scope": fetch_data.SCOPE_HOURLY,
                    "date_from": date_from, "date_to": date_to, "time_from": "1", "time_to": "24",
                    "lang": "en"},
            timeout=90,
        )
    except Exception:
        return 0, 0
    if resp.status_code != 200:
        return 0, 0
    try:
        payload = resp.json()
    except ValueError:
        return 0, 0
    hourly = payload.get("data", {}).get(numeric_id, {})
    n_total, n_negative = 0, 0
    for entry in hourly.values():
        if not isinstance(entry, list) or len(entry) < 3:
            continue
        raw = entry[2]
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        n_total += 1
        if value < 0:
            n_negative += 1
    return n_total, n_negative


def _months(start, end):
    cur = start.replace(day=1)
    while cur <= end:
        nxt = date(cur.year + 1, 1, 1) if cur.month == 12 else cur.replace(month=cur.month + 1)
        yield cur, min(nxt - timedelta(days=1), end)
        cur = nxt


id_map = fetch_data.resolve_ids(Path("munich_aq_final"))
months = list(_months(date(2019, 1, 1), date(2024, 12, 31)))

rows = []
for deby, name in STATIONS.items():
    num_id = id_map.get(deby)
    if not num_id:
        print(f"  ! could not resolve numeric id for {deby}")
        continue
    for comp_id, pol in fetch_data.COMP_MAP.items():
        n_total, n_neg = 0, 0
        for m0, m1 in months:
            t, n = fetch_one_counting_negatives(num_id, deby, comp_id,
                                                  m0.strftime("%Y-%m-%d"), m1.strftime("%Y-%m-%d"))
            n_total += t
            n_neg += n
            time.sleep(0.4)
        rows.append({"station": name, "pollutant": pol, "n_total": n_total, "n_negative": n_neg})
        print(f"  {name:<25} {pol:<5} total={n_total:<6} negative={n_neg}")

out = pd.DataFrame(rows)
out.to_csv("results/negative_readings_count.csv", index=False)
print("\n" + "=" * 60)
print(f"TOTAL negative readings across all stations/pollutants: {out['n_negative'].sum()}")
print(out.groupby('pollutant')['n_negative'].sum())
print("Saved results/negative_readings_count.csv")
