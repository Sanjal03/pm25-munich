"""
discover_nearby_stations.py
====================================================================
Finds REAL air quality stations near Munich by querying UBA's live
station list — the same endpoint your fetch_munich_aq_final.py
already uses successfully. No hardcoded station codes or guessed
coordinates: every station code, name, and coordinate pair here comes
directly from UBA's own API response.

This answers "are there more stations near Munich I could add?" with
actual data instead of a guess.

Usage:
  python discover_nearby_stations.py                  # 60km radius
  python discover_nearby_stations.py --radius 40       # tighter radius
"""

import argparse
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

import pandas as pd
import requests

STATIONS_URL = "https://www.umweltbundesamt.de/api/air_data/v2/stations/json"
MUNICH_CENTER = {"lat": 48.1374, "lon": 11.5755}

# Your current 8 stations (from fetch_munich_aq_final.py's STATIONS_MAP) —
# excluded from "new candidate" results since you already have them.
CURRENT_CODES = {"DEBY115", "DEBY037", "DEBY039", "DEBY089",
                  "DEBY189", "DEBY109", "DEBY007", "DEBY099"}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def fetch_all_stations():
    """
    Pulls UBA's FULL station list (all Länder, all types — traffic,
    background, industrial — not just the traffic-type stations your
    original fetcher targets). Confirmed field order from live response:
    [id, code, name, city, synonym, active_from, active_to,
     longitude, latitude, network_id, setting_id, type_id,
     network_code, network_name, setting_name, setting_short,
     type_name, street, street_nr, zip]
    """
    resp = requests.get(
        STATIONS_URL,
        params={"use": "airquality", "lang": "en",
                "date_from": "2023-01-01", "time_from": "1",
                "date_to": "2023-12-31", "time_to": "24"},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("data", {})


def run_discovery(radius_km=60):
    print("\n" + "=" * 80)
    print(f" REAL STATION DISCOVERY — within {radius_km} km of Munich")
    print(" (querying live UBA API — no hardcoded station list)")
    print("=" * 80)

    print("\nFetching full station list from UBA...")
    data = fetch_all_stations()
    print(f"✓ Received {len(data)} total stations (all of Germany, all types)")

    candidates = []
    for numeric_id, fields in data.items():
        if len(fields) < 17:
            continue
        code = fields[1]
        name = fields[2]
        city = fields[3]
        active_to = fields[6]
        try:
            lon = float(fields[7])
            lat = float(fields[8])
        except (TypeError, ValueError):
            continue
        network_code = fields[12]
        setting = fields[15]
        station_type = fields[16]

        dist = haversine_km(MUNICH_CENTER["lat"], MUNICH_CENTER["lon"], lat, lon)
        if dist <= radius_km:
            candidates.append({
                "code": code, "name": name, "city": city,
                "lat": lat, "lon": lon, "distance_km": round(dist, 1),
                "network": network_code, "setting": setting,
                "type": station_type, "active_to": active_to,
                "currently_active": active_to is None,
                "already_have": code in CURRENT_CODES,
            })

    df = pd.DataFrame(candidates).sort_values("distance_km")

    print(f"\n✓ Found {len(df)} stations within {radius_km} km of Munich "
          f"(any type, any status)")

    already = df[df["already_have"]]
    new_active = df[(~df["already_have"]) & (df["currently_active"])]
    new_inactive = df[(~df["already_have"]) & (~df["currently_active"])]

    print(f"\n── Already in your dataset ({len(already)}) ──")
    print(already[["code", "name", "distance_km", "type", "setting"]]
          .to_string(index=False))

    print(f"\n── NEW candidates, currently active ({len(new_active)}) ──")
    if len(new_active) > 0:
        print(new_active[["code", "name", "city", "distance_km", "type", "setting"]]
              .to_string(index=False))
    else:
        print("  (none found)")

    print(f"\n── NEW candidates, but INACTIVE / historical only ({len(new_inactive)}) ──")
    print("  (these stopped reporting at some point — don't use for 2019-2024 data)")
    if len(new_inactive) > 0:
        print(new_inactive[["code", "name", "city", "distance_km", "active_to"]]
              .head(10).to_string(index=False))

    out_dir = Path("munich_aq_final")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "nearby_station_candidates.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Full results saved to {out_path}")

    if len(new_active) > 0:
        print(f"""
NEXT STEP:
If any of the {len(new_active)} new active candidates above look useful
(e.g. fills your 12-30km coverage gap), add its CODE to STATIONS_MAP in
your existing fetch_munich_aq_final.py, then re-run:

  python fetch_munich_aq_final.py --validate-only

to confirm it actually has PM2.5 data before doing a full fetch. Not
every station measures every pollutant — validation catches that before
you spend time on a full 2019-2024 download.
        """)
    else:
        print(f"""
No new ACTIVE stations found within {radius_km} km beyond your current 8.
Try a larger --radius if you want to look further out, but be aware
this trades spatial coverage for weaker correlation with Munich (as
your spatial mismatch analysis already showed for Augsburg/Andechs).
        """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=60,
                         help="Search radius in km from Munich center")
    args = parser.parse_args()
    run_discovery(radius_km=args.radius)