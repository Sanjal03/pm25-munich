"""
Munich-Area Air Quality Fetcher — UBA API v2 measures endpoint (FINAL)
=======================================================================
Fetches PM2.5, PM10, NO2, O3 in REAL µg/m³ for Munich-area stations.

ROOT CAUSE OF ALL PREVIOUS FAILURES — now fixed:
  ✗ Wrong base URL: luftdaten.umweltbundesamt.de/api-proxy/measures
                    → redirects, ignores all params, returns error page
  ✓ Correct URL:    www.umweltbundesamt.de/api/air_data/v2/measures/json

Response format confirmed by live test:
  data[station_id][datetime_str] = [comp_id, scope_id, VALUE, date_end, index]
  VALUE at index 2 — real µg/m³ (O3 ~30–70, NO2 ~10–50, PM2.5 ~5–30)
  null = missing measurement

Component IDs (v2, confirmed working):
  1 = PM10
  3 = O3
  5 = NO2
  9 = PM2.5

Scope: 2 = hourly mean (1SMW)

Date range: 2019-01-01 to 2024-12-31
  Train: 2019–2023 | Test: 2024

Stations (8 fetched; 7 used in the final model panel — Allach has no
PM2.5 sensor, but its NO2/O3 are fetched because 02_preprocess.py uses them
as spatial neighbours when imputing 7-24h gaps at the other stations):
  DEBY115  München/Landshuter Allee  traffic hotspot   (highest PM2.5)
  DEBY037  München/Stachus           traffic urban
  DEBY039  München/Lothstraße        urban background
  DEBY089  München/Johanneskirchen   suburban
  DEBY189  München/Allach            suburban west (no PM2.5 sensor — excluded from modeling)
  DEBY109  Andechs/Rothenfeld        rural background 35km SW
  DEBY007  Augsburg/Bourges-Platz    urban 70km W
  DEBY099  Augsburg/LfU              suburban 70km W

Burghausen (DEBY012) and Oberaudorf (DEBY121) were tested as a 9-station
extension and are kept on the `extension-9-stations` branch (future work).

Usage:
  pip install requests pandas
  python pipeline/01_fetch_data.py --validate-only   # probe all stations first
  python pipeline/01_fetch_data.py                    # fetch 2019-2024
  python pipeline/01_fetch_data.py --no-resume        # force re-download
"""

import argparse
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── CORRECT API endpoints (confirmed working June 2026) ───────────────────────
# measures/json: returns real µg/m³ concentrations (not AQI index)
# stations/json: resolves DEBY codes → numeric IDs
MEASURES_URL = "https://www.umweltbundesamt.de/api/air_data/v2/measures/json"
STATIONS_URL = "https://www.umweltbundesamt.de/api/air_data/v2/stations/json"

# ── Component IDs & Scope (v2, live-tested) ───────────────────────────────────
COMP_MAP: dict[int, str] = {
    1: "PM10",
    3: "O3",
    5: "NO2",
    9: "PM2.5",
}
SCOPE_HOURLY = 2   # 1SMW = hourly mean

# ── Stations ──────────────────────────────────────────────────────────────────
# (name, environment_type, distance_from_munich_km)
STATIONS_MAP: dict[str, tuple[str, str, int]] = {
    "DEBY115": ("München/Landshuter Allee",  "traffic",           3),
    "DEBY037": ("München/Stachus",           "traffic-urban",     1),
    "DEBY039": ("München/Lothstraße",        "urban-background",  3),
    "DEBY089": ("München/Johanneskirchen",   "suburban",         12),
    "DEBY189": ("München/Allach",            "suburban",         14),
    "DEBY109": ("Andechs/Rothenfeld",        "rural-background", 35),
    "DEBY007": ("Augsburg/Bourges-Platz",    "urban-background", 70),
    "DEBY099": ("Augsburg/LfU",              "suburban",         70),
}

# ── HTTP session with retry ───────────────────────────────────────────────────
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (AirQualityResearch/1.0)",
        "Accept": "application/json",
    })
    retry = Retry(
        total=5, backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

SESSION = _session()


# ── Resolve DEBY codes → numeric station IDs ──────────────────────────────────
def resolve_ids(output_dir: Path) -> dict[str, str]:
    """Returns {DEBY_code: numeric_id_string}"""
    log.info("Resolving station IDs …")
    try:
        resp = SESSION.get(
            STATIONS_URL,
            params={"use": "airquality", "lang": "de",
                    "date_from": "2023-01-01", "time_from": "1",
                    "date_to": "2023-12-31", "time_to": "24"},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        log.error("Could not fetch station list: %s", exc)
        return {}

    import json
    (output_dir / "stations_raw.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # payload["data"] = {numeric_id: [fields...]}
    stations_data = payload.get("data", {})
    id_map: dict[str, str] = {}

    for numeric_id, fields in stations_data.items():
        fields_str = str(fields)
        for deby in STATIONS_MAP:
            if deby in fields_str and deby not in id_map:
                id_map[deby] = str(numeric_id)
                break

    log.info("  Resolved %d / %d", len(id_map), len(STATIONS_MAP))
    missing = [c for c in STATIONS_MAP if c not in id_map]
    if missing:
        log.warning("  Unresolved: %s", missing)

    pd.DataFrame([
        {"DEBY": k, "ID": v,
         "Name": STATIONS_MAP[k][0],
         "Type": STATIONS_MAP[k][1],
         "DistKm": STATIONS_MAP[k][2]}
        for k, v in id_map.items()
    ]).to_csv(output_dir / "station_id_map.csv", index=False)

    return id_map


# ── Fetch one station × one component × one date range ───────────────────────
def fetch_one(
    numeric_id: str,
    deby_code:  str,
    comp_id:    int,
    date_from:  str,
    date_to:    str,
) -> list[dict]:
    """
    Calls MEASURES_URL and parses real µg/m³ values.

    Confirmed response format (live test 2026-06-28):
      data[numeric_id][datetime_str] = [comp_id, scope_id, VALUE, date_end, index]
      VALUE (index 2) is the actual concentration in µg/m³, or null if missing.
    """
    try:
        resp = SESSION.get(
            MEASURES_URL,
            params={
                "station":    numeric_id,
                "component":  comp_id,
                "scope":      SCOPE_HOURLY,
                "date_from":  date_from,
                "date_to":    date_to,
                "time_from":  "1",
                "time_to":    "24",
                "lang":       "en",
            },
            timeout=90,
        )
    except requests.exceptions.RequestException as exc:
        log.debug("Network error %s comp=%d: %s", deby_code, comp_id, exc)
        return []

    if resp.status_code != 200:
        log.debug("HTTP %d for %s comp=%d", resp.status_code, deby_code, comp_id)
        return []

    try:
        payload = resp.json()
    except ValueError:
        log.debug("Invalid JSON for %s comp=%d", deby_code, comp_id)
        return []

    hourly = payload.get("data", {}).get(numeric_id, {})
    if not hourly:
        log.debug("No data key '%s' for %s comp=%d", numeric_id, deby_code, comp_id)
        return []

    name, env, dist = STATIONS_MAP.get(deby_code, (deby_code, "unknown", 0))
    pollutant = COMP_MAP[comp_id]
    records = []

    for dt_str, entry in hourly.items():
        # entry = [comp_id, scope_id, value, date_end, index]
        if not isinstance(entry, list) or len(entry) < 3:
            continue
        raw = entry[2]          # index 2 = actual µg/m³ value
        if raw is None:
            continue            # missing measurement — skip, don't fill
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value < 0:
            continue            # negative = sensor error

        records.append({
            "Station Name":  name,
            "Station Code":  deby_code,
            "Environment":   env,
            "Distance (km)": dist,
            "Pollutant":     pollutant,
            "Component ID":  comp_id,
            "Datetime":      dt_str,
            "Value (µg/m³)": round(value, 3),
        })

    return records


# ── Validation ────────────────────────────────────────────────────────────────
def validate(id_map: dict[str, str], pause: float, output_dir: Path) -> dict[str, str]:
    log.info("=" * 70)
    log.info("VALIDATION  %d stations × %d pollutants  (probe: 2023-06-01→07)",
             len(id_map), len(COMP_MAP))
    log.info("=" * 70)

    valid: dict[str, str] = {}
    rows = []

    for deby, num_id in id_map.items():
        name, env, dist = STATIONS_MAP[deby]
        found = []
        for comp_id, pol in COMP_MAP.items():
            recs = fetch_one(num_id, deby, comp_id, "2023-06-01", "2023-06-07")
            if recs:
                mean_v = sum(r["Value (µg/m³)"] for r in recs) / len(recs)
                log.info("  %-10s  %-35s  %-5s  %3dh  mean=%.1f µg/m³",
                         deby, name[:35], pol, len(recs), mean_v)
                found.append(pol)
            else:
                log.info("  %-10s  %-35s  %-5s  NO DATA", deby, name[:35], pol)
            time.sleep(pause)

        if found:
            valid[deby] = num_id
        rows.append({
            "DEBY": deby, "ID": num_id, "Name": name, "Type": env,
            "Dist(km)": dist, "Found": ", ".join(found) or "NONE",
            "HasPM25": "PM2.5" in found, "HasNO2": "NO2" in found,
        })

    log.info("-" * 70)
    log.info("Valid: %d/%d  |  PM2.5: %d  |  NO2: %d",
             len(valid), len(id_map),
             sum(1 for r in rows if r["HasPM25"]),
             sum(1 for r in rows if r["HasNO2"]))
    log.info("=" * 70)

    pd.DataFrame(rows).to_csv(output_dir / "station_validation.csv", index=False)
    log.info("Saved → station_validation.csv")
    return valid


# ── Month iterator ────────────────────────────────────────────────────────────
def _months(start: date, end: date):
    cur = start.replace(day=1)
    while cur <= end:
        nxt = (date(cur.year + 1, 1, 1) if cur.month == 12
               else cur.replace(month=cur.month + 1))
        yield cur, min(nxt - timedelta(days=1), end)
        cur = nxt


# ── Main fetch ────────────────────────────────────────────────────────────────
def fetch_all(
    start: date, end: date,
    output_dir: Path,
    valid: dict[str, str],
    resume: bool = True,
    pause: float = 0.5,
) -> Path:
    chunks = output_dir / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)

    months    = list(_months(start, end))
    n_calls   = len(months) * len(valid) * len(COMP_MAP)
    est_mins  = n_calls * (pause + 0.7) / 60

    log.info("=" * 70)
    log.info("FETCH  %s → %s  |  %d months × %d stations × %d pollutants",
             start, end, len(months), len(valid), len(COMP_MAP))
    log.info("~%d API calls  |  estimated ~%.0f min", n_calls, est_mins)
    log.info("=" * 70)

    for i, (m0, m1) in enumerate(months, 1):
        label = m0.strftime("%Y-%m")
        chunk = chunks / f"chunk_{label}.csv"

        if resume and chunk.exists() and chunk.stat().st_size > 200:
            log.info("[%3d/%d] %s  cached ✓", i, len(months), label)
            continue

        log.info("[%3d/%d] %s", i, len(months), label)
        rows: list[dict] = []

        for deby, num_id in valid.items():
            name = STATIONS_MAP[deby][0]
            for comp_id, pol in COMP_MAP.items():
                recs = fetch_one(num_id, deby, comp_id,
                                 m0.strftime("%Y-%m-%d"),
                                 m1.strftime("%Y-%m-%d"))
                if recs:
                    log.info("  %-10s  %-35s  %-5s  %3d rows",
                             deby, name[:35], pol, len(recs))
                    rows.extend(recs)
                else:
                    log.debug("  %-10s  %-35s  %-5s  0", deby, name[:35], pol)
                time.sleep(pause)

        if rows:
            (pd.DataFrame(rows)
               .sort_values(["Station Name", "Pollutant", "Datetime"])
               .to_csv(chunk, index=False))
            log.info("  → %s  (%d rows)", chunk.name, len(rows))
        else:
            chunk.touch()
            log.warning("  → no data for %s", label)

    # Merge
    log.info("Merging …")
    files = [f for f in sorted(chunks.glob("chunk_*.csv")) if f.stat().st_size > 200]
    if not files:
        raise RuntimeError("No data fetched — run --validate-only to debug")

    master = (
        pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
          .sort_values(["Station Name", "Pollutant", "Datetime"])
          .reset_index(drop=True)
    )
    out = output_dir / f"Munich_AQ_{start}_{end}.csv"
    master.to_csv(out, index=False)

    log.info("=" * 70)
    log.info("DONE  →  %s", out.name)
    log.info("Rows       : %d", len(master))
    log.info("Stations   : %d", master["Station Code"].nunique())
    log.info("Pollutants : %s", sorted(master["Pollutant"].unique()))
    log.info("Date range : %s → %s",
             master["Datetime"].min(), master["Datetime"].max())
    log.info("=" * 70)

    # Sanity check — values should be real µg/m³, not 0-5 index
    log.info("SANITY CHECK (expected: PM2.5~10, NO2~20, O3~50 µg/m³):")
    for pol in ["PM2.5", "PM10", "NO2", "O3"]:
        sub = master[master["Pollutant"] == pol]["Value (µg/m³)"]
        if sub.empty:
            log.warning("  %-5s  NO DATA", pol)
        else:
            ok = "✓" if sub.mean() > 3 else "✗ still looks like index values"
            log.info("  %-5s  mean=%6.1f  max=%6.0f  p99=%5.0f  %s",
                     pol, sub.mean(), sub.max(), sub.quantile(0.99), ok)

    summary = (
        master.groupby(["Station Code", "Station Name",
                        "Environment", "Distance (km)", "Pollutant"])
              .agg(n=("Value (µg/m³)", "count"),
                   mean=("Value (µg/m³)", "mean"),
                   max=("Value (µg/m³)", "max"))
              .round(1).reset_index()
    )
    summary.to_csv(output_dir / "coverage_summary.csv", index=False)
    log.info("\n%s", summary.to_string(index=False))

    return out


# ── CLI ───────────────────────────────────────────────────────────────────────
def _args():
    p = argparse.ArgumentParser(
        description="Fetch Munich-area PM2.5/PM10/NO2/O3 in µg/m³ — UBA v2 measures API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start",           default="2019-01-01")
    p.add_argument("--end",             default="2024-12-31")
    p.add_argument("--output-dir",      default="./munich_aq_final")
    p.add_argument("--no-resume",       action="store_true",
                   help="Ignore cached chunks, re-download all")
    p.add_argument("--validate-only",   action="store_true",
                   help="Probe all stations then exit — do this first")
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--pause",           type=float, default=0.5,
                   help="Seconds between API calls (be polite to the server)")
    p.add_argument("--debug",           action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_map = resolve_ids(out_dir)
    if not id_map:
        log.error("No station IDs resolved. Check internet connection.")
        raise SystemExit(1)

    if args.skip_validation:
        valid = id_map
        log.info("Skipping validation — using all %d stations", len(valid))
    else:
        valid = validate(id_map, args.pause, out_dir)

    if args.validate_only:
        log.info("Validation done. Check station_validation.csv")
        raise SystemExit(0)

    if not valid:
        log.error("No valid stations after validation.")
        raise SystemExit(1)

    fetch_all(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        output_dir=out_dir,
        valid=valid,
        resume=not args.no_resume,
        pause=args.pause,
    )