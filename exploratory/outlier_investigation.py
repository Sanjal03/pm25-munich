"""
02_outlier_investigation.py  (v3 — focused on 2023–2025, with visual context)
================================================================================
Finds extreme PM2.5 readings in the recent period and cross-checks whether
other stations spiked at the same time (real event) or it was isolated
(likely sensor error). Also plots the raw timeline (±48h) around each
event across ALL stations, so you can visually inspect it yourself —
not just read a verdict.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Inlined data loader (previously a separate data_loader.py — merged
#    in directly so this script has no other-file dependency) ─────────
CSV_PATH = Path("munich_aq_final/Munich_AQ_2019-01-01_2024-12-31.csv")
_CSV_FALLBACKS = [
    CSV_PATH,
    Path("Munich_AQ_2019-01-01_2024-12-31.csv"),
    Path("../munich_aq_final/Munich_AQ_2019-01-01_2024-12-31.csv"),
]
STATION_CODES = ["DEBY109", "DEBY007", "DEBY099", "DEBY189",
                  "DEBY089", "DEBY115", "DEBY039", "DEBY037",
                  "DEBY012", "DEBY121"]
STATION_NAMES = {
    "DEBY109": "Andechs/Rothenfeld", "DEBY007": "Augsburg/Bourges-Platz",
    "DEBY099": "Augsburg/LfU", "DEBY189": "München/Allach",
    "DEBY089": "München/Johanneskirchen", "DEBY115": "München/Landshuter Allee",
    "DEBY039": "München/Lothstraße", "DEBY037": "München/Stachus",
    "DEBY012": "Burghausen/Marktler Straße", "DEBY121": "Oberaudorf/Inntal-Autobahn",
}


def load_raw(csv_path=None, nrows=None):
    candidates = [csv_path] + _CSV_FALLBACKS if csv_path else _CSV_FALLBACKS
    for candidate in candidates:
        if candidate is not None and Path(candidate).exists():
            print(f"  (reading {Path(candidate).resolve()})")
            df = pd.read_csv(candidate, nrows=nrows)
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            return df
    tried = "\n    ".join(str(c) for c in candidates if c is not None)
    raise FileNotFoundError(
        f"Could not find the CSV. Tried:\n    {tried}\n"
        f"Edit CSV_PATH near the top of this script to point at your actual file."
    )


def pivot_station(df, station_code):
    sub = df[df["Station Code"] == station_code]
    wide = sub.pivot_table(index="Datetime", columns="Pollutant",
                            values="Value (µg/m³)", aggfunc="first")
    return wide.sort_index()


def pivot_all_stations(df, station_codes=STATION_CODES):
    return {code: pivot_station(df, code) for code in station_codes}
# ── end inlined loader ──────────────────────────────────────────────

EXTREME_THRESHOLD = 200
HIGH_THRESHOLD = 150

# Focus window — you flagged 2023-2025 specifically. Your data only runs
# through 2024-12-31, so this covers everything from 2023 onward that
# actually exists. Set FOCUS_START to None to scan the full 2019-2024
# range instead.
FOCUS_START = "2023-01-01"
FOCUS_END = None  # None = through the end of the data

CONTEXT_HOURS = 48   # how much timeline to show either side of each spike
MAX_PLOTS = 20        # cap on how many event plots to generate

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def is_nye_event(timestamp):
    month, day = timestamp.month, timestamp.day
    return (month == 12 and day == 31) or (month == 1 and day == 1)


def build_pm25_matrix(raw):
    """One column per station, PM2.5 only, indexed by Datetime."""
    pm25 = raw[raw["Pollutant"] == "PM2.5"]
    matrix = pm25.pivot_table(index="Datetime", columns="Station Code",
                               values="Value (µg/m³)", aggfunc="first")
    return matrix.sort_index()


def get_copollutant_profile(raw, station_code, timestamp, window_hours=1):
    lo = timestamp - pd.Timedelta(hours=window_hours)
    hi = timestamp + pd.Timedelta(hours=window_hours)
    window = raw[(raw["Station Code"] == station_code) &
                 (raw["Datetime"] >= lo) & (raw["Datetime"] <= hi) &
                 (raw["Pollutant"] != "PM2.5")]
    profile = {}
    for pol in ["PM10", "NO2", "O3"]:
        vals = window[window["Pollutant"] == pol]["Value (µg/m³)"]
        if len(vals) > 0:
            profile[pol] = {"mean": round(vals.mean(), 1), "max": round(vals.max(), 1)}
    return profile


def investigate_event(pm25_matrix, raw, station_code, timestamp, pm25_value):
    is_nye = is_nye_event(timestamp)

    lo = timestamp - pd.Timedelta(hours=3)
    hi = timestamp + pd.Timedelta(hours=3)
    window = pm25_matrix.loc[lo:hi]
    spike_per_station = window.max()
    max_other = spike_per_station.drop(station_code, errors="ignore").max()
    n_stations_high = (spike_per_station > EXTREME_THRESHOLD).sum()

    copol = get_copollutant_profile(raw, station_code, timestamp)

    if n_stations_high >= 4:
        verdict, reason = "CONFIRMED_REAL_EVENT", f"{n_stations_high} stations spiked >200"
    elif n_stations_high >= 2:
        verdict, reason = "LIKELY_REAL", f"{n_stations_high} stations spiked >200"
    elif pd.notna(max_other) and max_other > 100:
        verdict, reason = "LIKELY_REAL", f"other stations elevated (max={max_other:.1f})"
    else:
        verdict, reason = "LIKELY_SENSOR_ERROR", "isolated spike, others unremarkable"

    return {
        "station": station_code, "timestamp": timestamp, "pm25_value": pm25_value,
        "is_nye": is_nye, "verdict": verdict, "reason": reason,
        "max_other_stations": max_other, "n_stations_high": n_stations_high,
        "copollutants": copol,
    }


def plot_event_timeline(pm25_matrix, station_code, timestamp, event_num):
    """
    Plot the raw PM2.5 timeline across ALL stations for a window around
    this event, so you can visually see whether it's a shared spike
    (real event) or isolated to one station (likely sensor glitch).
    This is the "check the original timeline" step.
    """
    lo = timestamp - pd.Timedelta(hours=CONTEXT_HOURS)
    hi = timestamp + pd.Timedelta(hours=CONTEXT_HOURS)
    window = pm25_matrix.loc[lo:hi]

    fig, ax = plt.subplots(figsize=(11, 5))
    for code in window.columns:
        style = {"linewidth": 2.5, "zorder": 5} if code == station_code else \
                {"linewidth": 1, "alpha": 0.6, "zorder": 1}
        ax.plot(window.index, window[code],
                label=STATION_NAMES.get(code, code), **style)

    ax.axvline(timestamp, color="red", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title(f"Event {event_num}: {STATION_NAMES.get(station_code, station_code)} "
                 f"— {timestamp.strftime('%Y-%m-%d %H:%M')} "
                 f"(±{CONTEXT_HOURS}h context)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fname = RESULTS_DIR / f"outlier_event_{event_num:02d}_{station_code}.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def run_investigation():
    print("\n" + "="*90)
    print(" EXTREME PM2.5 OUTLIER INVESTIGATION")
    print("="*90)

    print("\nLoading data...")
    raw = load_raw()
    pm25_matrix = build_pm25_matrix(raw)
    print(f"✓ PM2.5 matrix: {pm25_matrix.shape[0]:,} timestamps × {pm25_matrix.shape[1]} stations")

    focus_matrix = pm25_matrix
    if FOCUS_START is not None:
        focus_matrix = pm25_matrix.loc[FOCUS_START:FOCUS_END]
        print(f"  Focusing on {FOCUS_START} → "
              f"{FOCUS_END or pm25_matrix.index.max().date()} "
              f"({len(focus_matrix):,} timestamps)")

    flat = focus_matrix.stack()
    flat.index.names = ["Datetime", "Station Code"]
    extremes = flat[flat > EXTREME_THRESHOLD].sort_values(ascending=False)

    print(f"\n✓ Found {len(extremes)} readings > {EXTREME_THRESHOLD} µg/m³ in this window")
    print(f"  PM2.5 range in this window: "
          f"{focus_matrix.min().min():.1f}–{focus_matrix.max().max():.1f} µg/m³")

    if len(extremes) == 0:
        print(f"\n✓ No readings above {EXTREME_THRESHOLD} µg/m³ in "
              f"{FOCUS_START or 'the full range'} onward.")
        print(f"  Trying the lower threshold ({HIGH_THRESHOLD} µg/m³) instead...")
        flat_high = focus_matrix.stack()
        flat_high.index.names = ["Datetime", "Station Code"]
        extremes = flat_high[flat_high > HIGH_THRESHOLD].sort_values(ascending=False)
        if len(extremes) == 0:
            print(f"  ✓ Nothing above {HIGH_THRESHOLD} µg/m³ either — this period looks clean.")
            return
        print(f"  ✓ Found {len(extremes)} readings > {HIGH_THRESHOLD} µg/m³")

    print("\n" + "="*90)
    print(" INVESTIGATION OF EACH EXTREME EVENT")
    print("="*90)

    investigations = []
    plot_paths = []
    for i, ((timestamp, station_code), value) in enumerate(extremes.items(), 1):
        inv = investigate_event(pm25_matrix, raw, station_code, timestamp, value)
        investigations.append(inv)
        if i <= MAX_PLOTS:
            path = plot_event_timeline(pm25_matrix, station_code, timestamp, i)
            plot_paths.append(path)

    for i, inv in enumerate(investigations, 1):
        name = STATION_NAMES.get(inv["station"], inv["station"])
        print(f"\n{i}. {name} ({inv['station']}) — {inv['timestamp']}")
        print(f"   PM2.5: {inv['pm25_value']:.1f} µg/m³  |  "
              f"{'NYE/NYD' if inv['is_nye'] else 'Regular day'}")
        print(f"   Verdict: {inv['verdict']} — {inv['reason']}")
        if inv["copollutants"]:
            for pol, v in inv["copollutants"].items():
                print(f"     {pol}: mean={v['mean']}  max={v['max']}")

    inv_df = pd.DataFrame(investigations)
    Path("results").mkdir(exist_ok=True)
    inv_df.to_csv(Path("results") / "outlier_investigation_verdicts.csv", index=False)
    print(f"\n✓ Saved verdicts to results/outlier_investigation_verdicts.csv")

    print("\n" + "="*90)
    print(" STATISTICS")
    print("="*90)
    print("\nVerdict breakdown:")
    print(inv_df["verdict"].value_counts().to_string())
    print("\nAffected stations:")
    print(inv_df["station"].map(STATION_NAMES).value_counts().to_string())
    nye_count = inv_df["is_nye"].sum()
    print(f"\nNYE/NYD events: {nye_count} of {len(inv_df)}")

    print("""
RECOMMENDATION FOR THESIS:
Add a sentence to your Data Integration section noting how many
extreme events were cross-validated as real (multi-station) vs.
flagged as likely single-station sensor errors, per the verdicts
above. Consider adding a boolean "suspected_sensor_error" column in
preprocessing for the isolated cases rather than deleting them.
    """)

    if plot_paths:
        print(f"✓ Saved {len(plot_paths)} timeline plots to results/ "
              f"(outlier_event_01_... through outlier_event_{len(plot_paths):02d}_...)")
        print("  Open a few of these — for a CONFIRMED_REAL_EVENT, you should see")
        print("  multiple stations' lines spike together around the red dashed line.")
        print("  For a LIKELY_SENSOR_ERROR, only one line should jump while the")
        print("  others stay flat — that visual gap is the actual evidence, the")
        print("  verdict text above is just a summary of it.")
        if len(extremes) > MAX_PLOTS:
            print(f"  ({len(extremes) - MAX_PLOTS} additional events found but not "
                  f"plotted — raise MAX_PLOTS at the top of the script if needed.)")


if __name__ == "__main__":
    run_investigation()