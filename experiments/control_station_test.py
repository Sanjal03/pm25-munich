"""
control_station_test.py — Nürnberg/Stuttgart control test (Appendix)
====================================================================
Does PM2.5 correlation with the reference station (München/Landshuter
Allee) fall off once you leave the Munich region entirely? Two stations in
clearly different regional airsheds serve as controls:

  Nürnberg/Muggenhof        DEBY058  urban background, Bavaria (BY)
  Stuttgart-Bad Cannstatt   DEBW013  urban background, Baden-Württemberg (BW)

Both are compared with the 7 study stations on the same metrics:
  - PM2.5 completeness over 2019-2024 (hourly)
  - Pearson r of daily-mean PM2.5 vs the reference (raw)
  - Pearson r after removing each station's own monthly climatology
    (deseasonalized) — separates day-to-day regional coupling from the
    shared "high in winter, low in summer" cycle every German city has

If the controls correlate about as well as the study stations, correlation
with the reference cannot by itself tell a regionally relevant station from
one that merely shares central-European weather and seasonality — the
thesis's argument for keeping the supervisor-confirmed 7-station scope.

The control stations are not modelled: DWD station 03379 (Munich) weather
is not valid for Nürnberg or Stuttgart.

Data: the controls are fetched once via fetch_one() from
pipeline/01_fetch_data.py into munich_aq_final/_control_validation/ and
reused from there afterwards (no API calls on re-runs).

Output:
  results/control_station_test.csv
  thesis_figs/figC1_control_correlation_vs_distance.png
  thesis_figs/figC2_control_monthly_pm25.png
"""

import importlib.util
import logging
import time
from datetime import date, timedelta
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REFERENCE_CSV = Path("munich_aq_final/Munich_AQ_2019-01-01_2024-12-31.csv")
REFERENCE = "München/Landshuter Allee"
REF_LATLON = (48.1496, 11.5365)
CACHE_DIR = Path("munich_aq_final/_control_validation")
OUT_FIGS = Path("thesis_figs")
N_EXPECTED = len(pd.date_range("2019-01-01", "2024-12-31 23:00", freq="h"))  # 52608

# name: (lat, lon, environment) — coordinates from the UBA station list
STUDY = {
    "München/Stachus":         (48.1373, 11.5648, "Traffic"),
    "München/Lothstraße":      (48.1546, 11.5547, "Urban background"),
    "München/Johanneskirchen": (48.1732, 11.6480, "Suburban"),
    "Andechs/Rothenfeld":      (47.9688, 11.2202, "Rural background"),
    "Augsburg/Bourges-Platz":  (48.3766, 10.8884, "Urban background"),
    "Augsburg/LfU":            (48.3260, 10.9031, "Suburban"),
}
CONTROLS = {
    "DEBY058": {"id": "492", "name": "Nürnberg/Muggenhof", "env": "urban-background",
                "latlon": (49.4622, 11.0248), "state": "BY"},
    "DEBW013": {"id": "224", "name": "Stuttgart-Bad Cannstatt", "env": "urban-background",
                "latlon": (48.8088, 9.2297), "state": "BW"},
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fbfaf7",
    "font.family": "serif",
    "font.size": 11.5, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.edgecolor": "#2b2b2b", "axes.linewidth": 1.1,
})
TYPE_COLORS = {"Traffic": "#c9782e", "Urban background": "#2b6a99",
               "Suburban": "#1f8a70", "Rural background": "#5c8a3a"}
CONTROL_COLOR = "#a63328"
LABEL_OFFSET = {"München/Lothstraße": (8, 2), "München/Stachus": (8, -12),
                "Augsburg/LfU": (-62, 6), "Augsburg/Bourges-Platz": (8, -12)}


def haversine_km(a, b):
    (lat1, lon1), (lat2, lon2) = a, b
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    h = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * atan2(sqrt(h), sqrt(1 - h))


def _cache_path(name: str) -> Path:
    safe = name.split(",")[0].lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}_full_2019_2024.csv"


def _months(start: date, end: date):
    cur = start.replace(day=1)
    while cur <= end:
        nxt = date(cur.year + 1, 1, 1) if cur.month == 12 else cur.replace(month=cur.month + 1)
        yield cur, min(nxt - timedelta(days=1), end)
        cur = nxt


def load_control(code: str, info: dict, pause: float = 0.5) -> pd.DataFrame:
    path = _cache_path(info["name"])
    if path.exists():
        return pd.read_csv(path)

    spec = importlib.util.spec_from_file_location(
        "fetch_data", Path(__file__).resolve().parent.parent / "pipeline" / "01_fetch_data.py")
    fetch_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch_data)
    fetch_data.STATIONS_MAP[code] = (info["name"], info["env"],
                                     round(haversine_km(REF_LATLON, info["latlon"])))
    rows = []
    months = list(_months(date(2019, 1, 1), date(2024, 12, 31)))
    log.info("Fetching %s (%s): %d months x 4 pollutants", info["name"], code, len(months))
    for m0, m1 in months:
        for comp_id in fetch_data.COMP_MAP:
            rows.extend(fetch_data.fetch_one(info["id"], code, comp_id,
                                             m0.strftime("%Y-%m-%d"), m1.strftime("%Y-%m-%d")))
            time.sleep(pause)
    df = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def hourly_pm25(df: pd.DataFrame, station: str | None = None) -> pd.Series:
    if station is not None:
        df = df[df["Station Name"] == station]
    df = df[df["Pollutant"] == "PM2.5"]
    s = df.set_index(pd.to_datetime(df["Datetime"]))["Value (µg/m³)"]
    return s[~s.index.duplicated()]


def deseasonalize(daily: pd.Series) -> pd.Series:
    return daily - daily.groupby(daily.index.month).transform("mean")


def compare(ref_daily: pd.Series, hourly: pd.Series) -> dict:
    daily = hourly.resample("D").mean()
    joined = pd.concat([ref_daily, daily], axis=1, join="inner").dropna()
    joined.columns = ["ref", "st"]
    return {
        "pm25_completeness_pct": round(100 * hourly.notna().sum() / N_EXPECTED, 1),
        "mean_pm25": round(hourly.mean(), 2),
        "n_days": len(joined),
        "r_daily": round(joined["ref"].corr(joined["st"]), 3),
        "r_deseasonalized": round(deseasonalize(joined["ref"]).corr(deseasonalize(joined["st"])), 3),
    }


def make_scatter(out: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for _, r in out.iterrows():
        control = r["group"] == "control"
        color = CONTROL_COLOR if control else TYPE_COLORS[r["environment"]]
        ax.scatter(r["distance_km"], r["r_daily"], s=150 if control else 110,
                   marker="D" if control else "o", color=color,
                   edgecolor="white", linewidth=1.2, zorder=3)
        label = r["station"].replace("München/", "").replace("Augsburg/", "Aug./")
        ax.annotate(label, (r["distance_km"], r["r_daily"]),
                    xytext=LABEL_OFFSET.get(r["station"], (8, -4)),
                    textcoords="offset points", fontsize=9.5)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=10, label=t) for t, c in TYPE_COLORS.items()]
    handles.append(plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=CONTROL_COLOR,
                              markersize=10, label="Control (outside study region)"))
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.95, edgecolor="#2b2b2b")
    ax.set_xlabel("Distance from München/Landshuter Allee (km)")
    ax.set_ylabel("Pearson r, daily-mean PM2.5")
    ax.set_title("Control Test: Correlation vs. Distance")
    ax.set_xlim(-5, out["distance_km"].max() + 35)
    ax.set_ylim(0.6, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / "figC1_control_correlation_vs_distance.png", dpi=200)
    plt.close(fig)


def make_monthly(series: dict[str, pd.Series]):
    fig, ax = plt.subplots(figsize=(11, 5))
    styles = {REFERENCE: ("#1a1a1a", "-"), "Nürnberg/Muggenhof": (CONTROL_COLOR, "--"),
              "Stuttgart-Bad Cannstatt": ("#6f4e91", "-.")}
    for name, s in series.items():
        color, ls = styles[name]
        ax.plot(s.resample("MS").mean(), color=color, linestyle=ls, linewidth=1.8, label=name)
    ax.set_ylabel("Monthly mean PM2.5 (µg/m³)")
    ax.set_title("Reference vs. Control Stations — Monthly Mean PM2.5, 2019–2024")
    ax.legend(fontsize=9.5, framealpha=0.95, edgecolor="#2b2b2b")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / "figC2_control_monthly_pm25.png", dpi=200)
    plt.close(fig)


def run():
    main = pd.read_csv(REFERENCE_CSV)
    ref_hourly = hourly_pm25(main, REFERENCE)
    ref_daily = ref_hourly.resample("D").mean()

    rows = []
    for name, (lat, lon, env) in STUDY.items():
        rows.append({"station": name, "group": "study", "environment": env,
                     "distance_km": round(haversine_km(REF_LATLON, (lat, lon)), 1),
                     **compare(ref_daily, hourly_pm25(main, name))})

    monthly = {REFERENCE: ref_hourly}
    for code, info in CONTROLS.items():
        hourly = hourly_pm25(load_control(code, info))
        monthly[info["name"]] = hourly
        rows.append({"station": info["name"], "group": "control", "environment": "Urban background",
                     "distance_km": round(haversine_km(REF_LATLON, info["latlon"]), 1),
                     **compare(ref_daily, hourly)})

    out = pd.DataFrame(rows).sort_values("distance_km")
    out.to_csv("results/control_station_test.csv", index=False)
    print(out.to_string(index=False))

    OUT_FIGS.mkdir(exist_ok=True)
    make_scatter(out)
    make_monthly(monthly)

    study = out[out["group"] == "study"]
    print(f"\nStudy stations: r_daily {study['r_daily'].min():.3f}–{study['r_daily'].max():.3f}, "
          f"deseasonalized {study['r_deseasonalized'].min():.3f}–{study['r_deseasonalized'].max():.3f}")
    print("✓ Saved: results/control_station_test.csv, thesis_figs/figC1_*.png, thesis_figs/figC2_*.png")


if __name__ == "__main__":
    run()
