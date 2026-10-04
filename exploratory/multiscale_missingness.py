"""
multiscale_missingness.py
======================================================================
Hour-of-day and day-of-week missingness views for PM2.5, to complement
your existing monthly heatmap (Figure 3.3).

IMPORTANT: your raw CSV has no explicit "missing" rows — a gap means
a (station, pollutant, hour) combination is simply ABSENT from the
file rather than present with a NaN value. So "missing" here is
computed by reindexing each station's PM2.5 series onto the full
expected hourly range and counting what's absent.
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
STATION_CODES = ["DEBY109", "DEBY007", "DEBY099",
                  "DEBY089", "DEBY115", "DEBY039", "DEBY037"]
STATION_NAMES = {
    "DEBY109": "Andechs/Rothenfeld", "DEBY007": "Augsburg/Bourges-Platz",
    "DEBY099": "Augsburg/LfU",
    "DEBY089": "München/Johanneskirchen", "DEBY115": "München/Landshuter Allee",
    "DEBY039": "München/Lothstraße", "DEBY037": "München/Stachus",
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

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def reindex_full_range(series, start="2019-01-01 00:00:00", end="2024-12-31 23:00:00"):
    full_range = pd.date_range(start, end, freq="h")
    return series.reindex(full_range)


def missingness_by_hour(pm25_dict):
    rows = {}
    for code, s in pm25_dict.items():
        full = reindex_full_range(s)
        rows[code] = full.groupby(full.index.hour).apply(lambda x: x.isna().mean() * 100)
    return pd.DataFrame(rows)


def missingness_by_dow(pm25_dict):
    rows = {}
    for code, s in pm25_dict.items():
        full = reindex_full_range(s)
        rows[code] = full.groupby(full.index.dayofweek).apply(lambda x: x.isna().mean() * 100)
    return pd.DataFrame(rows)


def compute_gap_runs(series_full):
    """
    Identify every run of consecutive missing hours in a reindexed
    (full hourly range) series. Returns one row per gap: start, end,
    length_hours.
    """
    is_na = series_full.isna()
    if not is_na.any():
        return pd.DataFrame(columns=["start", "end", "length_hours"])
    grp = (is_na != is_na.shift()).cumsum()
    only_na = pd.DataFrame({"grp": grp[is_na]}, index=series_full.index[is_na])
    runs = only_na.groupby("grp").apply(
        lambda g: pd.Series({"start": g.index.min(), "end": g.index.max(),
                              "length_hours": len(g)})
    )
    return runs.reset_index(drop=True)


def analyze_gap_runs(pm25_dict, short_threshold=24):
    """
    For each station: how many gaps are short (<= short_threshold, the
    same 24h your pipeline/02_preprocess.py already interpolates over) vs. long
    (structural, left as NaN for the model to handle natively)? This is
    the split your own pipeline already treats differently, so it's the
    most directly useful missingness cut for your Data/Methodology text.
    """
    all_runs = {}
    summary_rows = []
    for code, s in pm25_dict.items():
        full = reindex_full_range(s)
        runs = compute_gap_runs(full)
        all_runs[code] = runs
        if len(runs) == 0:
            summary_rows.append({
                "station": code, "n_gaps": 0, "n_short_gaps": 0, "n_long_gaps": 0,
                "total_missing_hours": 0, "max_gap_hours": 0, "median_gap_hours": 0,
            })
            continue
        short = runs[runs["length_hours"] <= short_threshold]
        long_ = runs[runs["length_hours"] > short_threshold]
        summary_rows.append({
            "station": code,
            "n_gaps": len(runs),
            "n_short_gaps": len(short),
            "n_long_gaps": len(long_),
            "total_missing_hours": int(runs["length_hours"].sum()),
            "max_gap_hours": int(runs["length_hours"].max()),
            "median_gap_hours": float(runs["length_hours"].median()),
        })
    summary_df = pd.DataFrame(summary_rows)
    return all_runs, summary_df


def plot_gap_distribution(all_runs):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = [1, 2, 4, 8, 12, 24, 48, 96, 168, 336, 672, 100000]
    bin_labels = ["1h", "2-3h", "4-7h", "8-11h", "12-23h", "24-47h",
                  "48-95h", "96-167h", "168-335h", "336-671h", "672h+"]
    codes = list(all_runs.keys())
    x = np.arange(len(bin_labels))
    width = 0.8 / max(len(codes), 1)

    for i, code in enumerate(codes):
        runs = all_runs[code]
        if len(runs) == 0:
            counts = np.zeros(len(bin_labels))
        else:
            counts, _ = np.histogram(runs["length_hours"], bins=bins)
        ax.bar(x + i * width, counts, width, label=STATION_NAMES.get(code, code))

    ax.set_xticks(x + width * len(codes) / 2)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_xlabel("Gap length")
    ax.set_ylabel("Number of gaps (count)")
    ax.set_yscale("log")
    ax.set_title("Distribution of PM2.5 Missing-Data Gap Lengths (2019–2024)")
    ax.axvline(x=5.5, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(5.6, ax.get_ylim()[1] * 0.7, "← 24h interpolation limit",
            fontsize=8, color="red")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "missingness_gap_lengths.png", dpi=150)
    plt.close(fig)
    print("✓ Saved: results/missingness_gap_lengths.png")


def plot_hourly(hourly_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in hourly_df.columns:
        ax.plot(hourly_df.index, hourly_df[col], marker='o', markersize=3,
                label=STATION_NAMES.get(col, col), linewidth=1.2)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Missing PM2.5 observations (%)")
    ax.set_title("PM2.5 Missingness by Hour of Day (2019–2024)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "missingness_hourly.png", dpi=150)
    plt.close(fig)
    print("✓ Saved: results/missingness_hourly.png")


def plot_daily(daily_df):
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(7)
    n = len(daily_df.columns)
    width = 0.8 / n
    for i, col in enumerate(daily_df.columns):
        ax.bar(x + i * width, daily_df[col].values, width,
               label=STATION_NAMES.get(col, col))
    ax.set_xticks(x + width * n / 2)
    ax.set_xticklabels(dow_labels)
    ax.set_ylabel("Missing PM2.5 observations (%)")
    ax.set_title("PM2.5 Missingness by Day of Week (2019–2024)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "missingness_daily.png", dpi=150)
    plt.close(fig)
    print("✓ Saved: results/missingness_daily.png")


def run_all():
    print("\n" + "="*80)
    print(" MULTI-SCALE MISSINGNESS ANALYSIS")
    print("="*80)

    print("\nLoading data...")
    raw = load_raw()
    print(f"✓ Loaded {len(raw):,} rows")

    stations_wide = pivot_all_stations(raw)
    pm25_dict = {code: wide["PM2.5"] for code, wide in stations_wide.items()
                 if "PM2.5" in wide.columns}
    print(f"✓ PM2.5 series for {len(pm25_dict)} stations")

    hourly_df = missingness_by_hour(pm25_dict)
    daily_df = missingness_by_dow(pm25_dict)

    print("\n── Missingness by hour of day (%) ──")
    print(hourly_df.round(2).to_string())
    print("\n── Missingness by day of week (%) ──")
    print(daily_df.round(2).to_string())

    plot_hourly(hourly_df)
    plot_daily(daily_df)

    print("\n" + "="*80)
    print(" GAP-DURATION ANALYSIS (most likely to reveal something new)")
    print("="*80)
    print("""
This splits gaps by LENGTH rather than by calendar position. Your own
pipeline/02_preprocess.py already treats short and long gaps differently
(interpolate limit=24 vs. "intentional long gaps, handled by model
natively") — this shows you exactly how many of each you actually
have, and how bad the worst ones are.
    """)
    all_runs, gap_summary = analyze_gap_runs(pm25_dict, short_threshold=24)
    gap_summary["display_name"] = gap_summary["station"].map(STATION_NAMES)
    print(gap_summary[["display_name", "n_gaps", "n_short_gaps", "n_long_gaps",
                        "total_missing_hours", "max_gap_hours",
                        "median_gap_hours"]].to_string(index=False))

    plot_gap_distribution(all_runs)

    worst = gap_summary.loc[gap_summary["max_gap_hours"].idxmax()]
    total_long_gap_stations = (gap_summary["n_long_gaps"] > 0).sum()
    print(f"""
Longest single gap: {worst['max_gap_hours']:.0f}h at {worst['display_name']}
Stations with at least one gap longer than {24}h: {total_long_gap_stations} of {len(gap_summary)}

If the longest gap here roughly matches the ~264h PM10 gap at Landshuter
Allee you're already handling in pipeline/02_preprocess.py, that's a useful
cross-check that this analysis lines up with what you already know about
your own data.

Suggested thesis sentence (Data Integration / Missing Value Patterns):
"Gap-length analysis shows the majority of missing PM2.5 observations
occur as short, isolated dropouts ({round(100*gap_summary['n_short_gaps'].sum()/max(gap_summary['n_gaps'].sum(),1))}% of
all gaps are ≤24h), consistent with sensor or transmission failures
rather than extended outages. A small number of longer structural gaps
(up to {int(gap_summary['max_gap_hours'].max())}h at {worst['display_name']}) are treated separately in the
missing-value handling strategy described in Section 4.X."
    """)

    hourly_range = hourly_df.max() - hourly_df.min()
    daily_range = daily_df.max() - daily_df.min()
    flat = (hourly_range < 0.5).all() and (daily_range < 0.5).all()

    print("\n" + "="*80)
    print(" INTERPRETATION")
    print("="*80)
    if flat:
        print("""
Both hourly and day-of-week missingness are essentially FLAT across
stations — no clustering at specific hours or weekdays. This supports
your Figure 3.3 conclusion: gaps look like isolated sensor/transmission
dropouts, not a scheduled process (e.g. routine maintenance windows).

Suggested sentence: "Missingness shows no systematic hour-of-day or
day-of-week pattern (range < 0.5 percentage points across stations),
consistent with isolated sensor or transmission failures rather than
a scheduled maintenance process."
        """)
    else:
        print(f"""
Some structure IS present (hourly range up to {hourly_range.max():.2f} pp,
daily range up to {daily_range.max():.2f} pp). Check which specific
hour/day spikes in the printed tables above — an early-morning cluster
often indicates a daily calibration window.
        """)


if __name__ == "__main__":
    run_all()