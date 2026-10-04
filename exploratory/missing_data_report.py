"""
missing_data_report.py
====================================================================
Quantifies exactly how much data is missing, per station and per
pollutant, in Munich_AQ_2019-01-01_2024-12-31.csv.

"Missing" here means: an (station, pollutant, hour) combination that
should exist in the expected hourly range but is absent from the raw
file entirely (your source data doesn't emit explicit NaN rows —
gaps are just missing rows).
"""

import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

CSV_PATH = Path("munich_aq_final/Munich_AQ_2019-01-01_2024-12-31.csv")
_CSV_FALLBACKS = [
    CSV_PATH,
    Path("Munich_AQ_2019-01-01_2024-12-31.csv"),
    Path("../munich_aq_final/Munich_AQ_2019-01-01_2024-12-31.csv"),
]

STATION_NAMES = {
    "DEBY109": "Andechs/Rothenfeld", "DEBY007": "Augsburg/Bourges-Platz",
    "DEBY099": "Augsburg/LfU", "DEBY189": "München/Allach",
    "DEBY089": "München/Johanneskirchen", "DEBY115": "München/Landshuter Allee",
    "DEBY039": "München/Lothstraße", "DEBY037": "München/Stachus",
    "DEBY012": "Burghausen/Marktler Straße", "DEBY121": "Oberaudorf/Inntal-Autobahn",
}

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_raw():
    for candidate in _CSV_FALLBACKS:
        if candidate.exists():
            print(f"  (reading {candidate.resolve()})")
            df = pd.read_csv(candidate)
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            return df
    raise FileNotFoundError(
        f"Could not find the CSV. Tried: {[str(c) for c in _CSV_FALLBACKS]}"
    )


def run_report():
    print("\n" + "=" * 90)
    print(" MISSING DATA REPORT")
    print("=" * 90)

    print("\nLoading data...")
    raw = load_raw()
    print(f"✓ Loaded {len(raw):,} rows")

    full_start = raw["Datetime"].min()
    full_end = raw["Datetime"].max()
    expected_range = pd.date_range(full_start, full_end, freq="h")
    n_expected = len(expected_range)

    print(f"  Date range: {full_start} → {full_end}")
    print(f"  Expected hourly timestamps per station-pollutant: {n_expected:,}")

    rows = []
    for station_code in sorted(raw["Station Code"].unique()):
        for pollutant in sorted(raw["Pollutant"].unique()):
            sub = raw[(raw["Station Code"] == station_code) &
                      (raw["Pollutant"] == pollutant)]
            n_present = sub["Datetime"].nunique()
            n_missing = n_expected - n_present
            pct_missing = 100 * n_missing / n_expected

            rows.append({
                "station_code": station_code,
                "display_name": STATION_NAMES.get(station_code, station_code),
                "pollutant": pollutant,
                "n_expected": n_expected,
                "n_present": n_present,
                "n_missing": n_missing,
                "pct_missing": round(pct_missing, 3),
            })

    report = pd.DataFrame(rows)

    print("\n" + "=" * 90)
    print(" MISSINGNESS BY STATION x POLLUTANT")
    print("=" * 90)
    print(report.sort_values(["station_code", "pollutant"])
          [["display_name", "pollutant", "n_present", "n_missing", "pct_missing"]]
          .to_string(index=False))

    print("\n" + "=" * 90)
    print(" SUMMARY BY STATION (averaged across all 4 pollutants)")
    print("=" * 90)
    by_station = report.groupby("display_name").agg(
        mean_pct_missing=("pct_missing", "mean"),
        max_pct_missing=("pct_missing", "max"),
        worst_pollutant=("pollutant", lambda s: report.loc[
            report.loc[s.index, "pct_missing"].idxmax(), "pollutant"]),
    ).round(3).sort_values("mean_pct_missing")
    print(by_station.to_string())

    print("\n" + "=" * 90)
    print(" SUMMARY BY POLLUTANT (averaged across all 8 stations)")
    print("=" * 90)
    by_pollutant = report.groupby("pollutant").agg(
        mean_pct_missing=("pct_missing", "mean"),
        max_pct_missing=("pct_missing", "max"),
    ).round(3).sort_values("mean_pct_missing")
    print(by_pollutant.to_string())

    overall_pct = report["n_missing"].sum() / report["n_expected"].sum() * 100
    print(f"\nOVERALL missingness across all stations x pollutants: {overall_pct:.3f}%")
    print(f"Total expected observations: {report['n_expected'].sum():,}")
    print(f"Total present observations:  {report['n_present'].sum():,}")
    print(f"Total missing observations:  {report['n_missing'].sum():,}")

    out_path = RESULTS_DIR / "missing_data_report.csv"
    report.to_csv(out_path, index=False)
    print(f"\n✓ Full report saved to {out_path}")

    worst = report.loc[report["pct_missing"].idxmax()]
    best = report.loc[report["pct_missing"].idxmin()]
    print(f"""
SUGGESTED THESIS TEXT (Data Integration / Missing Value Patterns):

"Missingness varies by station and pollutant, ranging from
{best['pct_missing']:.2f}% ({best['display_name']}, {best['pollutant']})
to {worst['pct_missing']:.2f}% ({worst['display_name']}, {worst['pollutant']}),
with an overall missingness of {overall_pct:.2f}% across all
station-pollutant combinations in the 2019-2024 study period."
    """)


if __name__ == "__main__":
    run_report()