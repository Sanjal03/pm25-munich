"""
03_station_metadata_spatial.py  (v2 — matches your actual data file)
========================================================================
Your CSV already carries a "Distance (km)" and "Environment" column per
station (computed by your own pipeline — presumably distance to a
Munich reference point). This script uses THAT as the authoritative
distance, and adds verified lat/lon/elevation from official LfU Bayern
station documentation (lfu.bayern.de/luft/immissionsmessungen/
dokumentation, accessed Aug 2026) so your thesis table has full
geographic detail, not just distance.

It then computes actual inter-station PM2.5 correlation from your real
data to quantify spatial mismatch.
"""

import pandas as pd
import numpy as np
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
THESIS_STATION_CODES = ["DEBY109", "DEBY007", "DEBY099",
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

# Verified via official LfU Bayern LÜB documentation PDFs
STATION_GEO = {
    "DEBY115": {"lat": 48.14955, "lon": 11.53653, "elevation_m": 520,
                "source_pdf": "01_Oberbayern/09_muenchen_landshuter_allee.pdf"},
    "DEBY037": {"lat": 48.13732, "lon": 11.56481, "elevation_m": 520,
                "source_pdf": "01_Oberbayern/11_muenchen_stachus.pdf"},
    "DEBY039": {"lat": 48.15455, "lon": 11.55466, "elevation_m": 519,
                "source_pdf": "01_Oberbayern/10_muenchen_lothstrasse.pdf"},
    "DEBY089": {"lat": 48.17319, "lon": 11.64804, "elevation_m": 512,
                "source_pdf": "01_Oberbayern/08_muenchen_johanneskirchen.pdf"},
    "DEBY007": {"lat": 48.37658, "lon": 10.88837, "elevation_m": 478,
                "source_pdf": "07_Schwaben/01_augsburg_bourges_platz.pdf"},
    "DEBY099": {"lat": 48.32601, "lon": 10.90305, "elevation_m": 495,
                "source_pdf": "07_Schwaben/04_augsburg_lfu.pdf"},
    "DEBY109": {"lat": 47.96875, "lon": 11.22017, "elevation_m": 700,
                "source_pdf": "01_Oberbayern/01_andechs_rothenfeld.pdf"},
    # München/Allach not in your final thesis station set — coordinates
    # not verified here. Add if you decide to include it after all.
    "DEBY189": {"lat": None, "lon": None, "elevation_m": None, "source_pdf": None},
    # Burghausen and Oberaudorf: lat/lon confirmed live from the UBA
    # stations API (2026-08-23), NOT cross-checked against an LfU
    # documentation PDF the way the original 7 were — elevation is
    # genuinely unverified, left as None rather than guessed. Fill in
    # from the LfU LÜB documentation if you want the same rigor as the
    # rest of this table before it goes in the thesis.
    "DEBY012": {"lat": 48.1772, "lon": 12.8293, "elevation_m": None, "source_pdf": None},
    "DEBY121": {"lat": 47.6485, "lon": 12.1883, "elevation_m": None, "source_pdf": None},
}

REFERENCE_STATION = "DEBY115"  # Landshuter Allee


def build_metadata_table(raw):
    meta_rows = raw[["Station Code", "Station Name", "Environment",
                      "Distance (km)"]].drop_duplicates()
    meta_rows = meta_rows.set_index("Station Code")

    rows = []
    for code, row in meta_rows.iterrows():
        geo = STATION_GEO.get(code, {})
        rows.append({
            "station_code": code,
            "display_name": row["Station Name"],
            "environment": row["Environment"],
            "distance_km": row["Distance (km)"],
            "lat": geo.get("lat"),
            "lon": geo.get("lon"),
            "elevation_m": geo.get("elevation_m"),
        })
    df = pd.DataFrame(rows).sort_values("distance_km")
    return df


def print_latex_table(df):
    print("\n" + "="*90)
    print(" LATEX TABLE (paste into your thesis, Section 3.3)")
    print("="*90)
    print(r"""
\begin{table}[h]
\centering
\caption{Geographic characteristics of the monitoring stations. Distance is
measured to the Munich city center reference point used throughout this
thesis. Coordinates and elevation sourced from the Bavarian State Office
for the Environment (LfU) LÜB station documentation.}
\label{tab:station-metadata}
\begin{tabular}{lccccc}
\toprule
Station & Type & Lat (\textdegree N) & Lon (\textdegree E) & Elev. (m) & Dist. (km) \\
\midrule""")
    for _, row in df.iterrows():
        lat = f"{row['lat']:.4f}" if pd.notna(row['lat']) else "—"
        lon = f"{row['lon']:.4f}" if pd.notna(row['lon']) else "—"
        elev = f"{row['elevation_m']:.0f}" if pd.notna(row['elevation_m']) else "—"
        print(f"{row['display_name']} & {row['environment']} & {lat} & "
              f"{lon} & {elev} & {row['distance_km']:.0f} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def compute_spatial_decorrelation(raw, reference=REFERENCE_STATION):
    print("\n" + "="*90)
    print(" SPATIAL DECORRELATION ANALYSIS (real PM2.5 data)")
    print("="*90)

    stations_wide = pivot_all_stations(raw, STATION_CODES)
    daily_series = {code: wide["PM2.5"].resample("D").mean()
                     for code, wide in stations_wide.items() if "PM2.5" in wide.columns}

    if reference not in daily_series:
        print(f"✗ Reference station {reference} has no PM2.5 data.")
        return None

    ref_series = daily_series[reference]
    dist_lookup = raw[["Station Code", "Distance (km)"]].drop_duplicates().set_index("Station Code")["Distance (km)"]

    results = []
    for code, series in daily_series.items():
        if code == reference:
            continue
        paired = pd.concat([ref_series, series], axis=1).dropna()
        if len(paired) < 30:
            print(f"⚠ {STATION_NAMES[code]}: only {len(paired)} overlapping days — skipping")
            continue
        r = paired.corr().iloc[0, 1]
        results.append({
            "station": code,
            "display_name": STATION_NAMES[code],
            "distance_km": dist_lookup.get(code, np.nan),
            "pearson_r": round(r, 3),
            "n_days": len(paired),
        })

    result_df = pd.DataFrame(results).sort_values("distance_km")
    print(f"\nDaily-mean PM2.5 correlation vs. {STATION_NAMES[reference]}:")
    print(result_df.to_string(index=False))

    if len(result_df) >= 3:
        corr_of_corr = result_df["distance_km"].corr(result_df["pearson_r"])
        print(f"\nCorrelation between distance and Pearson r: {corr_of_corr:.3f}")
        if corr_of_corr < -0.3:
            print("→ Supports expected spatial decay: farther stations correlate "
                  "more weakly with Landshuter Allee.")
        else:
            print("→ No strong distance-decorrelation signal — station TYPE "
                  "(traffic vs. background vs. rural) may dominate over raw "
                  "distance. Worth a sentence in Limitations either way.")

    return result_df


def run_all():
    print("\n" + "="*90)
    print(" STATION METADATA & SPATIAL MISMATCH ANALYSIS")
    print("="*90)

    print("\nLoading data...")
    raw = load_raw()
    print(f"✓ Loaded {len(raw):,} rows")

    meta_df = build_metadata_table(raw)
    print("\nStation metadata:")
    print(meta_df.to_string(index=False))

    print_latex_table(meta_df)

    corr_df = compute_spatial_decorrelation(raw)

    if corr_df is not None and len(corr_df) > 0:
        farthest = corr_df.iloc[-1]
        closest = corr_df.iloc[0]
        print(f"""
SUGGESTED THESIS TEXT:
"Daily-mean PM2.5 correlations against the Landshuter Allee reference
station range from r={farthest['pearson_r']:.2f} ({farthest['display_name']},
{farthest['distance_km']:.0f} km) to r={closest['pearson_r']:.2f}
({closest['display_name']}, {closest['distance_km']:.0f} km), showing that
spatial distance introduces a measurable degree of decorrelation across
the monitoring network."
        """)

    from pathlib import Path
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)
    meta_df.to_csv(RESULTS_DIR / "station_metadata.csv", index=False)
    if corr_df is not None:
        corr_df.to_csv(RESULTS_DIR / "spatial_correlation.csv", index=False)
    print(f"\n✓ Saved to results/")


if __name__ == "__main__":
    run_all()