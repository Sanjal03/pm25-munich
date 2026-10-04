"""
deseasonalized_correlation.py
================================================================================
Redo the spatial-correlation analysis with the shared seasonal cycle removed.

Raw daily-mean PM2.5 correlation against a Munich reference station is
dominated by the fact that every Central European city has high-PM2.5
winters and cleaner summers -- so even a station in a completely different
regional airshed (Stuttgart, 190km, different state) can show high raw
correlation just from sharing Germany's climate calendar, not from genuine
day-to-day regional atmospheric coupling with Munich.

Fix: subtract each station's own monthly climatology (its average PM2.5 for
that calendar month, across all years) before correlating. What's left is
each station's day-to-day deviation from its own normal seasonal pattern --
correlating THAT against Munich's deviations isolates genuine synoptic-scale
coupling (shared weather systems, transboundary transport) from "both places
have winter."

Reuses already-fetched data -- no new API calls.
"""

from pathlib import Path

import pandas as pd

REFERENCE_CSV = Path("munich_aq_final/Munich_AQ_2019-01-01_2024-12-31.csv")
REFERENCE_STATION_NAME = "München/Landshuter Allee"

# Existing 9-station stations (excluding reference itself) live in the main CSV
MAIN_CSV_STATIONS = {
    "München/Stachus": 1, "München/Lothstraße": 3, "München/Johanneskirchen": 12,
    "Andechs/Rothenfeld": 35, "Augsburg/Bourges-Platz": 70, "Augsburg/LfU": 70,
    "Burghausen/Marktler Straße": 93, "Oberaudorf/Inntal-Autobahn": 71,
}

# Standalone candidate/control CSVs, each already fetched separately
EXTRA_FILES = {
    "Neustadt a.d. Donau/Eining": ("munich_aq_final/_candidate_probe_neustadt/neustadt_full_2019_2024.csv", 81),
    "Trostberg/Schwimmbadstraße": ("munich_aq_final/_candidate_batch2/trostberg_full_2019_2024.csv", 73),
    "Mehring/Sportplatz": ("munich_aq_final/_candidate_batch2/mehring_full_2019_2024.csv", 90),
    "Kempten (Allgäu)/Westendstraße": ("munich_aq_final/_candidate_batch2/kempten_allgäu_full_2019_2024.csv", 105),
    "Oettingen/Goethestraße": ("munich_aq_final/_candidate_batch2/oettingen_full_2019_2024.csv", 116),
    "Neu-Ulm/Gabelsbergerstraße": ("munich_aq_final/_candidate_batch2/neu-ulm_full_2019_2024.csv", 120),
    "Nürnberg/Muggenhof (control)": ("munich_aq_final/_control_validation/nürnberg_muggenhof_full_2019_2024.csv", 165),
    "Stuttgart-Bad Cannstatt (control)": ("munich_aq_final/_control_validation/stuttgart_bad_cannstatt_full_2019_2024.csv", 190),
}


def daily_pm25(df: pd.DataFrame, station_name: str | None = None) -> pd.Series:
    if station_name is not None:
        df = df[df["Station Name"] == station_name]
    df = df[df["Pollutant"] == "PM2.5"].copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df.set_index("Datetime")["Value (µg/m³)"].resample("D").mean()


def deseasonalize(daily: pd.Series) -> pd.Series:
    """Subtract each station's own monthly climatology (mean PM2.5 for that
    calendar month, across all years) -- leaves day-to-day anomaly."""
    monthly_climatology = daily.groupby(daily.index.month).transform("mean")
    return daily - monthly_climatology


def main():
    main_raw = pd.read_csv(REFERENCE_CSV)
    ref_daily = daily_pm25(main_raw, REFERENCE_STATION_NAME)
    ref_anom = deseasonalize(ref_daily)

    rows = []

    for name, dist in MAIN_CSV_STATIONS.items():
        cand_daily = daily_pm25(main_raw, name)
        cand_anom = deseasonalize(cand_daily)

        raw_joined = pd.concat([ref_daily, cand_daily], axis=1, join="inner").dropna()
        anom_joined = pd.concat([ref_anom, cand_anom], axis=1, join="inner").dropna()
        raw_r = raw_joined.iloc[:, 0].corr(raw_joined.iloc[:, 1])
        anom_r = anom_joined.iloc[:, 0].corr(anom_joined.iloc[:, 1])
        rows.append({"station": name, "distance_km": dist,
                      "raw_r": round(raw_r, 3), "deseasonalized_r": round(anom_r, 3)})

    for name, (path, dist) in EXTRA_FILES.items():
        p = Path(path)
        if not p.exists():
            print(f"  ! missing file for {name}: {path}")
            continue
        cand_raw = pd.read_csv(p)
        cand_daily = daily_pm25(cand_raw)
        cand_anom = deseasonalize(cand_daily)

        raw_joined = pd.concat([ref_daily, cand_daily], axis=1, join="inner").dropna()
        anom_joined = pd.concat([ref_anom, cand_anom], axis=1, join="inner").dropna()
        raw_r = raw_joined.iloc[:, 0].corr(raw_joined.iloc[:, 1])
        anom_r = anom_joined.iloc[:, 0].corr(anom_joined.iloc[:, 1])
        rows.append({"station": name, "distance_km": dist,
                      "raw_r": round(raw_r, 3), "deseasonalized_r": round(anom_r, 3)})

    out = pd.DataFrame(rows).sort_values("deseasonalized_r", ascending=False)
    out["drop"] = (out["raw_r"] - out["deseasonalized_r"]).round(3)

    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/deseasonalized_spatial_correlation.csv", index=False)

    print(out.to_string(index=False))
    print(f"\nSaved to results/deseasonalized_spatial_correlation.csv")


if __name__ == "__main__":
    main()
