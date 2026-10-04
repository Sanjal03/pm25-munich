"""
data_chapter_figures.py
================================================================================
Regenerates every figure of the Data chapter at its final printed size
(text width ~ 6.3 in) so that fonts stay readable (>= 7 pt) when inserted with
\\includegraphics[width=\\textwidth]. Each figure is written as vector PDF
(for LaTeX) and as PNG (preview) to thesis_figs/data_chapter/.

Outputs (file name = name used in 03_Data.tex):
    study_area_map, pm25_time_series_2019_2024, correlation_matrix,
    outlier_nye_real, outlier_sensor_error, missing_data_heatmap,
    missingness_patterns (hourly + day-of-week + gap length, one figure)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

CSV = Path("munich_aq_final/Munich_AQ_2019-01-01_2024-12-31.csv")
OUT = Path("thesis_figs/data_chapter")
OUT.mkdir(parents=True, exist_ok=True)

W = 6.3  # text width in inches

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "serif", "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

# Station metadata (same values as Table tab:final_stations)
STATIONS = {
    "München/Landshuter Allee": ("Landshuter Allee", 48.1495, 11.5365, "Traffic"),
    "München/Stachus":          ("Stachus",          48.1373, 11.5648, "Traffic"),
    "München/Lothstraße":       ("Lothstraße",       48.1546, 11.5547, "Urban background"),
    "München/Johanneskirchen":  ("Johanneskirchen",  48.1732, 11.6480, "Suburban"),
    "Andechs/Rothenfeld":       ("Andechs/Rothenfeld", 47.9688, 11.2202, "Rural background"),
    "Augsburg/Bourges-Platz":   ("Augsburg/Bourges-Platz", 48.3766, 10.8884, "Urban background"),
    "Augsburg/LfU":             ("Augsburg/LfU",     48.3260, 10.9031, "Suburban"),
}
ORDER = list(STATIONS)
SHORT = {k: v[0] for k, v in STATIONS.items()}
TYPE_COL = {"Traffic": "#D55E00", "Urban background": "#0072B2",
            "Suburban": "#009E73", "Rural background": "#8C6D1F"}
ST_COL = {"München/Landshuter Allee": "#D55E00", "München/Stachus": "#E69F00",
          "München/Lothstraße": "#0072B2", "München/Johanneskirchen": "#009E73",
          "Andechs/Rothenfeld": "#8C6D1F", "Augsburg/Bourges-Platz": "#CC79A7",
          "Augsburg/LfU": "#882255"}
REF = "München/Landshuter Allee"
IDX = pd.date_range("2019-01-01", "2024-12-31 23:00", freq="h")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("saved", name)


def load_pm25():
    d = pd.read_csv(CSV, parse_dates=["Datetime"])
    d = d[(d.Pollutant == "PM2.5") & d["Station Name"].isin(ORDER)]
    w = d.pivot_table(index="Datetime", columns="Station Name",
                      values="Value (µg/m³)", aggfunc="first")
    return w.reindex(IDX)[ORDER]


pm = load_pm25()


# 1. Study-area map ----------------------------------------------------------
def fig_map():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 3.3),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    lat0 = STATIONS[REF][1]
    lon0 = STATIONS[REF][2]
    kx = 111.2 * np.cos(np.deg2rad(lat0))   # km per degree longitude
    ky = 111.2
    for r in (25, 50):
        a1.add_patch(Ellipse((lon0, lat0), 2 * r / kx, 2 * r / ky, fill=False,
                             ls="--" if r == 25 else ":", lw=0.9, ec="grey"))
    for name, (lab, lat, lon, typ) in STATIONS.items():
        a1.scatter(lon, lat, s=46, color=TYPE_COL[typ], ec="white", lw=0.8, zorder=3)
    a1.set_xlim(10.75, 11.95)
    a1.set_ylim(47.88, 48.46)
    a1.set_aspect(1 / np.cos(np.deg2rad(lat0)))
    a1.set_xlabel("Longitude (°E)")
    a1.set_ylabel("Latitude (°N)")
    a1.annotate("Augsburg\n(2 stations)", (10.896, 48.35), xytext=(11.02, 48.40),
                fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.6))
    a1.annotate("Andechs/\nRothenfeld", (11.2202, 47.9688), xytext=(10.85, 47.93),
                fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.6))
    a1.annotate("Munich\nstations", (11.56, 48.16), xytext=(11.62, 48.30),
                fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.6))
    a1.add_patch(plt.Rectangle((11.43, 48.12), 0.25, 0.07, fill=False,
                               ec="black", lw=0.8))
    a1.set_title("(a) Study region", loc="left")

    offs = {"München/Landshuter Allee": (-7, 4, "right"),
            "München/Stachus": (-7, -12, "right"),
            "München/Lothstraße": (6, 6, "left"),
            "München/Johanneskirchen": (-6, 7, "right")}
    for name, (dx, dy, ha) in offs.items():
        lab, lat, lon, typ = STATIONS[name]
        a2.scatter(lon, lat, s=60, color=TYPE_COL[typ], ec="white", lw=0.8, zorder=3)
        a2.annotate(lab.replace("Landshuter Allee", "Landshuter\nAllee"), (lon, lat),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8, ha=ha)
    a2.scatter(11.5755, 48.1374, marker="*", s=90, color="black", zorder=4)
    a2.annotate("City centre", (11.5755, 48.1374), xytext=(7, -10),
                textcoords="offset points", fontsize=7.5, ha="left")
    a2.set_xlim(11.43, 11.68)
    a2.set_ylim(48.12, 48.19)
    a2.set_aspect(1 / np.cos(np.deg2rad(lat0)))
    a2.set_xlabel("Longitude (°E)")
    a2.set_title("(b) Munich stations", loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=t, ms=6)
               for t, c in TYPE_COL.items()]
    handles.append(plt.Line2D([], [], marker="*", ls="", color="black", ms=8,
                              label="City centre"))
    handles.append(plt.Line2D([], [], ls="--", color="grey", label="25 km"))
    handles.append(plt.Line2D([], [], ls=":", color="grey", label="50 km from Landshuter Allee"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "study_area_map")


# 2. PM2.5 time series --------------------------------------------------------
def fig_timeseries():
    daily = pm.resample("D").mean()
    mean, lo, hi = daily.mean(axis=1), daily.min(axis=1), daily.max(axis=1)
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.fill_between(daily.index, lo, hi, color="#9ecae1", alpha=0.6, lw=0,
                    label="Range across the 7 stations")
    ax.plot(daily.index, mean, color="#08306b", lw=0.7, label="Network mean")
    ax.axvspan(pd.Timestamp("2020-03-22"), pd.Timestamp("2020-05-04"),
               color="grey", alpha=0.25, lw=0, label="COVID-19 lockdown window")
    pk = mean.idxmax()
    ax.annotate("New Year's Eve\nfireworks", (pk, hi.loc[pk]),
                xytext=(pk - pd.Timedelta(days=330), hi.loc[pk] * 0.80),
                fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.set_ylabel("PM$_{2.5}$, daily mean (µg/m³)")
    ax.set_xlim(IDX[0], IDX[-1])
    ax.set_ylim(0, None)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save(fig, "pm25_time_series_2019_2024")


# 3. Correlation matrix ---------------------------------------------------------
def fig_corr():
    corr = pd.read_csv("results/station_correlation_matrix.csv", index_col=0)
    corr = corr.loc[ORDER, ORDER]
    lab = [SHORT[n] for n in corr.index]
    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    im = ax.imshow(corr.values, cmap="Blues", vmin=0.65, vmax=1.0)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(lab, rotation=40, ha="right")
    ax.set_yticklabels(lab)
    ax.grid(False)
    for i in range(7):
        for j in range(7):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.88 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson $r$ (daily-mean PM$_{2.5}$)")
    fig.tight_layout()
    save(fig, "correlation_matrix")


# 4./5. Outlier profiles -----------------------------------------------------
def fig_event(station, when, name, hl_label):
    t = pd.Timestamp(when)
    win = pm.loc[t - pd.Timedelta(hours=48): t + pd.Timedelta(hours=48)]
    fig, ax = plt.subplots(figsize=(W, 2.9))
    for s in ORDER:
        hl = s == station
        ax.plot(win.index, win[s], color=ST_COL[s], lw=1.8 if hl else 0.9,
                alpha=1 if hl else 0.8, label=SHORT[s], zorder=3 if hl else 2)
    ax.axvline(t, color="black", ls=":", lw=0.8)
    ax.set_xlabel("Date and time (UTC)")
    ax.set_ylabel("PM$_{2.5}$ (µg/m³)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.set_ylim(0, None)
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32),
              frameon=False)
    ax.annotate(hl_label, (t, win[station].max()), xytext=(-8, -2),
                textcoords="offset points", fontsize=8, va="top", ha="right")
    fig.tight_layout()
    save(fig, name)


def fig_outliers():
    fig_event("München/Johanneskirchen", "2023-01-01 01:00", "outlier_nye_real",
              "Johanneskirchen\n539 µg/m³")
    s = pm["Augsburg/Bourges-Platz"]
    t = s.loc["2023-12-31"].idxmax()
    fig_event("Augsburg/Bourges-Platz", t, "outlier_sensor_error",
              f"Bourges-Platz\n{s.loc[t]:.0f} µg/m³")


# 6. Monthly missing heatmap ---------------------------------------------------
def fig_heatmap():
    miss = pm.isna().astype(float)
    m = miss.groupby([miss.index.year, miss.index.month]).mean().mul(100)
    mat = m.T.values
    fig, ax = plt.subplots(figsize=(W, 2.9))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=10)
    ax.set_yticks(range(7))
    ax.set_yticklabels([SHORT[s] for s in ORDER])
    ax.set_xticks(np.arange(0, 72, 12) + 5.5)
    ax.set_xticklabels(range(2019, 2025))
    for x in range(12, 72, 12):
        ax.axvline(x - 0.5, color="white", lw=1.2)
    ax.grid(False)
    ax.tick_params(axis="x", length=0)
    cb = fig.colorbar(im, ax=ax, pad=0.015, extend="max",
                      label="Missing PM$_{2.5}$ (% of hours in month)")
    fig.tight_layout()
    save(fig, "missing_data_heatmap")


# 7. Missingness patterns (hour of day, day of week, gap length) ---------------
def gap_runs(s):
    m = s.isna()
    run = (m != m.shift()).cumsum()
    return m.groupby(run).sum()[m.groupby(run).first()].astype(int)


def fig_patterns():
    miss = pm.isna().astype(float)
    hourly = miss.groupby(miss.index.hour).mean().mul(100)
    daily = miss.groupby(miss.index.dayofweek).mean().mul(100)
    bins = [1, 2, 4, 8, 12, 24, 48, 96, 168, 336, 672, 10 ** 6]
    blab = ["1", "2–3", "4–7", "8–11", "12–23", "24–47", "48–95", "96–167",
            "168–335", "336–671", "672+"]
    fig = plt.figure(figsize=(W, 6.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.28)
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for s in ORDER:
        a1.plot(hourly.index, hourly[s], color=ST_COL[s], lw=1.1, label=SHORT[s])
    a1.set_xlabel("Hour of day (UTC)"); a1.set_ylabel("Missing PM$_{2.5}$ (%)")
    a1.set_xticks(range(0, 24, 4)); a1.set_title("(a) By hour of day", loc="left")
    x = np.arange(7); w = 0.8 / 7
    for i, s in enumerate(ORDER):
        a2.bar(x + i * w, daily[s].values, w, color=ST_COL[s])
    a2.set_xticks(x + 0.4 - w / 2)
    a2.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    a2.set_ylabel("Missing PM$_{2.5}$ (%)"); a2.set_title("(b) By day of week", loc="left")
    a2.grid(axis="x", visible=False)
    x = np.arange(len(blab)); w = 0.8 / 7
    for i, s in enumerate(ORDER):
        r = gap_runs(pm[s])
        c, _ = np.histogram(r.values, bins=bins) if len(r) else (np.zeros(len(blab)), 0)
        a3.bar(x + i * w, c, w, color=ST_COL[s], label=SHORT[s])
    a3.set_yscale("log")
    a3.set_xticks(x + 0.4 - w / 2); a3.set_xticklabels(blab, rotation=30, ha="right")
    a3.set_xlabel("Length of consecutive gap (hours)")
    a3.set_ylabel("Number of gaps")
    a3.axvline(4.9, color="red", ls="--", lw=0.8)
    a3.text(5.0, a3.get_ylim()[1] * 0.6, "24 h interpolation limit",
            color="red", fontsize=7.5, va="top")
    a3.set_title("(c) Gap-length distribution", loc="left")
    a3.grid(axis="x", visible=False)
    h, l = a3.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.07))
    save(fig, "missingness_patterns")


if __name__ == "__main__":
    fig_map(); fig_timeseries(); fig_corr(); fig_outliers(); fig_heatmap(); fig_patterns()
