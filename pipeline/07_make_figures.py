"""
07_make_figures.py — Study area map + RQ1/RQ2 result figures
====================================================================
One-off figure generation script, separate from the phase1-3 pipeline.
Reads results/*.csv and results/*.json (7-station
results) and station coordinates, produces PNGs in thesis_figs/.
"""
import json
import warnings
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

OUT = Path("thesis_figs")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fbfaf7",
    "font.family": "serif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.edgecolor": "#2b2b2b", "axes.linewidth": 0.8,
})

MUNICH = {"lat": 48.1374, "lon": 11.5755}

STATIONS = {
    "München/Stachus":           {"lat": 48.1373, "lon": 11.5649, "type": "traffic-urban"},
    "München/Landshuter Allee":  {"lat": 48.1496, "lon": 11.5365, "type": "traffic"},
    "München/Lothstraße":        {"lat": 48.1545, "lon": 11.5547, "type": "urban-background"},
    "München/Johanneskirchen":   {"lat": 48.1732, "lon": 11.6480, "type": "suburban"},
    "Andechs/Rothenfeld":        {"lat": 47.9688, "lon": 11.2202, "type": "rural-background"},
    "Augsburg/Bourges-Platz":    {"lat": 48.3766, "lon": 10.8884, "type": "urban-background"},
    "Augsburg/LfU":              {"lat": 48.3260, "lon": 10.9031, "type": "suburban"},
}
EXCLUDED = {"München/Allach": {"lat": 48.1817, "lon": 11.4645, "reason": "no PM2.5 sensor"}}

TYPE_LABELS = {
    "traffic-urban": "Urban traffic", "traffic": "Traffic",
    "urban-background": "Urban background", "suburban": "Suburban",
    "suburban-background": "Suburban background", "rural-background": "Rural background",
    "rural-traffic": "Rural traffic",
}
TYPE_COLORS = {
    "traffic-urban": "#a63328", "traffic": "#c9782e",
    "urban-background": "#2b6a99", "suburban": "#1f8a70",
    "suburban-background": "#1f8a70", "rural-background": "#5c8a3a",
    "rural-traffic": "#6f4e91",
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


CORE = {"München/Stachus", "München/Landshuter Allee", "München/Lothstraße",
        "München/Johanneskirchen"}

# Custom per-station label offsets (points) — the 4 Munich-core stations and
# the 2 Augsburg stations sit close enough together that a uniform offset
# makes labels overlap.
LABEL_OFFSET = {
    "München/Stachus":            (10, -22),
    "München/Landshuter Allee":   (-95, 4),
    "München/Lothstraße":         (10, 10),
    "München/Johanneskirchen":    (10, 6),
    "Augsburg/Bourges-Platz":     (10, 8),
    "Augsburg/LfU":               (10, -16),
    "Andechs/Rothenfeld":         (10, 6),
}


def _draw_stations(ax, lat0, label_fontsize=9, show_rings=False,
                    km_per_deg_lat=111.0, km_per_deg_lon=None, dist_suffix=True,
                    only=None):
    km_per_deg_lon = km_per_deg_lon or 111.0 * cos(radians(lat0))

    if show_rings:
        for r in [25, 50, 75, 100]:
            circle = plt.Circle((MUNICH["lon"], MUNICH["lat"]), r / km_per_deg_lon,
                                 fill=False, color="#b8b3a8", linestyle=(0, (5, 3)),
                                 linewidth=1, zorder=1)
            ax.add_patch(circle)
            # label on the right edge of each ring, clear of the title and stations
            ax.annotate(f"{r} km", (MUNICH["lon"] + r / km_per_deg_lon, MUNICH["lat"]),
                        xytext=(3, 3), textcoords="offset points",
                        color="#8a8577", fontsize=8, style="italic")

    ax.scatter([MUNICH["lon"]], [MUNICH["lat"]], marker="*", s=420,
               color="#1a1a1a", edgecolor="white", linewidth=0.8, zorder=5)
    munich_offset = (10, 14) if show_rings else (-58, -4)
    ax.annotate("Munich", (MUNICH["lon"], MUNICH["lat"]), xytext=munich_offset,
                textcoords="offset points", fontsize=10.5, fontweight="bold", color="#1a1a1a")

    for name, s in EXCLUDED.items():
        if only and name not in only:
            continue
        ax.scatter([s["lon"]], [s["lat"]], marker="x", s=80, color="#aaaaaa",
                   linewidth=1.6, zorder=3)
        ax.annotate(f"{name.split('/')[-1]} (excluded, {s['reason']})",
                    (s["lon"], s["lat"]), xytext=(6, -10), textcoords="offset points",
                    fontsize=7.5, color="#999999", style="italic")

    for name, s in STATIONS.items():
        if only and name not in only:
            continue
        color = TYPE_COLORS.get(s["type"], "#333333")
        ax.scatter([s["lon"]], [s["lat"]], marker="o", s=190, color=color,
                   edgecolor="white", linewidth=1.3, zorder=4)
        label = name.replace("München/", "").replace("Augsburg/", "Aug./")
        if dist_suffix:
            dist = haversine_km(lat0, MUNICH["lon"], s["lat"], s["lon"])
            label = f"{label}\n{dist:.0f} km"
        offset = LABEL_OFFSET.get(name, (8, 8))
        ax.annotate(label, (s["lon"], s["lat"]), xytext=offset,
                    textcoords="offset points", fontsize=label_fontsize, linespacing=1.35,
                    color="#222222")


def _add_scale_bar(ax, km_per_deg_lon, x0_frac=0.045, y_frac=0.045, length_km=25):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0 = xlim[0] + x0_frac * (xlim[1] - xlim[0])
    y0 = ylim[0] + y_frac * (ylim[1] - ylim[0])
    length_deg = length_km / km_per_deg_lon
    ax.plot([x0, x0 + length_deg], [y0, y0], color="#1a1a1a", linewidth=2.2, solid_capstyle="butt")
    for xt in (x0, x0 + length_deg):
        ax.plot([xt, xt], [y0 - 0.008, y0 + 0.008], color="#1a1a1a", linewidth=2.2)
    ax.annotate(f"{length_km} km", (x0 + length_deg / 2, y0), xytext=(0, 6),
               textcoords="offset points", ha="center", fontsize=8.5)


def _add_north_arrow(ax, x_frac=0.045, y_frac=0.90):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = xlim[0] + x_frac * (xlim[1] - xlim[0])
    y0 = ylim[0] + (y_frac - 0.05) * (ylim[1] - ylim[0])
    y1 = ylim[0] + (y_frac + 0.05) * (ylim[1] - ylim[0])
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
               arrowprops=dict(arrowstyle="-|>", color="#1a1a1a", linewidth=1.8))
    ax.annotate("N", (x, y1), xytext=(0, 3), textcoords="offset points",
               ha="center", fontsize=10, fontweight="bold")


def make_map():
    fig, ax = plt.subplots(figsize=(11.5, 10))
    lat0 = MUNICH["lat"]
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * cos(radians(lat0))
    ax.set_aspect(km_per_deg_lat / km_per_deg_lon)

    ax.set_xlim(10.55, 12.2)
    ax.set_ylim(47.7, 48.55)

    _draw_stations(ax, lat0, show_rings=True, km_per_deg_lat=km_per_deg_lat,
                   km_per_deg_lon=km_per_deg_lon)
    _add_scale_bar(ax, km_per_deg_lon, x0_frac=0.80, y_frac=0.045)
    _add_north_arrow(ax)

    used_types = sorted({s["type"] for s in STATIONS.values()},
                        key=lambda t: list(TYPE_LABELS).index(t))
    type_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=TYPE_COLORS[t],
                            markeredgecolor="white", markersize=11, label=TYPE_LABELS[t])
                     for t in used_types]
    munich_handle = Line2D([0], [0], marker="*", color="w", markerfacecolor="#1a1a1a",
                           markeredgecolor="white", markersize=15, label="Munich (city centre)")
    excl_handle = Line2D([0], [0], marker="x", color="#aaaaaa", markersize=8,
                         linewidth=0, markeredgewidth=1.6, label="Excluded station")
    legend = ax.legend(handles=[munich_handle] + type_handles + [excl_handle],
                       loc="lower left", fontsize=9, framealpha=0.95,
                       edgecolor="#2b2b2b", title="Station type", title_fontsize=9.5)
    legend.get_frame().set_linewidth(0.9)

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Study Area\nMunich Region PM2.5 Monitoring Network",
                pad=14)

    # Zoomed inset for the tight Munich urban-core cluster
    core_lon_c, core_lat_c = 11.575, 48.163
    half_w, half_h = 0.075, 0.055
    rect_x0, rect_x1 = core_lon_c - half_w, core_lon_c + half_w
    rect_y0, rect_y1 = core_lat_c - half_h, core_lat_c + half_h
    ax.add_patch(plt.Rectangle((rect_x0, rect_y0), rect_x1 - rect_x0, rect_y1 - rect_y0,
                                fill=False, edgecolor="#2b2b2b", linewidth=1.3, zorder=6))

    axins = ax.inset_axes([0.40, 0.045, 0.30, 0.32])
    axins.set_aspect(km_per_deg_lat / km_per_deg_lon)
    _draw_stations(axins, lat0, label_fontsize=8.5, show_rings=False, only=CORE | {"München/Allach"})
    axins.set_xlim(rect_x0, rect_x1)
    axins.set_ylim(rect_y0, rect_y1)
    axins.set_xticks([]); axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("#2b2b2b")
        spine.set_linewidth(1.1)
    axins.set_title("Munich urban core (detail)", fontsize=9.5, fontweight="bold", pad=6)

    fig.tight_layout()
    fig.savefig(OUT / "fig13_study_area_map.png", dpi=220)
    fig.savefig(OUT / "fig13_study_area_map.pdf")
    plt.close(fig)
    print("Saved fig13_study_area_map.png")


RQ1_MODELS = ["catboost", "xgboost", "lightgbm", "dlinear", "patchtst"]
RQ1_ORDER = ["PM2.5_lags", "Co-pollutants", "Meteorology", "Temporal", "PM2.5_rolling"]
MODEL_COLORS = {"catboost": "#2b6a99", "xgboost": "#c9782e", "lightgbm": "#a68a4c",
                 "dlinear": "#5b6770", "patchtst": "#7d5a7e"}
MODEL_LABELS = {"catboost": "CatBoost", "xgboost": "XGBoost", "lightgbm": "LightGBM",
                 "dlinear": "DLinear", "patchtst": "PatchTST"}


def load_rq1_agg():
    rows = []
    for model in RQ1_MODELS:
        files = sorted(Path("results").glob(f"rq1_*_{model}.csv"))
        dfs = [pd.read_csv(f) for f in files]
        combined = pd.concat(dfs)
        combined = combined[combined["Configuration"] != "All features"]
        agg = combined.groupby("Configuration")["delta_MAE_pct"].mean()
        for cfg, val in agg.items():
            group = cfg.replace("Without ", "")
            rows.append({"model": model, "group": group, "delta_MAE_pct": val})
    return pd.DataFrame(rows)


def make_rq1_grouped_bar():
    df = load_rq1_agg()
    fig, ax = plt.subplots(figsize=(6.3, 3.7))
    n_groups, n_models = len(RQ1_ORDER), len(RQ1_MODELS)
    width = 0.8 / n_models
    x = np.arange(n_groups)

    for i, model in enumerate(RQ1_MODELS):
        vals = [df[(df.model == model) & (df.group == g)]["delta_MAE_pct"]
                .pipe(lambda s: s.iloc[0] if len(s) else np.nan) for g in RQ1_ORDER]
        bars = ax.bar(x + i * width - 0.8/2 + width/2, vals, width,
                      label=MODEL_LABELS[model], color=MODEL_COLORS[model])
        for b, v in zip(bars, vals):
            if np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, 1, "n/a", ha="center",
                       va="bottom", fontsize=6, rotation=90, color="#888888")
            else:
                ax.text(b.get_x() + b.get_width()/2, v + (1.5 if v >= 0 else -3.5),
                       f"{v:.1f}", ha="center", fontsize=6, rotation=90)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", " ") for g in RQ1_ORDER])
    ax.set_ylabel("Mean ΔMAE (%) when group removed")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(-8, max(75, df["delta_MAE_pct"].max() + 8))
    ax.set_xlim(-0.55, n_groups - 0.35)
    fig.tight_layout()
    fig.savefig(OUT / "fig14_rq1_ablation_grouped_bar.png", dpi=200)
    fig.savefig(OUT / "fig14_rq1_ablation_grouped_bar.pdf")
    plt.close(fig)
    print("Saved fig14_rq1_ablation_grouped_bar.png")


def make_rq1_heatmap():
    df = load_rq1_agg()
    pivot = df.pivot(index="model", columns="group", values="delta_MAE_pct")
    pivot = pivot.reindex(index=RQ1_MODELS, columns=RQ1_ORDER)
    pivot.index = [MODEL_LABELS[m] for m in pivot.index]

    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    masked = np.ma.masked_invalid(pivot.values)
    im = ax.imshow(masked, cmap="RdYlGn_r", aspect="auto",
                   vmin=-5, vmax=70)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_", " ") for c in pivot.columns], rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            txt = "n/a" if np.isnan(v) else f"{v:.1f}%"
            color = "white" if (not np.isnan(v) and abs(v) > 55) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean ΔMAE (%)")
    fig.tight_layout()
    fig.savefig(OUT / "fig15_rq1_ablation_heatmap.png", dpi=200)
    fig.savefig(OUT / "fig15_rq1_ablation_heatmap.pdf")
    plt.close(fig)
    print("Saved fig15_rq1_ablation_heatmap.png")


RQ2_MODELS = ["catboost", "xgboost", "lightgbm", "dlinear", "patchtst"]
STATION_ORDER = ["stachus", "landshuter_allee", "lothstrasse", "johanneskirchen",
                  "augsburg_bourges", "augsburg_lfu", "andechs"]
STATION_LABELS = {"stachus": "Stachus", "landshuter_allee": "Landshuter Allee",
                    "lothstrasse": "Lothstraße", "johanneskirchen": "Johanneskirchen",
                    "augsburg_bourges": "Augsburg/Bourges", "augsburg_lfu": "Augsburg/LfU",
                    "andechs": "Andechs"}


def load_rq2():
    rows = [json.loads(f.read_text()) for f in Path("results").glob("rq2_*.json")]
    return pd.DataFrame(rows)


def make_rq2_grouped_bar():
    df = load_rq2()
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    n_st, n_mod = len(STATION_ORDER), len(RQ2_MODELS)
    width = 0.8 / n_mod
    x = np.arange(n_st)

    for i, model in enumerate(RQ2_MODELS):
        vals = [df[(df.model == model) & (df.station == s)]["gap_pct"]
                .pipe(lambda v: v.iloc[0] if len(v) else np.nan) for s in STATION_ORDER]
        ax.bar(x + i * width - 0.8/2 + width/2, vals, width,
              label=MODEL_LABELS[model], color=MODEL_COLORS[model])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([STATION_LABELS[s] for s in STATION_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Raw lockdown gap (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7, ncol=5,
              columnspacing=1.0, handlelength=1.2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig16_rq2_gap_by_station_model.png", dpi=200)
    fig.savefig(OUT / "fig16_rq2_gap_by_station_model.pdf")
    plt.close(fig)
    print("Saved fig16_rq2_gap_by_station_model.png")


def make_rq2_heatmap():
    df = load_rq2()
    pivot = df.pivot(index="station", columns="model", values="gap_pct")
    pivot = pivot.reindex(index=STATION_ORDER, columns=RQ2_MODELS)
    pivot.index = [STATION_LABELS[s] for s in pivot.index]
    pivot.columns = [MODEL_LABELS[m] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    vmax = np.abs(pivot.values).max()
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:+.1f}%", ha="center", va="center", fontsize=7.5,
                   color="white" if abs(v) > vmax*0.6 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Lockdown gap (%)  [red: PM$_{2.5}$ below counterfactual]")
    fig.tight_layout()
    fig.savefig(OUT / "fig17_rq2_gap_heatmap.png", dpi=200)
    fig.savefig(OUT / "fig17_rq2_gap_heatmap.pdf")
    plt.close(fig)
    print("Saved fig17_rq2_gap_heatmap.png")


def make_rq2_summary():
    df = load_rq2()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))

    ax = axes[0]
    counts = df.assign(sign=np.where(df["gap"] > 0, "Positive\n(PM$_{2.5}$ below\ncounterfactual)",
                                      "Negative/zero")).groupby("sign").size()
    colors = ["#2980b9", "#c0392b"]
    ax.bar(counts.index, counts.values, color=colors[:len(counts)])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, f"{v}/{counts.sum()}\n({v/counts.sum()*100:.0f}%)",
               ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("Number of station × model runs")
    ax.set_title("(a) Sign of the raw gap (35 runs)")
    ax.set_ylim(0, counts.max() + 8)

    ax = axes[1]
    mean_gap = df.groupby("model")["gap"].mean().reindex(RQ2_MODELS)
    colors2 = [MODEL_COLORS[m] for m in mean_gap.index]
    bars = ax.bar([MODEL_LABELS[m] for m in mean_gap.index], mean_gap.values, color=colors2)
    ax.axhline(0, color="black", linewidth=0.8)
    for b, v in zip(bars, mean_gap.values):
        ax.text(b.get_x()+b.get_width()/2, v + (0.02 if v>=0 else -0.06),
               f"{v:+.2f}", ha="center", fontsize=7.5)
    ax.set_ylabel("Mean gap (µg/m³) across 7 stations")
    ax.set_ylim(0, mean_gap.max() * 1.15)
    ax.tick_params(axis="x", labelrotation=20)
    ax.set_title("(b) Mean raw gap by model")
    fig.tight_layout()
    fig.savefig(OUT / "fig18_rq2_summary.png", dpi=200)
    fig.savefig(OUT / "fig18_rq2_summary.pdf")
    plt.close(fig)
    print("Saved fig18_rq2_summary.png")


def _fmt(x, nd):
    """Signed number, with a plain 0.00 instead of -0.00 for tiny values."""
    r = round(x, nd)
    return f"{0.0:.{nd}f}" if r == 0 else f"{r:+.{nd}f}"


def make_rq2_timeseries(station, model_type="catboost"):
    from model_factory import (
        GATE_START, GATE_END, LOCKDOWN_START, LOCKDOWN_END,
        load_station_data, tabular_feature_cols, train_and_eval,
        recursive_counterfactual_forecast,
    )
    EXCLUDE_RANGE = (GATE_START, LOCKDOWN_END)

    df = load_station_data(station)
    feature_cols = tabular_feature_cols(df)
    gate = train_and_eval(model_type, df, feature_cols=feature_cols,
                          date_range=(GATE_START, GATE_END), exclude_range=EXCLUDE_RANGE,
                          return_model=True)
    model = gate["_model"]

    y_pred = recursive_counterfactual_forecast(model, df, feature_cols,
                                                LOCKDOWN_START, LOCKDOWN_END, model_type=model_type)
    y_true = df.loc[LOCKDOWN_START:LOCKDOWN_END, "target"].reindex(y_pred.index)

    # Pre-lockdown context: real observed values for 2 weeks before, for visual continuity
    context_start = pd.Timestamp(LOCKDOWN_START) - pd.Timedelta(days=14)
    context = df.loc[context_start:pd.Timestamp(LOCKDOWN_START) - pd.Timedelta(hours=1), "target"]

    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    ax.plot(context.index, context.values, color="#555555", linewidth=0.8,
           label="Observed (pre-lockdown context)")
    ax.plot(y_true.index, y_true.values, color="#a63328", linewidth=0.9, label="Observed (actual)")
    ax.plot(y_pred.index, y_pred.values, color="#2b6a99", linewidth=0.9, linestyle="--",
           label="Counterfactual (no-lockdown, recursive rollout)")
    ax.fill_between(y_true.index, y_true.values, y_pred.values,
                    where=(y_pred.values >= y_true.values), color="#2b6a99", alpha=0.15)
    ax.fill_between(y_true.index, y_true.values, y_pred.values,
                    where=(y_pred.values < y_true.values), color="#a63328", alpha=0.15)
    ax.axvline(pd.Timestamp(LOCKDOWN_START), color="black", linestyle=":", linewidth=1)
    ax.annotate("Lockdown starts\n(22 Mar 2020)", (pd.Timestamp(LOCKDOWN_START) - pd.Timedelta(hours=8), ax.get_ylim()[1]*0.90),
               fontsize=7, ha="right")

    gap = float(y_pred.mean() - y_true.mean())
    gap_pct = gap / float(y_true.mean()) * 100
    ax.set_title(f"{STATION_LABELS.get(station, station)}, {MODEL_LABELS[model_type]}: "
                f"mean gap {_fmt(gap, 2)} µg/m³ ({_fmt(gap_pct, 1)}%)")
    ax.set_ylabel("PM$_{2.5}$ (µg/m³)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=6.5, ncol=3, frameon=False)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.tight_layout()
    fname = f"fig{19 if station=='augsburg_lfu' else 20}_rq2_timeseries_{station}.png"
    fig.savefig(OUT / fname, dpi=200)
    fig.savefig(OUT / fname.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved {fname}")


SIMPLE_LABEL_OFFSET = {
    "München/Stachus":            (9, -13),
    "München/Landshuter Allee":   (-95, 2),
    "München/Lothstraße":         (9, 9),
    "München/Johanneskirchen":    (9, 5),
    "Andechs/Rothenfeld":         (9, 5),
    "Augsburg/Bourges-Platz":     (9, 7),
    "Augsburg/LfU":               (9, -15),
}


def make_map_simple():
    """Minimal version: plain dots for the 7 stations, one star for Munich.
    No rings, scale bar, north arrow, inset, or type colour-coding. The 4
    Munich-core stations sit within ~7km of each other and would collide at
    this scale even at large figure size, so they get one grouped label
    instead of 4 overlapping ones."""
    fig, ax = plt.subplots(figsize=(10.5, 9))
    lat0 = MUNICH["lat"]
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * cos(radians(lat0))
    ax.set_aspect(km_per_deg_lat / km_per_deg_lon)

    STATION_COLOR = "#2b6a99"

    for name, s in STATIONS.items():
        ax.scatter([s["lon"]], [s["lat"]], marker="o", s=170, color=STATION_COLOR,
                   edgecolor="white", linewidth=1.2, zorder=4)
        if name in CORE:
            continue
        label = name.replace("München/", "").replace("Augsburg/", "Aug./")
        offset = SIMPLE_LABEL_OFFSET.get(name, (9, 6))
        ax.annotate(label, (s["lon"], s["lat"]), xytext=offset, textcoords="offset points",
                    fontsize=10, color="#1a1a1a")

    ax.scatter([MUNICH["lon"]], [MUNICH["lat"]], marker="*", s=520,
               color="#a63328", edgecolor="white", linewidth=1, zorder=5)

    # One grouped label + leader line for the 4 overlapping Munich-core stations
    core_lons = [STATIONS[n]["lon"] for n in CORE]
    core_lats = [STATIONS[n]["lat"] for n in CORE]
    core_c = (sum(core_lons) / len(core_lons), sum(core_lats) / len(core_lats))
    label_pt = (core_c[0] + 0.14, core_c[1] - 0.10)
    ax.annotate("Munich +\nurban core\n(4 stations)", xy=core_c, xytext=label_pt,
                fontsize=9.5, ha="left", color="#1a1a1a", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color="#888888", linewidth=0.9,
                                shrinkA=0, shrinkB=8))

    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#a63328",
              markeredgecolor="white", markersize=17, label="Munich (city centre)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STATION_COLOR,
              markeredgecolor="white", markersize=11, label="PM2.5 monitoring station"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9.5, framealpha=0.95,
             edgecolor="#2b2b2b")

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Study Area — Munich Region PM2.5 Monitoring Network")
    ax.set_xlim(10.7, 11.95)
    ax.set_ylim(47.88, 48.46)

    fig.tight_layout()
    fig.savefig(OUT / "fig13_study_area_map.png", dpi=220)
    fig.savefig(OUT / "fig13_study_area_map.pdf")
    plt.close(fig)
    print("Saved fig13_study_area_map.png (simple version)")


if __name__ == "__main__":
    make_map_simple()
    make_rq1_grouped_bar()
    make_rq1_heatmap()
    make_rq2_grouped_bar()
    make_rq2_heatmap()
    make_rq2_summary()
    make_rq2_timeseries("augsburg_lfu")
    make_rq2_timeseries("johanneskirchen")
