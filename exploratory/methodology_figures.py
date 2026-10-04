"""
methodology_figures.py
================================================================================
Three schematic figures for the Methods chapter, matching the visual style
of pipeline/07_make_figures.py (same fonts, colors, rcParams) so they sit
consistently alongside the Data/Results figures:

  fig_M1_temporal_windows.png    -- train/gate/lockdown/test timeline
  fig_M2_recursive_rollout.png   -- counterfactual leakage-avoidance mechanism
  fig_M3_pipeline_overview.png   -- Phase 1 -> 2 -> 3 flow

These are schematics, not data plots -- no results/*.csv dependency.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from datetime import date

OUT = Path("thesis_figs")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fbfaf7",
    "font.family": "serif",
    "font.size": 11.5, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.edgecolor": "#2b2b2b", "axes.linewidth": 1.1,
})

TRAIN_COLOR = "#2b6a99"
GATE_COLOR = "#c9782e"
LOCKDOWN_COLOR = "#a63328"
TEST_COLOR = "#1f8a70"


# ── Figure M1: Temporal windows ─────────────────────────────────────────────
def make_temporal_windows():
    fig, ax = plt.subplots(figsize=(11, 3.2))

    segments = [
        (date(2019, 1, 1), date(2020, 1, 15), TRAIN_COLOR, "Train"),
        (date(2020, 1, 15), date(2020, 3, 21), GATE_COLOR, "Validity\nGate"),
        (date(2020, 3, 22), date(2020, 5, 4), LOCKDOWN_COLOR, "Lockdown\n(counterfactual)"),
        (date(2020, 5, 5), date(2024, 1, 1), TRAIN_COLOR, "Train"),
        (date(2024, 1, 1), date(2024, 12, 31), TEST_COLOR, "Test (2024)"),
    ]

    y0, height = 0.3, 0.5
    for start, end, color, label in segments:
        width_days = (mdates.date2num(end) - mdates.date2num(start))
        ax.barh(y0, width_days, left=mdates.date2num(start), height=height,
                color=color, edgecolor="#2b2b2b", linewidth=0.8, align="edge")

    # Labels for the two narrow, important windows (annotated above with leader lines)
    gate_mid = mdates.date2num(date(2020, 2, 17))
    lock_mid = mdates.date2num(date(2020, 4, 12))
    ax.annotate("Validity Gate\n(15 Jan–21 Mar 2020)\nheld out, evaluated,\nnever trained on",
                xy=(gate_mid, y0 + height), xytext=(gate_mid - 240, y0 + 1.15),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="-", color="#555555", linewidth=0.9))
    ax.annotate("Lockdown window\n(22 Mar–4 May 2020)\nexcluded from training,\nrecursive counterfactual here",
                xy=(lock_mid, y0 + height), xytext=(lock_mid + 300, y0 + 2.15),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="-", color="#555555", linewidth=0.9))
    ax.annotate("Test (2024)\nordinary 1-step-ahead\nevaluation",
                xy=(mdates.date2num(date(2024, 6, 1)), y0 + height),
                xytext=(mdates.date2num(date(2024, 6, 1)), y0 + 1.15),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="-", color="#555555", linewidth=0.9))

    ax.set_xlim(mdates.date2num(date(2019, 1, 1)), mdates.date2num(date(2025, 1, 1)))
    ax.set_ylim(-0.3, 3.4)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_yticks([])
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.set_title("Temporal Structure of the Experimental Design", pad=18)

    handles = [plt.Rectangle((0, 0), 1, 1, color=TRAIN_COLOR, label="Training data"),
               plt.Rectangle((0, 0), 1, 1, color=GATE_COLOR, label="Validity Gate window"),
               plt.Rectangle((0, 0), 1, 1, color=LOCKDOWN_COLOR, label="Lockdown window"),
               plt.Rectangle((0, 0), 1, 1, color=TEST_COLOR, label="Standard test set")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=4, fontsize=9.5, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_M1_temporal_windows.png", dpi=220)
    fig.savefig(OUT / "fig_M1_temporal_windows.pdf")
    plt.close(fig)
    print("Saved fig_M1_temporal_windows.png")


# ── Figure M2: Recursive counterfactual rollout ─────────────────────────────
def make_recursive_rollout():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Recursive Counterfactual Rollout — Leakage-Avoidance Mechanism", pad=14)

    def box(x, y, w, h, text, color, fontsize=9.5, textcolor="white"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                            facecolor=color, edgecolor="#2b2b2b", linewidth=1.0, zorder=2)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, zorder=3, linespacing=1.3)
        return b

    def arrow(xy1, xy2, color="#2b2b2b", style="-|>", lw=1.4, connectionstyle=None):
        a = FancyArrowPatch(xy1, xy2, arrowstyle=style, color=color, linewidth=lw,
                             mutation_scale=14, zorder=1,
                             connectionstyle=connectionstyle)
        ax.add_patch(a)

    # Real history box
    box(0.3, 3.6, 2.0, 1.0, "Real observed\nPM2.5 history\n(pre-lockdown)", TRAIN_COLOR)

    # Model boxes at t, t+1, t+2
    model_x = [3.1, 5.3, 7.5]
    for i, mx in enumerate(model_x):
        box(mx, 3.6, 1.6, 1.0, f"Model\n$t{'+'+str(i) if i else ''}$", "#4a4a4a")

    # Prediction outputs below each model
    for i, mx in enumerate(model_x):
        box(mx, 1.6, 1.6, 0.85, f"$\\hat{{y}}(t{'+'+str(i) if i else ''})$", LOCKDOWN_COLOR, fontsize=11)

    # Arrows: history -> first model
    arrow((2.3, 4.1), (3.1, 4.1))
    # Arrows: model -> prediction (down)
    for mx in model_x:
        arrow((mx + 0.8, 3.6), (mx + 0.8, 2.45))
    # Arrows: prediction feeds forward into NEXT model's lag features (the key mechanism)
    for i in range(len(model_x) - 1):
        x1 = model_x[i] + 0.8
        x2 = model_x[i + 1]
        arrow((x1, 2.0), (x2, 3.75), color=LOCKDOWN_COLOR, lw=1.8,
              connectionstyle="arc3,rad=-0.35")

    ax.text(5.0, 0.55,
            "Each prediction becomes the “history” used to build the NEXT hour's lag/rolling features\n"
            "— never the real (already lockdown-affected) observed value",
            ha="center", fontsize=9.5, color=LOCKDOWN_COLOR, style="italic")

    # Real exogenous inputs (weather/co-pollutants/temporal) — parallel, unbroken
    box(3.1, 5.1, 6.0, 0.75, "Weather, co-pollutants, and calendar features — real observed values throughout",
        TEST_COLOR, fontsize=9.5)
    for mx in model_x:
        arrow((mx + 0.8, 5.1), (mx + 0.8, 4.6), color=TEST_COLOR, lw=1.2)

    fig.tight_layout()
    fig.savefig(OUT / "fig_M2_recursive_rollout.png", dpi=220)
    fig.savefig(OUT / "fig_M2_recursive_rollout.pdf")
    plt.close(fig)
    print("Saved fig_M2_recursive_rollout.png")


# ── Figure M3: Three-phase pipeline overview ────────────────────────────────
def make_pipeline_overview():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Three-Phase Experimental Design", pad=14)

    def box(x, y, w, h, title, subtitle, color):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.12",
                            facecolor=color, edgecolor="#2b2b2b", linewidth=1.2, zorder=2)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2 + 0.35, title, ha="center", va="center",
                fontsize=12, color="white", fontweight="bold", zorder=3)
        ax.text(x + w/2, y + h/2 - 0.35, subtitle, ha="center", va="center",
                fontsize=9, color="white", zorder=3, linespacing=1.4)
        return b

    def arrow(xy1, xy2, label=None):
        a = FancyArrowPatch(xy1, xy2, arrowstyle="-|>", color="#2b2b2b",
                             linewidth=1.6, mutation_scale=16, zorder=1)
        ax.add_patch(a)
        if label:
            mx, my = (xy1[0] + xy2[0]) / 2, (xy1[1] + xy2[1]) / 2 + 0.55
            ax.text(mx, my, label, ha="center", fontsize=8.5, color="#555555", style="italic")

    box(0.3, 1.0, 3.0, 2.0, "Phase 1", "Benchmark all models\n+ Validity Gate check", TRAIN_COLOR)
    box(4.3, 1.0, 3.0, 2.0, "Phase 2 (RQ1)", "Leave-one-group-out\nfeature ablation", GATE_COLOR)
    box(8.3, 1.0, 3.0, 2.0, "Phase 3 (RQ2)", "Recursive counterfactual\nlockdown rollout", LOCKDOWN_COLOR)

    arrow((3.3, 2.0), (4.3, 2.0), "Champion model")
    arrow((7.3, 2.0), (8.3, 2.0), "Top-3 models")

    fig.tight_layout()
    fig.savefig(OUT / "fig_M3_pipeline_overview.png", dpi=220)
    fig.savefig(OUT / "fig_M3_pipeline_overview.pdf")
    plt.close(fig)
    print("Saved fig_M3_pipeline_overview.png")


if __name__ == "__main__":
    make_temporal_windows()
    make_recursive_rollout()
    make_pipeline_overview()
