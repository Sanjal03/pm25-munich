"""
09_make_fig21_horizon.py — Figure 21: Recursive Rollout Error Accumulation
============================================================================
Consumes results/rq2_horizon_daily_summary.csv (from
08_rq2_recursive_horizon_check.py) and the real results/rq2_*.json lockdown
gaps, produces thesis_figs/fig21_rq2_horizon_error_accumulation.png.

Two lines: daily MAE (does error keep climbing, or plateau?) and cumulative
MAE (mean error over [0, day] — the fair comparison point against a 44-day
gap estimate, since the reported RQ2 gap is itself a whole-window mean, not
a single hour). A horizontal reference line marks the mean |lockdown gap|
across the real 35 RQ2 runs, so the reader can see at a glance whether
normal-period rollout drift is small relative to the effect being claimed.
"""
import json
import warnings
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT = Path("thesis_figs")
OUT.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fbfaf7",
    "font.family": "serif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.edgecolor": "#2b2b2b", "axes.linewidth": 1.1,
})


def mean_abs_lockdown_gap() -> float:
    gaps = []
    for f in RESULTS_DIR.glob("rq2_*.json"):
        if f.name.startswith("rq2_horizon"):
            continue
        d = json.loads(f.read_text())
        if d.get("gap") is not None:
            gaps.append(abs(d["gap"]))
    return sum(gaps) / len(gaps)


def make_figure():
    daily = pd.read_csv(RESULTS_DIR / "rq2_horizon_daily_summary.csv")
    ref_gap = mean_abs_lockdown_gap()

    fig, ax = plt.subplots(figsize=(6.3, 3.0))

    ax.plot(daily["day_index"], daily["MAE"], color="#a63328", lw=1.2,
             marker="o", ms=2.2, label="Daily MAE (normal-period rollout)")
    ax.fill_between(daily["day_index"],
                     (daily["MAE"] - daily["MAE_std"]).clip(lower=0),
                     daily["MAE"] + daily["MAE_std"],
                     color="#a63328", alpha=0.12, lw=0)
    ax.plot(daily["day_index"], daily["cumulative_MAE"], color="#2b6a99",
             lw=1.4, ls="--", label="Cumulative MAE (mean over [0, day])")

    ax.axhline(ref_gap, color="#1f8a70", lw=1.2, ls=":",
                label=f"Mean |lockdown gap| across real RQ2 runs ({ref_gap:.2f} µg/m³)")

    ax.set_xlabel("Day into recursive rollout (0 = first hour)")
    ax.set_ylabel("Mean Absolute Error (µg/m³)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="upper right", frameon=True)
    ax.grid(alpha=0.25, lw=0.6)

    fig.tight_layout()
    fig.savefig(OUT / "fig21_rq2_horizon_error_accumulation.png", dpi=200)
    fig.savefig(OUT / "fig21_rq2_horizon_error_accumulation.pdf")
    print(f"✓ Saved: {OUT / 'fig21_rq2_horizon_error_accumulation.png'}")

    print(f"\nDay-0 MAE:              {daily['MAE'].iloc[0]:.3f} µg/m³")
    print(f"Final-day MAE:          {daily['MAE'].iloc[-1]:.3f} µg/m³")
    print(f"Full-horizon cum. MAE:  {daily['cumulative_MAE'].iloc[-1]:.3f} µg/m³")
    print(f"Mean |real lockdown gap|: {ref_gap:.3f} µg/m³")


if __name__ == "__main__":
    make_figure()
