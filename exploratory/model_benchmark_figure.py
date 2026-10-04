"""
model_benchmark_figure.py
================================================================================
Horizontal bar chart of Phase 1 benchmark results (mean MAE across 7 final
stations, validity-gate window), sorted best-to-worst -- same visual
convention as a typical model-comparison figure (lower is better), restricted
to 2 colors: the champion highlighted, everything else in the secondary color.
For the RESULTS chapter (this is what happened when the benchmark ran, not a
methodology description).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("thesis_figs")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fbfaf7",
    "font.family": "serif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#2b2b2b", "axes.linewidth": 1.1,
})

COLOR_CHAMPION = "#2b6a99"
COLOR_OTHER = "#5b6770"

LABELS = {"catboost": "CatBoost", "xgboost": "XGBoost", "lightgbm": "LightGBM",
          "dlinear": "DLinear", "patchtst": "PatchTST", "persistence": "Persistence"}

df = pd.read_csv("results/phase1_ranking.csv")
df = df.sort_values("mean_MAE", ascending=True).reset_index(drop=True)
df["label"] = df["model"].map(LABELS)
df["color"] = [COLOR_CHAMPION if i == 0 else COLOR_OTHER for i in range(len(df))]

fig, ax = plt.subplots(figsize=(5.4, 2.6))
y = range(len(df))[::-1]  # best model at top

bars = ax.barh(y, df["mean_MAE"], color=df["color"], edgecolor="#2b2b2b",
                linewidth=0.9, height=0.6, zorder=3)

for yi, val in zip(y, df["mean_MAE"]):
    ax.text(val + 0.015, yi, f"{val:.3f}", va="center", fontsize=8, color="#1a1a1a")

ax.set_yticks(list(y))
ax.set_yticklabels(df["label"], fontsize=8)
ax.set_xlabel("Mean MAE across 7 stations, µg/m$^3$ (lower is better)")
ax.set_xlim(0, max(df["mean_MAE"]) * 1.15)
ax.grid(axis="x", color="#dddddd", linewidth=0.7, zorder=0)

fig.tight_layout()
fig.savefig(OUT / "fig_R1_model_benchmark.png", dpi=220)
fig.savefig(OUT / "fig_R1_model_benchmark.pdf")
plt.close(fig)
print("Saved fig_R1_model_benchmark.png")
print(df[["label", "mean_MAE", "mean_R2"]])
