"""
station_correlation_matrix_figure.py
================================================================================
Full 7x7 pairwise Pearson correlation matrix (daily-mean PM2.5, complete-case
days) for the final 7 stations, as a heatmap. Complements the single-reference
correlation-vs-distance figure by showing the full pairwise structure --
e.g. whether the two Augsburg stations cluster tightly with each other,
whether Andechs is isolated from the whole network or just the reference.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("thesis_figs")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fbfaf7",
    "font.family": "serif",
    "font.size": 11.5, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.edgecolor": "#2b2b2b", "axes.linewidth": 1.1,
})

corr = pd.read_csv("results/station_correlation_matrix.csv", index_col=0)
short_labels = [n.replace("München/", "").replace("Augsburg/", "Augsburg/\n")
                for n in corr.index]

fig, ax = plt.subplots(figsize=(8.5, 7.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=0.65, vmax=1.0)

ax.set_xticks(range(len(short_labels)))
ax.set_yticks(range(len(short_labels)))
ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=9.5)
ax.set_yticklabels(short_labels, fontsize=9.5)

for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.values[i, j]
        color = "white" if val > 0.9 or val < 0.7 else "#1a1a1a"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9.5, color=color)

ax.set_title("Pairwise PM$_{2.5}$ Correlation Between Stations")
cbar = fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson correlation ($r$)")

fig.tight_layout()
fig.savefig(OUT / "fig_D_correlation_matrix.png", dpi=220)
fig.savefig(OUT / "fig_D_correlation_matrix.pdf")
plt.close(fig)
print("Saved fig_D_correlation_matrix.png")
