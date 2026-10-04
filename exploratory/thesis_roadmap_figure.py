"""
thesis_roadmap_figure.py
================================================================================
Simple vertical roadmap of the thesis structure, for the Introduction chapter.
Matches the visual style of the Methods chapter figures (same fonts, colors).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("thesis_figs")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "serif",
    "font.size": 10,
})

COLOR_A = "#2b6a99"
COLOR_B = "#5b6770"

CHAPTERS = [
    ("Chapter 1 — Introduction", "Motivation, research questions", COLOR_A),
    ("Chapter 2 — Literature Review", "Prior work, research gap", COLOR_B),
    ("Chapter 3 — Data", "Study area, sources, data quality", COLOR_A),
    ("Chapter 4 — Methodology", "Models, features, experimental design", COLOR_B),
    ("Chapter 5 — Results", "Benchmark, RQ1 ablation, RQ2 counterfactual", COLOR_A),
    ("Chapter 6 — Discussion", "Interpretation, limitations", COLOR_B),
    ("Chapter 7 — Conclusion", "Summary, contributions, future work", COLOR_A),
]

n = len(CHAPTERS)
box_h, gap = 1.0, 0.35
total_h = n * box_h + (n - 1) * gap

fig, ax = plt.subplots(figsize=(6.3, 6.4))
ax.set_xlim(0, 10)
ax.set_ylim(-0.2, total_h + 0.2)
ax.axis("off")

y = total_h
for i, (title, subtitle, color) in enumerate(CHAPTERS):
    y_top = y
    y_bot = y - box_h
    box = FancyBboxPatch((0.5, y_bot), 9.0, box_h,
                          boxstyle="round,pad=0.08,rounding_size=0.1",
                          facecolor=color, edgecolor="#2b2b2b", linewidth=1.1, zorder=2)
    ax.add_patch(box)
    ax.text(5.0, y_bot + box_h * 0.64, title, ha="center", va="center",
            fontsize=11.5, color="white", fontweight="bold", zorder=3)
    ax.text(5.0, y_bot + box_h * 0.29, subtitle, ha="center", va="center",
            fontsize=9.5, color="white", zorder=3)

    if i < n - 1:
        arrow = FancyArrowPatch((5.0, y_bot), (5.0, y_bot - gap),
                                 arrowstyle="-|>", color="#2b2b2b",
                                 linewidth=1.4, mutation_scale=12, zorder=1)
        ax.add_patch(arrow)

    y = y_bot - gap

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "fig_I1_thesis_roadmap.png", dpi=220)
fig.savefig(OUT / "fig_I1_thesis_roadmap.pdf")
plt.close(fig)
print("Saved fig_I1_thesis_roadmap.png")
