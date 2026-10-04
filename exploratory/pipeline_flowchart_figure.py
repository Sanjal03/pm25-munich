"""
pipeline_flowchart_figure.py
================================================================================
Print-size (6.3 in wide) two-row flowchart of the thesis pipeline. Coordinates
are in inches, so a 7 pt font stays 7 pt when the figure is set at full text
width. Parallelogram = data, rectangle = process, diamond = decision.
Row 1 runs left to right, row 2 right to left.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

OUT = Path("thesis_figs")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "font.family": "serif"})
COLOR_A, COLOR_B, EDGE = "#2b6a99", "#5b6770", "#2b2b2b"
FS = 7.2

fig, ax = plt.subplots(figsize=(6.3, 3.5))
ax.set_xlim(0, 6.3)
ax.set_ylim(0, 3.5)
ax.axis("off")

W, H = 1.38, 0.86


def rect(cx, cy, text, color=COLOR_A, w=W, h=H):
    ax.add_patch(Rectangle((cx - w/2, cy - h/2), w, h, facecolor=color,
                           edgecolor=EDGE, linewidth=0.8, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FS, color="white",
            zorder=3, linespacing=1.25)


def parallelogram(cx, cy, text, color=COLOR_B, w=W - 0.12, h=H, skew=0.14):
    pts = [(cx - w/2 + skew, cy + h/2), (cx + w/2 + skew, cy + h/2),
           (cx + w/2 - skew, cy - h/2), (cx - w/2 - skew, cy - h/2)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=EDGE,
                         linewidth=0.8, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FS, color="white",
            zorder=3, linespacing=1.25)


def diamond(cx, cy, text, color=COLOR_B, w=1.45, h=1.05):
    pts = [(cx, cy + h/2), (cx + w/2, cy), (cx, cy - h/2), (cx - w/2, cy)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=EDGE,
                         linewidth=0.8, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FS - 0.3, color="white",
            zorder=3, linespacing=1.2)


def arrow(p1, p2, head=True):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>" if head else "-",
                                 color=EDGE, linewidth=0.9, mutation_scale=8,
                                 zorder=1, shrinkA=0, shrinkB=0))


X = [0.82, 2.35, 3.88, 5.45]
Y1, Y2 = 2.95, 0.95

# Row 1 (left -> right)
parallelogram(X[0], Y1, "Data collection\nUBA air quality,\nDWD weather,\n2019–2024")
rect(X[1], Y1, "Preprocessing\nimputation,\npercentile cap,\nUTC alignment")
rect(X[2], Y1, "Feature engineering\n36 features: PM$_{2.5}$\nlags, co-pollutants,\nweather, time")
rect(X[3], Y1, "Training and\nPhase 1 benchmark\n5 models +\nPersistence")
arrow((X[0] + 0.62, Y1), (X[1] - W/2, Y1))
arrow((X[1] + W/2, Y1), (X[2] - W/2, Y1))
arrow((X[2] + W/2, Y1), (X[3] - W/2, Y1))

# Fork below the benchmark box
FORK = 2.05
arrow((X[3], Y1 - H/2), (X[3], FORK), head=False)
arrow((X[3], FORK), (X[3], Y2 + H/2))
GX = X[2] + 0.1
arrow((X[3], FORK), (GX, FORK), head=False)
arrow((GX, FORK), (GX, Y2 + 0.53))

# Row 2 (right -> left)
rect(X[3], Y2, "RQ1: feature-group\nablation (champion,\nleave-one-group-out)")
diamond(GX, Y2, "Validity gate\nR$^2\\geq$0.80,\n|Bias|<1.5\npassed?")
rect(X[1] + 0.05, Y2, "RQ2: recursive\ncounterfactual,\n5 models, rollout\n+ drift correction")
rect(X[0] - 0.02, Y2, "Interpretation:\nfeature importance,\nlockdown effect")
arrow((GX - 0.725, Y2), (X[1] + 0.05 + W/2, Y2))
ax.text(GX - 0.80, Y2 + 0.1, "yes", fontsize=6.5, ha="right", color="#333333")
arrow((X[1] + 0.05 - W/2, Y2), (X[0] - 0.02 + W/2, Y2))

# "No" branch
arrow((GX, Y2 - 0.53), (GX, 0.36))
ax.text(GX + 0.12, 0.43, "no", fontsize=6.5, ha="left", color="#333333")
ax.text(GX, 0.2, "excluded from RQ2", fontsize=6.5, ha="center", color="#333333")

# RQ1 result also feeds the interpretation
ax.plot([X[3], X[3]], [Y2 - H/2, 0.08], color=EDGE, lw=0.9, zorder=1)
ax.plot([X[3], X[0] - 0.02], [0.08, 0.08], color=EDGE, lw=0.9, zorder=1)
arrow((X[0] - 0.02, 0.08), (X[0] - 0.02, Y2 - H/2))

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(OUT / "fig_M4_pipeline_flowchart.png", dpi=300)
fig.savefig(OUT / "fig_M4_pipeline_flowchart.pdf")
plt.close(fig)
print("Saved fig_M4_pipeline_flowchart.png")
