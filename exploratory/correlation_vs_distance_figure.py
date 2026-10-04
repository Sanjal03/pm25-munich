"""
correlation_vs_distance_figure.py
================================================================================
Scatter plot: PM2.5 correlation vs distance from the reference station
(München/Landshuter Allee), labeled by station and colored by environment
type -- makes the "not distance-dependent, environment type matters more"
finding visible at a glance. Matches the style of pipeline/07_make_figures.py.

Uses the raw correlation values quoted in the Data chapter text (0.944,
0.950, 0.912, 0.692, 0.885, 0.886) and the user's own precise distances
from Landshuter Allee, so the figure is consistent with the prose and table.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("thesis_figs")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fbfaf7",
    "font.family": "serif",
    "font.size": 11.5, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.edgecolor": "#2b2b2b", "axes.linewidth": 1.1,
})

TYPE_COLORS = {
    "Traffic": "#c9782e",
    "Urban background": "#2b6a99",
    "Suburban": "#1f8a70",
    "Rural background": "#5c8a3a",
}

# station: (distance_km, r, type), ordered by distance (closest first)
DATA = [
    ("München/Lothstraße",       1.5,  0.950, "Urban background"),
    ("München/Stachus",          2.5,  0.944, "Traffic"),
    ("München/Johanneskirchen",  8.7,  0.912, "Suburban"),
    ("Andechs/Rothenfeld",       30.2, 0.692, "Rural background"),
    ("Augsburg/LfU",             51.1, 0.886, "Suburban"),
    ("Augsburg/Bourges-Platz",   54.8, 0.885, "Urban background"),
]

fig, ax = plt.subplots(figsize=(9.5, 6))

x = range(len(DATA))
bar_colors = [TYPE_COLORS[etype] for _, _, _, etype in DATA]
bars = ax.bar(x, [r for _, _, r, _ in DATA], color=bar_colors,
              edgecolor="#2b2b2b", linewidth=1.0, width=0.62, zorder=3)

for i, (name, dist, r, etype) in enumerate(DATA):
    ax.text(i, r + 0.012, f"$r$={r:.3f}", ha="center", fontsize=10, zorder=4)
    ax.text(i, 0.605, f"{dist:.1f} km", ha="center", fontsize=9.5, color="#555555")

short_labels = [n.replace("München/", "").replace("Augsburg/", "Augsburg\n")
                for n, _, _, _ in DATA]
ax.set_xticks(list(x))
ax.set_xticklabels(short_labels, fontsize=10)

# Highlight the key comparison: Andechs vs. the two Augsburg stations
ax.annotate("", xy=(3.15, 0.705), xytext=(4.85, 0.80),
            arrowprops=dict(arrowstyle="<->", color="#a63328", linewidth=1.4,
                             connectionstyle="arc3,rad=-0.2"))
ax.text(2.7, 0.975, "Farther away, but\nHIGHER correlation",
        ha="center", fontsize=10, color="#a63328", style="italic", linespacing=1.3)

ax.set_ylabel("Pearson correlation ($r$) with reference station\n(daily-mean PM$_{2.5}$)")
ax.set_xlabel("Station (ordered by distance from reference station, closest → farthest)")
ax.set_title("PM$_{2.5}$ Correlation Does Not Decrease Monotonically with Distance")
ax.set_ylim(0.6, 1.05)

handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=t) for t, c in TYPE_COLORS.items()]
ax.legend(handles=handles, loc="upper right", fontsize=9.5, framealpha=0.95,
          edgecolor="#2b2b2b", title="Station type", ncol=1)

fig.tight_layout()
fig.savefig(OUT / "fig_D_correlation_vs_distance.png", dpi=220)
fig.savefig(OUT / "fig_D_correlation_vs_distance.pdf")
plt.close(fig)
print("Saved fig_D_correlation_vs_distance.png")
