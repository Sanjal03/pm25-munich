"""
feature_correlation_matrix.py
================================================================================
Pooled correlation matrix across PM2.5, co-pollutants, and meteorological
features -- all 7 final stations' hourly readings stacked together (so met
columns, which are shared across stations, repeat once per station). Matches
the style of a typical feature-correlation heatmap (e.g. Fig 1 in reference
papers) but restricted to a 2-hue diverging colormap per user's style request.
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
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "serif", "font.size": 11,
})

STATIONS = ["andechs_rothenfeld", "aug_bourges-platz", "aug_lfu",
            "mun_johanneskirchen", "mun_landshuter_allee", "mun_lothstrasse", "mun_stachus"]

df = pd.read_csv("Munich_AQ_clean.csv")

MET_COLS = ["temp_c", "humidity_pct", "wind_speed", "wind_u", "wind_v",
            "pressure", "pressure_diff_24h"]

rows = []
for st in STATIONS:
    cols = {"pm25": f"{st}_pm25", "pm10": f"{st}_pm10",
            "no2": f"{st}_no2", "o3": f"{st}_o3"}
    sub = pd.DataFrame({k: df[v] for k, v in cols.items() if v in df.columns})
    for c in MET_COLS:
        sub[c] = df[c]
    rows.append(sub)

pooled = pd.concat(rows, ignore_index=True)
pooled = pooled.rename(columns={
    "pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "o3": "O3",
    "temp_c": "Temperature", "humidity_pct": "Humidity",
    "wind_speed": "Wind Speed", "wind_u": "Wind U", "wind_v": "Wind V",
    "pressure": "Pressure", "pressure_diff_24h": "Pressure Δ 24h",
})

ORDER = ["PM2.5", "PM10", "NO2", "O3", "Temperature", "Humidity",
         "Wind Speed", "Wind U", "Wind V", "Pressure", "Pressure Δ 24h"]
corr = pooled[ORDER].corr()
corr.round(3).to_csv("results/feature_correlation_matrix.csv")

fig, ax = plt.subplots(figsize=(9.5, 8.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

ax.set_xticks(range(len(ORDER)))
ax.set_yticks(range(len(ORDER)))
ax.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=9.5)
ax.set_yticklabels(ORDER, fontsize=9.5)

for i in range(len(ORDER)):
    for j in range(len(ORDER)):
        val = corr.values[i, j]
        color = "white" if abs(val) > 0.6 else "#1a1a1a"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.5, color=color)

ax.set_title("Correlation Matrix: PM$_{2.5}$, Co-Pollutants, and Meteorological Features")
cbar = fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson correlation ($r$)")

fig.tight_layout()
fig.savefig(OUT / "fig_D_feature_correlation_matrix.png", dpi=220)
fig.savefig(OUT / "fig_D_feature_correlation_matrix.pdf")
plt.close(fig)
print("Saved fig_D_feature_correlation_matrix.png")
print(corr.round(2))
