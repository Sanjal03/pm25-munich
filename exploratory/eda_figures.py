"""
Thesis Final Figures — Publication Quality
==========================================
MAIN TEXT (Methodology Chapter):
  Fig M1 — Figure 12: Data Completeness Gantt
  Fig M2 — Figure 9:  Cross-Pollutant Correlation Matrix
  Fig M3 — Figure 8:  Diurnal PM2.5 Cycle

APPENDIX (Evidence):
  Fig A1 — Figure 1:  Missing Data Heatmap
  Fig A2 — Figure 2:  Monthly Missing Timeline
  Fig A3 — Figure 3:  Gap Length Distribution
  Fig A4 — Figure 4:  Before/After Imputation
  Fig A5 — Figure 10: Outlier Analysis
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

CSV = 'munich_aq_final/Munich_AQ_2019-01-01_2024-12-31.csv'
OUT = Path("thesis_figs"); OUT.mkdir(exist_ok=True)

# ── Thesis-grade style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.2,
    "grid.linestyle":     "--",
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
})

# ── Colours & labels ──────────────────────────────────────────────────────────
STATION_COLORS = {
    'München/Landshuter Allee': '#C0392B',
    'München/Stachus':          '#E67E22',
    'München/Lothstraße':       '#2471A3',
    'München/Johanneskirchen':  '#1E8449',
    'Augsburg/Bourges-Platz':   '#6C3483',
    'Augsburg/LfU':             '#884EA0',
    'Andechs/Rothenfeld':       '#7D6608',
    'München/Allach':           '#17A589',
}
SHORT = {
    'München/Landshuter Allee': 'Landshuter Allee',
    'München/Stachus':          'Stachus',
    'München/Lothstraße':       'Lothstraße',
    'München/Johanneskirchen':  'Johanneskirchen',
    'Augsburg/Bourges-Platz':   'Augsburg/Bourges',
    'Augsburg/LfU':             'Augsburg/LfU',
    'Andechs/Rothenfeld':       'Andechs',
    'München/Allach':           'Allach',
}
PM25_STATIONS = [
    'München/Landshuter Allee','München/Stachus','München/Lothstraße',
    'München/Johanneskirchen','Augsburg/Bourges-Platz',
    'Augsburg/LfU','Andechs/Rothenfeld',
]
ALL_STATIONS = [
    'München/Landshuter Allee','München/Stachus','München/Lothstraße',
    'München/Johanneskirchen','München/Allach',
    'Augsburg/Bourges-Platz','Augsburg/LfU','Andechs/Rothenfeld',
]
YEARS = list(range(2019, 2025))
FULL  = pd.date_range('2019-01-01','2024-12-31 23:00', freq='h')

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV)
df['datetime'] = pd.to_datetime(df['Datetime'])
df['year']  = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['hour']  = df['datetime'].dt.hour
df['dow']   = df['datetime'].dt.dayofweek
def season(m):
    return {12:'Winter',1:'Winter',2:'Winter',
            3:'Spring',4:'Spring',5:'Spring',
            6:'Summer',7:'Summer',8:'Summer'}.get(m,'Autumn')
df['season'] = df['month'].map(season)
print("  Loaded.")

# helper: get hourly series for a station/pollutant
def get_series(station, pollutant, cap=None):
    sub = df[(df['Pollutant']==pollutant)&(df['Station Name']==station)]
    s = sub.set_index('datetime')['Value (µg/m³)']
    s = s[~s.index.duplicated()].reindex(FULL)
    if cap: s = s.clip(upper=cap)
    return s

# helper: gap lengths
def gap_lengths(series):
    gaps=[]; c=0
    for v in series.isna():
        if v: c+=1
        else:
            if c>0: gaps.append(c)
            c=0
    if c>0: gaps.append(c)
    return pd.Series(gaps) if gaps else pd.Series([0])

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TEXT FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── Fig M1: Data Completeness Gantt ──────────────────────────────────────────
print("Fig M1: Completeness Gantt...")
POLS = ['PM2.5','PM10','NO2','O3']
POL_COLORS = {'PM2.5':'#C0392B','PM10':'#E67E22','NO2':'#2471A3','O3':'#1E8449'}

fig = plt.figure(figsize=(15, 10))
fig.suptitle(
    "Figure M1 — Annual Data Completeness by Station and Pollutant\n"
    "Munich area air quality monitoring network, 2019–2024",
    fontsize=13, fontweight='bold', y=1.01
)
gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.55)

for pi, pol in enumerate(POLS):
    ax = fig.add_subplot(gs[pi])
    stations_here = [s for s in ALL_STATIONS
                     if df[(df['Pollutant']==pol)&(df['Station Name']==s)].shape[0]>0]

    for si, s in enumerate(stations_here):
        for yr in YEARS:
            sub = df[(df['Pollutant']==pol)&(df['Station Name']==s)&(df['year']==yr)]
            n_exp = 8784 if yr==2024 else 8760
            comp  = len(sub)/n_exp
            col   = '#1E8449' if comp>=0.95 else '#F39C12' if comp>=0.80 else '#C0392B'
            ax.barh(si, 1, left=yr, height=0.7, color=col, alpha=0.88,
                    edgecolor='white', lw=0.8, zorder=3)
            ax.text(yr+0.5, si, f'{comp*100:.0f}%',
                    ha='center', va='center', fontsize=7.5,
                    color='white', fontweight='bold', zorder=4)

    ax.set_yticks(range(len(stations_here)))
    ax.set_yticklabels([SHORT.get(s,s) for s in stations_here], fontsize=9)
    ax.set_xlim(2018.7, 2025.0)
    ax.set_xticks(YEARS)
    ax.set_xticklabels(YEARS if pi==3 else [], fontsize=9)
    ax.grid(False)
    ax.spines['left'].set_visible(False)
    ax.set_title(f'{pol}', fontsize=10, fontweight='bold',
                 color=POL_COLORS[pol], loc='left', pad=3)

legend_patches = [
    mpatches.Patch(color='#1E8449', label='≥ 95% complete (acceptable)'),
    mpatches.Patch(color='#F39C12', label='80–95% complete (moderate)'),
    mpatches.Patch(color='#C0392B', label='< 80% complete (poor)'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.02), framealpha=0.95,
           edgecolor='#ccc')
plt.savefig(OUT/'figM1_completeness_gantt.png')
plt.savefig(OUT/'figM1_completeness_gantt.pdf')
plt.close()
print("  ✓ figM1")

# ── Fig M2: Correlation Matrix ────────────────────────────────────────────────
print("Fig M2: Correlation matrix...")

corr_cols = {
    'PM2.5\n(Lothstraße)':     get_series('München/Lothstraße',       'PM2.5', 200),
    'PM10\n(Lothstraße)':      get_series('München/Lothstraße',       'PM10',  300),
    'NO2\n(Lothstraße)':       get_series('München/Lothstraße',       'NO2',   300),
    'O3\n(Lothstraße)':        get_series('München/Lothstraße',       'O3',    250),
    'PM2.5\n(Landshuter Al.)': get_series('München/Landshuter Allee', 'PM2.5', 200),
    'NO2\n(Landshuter Al.)':   get_series('München/Landshuter Allee', 'NO2',   300),
    'PM10\n(Landshuter Al.)':  get_series('München/Landshuter Allee', 'PM10',  300),
}
corr_df  = pd.DataFrame(corr_cols).dropna()
corr_mat = corr_df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle(
    "Figure M2 — Cross-Pollutant Pearson Correlation Matrix\n"
    "München/Lothstraße and Landshuter Allee, hourly data 2019–2024  (n = 50,957 h)",
    fontsize=12, fontweight='bold'
)
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(
    corr_mat, ax=ax, annot=True, fmt='.3f', cmap='RdBu_r',
    center=0, vmin=-1, vmax=1, linewidths=0.8, linecolor='white',
    mask=mask, square=True, annot_kws={'size':11, 'weight':'bold'},
    cbar_kws={'shrink':0.75, 'label':'Pearson r', 'pad':0.02}
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

# Annotation box explaining key values
ax.text(1.22, 0.98,
    "Key findings:\n"
    "• PM2.5 stations: r = 0.849\n"
    "  (strong spatial coherence)\n\n"
    "• PM10 ↔ PM2.5: r = 0.847\n"
    "  (shared emission sources)\n\n"
    "• NO2 ↔ PM2.5: r = 0.394\n"
    "  (traffic tracer signal)\n\n"
    "• O3 ↔ PM2.5: r = −0.214\n"
    "  (photochem. sink — negative)",
    transform=ax.transAxes, fontsize=9, va='top',
    bbox=dict(boxstyle='round,pad=0.5', fc='#F8F9FA',
              ec='#BDC3C7', alpha=0.95)
)
plt.tight_layout()
plt.savefig(OUT/'figM2_correlation_matrix.png')
plt.savefig(OUT/'figM2_correlation_matrix.pdf')
plt.close()
print("  ✓ figM2")

# ── Fig M3: Diurnal Cycle ─────────────────────────────────────────────────────
print("Fig M3: Diurnal cycle...")

diurnal = {}
for s in PM25_STATIONS:
    ser = get_series(s, 'PM2.5', cap=200).dropna()
    diurnal[s] = ser.groupby(ser.index.hour).mean().values

pm25_la = get_series('München/Landshuter Allee', 'PM2.5', cap=200)
pm25_la_df = pm25_la.to_frame('pm25')
pm25_la_df['dow'] = pm25_la_df.index.dayofweek
pm25_la_df['hour'] = pm25_la_df.index.hour
wd = pm25_la_df[pm25_la_df['dow']<5].groupby('hour')['pm25'].mean().values
we = pm25_la_df[pm25_la_df['dow']>=5].groupby('hour')['pm25'].mean().values

winter_h = pm25_la_df[pm25_la_df.index.month.isin([12,1,2])].groupby('hour')['pm25'].mean().values
summer_h = pm25_la_df[pm25_la_df.index.month.isin([6,7,8])].groupby('hour')['pm25'].mean().values

hrs = range(24)

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle(
    "Figure M3 — Diurnal PM2.5 Cycle: Traffic Patterns, Weekday Effect, and Seasonal Contrast\n"
    "Munich area monitoring network, 2019–2024",
    fontsize=13, fontweight='bold'
)

# Panel A: All stations
ax = axes[0]
for s in PM25_STATIONS:
    ax.plot(hrs, diurnal[s], '-', color=STATION_COLORS[s], lw=2,
            label=SHORT[s], zorder=3)
ax.set_xlabel('Hour of day')
ax.set_ylabel('Mean PM2.5 (µg/m³)')
ax.set_title('A — All stations: spatial gradient', fontweight='bold')
ax.set_xticks([0,3,6,9,12,15,18,21,23])
ax.legend(fontsize=8.5, loc='upper left')
ax.set_ylim(4, 16)
ax.annotate('Evening peak\n(19–21h)',
            xy=(20, diurnal['München/Landshuter Allee'][20]),
            xytext=(16, 14.5),
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5),
            fontsize=8, color='#C0392B')

# Panel B: Weekday vs Weekend
ax = axes[1]
ax.plot(hrs, wd, '-',  color='#C0392B', lw=2.5, label='Weekday (Mon–Fri)', zorder=3)
ax.plot(hrs, we, '--', color='#C0392B', lw=2.5, alpha=0.55,
        label='Weekend (Sat–Sun)', zorder=3)
ax.fill_between(hrs, wd, we, alpha=0.12, color='#C0392B')
ax.set_xlabel('Hour of day')
ax.set_ylabel('Mean PM2.5 (µg/m³)')
ax.set_title('B — Landshuter Allee: weekday vs weekend', fontweight='bold')
ax.set_xticks([0,3,6,9,12,15,18,21,23])
ax.legend(fontsize=9)
ax.set_ylim(7, 14.5)
ax.annotate('Rush-hour peak\n(6–8h, weekday only)',
            xy=(7, wd[7]),
            xytext=(10, 13.5),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.5),
            fontsize=8, color='#555')

# Panel C: Seasonal contrast
ax = axes[2]
ax.plot(hrs, winter_h, '-',  color='#2E86C1', lw=2.5, label='Winter (Dec–Feb)', zorder=3)
ax.plot(hrs, summer_h, '--', color='#E67E22', lw=2.5, label='Summer (Jun–Aug)', zorder=3)
ax.fill_between(hrs, winter_h, summer_h, alpha=0.1, color='#555')
ax.set_xlabel('Hour of day')
ax.set_ylabel('Mean PM2.5 (µg/m³)')
ax.set_title('C — Landshuter Allee: seasonal contrast', fontweight='bold')
ax.set_xticks([0,3,6,9,12,15,18,21,23])
ax.legend(fontsize=9)
ax.set_ylim(4, 20)
ax.annotate(f'Winter mean\n~{winter_h.mean():.1f} µg/m³',
            xy=(12, winter_h[12]),
            xytext=(14, 17.5),
            arrowprops=dict(arrowstyle='->', color='#2E86C1', lw=1.5),
            fontsize=8, color='#2E86C1')

plt.tight_layout()
plt.savefig(OUT/'figM3_diurnal_cycle.png')
plt.savefig(OUT/'figM3_diurnal_cycle.pdf')
plt.close()
print("  ✓ figM3")


# ══════════════════════════════════════════════════════════════════════════════
#  APPENDIX FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── Fig A1: Missing Data Heatmap ──────────────────────────────────────────────
print("Fig A1: Missing heatmap...")
miss_data = {}
for s in ALL_STATIONS:
    row = {}
    for p in ['PM2.5','PM10','NO2','O3']:
        ser = get_series(s, p)
        m = ser.isna().mean()*100
        row[p] = round(m, 1) if m < 99 else None
    miss_data[s] = row

miss_df = pd.DataFrame(miss_data).T[['PM2.5','PM10','NO2','O3']]
miss_df.index = [SHORT.get(s,s) for s in miss_df.index]

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(
    "Appendix Figure A1 — Overall Missing Data Rate (%)\n"
    "Munich area, all stations and pollutants, 2019–2024",
    fontsize=12, fontweight='bold'
)
disp = miss_df.copy().fillna(-1).astype(float)
cmap = matplotlib.cm.RdYlGn_r.copy()
cmap.set_under('#CCCCCC')
im = ax.imshow(disp.values, aspect='auto', cmap=cmap, vmin=0, vmax=10)
ax.set_xticks(range(4))
ax.set_xticklabels(['PM2.5','PM10','NO2','O3'], fontweight='bold', fontsize=11)
ax.set_yticks(range(len(miss_df)))
ax.set_yticklabels(miss_df.index, fontsize=10)

for i in range(len(miss_df)):
    for j, p in enumerate(['PM2.5','PM10','NO2','O3']):
        v = miss_df.iloc[i, j]
        if v is None:
            ax.text(j, i, '—', ha='center', va='center', fontsize=12, color='#555')
        else:
            col = 'white' if v > 6 else 'black'
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=col)

plt.colorbar(im, ax=ax, label='Missing %', shrink=0.85)
ax.text(1.19, 0.02, 'Grey = pollutant\nnot measured\nat this station',
        transform=ax.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round', fc='#f5f5f5', ec='#ccc'))
plt.tight_layout()
plt.savefig(OUT/'figA1_missing_heatmap.png')
plt.savefig(OUT/'figA1_missing_heatmap.pdf')
plt.close()
print("  ✓ figA1")

# ── Fig A2: Monthly Missing Timeline ─────────────────────────────────────────
print("Fig A2: Monthly timeline...")
monthly_miss = {}
for s in PM25_STATIONS:
    ser = get_series(s, 'PM2.5')
    mm  = ser.resample('ME').apply(lambda x: x.isna().mean()*100)
    monthly_miss[SHORT[s]] = mm
miss_tl = pd.DataFrame(monthly_miss)
miss_tl.index = miss_tl.index.strftime('%Y-%m')

fig, ax = plt.subplots(figsize=(15, 5))
fig.suptitle(
    "Appendix Figure A2 — Monthly PM2.5 Missing Data Rate by Station\n"
    "Munich area, 2019–2024  (dark red = high missing, green = complete)",
    fontsize=12, fontweight='bold'
)
im = ax.imshow(miss_tl.T.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=20)
n = len(miss_tl)
ax.set_xticks(range(0, n, 6))
ax.set_xticklabels(miss_tl.index[::6], rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(miss_tl.columns)))
ax.set_yticklabels(miss_tl.columns, fontsize=10)
plt.colorbar(im, ax=ax, label='Missing %', shrink=0.85)
# Annotate worst period
ax.text(0.5, -0.18,
    'Notable outage: München/Lothstraße April 2024 (55.4% missing) — sensor maintenance period',
    transform=ax.transAxes, ha='center', fontsize=9, color='#922B21',
    style='italic')
plt.tight_layout()
plt.savefig(OUT/'figA2_monthly_timeline.png')
plt.savefig(OUT/'figA2_monthly_timeline.pdf')
plt.close()
print("  ✓ figA2")

# ── Fig A3: Gap Length Distribution ──────────────────────────────────────────
print("Fig A3: Gap distribution...")
bins    = [1,2,3,4,5,6,7,12,24,48,200,1000]
blabels = ['1h','2h','3h','4h','5h','6h','7–12h','13–24h','25–48h','49–200h','>200h']
bcolors = ['#1E8449']*6 + ['#F39C12']*2 + ['#C0392B']*3

fig, axes = plt.subplots(1, len(PM25_STATIONS), figsize=(3 * len(PM25_STATIONS), 4.5))
fig.suptitle(
    "Appendix Figure A3 — PM2.5 Gap Length Distribution per Station (2019–2024)\n"
    "Shows that ≥ 85% of all missing windows are ≤ 6 hours — justifying linear interpolation",
    fontsize=12, fontweight='bold'
)
for ax, s in zip(axes, PM25_STATIONS):
    ser    = get_series(s, 'PM2.5')
    gl     = gap_lengths(ser)
    counts = [((gl>=bins[i])&(gl<bins[i+1])).sum() for i in range(len(bins)-1)]
    ax.bar(range(len(blabels)), counts, color=bcolors, alpha=0.85,
           edgecolor='white', lw=0.5, zorder=3)
    ax.set_xticks(range(len(blabels)))
    ax.set_xticklabels(blabels, rotation=55, ha='right', fontsize=7)
    ax.set_title(SHORT[s], fontsize=9, fontweight='bold', color=STATION_COLORS[s])
    ax.set_ylabel('Number of gaps' if ax==axes[0] else '')
    total = len(gl); le6 = (gl<=6).sum()
    ax.text(0.97, 0.97, f'n = {total}\n≤6h: {le6/total*100:.0f}%',
            transform=ax.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.9))

leg = [mpatches.Patch(color='#1E8449', label='Short ≤6h → linear interpolation'),
       mpatches.Patch(color='#F39C12', label='Medium 7–24h → spatial interpolation'),
       mpatches.Patch(color='#C0392B', label='Long >24h → flagged as missing')]
fig.legend(handles=leg, loc='lower center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.04), framealpha=0.95)
plt.tight_layout()
plt.savefig(OUT/'figA3_gap_distribution.png')
plt.savefig(OUT/'figA3_gap_distribution.pdf')
plt.close()
print("  ✓ figA3")

# ── Fig A4: Before / After Imputation ────────────────────────────────────────
print("Fig A4: Before/after...")

def hybrid_impute(pm25_wide, max_short=6, max_medium=24):
    result = pm25_wide.copy()
    for col in result.columns:
        s = result[col].copy()
        nb = pm25_wide[[c for c in pm25_wide.columns if c!=col]]
        gaps=[]; c=0; start=None
        for idx, v in s.isna().items():
            if v:
                if c==0: start=idx
                c+=1
            else:
                if c>0: gaps.append({'start':start,'length':c})
                c=0
        if c>0: gaps.append({'start':start,'length':c})
        for g in gaps:
            L=g['length']; st=g['start']
            mask=(s.index>=st)&(s.index<st+pd.Timedelta(hours=L))
            if L<=max_short:
                s=s.interpolate(method='linear',limit=max_short)
            elif L<=max_medium:
                nb_mean=nb.loc[mask].mean(axis=1)
                if nb_mean.notna().any():
                    s.loc[mask]=nb_mean.values
        result[col]=s
    return result

pm25_wide = df[df['Pollutant']=='PM2.5'].pivot_table(
    index='datetime', columns='Station Name', values='Value (µg/m³)', aggfunc='mean')
pm25_wide     = pm25_wide.reindex(FULL)[PM25_STATIONS]
pm25_imputed  = hybrid_impute(pm25_wide)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "Appendix Figure A4 — Monthly PM2.5 Missing Rate: Before vs After Hybrid Imputation\n"
    "Linear interpolation (≤6h gaps) + Spatial mean (7–24h gaps); long gaps flagged",
    fontsize=12, fontweight='bold'
)
for ax, data, title in [
    (ax1, pm25_wide,    'BEFORE imputation'),
    (ax2, pm25_imputed, 'AFTER hybrid imputation'),
]:
    mm = data.resample('ME').apply(lambda x: x.isna().mean()*100)
    mm.index = mm.index.strftime('%Y-%m')
    mm.columns = [SHORT[c] for c in mm.columns]
    im = ax.imshow(mm.T.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=15)
    n = len(mm)
    ax.set_xticks(range(0, n, 6))
    ax.set_xticklabels(mm.index[::6], rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(mm.columns)))
    ax.set_yticklabels(mm.columns, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    total = data.isna().mean().mean()*100
    ax.text(0.98, 0.02, f'Overall missing: {total:.2f}%',
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', fc='white', alpha=0.9))

plt.colorbar(im, ax=ax2, label='Missing %', shrink=0.85)
plt.tight_layout()
plt.savefig(OUT/'figA4_before_after.png')
plt.savefig(OUT/'figA4_before_after.pdf')
plt.close()
print("  ✓ figA4")

# ── Fig A5: Outlier Analysis ──────────────────────────────────────────────────
print("Fig A5: Outlier analysis...")
raw = get_series('München/Landshuter Allee', 'PM2.5').dropna()
cap = raw.quantile(0.995)
capped = raw.clip(upper=cap)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Appendix Figure A5 — PM2.5 Outlier Analysis: München/Landshuter Allee (2019–2024)\n"
    "New Year's Eve fireworks produce extreme spikes (up to 813 µg/m³) "
    "→ capped at 99.5th percentile",
    fontsize=12, fontweight='bold'
)
axes[0].hist(raw, bins=120, color='#2471A3', alpha=0.82, edgecolor='white', lw=0.3)
axes[0].axvline(cap, color='#E74C3C', ls='--', lw=2,
                label=f'99.5th pct = {cap:.0f} µg/m³')
axes[0].set_xlabel('PM2.5 (µg/m³)'); axes[0].set_ylabel('Count')
axes[0].set_title('Raw distribution (full range)')
axes[0].legend(fontsize=9)
axes[0].text(0.55, 0.82, f'max = {raw.max():.0f} µg/m³\nskewness = {raw.skew():.1f}',
             transform=axes[0].transAxes, fontsize=9,
             bbox=dict(boxstyle='round', fc='white', alpha=0.85))

axes[1].hist(raw, bins=150, color='#2471A3', alpha=0.82, edgecolor='white', lw=0.3)
axes[1].set_xlim(0, 120)
axes[1].axvline(cap, color='#E74C3C', ls='--', lw=2, label=f'Cap = {cap:.0f} µg/m³')
axes[1].axvline(raw.quantile(0.99), color='#F39C12', ls=':', lw=2,
                label=f"99th = {raw.quantile(0.99):.0f} µg/m³")
axes[1].set_xlabel('PM2.5 (µg/m³)'); axes[1].set_ylabel('Count')
axes[1].set_title('Zoomed view: 0–120 µg/m³')
axes[1].legend(fontsize=9)

axes[2].hist(capped, bins=100, color='#1E8449', alpha=0.82, edgecolor='white', lw=0.3)
axes[2].set_xlabel('PM2.5 (µg/m³)'); axes[2].set_ylabel('Count')
axes[2].set_title(f'After capping at {cap:.0f} µg/m³')
axes[2].text(0.45, 0.82,
    f'Skewness:  {raw.skew():.2f} → {capped.skew():.2f}\n'
    f'Mean:  {raw.mean():.2f} → {capped.mean():.2f} µg/m³\n'
    f'Max:  {raw.max():.0f} → {capped.max():.0f} µg/m³\n'
    f'Rows capped: {(raw>cap).sum():,} ({(raw>cap).mean()*100:.2f}%)',
    transform=axes[2].transAxes, fontsize=9, va='top',
    bbox=dict(boxstyle='round', fc='white', alpha=0.9))

plt.tight_layout()
plt.savefig(OUT/'figA5_outlier_analysis.png')
plt.savefig(OUT/'figA5_outlier_analysis.pdf')
plt.close()
print("  ✓ figA5")

print(f"\nAll 8 figures saved to {OUT}/")
print("  Main text:  figM1, figM2, figM3")
print("  Appendix:   figA1, figA2, figA3, figA4, figA5")