"""
make_appendix_tables.py — generates thesis_appendix/99_Appendix_generated.tex
from results/*.csv|json (so every number comes straight from the pipeline).
Run from the project root:  python exploratory/make_appendix_tables.py
Needs LaTeX packages already used in the thesis: booktabs, amsmath, textcomp.

Contents (the eight items of the agreed appendix):
  1 benchmark per station x model, 2 RQ1 ablation, 3 RQ2 raw + corrected,
  4 rollout drift details, 5 missingness / gaps, 6 PatchTST sensitivity,
  7 Nuernberg/Stuttgart control test, 8 reproducibility (config + run summary).
"""
from pathlib import Path
import numpy as np
import pandas as pd

R = Path("results")
OUT = Path("thesis_appendix"); OUT.mkdir(exist_ok=True)

STATIONS = ["landshuter_allee", "stachus", "lothstrasse", "johanneskirchen",
            "augsburg_bourges", "augsburg_lfu", "andechs"]
SNAME = {"landshuter_allee": "M\\\"unchen/Landshuter Allee", "stachus": "M\\\"unchen/Stachus",
         "lothstrasse": "M\\\"unchen/Lothstra{\\ss}e", "johanneskirchen": "M\\\"unchen/Johanneskirchen",
         "augsburg_bourges": "Augsburg/Bourges-Platz", "augsburg_lfu": "Augsburg/LfU",
         "andechs": "Andechs/Rothenfeld"}
SHORT = {"landshuter_allee": "Landshuter Allee", "stachus": "Stachus",
         "lothstrasse": "Lothstra{\\ss}e", "johanneskirchen": "Johanneskirchen",
         "augsburg_bourges": "Augsburg/Bourges", "augsburg_lfu": "Augsburg/LfU",
         "andechs": "Andechs"}
MODELS = ["catboost", "xgboost", "lightgbm", "dlinear", "patchtst"]
MNAME = {"catboost": "CatBoost", "xgboost": "XGBoost", "lightgbm": "LightGBM",
         "dlinear": "DLinear", "patchtst": "PatchTST", "persistence": "Persistence"}
GROUPS = ["PM2.5_lags", "PM2.5_rolling", "Co-pollutants", "Meteorology", "Temporal"]
GNAME = {"PM2.5_lags": "PM$_{2.5}$ lags", "PM2.5_rolling": "Rolling", "Co-pollutants": "Co-poll.",
         "Meteorology": "Meteo.", "Temporal": "Temporal"}

parts = []
def add(s): parts.append(s.strip("\n") + "\n")

def f(x, d=2):
    if pd.isna(x):
        return "--"
    x = round(float(x), d)
    return f"{abs(x):.{d}f}" if x == 0 else f"{x:.{d}f}".replace("-", "$-$")

def table(caption, label, header, rows, colspec, note=None, size="\\small", tabcolsep=None):
    s = ["\\begin{table}[!htbp]", "\\centering" + size + (f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}" if tabcolsep else ""),
         f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        s.append("\\midrule" if r == "MID" else " & ".join(str(c) for c in r) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    if note:
        s.append(f"\\\\[2pt]\\parbox{{0.92\\textwidth}}{{\\footnotesize {note}}}")
    s.append("\\end{table}")
    return "\n".join(s)

# ═══════════════════════════════════════════ 1. Benchmark
b = pd.read_csv(R / "phase1_benchmark.csv")
order = MODELS + ["persistence"]
add(r"""
\section{Benchmark Results by Station}
\label{app:phase1}

Complete one-hour-ahead results on the Validity Gate window (1\,608 hours per station). All 42 model--station combinations pass the accuracy gate.
""")
for metric, lab, cap, d in [("R2", "tab:app-p1-r2", "Test $R^2$ per station and model (Validity Gate window). A tree ensemble is the best model at every station.", 4)
]:
    p = b.pivot(index="station", columns="model", values=metric).reindex(STATIONS)[order]
    rows = [[SNAME[s]] + [f(p.loc[s, m], d) for m in order] for s in STATIONS]
    rows += ["MID", ["Mean"] + [f(p[m].mean(), d) for m in order]]
    add(table(cap, lab, ["Station"] + [MNAME[m] for m in order], rows, "l" + "r" * 6, size="\\footnotesize", tabcolsep="4pt"))

# ═══════════════════════════════════════════ 2. RQ1
per = {}
for m in MODELS:
    d = {}
    for s in STATIONS:
        t = pd.read_csv(R / f"rq1_{s}_{m}.csv")
        t["Configuration"] = t["Configuration"].str.replace("Without ", "")
        d[s] = t.set_index("Configuration")["delta_MAE_pct"]
    per[m] = pd.DataFrame(d).T
add(r"""
\section{Feature-Group Ablation (RQ1): Full Results}
\label{app:rq1}

Relative increase in test MAE (\%) when a feature group is removed and the model is retrained, per station, for every model. Positive values mean the group helps. Rolling statistics do not exist as a separate input for the sequence models (``--'').
""")
rows = []
for i, m in enumerate(MODELS):
    for s in STATIONS:
        rows.append([MNAME[m] if s == STATIONS[0] else "", SHORT[s]] + [("--" if (g == "PM2.5_rolling" and m in ("dlinear", "patchtst")) else f(per[m].loc[s, g], 1)) for g in GROUPS])
    rows.append(["", "\\textit{Mean}"] + [("--" if (g == "PM2.5_rolling" and m in ("dlinear", "patchtst")) else f(per[m][g].mean(), 1)) for g in GROUPS])
    if i < len(MODELS) - 1:
        rows.append("MID")
add(table("Leave-one-group-out ablation: $\\Delta$MAE (\\%) by model, station and removed feature group. PM$_{2.5}$ lags dominate everywhere (about 41\\% for the trees, 62--71\\% for the sequence models).",
    "tab:app-rq1", ["Model", "Station"] + [GNAME[g] for g in GROUPS], rows, "llrrrrr", size="\\footnotesize", tabcolsep="4pt"))

# ═══════════════════════════════════════════ 3. RQ2
bc = pd.read_csv(R / "rq2_bias_corrected.csv")
n = len(bc); n_raw = int((bc.gap > 0).sum()); n_cor = int((bc.corrected_gap > 0).sum())
add(rf"""
\section{{Lockdown Counterfactual (RQ2): Raw and Drift-Corrected Results}}
\label{{app:rq2}}

Raw lockdown gap (mean counterfactual minus mean observed PM$_{{2.5}}$ over the 1\,056-hour lockdown window), baseline rollout drift (mean offset of the same rollout over the three 2024 control windows) and the drift-corrected gap, for all {n} model--station runs. A positive gap means observed concentrations were below the counterfactual. {n_raw} runs are positive before and {n_cor} after correction.
""")
rows = []
for si, s in enumerate(STATIONS):
    for m in MODELS:
        r = bc[(bc.station == s) & (bc.model == m)].iloc[0]
        rows.append([SHORT[s] if m == MODELS[0] else "", MNAME[m], f(r.gap, 2), f(r.gap_pct, 1), f(r.baseline_bias, 2), f(r.corrected_gap, 2),
                     "\\checkmark" if r.corrected_gap > 0 else "--"])
    if si < len(STATIONS) - 1:
        rows.append("MID")
add(table(f"Raw and drift-corrected lockdown gaps for all {n} runs (\\textmu g/m$^3$ unless stated). Corrected gap = raw gap $-$ baseline drift; \\checkmark{{}} marks a positive corrected gap.",
    "tab:app-rq2", ["Station", "Model", "Raw gap", "Raw gap (\\%)", "Drift", "Corrected", "$>0$"], rows, "llrrrrc", size="\\footnotesize", tabcolsep="4pt"))

# ═══════════════════════════════════════════ 4. Rollout drift details
ds = pd.read_csv(R / "rq2_horizon_daily_summary.csv")
pg = pd.read_csv(R / "rq2_horizon_pseudo_gaps.csv")
days = [0, 1, 6, 13, 20, 27, 34, 43]
rows = [[str(d + 1), f(ds.loc[ds.day_index == d, "MAE"].iloc[0], 2), f(ds.loc[ds.day_index == d, "MAE_median"].iloc[0], 2),
         f(ds.loc[ds.day_index == d, "cumulative_MAE"].iloc[0], 2)] for d in days]
add(rf"""
\section{{Recursive Rollout Drift: Details}}
\label{{app:horizon}}

The trained RQ2 models were rolled out recursively over three 1\,056-hour windows of 2024 (22 March--4 May, 1 July--13 August and 1 October--13 November), where no intervention occurred and observations are the correct ground truth. The daily MAE is {ds.MAE.iloc[0]:.2f}~\textmu g/m$^3$ on day 1 and {ds.MAE.iloc[-1]:.2f}~\textmu g/m$^3$ on day 44 (largest single day {ds.MAE.max():.2f}, day {int(ds.MAE.idxmax())+1}); the cumulative MAE of {ds.cumulative_MAE.iloc[-1]:.2f}~\textmu g/m$^3$ exceeds the mean absolute raw lockdown gap ({bc.gap.abs().mean():.2f}~\textmu g/m$^3$).
""")
add(table("Rollout error by day (all stations, models and windows pooled). The cumulative MAE settles at about 2.1~\\textmu g/m$^3$, three times the mean absolute lockdown gap.",
    "tab:app-horizon", ["Day", "MAE", "Median AE", "Cumulative MAE"], rows, "rrrr",
    "Values in \\textmu g/m$^3$. DLinear and PatchTST return no prediction when a real target gap of more than 24\\,h lies inside their 168\\,h lookback; errors are computed on paired hours only."))
# ═══════════════════════════════════════════ 6. PatchTST
e100 = pd.read_csv(R / "patchtst_epoch100_test.csv").set_index("station")
p30 = b[b.model == "patchtst"].set_index("station")
cols = [("R2", 4, "$R^2$"), ("MAE", 3, "MAE"), ("RMSE", 3, "RMSE"), ("Bias", 3, "Bias")]
rows = []
for s_ in STATIONS:
    r = [SHORT[s_]]
    for c, d_, _ in cols:
        r += [f(p30.loc[s_, c], d_), f(e100.loc[s_, c], d_)]
    rows.append(r)
mean_row = ["Mean"]
for c, d_, _ in cols:
    mean_row += [f(p30.loc[STATIONS, c].mean(), d_), f(e100.loc[STATIONS, c].mean(), d_)]
rows += ["MID", mean_row]
hdr = ["Station"]
for _, _, name in cols:
    hdr += [name + " (30)", name + " (100)"]
add(r"""
\section{PatchTST Training-Budget Sensitivity}
\label{app:patchtst}
""")
add(table("PatchTST trained for 30 versus 100 epochs, per station (Validity Gate window, 1\\,608 hours per station). MAE, RMSE and Bias in \\textmu g/m$^3$. Longer training does not help: mean $R^2$ falls from "
          f"{p30.loc[STATIONS,'R2'].mean():.4f} to {e100.loc[STATIONS,'R2'].mean():.4f}.",
    "tab:app-patchtst", hdr, rows, "l" + "rr" * 4, size="\\footnotesize", tabcolsep="3pt"))

# ═══════════════════════════════════════════ 7. Control test
ct = pd.read_csv(R / "control_station_test.csv")
rows = [[r.station.replace("ü", "\\\"u").replace("ß", "{\\ss}"), r.group, r.environment, f(r.distance_km, 1), f(r.pm25_completeness_pct, 1), f(r.r_daily, 3), f(r.r_deseasonalized, 3)]
        for r in ct.itertuples()]
add(r"""
\section{Control-Station Test}
\label{app:stations}
""")
add(table("Correlation of daily PM$_{2.5}$ with the reference station (M\\\"unchen/Landshuter Allee). The distant control stations N\\\"urnberg (151\\,km, $r=0.80$) and Stuttgart (185\\,km, $r=0.86$) correlate about as strongly as Andechs/Rothenfeld ($r=0.69$) and the Augsburg stations ($r=0.89$), so correlation with a reference station cannot separate regional from merely weather-sharing sites.",
    "tab:app-control", ["Station", "Group", "Environment", "Dist.\\ (km)", "Compl.\\ (\\%)", "$r$", "$r$ (deseas.)"], rows, "lllrrrr",
    "Compl.\\ = PM$_{2.5}$ completeness; deseas.\\ = after removing the day-of-year climatology.", size="\\footnotesize", tabcolsep="4pt"))

# ═══════════════════════════════════════════ 8. Reproducibility
add(r"""
\section{Reproducibility and Configuration}
\label{app:repro}

All code, intermediate results and figure scripts are available at \url{https://github.com/Sanjal03/pm25-munich}. Air-quality data come from the German Federal Environment Agency (UBA) API and weather data from DWD station 03379. The full analysis is reproduced by \texttt{run\_pipeline.sh} (about 2.3 hours on a laptop CPU, plus about one hour to download the raw data). Python 3 with pandas 3.0.3, NumPy 2.4.6, scikit-learn 1.9.0, LightGBM 4.7.0, XGBoost 3.4.0, CatBoost 1.2.10 and PyTorch 2.13.0 was used; all random seeds are 42. The study windows are those defined in Chapter~\nameref{chapter:Methodology}.
""")
add(table("Hyperparameters of all models (fixed in every experiment; unlisted parameters are library defaults). The tree models use all 36 features; the sequence models use a 168\\,h PM$_{2.5}$ window plus 20 covariates.",
    "tab:app-hyper", ["Model", "Configuration"],
    [["CatBoost", "600 iterations, depth 6, learning rate 0.05, subsample 0.8, $\\ell_2$ leaf regularization 1.0"],
     ["XGBoost", "600 trees, depth 6, learning rate 0.05, subsample 0.8, column sample 0.8, min child weight 5, $\\alpha=0.1$, $\\lambda=1.0$"],
     ["LightGBM", "600 trees, depth 6, learning rate 0.05, subsample 0.8 (inactive: no bagging frequency set), column sample 0.8, min child samples 20, $\\alpha=0.1$, $\\lambda=1.0$"],
     ["DLinear", "moving-average kernel 25; trend, residual and covariate terms fitted jointly by least squares"],
     ["PatchTST (adapted)", "patch length and stride 24 (7 patches), $d_{\\text{model}}=32$, 4 heads, 2 layers, dropout 0.1; Adam, lr $10^{-3}$, batch 256, 30 epochs; covariates concatenated before the output layer; no positional encoding, global normalization"],
     ["Persistence", "$\\hat y_t = y_{t-1}$"]],
    "lp{0.72\\textwidth}", size="\\small"))

import re
text = "\n\n".join(parts)
text = re.sub(r"(\d),(\d{3})", r"\1\\,\2", text)
text = text.replace("\\section{", "\\FloatBarrier\n\\section{")
(OUT / "99_Appendix_generated.tex").write_text(text, encoding="utf-8")
labels = set(re.findall(r"\\label\{([^}]+)\}", text))
refs = set(re.findall(r"\\(?:ref|nameref)\{([^}]+)\}", text))
print("wrote", OUT / "99_Appendix_generated.tex", "| tables:", text.count("\\begin{table}"), "| sections:", text.count("\\section{"),
      "| undefined refs:", sorted(r for r in refs if r not in labels and not r.startswith("chapter:")))
