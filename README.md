# PM2.5 Munich — Bachelor Thesis Pipeline

Hourly PM2.5 forecasting in the Munich region (2019–2024). Bachelor thesis, TUM Management & Technology, supervised by Zhenyu Wang.

The thesis answers two research questions through a Champion/Challenger model comparison:

- **RQ1**: Which feature groups (PM2.5 history, co-pollutants, meteorology, temporal encodings) matter most for PM2.5 prediction?
- **RQ2**: Can a validity-gated recursive counterfactual model quantify a COVID-19 lockdown effect on PM2.5?

## Data

- **Air quality**: hourly PM2.5, PM10, NO2 and O3 from the UBA (Umweltbundesamt) `air_data/v2/measures` API, in µg/m³. Fetch logic is in `pipeline/01_fetch_data.py`.
- **Meteorology**: temperature, humidity, wind and pressure from DWD station 03379 (`Temperature.txt`, `Wind.txt`, `Pressure.txt`, not tracked in git).
- **Stations (7)**: München/Stachus, München/Landshuter Allee (reference station), München/Lothstraße, München/Johanneskirchen, Augsburg/Bourges-Platz, Augsburg/LfU, Andechs/Rothenfeld. München/Allach is also fetched. It has no PM2.5 sensor, but its NO2/O3 are used as spatial neighbours during gap imputation.
- **Period**: 2019–2024, split chronologically: train 2019–2023, test 2024, no shuffling. Outlier thresholds are learned on 2019–2023 only.
- **Windows** (defined once in `model_factory.py`): Validity Gate 15 Jan 00:00 – 21 Mar 23:00 2020 (1,608 h); lockdown 22 Mar 00:00 – 4 May 23:00 2020 (1,056 h). RQ2 models never see either window in training.

## Task definition

Every model estimates PM2.5 at hour *t* from the station's PM2.5 up to *t*−1 and the covariates observed **at** *t*: PM10, NO2, O3, `pm25_background` (mean PM2.5 of the other 6 stations), DWD weather and calendar features. This is one-hour-ahead estimation with concurrently observed network covariates, not a pure forecast from information available before *t*.

A 9-station extension (adding Burghausen/Marktler Straße, 93 km E, and Oberaudorf/Inntal-Autobahn, 71 km S) lives on the [`extension-9-stations`](../../tree/extension-9-stations) branch as future work. It is not part of the thesis results.

Raw and intermediate data (`munich_aq_final/`, `Munich_AQ_clean.csv`, `features/`, the DWD `.txt` files) are not tracked in git because they total more than 400 MB. `01_fetch_data.py` fetches them from public sources; the later pipeline steps derive them.

## Project structure

```
pipeline/                        the thesis pipeline, run in this order
  01_fetch_data.py                 raw AQ data from the UBA API -> munich_aq_final/
  02_preprocess.py                 merge DWD met, impute gaps, cap outliers -> Munich_AQ_clean.csv
  03_feature_engineering.py        36 features per station -> features/{station}_features.csv
  model_factory.py                 shared models + train_and_eval(); imported, not run
  04_phase1_benchmark.py           Phase 1: 5 models + Persistence x 7 stations, champion selection
  05_phase2_rq1_ablation.py        RQ1: leave-one-group-out ablation (champion)
  _rq1_remaining_models.py         RQ1: same ablation for the other 4 models
  06_phase3_rq2_counterfactual.py  RQ2: validity gate + recursive lockdown counterfactual (top 3)
  _rq2_remaining_models.py         RQ2: same for the other 2 models
  08_rq2_recursive_horizon_check.py  RQ2 robustness: rollout drift over 3 matched 2024 windows
  10_rq2_drift_correction.py       RQ2 robustness: lockdown gaps corrected for that drift
  07_make_figures.py               study area map + RQ1/RQ2 figures -> thesis_figs/fig13-20
  09_make_fig21_horizon.py         horizon error-accumulation figure -> thesis_figs/fig21

experiments/                     robustness checks cited in the thesis
  patchtst_epoch_check.py          PatchTST 30 vs 100 epochs (undertraining check)
  leakage_sensitivity_check.py     outlier caps: train-only vs full-data thresholds
  control_station_test.py          Nürnberg/Stuttgart control test (Appendix)

exploratory/                     one-off data investigations (station discovery, candidate
                                 validation, DST, outliers, missingness, EDA/Methods figures)
archive/                         deprecated, kept for history, do not run

run_pipeline.sh                  runs 02 -> 10 and all experiments in order
```

Run every script **from the project root**, for example `python pipeline/04_phase1_benchmark.py`. The scripts use relative paths such as `results/`.

## Setup and reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline/01_fetch_data.py   # once; about 1 h of API calls
./run_pipeline.sh                  # everything else; about 2.5 h on a laptop CPU
```

## Models

- **Tabular** (the full 36-feature table): LightGBM, XGBoost, CatBoost
- **Sequence** (a raw 168-hour PM2.5 lookback plus exogenous covariates): DLinear, and PatchTST-lite. PatchTST-lite is a simplified PatchTST-style patch Transformer after Nie et al. (2023). Unlike the original it takes exogenous covariates (concatenated before the output head), has no positional encoding, and uses global rather than per-instance (RevIN) normalisation.
- **Baseline**: Persistence, y(t) = y(t−1)

All models share one entry point, `train_and_eval()` in `model_factory.py`, so the same preprocessing and splits apply to every model.

## Results (7 stations, reproduced with `run_pipeline.sh`, 30 Sep 2026)

**Phase 1: Champion.** Mean R² on the Validity Gate window (1,608 h, excluded from training), across 7 stations:

| Rank | Model | Mean R² | Mean MAE (µg/m³) |
|---|---|---|---|
| 1 | **CatBoost** (Champion) | 0.9540 | 0.968 |
| 2 | XGBoost | 0.9535 | 0.974 |
| 3 | LightGBM | 0.9530 | 0.977 |
| 4 | DLinear | 0.9447 | 1.067 |
| 5 | PatchTST | 0.9435 | 1.104 |
| 6 | Persistence | 0.9125 | 1.218 |

The three tree ensembles are separated by 0.0010 R². Architecture class, not the individual model, decides the ranking.

**PatchTST 30 vs 100 epochs** (`experiments/patchtst_epoch_check.py`): mean R² drops from 0.9435 to 0.9371, so undertraining does not explain PatchTST's gap to the tree models.

**RQ1: feature-group ablation.** Mean ΔMAE% across 7 stations when a group is removed (positive = the group helps):

| Feature group removed | CatBoost | XGBoost | LightGBM | DLinear | PatchTST |
|---|---|---|---|---|---|
| PM2.5 lags | +41.5% | +40.8% | +41.5% | +70.7% | +62.0% |
| Co-pollutants | +6.2% | +7.0% | +6.5% | +6.8% | +3.4% |
| Meteorology | +0.9% | +1.8% | +1.1% | +0.3% | −1.6% |
| Temporal | −1.3% | −1.2% | −1.5% | −0.2% | −2.0% |
| PM2.5 rolling stats | −1.0% | −0.9% | −0.9% | n/a | n/a |

Sequence models have no separate rolling-statistic features; their 168-hour lookback already contains that information, so the row does not apply to them.

**RQ2: lockdown counterfactual** (1,056 h, recursive rollout, 5 models × 7 stations = 35 runs):
- All 35 runs pass the validity gate (R² ≥ 0.80 and |bias| < 1.5 µg/m³; the lowest R² is 0.911).
- **29/35 (83%)** show observed PM2.5 below the counterfactual. Mean gap by model ranges from +0.08 µg/m³ (XGBoost) to +0.93 µg/m³ (PatchTST); the Champion is at +0.56.
- The strongest station is Augsburg/LfU (mean +10.2%, positive in all 5 models). The least consistent is Johanneskirchen (positive in 2 of 5 models; CatBoost gap 0.0%).
- PatchTST's gaps are the least stable across reruns (e.g. Johanneskirchen −3.2% → +13.0% after the threshold fix), so conclusions rest on agreement across models, not on any single run.

**RQ2 robustness: drift correction** (`08` + `10`). Running the same models and the same rollout over three 1,056-hour windows in 2024, where nothing unusual happened, gives a cumulative MAE of 2.10 µg/m³ (day 0: 1.48, final day: 2.15). That is larger than the mean |lockdown gap| of 0.68 µg/m³. Subtracting each run's own 2024 drift:

| Station | Positive (raw) | Positive (drift-corrected) |
|---|---|---|
| Augsburg/LfU | 5 | 5 |
| Andechs/Rothenfeld | 4 | 5 |
| Augsburg/Bourges-Platz | 5 | 2 |
| München/Johanneskirchen | 2 | 1 |
| München/Landshuter Allee | 5 | 0 |
| München/Lothstraße | 4 | 0 |
| München/Stachus | 4 | 0 |
| **Total** | **29/35 (83%)** | **13/35 (37%)** |

Only Augsburg/LfU and Andechs/Rothenfeld show an effect that survives correction for the model's own rollout drift.

**Outlier thresholds** (`experiments/leakage_sensitivity_check.py`). The pipeline learns both thresholds on 2019–2023 only. Earlier versions used 2019–2024; the difference per PM2.5 threshold was:
- 99.5th-percentile cap (`02`), on raw measurements: at most 2.4% difference; 4 of 7 stations identical.
- 99.5th-percentile cap (`02`), on the imputed series as the code applies it: at most 2.9%; 5 of 7 identical.
- Q3 + 3·IQR target fence (`03`): at most 10.0% (Augsburg/Bourges-Platz 36 vs 40 µg/m³); 4 of 7 identical.

**Control stations, Appendix** (`experiments/control_station_test.py`). Two stations outside the study region are compared with the reference station, München/Landshuter Allee. Nürnberg/Muggenhof (151 km, BY) correlates at r = 0.796 and Stuttgart-Bad Cannstatt (185 km, BW) at r = 0.860, on daily-mean PM2.5 over 2019–2024. The study stations range from 0.692 (Andechs/Rothenfeld) to 0.950. Removing the seasonal cycle barely changes these values (0.794 and 0.861). Correlation with the reference therefore cannot, on its own, separate a regionally relevant station from a distant one. The controls are not modelled, because Munich DWD weather does not apply there.

## Figures

- `07_make_figures.py` → `thesis_figs/fig13`–`fig20`: study area map, RQ1 bar chart and heatmap, RQ2 bar chart, heatmap and summary, and the CatBoost time series for Augsburg/LfU and Johanneskirchen (the strongest and weakest effects).
- `09_make_fig21_horizon.py` → `fig21`: rollout error accumulation.
- `experiments/control_station_test.py` → `figC1` (correlation vs distance) and `figC2` (monthly PM2.5, reference vs controls).
- `exploratory/*_figure.py` and `exploratory/eda_figures.py` → Data and Methods figures (`fig_D_*`, `fig_M*`, `figM*`, `figA*`).

## Known limitations

- **Concurrent covariates** (see Task definition). In the RQ2 rollout, co-pollutants and `pm25_background` stay at their observed lockdown values. Those were themselves lowered by the lockdown, so the counterfactual is partly anchored to the observed outcome and the reported gaps are conservative (lower bounds). Lagging all covariates would make this a true forecast; that is left for future work.
- **PatchTST-lite** is a simplified variant (see Models), not a reproduction of the published architecture.
- **Sequence-model gaps.** DLinear and PatchTST return NaN when a real target gap longer than 24 h falls inside their 168-hour lookback. This does not happen in the 2020 lockdown window. It does happen in three 2024 horizon windows, where `10_rq2_drift_correction.py` averages predictions and actuals over paired hours only.
- **Not in this repo.** No runnable code exists for the ensemble, hyperparameter-tuning or imputation-sensitivity experiments mentioned in early drafts. They are not part of the thesis.
