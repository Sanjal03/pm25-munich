#!/usr/bin/env bash
# Reproduce every thesis result from the raw data, in order.
# Run from the project root with the venv active. 01_fetch_data.py is not
# included (it hits the UBA API for ~1h); run it once first if
# munich_aq_final/ is empty.
set -euo pipefail
PY=${PY:-python}

$PY pipeline/02_preprocess.py
$PY pipeline/03_feature_engineering.py
$PY pipeline/04_phase1_benchmark.py            # Phase 1: champion
$PY pipeline/05_phase2_rq1_ablation.py         # RQ1: champion
$PY pipeline/_rq1_remaining_models.py          # RQ1: other 4 models
$PY pipeline/06_phase3_rq2_counterfactual.py   # RQ2: top-3 models
$PY pipeline/_rq2_remaining_models.py          # RQ2: other 2 models
$PY pipeline/08_rq2_recursive_horizon_check.py # RQ2: 2024 rollout drift
$PY pipeline/10_rq2_drift_correction.py        # RQ2: drift-corrected gaps
$PY pipeline/07_make_figures.py
$PY pipeline/09_make_fig21_horizon.py

$PY experiments/patchtst_epoch_check.py
$PY experiments/leakage_sensitivity_check.py
$PY experiments/control_station_test.py
