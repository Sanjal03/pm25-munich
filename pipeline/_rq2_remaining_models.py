"""
Driver to run RQ2 for the models 06_phase3_rq2_counterfactual.py's
default (top-3 by Phase 1 R²) doesn't cover. Run after 06. Reuses
run_rq2_experiment() directly (not run_all(), which overwrites the summary
CSV wholesale) and merges the new rows into the existing summary instead of
replacing it.
"""
import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "phase3", Path(__file__).parent / "06_phase3_rq2_counterfactual.py"
)
phase3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase3)

new_rows = []
from model_factory import ALL_MODELS

top3 = phase3.pick_rq2_models()
for model_type in [m for m in ALL_MODELS if m not in top3]:
    for station in phase3.STATIONS:
        print(f"\n{'='*70}\n  RQ2 | Station: {station} | Model: {model_type}\n{'='*70}")
        try:
            new_rows.append(phase3.run_rq2_experiment(station, model_type))
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            new_rows.append({"station": station, "model": model_type,
                              "sanity_valid": False, "error": str(e)})

new_df = pd.DataFrame(new_rows)
summary_path = Path("results") / "phase3_rq2_summary.csv"
existing = pd.read_csv(summary_path)
combined = pd.concat([existing, new_df], ignore_index=True)
combined = combined.drop_duplicates(subset=["station", "model"], keep="last")
combined.to_csv(summary_path, index=False)

print(f"\n✓ Merged — {len(combined)} total runs — {summary_path}")
