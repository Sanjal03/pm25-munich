"""
Driver to run RQ1 ablation for every model 05_phase2_rq1_ablation.py
doesn't cover by default (it only auto-picks the Phase 1 champion). Reuses
run_rq1_experiment() directly so methodology is identical. Run after 05.
"""
import importlib.util
from pathlib import Path

from model_factory import ALL_MODELS

spec = importlib.util.spec_from_file_location(
    "phase2", Path(__file__).parent / "05_phase2_rq1_ablation.py"
)
phase2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase2)

champion = phase2.pick_rq1_model()
remaining = [m for m in ALL_MODELS if m != champion]

for model_type in remaining:
    for station in phase2.STATIONS:
        print(f"\n{'='*70}\n  RQ1 | Station: {station} | Model: {model_type}\n{'='*70}")
        phase2.run_rq1_experiment(station, model_type)

print(f"\n✓ Done — RQ1 ablation regenerated for {remaining}")
