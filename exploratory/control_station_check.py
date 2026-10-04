import importlib.util, time
from pathlib import Path
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "fetch_data", Path(__file__).resolve().parent.parent / "pipeline" / "01_fetch_data.py"
)
fetch_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_data)

targets = {"DEBW013": "Stuttgart-Bad Cannstatt"}
fetch_data.STATIONS_MAP = {k: (v, "urban-background", 0) for k, v in targets.items()}

out_dir = Path("munich_aq_final/_control_probe")
out_dir.mkdir(parents=True, exist_ok=True)
id_map = fetch_data.resolve_ids(out_dir)

for deby, num_id in id_map.items():
    found = []
    for comp_id, pol in fetch_data.COMP_MAP.items():
        recs = fetch_data.fetch_one(num_id, deby, comp_id, "2023-06-01", "2023-06-07")
        if recs:
            found.append(pol)
        time.sleep(0.5)
    print(deby, targets[deby], "id=", num_id, "->", found)
