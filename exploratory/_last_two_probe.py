import importlib.util, time
from pathlib import Path
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "fetch_data", Path(__file__).resolve().parent.parent / "pipeline" / "01_fetch_data.py"
)
fetch_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_data)

targets = {"DEBY110": "Augsburg/Karlstraße", "DEUB044": "Zugspitze Schneefernerhaus"}
fetch_data.STATIONS_MAP.update({k: (v, "unknown", 0) for k, v in targets.items()})
fetch_data.STATIONS_MAP = {k: v for k, v in fetch_data.STATIONS_MAP.items() if k in targets}

id_map = fetch_data.resolve_ids(Path("munich_aq_final/_last_two_probe"))
rows = []
for deby, num_id in id_map.items():
    found = []
    for comp_id, pol in fetch_data.COMP_MAP.items():
        recs = fetch_data.fetch_one(num_id, deby, comp_id, "2023-06-01", "2023-06-07")
        if recs:
            found.append(pol)
        time.sleep(0.5)
    rows.append({"DEBY": deby, "Name": targets[deby], "Found": ", ".join(found) or "NONE",
                 "HasPM25": "PM2.5" in found})
    print(deby, targets[deby], "->", found)

Path("munich_aq_final/_last_two_probe").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv("munich_aq_final/_last_two_probe/station_validation.csv", index=False)
