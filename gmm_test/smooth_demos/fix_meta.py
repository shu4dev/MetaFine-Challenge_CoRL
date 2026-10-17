"""Make a to_lerobot.py dataset match the original's schema (feature names/order, string task column,
robot_type) so aggregate/training treat it like demos/CoRL/toggle_switch_table/mixed/lerobot.

    python gmm_test/smooth_demos/fix_meta.py <dataset root>
"""
import glob
import json
import sys

import pandas as pd

O = "demos/CoRL/toggle_switch_table/mixed/lerobot"
R = sys.argv[1]
o = json.load(open(f"{O}/meta/info.json"))
od = pd.read_parquet(f"{O}/data/chunk-000/file-000.parquet")
n = json.load(open(f"{R}/meta/info.json"))
for k in ("action", "observation.state", "task"):
    n["features"][k] = o["features"][k]
n["features"] = {k: n["features"][k] for k in o["features"]}
n["robot_type"] = o["robot_type"]
json.dump(n, open(f"{R}/meta/info.json", "w"), indent=4)
tasks = pd.read_parquet(f"{R}/meta/tasks.parquet")
inv = {v: k for k, v in tasks["task_index"].items()}
for f in glob.glob(f"{R}/data/*/*.parquet"):
    d = pd.read_parquet(f)
    d["task"] = d["task_index"].map(inv)
    assert d["task"].notna().all()
    d[list(od.columns)].to_parquet(f, index=False)
print(R, "fixed;", d["task"].value_counts().to_dict())
