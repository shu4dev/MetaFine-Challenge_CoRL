"""Is the baseline jerkier than the expert demos? Same LDLJ/SPARC code as the reports."""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from make_report import log_dimensionless_jerk, sparc

def metrics(pos, dt):
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    return log_dimensionless_jerk(pos, dt), sparc(speed, dt), (len(pos)) * dt

d = np.load(REPO / "gmm_test/demo_tcp_t3.npz")
demos = {}
for k in d.files:
    ep, name = k.split("|"); demos.setdefault(ep, {})[name] = d[k]
dt = 0.05
rows = [("demo", ep, *metrics(v["tcp_pose"][1:, :3], dt)) for ep, v in demos.items()]
for run in sys.argv[1:]:
    e = pd.read_csv(f"{run}/episodes.csv")
    for _, r in e[e.success & (e.condition == "clean")].iterrows():
        z = np.load(f"{run}/{r.traj_file}")
        rows.append((Path(run).name, r.episode_id, *metrics(z["tcp_pose"][:, :3], float(z["dt"]))))
t = pd.DataFrame(rows, columns=["source", "ep", "ldlj", "sparc", "time_s"])
print(t.groupby("source")[["ldlj", "sparc", "time_s"]].median().round(2).to_string())
print(t.groupby("source").size().to_string())
v = demos["red/traj_0"]
print("demo tcp start", v["tcp_pose"][0, :3], "moves", np.ptp(v["tcp_pose"][:, :3], 0))
z = np.load(glob.glob(f"{sys.argv[-1]}/episodes/*.npz")[0]); print("rollout tcp start", z["tcp_pose"][0, :3])
