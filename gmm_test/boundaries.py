"""Where is the jerk? Mean squared TCP jerk by step position within the replan cycle."""
import sys
import numpy as np, pandas as pd
run, n = sys.argv[1], int(sys.argv[2])
e = pd.read_csv(f"{run}/episodes.csv")
acc = np.zeros(n); cnt = np.zeros(n); total = []
for _, r in e[e.success].iterrows():
    z = np.load(f"{run}/{r.traj_file}"); p = z["tcp_pose"][:, :3].astype(float); dt = float(z["dt"])
    j = np.diff(p, n=3, axis=0) / dt**3          # jerk between steps t..t+3
    js = (j**2).sum(1)
    for t, v in enumerate(js):
        acc[(t + 2) % n] += v; cnt[(t + 2) % n] += 1   # centre of the 4-point stencil
m = acc / cnt
near = [(k % n) for k in (-1, 0, 1, 2)]
share = m[near].sum() / m.sum()
print(f"{run} replan {n}: steps within ±2 of a replan hold {share:.0%} of jerk (they are {len(near)/n:.0%} of steps)")
print("per-position mean sq jerk (rel. to median):", np.round(m / np.median(m), 1).tolist())
