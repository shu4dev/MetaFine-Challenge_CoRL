"""Where does the gripper go? For each eval episode, the TCP point nearest the switch vs the red and blue
slider grasp points (world frame, from the env at the same seed). Tells "chose the wrong slider" apart
from "aimed at the right one but not precisely enough" (sliders are ~1 cm apart).

    python gmm_test/color_mask/aim.py            # builds slider_pos.npz once (CPU), then prints per run
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
POS = REPO / "gmm_test/color_mask/slider_pos.npz"


def slider_positions():
    import gymnasium as gym
    import core.env  # noqa: F401
    from utils.grasp_compute import get_grasp_pose_from_config
    seeds = json.load(open(REPO / "eval/seeds/dev/toggle_switch_table_100.json"))["seeds"]
    out = {}
    for c in ("red", "blue"):
        env = gym.make("toggle_switch_table", obs_mode="none", control_mode="pd_joint_pos", sim_backend="physx_cpu",
                       object_name="100920", part_name="button", target_switch=c)
        P = []
        for s in seeds:
            env.reset(seed=int(s))
            b = env.unwrapped
            P.append(np.array(get_grasp_pose_from_config(b, "button", grasp_id=int(b.target_switch_id)).p))
        out[c] = np.array(P)
        env.close()
    np.savez(POS, seeds=np.array(seeds), **out)


def aim(run):
    z = np.load(POS)
    idx = {int(s): i for i, s in enumerate(z["seeds"])}
    e = pd.read_csv(REPO / "runs" / run / "episodes.csv")
    rows = []
    for _, r in e.iterrows():
        i = idx[int(r.seed)]
        red, blue = z["red"][i], z["blue"][i]
        tgt, oth = (red, blue) if r.variant == "red" else (blue, red)
        p = np.load(REPO / "runs" / run / r.traj_file)["tcp_pose"][:, :3]
        mid = (red + blue) / 2
        u = (oth - tgt) / np.linalg.norm(oth - tgt)
        t = np.clip((p - tgt) @ u, 0, np.linalg.norm(oth - tgt))
        k = int(np.argmin(np.linalg.norm(p - tgt - t[:, None] * u, axis=1)))  # nearest the slider segment: no pull to the middle
        rows.append(dict(dist_mid=np.linalg.norm(p[k] - mid), along=float((p[k] - tgt) @ u),
                         gap=float(np.linalg.norm(oth - tgt)), cond=r.condition, res=r.toggled))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if not POS.exists():
        slider_positions()
    runs = sys.argv[1:] or ["2026-10-11_041055", "2026-10-17_234514", "2026-10-18_061659",
                            "2026-10-18_101112", "2026-10-18_163311"]
    for run in runs:
        a = aim(run)
        near = a[a.dist_mid < 0.03]  # reached the switch
        g = near.gap.median()
        print(f"{run}: reached {len(near)}/200, gap {100*g:.2f} cm; position along target->other axis "
              f"(0 = target, {100*g:.1f} = other): median {100*near.along.median():.2f} cm, "
              f"closer to target {int((near.along < g/2).sum())}, closer to other {int((near.along >= g/2).sum())}")
        h, _ = np.histogram(near.along / g, bins=[-9, -0.25, 0.25, 0.75, 1.25, 9])
        print("   share of gap: <-.25 | target +-.25 | middle | other +-.25 | >1.25 :", h.tolist())
