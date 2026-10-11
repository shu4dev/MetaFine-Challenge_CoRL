"""TCP path of the LeRobot training demos (the delta-action replays pi0 imitates), by forward
kinematics of observation.state (qpos). Checked against the env-state path on the source demos."""
import importlib, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tools"))
import core.env  # noqa: F401
from mani_skill.utils import common
from utils.eval_common import load_task_config
from parity_check import make_env, TASK_MODULES

task = "toggle_switch_table"
cfg = load_task_config(str(REPO / "eval/configs/toggle_switch_table.yaml"))
env = make_env(task, importlib.import_module(TASK_MODULES[task]), cfg, "red")
env.reset(seed=0)
base = env.unwrapped
robot = base.agent.robot

def fk(q):
    robot.set_qpos(torch.as_tensor(q, dtype=torch.float32)[None])
    return common.to_numpy(base.agent.tcp.pose.raw_pose)[0]

# check: FK of the source demo qpos reproduces the env-state TCP
src = np.load(REPO / "gmm_test/demo_tcp_t3.npz")
q, t = src["red/traj_0|qpos"], src["red/traj_0|tcp_pose"]
err = max(np.abs(fk(q[i])[:3] - t[i, :3]).max() for i in range(0, len(q), 7))
print(f"FK check max position error {err:.2e} m")

p = pd.read_parquet(REPO / "demos/CoRL/toggle_switch_table/mixed/lerobot/data/chunk-000/file-000.parquet",
                    columns=["observation.state", "action", "episode_index"])
S, A, E = np.stack(p["observation.state"].values), np.stack(p.action.values), p.episode_index.values
tcp = np.stack([fk(s) for s in S]).astype(np.float32)
np.savez_compressed(REPO / "gmm_test/lerobot_demos_t3.npz", tcp_pose=tcp, qpos=S, action=A, episode_index=E)
print("saved", len(tcp))
