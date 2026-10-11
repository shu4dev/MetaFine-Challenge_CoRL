"""Replay each T3 source demo's recorded env states and save the TCP pose per frame,
in the same layout as the eval rollouts (tcp_pose (T, 7), qpos (T, 9), action (T, 8), dt)."""
import importlib, sys
from pathlib import Path
import h5py, numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tools"))
import core.env  # noqa: F401
from mani_skill.utils import common
from mani_skill.trajectory import utils as trajectory_utils
from utils.eval_common import load_task_config
from parity_check import make_env, TASK_MODULES


def h5_to_dict(g):
    return {k: h5_to_dict(v) if isinstance(v, h5py.Group) else v[:] for k, v in g.items()}

task = "toggle_switch_table"
mod = importlib.import_module(TASK_MODULES[task])
cfg = load_task_config(str(REPO / "eval/configs/toggle_switch_table.yaml"))
out = {}
for variant in ("red", "blue"):
    env = make_env(task, mod, cfg, variant)
    env.reset(seed=0)
    base = env.unwrapped
    f = h5py.File(REPO / f"demos/CoRL/{task}/{variant}/{task}_{variant}.h5", "r")
    for key in f:
        states = trajectory_utils.dict_to_list_of_dicts(h5_to_dict(f[key]["env_states"]))
        tcp, qpos = [], []
        for s in states:
            base.set_state_dict(s)
            tcp.append(common.to_numpy(base.agent.tcp.pose.raw_pose)[0])
            qpos.append(common.to_numpy(base.agent.robot.get_qpos())[0])
        out[f"{variant}/{key}"] = dict(tcp_pose=np.array(tcp, np.float32), qpos=np.array(qpos, np.float32),
                                       action=f[key]["actions"][:].astype(np.float32))
    env.close()
    print(variant, len(f))
np.savez_compressed(REPO / "gmm_test/demo_tcp_t3.npz",
                    **{f"{k}|{n}": v for k, d in out.items() for n, v in d.items()})
print("saved", len(out))
