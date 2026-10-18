"""First camera frames of the 100 dev scenes under clean / strongest light / strongest camera shift.

    python gmm_test/color_mask/render_first_frames.py   -> gmm_test/color_mask/first_frames.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from eval.eval_toggle_switch import _make_env, _variant_extra  # noqa: E402
from utils.eval_common import build_base_env_kwargs, build_perception_profiles, load_task_config  # noqa: E402

cfg = load_task_config(str(REPO / "eval/configs/toggle_switch_table.yaml"))
seeds = json.load(open(REPO / "eval/seeds/dev/toggle_switch_table_100.json"))["seeds"]
profiles = build_perception_profiles(build_base_env_kwargs(cfg, extra=_variant_extra(cfg, "red")), cfg)
out = {}
for name in ["clean", "light_l3", "cam_l3"]:
    env = _make_env(profiles[name])
    B, H = [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        sd = obs["sensor_data"]
        B.append(np.asarray(sd["base_camera"]["rgb"])); H.append(np.asarray(sd["hand_camera"]["rgb"]))
    env.close()
    out[f"{name}_base"], out[f"{name}_hand"] = np.stack(B), np.stack(H)
    print(name, out[f"{name}_base"].shape, flush=True)
np.savez_compressed(REPO / "gmm_test/color_mask/first_frames.npz", seeds=np.array(seeds), **out)
