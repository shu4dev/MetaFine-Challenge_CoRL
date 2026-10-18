#!/usr/bin/env python3
"""lerobot-train with the T3 target mask (tools/target_mask.py) in pi0's third image slot.

    python gmm_test/color_mask/train_mask.py [lerobot-train args, incl. --dataset.root=...]

Before training, one mask per episode is made from that episode's first base-camera frame and its
instruction colour (same as eval). In training forward it is added to the batch by episode_index.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.pi0 import modeling_pi0 as M  # noqa: E402
from tools.target_mask import BASE, SLOT, add_offset, mask_image, target_offset, to_uint8  # noqa: E402

HINT = os.environ.get("TARGET_HINT", "view")  # view | offset | both: what the model gets (eval: --target-mask / --target-offset)

root = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--dataset.root="))
ds = LeRobotDataset(Path(root).name, root=root, video_backend="pyav")
start = ds.meta.episodes["dataset_from_index"]
MASKS, OFFS, missing = [], [], 0
for ep in range(ds.meta.total_episodes):
    x = ds[int(start[ep])]
    colour = "red" if " red " in x["task"] else "blue"
    m = mask_image(to_uint8(x["observation.images.base_camera"]), colour)
    missing += not m.any()
    MASKS.append(m)
    OFFS.append(target_offset(to_uint8(x["observation.images.base_camera"]), colour))
MASKS, OFFS = torch.from_numpy(np.stack(MASKS)), torch.from_numpy(np.stack(OFFS))
print(f"[train_mask] hint={HINT}: {len(MASKS)} episode masks, {missing} empty; "
      f"{int((OFFS == 0).all(1).sum())} empty offsets", flush=True)
del ds

_forward = M.PI0Policy.forward


def forward(self, batch, reduction="mean"):
    ep = batch["episode_index"].long().cpu()
    if HINT in ("view", "both"):
        batch = {**batch, SLOT: MASKS[ep].to(batch[BASE].device)}
    if HINT in ("offset", "both"):
        batch = add_offset(batch, OFFS[ep])
    return _forward(self, batch, reduction)


M.PI0Policy.forward = forward

if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import main

    main()
