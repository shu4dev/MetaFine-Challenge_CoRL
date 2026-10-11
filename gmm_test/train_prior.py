#!/usr/bin/env python3
"""lerobot-train with the motion-prior loss added (same arguments as lerobot-train).

    PRIOR_LAMBDA=1e-3 python gmm_test/train_prior.py --policy.path=... [lerobot-train args]

loss = flow-matching loss + PRIOR_LAMBDA * aux, where aux is the (1 - t)-weighted mean of
-log p(window) over all 10-step windows of the one-step denoised action chunk
a_hat = x_t - t * v_pred (unnormalized, 7 arm joints, windows touching padding masked).
aux is computed and logged even when PRIOR_LAMBDA=0, so the control run logs it too.
"""
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prior import W, MotionPrior  # noqa: E402

from lerobot.policies.pi0 import modeling_pi0 as M  # noqa: E402

LAMBDA = float(os.environ.get("PRIOR_LAMBDA", "0"))
PRIOR = MotionPrior.load(os.environ.get("PRIOR_PATH", HERE / "prior_planner_ctxfree.npz"))
STATS = json.load(open(os.environ["ACTION_STATS"]))["action"]
AUX_LOG = os.environ.get("AUX_LOG")
_mean = torch.tensor(STATS["mean"][:7])
_std = torch.tensor(STATS["std"][:7])
_stash: dict = {}
_step = [0]

_sample_noise, _sample_time, _apply_checkpoint = (
    M.PI0Pytorch.sample_noise, M.PI0Pytorch.sample_time, M.PI0Pytorch._apply_checkpoint)


def sample_noise(self, shape, device):
    _stash["noise"] = _sample_noise(self, shape, device)
    return _stash["noise"]


def sample_time(self, bsize, device):
    _stash["time"] = _sample_time(self, bsize, device)
    return _stash["time"]


def apply_checkpoint(self, func, *args, **kwargs):
    out = _apply_checkpoint(self, func, *args, **kwargs)
    if getattr(func, "__name__", "") == "action_out_proj_func":
        _stash["v"] = out
    return out


M.PI0Pytorch.sample_noise = sample_noise
M.PI0Pytorch.sample_time = sample_time
M.PI0Pytorch._apply_checkpoint = apply_checkpoint
_forward = M.PI0Policy.forward


def forward(self, batch, reduction="mean"):
    _stash.clear()
    loss, out = _forward(self, batch, reduction)
    if not self.training or "v" not in _stash:
        return loss, out
    v = _stash["v"].float()
    t = _stash["time"][:, None, None]
    actions = M.pad_vector(batch[M.ACTION], self.config.max_action_dim)
    x_t = t * _stash["noise"] + (1 - t) * actions
    a_hat = (x_t - t * v)[:, :, :7] * _std.to(v.device) + _mean.to(v.device)   # (B, 50, 7)
    wins = a_hat.unfold(1, W, 1).permute(0, 1, 3, 2)                            # (B, N, W, 7)
    B, N = wins.shape[:2]
    valid = torch.ones(B, N, dtype=torch.bool, device=v.device)
    if "action_is_pad" in batch:
        valid = ~batch["action_is_pad"].unfold(1, W, 1).any(-1)
    PRIOR.to(v.device)
    lp = PRIOR.log_prob(wins.reshape(-1, W, 7)).reshape(B, N).float()
    w = (1 - t[:, 0, 0])[:, None] * valid
    aux = -(lp * w).sum() / w.sum().clamp_min(1e-6)
    if LAMBDA > 0:
        loss = loss + LAMBDA * aux
    out["aux_nll"] = aux.item()
    _step[0] += 1
    if AUX_LOG:
        with open(AUX_LOG, "a") as f:
            f.write(f"{_step[0]},{out['loss']:.6f},{aux.item():.4f},{valid.float().mean().item():.3f}\n")
    return loss, out


M.PI0Policy.forward = forward

if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import main

    main()
