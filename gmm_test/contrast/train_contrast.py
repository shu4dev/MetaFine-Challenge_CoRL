#!/usr/bin/env python3
"""lerobot-train with a "wrong colour word" contrastive loss (same arguments as lerobot-train).

    CONTRAST_MARGIN=0.05 CONTRAST_LAMBDA=1 python gmm_test/contrast/train_contrast.py [lerobot-train args]

For each sample, the normal flow-matching loss L_right uses the real instruction. A second pass
uses the same images, state, actions, noise and time but with red <-> blue swapped in the
instruction (L_wrong). The extra loss

    CONTRAST_LAMBDA * mean_b relu(CONTRAST_MARGIN - (L_wrong_b - L_right_b.detach()))

pushes the prediction for the wrong word away from the demo's actions, so the actions must depend
on the colour word. Its backward runs inside forward (separate graph) to keep peak memory low.
Before training, the LoRA 20k model gives L_wrong - L_right ~ 0.0001 (median): it ignores the word.
"""
import os

import torch

from lerobot.policies.pi0 import modeling_pi0 as M
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

RED, BLUE = 3118, 3868          # PaliGemma token ids of " red" / " blue" (checked with the tokenizer)
MARGIN = float(os.environ.get("CONTRAST_MARGIN", "0.05"))
LAMBDA = float(os.environ.get("CONTRAST_LAMBDA", "1"))
LOG = os.environ.get("CONTRAST_LOG")
_stash: dict = {}
_step = [0]

_sample_noise, _sample_time = M.PI0Pytorch.sample_noise, M.PI0Pytorch.sample_time


def sample_noise(self, shape, device):
    _stash["noise"] = _sample_noise(self, shape, device)
    return _stash["noise"]


def sample_time(self, bsize, device):
    _stash["time"] = _sample_time(self, bsize, device)
    return _stash["time"]


M.PI0Pytorch.sample_noise = sample_noise
M.PI0Pytorch.sample_time = sample_time
_forward = M.PI0Policy.forward


def forward(self, batch, reduction="mean"):
    _stash.clear()
    if not self.training:
        return _forward(self, batch, reduction)
    per_sample, out = _forward(self, batch, "none")
    loss = per_sample.mean() if reduction == "mean" else per_sample
    l_right = per_sample.detach()
    tok = batch[OBS_LANGUAGE_TOKENS]
    has = (tok == RED).any(1) & ~(tok == BLUE).any(1) | (tok == BLUE).any(1) & ~(tok == RED).any(1)
    assert has.all(), "every instruction must contain exactly one of red / blue"
    swapped = tok.clone()
    swapped[tok == RED] = BLUE
    swapped[tok == BLUE] = RED
    images, img_masks = self._preprocess_images(batch)
    state, actions = self.prepare_state(batch), self.prepare_action(batch)
    noise, time = _stash["noise"], _stash["time"]
    d = self.config.output_features["action"].shape[0]
    l_wrong = self.model.forward(images, img_masks, swapped, batch[OBS_LANGUAGE_ATTENTION_MASK], state, actions,
                                 noise=noise, time=time)[:, :, :d].mean((1, 2))
    term = torch.relu(MARGIN - (l_wrong - l_right)).mean()
    if LAMBDA > 0 and term.requires_grad:
        (LAMBDA * term).backward()       # gradients add to the main loss's in the optimizer step
    out["contrast"] = term.item()
    _step[0] += 1
    if LOG:
        with open(LOG, "a") as f:
            f.write(f"{_step[0]},{out['loss']:.6f},{l_right.mean().item():.6f},{l_wrong.mean().item():.6f},{term.item():.6f}\n")
    return loss, out


M.PI0Policy.forward = forward

if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import main

    main()
