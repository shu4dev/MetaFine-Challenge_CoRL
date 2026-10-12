"""Contrastive instruction guidance for pi0 ("steer away from the other instruction").

At every denoising step pi0 predicts a velocity v for the real instruction and v_neg for a
negative instruction made by swapping one word (e.g. red <-> blue). The step uses

    v = v_pos + w * (v_pos - v_neg)

so whatever the two instructions agree on is unchanged and only the part that depends on the
swapped word is amplified. w = 0 is plain pi0. Both rows share the same noise and stay
identical, so the policy still returns one action chunk. No retraining needed.
"""
from __future__ import annotations

import torch


def _word_swap_ids(preprocessor, a: str, b: str) -> dict[int, int]:
    """Token ids that differ between "... a ..." and "... b ..." (same sentence otherwise)."""
    tok = next(s.input_tokenizer for s in preprocessor.steps if getattr(s, "input_tokenizer", None) is not None)
    ia = tok(f"Toggle the {a} switch on the table")["input_ids"]
    ib = tok(f"Toggle the {b} switch on the table")["input_ids"]
    if len(ia) != len(ib):
        raise ValueError(f"'{a}' and '{b}' tokenize to different lengths")
    diff = [(x, y) for x, y in zip(ia, ib) if x != y]
    if len(diff) != 1:
        raise ValueError(f"expected one differing token, got {diff}")
    x, y = diff[0]
    return {x: y, y: x}


def enable_guidance(policy, preprocessor, w: float, swap: tuple[str, str] = ("red", "blue")) -> None:
    model = policy.model
    swap_ids = _word_swap_ids(preprocessor, *swap)
    sample_actions, denoise_step = model.sample_actions, model.denoise_step

    def guided_denoise_step(*args, **kwargs):
        v = denoise_step(*args, **kwargs)                 # (2, chunk, dim): row 0 real, row 1 swapped
        g = v[:1] + w * (v[:1] - v[1:])
        return g.expand_as(v)

    def guided_sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kw):
        if lang_tokens.shape[0] != 1:
            raise ValueError("guidance supports batch size 1")
        neg = lang_tokens.clone()
        for x, y in swap_ids.items():
            neg[lang_tokens == x] = y
        if torch.equal(neg, lang_tokens):
            raise ValueError("instruction contains neither swap word")
        if noise is None:
            noise = model.sample_noise((1, model.config.chunk_size, model.config.max_action_dim), state.device)
        model.denoise_step = guided_denoise_step
        try:
            out = sample_actions([torch.cat([i, i]) for i in images], [torch.cat([m, m]) for m in img_masks],
                                 torch.cat([lang_tokens, neg]), torch.cat([lang_masks, lang_masks]),
                                 torch.cat([state, state]), noise=noise.expand(2, -1, -1).contiguous(), **kw)
        finally:
            model.denoise_step = denoise_step
        return out[:1]

    model.sample_actions = guided_sample_actions
