"""Target view for T3: a zoomed view of the switch with a green dot on the instructed slider,
fed to pi0's unused third image slot.

The switch is a red box with a blue and a dull-red slider, each ~3 px wide in pi0's 224 px view.
From the episode's FIRST base-camera frame (camera and switch don't move; later the arm covers it):
  box  = largest bright-red blob (hue red, saturation > 200)
  blue = blue pixels inside the box outline
  red  = dull-red pixels (saturation 50-185) inside the box outline, the blob nearest the blue one
The view is a CROP x CROP px crop of that frame centred on the box, with a green Gaussian dot
(sigma 2.5 px before zoom) on the instructed slider, resized to 224x224 (~4.7x zoom, so the two sliders
end up ~20-50 px apart instead of 2-5). Given as observation.images.right_wrist_0_rgb, kept for the
whole episode. A plain full-frame spot was rejected: red and blue spots would be only 2-5 px apart.
Checked on the 100 dev scenes: clean, strongest light and strongest camera shift all 100/100.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch

RED, BLUE = 3118, 3868          # PaliGemma token ids of " red" / " blue"
SLOT = "observation.images.right_wrist_0_rgb"
BASE = "observation.images.base_0_rgb"
STATE = "observation.state"
SIGMA = 2.5
CROP = 48
OUT = 224


def find_target(rgb: np.ndarray, colour: str):
    """((x, y) of the instructed slider, (x, y) of the box centre) in a uint8 HxWx3 image, or None."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    red = (h < 10) | (h > 165)
    n, lab, st, _ = cv2.connectedComponentsWithStats((red & (s > 200) & (v > 60)).astype(np.uint8))
    if n < 2:
        return None
    box = 1 + int(np.argmax(st[1:, 4]))
    centre = tuple(float(c) for c in np.argwhere(lab == box)[:, ::-1].mean(0))
    face = np.zeros(h.shape, np.uint8)
    cv2.fillConvexPoly(face, cv2.convexHull(np.argwhere(lab == box)[:, ::-1]), 1)
    blue = (h > 90) & (h < 130) & (s > 60) & (v > 50) & (face > 0)
    if blue.sum() < 3:
        return None
    by, bx = np.argwhere(blue).mean(0)
    if colour == "blue":
        return (float(bx), float(by)), centre
    pink = (red & (s > 50) & (s < 185) & (v > 110) & (face > 0)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(pink)
    if n < 2:
        return None
    near = [i for i in range(1, n) if np.hypot(cen[i, 0] - bx, cen[i, 1] - by) < 20] or list(range(1, n))
    i = max(near, key=lambda i: st[i, 4])
    return (float(cen[i, 0]), float(cen[i, 1])), centre


def mask_image(rgb: np.ndarray, colour: str) -> np.ndarray:
    """float32 (3, 224, 224) in [0, 1]: zoomed switch with a green dot on the target; zeros if not found."""
    found = find_target(rgb, colour)
    if found is None:
        return np.zeros((3, OUT, OUT), np.float32)
    (tx, ty), (cx, cy) = found
    H, W = rgb.shape[:2]
    x0 = int(np.clip(round(cx) - CROP // 2, 0, W - CROP)); y0 = int(np.clip(round(cy) - CROP // 2, 0, H - CROP))
    crop = rgb[y0:y0 + CROP, x0:x0 + CROP].astype(np.float32) / 255
    yy, xx = np.mgrid[:CROP, :CROP]
    a = np.exp(-((xx + x0 - tx) ** 2 + (yy + y0 - ty) ** 2) / (2 * SIGMA**2))[..., None]
    crop = crop * (1 - a) + np.array([0, 1, 0], np.float32) * a
    return cv2.resize(crop, (OUT, OUT), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1).copy()

def target_offset(rgb: np.ndarray, colour: str) -> np.ndarray:
    """float32 (2,): (instructed - other slider) in px / 10 (~0.4-1.1 at 512 px): which way to shift the aim.
    Appended to the state, so it reaches the action expert directly. Zeros if either slider isn't found."""
    a, b = find_target(rgb, colour), find_target(rgb, "blue" if colour == "red" else "red")
    if a is None or b is None:
        return np.zeros(2, np.float32)
    return (np.subtract(a[0], b[0]) / 10).astype(np.float32)


def add_offset(batch: dict, off: torch.Tensor) -> dict:
    """Append the offset to the (already normalised) state; pi0 pads the state to 32 dims anyway."""
    s = batch[STATE]
    return {**batch, STATE: torch.cat([s, off.to(s.device, s.dtype)], -1)}


def colour_of(tokens: torch.Tensor) -> str:
    """'red' / 'blue' from one row of language tokens."""
    r, b = bool((tokens == RED).any()), bool((tokens == BLUE).any())
    if r == b:
        raise ValueError("instruction must contain exactly one of red / blue")
    return "red" if r else "blue"


def to_uint8(img: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0, 1] -> HxWx3 uint8."""
    return (img.permute(1, 2, 0).float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


def enable_target_mask(policy, view: bool = True, offset: bool = False) -> None:
    """Eval: compute the view / offset on the first chunk of each episode, reuse until reset()."""
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

    predict, reset = policy.predict_action_chunk, policy.reset
    state = {"mask": None, "off": None}

    def reset_(*a, **kw):
        state["mask"] = state["off"] = None
        return reset(*a, **kw)

    def predict_(batch, **kw):
        if state["mask"] is None:
            masks = [mask_image(to_uint8(img), colour_of(tok))
                     for img, tok in zip(batch[BASE], batch[OBS_LANGUAGE_TOKENS])]
            if not any(m.any() for m in masks):
                print("[target_mask] WARNING: slider not found; mask is empty", flush=True)
            state["mask"] = torch.from_numpy(np.stack(masks))
            state["off"] = torch.from_numpy(np.stack([target_offset(to_uint8(img), colour_of(tok))
                                                      for img, tok in zip(batch[BASE], batch[OBS_LANGUAGE_TOKENS])]))
            if os.environ.get("TARGET_MASK_DEBUG"):  # save what the policy sees, for checking
                from PIL import Image
                Image.fromarray((masks[0].transpose(1, 2, 0) * 255).astype(np.uint8)).save(
                    f"{os.environ['TARGET_MASK_DEBUG']}/view_{colour_of(batch[OBS_LANGUAGE_TOKENS][0])}_{id(masks)}.png")
        if view:
            batch = {**batch, SLOT: state["mask"].to(batch[BASE].device)}
        if offset:
            batch = add_offset(batch, state["off"])
        return predict(batch, **kw)

    policy.reset, policy.predict_action_chunk = reset_, predict_


if __name__ == "__main__":  # self-check on the rendered dev frames
    import sys
    z = np.load(sys.argv[1] if len(sys.argv) > 1 else "gmm_test/color_mask/first_frames.npz")
    for k in ("clean", "light_l3", "cam_l3"):
        for c in ("red", "blue"):
            pts = [find_target(im, c) for im in z[f"{k}_base"]]
            assert all(p is not None for p in pts), (k, c)
        d = [np.hypot(*np.subtract(find_target(im, "red")[0], find_target(im, "blue")[0])) for im in z[f"{k}_base"]]
        print(f"{k}: found 100/100 both colours; red-blue distance {np.min(d):.1f}-{np.max(d):.1f} px at 512")
        o = np.array([target_offset(im, "red") for im in z[f"{k}_base"]])
        assert np.allclose(o, -np.array([target_offset(im, "blue") for im in z[f"{k}_base"]])) and o.any(1).all()
    from PIL import Image
    tiles = [np.concatenate([mask_image(z[f"{k}_base"][j], c) for c in ("red", "blue")], 1) for k in ("clean", "light_l3", "cam_l3") for j in range(4)]
    Image.fromarray((np.concatenate(tiles, 2).transpose(1, 2, 0) * 255).astype(np.uint8)).save(sys.argv[2] if len(sys.argv) > 2 else "target_views.png")
