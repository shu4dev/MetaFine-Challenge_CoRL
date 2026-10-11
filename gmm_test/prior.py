"""Motion prior: a GMM over smoothness features of short windows of arm actions.

Features of one window (W consecutive actions, 7 arm joints, delta-action units):
log RMS of the actions, of their first difference and of their second difference, per joint
(21 numbers). Context-free for now (no instruction or stage). ``log_prob`` is differentiable
in torch so it can be used as a training loss.
"""
from __future__ import annotations

import numpy as np
import torch

W = 10        # window length in actions (0.5 s at 20 Hz)
EPS = 1e-4


def features(win: torch.Tensor) -> torch.Tensor:
    """win (B, W, 7) -> (B, 21)."""
    def lrms(x):
        return 0.5 * torch.log((x ** 2).mean(dim=1) + EPS ** 2)
    d1 = win[:, 1:] - win[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    return torch.cat([lrms(win), lrms(d1), lrms(d2)], dim=1)


def windows(seq: np.ndarray, stride: int = 1) -> np.ndarray:
    """seq (T, 7) -> (N, W, 7) sliding windows."""
    if len(seq) < W:
        return np.zeros((0, W, seq.shape[1]), seq.dtype)
    idx = np.arange(0, len(seq) - W + 1, stride)[:, None] + np.arange(W)
    return seq[idx]


class MotionPrior(torch.nn.Module):
    def __init__(self, weights, means, covariances, feat_mean, feat_std):
        super().__init__()
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)
        self.register_buffer("log_w", torch.log(f(weights)))
        self.register_buffer("means", f(means))
        prec_chol = torch.linalg.cholesky(torch.linalg.inv(f(covariances)))  # (K, D, D)
        self.register_buffer("prec_chol", prec_chol)
        self.register_buffer("log_det", torch.log(torch.diagonal(prec_chol, dim1=-2, dim2=-1)).sum(-1))
        self.register_buffer("feat_mean", f(feat_mean))
        self.register_buffer("feat_std", f(feat_std))

    def log_prob_features(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, D) raw features -> (B,) log density of the standardized features."""
        z = ((x.double() - self.feat_mean) / self.feat_std)
        diff = z[:, None, :] - self.means[None]                         # (B, K, D)
        y = torch.einsum("bkd,kde->bke", diff, self.prec_chol)          # (B, K, D)
        D = z.shape[1]
        comp = -0.5 * (y ** 2).sum(-1) + self.log_det - 0.5 * D * np.log(2 * np.pi)
        return torch.logsumexp(comp + self.log_w, dim=1).to(x.dtype)

    def log_prob(self, win: torch.Tensor) -> torch.Tensor:
        """win (B, W, 7) delta-action windows -> (B,)."""
        return self.log_prob_features(features(win))

    def save(self, path):
        np.savez(path, **{k: v.cpu().numpy() for k, v in self.state_dict().items()}, W=W)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls.__new__(cls)
        torch.nn.Module.__init__(m)
        for k in ("log_w", "means", "prec_chol", "log_det", "feat_mean", "feat_std"):
            m.register_buffer(k, torch.as_tensor(z[k]))
        return m
