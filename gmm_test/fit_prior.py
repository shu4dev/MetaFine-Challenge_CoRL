"""Fit the motion prior on the planner demos (as delta actions) and run the sanity checks:
1. torch log_prob matches sklearn
2. scores: planner (held out) vs training demos vs baseline rollouts
3. noise lowers the score
4. following the gradient makes baseline action windows smoother
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prior import W, MotionPrior, features, windows  # noqa: E402

rng = np.random.default_rng(0)
f = lambda w: features(torch.as_tensor(w, dtype=torch.float64)).numpy()

# planner demos as delta actions, split by episode
d = np.load(HERE / "demo_tcp_t3.npz")
eps = sorted({k.split("|")[0] for k in d.files})
planner = {ep: (d[ep + "|action"][:, :7] - d[ep + "|qpos"][:-1, :7]) / 0.1 for ep in eps}
val_eps = set(rng.choice(eps, size=30, replace=False))
tr = np.concatenate([windows(planner[e]) for e in eps if e not in val_eps])
va = np.concatenate([windows(planner[e]) for e in eps if e in val_eps])

# training demos (LeRobot) and baseline rollouts (LoRA 20k, replan 25)
L = np.load(HERE / "lerobot_demos_t3.npz")
demo = np.concatenate([windows(L["action"][L["episode_index"] == i, :7]) for i in np.unique(L["episode_index"])])
run = HERE.parent / "runs/2026-10-11_041055"
e = pd.read_csv(run / "episodes.csv")
roll = np.concatenate([windows(np.load(run / r.traj_file)["action"][:, :7]) for _, r in e[e.success].iterrows()])

Ftr = f(tr)
mu, sd = Ftr.mean(0), Ftr.std(0)
z = lambda F: (F - mu) / sd
best = None
for K in (1, 2, 4, 8, 16):
    g = GaussianMixture(K, covariance_type="full", reg_covar=1e-3, random_state=0, n_init=2).fit(z(Ftr))
    ll = g.score(z(f(va)))
    print(f"K={K:2d} val log-lik {ll:7.2f}")
    if best is None or ll > best[0]:
        best = (ll, K, g)
_, K, g = best
prior = MotionPrior(g.weights_, g.means_, g.covariances_, mu, sd)
prior.save(HERE / "prior_planner_ctxfree.npz")
print(f"chose K={K}; saved prior_planner_ctxfree.npz")

# 1. torch matches sklearn
ref = g.score_samples(z(f(va[:1000])))
ours = prior.log_prob(torch.as_tensor(va[:1000], dtype=torch.float64)).numpy()
print(f"check 1: max |torch - sklearn| = {np.abs(ref - ours).max():.2e}")

# 2. scores
lp = lambda w: prior.log_prob(torch.as_tensor(w, dtype=torch.float64)).numpy()
print("check 2: median log-prob  planner(held out) %.1f | training demos %.1f | baseline rollouts %.1f"
      % (np.median(lp(va)), np.median(lp(demo)), np.median(lp(roll))))

# 3. noise at 0.25 x action std lowers the score
std = tr.reshape(-1, 7).std(0)
noisy = va + rng.normal(size=va.shape) * 0.25 * std
print(f"check 3: noise lowers the score for {np.mean(lp(noisy) < lp(va)):.0%} of held-out windows")

# 4. gradient ascent on log-prob, staying close to the original actions
x0 = torch.as_tensor(roll[rng.choice(len(roll), 2000, replace=False)], dtype=torch.float64)
x = x0.clone().requires_grad_(True)
opt = torch.optim.Adam([x], lr=1e-3)
for _ in range(200):
    loss = -prior.log_prob(x).mean() + 10.0 * ((x - x0) ** 2).sum((1, 2)).mean()
    opt.zero_grad(); loss.backward(); opt.step()
rough = lambda w: (w[:, 2:] - 2 * w[:, 1:-1] + w[:, :-2]).abs().mean().item()
print("check 4: roughness of baseline windows %.4f -> %.4f after following the prior "
      "(planner %.4f, training demos %.4f); mean action change %.4f"
      % (rough(x0), rough(x.detach()), rough(torch.as_tensor(va)), rough(torch.as_tensor(demo)),
         (x.detach() - x0).abs().mean().item()))
