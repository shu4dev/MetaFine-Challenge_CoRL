"""Compare the prior-test runs: success (paired McNemar), smoothness on episodes where both succeeded,
GMM score of the rollouts, and the training aux trend."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import binomtest, wilcoxon
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO / "gmm_test"))
from make_report import log_dimensionless_jerk, sparc
from prior import MotionPrior, windows
prior = MotionPrior.load(REPO / "gmm_test/prior_planner_ctxfree.npz")
RUNS = {"baseline": "2026-10-11_041055", "control (λ=0)": "2026-10-11_133203",
        "λ=1e-4": "2026-10-11_145122", "λ=1e-3": "2026-10-11_161107"}

def ep_metrics(run, r):
    z = np.load(REPO / "runs" / run / r.traj_file); dt = float(z["dt"]); pos = z["tcp_pose"][:, :3]
    sp = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    a = z["action"][:, :7]
    w = windows(a)
    lp = np.median(prior.log_prob(torch.as_tensor(w, dtype=torch.float64)).numpy()) if len(w) else np.nan
    return log_dimensionless_jerk(pos, dt), sparc(sp, dt), np.abs(np.diff(a, 2, axis=0)).mean(), lp

E = {}
for k, run in RUNS.items():
    e = pd.read_csv(REPO / "runs" / run / "episodes.csv").set_index("episode_id")
    m = [ep_metrics(run, r) if r.success else (np.nan,) * 4 for _, r in e.iterrows()]
    e[["ldlj", "sparc", "rough", "gmm"]] = m
    E[k] = e

def mcnemar(a, b):
    b01 = int((~a & b).sum()); b10 = int((a & ~b).sum())
    return b01, b10, (binomtest(b01, b01 + b10).pvalue if b01 + b10 else 1.0)

ctl = E["control (λ=0)"]
for ref_name in ["control (λ=0)", "baseline"]:
    ref = E[ref_name]
    print(f"\n=== vs {ref_name} ===")
    for k, e in E.items():
        if k == ref_name: continue
        ids = ref.index.intersection(e.index)
        a, b = ref.loc[ids, "success"].astype(bool), e.loc[ids, "success"].astype(bool)
        g, l, p = mcnemar(a, b)
        both = ids[a & b]
        row = f"{k:14s} SR {b.mean():.3f} vs {a.mean():.3f}  gained {g} lost {l} p={p:.3f} | both-succeeded n={len(both)}"
        for col in ["ldlj", "sparc", "rough", "gmm"]:
            x, y = ref.loc[both, col], e.loc[both, col]
            pw = wilcoxon(y, x).pvalue
            row += f" | {col} {y.median():.3f} vs {x.median():.3f} (p={pw:.3f})"
        print(row)

print("\n=== all successful episodes (median) ===")
for k, e in E.items():
    s = e[e.success]
    print(f"{k:14s} n={len(s):3d} ldlj {s.ldlj.median():.2f} sparc {s.sparc.median():.2f} rough {s.rough.median():.4f} gmm {s.gmm.median():.1f} time {s.completion_time_s.median():.2f}")

print("\n=== training aux (-log p, lower = smoother per GMM), mean of first/last 200 steps ===")
for L in ["0", "1e-4", "1e-3"]:
    a = pd.read_csv(REPO / f"gmm_test/aux_gmmtest_lambda{L}.csv", header=None, names=["step", "loss", "aux", "valid"])
    print(f"λ={L:5s} n={len(a)} loss {a.loss[:200].mean():.4f} -> {a.loss[-200:].mean():.4f} | aux {a.aux[:200].mean():.2f} -> {a.aux[-200:].mean():.2f}")

# ---- PDF: score tables (front), a GMM verdict page, then the standard report pages ----
import textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import make_report as MR

rows = []
for k, e in E.items():
    if k == "control (λ=0)":
        rows.append([k, f"{e.success.mean():.3f}", "-", "-", "-", "-", "-"]); continue
    ids = ctl.index.intersection(e.index)
    a, b = ctl.loc[ids, "success"].astype(bool), e.loc[ids, "success"].astype(bool)
    g, l, p = mcnemar(a, b)
    both = ids[a & b]
    cell = lambda c, f: f"{e.loc[both, c].median():{f}} vs {ctl.loc[both, c].median():{f}} (p={wilcoxon(e.loc[both, c], ctl.loc[both, c]).pvalue:.2f})"
    rows.append([k, f"{b.mean():.3f}", f"+{g} / -{l} (p={p:.2f})", str(len(both)),
                 cell("ldlj", ".2f"), cell("rough", ".4f"), cell("gmm", ".1f")])
aux = {L: pd.read_csv(REPO / f"gmm_test/aux_gmmtest_lambda{L}.csv", header=None, names=["step", "loss", "aux", "valid"])
       for L in ["0", "1e-4", "1e-3"]}

out = REPO / "reports/t3_gmm_prior_test.pdf"
runs = [MR.Run(REPO / "runs" / r, i) for i, r in enumerate(RUNS.values())]
with PdfPages(out) as pdf:
    for run in runs:
        MR.page_score_table(pdf, run)
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("T3 GMM motion prior: does it cut jerk at matched success?", fontsize=14, weight="bold")
    ax = fig.add_axes([0.03, 0.55, 0.94, 0.33]); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["Model", "Understanding\nSR", "Successes\ngained / lost\nvs control",
                 "Scenes both\nsucceeded", "Jerk (LDLJ)\ncloser to 0 = smoother\nthis vs control", "Action roughness\nlower = smoother\nthis vs control",
                 "GMM score\nhigher = planner-like\nthis vs control"],
                 loc="upper center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 2.2)
    for (r, c), cl in t.get_celld().items():
        if r == 0: cl.set_height(cl.get_height() * 2)
    ax2 = fig.add_axes([0.07, 0.08, 0.4, 0.38])
    for L, a in aux.items():
        ax2.plot(a.step, a.aux.rolling(100).mean(), label=f"λ={L}")
    ax2.set_xlabel("training step"); ax2.set_ylabel("prior loss (−log p, lower = more planner-like)")
    ax2.set_title("Prior loss during training (100-step mean)", fontsize=10); ax2.legend()
    note = ("All rows: LoRA 20k + 2000 more LoRA steps, same seed, 100 dev scenes, Understanding, replan 25, "
            "torch seed 0. The control (λ=0) is the fair comparison; the 20k baseline is in the score tables. "
            "Smoothness compares only scenes where both models succeeded (paired Wilcoxon). "
            "VERDICT: No. The prior loss trained down hard at λ=1e-3 and the rollouts score as more "
            "planner-like by the GMM (p<0.001), but jerk (LDLJ) and action roughness did not change, and "
            "success did not change either (0.570 vs 0.575). The GMM was satisfied without the motion getting "
            "smoother: it rewards the planner's joint pattern, not low jerk itself. "
            "Next: a prior built only on roughness features, a larger λ, and chunk blending "
            "(about 1/3 of jerk is at replan boundaries, which a per-chunk prior cannot reach).")
    fig.text(0.53, 0.45, "\n".join(textwrap.wrap(note, 62)), va="top", fontsize=9)
    pdf.savefig(fig); plt.close(fig)
    MR.page_summary(pdf, runs)
    MR.page_task(pdf, "T3", "toggle_switch_table", runs)
    MR.page_motion(pdf, "T3", "toggle_switch_table", runs)
print("wrote", out)
