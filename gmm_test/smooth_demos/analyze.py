"""Smooth-demo test: success (paired McNemar) and smoothness (all successes + episodes where both succeeded)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest, mannwhitneyu, wilcoxon
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from make_report import log_dimensionless_jerk, sparc
RUNS = {"baseline": "2026-10-11_041055", "control 2k": "2026-10-11_133203", "smooth 2k": "2026-10-17_195552", "smooth 4k": "2026-10-17_234514", "smooth 6k": "2026-10-17_235508"}

def metrics(run, r):
    z = np.load(REPO / "runs" / run / r.traj_file); dt = float(z["dt"]); pos = z["tcp_pose"][:, :3]
    a = z["action"][:, :7]
    return (log_dimensionless_jerk(pos, dt), sparc(np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt, dt),
            np.abs(np.diff(a, 2, axis=0)).mean(), np.abs(a).max(1).mean())

E = {}
for k, run in RUNS.items():
    e = pd.read_csv(REPO / "runs" / run / "episodes.csv").set_index("episode_id")
    e[["ldlj", "sparc", "rough", "mean_max_abs_a"]] = [metrics(run, r) for _, r in e.iterrows()]
    e["success"] = e.success.astype(bool)
    E[k] = e
for k, e in E.items():
    s = e[e.success]
    print(f"{k:11s} SR {e.success.mean():.3f} | successes n={len(s)}: ldlj {s.ldlj.median():.2f} sparc {s.sparc.median():.2f} "
          f"rough {s.rough.median():.4f} | ALL eps rough {e.rough.median():.4f} ldlj {e.ldlj.median():.2f} time {s.completion_time_s.median():.2f}")
for name in ["smooth 2k", "smooth 4k", "smooth 6k"]:
  new = E[name]; print("==", name)
  for ref in ["control 2k", "baseline"]:
      r = E[ref]; ids = r.index.intersection(new.index)
      a, b = r.loc[ids, "success"], new.loc[ids, "success"]
      g, l = int((~a & b).sum()), int((a & ~b).sum())
      both = ids[a & b]
      out = f"vs {ref}: gained {g} lost {l} p={binomtest(g, g + l).pvalue:.3f} | both ok n={len(both)}"
      for c in ["ldlj", "sparc", "rough"]:
          out += f" | {c} {new.loc[both, c].median():.4f} vs {r.loc[both, c].median():.4f} p={wilcoxon(new.loc[both, c], r.loc[both, c]).pvalue:.3g}"
      out += f" | ALL rough p={mannwhitneyu(new.rough, r.rough).pvalue:.3g}"
      print(out)

# ---- PDF: this run's score table first, a verdict page, then the standard report pages ----
import textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import make_report as MR

def counts(e):
    R, B = e.red_toggled.astype(bool), e.blue_toggled.astype(bool)
    wrong = ((e.variant == "red") & B & ~R) | ((e.variant == "blue") & R & ~B)
    return int(wrong.sum()), int((R & B).sum()), int((~R & ~B).sum())

ctl = E["control 2k"]; rows = []
for k, e in E.items():
    w, bo, no = counts(e)
    if k == "control 2k":
        rows.append([k, f"{e.success.mean():.3f}", "-", "-", "-", "-", "-", f"{w} / {bo} / {no}"]); continue
    ids = ctl.index.intersection(e.index)
    a, b = ctl.loc[ids, "success"], e.loc[ids, "success"]
    g, l = int((~a & b).sum()), int((a & ~b).sum())
    both = ids[a & b]
    cell = lambda c, f: f"{e.loc[both, c].median():{f}} vs {ctl.loc[both, c].median():{f}}\n(p={wilcoxon(e.loc[both, c], ctl.loc[both, c]).pvalue:.1g})"
    rows.append([k, f"{b.mean():.3f}", f"+{g} / -{l}\n(p={binomtest(g, g + l).pvalue:.3f})", str(len(both)),
                 cell("ldlj", ".2f"), cell("sparc", ".2f"), cell("rough", ".4f"), f"{w} / {bo} / {no}"])

out = REPO / "reports/t3_smooth_demos_test.pdf"
runs = [MR.Run(REPO / "runs" / r, i) for i, r in enumerate(RUNS.values())]
with PdfPages(out) as pdf:
    MR.page_score_table(pdf, runs[3])
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("T3 smooth demos: does training on continuous-motion demos make pi0 smoother?", fontsize=14, weight="bold")
    ax = fig.add_axes([0.03, 0.52, 0.94, 0.36]); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["Model", "Understanding\nSR", "Successes\ngained / lost\nvs control",
                 "Scenes both\nsucceeded", "Jerk (LDLJ)\ncloser to 0 = smoother\nthis vs control",
                 "SPARC\ncloser to 0 = smoother\nthis vs control", "Action roughness\nlower = smoother\nthis vs control",
                 "Wrong only /\nboth / none\n(of 200)"], loc="upper center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 3.2)
    for (r, c), cl in t.get_celld().items():
        if r == 0: cl.set_height(cl.get_height() * 2)
    note = ("Control and smooth: LoRA 20k + a fresh LoRA trained 2000 steps, same seed and settings. Control used the "
            "original 200 demos; smooth used the same scenes remade as one continuous joint motion "
            "(gmm_test/smooth_demos/gen_smooth.py; 192 of 200 usable). Demo roughness after conversion: 0.0115 -> 0.0027 "
            "(10-scene check). Eval: 100 dev scenes x red/blue, replan 25, torch seed 0. Smoothness is compared only on "
            "scenes where both models succeeded (paired Wilcoxon), so it is not 'smoother because it did less'. "
            "VERDICT: YES at 4k steps. Smooth 2k + 2000 more steps (4k total) is much smoother on matched successes "
            "(roughness 0.0051 vs 0.0123, LDLJ -9.3 vs -10.3, SPARC -1.55 vs -1.79, all p<1e-7), finishes faster "
            "(4.15 s vs 4.40 s) and has the SAME success as the control (0.575, +38/-38) with fewer 'none' (23 vs 29). "
            "Vs the 20k baseline (0.62): +31/-40, p=0.34, not significant. At 2k steps success had dropped (0.445, the "
            "model was mid-switch from the old style); at 6k it is 0.505 (+34/-48 vs control, p=0.15), so 4k is the best "
            "point. Rows 4k/6k continue from smooth 2k with a fresh LoRA (one cosine schedule over the extra 4000 "
            "steps, checkpoints at +2000 and +4000).")
    fig.text(0.05, 0.36, "\n".join(textwrap.wrap(note, 130)), va="top", fontsize=9.5)
    pdf.savefig(fig); plt.close(fig)
    for i in (4, 2, 1, 0): MR.page_score_table(pdf, runs[i])
    MR.page_summary(pdf, runs)
    MR.page_task(pdf, "T3", "toggle_switch_table", runs)
    MR.page_motion(pdf, "T3", "toggle_switch_table", runs)
print("wrote", out)
