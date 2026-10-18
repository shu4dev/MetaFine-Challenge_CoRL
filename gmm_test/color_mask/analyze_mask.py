"""Target-view test: success, red/blue confusion and smoothness of every run listed in run_mask_test.log
("RUN <name> <step> <folder>") vs smooth 4k (same model, no view) and the LoRA 20k baseline.
Writes reports/t3_target_view_test.pdf (front page: score table of the best target-view run)."""
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import make_report as MR  # noqa: E402
from make_report import log_dimensionless_jerk  # noqa: E402

REF = {"baseline (LoRA 20k)": "2026-10-11_041055", "smooth 4k (no view)": "2026-10-17_234514"}
RUNS = dict(REF)
for line in open(REPO / "gmm_test/color_mask/run_mask_test.log"):
    if line.startswith("RUN "):
        _, name, step, folder = line.split()
        RUNS[f"{name.replace('smooth4k_', '')} +{int(step)}"] = Path(folder).name


def load(run):
    e = pd.read_csv(REPO / "runs" / run / "episodes.csv").set_index("episode_id")
    e["success"] = e.success.astype(bool)
    R, B = e.red_toggled.astype(bool), e.blue_toggled.astype(bool)
    e["wrong"] = ((e.variant == "red") & B & ~R) | ((e.variant == "blue") & R & ~B)
    e["both"], e["none"] = R & B, ~R & ~B
    e["right"] = ~e.wrong & ~e.both & ~e["none"]
    rough, ldlj = [], []
    for _, r in e.iterrows():
        z = np.load(REPO / "runs" / run / r.traj_file)
        a = z["action"][:, :7]
        rough.append(np.abs(np.diff(a, 2, axis=0)).mean())
        ldlj.append(log_dimensionless_jerk(z["tcp_pose"][:, :3], float(z["dt"])))
    e["rough"], e["ldlj"] = rough, ldlj
    return e


E = {k: load(r) for k, r in RUNS.items()}
ref = E["smooth 4k (no view)"]
rows = []
for k, e in E.items():
    row = [k, f"{e.success.mean():.3f}", f"{int(e.right.sum())} / {int(e.wrong.sum())} / {int(e.both.sum())} / {int(e['none'].sum())}"]
    if e is ref:
        rows.append(row + ["-", "-", f"{e[e.success].rough.median():.4f}"]); continue
    ids = ref.index.intersection(e.index)
    a, b = ref.loc[ids, "success"], e.loc[ids, "success"]
    g, l = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(g, g + l).pvalue if g + l else 1.0
    both = ids[a & b]
    pr = wilcoxon(e.loc[both, "rough"], ref.loc[both, "rough"]).pvalue if len(both) > 10 else float("nan")
    rows.append(row + [f"+{g} / -{l}\n(p={p:.3f})", str(len(both)),
                       f"{e.loc[both, 'rough'].median():.4f} vs {ref.loc[both, 'rough'].median():.4f}\n(p={pr:.2g})"])
    print(f"{k:22s} SR {b.mean():.3f}  vs smooth4k +{g}/-{l} p={p:.3f}  right/wrong/both/none {row[2]}")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

mask_runs = [k for k in RUNS if k.startswith("mask")]
order = ([max(mask_runs, key=lambda k: E[k].success.mean())] if mask_runs else []) + \
        [k for k in RUNS if k not in mask_runs[:0]]
runs = {k: MR.Run(REPO / "runs" / r, i) for i, (k, r) in enumerate(RUNS.items())}
out = REPO / "reports/t3_target_view_test.pdf"
with PdfPages(out) as pdf:
    if mask_runs:
        MR.page_score_table(pdf, runs[order[0]])
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("T3 target view: does showing pi0 a zoomed switch with a dot on the target fix red/blue?",
                 fontsize=13, weight="bold")
    ax = fig.add_axes([0.03, 0.40, 0.94, 0.50]); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["Model (+ = extra LoRA steps)", "Understanding\nSR",
                 "Right only / wrong only /\nboth / none (of 200)", "Successes gained / lost\nvs smooth 4k",
                 "Scenes both\nsucceeded", "Action roughness, lower = smoother\nthis vs smooth 4k"],
                 loc="upper center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 2.6)
    note = ("All rows start from the same models and use the 100 dev scenes x red/blue, Understanding, replan 25, "
            "torch seed 0. 'mask +N': smooth 4k + a fresh LoRA trained N steps on the smooth demos with the "
            "target view (tools/target_mask.py: 48 px crop of the first base-camera frame around the switch, "
            "green dot on the instructed slider, 224 px) in pi0's third image slot, at train and test time. "
            "'nomask +N': same training without the view (control for 'more training'). Smoothness compares "
            "only scenes both models solved. Success also counts 'both' (pressing both sliders), so 'right only' "
            "is the real test of picking the colour.")
    fig.text(0.05, 0.33, "\n".join(textwrap.wrap(note, 130)), va="top", fontsize=9.5)
    pdf.savefig(fig); plt.close(fig)
    for k in RUNS:
        if not mask_runs or k != order[0]:
            MR.page_score_table(pdf, runs[k])
    MR.page_summary(pdf, list(runs.values()))
    MR.page_task(pdf, "T3", "toggle_switch_table", list(runs.values()))
    MR.page_motion(pdf, "T3", "toggle_switch_table", list(runs.values()))
if "mask +6000" in E and "nomask +6000" in E:  # the fair comparison: same training, with vs without the view
    a, b = E["nomask +6000"].right, E["mask +6000"].right.loc[E["nomask +6000"].index]
    g, l = int((~a & b).sum()), int((a & ~b).sum())
    print(f"right-only, mask +6000 vs nomask +6000: +{g}/-{l} p={binomtest(g, g + l).pvalue:.3f}")
print("wrote", out)
