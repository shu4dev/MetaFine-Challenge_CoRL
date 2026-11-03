#!/usr/bin/env python3
"""Summarize the seed baselines (seed-baseline summary): one table row per run, two figures.

    python tools/summarize_seed_baselines.py                      # s1001 and s1002 -> stdout table, report/figures/*.png
    python tools/summarize_seed_baselines.py --runs pi0_toggle_switch_table_s1002

Reads logs/<run>.train.log and logs/<run>.status (lerobot-train through tools/launch_seed_baselines.sh and
run_seed.sh) and the newest eval_runs/<run>/<timestamp>/ from tools/run_eval.py (episodes.csv, episodes/*.npz,
official/understanding_summary.json). LDLJ, SPARC and completion time are medians over successful episodes,
as in gmm_test/smooth_demos/analyze.py. The seed-1000 row is transcribed from the midterm report
(Josh's run; its trajectories are not in git, so its smoothness cells read n/r).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from make_report import log_dimensionless_jerk, sparc  # noqa: E402  (branch t3-lora-baseline)

LOG_FREQ = 200  # --log_freq of the launcher; lerobot prints the step rounded ("5K"), so the line index gives the step
SEED1000 = {"run": "pi0_toggle_switch_table_s1000 (Josh, midterm)", "seed": 1000, "steps": 20000, "success": 0.62,
            "n": 200, "right_only": 68, "counts": "68 / 47 / 66 / 19", "ldlj": None, "sparc": None, "time_s": None,
            "wall_h": 12.0, "gpu": "RTX 5070 12 GB (10k + resume 10k)"}


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return float("nan"), float("nan")
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def loss_curve(log: Path) -> tuple[np.ndarray, np.ndarray]:
    txt = log.read_text(errors="replace").replace("\r", "\n")
    losses = [float(m) for m in re.findall(r"ot_train\.py:\d+ step:\S+ .*?loss:([0-9.eE+-]+)", txt)]
    return np.arange(1, len(losses) + 1) * LOG_FREQ, np.array(losses)


def status(path: Path) -> dict:
    out = {"gpu": None, "wall_h": None}
    if not path.exists():
        return out
    t: dict[str, str] = {}
    for line in path.read_text().splitlines():
        when = line.split()[0] if line.split() else ""
        if "START train" in line:
            t["start"] = when
            m = re.search(r"gpu=(.*)$", line)
            out["gpu"] = m.group(1).strip() if m else None
        elif "END train" in line:
            t["end"] = when
    if "start" in t and "end" in t:
        out["wall_h"] = (datetime.fromisoformat(t["end"]) - datetime.fromisoformat(t["start"])).total_seconds() / 3600
    return out


def latest_eval(run: str) -> Path | None:
    d = REPO / "eval_runs" / run
    found = sorted(d.glob("*/official/understanding_summary.json")) if d.exists() else []
    return found[-1].parent.parent if found else None


def _bool(col: pd.Series) -> pd.Series:
    return col.map(lambda v: str(v).strip().lower() == "true")


def eval_metrics(run_dir: Path) -> dict:
    summ = json.loads((run_dir / "official" / "understanding_summary.json").read_text())
    e = pd.read_csv(run_dir / "episodes.csv")
    e["success"] = _bool(e["success"])
    red, blue = _bool(e["red_toggled"]), _bool(e["blue_toggled"])
    right = ((e.variant == "red") & red & ~blue) | ((e.variant == "blue") & blue & ~red)
    wrong = ((e.variant == "red") & blue & ~red) | ((e.variant == "blue") & red & ~blue)
    ldlj, sp = [], []
    for _, r in e.iterrows():
        z = np.load(run_dir / r.traj_file)
        dt = float(z["dt"])
        pos = z["tcp_pose"][:, :3]
        ldlj.append(log_dimensionless_jerk(pos, dt))
        sp.append(sparc(np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt, dt))
    e["ldlj"], e["sparc"] = ldlj, sp
    s = e[e.success]
    med = (lambda c: float(s[c].median()) if len(s) else float("nan"))
    return {"success": float(summ["success_rate"]), "n": int(summ["n_episodes"]), "n_success": int(len(s)),
            "per_variant": summ.get("per_variant_success_rate"), "right_only": int(right.sum()),
            "counts": f"{int(right.sum())} / {int(wrong.sum())} / {int((red & blue).sum())} / {int((~red & ~blue).sum())}",
            "ldlj": med("ldlj"), "sparc": med("sparc"), "time_s": med("completion_time_s"),
            "eval_dir": str(run_dir.relative_to(REPO))}


def fmt(x, nd: int = 2) -> str:
    return "n/r" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["pi0_toggle_switch_table_s1001", "pi0_toggle_switch_table_s1002"])
    ap.add_argument("--figures", type=Path, default=REPO / "report" / "figures")
    a = ap.parse_args()

    rows = [dict(SEED1000)]
    curves = {}
    for run in a.runs:
        seed = int(run.rsplit("s", 1)[1])
        st = status(REPO / "logs" / f"{run}.status")
        log = REPO / "logs" / f"{run}.train.log"
        if log.exists():
            curves[run] = loss_curve(log)
        row = {"run": run, "seed": seed, "steps": 20000, "gpu": st["gpu"], "wall_h": st["wall_h"], "success": None,
               "n": 0, "right_only": None, "counts": "pending", "ldlj": None, "sparc": None, "time_s": None}
        ev = latest_eval(run)
        if ev is not None:
            row.update(eval_metrics(ev))
        rows.append(row)

    print("| Run | Seed | Steps | Success [Wilson 95%] | Right / wrong / both / none | LDLJ | SPARC | Time (s) | Train wall | GPU |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["success"] is None:
            succ = "pending"
        else:
            lo, hi = wilson(r["success"], r["n"])
            succ = f"{r['success']:.3f} [{lo:.2f}, {hi:.2f}] (n={r['n']})"
        wall = "pending" if r["wall_h"] is None else f"{r['wall_h']:.1f} h"
        print(f"| {r['run']} | {r['seed']} | {r['steps']} | {succ} | {r['counts']} | {fmt(r['ldlj'], 1)} | "
              f"{fmt(r['sparc'])} | {fmt(r['time_s'])} | {wall} | {r['gpu'] or 'n/r'} |")
    for r in rows[1:]:
        if r.get("per_variant"):
            print(f"{r['run']}: per-variant success {r['per_variant']}, successes {r['n_success']}, eval {r['eval_dir']}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a.figures.mkdir(parents=True, exist_ok=True)
    if curves:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for run, (x, y) in curves.items():
            ax.plot(x, y, lw=1, label=f"seed {run.rsplit('s', 1)[1]}")
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel("flow-matching loss")
        ax.set_title("pi0 LoRA on T3: training loss, 20k steps in one run")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(a.figures / "seed_baselines_loss.png", dpi=150)
        print("wrote", a.figures / "seed_baselines_loss.png")
    done = [r for r in rows if r["success"] is not None]
    if done:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        xs = np.arange(len(done))
        p = np.array([r["success"] for r in done])
        ci = np.array([wilson(r["success"], r["n"]) for r in done])
        ax.errorbar(xs, p, yerr=[p - ci[:, 0], ci[:, 1] - p], fmt="o", capsize=4)
        ax.axhline(0.60, ls="--", c="gray", lw=1)
        ax.text(len(done) - 0.55, 0.615, "organizers' pi0 0.60", fontsize=8, color="gray", ha="right")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"seed {r['seed']}" for r in done])
        ax.set_ylim(0, 1)
        ax.set_ylabel("understanding success, 200 episodes")
        ax.set_title("T3 pi0 baseline per seed (Wilson 95%)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(a.figures / "seed_baselines_success.png", dpi=150)
        print("wrote", a.figures / "seed_baselines_success.png")


if __name__ == "__main__":
    main()
