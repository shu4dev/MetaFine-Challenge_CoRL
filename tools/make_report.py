#!/usr/bin/env python3
"""Turn one or more run folders (from tools/run_eval.py) into a PDF report.

The front page of every report is a table of every score for each run: the organizers'
30K-step pi0 value (``eval_runs/<task>/metafine_report.json``), this run's value, what it
measures, and whether it improved. After that come all five graded tasks and per-task plots::

    python tools/make_report.py runs/2026-10-10_060512                 # -> runs/.../report.pdf
    python tools/make_report.py runs/A runs/B --out reports/a_vs_b.pdf  # compare runs

Reads only CSV / JSON / npz, so it needs no simulator or GPU.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

_REPO = Path(__file__).resolve().parent.parent

TASKS = [
    ("T1", "grasp_part"),
    ("T2", "grasp_move_mug"),
    ("T3", "toggle_switch_table"),
    ("T4", "put_blocks_into_boxes"),
    ("T5", "insert_letter"),
]
CAM_LEVELS = ["clean", "cam_l1", "cam_l2", "cam_l3"]
LIGHT_LEVELS = ["clean", "light_l1", "light_l2", "light_l3"]
REF_LABEL = "Organizers' pi0 (reference)"
REF_NOTE = ("Organizers' pi0: 30k steps, batch 16, their own 20 seeds. Different scenes and "
            "training budget from our runs, so it is a reference point, not a paired comparison.")
PAGE = (8.5, 11)
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
REF_COLOR = "#6b7280"


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #

class Run:
    def __init__(self, path: Path, index: int):
        self.path = path
        self.info = json.loads((path / "run.json").read_text())
        self.task = self.info["task_id"]
        self.episodes = pd.read_csv(path / "episodes.csv")
        report = path / "official" / "metafine_report.json"
        self.report = json.loads(report.read_text()) if report.exists() else {}
        und = path / "official" / "understanding_summary.json"
        self.understanding = json.loads(und.read_text()) if und.exists() else {}
        self.color = COLORS[index % len(COLORS)]
        note = self.info.get("note") or ""
        self.label = path.name + (f" ({note.split(';')[0]})" if note else "")


def load_reference(task: str) -> dict:
    path = _REPO / "eval_runs" / task / "metafine_report.json"
    return json.loads(path.read_text()) if path.exists() else {}


def headline(report: dict) -> dict:
    """Perception AUSC, Understanding SR and Behavior score from a metafine_report."""
    m = report.get("metrics") or {}
    beh = m.get("behavior") or {}
    behavior = beh.get("mean_stage_fraction")
    if behavior is None:
        behavior = beh.get("success_rate")
    return {
        "perception": (m.get("perception") or {}).get("ausc_mean"),
        "understanding": (m.get("understanding") or {}).get("success_rate"),
        "behavior": behavior,
    }


# --------------------------------------------------------------------------- #
# Statistics and motion metrics                                               #
# --------------------------------------------------------------------------- #

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def log_dimensionless_jerk(pos: np.ndarray, dt: float) -> float:
    """LDLJ of a 3-D path (Balasubramanian et al. 2015); closer to 0 is smoother."""
    if len(pos) < 6:
        return float("nan")
    vel = np.gradient(pos, dt, axis=0)
    jerk = np.gradient(np.gradient(vel, dt, axis=0), dt, axis=0)
    duration = (len(pos) - 1) * dt
    v_peak = np.linalg.norm(vel, axis=1).max()
    if v_peak <= 1e-9:
        return float("nan")
    integral = np.sum(np.sum(jerk ** 2, axis=1)) * dt
    return float(-np.log(duration ** 3 / v_peak ** 2 * integral))


def sparc(speed: np.ndarray, dt: float, fc: float = 10.0, amp_th: float = 0.05,
          pad_level: int = 4) -> float:
    """Spectral arc length of a speed profile (Balasubramanian et al. 2015); closer to 0 is smoother."""
    if len(speed) < 4 or np.max(speed) <= 1e-9:
        return float("nan")
    n_fft = int(2 ** (np.ceil(np.log2(len(speed))) + pad_level))
    freq = np.arange(n_fft) / (n_fft * dt)
    mag = np.abs(np.fft.fft(speed, n_fft))
    mag = mag / mag.max()
    keep = freq <= fc
    freq, mag = freq[keep], mag[keep]
    above = np.nonzero(mag >= amp_th)[0]
    freq, mag = freq[: above[-1] + 1], mag[: above[-1] + 1]
    dfreq = np.diff(freq) / (freq[-1] - freq[0])
    return float(-np.sum(np.sqrt(dfreq ** 2 + np.diff(mag) ** 2)))


def motion_table(run: Run) -> pd.DataFrame:
    """Per-episode smoothness of the end-effector path, successful clean episodes only."""
    rows = []
    for _, ep in run.episodes.iterrows():
        if not ep["success"] or ep["condition"] != "clean":
            continue
        data = np.load(run.path / ep["traj_file"])
        tcp = data["tcp_pose"]
        if tcp.ndim != 2 or len(tcp) < 6:
            continue
        dt = float(data["dt"])
        pos = tcp[:, :3]
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
        rows.append({"episode_id": ep["episode_id"], "ldlj": log_dimensionless_jerk(pos, dt),
                     "sparc": sparc(speed, dt), "time_s": ep["completion_time_s"],
                     "speed": speed, "dt": dt})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Pages                                                                       #
# --------------------------------------------------------------------------- #

def _fmt(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"


def page_summary(pdf: PdfPages, runs: list[Run]) -> None:
    fig = plt.figure(figsize=PAGE)
    fig.text(0.07, 0.95, "MetaFine results", fontsize=20, weight="bold")
    fig.text(0.07, 0.925, f"Generated {datetime.now():%Y-%m-%d %H:%M}", fontsize=9, color="#555")

    y = 0.88
    fig.text(0.07, y, "Runs", fontsize=12, weight="bold")
    for run in runs:
        y -= 0.022
        info = run.info
        policy = info["policy"].get("path") or info["policy"].get("url") or "?"
        line = (f"■ {run.path.name}   task={run.task}   episodes={len(run.episodes)}   "
                f"status={info.get('status')}   git={info.get('git', {}).get('commit', '')[:8]}"
                f"{'+dirty' if info.get('git', {}).get('dirty') else ''}")
        fig.text(0.07, y, line, fontsize=8.5, color=run.color)
        for extra in [f"policy: {policy}", f"note: {info.get('note', '')}"]:
            for wrapped in textwrap.wrap(extra, 110):
                y -= 0.017
                fig.text(0.095, y, wrapped, fontsize=7.5, color="#444")

    y -= 0.05
    fig.text(0.07, y, "Scores on the five graded tasks", fontsize=12, weight="bold")
    columns = ["Task", "Policy", "Perception\nAUSC", "Understanding\nSR", "Behavior"]
    cells, colors = [], []
    for tid, task in TASKS:
        ref = headline(load_reference(task))
        cells.append([f"{tid} {task}", REF_LABEL, _fmt(ref["perception"]),
                      _fmt(ref["understanding"]), _fmt(ref["behavior"])])
        colors.append(REF_COLOR)
        for run in runs:
            if run.task == task:
                h = headline(run.report)
                cells.append(["", run.path.name, _fmt(h["perception"]),
                               _fmt(h["understanding"]), _fmt(h["behavior"])])
                colors.append(run.color)
        if not any(run.task == task for run in runs):
            cells.append(["", "our runs", "not evaluated", "not evaluated", "not evaluated"])
            colors.append("#999999")
    height = 0.024 * (len(cells) + 2)
    ax = fig.add_axes([0.07, y - 0.015 - height, 0.86, height])
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=columns, loc="upper left", cellLoc="left",
                     colWidths=[0.27, 0.33, 0.13, 0.15, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(cell.get_height() * 1.6)
        elif c > 0:
            cell.get_text().set_color(colors[r - 1])

    notes = [
        "Behavior = mean stage progress for multi-stage tasks (T2, T4), final success rate otherwise.",
        "Perception AUSC = mean of the camera and light success-curve areas (clean + 3 levels each).",
        REF_NOTE,
    ]
    yn = y - 0.05 - height
    for note in notes:
        for wrapped in textwrap.wrap(note, 120):
            fig.text(0.07, yn, wrapped, fontsize=7.5, color="#555")
            yn -= 0.016
    pdf.savefig(fig)
    plt.close(fig)


# What "success" means per task, and how T3 outcomes read; other tasks use generic wording.
GOAL = {"toggle_switch_table": "toggled the instructed switch"}
OUTCOME_VERB = {"toggle_switch_table": "toggled"}
OUTCOME_MEANING = {
    "toggle_switch_table": {"correct": "at episode end only the instructed switch was flipped",
                            "wrong": "at episode end only the wrong switch was flipped",
                            "both": "at episode end both switches were flipped",
                            "none": "at episode end neither switch was flipped"},
}
# Count rows are end-of-episode outcomes, not the success check (which can add contact/hold rules).
COUNT_NOTE = " (position only; success also needs the official success check)"
LEVEL_WORD = {"l1": "mild", "l2": "medium", "l3": "strong"}
NO_REF = "Can't tell, no 30K value"


def _num(v) -> float | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def _verdict_rate(ref, ours) -> str:
    """Higher is better; any drop or tie is a No."""
    ref, ours = _num(ref), _num(ours)
    if ref is None:
        return NO_REF
    if ours is None:
        return "Can't tell, not measured in this run"
    d = round(round(ours, 2) - round(ref, 2), 2)  # match the rounded values in the table
    if d == 0:
        return "No, same"
    return f"Yes, {d:.2f} higher" if d > 0 else f"No, {-d:.2f} lower"


def _verdict_count(kind: str, variant: str, ref: float, ours: float, sr_ref, sr_ours,
                   none_ref: float, none_ours: float) -> str:
    """Compare outcome shares (runs can have different episode counts). A share only improves
    if it moved the good way AND the robot did not just act less (more "none") or succeed less
    (lower success rate for that instruction)."""
    d = round(round(ours, 2) - round(ref, 2), 2)
    if d == 0:
        return "No, same"
    good = d > 0 if kind == "correct" else d < 0
    word = "higher" if d > 0 else "lower"
    if not good:
        return f"No, {abs(d):.2f} {word}"
    sr_fell = (_num(sr_ref) is not None and _num(sr_ours) is not None
               and round(sr_ours, 2) < round(sr_ref, 2))
    moved_less = kind in ("wrong", "both") and round(none_ours, 2) > round(none_ref, 2)
    if moved_less and sr_fell:
        return (f'No, {abs(d):.2f} {word}, but the robot moved less ("none" {none_ref:.2f} → '
                f'{none_ours:.2f}) and the {variant} success rate fell from {sr_ref:.2f} to {sr_ours:.2f}')
    if moved_less:
        return (f'No, it\'s lower because the robot moved less '
                f'("none" {none_ref:.2f} → {none_ours:.2f})')
    if sr_fell:
        return (f"No, {abs(d):.2f} {word} by end position, but the "
                f"{variant} success rate fell from {sr_ref:.2f} to {sr_ours:.2f}")
    return f"Yes, {abs(d):.2f} {word}"


def score_rows(run: Run) -> list[dict]:
    """Every score for one run: name, organizers' 30K value, this run, meaning, improved?"""
    task, tid = run.task, dict((t, i) for i, t in TASKS)[run.task]
    ref_m = load_reference(task).get("metrics") or {}
    our_m = run.report.get("metrics") or {}
    rp, op = ref_m.get("perception") or {}, our_m.get("perception") or {}
    ru, ou = ref_m.get("understanding") or {}, our_m.get("understanding") or {}
    rb, ob = ref_m.get("behavior") or {}, our_m.get("behavior") or {}
    rows: list[dict] = []

    def rate(name, ref, ours, meaning, graded=False):
        rows.append({"name": name, "ref": _fmt(ref), "ours": _fmt(ours), "meaning": meaning,
                     "verdict": _verdict_rate(ref, ours), "graded": graded})

    # Perception
    perc = run.episodes[run.episodes["protocol"] == "perception"]
    n_scenes = int((perc["condition"] == "clean").sum()) or "the"
    rc, oc = rp.get("curves") or {}, op.get("curves") or {}
    rate("Clean success rate", rp.get("clean_success_rate"), op.get("clean_success_rate"),
         f"Fraction of {n_scenes} normal scenes where the robot "
         f"{GOAL.get(task, 'completed the instruction')}")
    for axis, label, what in [("camera", "Camera jitter", "camera position/angle shift"),
                              ("light", "Lighting", "lighting change")]:
        for level in ("l1", "l2", "l3"):
            key = f"{'cam' if axis == 'camera' else 'light'}_{level}"
            word = LEVEL_WORD[level]
            desc = what if level == "l1" or axis == "light" else "camera shift"
            rate(f"{label} {level.upper()}", (rc.get(axis) or {}).get(key),
                 (oc.get(axis) or {}).get(key), f"Success rate with {word} {desc}")
    rate("Perception AUSC (camera)", rp.get("ausc_camera"), op.get("ausc_camera"),
         "Area under the clean→L3 camera curve; how well success holds up as the camera shifts")
    rate("Perception AUSC (light)", rp.get("ausc_light"), op.get("ausc_light"),
         "Area under the clean→L3 lighting curve; how well success holds up as lighting changes")
    rate("Perception AUSC (graded)", rp.get("ausc_mean"), op.get("ausc_mean"),
         "Average of the two AUSCs; the official Perception score", graded=True)

    # Understanding
    variants = list((ou.get("per_variant_success_rate") or ru.get("per_variant_success_rate")
                     or {}).keys())
    n_und = len(run.episodes[run.episodes["protocol"] == "understanding"]) or "all"
    quoted = ", ".join(f'"{v}"' for v in variants)
    rate("Understanding SR (graded)", ru.get("success_rate"), ou.get("success_rate"),
         f"Success rate when each scene is run with every instruction ({quoted}); "
         f"{n_und} episodes", graded=True)
    rsr, osr = ru.get("per_variant_success_rate") or {}, ou.get("per_variant_success_rate") or {}
    for v in variants:
        rate(f'Understanding SR, "{v}"', rsr.get(v), osr.get(v),
             f'Success rate when the instruction is "{v}"')

    # Confusion counts: what the robot did for each instruction
    rconf, oconf = ru.get("confusion") or {}, ou.get("confusion") or {}
    verb = OUTCOME_VERB.get(task, "outcome")
    words = OUTCOME_MEANING.get(task, {})
    for v in variants:
        r, o = rconf.get(v) or {}, oconf.get(v) or {}
        outcomes = [v] + [x for x in variants if x != v]
        outcomes += [x for x in sorted(set(r) | set(o)) if x not in outcomes and x != "none"]
        outcomes += ["none"]
        for out in outcomes:
            kind = ("correct" if out == v else "none" if out == "none"
                    else "both" if out == "both" else "wrong")
            meaning = words.get(kind) or {
                "correct": "outcome matched the instruction",
                "wrong": f'outcome was "{out}" (a different target)',
                "both": f'outcome was "{out}"', "none": "no target reached"}[kind]
            row = {"name": f"Told {v} → {verb} {out}", "graded": False,
                   "meaning": "Share of episodes: " + meaning
                   + (COUNT_NOTE if kind in ("correct", "both") else "")}
            n_r, n_o = sum(r.values()), sum(o.values())
            share = lambda c, n: f"{c}/{n} ({c / n:.2f})" if n else "—"
            if not rconf or not n_r:
                row.update(ref="not in JSON", ours=share(int(o.get(out, 0)), n_o), verdict=NO_REF)
            elif not n_o:
                row.update(ref=share(int(r.get(out, 0)), n_r), ours="—",
                           verdict="Can't tell, not measured in this run")
            else:
                ref_c, our_c = int(r.get(out, 0)), int(o.get(out, 0))
                row.update(ref=share(ref_c, n_r), ours=share(our_c, n_o), verdict=_verdict_count(
                    kind, v, ref_c / n_r, our_c / n_o, rsr.get(v), osr.get(v),
                    r.get("none", 0) / n_r, o.get("none", 0) / n_o))
            rows.append(row)

    # Behavior
    if rb.get("mean_stage_fraction") is not None or ob.get("mean_stage_fraction") is not None:
        rate("Behavior success rate", rb.get("success_rate"), ob.get("success_rate"),
             "Fraction of episodes that finished every stage of the task")
        rate("Behavior stage progress (graded)", rb.get("mean_stage_fraction"),
             ob.get("mean_stage_fraction"),
             "Average fraction of task stages completed per episode; the official Behavior score",
             graded=True)
    else:
        same = (_num(ob.get("success_rate")) is not None
                and _num(ob.get("success_rate")) == _num(ou.get("success_rate")))
        rate("Behavior (graded)", rb.get("success_rate"), ob.get("success_rate"),
             f"Final task success rate ({tid} is single-stage"
             + (", so equals Understanding SR)" if same else ")"), graded=True)

    # Motion (proposal metrics; the organizers' JSON has none)
    tab = motion_table(run)
    n_ok = len(tab)
    med = (lambda c: tab[c].median()) if n_ok else (lambda c: None)
    for name, col, unit, meaning in [
        ("Log dimensionless jerk", "ldlj", "",
         "Smoothness of the gripper path on successful clean episodes; closer to 0 = smoother"),
        ("SPARC", "sparc", "",
         "Smoothness from the speed profile's frequency content; closer to 0 = smoother"),
        ("Completion time", "time_s", " s", "Median time to finish on successful clean episodes"),
    ]:
        v = med(col)
        rows.append({"name": name, "ref": "not in JSON",
                     "ours": "no successes" if v is None else f"{v:.2f}{unit}",
                     "meaning": meaning + (f" (n={n_ok})" if n_ok else ""),
                     "verdict": NO_REF, "graded": False})
    return rows


def page_score_table(pdf: PdfPages, run: Run) -> None:
    """Front page: every score for this run, the organizers' 30K value, and whether it improved."""
    tid = dict((t, i) for i, t in TASKS)[run.task]
    rows = score_rows(run)
    columns = ["Score", "30K JSON\n(organizers)", "This run", "What it measures", "Improved?"]
    widths = [0.19, 0.11, 0.11, 0.32, 0.27]
    wrap = [24, 14, 14, 42, 36]
    lines_per_page = 54
    pages, current, used = [], [], 0
    for row in rows:
        cells = [row["name"], row["ref"], row["ours"], row["meaning"], row["verdict"]]
        cells = ["\n".join(textwrap.wrap(c, w)) or c for c, w in zip(cells, wrap)]
        n_lines = max(c.count("\n") + 1 for c in cells)
        if used + n_lines > lines_per_page and current:
            pages.append(current)
            current, used = [], 0
        current.append((cells, n_lines, row["graded"]))
        used += n_lines
    pages.append(current)

    for page_i, chunk in enumerate(pages):
        fig = plt.figure(figsize=PAGE)
        cont = " (continued)" if page_i else ""
        fig.text(0.05, 0.965, f"{tid} {run.task}: every score{cont}", fontsize=14, weight="bold")
        fig.text(0.05, 0.945, f"This run: {run.label}", fontsize=8, color="#444")
        line_h = 0.0128
        header_h = 2 * line_h + 0.006
        total_h = header_h + sum(n * line_h + 0.004 for _, n, _ in chunk)
        ax = fig.add_axes([0.05, 0.93 - total_h, 0.90, total_h])
        ax.axis("off")
        table = ax.table(cellText=[c for c, _, _ in chunk], colLabels=columns,
                         colWidths=widths, loc="upper left", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#cccccc")
            cell.PAD = 0.03
            if r == 0:
                cell.set_height(header_h / total_h)
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f3f4f6")
            else:
                _, n, graded = chunk[r - 1]
                cell.set_height((n * line_h + 0.004) / total_h)
                text = cell.get_text()
                if graded:
                    text.set_weight("bold")
                if c == 4:
                    verdict = text.get_text()
                    text.set_color("#15803d" if verdict.startswith("Yes")
                                   else "#b91c1c" if verdict.startswith("No") else "#555555")
        pdf.savefig(fig)
        plt.close(fig)


def _curve_with_ci(ax, run: Run, levels: list[str]) -> None:
    perc = run.episodes[run.episodes["protocol"] == "perception"]
    rates, lo, hi = [], [], []
    for level in levels:
        sub = perc[perc["condition"] == level]
        k, n = int(sub["success"].sum()), len(sub)
        rates.append(k / n if n else np.nan)
        a, b = wilson(k, n)
        lo.append(a)
        hi.append(b)
    x = np.arange(len(levels))
    ax.plot(x, rates, "o-", color=run.color, label=run.path.name)
    ax.fill_between(x, lo, hi, color=run.color, alpha=0.12)


def page_task(pdf: PdfPages, task_label: str, task: str, runs: list[Run]) -> None:
    ref = (load_reference(task).get("metrics") or {})
    fig, axes = plt.subplots(3, 2, figsize=PAGE)
    fig.suptitle(f"{task_label} {task}: what the scores are made of", fontsize=14,
                 weight="bold", x=0.07, ha="left")

    for ax, levels, key, title in [
        (axes[0, 0], CAM_LEVELS, "camera", "Perception: camera jitter"),
        (axes[0, 1], LIGHT_LEVELS, "light", "Perception: lighting change"),
    ]:
        curve = ((ref.get("perception") or {}).get("curves") or {}).get(key) or {}
        if curve:
            ax.plot(range(len(levels)), [curve.get(lv, np.nan) for lv in levels], "s--",
                    color=REF_COLOR, label=REF_LABEL)
        for run in runs:
            _curve_with_ci(ax, run, levels)
        ax.set_xticks(range(len(levels)), levels)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("success rate")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="lower left")

    ax = axes[1, 0]
    ref_variants = (ref.get("understanding") or {}).get("per_variant_success_rate") or {}
    variants = list(ref_variants) or sorted(runs[0].episodes["variant"].dropna().unique())
    width = 0.8 / (len(runs) + 1)
    x = np.arange(len(variants))
    ax.bar(x, [ref_variants.get(v, np.nan) for v in variants], width, color=REF_COLOR,
           label=REF_LABEL)
    for i, run in enumerate(runs, start=1):
        und = run.episodes[run.episodes["protocol"] == "understanding"]
        rates, errs = [], [[], []]
        for v in variants:
            sub = und[und["variant"] == v]
            k, n = int(sub["success"].sum()), len(sub)
            p = k / n if n else np.nan
            a, b = wilson(k, n)
            rates.append(p)
            errs[0].append(p - a if n else 0)
            errs[1].append(b - p if n else 0)
        ax.bar(x + i * width, rates, width, yerr=errs, capsize=2, color=run.color,
               label=run.path.name)
    ax.set_xticks(x + width * len(runs) / 2, variants)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title("Understanding: success per instruction", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    confusion = runs[0].understanding.get("confusion") or {}
    if confusion:
        rows = list(confusion)
        # Instructed variants first so correct outcomes sit on the diagonal.
        outcomes = rows + sorted({o for row in confusion.values() for o in row} - set(rows))
        mat = np.array([[confusion[r].get(o, 0) for o in outcomes] for r in rows])
        ax.imshow(mat, cmap="Blues")
        for (i, j), val in np.ndenumerate(mat):
            ax.text(j, i, str(val), ha="center", va="center", fontsize=9,
                    color="white" if val > mat.max() / 2 else "black")
        ax.set_xticks(range(len(outcomes)), outcomes)
        ax.set_yticks(range(len(rows)), rows)
        ax.set_xlabel("what the robot actually did")
        ax.set_ylabel("instructed")
        ax.set_title(f"Understanding confusion ({runs[0].path.name})", fontsize=10, loc="left")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No confusion matrix for this task", ha="center", va="center")

    ax = axes[2, 0]
    for run in runs:
        clean = run.episodes[run.episodes["condition"] == "clean"]
        for ok, style in [(True, "-"), (False, ":")]:
            t = clean[clean["success"] == ok]["completion_time_s"]
            if len(t):
                ax.hist(t, bins=15, histtype="step", linestyle=style, color=run.color,
                        label=f"{run.path.name} {'success' if ok else 'failure'} (n={len(t)})")
    ax.set_xlabel("episode time (s); failures run to the time limit")
    ax.set_ylabel("episodes")
    ax.set_title("Episode time, clean scenes", fontsize=10, loc="left")
    ax.legend(fontsize=6)

    ax = axes[2, 1]
    ax.axis("off")
    lines = []
    for run in runs:
        e = run.episodes
        lines.append(f"{run.path.name}")
        for proto in ["understanding", "perception"]:
            sub = e[e["protocol"] == proto]
            if len(sub):
                k, n = int(sub["success"].sum()), len(sub)
                a, b = wilson(k, n)
                lines.append(f"  {proto}: {k}/{n} = {k / n:.2f}  (95% CI {a:.2f}–{b:.2f})")
    lines += ["", *textwrap.wrap(REF_NOTE, 60)]
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=8, family="monospace")

    fig.tight_layout(rect=(0.03, 0.02, 0.97, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def page_motion(pdf: PdfPages, task_label: str, task: str, runs: list[Run]) -> None:
    tables = {run.path.name: motion_table(run) for run in runs}
    fig, axes = plt.subplots(3, 2, figsize=PAGE)
    fig.suptitle(f"{task_label} {task}: motion quality (end-effector path)", fontsize=14,
                 weight="bold", x=0.07, ha="left")

    for ax, metric, title in [
        (axes[0, 0], "ldlj", "Log dimensionless jerk (closer to 0 = smoother)"),
        (axes[0, 1], "sparc", "SPARC (closer to 0 = smoother)"),
        (axes[1, 0], "time_s", "Completion time, successful episodes (s)"),
    ]:
        data = [tables[r.path.name][metric].dropna() if len(tables[r.path.name]) else []
                for r in runs]
        if any(len(d) for d in data):
            parts = ax.boxplot(data, patch_artist=True, widths=0.5)
            for patch, run in zip(parts["boxes"], runs):
                patch.set_facecolor(run.color)
                patch.set_alpha(0.35)
            ax.set_xticks(range(1, len(runs) + 1),
                          [f"{r.path.name}\n(n={len(d)})" for r, d in zip(runs, data)], fontsize=7)
        else:
            ax.text(0.5, 0.5, "no successful clean episodes", ha="center", va="center")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    for run in runs:
        tab = tables[run.path.name]
        for i, (_, row) in enumerate(tab.head(5).iterrows()):
            t = np.arange(len(row["speed"])) * row["dt"]
            ax.plot(t, row["speed"], color=run.color, alpha=0.6, lw=1,
                    label=run.path.name if i == 0 else None)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("end-effector speed (m/s)")
    ax.set_title("Speed profiles (first 5 successful episodes)", fontsize=10, loc="left")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)

    ax = axes[2, 0]
    ax.axis("off")
    lines = ["Medians over successful clean episodes"]
    for run in runs:
        tab = tables[run.path.name]
        if len(tab):
            lines.append(f"{run.path.name}: LDLJ {tab['ldlj'].median():.2f}  "
                         f"SPARC {tab['sparc'].median():.2f}  time {tab['time_s'].median():.2f}s"
                         f"  (n={len(tab)})")
        else:
            lines.append(f"{run.path.name}: no successful clean episodes")
    if len(runs) >= 2:
        a, b = (tables[r.path.name] for r in runs[:2])
        if len(a) and len(b):
            both = a.merge(b, on="episode_id", suffixes=("_a", "_b"))
            lines += ["", f"Paired ({runs[0].path.name} vs {runs[1].path.name}):",
                      f"  episodes both solved: {len(both)}"]
            if len(both):
                lines.append(f"  median LDLJ difference (b - a): "
                             f"{(both['ldlj_b'] - both['ldlj_a']).median():+.2f}")
                lines.append(f"  median time difference (b - a): "
                             f"{(both['time_s_b'] - both['time_s_a']).median():+.2f}s")
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=8, family="monospace")

    ax = axes[2, 1]
    ax.axis("off")
    note = ("Computed on the whole episode from the end-effector position, sampled at the "
            "control rate. Stage-specific values (approach / transport) need stage labels, "
            "which are not implemented yet. The organizers did not publish trajectories, so "
            "there is no reference motion here; add an expert-planner run for that.")
    ax.text(0, 1, "\n".join(textwrap.wrap(note, 60)), va="top", fontsize=8, color="#555")

    fig.tight_layout(rect=(0.03, 0.02, 0.97, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def make_report(run_dirs: list[Path], out: Path) -> Path:
    runs = [Run(p, i) for i, p in enumerate(run_dirs)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for run in runs:
            page_score_table(pdf, run)
        page_summary(pdf, runs)
        for tid, task in TASKS:
            task_runs = [r for r in runs if r.task == task]
            if task_runs:
                page_task(pdf, tid, task, task_runs)
                page_motion(pdf, tid, task, task_runs)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="run folders from tools/run_eval.py")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out
    if out is None:
        out = (args.runs[0] / "report.pdf" if len(args.runs) == 1
               else _REPO / "reports" / f"{datetime.now():%Y-%m-%d_%H%M%S}.pdf")
    print(f"wrote {make_report(args.runs, out)}")


if __name__ == "__main__":
    main()
