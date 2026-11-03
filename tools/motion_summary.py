#!/usr/bin/env python3
"""Windowed motion statistics of the CoRL demos by instruction variant and gripper stage (issue #20).

Reads the same LeRobot files as ``tools/build_index.py`` (``meta/info.json``, the tasks table
and ``data/chunk-*/*.parquet``; no videos). For every episode the 7 arm-joint columns of
``action`` (pd_joint_delta_pos: 7 joint deltas + 1 gripper) are cut into windows of
``--window`` steps (default 50, one pi0 action chunk = 2.5 s) with stride ``--stride``
(default 10). Per window, per joint and then averaged over joints:

* action change: mean |a[t+1] - a[t]| (first difference);
* roughness: mean |a[t+2] - 2 a[t+1] + a[t]| (second difference, the proxy behind the
  "2.4x smoother" number in Josh's update);
* band fractions: share of the rFFT energy of the raw window in 0-1, 1-3, 3-6 and 6-10 Hz at
  fs = 20 Hz (control runs at 20 Hz, dt = 0.05 s; the ``fps: 30`` in info.json is a label).
  The DC bin counts towards 0-1 Hz; joints with no energy in the window are left out of the
  average. With fs = 20 the four bands cover every bin up to Nyquist, so they sum to 1;
* mad (CSV only): mean over steps of the L2 norm across joints of the first difference divided
  by dt, the ``compute_mad`` of ``utils/util.py``.

Action units are the normalized joint deltas of pd_joint_delta_pos (1.0 = 0.1 rad), so every
amplitude metric is in those units.

Stage proxy from the gripper: the last two ``observation.state`` columns are the finger joints
(about 0.04 m when open, near 0 when closed). A frame is closed when their mean is below
``--closed-thresh`` (0.02 m); a window is "closed" (transport / contact) when more than half
of its frames are closed, otherwise "open" (approach / retreat).

Outputs:

* ``docs/MOTION_SUMMARY.md``: one table per task (rows variant x stage plus an "all" row per
  variant; columns: windows, episodes, mean (median) of each metric) and Kruskal-Wallis
  p-values (scipy.stats.kruskal) across variants within each stage and across stages within
  each variant. Windows overlap, so the tests use one value per episode (the mean over its
  windows of that stage) unless ``--kw-unit window`` is given. As issue #20 asks, the table and
  the tests use the training episodes of ``data/split_v1.json`` when that file exists
  (``--split-json``, ``--subset train|val|all``);
* ``data/motion_summary.csv``: one row per window with its tags (including ``split``) and
  metrics, always every window.

Laptop commands (meta/ and data/ of the two tasks only, a few MB, no videos):

    huggingface-cli download hiangx/MetaFine_CoRL26 --repo-type dataset --local-dir demos/CoRL \\
        --include "toggle_switch_table/mixed/lerobot/meta/*" "toggle_switch_table/mixed/lerobot/data/*" \\
                  "grasp_move_mug/mixed/lerobot/meta/*" "grasp_move_mug/mixed/lerobot/data/*"
    python tools/motion_summary.py
    python tools/motion_summary.py --tasks toggle_switch_table --window 50 --stride 10 --fs 20 \\
        --out-dir docs --csv data/motion_summary.csv

Only numpy, pandas, pyarrow and scipy are needed. Developed against a synthetic LeRobot v3
dataset; the points marked "verify on the real files" were not checked on the download.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from build_index import (  # noqa: E402
    CONTROL_HZ,
    DEFAULT_TASKS,
    build_episode_index,
    display_path,
    lerobot_dir,
    load_info,
    load_variants,
    read_frames,
    stack_list_column,
    variant_order,
)

N_ARM_JOINTS = 7  # action = 7 joint deltas + 1 gripper; verify on the real files
N_STATE_DIMS = 9  # observation.state = 7 arm joints + 2 finger joints; verify on the real files
N_FINGER_JOINTS = 2
STAGES = ("open", "closed")
BANDS = ((0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 10.0))
BAND_COLS = ("frac_0_1", "frac_1_3", "frac_3_6", "frac_6_10")
BAND_LABELS = ("0-1 Hz", "1-3 Hz", "3-6 Hz", "6-10 Hz")
METRICS = ("action_change", "roughness") + BAND_COLS
METRIC_LABELS = ("action change", "roughness") + BAND_LABELS
CSV_COLUMNS = (
    "task_id", "episode_index", "variant", "instruction", "split", "start", "end", "stage", "closed_frac",
    "mean_abs_action", "action_change", "mad", "roughness", *BAND_COLS,
)
DEFAULT_SPLIT_JSON = REPO_ROOT / "data" / "split_v1.json"


def band_fractions(win: np.ndarray, fs: float, bands: Sequence[tuple[float, float]] = BANDS, eps: float = 1e-12) -> np.ndarray:
    """Energy fractions of ``win`` (steps, joints) per band, per joint, averaged over the joints with energy.

    The fraction is band energy over the total energy up to Nyquist (DC included, counted in the
    first band). Returns NaN for every band when no joint has energy.
    """
    spec = np.abs(np.fft.rfft(win, axis=0)) ** 2  # (freqs, joints)
    freqs = np.fft.rfftfreq(win.shape[0], d=1.0 / fs)
    total = spec.sum(axis=0)
    keep = total > eps
    if not keep.any():
        return np.full(len(bands), np.nan)
    out = []
    for k, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs <= hi) if k == 0 else (freqs > lo) & (freqs <= hi)
        out.append(float((spec[mask][:, keep].sum(axis=0) / total[keep]).mean()))
    return np.asarray(out)


def window_metrics(win: np.ndarray, fs: float) -> dict[str, float]:
    """action change, mad, roughness, mean |a| and the band fractions of one window (steps, joints)."""
    d1 = np.diff(win, axis=0)
    d2 = np.diff(win, n=2, axis=0)
    out = {
        "mean_abs_action": float(np.abs(win).mean()),
        "action_change": float(np.abs(d1).mean()),
        "mad": float(np.linalg.norm(d1 * fs, ord=2, axis=1).mean()),  # utils/util.py::compute_mad, dt = 1 / fs
        "roughness": float(np.abs(d2).mean()),
    }
    out.update(zip(BAND_COLS, (float(x) for x in band_fractions(win, fs))))
    return out


def episode_windows(arm: np.ndarray, closed: np.ndarray, window: int, stride: int, fs: float) -> list[dict]:
    """Windows of one episode: ``arm`` (steps, joints) actions and ``closed`` (steps,) gripper flags."""
    rows = []
    for start in range(0, len(arm) - window + 1, stride):
        end = start + window
        closed_frac = float(np.mean(closed[start:end]))
        rows.append(
            {
                "start": start,
                "end": end,
                "stage": "closed" if closed_frac > 0.5 else "open",
                "closed_frac": closed_frac,
                **window_metrics(arm[start:end], fs),
            }
        )
    return rows


def summarise_task(
    task: str, root: Path, variants, window: int, stride: int, fs: float, closed_thresh: float, split_lookup: Mapping[int, str] | None = None
) -> tuple[pd.DataFrame, list[dict], dict]:
    """Window table, episode records and run facts (skipped episodes, warnings) of one task.

    ``split_lookup`` maps episode_index to "train" or "val"; episodes outside it (or every
    episode when it is None) get split "none".
    """
    frames = read_frames(root, ["episode_index", "task_index", "action", "observation.state"], optional=["index", "frame_index", "task"])
    records, warnings = build_episode_index(task, root, variants, frames=frames)
    by_episode = {int(r["episode_index"]): r for r in records}
    sort_col = "frame_index" if "frame_index" in frames.columns else "index"
    rows: list[dict] = []
    skipped = 0
    dims_checked = False
    for ep, g in frames.groupby("episode_index", sort=True):
        g = g.sort_values(sort_col)
        actions = stack_list_column(g["action"].tolist())
        states = stack_list_column(g["observation.state"].tolist())
        if not dims_checked:
            dims_checked = True
            if actions.shape[1] != N_ARM_JOINTS + 1 or states.shape[1] != N_STATE_DIMS:
                warnings.append(
                    f"action has {actions.shape[1]} dims and observation.state {states.shape[1]} "
                    f"(expected {N_ARM_JOINTS + 1} and {N_STATE_DIMS}); the first {N_ARM_JOINTS} action "
                    f"columns and the last {N_FINGER_JOINTS} state columns are still used"
                )
        if len(actions) < window:
            skipped += 1
            continue
        closed = states[:, -N_FINGER_JOINTS:].mean(axis=1) < closed_thresh
        rec = by_episode[int(ep)]
        split = split_lookup.get(int(ep), "none") if split_lookup is not None else "none"
        for w in episode_windows(actions[:, :N_ARM_JOINTS], closed, window, stride, fs):
            rows.append({"task_id": task, "episode_index": int(ep), "variant": rec["variant"], "instruction": rec["instruction"], "split": split, **w})
    df = pd.DataFrame(rows, columns=list(CSV_COLUMNS))
    facts = {"n_episodes": len(records), "skipped": skipped, "warnings": warnings}
    return df, records, facts


def summary_table(df: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    """Rows variant x stage (plus "all" per variant) with window and episode counts, mean and median of each metric."""
    rows = []
    for v in order:
        for stage in (*STAGES, "all"):
            sub = df[df["variant"] == v] if stage == "all" else df[(df["variant"] == v) & (df["stage"] == stage)]
            row: dict = {"variant": v, "stage": stage, "windows": int(len(sub)), "episodes": int(sub["episode_index"].nunique())}
            for m in METRICS:
                row[f"{m}_mean"] = float(sub[m].mean()) if len(sub) else float("nan")
                row[f"{m}_median"] = float(sub[m].median()) if len(sub) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def kruskal_p(groups: Sequence[np.ndarray]) -> float:
    """Kruskal-Wallis p-value over groups with at least 2 finite values each; NaN when fewer than 2 groups qualify."""
    from scipy.stats import kruskal

    clean = []
    for g in groups:
        g = np.asarray(g, dtype=float)
        g = g[np.isfinite(g)]
        if len(g) >= 2:
            clean.append(g)
    if len(clean) < 2:
        return float("nan")
    values = np.concatenate(clean)
    if np.all(values == values[0]):  # identical values: no ranks to compare
        return float("nan")
    try:
        return float(kruskal(*clean).pvalue)
    except ValueError:
        return float("nan")


def kruskal_table(df: pd.DataFrame, order: Sequence[str], unit: str = "episode") -> list[dict]:
    """Kruskal-Wallis p-values across variants within each stage and across stages within each variant.

    ``unit`` "episode" first averages the windows of each (variant, stage, episode), so overlapping
    windows of one episode count once; "window" uses every window as a sample.
    """
    if df.empty:
        return []
    data = df.groupby(["variant", "stage", "episode_index"], as_index=False)[list(METRICS)].mean() if unit == "episode" else df
    rows = []
    for stage in STAGES:
        sub = data[data["stage"] == stage]
        groups = [v for v in order if (sub["variant"] == v).any()]
        row = {"comparison": f"across variants, stage {stage}", "groups": ", ".join(groups) or "-"}
        for m in METRICS:
            row[m] = kruskal_p([sub.loc[sub["variant"] == v, m].to_numpy() for v in groups])
        rows.append(row)
    for v in order:
        sub = data[data["variant"] == v]
        groups = [s for s in STAGES if (sub["stage"] == s).any()]
        row = {"comparison": f"across stages, variant {v}", "groups": ", ".join(groups) or "-"}
        for m in METRICS:
            row[m] = kruskal_p([sub.loc[sub["stage"] == s, m].to_numpy() for s in groups])
        rows.append(row)
    return rows


def _fmt(value: float, spec: str) -> str:
    return "n/a" if value is None or not np.isfinite(value) else format(value, spec)


def md_table(header: Sequence[str], rows: Sequence[Sequence[str]], n_text_cols: int) -> str:
    sep = ["---"] * n_text_cols + ["---:"] * (len(header) - n_text_cols)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render_task(
    task: str, root: Path, info: dict, df: pd.DataFrame, sel: pd.DataFrame, subset: str, records: Sequence[dict], facts: dict,
    order: Sequence[str], table: pd.DataFrame, kw_rows: Sequence[dict], unit: str,
) -> str:
    instructions = {}
    for r in records:
        instructions.setdefault(r["variant"], r["instruction"])
    lines = [f"## {task}", ""]
    lines.append(
        f"{len(records)} episodes ({info.get('total_frames')} frames per info.json), {len(df)} windows; "
        f"{facts['skipped']} episode(s) shorter than the window skipped. Source: `{display_path(root)}`."
    )
    if subset == "all":
        lines.append("Table and tests use every window.")
    else:
        lines.append(
            f"Table and tests use the {len(sel)} {subset} windows from {sel['episode_index'].nunique()} {subset} episodes; "
            f"the CSV holds every window with its split."
        )
    lines.append("")
    lines += [f"- {v}: \"{instructions[v]}\"" for v in order if v in instructions]
    lines.append("")
    header = ["variant", "stage", "windows", "episodes", *METRIC_LABELS]
    rows = []
    for _, r in table.iterrows():
        cells = [r["variant"], r["stage"], str(int(r["windows"])), str(int(r["episodes"]))]
        for m in METRICS:
            spec = ".3f" if m in BAND_COLS else ".3g"
            cells.append(f"{_fmt(r[f'{m}_mean'], spec)} ({_fmt(r[f'{m}_median'], spec)})")
        rows.append(cells)
    lines.append(md_table(header, rows, 2))
    lines.append("")
    unit_text = "one value per episode, the mean over its windows of that stage" if unit == "episode" else "every window is one sample (windows overlap, so these are optimistic)"
    lines.append(f"Kruskal-Wallis p-values ({unit_text}):")
    lines.append("")
    header = ["comparison", "groups", *METRIC_LABELS]
    rows = [[r["comparison"], r["groups"], *[_fmt(r[m], ".2g") for m in METRICS]] for r in kw_rows]
    lines.append(md_table(header, rows, 2))
    if facts["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        lines += [f"- {w}" for w in facts["warnings"]]
    lines.append("")
    return "\n".join(lines)


def render_document(sections: Sequence[str], args: argparse.Namespace, split_path: Path | None, subset: str) -> str:
    cmd = "python tools/motion_summary.py " + " ".join(
        [
            f"--demos-root {display_path(args.demos_root)}",
            "--tasks " + " ".join(args.tasks),
            f"--window {args.window}",
            f"--stride {args.stride}",
            f"--fs {args.fs:g}",
            f"--closed-thresh {args.closed_thresh:g}",
            f"--kw-unit {args.kw_unit}",
            f"--subset {subset}",
        ]
        + ([f"--split-json {display_path(split_path)}"] if split_path else [])
    )
    subset_text = (
        f"Tables and tests use the {subset} episodes of `{display_path(split_path)}` (whole-episode split, see tools/build_index.py)."
        if split_path and subset != "all"
        else "Tables and tests use every window (no split applied)."
    )
    head = [
        "# Motion summary of the CoRL demos",
        "",
        f"Generated by `{cmd}` from the `action` and `observation.state` columns of the LeRobot frame tables (no videos).",
        "",
        subset_text,
        "",
        f"Settings: window {args.window} steps ({args.window / args.fs:.2f} s), stride {args.stride}, fs {args.fs:g} Hz "
        f"(dt {1.0 / args.fs:.3f} s; the fps label in info.json is not used), gripper closed when the mean of the two "
        f"finger joints is below {args.closed_thresh:g} m, window stage by majority of its frames.",
        "",
        "Metrics per window over the 7 arm-joint columns of `action` (pd_joint_delta_pos, normalized joint deltas with "
        "1.0 = 0.1 rad), cells show mean (median) over windows:",
        "",
        "- action change: mean |a[t+1] - a[t]| (first difference)",
        "- roughness: mean |a[t+2] - 2 a[t+1] + a[t]| (second difference, the proxy behind the \"2.4x smoother\" number)",
        "- band fractions: share of the rFFT energy of the raw window in 0-1, 1-3, 3-6 and 6-10 Hz, per joint then averaged "
        "over the joints with energy; the DC bin counts towards 0-1 Hz, so a window of steady motion scores close to 1 there",
        "",
        "Stages: \"open\" = approach / retreat (fingers apart), \"closed\" = transport / contact (fingers together), \"all\" = every window of the variant.",
        "",
    ]
    notes = [
        "## Notes",
        "",
        "- Windows overlap (stride smaller than the window) and several windows come from one episode, so window-level tests "
        "overstate significance; the p-values above use one value per episode unless `--kw-unit window` was given.",
        "- The stage proxy is the gripper only: a task whose gripper never closes (or never opens) has one empty stage row, "
        "and the across-stage test for that variant is n/a. Verify on the real files that the finger joints read about 0.04 m "
        "when open and near 0 when closed.",
        "- Kruskal-Wallis compares distributions, not means; a small p-value with similar medians means a shape difference.",
        "- `data/motion_summary.csv` holds every window (" + ", ".join(CSV_COLUMNS) + ") for other cuts; `mad` is the "
        "mean L2 norm of the first difference over dt (`utils/util.py::compute_mad`), `split` is train / val from the split "
        "file or none.",
        "",
    ]
    return "\n".join(head + list(sections) + notes)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos-root", type=Path, default=REPO_ROOT / "demos" / "CoRL", help="demo root (default: demos/CoRL)")
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), help="task ids (default: %(default)s)")
    p.add_argument("--window", type=int, default=50, help="window length in steps (default: 50, one pi0 chunk)")
    p.add_argument("--stride", type=int, default=10, help="window stride in steps (default: 10)")
    p.add_argument("--fs", type=float, default=float(CONTROL_HZ), help="control rate in Hz for the spectra (default: 20)")
    p.add_argument("--closed-thresh", type=float, default=0.02, help="gripper closed when the mean finger joint is below this (default: 0.02 m)")
    p.add_argument("--kw-unit", choices=("episode", "window"), default="episode", help="sample unit of the Kruskal-Wallis tests (default: episode)")
    p.add_argument(
        "--split-json", type=Path, default=None,
        help="split file from tools/build_index.py (default: data/split_v1.json when it exists, else no split)",
    )
    p.add_argument(
        "--subset", choices=("train", "val", "all"), default=None,
        help="episodes used for the tables and tests (default: train when a split file is available, else all); the CSV always has every window",
    )
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs", help="where MOTION_SUMMARY.md goes (default: docs/)")
    p.add_argument("--csv", type=Path, default=REPO_ROOT / "data" / "motion_summary.csv", help="per-window CSV (default: data/motion_summary.csv)")
    args = p.parse_args(argv)
    if args.window < 4:
        p.error("--window must be at least 4 steps (second differences and a spectrum need it)")
    if args.stride < 1:
        p.error("--stride must be at least 1")
    if args.fs <= 0:
        p.error("--fs must be positive")
    if args.split_json is not None and not args.split_json.exists():
        p.error(f"--split-json {args.split_json} does not exist (run tools/build_index.py first)")
    split_path = args.split_json if args.split_json is not None else (DEFAULT_SPLIT_JSON if DEFAULT_SPLIT_JSON.exists() else None)
    subset = args.subset or ("train" if split_path else "all")
    if subset != "all" and split_path is None:
        p.error(f"--subset {subset} needs a split file (--split-json, or data/split_v1.json from tools/build_index.py)")
    split_tasks = json.loads(split_path.read_text(encoding="utf-8"))["tasks"] if split_path else {}
    if split_path:
        print(f"split: {split_path} ({subset} episodes for the tables and tests)")
    else:
        print("split: none (every window is used; run tools/build_index.py to get data/split_v1.json)")

    sections: list[str] = []
    tables: list[pd.DataFrame] = []
    for task in args.tasks:
        root = lerobot_dir(args.demos_root, task)
        if not (root / "meta" / "info.json").exists():
            print(f"[FAIL] {task}: {root / 'meta' / 'info.json'} missing (download meta/* and data/* first)")
            return 2
        info = load_info(root)
        variants = load_variants(task)
        lookup: dict[int, str] | None = None
        task_subset = subset
        if split_path:
            if task in split_tasks:
                lookup = {int(e): "train" for e in split_tasks[task].get("train", [])}
                lookup.update({int(e): "val" for e in split_tasks[task].get("val", [])})
            else:
                print(f"[WARN] {task}: not in {split_path}; every window is used for this task")
                task_subset = "all"
        df, records, facts = summarise_task(task, root, variants, args.window, args.stride, args.fs, args.closed_thresh, lookup)
        sel = df if task_subset == "all" else df[df["split"] == task_subset]
        order = variant_order(records, variants)
        table = summary_table(sel, order)
        kw_rows = kruskal_table(sel, order, args.kw_unit)
        sections.append(render_task(task, root, info, df, sel, task_subset, records, facts, order, table, kw_rows, args.kw_unit))
        tables.append(df)

        print(
            f"{task}: {len(records)} episodes, {len(df)} windows ({len(sel)} in the {task_subset} subset), "
            f"{facts['skipped']} skipped (shorter than {args.window})"
        )
        print(f"  {'variant':<10} {'stage':<7} {'windows':>7} {'episodes':>8} {'change':>9} {'rough':>9} {'0-1':>6} {'1-3':>6} {'3-6':>6} {'6-10':>6}")
        for _, r in table.iterrows():
            print(
                f"  {r['variant']:<10} {r['stage']:<7} {int(r['windows']):>7} {int(r['episodes']):>8} "
                f"{_fmt(r['action_change_mean'], '.3g'):>9} {_fmt(r['roughness_mean'], '.3g'):>9} "
                + " ".join(f"{_fmt(r[f'{b}_mean'], '.3f'):>6}" for b in BAND_COLS)
            )
        for w in facts["warnings"]:
            print(f"    ! {w}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    md_path = args.out_dir / "MOTION_SUMMARY.md"
    md_path.write_text(render_document(sections, args, split_path, subset), encoding="utf-8")
    pd.concat(tables, ignore_index=True).to_csv(args.csv, index=False, float_format="%.6g")
    print(f"wrote {md_path} and {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
