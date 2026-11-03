#!/usr/bin/env python3
"""Build the episode index and the train/val split of the CoRL demos (issue #16).

For each task (default: T3 ``toggle_switch_table`` and T2 ``grasp_move_mug``) the script
reads ``<demos-root>/<task>/mixed/lerobot/``:

* ``meta/info.json``: totals, fps label and codebase version;
* ``meta/tasks.parquet`` (LeRobot v3: the instruction is the pandas index or a ``task``
  column, ``task_index`` is a column) or ``meta/tasks.jsonl`` (LeRobot v2.x):
  ``task_index`` to instruction string;
* ``data/chunk-*/*.parquet``: the frame table (``episode_index``, ``index``, ``task_index``
  and, when present, ``task``). ``meta/episodes`` is only used to cross-check lengths.

Videos are not needed. Every episode becomes one record carrying its instruction string
and the variant name (red / blue, left / right / forward) looked up in
``eval/configs/<task>.yaml``; a task without a config gets the first colour or direction
word of its instruction.

Outputs (default ``data/``):

* ``index.json``: ``{"episodes": [...], "tasks": {...}}`` with one record per episode:
  task_id, episode_index, task_index, instruction, variant, n_frames, first_index,
  last_index (global frame indices, inclusive) and data_file;
* ``split_v1.json``: 85/15 train/val split of whole episodes, stratified by variant,
  drawn with ``numpy.random.default_rng(seed)`` (seed 0, one generator per task),
  listing episode_index per task and split.

Episodes are keyed by the LeRobot episode_index, never by the planner seed: the T3 blue
replay JSON carries a duplicated seed (docs/SETUP.md), so seed-keyed joins are ambiguous.
Not in this index: the replay h5 trajectory key and ``episode_seed`` asked for in issue #16.
They need the replay JSON and h5, which this download skips, plus the mixed episode order
(red 0-99 then blue 100-199 is likely but unverified) and the first-frame qpos check against
``obs/agent/qpos``; verify on the real files before adding that join.

Laptop commands (meta/ and data/ of the two tasks only, a few MB, no videos):

    huggingface-cli download hiangx/MetaFine_CoRL26 --repo-type dataset --local-dir demos/CoRL \\
        --include "toggle_switch_table/mixed/lerobot/meta/*" "toggle_switch_table/mixed/lerobot/data/*" \\
                  "grasp_move_mug/mixed/lerobot/meta/*" "grasp_move_mug/mixed/lerobot/data/*"
    python tools/build_index.py
    python tools/build_index.py --tasks toggle_switch_table --val-frac 0.15 --seed 0 --out-dir data

``hf download`` takes the same arguments on recent huggingface_hub releases. Only numpy,
pandas and pyarrow are needed. The script was developed against a synthetic LeRobot v3
dataset; the points marked "verify on the real files" were not checked on the download.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from check_demos import CONFIG_DIR, EPISODES_PER_VARIANT, EXPECTED, parse_task_config  # noqa: E402

DEFAULT_TASKS = ("toggle_switch_table", "grasp_move_mug")
INDEX_VERSION = "v1"
SPLIT_VERSION = "v1"
CONTROL_HZ = 20  # control_freq of the demos (dt = 0.05 s); the fps label in info.json is 30
# Fallback for a task without an eval config: the first of these whole words in the instruction.
VARIANT_WORDS = ("red", "blue", "green", "left", "right", "forward", "cap", "body")


def lerobot_dir(demos_root: Path, task: str) -> Path:
    """The mixed LeRobot dataset of ``task`` under ``demos_root``."""
    return demos_root / task / "mixed" / "lerobot"


def display_path(path: Path) -> str:
    """``path`` relative to the repo root when it lies inside it (keeps committed files free of home directories), else as given."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_variants(task: str) -> dict[str, str]:
    """Variant name to instruction string from eval/configs/<task>.yaml; {} when there is no config."""
    path = CONFIG_DIR / f"{task}.yaml"
    return parse_task_config(path)["variants"] if path.exists() else {}


def variant_for(instruction: str, variants: Mapping[str, str]) -> str:
    """Variant name of an instruction: exact config match first, then the first colour or direction word."""
    for name, text in variants.items():
        if text == instruction:
            return name
    words = set(re.findall(r"[a-z]+", instruction.lower()))
    for word in VARIANT_WORDS:
        if word in words:
            return word
    return "unknown"


def variant_order(records: Sequence[Mapping], variants: Mapping[str, str]) -> list[str]:
    """Variants present in ``records``, in config order, then any others alphabetically."""
    seen = {str(r["variant"]) for r in records}
    ordered = [v for v in variants if v in seen]
    return ordered + sorted(seen - set(ordered))


def load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))


def load_task_table(meta_dir: Path) -> dict[int, str]:
    """Return task_index to instruction from ``tasks.parquet`` (v3) or ``tasks.jsonl`` (v2.x).

    LeRobot v3 writes ``pd.DataFrame({"task_index": ...}, index=instructions).to_parquet``, so the
    instruction arrives as the ``__index_level_0__`` column; older writers use a ``task`` column.
    """
    parquet = meta_dir / "tasks.parquet"
    jsonl = meta_dir / "tasks.jsonl"
    if parquet.exists():
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(parquet)
        names = table.column_names
        text_col = next((c for c in ("task", "__index_level_0__") if c in names), None)
        if text_col is None:
            # verify on the real files: fall back to the first string column
            text_col = next(
                (c for c in names if pa.types.is_string(table.schema.field(c).type) or pa.types.is_large_string(table.schema.field(c).type)),
                None,
            )
        if text_col is None:
            raise ValueError(f"{parquet}: no instruction column in {names}")
        texts = [str(t) for t in table.column(text_col).to_pylist()]
        if "task_index" in names:
            indices = [int(i) for i in table.column("task_index").to_pylist()]
        else:
            indices = list(range(len(texts)))  # row order; verify on the real files
        return dict(zip(indices, texts))
    if jsonl.exists():
        table: dict[int, str] = {}
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                table[int(rec["task_index"])] = str(rec["task"])
        return table
    raise FileNotFoundError(f"{meta_dir}: neither tasks.parquet nor tasks.jsonl (download meta/* first)")


def load_episode_lengths(meta_dir: Path) -> dict[int, int]:
    """episode_index to length from ``meta/episodes/*/*.parquet`` (v3) or ``meta/episodes.jsonl`` (v2.x); {} when absent."""
    import pyarrow.parquet as pq

    lengths: dict[int, int] = {}
    files = sorted((meta_dir / "episodes").glob("*/*.parquet"))
    if files:
        for f in files:
            names = pq.read_schema(f).names
            if "episode_index" in names and "length" in names:
                t = pq.read_table(f, columns=["episode_index", "length"])
                lengths.update(
                    zip((int(i) for i in t.column("episode_index").to_pylist()), (int(n) for n in t.column("length").to_pylist()))
                )
        return lengths
    jsonl = meta_dir / "episodes.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                lengths[int(rec["episode_index"])] = int(rec["length"])
    return lengths


def data_files(root: Path) -> list[Path]:
    """The frame tables of a dataset: v3 ``data/chunk-*/file-*.parquet`` or v2.x ``data/chunk-*/episode_*.parquet``."""
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"{root / 'data'}: no parquet files (download data/* first)")
    return files


def read_frames(root: Path, columns: Sequence[str], optional: Sequence[str] = ()):
    """Concatenate ``columns`` (and the ``optional`` ones that exist) of every frame table.

    Adds ``data_file``, the parquet path relative to ``root``. List features such as ``action``
    arrive as object columns holding one numpy array per frame (see ``stack_list_column``).
    """
    import pandas as pd
    import pyarrow.parquet as pq

    parts = []
    for f in data_files(root):
        names = pq.read_schema(f).names
        missing = [c for c in columns if c not in names]
        if missing:
            raise ValueError(f"{f}: missing columns {missing}; present: {names}")
        cols = list(columns) + [c for c in optional if c in names]
        df = pq.read_table(f, columns=cols).to_pandas()
        df["data_file"] = f.relative_to(root).as_posix()
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def stack_list_column(values) -> np.ndarray:
    """Object column of equal-length arrays (a LeRobot list feature) to a 2-D float64 array (frames, dims)."""
    return np.stack([np.asarray(v, dtype=np.float64) for v in values])


def build_episode_index(task: str, root: Path, variants: Mapping[str, str], frames=None) -> tuple[list[dict], list[str]]:
    """Return (records, warnings): one record per episode of ``root`` and the consistency checks that failed.

    ``frames`` may be a frame table already read with ``read_frames`` (it must hold
    ``episode_index`` and ``task_index``); otherwise the scalar columns are read here.
    """
    if frames is None:
        frames = read_frames(root, ["episode_index", "task_index"], optional=["index", "frame_index", "task"])
    task_table = load_task_table(root / "meta")
    lengths = load_episode_lengths(root / "meta")
    warnings: list[str] = []
    if "index" not in frames.columns:
        warnings.append("no 'index' column in the frame table; using the row order as the global frame index")
        frames = frames.assign(index=np.arange(len(frames)))

    records: list[dict] = []
    for ep, g in frames.groupby("episode_index", sort=True):
        ep = int(ep)
        task_idx = sorted({int(t) for t in g["task_index"].tolist()})
        if len(task_idx) != 1:
            warnings.append(f"episode {ep}: several task_index values {task_idx}; using the first")
        ti = task_idx[0]
        instruction = task_table.get(ti)
        if "task" in g.columns:
            # The mixed datasets also carry the instruction per frame; verify on the real files.
            texts = sorted({str(t) for t in g["task"].dropna().tolist()})
            if texts:
                if instruction is not None and texts != [instruction]:
                    warnings.append(f"episode {ep}: 'task' column {texts} differs from the tasks table {instruction!r}")
                instruction = texts[0]
        if instruction is None:
            raise ValueError(f"{task}: episode {ep} has task_index {ti}, which is not in the tasks table {task_table}")

        n = int(len(g))
        first, last = int(g["index"].min()), int(g["index"].max())
        if last - first + 1 != n:
            warnings.append(f"episode {ep}: global index {first}..{last} is not contiguous for {n} frames")
        if lengths and lengths.get(ep) not in (None, n):
            warnings.append(f"episode {ep}: {n} frames in the data but meta/episodes says {lengths.get(ep)}")
        variant = variant_for(instruction, variants)
        if variant == "unknown":
            warnings.append(f"episode {ep}: no variant for instruction {instruction!r}")
        files = sorted(set(g["data_file"].tolist()))
        rec = {
            "task_id": task,
            "episode_index": ep,
            "task_index": ti,
            "instruction": instruction,
            "variant": variant,
            "n_frames": n,
            "first_index": first,
            "last_index": last,
            "data_file": files[0],
        }
        if len(files) > 1:
            rec["data_files"] = files
            warnings.append(f"episode {ep}: frames spread over {len(files)} files")
        records.append(rec)
    return records, warnings


def task_summary(task: str, root: Path, info: Mapping, records: Sequence[Mapping], variants: Mapping[str, str]) -> dict:
    """Per-task block of index.json: dataset facts and per-variant counts."""
    order = variant_order(records, variants)
    episodes = Counter(str(r["variant"]) for r in records)
    frames: Counter = Counter()
    instructions: dict[str, str] = {}
    for r in records:
        frames[str(r["variant"])] += int(r["n_frames"])
        instructions.setdefault(str(r["variant"]), str(r["instruction"]))
    return {
        "lerobot_dir": display_path(root),
        "codebase_version": info.get("codebase_version"),
        "fps_label": info.get("fps"),
        "control_hz": CONTROL_HZ,
        "info_total_episodes": info.get("total_episodes"),
        "info_total_frames": info.get("total_frames"),
        "n_episodes": len(records),
        "n_frames": int(sum(frames.values())),
        "variants": order,
        "instructions": {v: instructions[v] for v in order},
        "episodes_per_variant": {v: episodes[v] for v in order},
        "frames_per_variant": {v: frames[v] for v in order},
    }


def stratified_split(records: Sequence[Mapping], val_frac: float, seed: int) -> dict:
    """Split one task's episodes into train/val by whole episode, stratified by variant.

    The task gets round(val_frac * n) validation episodes (at least 1 when val_frac > 0 and
    n >= 2, at most n - 1), shared out over the variants by the largest remainder of
    val_frac * n_variant so every variant keeps about the same ratio (100 episodes per variant
    and 0.15 give exactly 15). Episodes inside a variant are drawn with
    ``numpy.random.default_rng(seed)``; variants are visited in sorted order and remainder ties
    are broken by the same generator, so the result depends only on the records and the seed.
    """
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    rng = np.random.default_rng(seed)
    by_variant: dict[str, list[int]] = defaultdict(list)
    for r in records:
        by_variant[str(r["variant"])].append(int(r["episode_index"]))
    variants = sorted(by_variant)
    n_total = sum(len(ids) for ids in by_variant.values())
    n_val = int(round(val_frac * n_total))
    if val_frac > 0 and n_total >= 2:
        n_val = max(n_val, 1)
    n_val = min(n_val, max(n_total - 1, 0))

    quota = {v: val_frac * len(by_variant[v]) for v in variants}
    n_val_v = {v: min(int(np.floor(quota[v])), len(by_variant[v])) for v in variants}
    order = sorted(variants, key=lambda v: (-(quota[v] - n_val_v[v]), rng.random()))
    remaining = n_val - sum(n_val_v.values())
    assert remaining >= 0, "floors of the quotas exceed the task total"
    for v in itertools.cycle(order):
        if remaining <= 0:
            break
        if n_val_v[v] < len(by_variant[v]):
            n_val_v[v] += 1
            remaining -= 1

    out: dict = {"train": [], "val": [], "by_variant": {}}
    for v in variants:
        ids = np.array(sorted(by_variant[v]), dtype=int)
        perm = ids[rng.permutation(len(ids))]
        val = sorted(int(x) for x in perm[: n_val_v[v]])
        train = sorted(int(x) for x in perm[n_val_v[v] :])
        out["by_variant"][v] = {"train": train, "val": val}
        out["train"] += train
        out["val"] += val
    out["train"].sort()
    out["val"].sort()
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos-root", type=Path, default=REPO_ROOT / "demos" / "CoRL", help="demo root (default: demos/CoRL)")
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), help="task ids (default: %(default)s)")
    p.add_argument("--val-frac", type=float, default=0.15, help="validation share of the episodes (default: 0.15)")
    p.add_argument("--seed", type=int, default=0, help="numpy RNG seed of the split (default: 0)")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data", help="where index.json and split_v1.json go (default: data/)")
    args = p.parse_args(argv)
    if not 0.0 <= args.val_frac < 1.0:
        p.error("--val-frac must be in [0, 1)")

    index: dict = {"version": INDEX_VERSION, "demos_root": display_path(args.demos_root), "tasks": {}, "episodes": []}
    split: dict = {
        "version": SPLIT_VERSION,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "unit": "episode",
        "stratify": "variant",
        "rng": "numpy.random.default_rng(seed), one generator per task, variants visited in sorted order",
        "tasks": {},
    }
    for task in args.tasks:
        root = lerobot_dir(args.demos_root, task)
        if not (root / "meta" / "info.json").exists():
            print(f"[FAIL] {task}: {root / 'meta' / 'info.json'} missing (download meta/* and data/* first)")
            return 2
        info = load_info(root)
        variants = load_variants(task)
        if not variants:
            print(f"[WARN] {task}: no eval/configs/{task}.yaml; variants come from the instruction words")
        records, warnings = build_episode_index(task, root, variants)
        summary = task_summary(task, root, info, records, variants)
        for key, value in (("total_episodes", summary["n_episodes"]), ("total_frames", summary["n_frames"])):
            if info.get(key) not in (None, value):
                warnings.append(f"info.json {key} = {info.get(key)} but the data has {value}")
        expected = EXPECTED.get(task)  # README totals (demos/CoRL/README.md), 100 episodes per variant
        if expected:
            if (summary["n_episodes"], summary["n_frames"]) != (expected["episodes"], expected["frames"]):
                warnings.append(
                    f"README says {expected['episodes']} episodes / {expected['frames']} frames, "
                    f"the data has {summary['n_episodes']} / {summary['n_frames']}"
                )
            for v in summary["variants"]:
                if summary["episodes_per_variant"][v] != EPISODES_PER_VARIANT:
                    warnings.append(f"variant {v!r}: {summary['episodes_per_variant'][v]} episodes, README says {EPISODES_PER_VARIANT}")
        task_split = stratified_split(records, args.val_frac, args.seed)

        print(
            f"{task}: {summary['n_episodes']} episodes, {summary['n_frames']} frames "
            f"(info.json: {info.get('total_episodes')} / {info.get('total_frames')}), "
            f"LeRobot {info.get('codebase_version')}, fps label {info.get('fps')}, control {CONTROL_HZ} Hz"
        )
        print(f"  {'variant':<10} {'episodes':>8} {'frames':>8} {'train':>6} {'val':>5}")
        for v in summary["variants"]:
            by_v = task_split["by_variant"][v]
            print(
                f"  {v:<10} {summary['episodes_per_variant'][v]:>8} {summary['frames_per_variant'][v]:>8} "
                f"{len(by_v['train']):>6} {len(by_v['val']):>5}"
            )
        for w in warnings:
            print(f"    ! {w}")

        index["tasks"][task] = summary
        index["episodes"] += records
        split["tasks"][task] = {"n_train": len(task_split["train"]), "n_val": len(task_split["val"]), **task_split}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "index.json"
    split_path = args.out_dir / f"split_{SPLIT_VERSION}.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    split_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {index_path} ({len(index['episodes'])} episodes) and {split_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
