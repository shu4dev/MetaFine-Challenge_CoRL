#!/usr/bin/env python3
"""Check downloaded CoRL demos against the counts and instruction strings we rely on.

For each task (default: T2 ``grasp_move_mug`` and T3 ``toggle_switch_table``):

* ``mixed/lerobot/meta/info.json`` has the expected episode and frame totals;
* every instruction variant has 100 episodes in the mixed dataset;
* the instruction strings in the LeRobot ``tasks.parquet`` match the
  ``variants`` block of ``eval/configs/<task>.yaml`` exactly;
* every replay JSON in ``train_demo_jsons`` holds 100 episodes with unique
  integer ``episode_seed`` values.

It also checks that ``eval/configs/*.yaml`` defines the 14 instruction strings
of the five competition tasks.

Only pyarrow and the standard library are needed, so it runs on a laptop
without the simulator stack.

Usage:
    python tools/check_demos.py
    python tools/check_demos.py --tasks toggle_switch_table --root demos/CoRL

Exit code 0 when every check passes, 2 when any check fails.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "eval" / "configs"

EPISODES_PER_VARIANT = 100
N_INSTRUCTIONS = 14  # 2 + 3 + 2 + 3 + 4 variants over T1-T5

# Totals of the mixed LeRobot datasets (demos/CoRL/README.md).
EXPECTED = {
    "grasp_part": {"episodes": 200, "frames": 30_670},
    "grasp_move_mug": {"episodes": 300, "frames": 39_395},
    "toggle_switch_table": {"episodes": 200, "frames": 20_567},
    "put_blocks_into_boxes": {"episodes": 300, "frames": 156_151},
    "insert_letter": {"episodes": 400, "frames": 115_894},
}
DEFAULT_TASKS = ("grasp_move_mug", "toggle_switch_table")

# Upstream defects in hiangx/MetaFine_CoRL26 (revision 0e86f4a), reported as
# warnings so any new mismatch still fails. In the T3 blue RGB replay JSON,
# episodes 98 and 99 both carry seed 729406 and source seed 723909 is absent;
# their LeRobot trajectories differ, so only the seed label is wrong.
KNOWN_DUPLICATE_SEEDS = {
    "toggle_switch_table_blue.rgb.pd_joint_delta_pos.physx_cpu.json": {729406},
}


def parse_task_config(path: Path) -> dict:
    """Read ``variants`` and ``train_demo_jsons`` from an eval config.

    The configs are flat YAML; a small line parser keeps this script free of
    a PyYAML dependency.
    """
    variants: dict[str, str] = {}
    demo_jsons: list[str] = []
    block = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            block = line.split(":", 1)[0].strip()
            continue
        if block == "variants":
            m = re.match(r'\s+([^:\s]+):\s*"(.*)"\s*$', line)
            if m:
                variants[m.group(1)] = m.group(2)
        elif block == "train_demo_jsons":
            m = re.match(r"\s+-\s*(\S+)\s*$", line)
            if m:
                demo_jsons.append(m.group(1))
    return {"variants": variants, "train_demo_jsons": demo_jsons}


def compare_instructions(expected: Mapping[str, str], found: Iterable[str]) -> list[str]:
    """Return one error per instruction string that differs between config and demos.

    ``expected`` maps variant name to the config string; ``found`` holds the
    strings read from the demos. An empty list means an exact match.
    """
    expected_set = set(expected.values())
    found_set = set(found)
    errors = [
        f"variant {name!r}: config string {text!r} not found in demos"
        for name, text in expected.items()
        if text not in found_set
    ]
    errors += [f"demo string {text!r} is not in the config" for text in sorted(found_set - expected_set)]
    return errors


def _read_tasks_parquet(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    # LeRobot v3 writes the task string as the (pandas) index column.
    for name in ("task", "__index_level_0__"):
        if name in table.column_names:
            return [str(t) for t in table.column(name).to_pylist()]
    raise ValueError(f"{path}: no task column in {table.column_names}")


def _episode_task_counts(meta_dir: Path) -> Counter:
    import pyarrow.parquet as pq

    counts: Counter = Counter()
    files = sorted((meta_dir / "episodes").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"{meta_dir / 'episodes'}: no parquet files")
    for f in files:
        for tasks in pq.read_table(f, columns=["tasks"]).column("tasks").to_pylist():
            for task in tasks:
                counts[task] += 1
    return counts


def check_replay_json(path: Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one replay JSON."""
    if not path.exists():
        return [f"{path}: missing"], []
    episodes = json.loads(path.read_text(encoding="utf-8")).get("episodes", [])
    seeds = [ep.get("episode_seed") for ep in episodes]
    errors, warnings = [], []
    if len(episodes) != EPISODES_PER_VARIANT:
        errors.append(f"{path.name}: {len(episodes)} episodes, expected {EPISODES_PER_VARIANT}")
    bad = [s for s in seeds if not isinstance(s, int) or isinstance(s, bool)]
    if bad:
        errors.append(f"{path.name}: {len(bad)} non-integer episode_seed values")
    dups = {s for s, n in Counter(seeds).items() if n > 1}
    known = KNOWN_DUPLICATE_SEEDS.get(path.name, set())
    if dups - known:
        errors.append(f"{path.name}: duplicate episode_seed values {sorted(dups - known)}")
    if dups & known:
        warnings.append(f"{path.name}: known upstream duplicate episode_seed {sorted(dups & known)}")
    return errors, warnings


def check_task(task: str, root: Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one task's demos."""
    cfg = parse_task_config(CONFIG_DIR / f"{task}.yaml")
    meta = root / task / "mixed" / "lerobot" / "meta"
    info_path = meta / "info.json"
    if not info_path.exists():
        return [f"{info_path}: missing (download the demos first)"], []

    errors, warnings = [], []
    info = json.loads(info_path.read_text(encoding="utf-8"))
    exp = EXPECTED[task]
    for key, info_key in (("episodes", "total_episodes"), ("frames", "total_frames")):
        if info.get(info_key) != exp[key]:
            errors.append(f"{info_key} = {info.get(info_key)}, expected {exp[key]}")

    errors += compare_instructions(cfg["variants"], _read_tasks_parquet(meta / "tasks.parquet"))

    counts = _episode_task_counts(meta)
    for name, text in cfg["variants"].items():
        if counts.get(text, 0) != EPISODES_PER_VARIANT:
            errors.append(f"variant {name!r}: {counts.get(text, 0)} episodes, expected {EPISODES_PER_VARIANT}")

    for rel in cfg["train_demo_jsons"]:
        # Config paths are repo-relative (demos/CoRL/...); re-root them on --root.
        rel_path = Path(rel)
        if rel_path.parts[:2] == ("demos", "CoRL"):
            rel_path = Path(*rel_path.parts[2:])
        json_errors, json_warnings = check_replay_json(root / rel_path)
        errors += json_errors
        warnings += json_warnings
    return errors, warnings


def check_instruction_total() -> list[str]:
    strings = [s for f in sorted(CONFIG_DIR.glob("*.yaml")) for s in parse_task_config(f)["variants"].values()]
    if len(strings) != N_INSTRUCTIONS or len(set(strings)) != N_INSTRUCTIONS:
        return [f"eval/configs defines {len(strings)} instruction strings ({len(set(strings))} unique), expected {N_INSTRUCTIONS}"]
    return []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=REPO_ROOT / "demos" / "CoRL", help="demo root (default: demos/CoRL)")
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=sorted(EXPECTED))
    args = p.parse_args(argv)

    failed = False
    for label, (errors, warnings) in [("eval/configs", (check_instruction_total(), []))] + [
        (task, check_task(task, args.root)) for task in args.tasks
    ]:
        status = "FAIL" if errors else ("WARN" if warnings else "OK")
        print(f"[{status}] {label}")
        for e in errors:
            print(f"    - {e}")
        for w in warnings:
            print(f"    ! {w}")
        failed |= bool(errors)

    print("CHECK_DEMOS: " + ("FAIL" if failed else "PASS"))
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
