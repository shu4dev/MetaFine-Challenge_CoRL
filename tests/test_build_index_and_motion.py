"""Tests for tools/build_index.py and tools/motion_summary.py on a synthetic LeRobot v3 dataset.

No download, no simulator, no GPU: a tiny dataset (two tasks, six episodes each) is written to a
temp dir in the layout of ``demos/CoRL/<task>/mixed/lerobot`` (meta/info.json, meta/tasks.parquet,
meta/episodes/chunk-000/file-000.parquet, data/chunk-000/file-000.parquet) and both tools run on it.

Run with ``python3 -m pytest tests/test_build_index_and_motion.py -q``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # motion_summary does "from build_index import ..."
    spec.loader.exec_module(module)
    return module


build_index = _load_tool("build_index")
motion_summary = _load_tool("motion_summary")

FS = 20.0
WINDOW, STRIDE = 50, 10
INSTRUCTIONS = {
    "toggle_switch_table": {
        "red": "Toggle the red switch on the table",
        "blue": "Toggle the blue switch on the table",
    },
    "grasp_move_mug": {
        "left": "Grasp the mug by the handle and move it to the left",
        "right": "Grasp the mug by the handle and move it to the right",
        "forward": "Grasp the mug by the handle and move it forward",
    },
}
# variant -> (episodes, 8 Hz jitter added to the arm actions)
PLAN = {
    "toggle_switch_table": [("red", 3, False), ("blue", 3, True)],
    "grasp_move_mug": [("left", 2, False), ("right", 2, False), ("forward", 2, False)],
}
LENGTHS = {"toggle_switch_table": [120, 160, 200, 130, 170, 300], "grasp_move_mug": [150, 180, 210, 240, 270, 300]}
# Only the T3 frame table carries the per-frame "task" string column (the real mixed datasets
# seem to; grasp_move_mug exercises the tasks-table path).
WITH_TASK_COLUMN = {"toggle_switch_table": True, "grasp_move_mug": False}


def _episode(rng: np.random.Generator, n: int, jitter: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One synthetic episode: action (n, 8), observation.state (n, 9), gripper closed flags (n,)."""
    t = np.arange(n) / FS
    arm = 0.02 * np.sin(2 * np.pi * 0.3 * t[:, None] + rng.uniform(0, 2 * np.pi, size=7))
    if jitter:
        arm = arm + 0.015 * np.sin(2 * np.pi * 8.0 * t[:, None] + rng.uniform(0, 2 * np.pi, size=7))
    arm = arm + rng.normal(0.0, 1e-4, size=arm.shape)
    closed = np.zeros(n, dtype=bool)
    closed[int(0.4 * n) : int(0.85 * n)] = True  # grasp in the middle, release near the end
    fingers = np.where(closed, 0.002, 0.04)
    gripper = np.where(closed, -1.0, 1.0)
    action = np.concatenate([arm, gripper[:, None]], axis=1).astype(np.float32)
    qpos = 0.1 + np.cumsum(arm, axis=0)
    state = np.concatenate([qpos, fingers[:, None], fingers[:, None]], axis=1).astype(np.float32)
    return action, state, closed


def _list_array(rows: list[np.ndarray]) -> pa.Array:
    return pa.array([np.asarray(r, dtype=np.float32) for r in rows], type=pa.list_(pa.float32()))


def write_lerobot_v3(root: Path, task: str, rng: np.random.Generator) -> list[dict]:
    """Write a LeRobot v3 style dataset for ``task`` under ``root`` and return the ground truth per episode."""
    instructions = list(INSTRUCTIONS[task].values())
    variants = list(INSTRUCTIONS[task].keys())
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    # tasks.parquet exactly as lerobot 0.4.4 writes it: instruction as the pandas index.
    pd.DataFrame({"task_index": list(range(len(instructions)))}, index=instructions).to_parquet(root / "meta" / "tasks.parquet")

    truth, ep_meta, columns = [], [], {k: [] for k in ("action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index", "task")}
    lengths = iter(LENGTHS[task])
    ep_idx, global_idx = 0, 0
    for variant, n_eps, jitter in PLAN[task]:
        for _ in range(n_eps):
            n = next(lengths)
            action, state, closed = _episode(rng, n, jitter)
            task_index = variants.index(variant)
            columns["action"] += list(action)
            columns["observation.state"] += list(state)
            columns["timestamp"] += list((np.arange(n) / 30.0).astype(np.float32))
            columns["frame_index"] += list(range(n))
            columns["episode_index"] += [ep_idx] * n
            columns["index"] += list(range(global_idx, global_idx + n))
            columns["task_index"] += [task_index] * n
            columns["task"] += [instructions[task_index]] * n
            ep_meta.append(
                {
                    "episode_index": ep_idx,
                    "tasks": [instructions[task_index]],
                    "length": n,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": global_idx,
                    "dataset_to_index": global_idx + n,
                }
            )
            truth.append(
                {
                    "episode_index": ep_idx,
                    "variant": variant,
                    "instruction": instructions[task_index],
                    "task_index": task_index,
                    "n_frames": n,
                    "first_index": global_idx,
                    "last_index": global_idx + n - 1,
                    "closed": closed,
                    "jitter": jitter,
                }
            )
            ep_idx += 1
            global_idx += n

    arrays = {
        "action": _list_array(columns["action"]),
        "observation.state": _list_array(columns["observation.state"]),
        "timestamp": pa.array(columns["timestamp"], type=pa.float32()),
        "frame_index": pa.array(columns["frame_index"], type=pa.int64()),
        "episode_index": pa.array(columns["episode_index"], type=pa.int64()),
        "index": pa.array(columns["index"], type=pa.int64()),
        "task_index": pa.array(columns["task_index"], type=pa.int64()),
    }
    if WITH_TASK_COLUMN[task]:
        arrays["task"] = pa.array(columns["task"], type=pa.string())
    pq.write_table(pa.table(arrays), root / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.Table.from_pylist(ep_meta), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    features = {
        "action": {"dtype": "float32", "shape": [8], "names": [f"action_{i}" for i in range(8)]},
        "observation.state": {"dtype": "float32", "shape": [9], "names": [f"joint_{i}" for i in range(9)]},
        "observation.images.base_camera": {"dtype": "video", "shape": [512, 512, 3], "names": ["height", "width", "channels"]},
        "observation.images.hand_camera": {"dtype": "video", "shape": [512, 512, 3], "names": ["height", "width", "channels"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if WITH_TASK_COLUMN[task]:
        features["task"] = {"dtype": "string", "shape": [1], "names": None}
    info = {
        "codebase_version": "v3.0",
        "robot_type": "panda",
        "total_episodes": ep_idx,
        "total_frames": global_idx,
        "total_tasks": len(instructions),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": 30,
        "splits": {"train": f"0:{ep_idx}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")
    return truth


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Synthetic demos root plus the ground truth per task, and both tools run once."""
    base = tmp_path_factory.mktemp("synthetic")
    demos_root = base / "demos" / "CoRL"
    rng = np.random.default_rng(1234)
    truth = {task: write_lerobot_v3(build_index.lerobot_dir(demos_root, task), task, rng) for task in PLAN}

    out_dir = base / "data"
    docs_dir = base / "docs"
    csv_path = out_dir / "motion_summary.csv"
    assert build_index.main(["--demos-root", str(demos_root), "--out-dir", str(out_dir), "--val-frac", "0.15", "--seed", "0"]) == 0
    assert motion_summary.main(
        [
            "--demos-root", str(demos_root), "--out-dir", str(docs_dir), "--csv", str(csv_path),
            "--window", str(WINDOW), "--stride", str(STRIDE), "--fs", str(FS),
            "--split-json", str(out_dir / "split_v1.json"),
        ]
    ) == 0
    return {
        "demos_root": demos_root,
        "truth": truth,
        "index": json.loads((out_dir / "index.json").read_text(encoding="utf-8")),
        "split": json.loads((out_dir / "split_v1.json").read_text(encoding="utf-8")),
        "md_path": docs_dir / "MOTION_SUMMARY.md",
        "md": (docs_dir / "MOTION_SUMMARY.md").read_text(encoding="utf-8"),
        "csv": pd.read_csv(csv_path),
    }


# --- build_index -----------------------------------------------------------


def test_index_has_every_episode_with_its_variant(synthetic):
    episodes = synthetic["index"]["episodes"]
    assert len(episodes) == 12
    for task, truth in synthetic["truth"].items():
        got = {r["episode_index"]: r for r in episodes if r["task_id"] == task}
        assert sorted(got) == [t["episode_index"] for t in truth]
        for t in truth:
            r = got[t["episode_index"]]
            assert r["variant"] == t["variant"]
            assert r["instruction"] == t["instruction"]
            assert r["task_index"] == t["task_index"]
            assert r["n_frames"] == t["n_frames"]
            assert (r["first_index"], r["last_index"]) == (t["first_index"], t["last_index"])
            assert r["data_file"] == "data/chunk-000/file-000.parquet"


def test_index_task_block_counts(synthetic):
    tasks = synthetic["index"]["tasks"]
    assert tasks["toggle_switch_table"]["episodes_per_variant"] == {"red": 3, "blue": 3}
    assert tasks["grasp_move_mug"]["episodes_per_variant"] == {"left": 2, "right": 2, "forward": 2}
    assert tasks["toggle_switch_table"]["instructions"] == INSTRUCTIONS["toggle_switch_table"]
    assert tasks["toggle_switch_table"]["n_frames"] == sum(LENGTHS["toggle_switch_table"])
    assert tasks["toggle_switch_table"]["codebase_version"] == "v3.0"
    assert tasks["toggle_switch_table"]["control_hz"] == 20


def test_split_is_disjoint_and_about_85_15(synthetic):
    split = synthetic["split"]
    assert split["val_frac"] == 0.15 and split["seed"] == 0 and split["unit"] == "episode"
    for task, truth in synthetic["truth"].items():
        s = split["tasks"][task]
        train, val = set(s["train"]), set(s["val"])
        assert not train & val
        assert train | val == {t["episode_index"] for t in truth}
        assert s["n_train"] == len(train) and s["n_val"] == len(val)
        # round(0.15 * 6) = 1 validation episode: 1/6 is as close to 15% as six episodes allow
        assert len(val) == 1
        by_variant = s["by_variant"]
        assert set(by_variant) == {t["variant"] for t in truth}
        assert sorted(sum((v["train"] + v["val"] for v in by_variant.values()), [])) == sorted(train | val)


def test_stratified_split_exact_on_100_per_variant():
    records = [{"episode_index": i, "variant": "red" if i < 100 else "blue"} for i in range(200)]
    split = build_index.stratified_split(records, 0.15, 0)
    assert len(split["val"]) == 30 and len(split["train"]) == 170
    for v in ("red", "blue"):
        assert len(split["by_variant"][v]["val"]) == 15
        assert len(split["by_variant"][v]["train"]) == 85
    assert not set(split["train"]) & set(split["val"])
    assert split == build_index.stratified_split(records, 0.15, 0)  # deterministic
    assert split["val"] != build_index.stratified_split(records, 0.15, 1)["val"]
    # T2-like: three variants of 100 give 15 each
    records = [{"episode_index": i, "variant": ("left", "right", "forward")[i // 100]} for i in range(300)]
    split = build_index.stratified_split(records, 0.15, 0)
    assert [len(split["by_variant"][v]["val"]) for v in ("forward", "left", "right")] == [15, 15, 15]


def test_variant_lookup_with_and_without_config():
    cfg = build_index.load_variants("toggle_switch_table")
    assert build_index.variant_for("Toggle the blue switch on the table", cfg) == "blue"
    assert build_index.variant_for("Toggle the blue switch on the table", {}) == "blue"
    assert build_index.variant_for("Grasp the mug by the handle and move it forward", {}) == "forward"
    assert build_index.variant_for("Grasp the mug by the handle and move it to the left", {}) == "left"
    assert build_index.variant_for("Do something else", {}) == "unknown"


def test_task_table_readers(tmp_path):
    texts = ["Toggle the red switch on the table", "Toggle the blue switch on the table"]
    # v3: instruction as the pandas index (__index_level_0__ on disk)
    v3 = tmp_path / "v3"
    v3.mkdir()
    pd.DataFrame({"task_index": [0, 1]}, index=texts).to_parquet(v3 / "tasks.parquet")
    assert build_index.load_task_table(v3) == {0: texts[0], 1: texts[1]}
    # v3 variant with an explicit "task" column and shuffled task_index
    v3b = tmp_path / "v3b"
    v3b.mkdir()
    pd.DataFrame({"task_index": [1, 0], "task": texts}).to_parquet(v3b / "tasks.parquet", index=False)
    assert build_index.load_task_table(v3b) == {1: texts[0], 0: texts[1]}
    # v2.x: tasks.jsonl
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "tasks.jsonl").write_text("\n".join(json.dumps({"task_index": i, "task": t}) for i, t in enumerate(texts)) + "\n")
    assert build_index.load_task_table(v2) == {0: texts[0], 1: texts[1]}
    with pytest.raises(FileNotFoundError):
        build_index.load_task_table(tmp_path / "missing")


# --- motion_summary --------------------------------------------------------


def _expected_windows(n: int) -> int:
    return (n - WINDOW) // STRIDE + 1


def test_motion_outputs_exist_with_expected_rows(synthetic):
    md, csv = synthetic["md"], synthetic["csv"]
    assert synthetic["md_path"].exists()
    assert list(csv.columns) == list(motion_summary.CSV_COLUMNS)
    assert len(csv) == sum(_expected_windows(n) for task in LENGTHS for n in LENGTHS[task])
    for task, truth in synthetic["truth"].items():
        assert f"## {task}" in md
        sub = csv[csv["task_id"] == task]
        assert sub.groupby("episode_index").size().to_dict() == {t["episode_index"]: _expected_windows(t["n_frames"]) for t in truth}
        for variant in INSTRUCTIONS[task]:
            for stage in ("open", "closed", "all"):
                assert f"| {variant} | {stage} |" in md
            assert set(sub.loc[sub["variant"] == variant, "stage"]) == {"open", "closed"}
    assert "across variants, stage open" in md and "across stages, variant red" in md
    assert "Kruskal-Wallis" in md


def test_csv_split_column_follows_split_v1(synthetic):
    csv, split, md = synthetic["csv"], synthetic["split"], synthetic["md"]
    assert set(csv["split"]) == {"train", "val"}
    for task in PLAN:
        sub = csv[csv["task_id"] == task]
        assert set(sub.loc[sub["split"] == "val", "episode_index"]) == set(split["tasks"][task]["val"])
        assert set(sub.loc[sub["split"] == "train", "episode_index"]) == set(split["tasks"][task]["train"])
        n_train = int((sub["split"] == "train").sum())
        assert f"Table and tests use the {n_train} train windows" in md
    assert "Tables and tests use the train episodes of" in md


def test_band_fractions_sum_to_one(synthetic):
    csv = synthetic["csv"]
    sums = csv[list(motion_summary.BAND_COLS)].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)
    assert (csv[list(motion_summary.BAND_COLS)] >= 0).all().all()
    assert (csv["action_change"] > 0).all() and (csv["roughness"] > 0).all() and (csv["mad"] > 0).all()


def test_high_frequency_variant_has_more_6_10_hz_energy(synthetic):
    csv = synthetic["csv"]
    t3 = csv[csv["task_id"] == "toggle_switch_table"]
    red = t3[t3["variant"] == "red"]
    blue = t3[t3["variant"] == "blue"]  # 8 Hz jitter added in the synthetic data
    assert blue["frac_6_10"].mean() > 3 * red["frac_6_10"].mean()
    assert blue["frac_6_10"].mean() > 0.2 and red["frac_6_10"].mean() < 0.05
    assert red["frac_0_1"].mean() > 0.8
    assert blue["roughness"].mean() > 3 * red["roughness"].mean()
    for stage in ("open", "closed"):
        assert blue.loc[blue["stage"] == stage, "frac_6_10"].mean() > red.loc[red["stage"] == stage, "frac_6_10"].mean()
    # the markdown table shows it too: the blue "all" row carries a larger 6-10 Hz mean than the red one
    md_rows = {line.split("|")[1].strip() + "/" + line.split("|")[2].strip(): line for line in synthetic["md"].splitlines() if line.startswith("| red |") or line.startswith("| blue |")}
    red_all = float(md_rows["red/all"].split("|")[-2].split("(")[0])
    blue_all = float(md_rows["blue/all"].split("|")[-2].split("(")[0])
    assert blue_all > red_all


def test_stage_labels_follow_the_gripper(synthetic):
    csv = synthetic["csv"]
    for task, truth in synthetic["truth"].items():
        for t in truth:
            sub = csv[(csv["task_id"] == task) & (csv["episode_index"] == t["episode_index"])]
            for _, w in sub.iterrows():
                frac = t["closed"][int(w["start"]) : int(w["end"])].mean()
                assert w["closed_frac"] == pytest.approx(frac)
                assert w["stage"] == ("closed" if frac > 0.5 else "open")


def test_window_metrics_on_known_signals():
    t = np.arange(WINDOW) / FS
    const = np.full((WINDOW, 7), 0.01)
    m = motion_summary.window_metrics(const, FS)
    assert m["action_change"] == pytest.approx(0.0) and m["roughness"] == pytest.approx(0.0) and m["mad"] == pytest.approx(0.0)
    assert m["frac_0_1"] == pytest.approx(1.0)  # all energy in the DC bin
    ramp = np.repeat(np.arange(WINDOW)[:, None] * 0.001, 7, axis=1)  # first difference 0.001 on 7 joints
    m = motion_summary.window_metrics(ramp, FS)
    assert m["action_change"] == pytest.approx(0.001)
    assert m["mad"] == pytest.approx(0.001 * FS * np.sqrt(7))  # L2 over joints of the difference over dt
    assert m["roughness"] == pytest.approx(0.0, abs=1e-12)
    fast = np.repeat(np.sin(2 * np.pi * 8.0 * t)[:, None], 7, axis=1)  # 8 Hz, an exact bin at 20 Hz / 50 steps
    m = motion_summary.window_metrics(fast, FS)
    assert m["frac_6_10"] == pytest.approx(1.0, abs=1e-6)
    zeros = np.zeros((WINDOW, 7))
    assert np.isnan(motion_summary.band_fractions(zeros, FS)).all()
    mixed = np.concatenate([fast[:, :3], zeros[:, :4]], axis=1)  # silent joints are left out of the average
    assert motion_summary.band_fractions(mixed, FS)[3] == pytest.approx(1.0, abs=1e-6)


def test_kruskal_table_shapes(synthetic):
    csv = synthetic["csv"]
    t3 = csv[csv["task_id"] == "toggle_switch_table"]
    rows = motion_summary.kruskal_table(t3, ["red", "blue"], unit="episode")
    assert [r["comparison"] for r in rows] == [
        "across variants, stage open",
        "across variants, stage closed",
        "across stages, variant red",
        "across stages, variant blue",
    ]
    for r in rows[:2]:
        assert r["groups"] == "red, blue"
        assert 0.0 <= r["frac_6_10"] <= 1.0
    assert np.isnan(motion_summary.kruskal_p([np.array([1.0, 2.0])]))
    assert np.isnan(motion_summary.kruskal_p([np.array([1.0, 1.0]), np.array([1.0, 1.0])]))
