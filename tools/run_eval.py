#!/usr/bin/env python3
"""Run the official MetaFine eval for one task and store everything in one run folder.

The official ``eval/eval_<task>.py`` runs unchanged, so the scores are the official
ones. A thin wrapper around its ``run_episode`` also saves each episode's per-step
trajectory, and the per-episode results are flattened into ``episodes.csv``::

    runs/<YYYY-MM-DD_HHMMSS>/
      run.json        what ran: command, git commit, policy, task, seeds file, dt, status
      config.yaml     copy of the task config used
      seeds.json      copy of the seed file used
      episodes.csv    one row per episode
      episodes/       one .npz per episode: action, qpos, tcp_pose per step
      official/       the official eval outputs, untouched (summaries + metafine_report.json)

Example::

    python tools/run_eval.py --task toggle_switch_table \\
        --policy-path outputs/<run>/checkpoints/010000/pretrained_model \\
        --seeds eval/seeds/dev/toggle_switch_table_20.json --mode both

``--n-action-steps N`` makes pi0 replan every N actions instead of the checkpoint's value.
``--guidance W`` adds contrastive instruction guidance (tools/guidance.py).
``--target-mask`` feeds the T3 target view (tools/target_mask.py) into pi0's third image slot.
Any argument not listed below is passed through to the official eval script
(e.g. ``--n-seeds 5``, ``--torch-seed 0``, ``--save-video``).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

TASK_MODULES = {
    "grasp_part": "eval.eval_grasp_part",
    "grasp_move_mug": "eval.eval_grasp_move_mug",
    "toggle_switch_table": "eval.eval_toggle_switch",
    "put_blocks_into_boxes": "eval.eval_put_blocks",
    "insert_letter": "eval.eval_insert_letter",
}
BASE_COLUMNS = [
    "episode_id", "protocol", "condition", "seed", "variant", "instruction",
    "success", "episode_length", "completion_time_s", "traj_file",
]


def _git_info() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=_REPO, capture_output=True,
                              text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


class TrajectoryRecorder:
    """Wraps ``run_episode`` to keep per-step robot state; results are unchanged."""

    def __init__(self, run_episode, out_dir: Path):
        self.run_episode = run_episode
        self.out_dir = out_dir
        self.calls: list[dict] = []
        self.dt: float | None = None

    def __call__(self, env, *args, on_step=None, **kwargs):
        base = env.unwrapped
        qpos: list[np.ndarray] = []
        tcp: list[np.ndarray] = []

        def record(step_i, obs, info):
            try:
                qpos.append(_to_np(base.agent.robot.get_qpos()))
                tcp.append(_to_np(base.agent.tcp.pose.raw_pose))
            except AttributeError:
                pass
            if on_step is not None:
                on_step(step_i, obs, info)

        ep = self.run_episode(env, *args, on_step=record, **kwargs)
        if self.dt is None:
            self.dt = 1.0 / float(getattr(base, "control_freq", 20))
        idx = len(self.calls)
        path = self.out_dir / f"_call{idx:05d}.npz"
        np.savez_compressed(
            path,
            action=np.asarray(ep["actions"], dtype=np.float32),
            qpos=np.asarray(qpos, dtype=np.float32),
            tcp_pose=np.asarray(tcp, dtype=np.float32),
            success=bool(ep["success"]),
            seed=int(ep["seed"]),
            instruction=str(kwargs.get("task", "")),
            dt=np.float32(self.dt),
        )
        self.calls.append({"seed": int(ep["seed"]), "instruction": str(kwargs.get("task", "")),
                           "path": path})
        return ep


def _official_episodes(official: Path, cfg: dict, mode: str) -> list[dict]:
    """Per-episode rows from the official summaries, in the order the eval ran them."""
    rows: list[dict] = []
    perc_file = official / "perception_summary.json"
    if mode in ("perception", "both") and perc_file.exists():
        perc = json.loads(perc_file.read_text())
        variant = perc["meta"]["variant"]
        for cond in perc["meta"]["perception_profiles"]:
            for d in perc[cond]["details"]:
                rows.append({"protocol": "perception", "condition": cond, "variant": variant,
                             "instruction": perc["meta"]["instruction"], **d})
    und_file = official / "understanding_summary.json"
    if mode in ("understanding", "both") and und_file.exists():
        by_text = {text: v for v, text in (cfg.get("variants") or {}).items()}
        for d in json.loads(und_file.read_text())["episodes"]:
            rows.append({"protocol": "understanding", "condition": "clean",
                         "variant": by_text.get(d.get("instruction"), ""), **d})
    return rows


def build_episodes_csv(run_dir: Path, recorder: TrajectoryRecorder, cfg: dict, mode: str) -> int:
    rows = _official_episodes(run_dir / "official", cfg, mode)
    if len(rows) != len(recorder.calls):
        raise RuntimeError(f"{len(rows)} official episodes but {len(recorder.calls)} recorded")
    extra_cols: list[str] = []
    out_rows = []
    for row, call in zip(rows, recorder.calls):
        if row["seed"] != call["seed"] or row["instruction"] != call["instruction"]:
            raise RuntimeError(f"episode order mismatch: official {row['seed']}/{row['instruction']}"
                               f" vs recorded {call['seed']}/{call['instruction']}")
        prefix = "und" if row["protocol"] == "understanding" else "perc"
        episode_id = f"{prefix}-{row['condition']}-s{row['seed']:06d}-{row['variant']}"
        traj = run_dir / "episodes" / f"{episode_id}.npz"
        call["path"].rename(traj)
        out = {
            "episode_id": episode_id,
            "completion_time_s": round(row["episode_length"] * recorder.dt, 4),
            "traj_file": str(traj.relative_to(run_dir)),
            **{k: v for k, v in row.items() if k != "video" and not isinstance(v, (dict, list))},
        }
        extra_cols += [k for k in out if k not in BASE_COLUMNS and k not in extra_cols]
        out_rows.append(out)
    with open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS + extra_cols)
        writer.writeheader()
        writer.writerows(out_rows)
    return len(out_rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(TASK_MODULES))
    p.add_argument("--seeds", required=True, help="seed JSON from eval.select_eval_seeds")
    p.add_argument("--policy-path", default=None)
    p.add_argument("--policy-url", default=None)
    p.add_argument("--config", default=None,
                   help="task config to use instead of eval/configs/<task>.yaml "
                        "(e.g. a copy with a larger perception.n_episodes)")
    p.add_argument("--mode", choices=["perception", "understanding", "both"], default="both")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="replan every N actions instead of the checkpoint's n_action_steps "
                        "(pi0 default 50); --policy-path only")
    p.add_argument("--guidance", type=float, default=None,
                   help="contrastive instruction guidance weight w (tools/guidance.py); --policy-path only")
    p.add_argument("--guidance-swap", default="red,blue",
                   help="the two words swapped to make the negative instruction")
    p.add_argument("--target-mask", action="store_true",
                   help="T3 target view in the third image slot (tools/target_mask.py); --policy-path only")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--note", default="", help="free-text note stored in run.json")
    args, passthrough = p.parse_known_args()
    if bool(args.policy_path) == bool(args.policy_url):
        raise SystemExit("pass exactly one of --policy-path / --policy-url")

    from utils.eval_common import load_task_config

    config_path = (Path(args.config).resolve() if args.config
                   else _REPO / "eval" / "configs" / f"{args.task}.yaml")
    cfg = load_task_config(str(config_path))
    started = datetime.now(timezone.utc)
    run_dir = _REPO / args.runs_dir / started.astimezone().strftime("%Y-%m-%d_%H%M%S")
    (run_dir / "episodes").mkdir(parents=True)
    shutil.copy(config_path, run_dir / "config.yaml")
    shutil.copy(args.seeds, run_dir / "seeds.json")

    run_info = {
        "status": "running",
        "task_id": args.task,
        "mode": args.mode,
        "policy": {"path": args.policy_path, "url": args.policy_url},
        "config_file": str(config_path),
        "n_action_steps": args.n_action_steps,
        "guidance": args.guidance, "guidance_swap": args.guidance_swap if args.guidance is not None else None,
        "target_mask": args.target_mask,
        "seeds_file": str(args.seeds),
        "eval_passthrough_args": passthrough,
        "command": " ".join([sys.executable, *sys.argv]),
        "git": _git_info(),
        "note": args.note,
        "started_utc": started.isoformat(timespec="seconds"),
    }
    _write_json(run_dir / "run.json", run_info)
    print(f"[run_eval] run folder: {run_dir}")

    module = importlib.import_module(TASK_MODULES[args.task])
    recorder = TrajectoryRecorder(module.run_episode, run_dir / "episodes")
    module.run_episode = recorder
    if args.n_action_steps is not None or args.guidance is not None or args.target_mask:
        if not args.policy_path:
            raise SystemExit("--n-action-steps / --guidance / --target-mask need --policy-path")
        load = module.load_pi0_policy

        def load_with_overrides(*a, **kw):
            policy, pre, post = load(*a, **kw)
            if args.n_action_steps is not None:
                policy.config.n_action_steps = args.n_action_steps
                policy.reset()  # rebuild the action queue with the new length
            if args.guidance is not None:
                from tools.guidance import enable_guidance
                enable_guidance(policy, pre, args.guidance, tuple(args.guidance_swap.split(",")))
            if args.target_mask:
                from tools.target_mask import enable_target_mask
                enable_target_mask(policy)
            return policy, pre, post

        module.load_pi0_policy = load_with_overrides

    policy_args = (["--policy-path", args.policy_path] if args.policy_path
                   else ["--policy-url", args.policy_url])
    sys.argv = [module.__file__, "--config", str(config_path), "--seeds", str(args.seeds),
                "--mode", args.mode, "--record-dir", str(run_dir / "official"),
                *policy_args, *passthrough]
    try:
        module.main()
        from utils.eval_report import build_report, _load
        report = build_report(
            task_id=args.task,
            perception=_load(str(run_dir / "official" / "perception_summary.json")),
            understanding=_load(str(run_dir / "official" / "understanding_summary.json")),
        )
        _write_json(run_dir / "official" / "metafine_report.json", report)
        n = build_episodes_csv(run_dir, recorder, cfg, args.mode)
        run_info.update(status="done", n_episodes=n, dt=recorder.dt,
                        metrics=report.get("metrics"))
    except BaseException as exc:
        run_info.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        run_info["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_json(run_dir / "run.json", run_info)
    print(f"[run_eval] {run_info['n_episodes']} episodes -> {run_dir / 'episodes.csv'}")
    try:
        from tools.make_report import make_report
        print(f"[run_eval] report -> {make_report([run_dir], run_dir / 'report.pdf')}")
    except Exception as exc:  # the run itself is complete; a report bug must not hide that
        print(f"[run_eval] report failed ({type(exc).__name__}: {exc}); "
              f"rerun: python tools/make_report.py {run_dir}")


if __name__ == "__main__":
    # Same as the official eval scripts: avoid OpenBLAS / Vulkan thread fights.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
