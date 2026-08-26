#!/usr/bin/env python3
"""Record side-by-side base+hand camera videos for any task, via a policy server.

Useful for eyeballing a submission without running (or modifying) the scoring
scripts — ``eval/eval_put_blocks.py`` has no video support at all, and the
tasks that do save one video per episode, which is far more than you want when
you just need to look at the behaviour.

    python tools/record_video.py --config eval/configs/put_blocks_into_boxes.yaml \\
        --policy-url http://127.0.0.1:8080 \\
        --seeds-file /tmp/put_blocks_dev.json --n-episodes 2 \\
        --out-dir eval_runs/videos_put_blocks

Point it at a checkpoint with ``--ckpt`` instead of ``--policy-url`` to record
the in-process path.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import core.env  # noqa: F401,E402
from tools.parity_check import TASK_MODULES, make_env  # noqa: E402
from utils.eval_common import (  # noqa: E402
    load_pi0_policy,
    load_remote_policy,
    load_seed_list,
    load_task_config,
    obs_to_frame,
    resolve_path,
    run_episode,
    save_rgb_video,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True)
    p.add_argument("--policy-url", default=None)
    p.add_argument("--ckpt", default=None, help="in-process alternative to --policy-url")
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--action-dim", type=int, default=8)
    p.add_argument("--seeds-file", default=None)
    p.add_argument("--seeds", default=None, help="Comma-separated seeds")
    p.add_argument("--variants", default=None,
                   help="Comma-separated variants (default: all)")
    p.add_argument("--n-episodes", type=int, default=2,
                   help="Episodes per variant")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--video-fps", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.ckpt) == bool(args.policy_url):
        raise SystemExit("pass exactly one of --ckpt / --policy-url")

    cfg = load_task_config(args.config)
    task_id = cfg["task_id"]
    mod = importlib.import_module(TASK_MODULES[task_id])
    variants = dict(cfg.get("variants") or {})
    wanted = ([v.strip() for v in args.variants.split(",") if v.strip()]
              if args.variants else list(variants))
    max_steps = int(args.max_steps or cfg.get("max_episode_steps", 300))

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    elif args.seeds_file:
        seeds = load_seed_list(args.seeds_file)
    else:
        raise SystemExit("pass --seeds or --seeds-file")
    seeds = seeds[: args.n_episodes]

    if args.policy_url:
        policy, pre, post = load_remote_policy(
            args.policy_url, task_id, action_dim=args.action_dim)
    else:
        policy, pre, post = load_pi0_policy(
            args.ckpt, device=args.device, tokenizer_path=args.tokenizer_path)

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"task={task_id} variants={wanted} seeds={seeds} max_steps={max_steps}")

    written = []
    for variant in wanted:
        instruction = variants[variant]
        env = make_env(task_id, mod, cfg, variant)
        try:
            for seed in seeds:
                frames: list[np.ndarray] = []

                def on_step(_t: int, obs: Any, _info: Any) -> None:
                    try:
                        frames.append(obs_to_frame(obs))
                    except KeyError:
                        pass

                ep = run_episode(env, policy, pre, post, task=instruction,
                                 seed=seed, max_steps=max_steps, on_step=on_step)
                name = (f"seed{int(seed):06d}_{variant}_succ{int(ep['success'])}"
                        f"_len{ep['episode_length']:03d}.mp4")
                path = out_dir / name
                if frames:
                    save_rgb_video(frames, path, fps=args.video_fps)
                    written.append(path)
                    print(f"  {name}  ({len(frames)} frames)")
                else:
                    print(f"  {name}: NO FRAMES captured")
        finally:
            try:
                env.close()
            except Exception:
                pass

    print(f"wrote {len(written)} video(s) to {out_dir}")


if __name__ == "__main__":
    main()
