#!/usr/bin/env python3
"""Prove the policy-server path is transparent: same policy, same actions.

Runs the *same* episode twice against the *same* env — once with π0 loaded
in-process (what ``eval_*.py`` has always done) and once driven over HTTP by
``submission/pi0_policy_server.py`` — then compares the two action sequences
step by step.

π0 samples flow-matching noise every action chunk, so both sides must start
from the same torch RNG state or the actions legitimately differ. Launch the
server with ``--torch-seed K`` and pass the same ``--torch-seed K`` here.

Usage (server must already be running with the same checkpoint + seed)::

    python submission/pi0_policy_server.py --ckpt <CKPT> \\
        --tokenizer-path <TOK> --port 8080 --torch-seed 0 &

    python tools/parity_check.py --config eval/configs/toggle_switch_table.yaml \\
        --ckpt <CKPT> --tokenizer-path <TOK> \\
        --policy-url http://127.0.0.1:8080 --torch-seed 0 \\
        --variant red --seeds 12345
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import core.env  # noqa: F401,E402
from utils.eval_common import (  # noqa: E402
    build_base_env_kwargs,
    dump_json,
    load_pi0_policy,
    load_remote_policy,
    load_task_config,
    resolve_path,
    run_episode,
)

# Mirrors how each eval_*.py builds its Understanding-mode envs.
TASK_MODULES = {
    "grasp_part": "eval.eval_grasp_part",
    "grasp_move_mug": "eval.eval_grasp_move_mug",
    "toggle_switch_table": "eval.eval_toggle_switch",
    "put_blocks_into_boxes": "eval.eval_put_blocks",
    "insert_letter": "eval.eval_insert_letter",
}


def make_env(task_id: str, mod, cfg: dict, variant: str):
    if task_id == "grasp_move_mug":
        return mod._make_env(cfg, variant)
    if task_id == "toggle_switch_table":
        extra = mod._variant_extra(cfg, variant)
    elif task_id == "grasp_part":
        extra = {
            "object_name": cfg.get("object_name", "3558"),
            "part_name": variant,
            "eval_require_correct_part": True,
        }
    elif task_id == "put_blocks_into_boxes":
        extra = {"special_cube": variant}
    elif task_id == "insert_letter":
        extra = {"target_letter": variant}
    else:
        raise ValueError(f"unknown task_id {task_id}")
    return mod._make_env(build_base_env_kwargs(cfg, extra=extra))


def record_episode(env, policy, preprocessor, postprocessor, *, task, seed,
                   max_steps, default_last_action=0.0):
    """One in-process episode that also keeps every observation it fed the policy.

    The stepping logic mirrors ``utils.eval_common.run_episode`` exactly; the
    only addition is capturing the pre-preprocessor batch so the very same
    observations can be replayed through the policy server.
    """
    import torch

    from utils.eval_common import _extract_success, adapt_action, maniskill_obs_to_batch

    policy.reset()
    obs, info = env.reset(seed=int(seed))
    batches, actions = [], []
    last_info = info
    env_dim = int(np.prod(env.action_space.shape))
    for _ in range(int(max_steps)):
        batch = maniskill_obs_to_batch(obs, task)
        batches.append(batch)
        batch = preprocessor(batch)
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = postprocessor(action)
        act_np = np.asarray(action, dtype=np.float32)
        if act_np.ndim == 1:
            act_np = act_np[None, :]
        act_np = adapt_action(act_np, env_dim, default_last_action)
        obs, _rew, _term, _trunc, info = env.step(act_np[0])
        last_info = info
        actions.append(act_np[0].copy())
        if _extract_success(info):
            break
    return {"success": bool(_extract_success(last_info)),
            "episode_length": len(actions),
            "actions": actions,
            "batches": batches}


def replay_through_server(remote, batches) -> list:
    """Feed recorded observations to the policy server, collect its actions."""
    from utils.eval_common import adapt_action

    remote.reset()
    out = []
    for batch in batches:
        action = remote.select_action(batch)
        act_np = np.asarray(action, dtype=np.float32)
        if act_np.ndim == 1:
            act_np = act_np[None, :]
        out.append(adapt_action(act_np, 8, 0.0)[0].copy())
    return out


def compare(a: list, b: list) -> dict:
    """Compare two action sequences; report where and by how much they differ."""
    n = min(len(a), len(b))
    arr_a = np.asarray(a[:n], dtype=np.float64)
    arr_b = np.asarray(b[:n], dtype=np.float64)
    exact = [i for i in range(n) if not np.array_equal(arr_a[i], arr_b[i])]
    diff = np.abs(arr_a - arr_b) if n else np.zeros((0, 0))
    return {
        "steps_compared": n,
        "len_in_process": len(a),
        "len_remote": len(b),
        "identical": len(a) == len(b) and not exact,
        "n_steps_differing": len(exact),
        "first_differing_step": exact[0] if exact else None,
        "max_abs_diff": float(diff.max()) if n else 0.0,
        "mean_abs_diff": float(diff.mean()) if n else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True, help="π0 checkpoint for the in-process side")
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--policy-url", default="http://127.0.0.1:8080")
    p.add_argument("--device", default="cuda")
    p.add_argument("--variant", default=None, help="Instruction variant (default: first)")
    p.add_argument("--seeds", default=None, help="Comma-separated env seeds (default: first from --seeds-file)")
    p.add_argument("--seeds-file", default=None)
    p.add_argument("--max-steps", type=int, default=None, help="Override episode horizon")
    p.add_argument("--action-dim", type=int, default=8)
    p.add_argument("--torch-seed", type=int, default=0,
                   help="Must match the server's --torch-seed")
    p.add_argument("--out", default=None, help="Write the report JSON here")
    p.add_argument(
        "--mode",
        choices=["rollout", "replay"],
        default="replay",
        help="replay (default): record one in-process episode and feed the SAME "
             "observations to the server — isolates the HTTP path from any env "
             "non-determinism. rollout: run the episode twice, once per path "
             "(only meaningful when env.reset(seed) is fully reproducible).",
    )
    return p.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    cfg = load_task_config(args.config)
    task_id = cfg["task_id"]
    mod = importlib.import_module(TASK_MODULES[task_id])

    variants = dict(cfg.get("variants") or {})
    variant = args.variant or next(iter(variants))
    instruction = variants[variant]
    max_steps = int(args.max_steps or cfg.get("max_episode_steps", 300))

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    elif args.seeds_file:
        from utils.eval_common import load_seed_list

        seeds = load_seed_list(args.seeds_file)[:1]
    else:
        raise SystemExit("pass --seeds or --seeds-file")

    print(f"task={task_id} variant={variant!r} instruction={instruction!r}")
    print(f"seeds={seeds} max_steps={max_steps} torch_seed={args.torch_seed} mode={args.mode}")

    remote, r_pre, r_post = load_remote_policy(
        args.policy_url, task_id, action_dim=args.action_dim,
    )
    policy, pre, post = load_pi0_policy(
        args.ckpt, device=args.device, tokenizer_path=args.tokenizer_path,
    )

    env = make_env(task_id, mod, cfg, variant)
    reports = []
    try:
        # Seed ONCE, not per episode: the server's RNG advances continuously
        # across episodes, so reseeding only this side would desynchronise the
        # two noise streams from episode 2 onwards.
        torch.manual_seed(int(args.torch_seed))
        for seed in seeds:
            if args.mode == "replay":
                ep_a = record_episode(env, policy, pre, post, task=instruction,
                                      seed=seed, max_steps=max_steps)
                print(f"  [in-process] seed={seed} success={ep_a['success']} "
                      f"len={ep_a['episode_length']}")
                actions_b = replay_through_server(remote, ep_a["batches"])
                print(f"  [replay    ] seed={seed} "
                      f"{len(actions_b)} observations replayed to the server")
                ep_b = {"actions": actions_b, "success": ep_a["success"]}
            else:
                ep_a = run_episode(env, policy, pre, post, task=instruction,
                                   seed=seed, max_steps=max_steps)
                print(f"  [in-process] seed={seed} success={ep_a['success']} "
                      f"len={ep_a['episode_length']}")
                ep_b = run_episode(env, remote, r_pre, r_post, task=instruction,
                                   seed=seed, max_steps=max_steps)
                print(f"  [remote    ] seed={seed} success={ep_b['success']} "
                      f"len={ep_b['episode_length']}")

            rep = compare(ep_a["actions"], ep_b["actions"])
            rep.update({
                "seed": int(seed),
                "success_in_process": bool(ep_a["success"]),
                "success_remote": bool(ep_b["success"]),
                "success_match": bool(ep_a["success"]) == bool(ep_b["success"]),
            })
            reports.append(rep)
            verdict = "IDENTICAL" if rep["identical"] else "DIFFERS"
            print(f"  → {verdict}: steps={rep['steps_compared']} "
                  f"differing={rep['n_steps_differing']} "
                  f"first={rep['first_differing_step']} "
                  f"max|Δ|={rep['max_abs_diff']:.3e}")
    finally:
        try:
            env.close()
        except Exception:
            pass

    all_identical = all(r["identical"] for r in reports)
    summary = {
        "task_id": task_id,
        "variant": variant,
        "instruction": instruction,
        "torch_seed": args.torch_seed,
        "policy_url": args.policy_url,
        "ckpt": args.ckpt,
        "max_steps": max_steps,
        "mode": args.mode,
        "all_identical": all_identical,
        "episodes": reports,
    }
    if args.out:
        dump_json(resolve_path(args.out), summary)
        print(f"saved {resolve_path(args.out)}")
    print("PARITY:", "PASS — server path is transparent" if all_identical
          else "FAIL — see first_differing_step")
    sys.exit(0 if all_identical else 1)


if __name__ == "__main__":
    main()
