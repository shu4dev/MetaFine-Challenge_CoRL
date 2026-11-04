#!/usr/bin/env python3
"""Measure how much T3 success changes when only the torch (sampling) seed changes.

The same merged model is evaluated on the same scene seeds several times, each time with a
different ``--torch-seed``. Differences between those runs come only from pi0's random noise
at inference, so they show how big a "real" difference between two models must be.

Run (lab machine, after ``source ~/venvs/metafine/env.sh``; about 10 min per torch seed)::

    python tools/seed_variance.py \\
        --policy-path outputs/lora_pi0_toggle_switch_table_s1000/checkpoints/020000/merged_model \\
        --torch-seeds 0 1 2 3 4

Each torch seed gets its own folder ``<out-dir>/seed<N>/<timestamp>/`` (a normal run_eval run).
Finished seeds are skipped, so an interrupted sweep can simply be restarted. To only
re-summarise existing runs::

    python tools/seed_variance.py --aggregate-only --out-dir runs_seedvar

Outputs in ``<out-dir>``: seed_variance.csv (one row per torch seed), seed_variance_summary.txt,
and, if matplotlib is installed, one PNG per figure.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
TRUE = {"true", "1", "1.0", "yes"}


def truthy(value) -> bool:
    return str(value).strip().lower() in TRUE


def latest_csv(seed_dir: Path) -> Path | None:
    csvs = sorted(seed_dir.glob("*/episodes.csv"))
    return csvs[-1] if csvs else None


def read_episodes(path: Path) -> dict[str, dict]:
    """episode_id -> {success, outcome} for the understanding protocol."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        need = {"episode_id", "variant", "success", "red_toggled", "blue_toggled"}
        if not need <= set(cols):
            raise SystemExit(f"{path}: missing columns {sorted(need - set(cols))}; found {cols}")
        out = {}
        for row in reader:
            if row.get("protocol", "understanding") != "understanding":
                continue
            red, blue = truthy(row["red_toggled"]), truthy(row["blue_toggled"])
            target_is_red = row["variant"].strip().lower() == "red"
            target, other = (red, blue) if target_is_red else (blue, red)
            outcome = ("both" if target and other else "right" if target
                       else "wrong" if other else "none")
            out[row["episode_id"]] = {"success": truthy(row["success"]), "outcome": outcome}
        return out


def run_sweep(args) -> None:
    out_dir = Path(args.out_dir)
    for seed in args.torch_seeds:
        seed_dir = out_dir / f"seed{seed}"
        if latest_csv(_REPO / seed_dir) is not None:
            print(f"[seed_variance] torch seed {seed}: already done, skipping")
            continue
        cmd = [sys.executable, str(_REPO / "tools" / "run_eval.py"),
               "--task", "toggle_switch_table",
               "--policy-path", args.policy_path,
               "--seeds", args.seeds,
               "--mode", "understanding",
               "--n-action-steps", str(args.n_action_steps),
               "--runs-dir", str(seed_dir),
               "--note", f"seed variance sweep, torch seed {seed}",
               "--torch-seed", str(seed)]
        print(f"[seed_variance] torch seed {seed}: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=_REPO, check=True)


def summarise(out_dir: Path) -> None:
    per_seed: dict[str, dict[str, dict]] = {}
    for seed_dir in sorted(out_dir.glob("seed*"), key=lambda p: p.name):
        csv_path = latest_csv(seed_dir)
        if csv_path is not None:
            per_seed[seed_dir.name.removeprefix("seed")] = read_episodes(csv_path)
    if len(per_seed) < 2:
        raise SystemExit(f"need at least 2 finished torch seeds in {out_dir}, found {len(per_seed)}")

    ids = set.intersection(*(set(eps) for eps in per_seed.values()))
    rows = []
    for seed, eps in per_seed.items():
        eps = {k: eps[k] for k in ids}
        n = len(eps)
        counts = {k: sum(e["outcome"] == k for e in eps.values()) for k in ("right", "wrong", "both", "none")}
        rows.append({"torch_seed": seed, "n": n,
                     "success_rate": round(sum(e["success"] for e in eps.values()) / n, 4), **counts})

    csv_path = out_dir / "seed_variance.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = rows[0]["n"]
    lines = [f"torch seeds: {', '.join(r['torch_seed'] for r in rows)}   episodes per seed: {n}", ""]
    lines.append(f"{'torch seed':>10} {'SR':>7} {'right':>6} {'wrong':>6} {'both':>6} {'none':>6}")
    for r in rows:
        lines.append(f"{r['torch_seed']:>10} {r['success_rate']:>7.3f} {r['right']:>6} "
                     f"{r['wrong']:>6} {r['both']:>6} {r['none']:>6}")
    lines.append("")
    for name, vals in (("success rate", [r["success_rate"] for r in rows]),
                       ("right only (episodes)", [r["right"] for r in rows])):
        lines.append(f"{name}: mean {statistics.mean(vals):.3f}  std {statistics.stdev(vals):.3f}  "
                     f"min {min(vals):.3f}  max {max(vals):.3f}  range {max(vals) - min(vals):.3f}")
    p = statistics.mean(r["success_rate"] for r in rows)
    lines.append(f"binomial std of SR if episodes were independent: {math.sqrt(p * (1 - p) / n):.3f}")

    flips = sum(len({eps[i]["success"] for eps in per_seed.values()}) > 1 for i in ids)
    lines.append(f"episodes whose success changes with the torch seed: {flips}/{len(ids)} "
                 f"({flips / len(ids):.0%})")
    summary = "\n".join(lines)
    (out_dir / "seed_variance_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"\nwrote {csv_path}")
    plot(out_dir, rows)


def plot(out_dir: Path, rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping figures")
        return
    seeds = [r["torch_seed"] for r in rows]
    for key, label, fname in (("success_rate", "success rate (200 episodes)", "seed_variance_sr.png"),
                              ("right", "right-only episodes (of 200)", "seed_variance_right_only.png")):
        vals = [r[key] for r in rows]
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.axhline(statistics.mean(vals), color="0.6", linewidth=1)
        ax.plot(seeds, vals, "o", color="tab:blue")
        ax.set_xlabel("torch seed")
        ax.set_ylabel(label)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=200)
        plt.close(fig)
        print(f"wrote {out_dir / fname}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-path", help="merged model folder (needed unless --aggregate-only)")
    p.add_argument("--seeds", default="eval/seeds/dev/toggle_switch_table_100.json",
                   help="scene seed file (same scenes for every torch seed)")
    p.add_argument("--torch-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-action-steps", type=int, default=25)
    p.add_argument("--out-dir", default="runs_seedvar")
    p.add_argument("--aggregate-only", action="store_true", help="skip running, only summarise")
    args = p.parse_args()
    if not args.aggregate_only:
        if not args.policy_path:
            raise SystemExit("--policy-path is required unless --aggregate-only")
        run_sweep(args)
    summarise(_REPO / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))


if __name__ == "__main__":
    main()
