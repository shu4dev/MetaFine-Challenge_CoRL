#!/usr/bin/env python3
"""Append or update one run in registry/runs.jsonl (the team's registry-line format, issue #14).

Examples:
  python tools/register_run.py --run-name pi0_toggle_switch_table_s1001 --method pi0 --task-id toggle_switch_table \
      --seed 1001 --role matrix --status training --checkpoint-path outputs/pi0_toggle_switch_table_s1001/checkpoints/020000/pretrained_model
  python tools/register_run.py --run-name pi0_toggle_switch_table_s1001 --update --status done \
      --report-paths eval_runs/pi0_toggle_switch_table_s1001/official/metafine_report.json
A duplicate run_name without --update exits with code 2.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REQUIRED = ["run_name", "method", "task_id", "seed", "role", "split", "config_path", "git_sha", "lerobot_source",
            "base_ckpt_revision", "dataset_fingerprint", "env_fingerprint", "checkpoint_path", "hf_repo", "chunk_size",
            "aux_loss", "aux_lambda", "aux_prior_path", "status", "report_paths"]
ROLES = {"pilot", "smoke", "sweep", "matrix"}


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default="registry/runs.jsonl")
    ap.add_argument("--update", action="store_true", help="patch the existing line instead of appending")
    for k in REQUIRED:
        ap.add_argument("--" + k.replace("_", "-"), default=None)
    a = ap.parse_args()
    reg = Path(a.registry)
    rows = [json.loads(l) for l in reg.read_text().splitlines() if l.strip()] if reg.exists() else []
    if not a.run_name:
        ap.error("--run-name is required")
    given = {k: getattr(a, k) for k in REQUIRED if getattr(a, k) is not None}
    if "seed" in given:
        given["seed"] = int(given["seed"])
    if "chunk_size" in given:
        given["chunk_size"] = int(given["chunk_size"])
    if "aux_lambda" in given:
        given["aux_lambda"] = float(given["aux_lambda"])
    if "report_paths" in given:
        given["report_paths"] = [p for p in given["report_paths"].split(",") if p]
    if "role" in given and given["role"] not in ROLES:
        ap.error(f"--role must be one of {sorted(ROLES)}")
    idx = next((i for i, r in enumerate(rows) if r.get("run_name") == a.run_name), None)
    if idx is None and a.update:
        print(f"no run named {a.run_name}", file=sys.stderr); return 2
    if idx is not None and not a.update:
        print(f"duplicate run_name {a.run_name}; use --update", file=sys.stderr); return 2
    if a.update:
        rows[idx].update(given)
    else:
        row = {k: None for k in REQUIRED}
        row.update({"git_sha": git_sha(), "chunk_size": 50, "aux_loss": "none", "status": "planned", "report_paths": []})
        row.update(given)
        rows.append(row)
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(("updated " if a.update else "registered ") + a.run_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
