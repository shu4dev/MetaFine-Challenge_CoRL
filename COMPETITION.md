# MetaFine Competition — Participant Guide

This document describes the **public competition release** of MetaFine: five manipulation tasks (T1–T5), the evaluation protocol, and what is (and is not) included in this repository.

> 🏆 **Registration, key dates, and announcements** live on the [competition page](https://robofinemani2026.github.io/competition.html). Register there first — this document covers the technical side only.

## Tasks (T1–T5)

| ID | Env | Instruction variants | Stages |
|---|---|---|---|
| T1 | `grasp_part` | cap / body | 1 — grasp instructed bottle part |
| T2 | `grasp_move_mug` | left / right / forward | 2 — grasp mug handle, translate 10 cm |
| T3 | `toggle_switch_table` | red / blue | 1 — flip the switch of a specific color |
| T4 | `put_blocks_into_boxes` | red / blue / green (special cube) | 3 — place special cube left, others right |
| T5 | `insert_letter` | C / o / R / L | 1 — insert instructed peg into slot |

Full sensor settings, success criteria, and baseline scores: [eval/README.md](eval/README.md).

## What this repo contains

| Included | Not included |
|---|---|
| Simulation environments, skills, predicates | **Official eval seeds** |
| `eval/` harness (Perception + Understanding + Behavior) | Policy checkpoints (participants train their own) |
| Competition assets (`3558`, `8848`, `100920`, `table.glb`) | Full CoRL demo HDF5 / LeRobot trees (download separately) |
| Task-graph YAMLs for T2 | Internal cluster logs / operational scripts |
| π0 baseline **reports** (`eval_runs/*/metafine_report.json`) | Evaluation videos from organizer runs |

## Downloads

### Training data (CoRL mixed demos)

Download from Hugging Face ([`hiangx/MetaFine_CoRL26`](https://huggingface.co/datasets/hiangx/MetaFine_CoRL26)) so paths match `demos/CoRL/` in this repo:

```bash
huggingface-cli download hiangx/MetaFine_CoRL26 --repo-type dataset --local-dir demos/CoRL
```

See [demos/CoRL/README.md](demos/CoRL/README.md) for per-task layout.

### Extended asset library

The competition bundle ships only the three articulated assets used by T1–T3 plus the table mesh. The full 40+ PartNet-Mobility subset is part of the MetaFine platform release ([metafine.github.io](https://metafine.github.io/)) — not required for the competition.

## Evaluation protocol

Three orthogonal metrics (see [eval/README.md](eval/README.md)):

1. **Perception** — camera + light domain randomization sweeps → AUSC (area under success curve).
2. **Understanding** — fixed scene seeds, varied language instructions → success rate + confusion matrix.
3. **Behavior** — final task success (or stage progress for multi-stage tasks).

All tasks use 512×512 RGB observations, `pd_joint_delta_pos` control, and the same `evaluate()` predicates as training demo replay.

Participant policies are evaluated through a unified HTTP `policy-server` interface. The evaluation harness owns the simulator and seeds; your server receives observations and returns actions.

### 1. Implement the policy interface

Evaluation uses the unified `policy-server` interface. Customize `Policy` in `submission/policy_server.py`: load weights once in `__init__`, clear per-episode state in `reset`, and return one action from `act`. Leave the protocol plumbing unchanged.

```python
class Policy:
    def __init__(self):
        self.model = load_your_model(...)  # loaded once

    def reset(self, task_id, instruction, action_dim):
        ...  # clear action chunks / recurrent state

    def act(self, state, images, instruction, step):
        return action  # shape: (action_dim,)
```

See `submission/pi0_policy_server.py` for a complete runnable π0 example.

### 2. Self-test the protocol

This uses fake observations and does not launch the simulator:

```bash
python submission/policy_server.py --self-test
```

### 3. Generate local dev seeds

Official seeds are hidden; local seeds are only for debugging:

```bash
python -m eval.select_eval_seeds --config eval/configs/toggle_switch_table.yaml \
  --n-seeds 5 --rng-seed 42 --out /tmp/toggle_dev.json
```

### 4. Start the policy server

```bash
# Your policy
python submission/policy_server.py --port 8080 &

# Or the complete π0 example
python submission/pi0_policy_server.py --port 8080 \
  --ckpt outputs/pi0_toggle_mixed/checkpoints/030000/pretrained_model \
  --tokenizer-path /path/to/paligemma-3b-pt-224 &
```

### 5. Run evaluation

```bash
python -m eval.eval_toggle_switch \
  --policy-url http://127.0.0.1:8080 \
  --seeds /tmp/toggle_dev.json --mode both \
  --record-dir eval_runs/toggle
```

Use `eval.eval_grasp_part`, `eval.eval_grasp_move_mug`, `eval.eval_put_blocks`, or `eval.eval_insert_letter` for the other tasks. Add `--save-video` where supported; for `put_blocks`, use `tools/record_video.py`.

## Seeds — critical policy

**Official competition evaluation uses hidden seeds that are never published.**

Why:

- `select_eval_seeds.py` is **deterministic** for a given `--rng-seed`.
- Training demo JSONs list all training `episode_seed` values (public).
- Given the script, configs, and a known `--rng-seed`, anyone could reconstruct the official seed list.

Therefore:

| Party | Seeds |
|---|---|
| Organizers | Private list from a **secret** `--rng-seed`; stored only on evaluation servers |
| Participants | Generate **local dev seeds** for debugging (`--rng-seed 42` in docs is an example only) |

Your locally generated seeds **will not** match the official hidden set. This is expected.

## Submission

Final submission format and upload portal will be announced on the [competition page](https://robofinemani2026.github.io/competition.html). Expected deliverables:

- Trained policy checkpoint(s) per task or a single multi-task checkpoint.
- Optional: self-reported local eval logs on **your own** dev seeds (not used for official ranking).

Organizers re-run submitted checkpoints on the **hidden seed set** with the shipped `eval/eval_*.py` scripts.

## Experimental environments

The platform registers 20 Gym environments total. Only the five tasks above are scored in this competition. Others (`align_to_part`, `door_env`, `stand_up`, …) are research/experimental and may have incomplete `evaluate()` implementations.

## Support

- Evaluation details: [eval/README.md](eval/README.md)
- Data layout: [demos/CoRL/README.md](demos/CoRL/README.md)
- Issues: [GitHub Issues](https://github.com/Hiangx-robotics/MetaFine-Challenge_CoRL/issues)
