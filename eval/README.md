# MetaFine evaluation — π0 baseline

Per-task evaluation for the MetaFine challenge metrics:

| Metric | Protocol |
|---|---|
| **Perception** | Sweep camera (position + rotation jointly) and ambient light; report AUSC per axis and their mean. |
| **Understanding** | Fix scene seeds; vary only the language instruction; report success rate + confusion matrix. |
| **Behavior** | Final task success (single-stage) or stage progress (multi-stage). |

Sensor settings live in `eval/configs/<task>.yaml` (copied from training replay metadata). Do **not** rely on env defaults (`224×224`) — they diverge from the collected demos (`512×512`).

---

## Tasks

Numbered T1–T5 in table order.

### T1 — `grasp_part`

| | |
|---|---|
| Env | `grasp_part` / bottle asset `3558` |
| Stages | 1 (grasp the instructed part) |
| Variants | `cap` → *"Grasp the cap of the bottle"*; `body` → *"Grasp the body of the bottle"* |
| Horizon | 300 steps |
| Sensor | 512×512, FOV ≈ 70°, `pd_joint_delta_pos`, `obs_mode=rgb` |
| Seeds (local dev) | Generate with `select_eval_seeds.py` — **not** the hidden competition set |

### T2 — `grasp_move_mug`

| | |
|---|---|
| Env | `multi_skill` + `configs/t2_mug_move_{left,right,forward}.yaml` |
| Stages | 2 (grasp handle, then translate 10 cm) |
| Variants | `left` / `right` / `forward` |
| Horizon | 400 steps |
| Sensor | 512×512, FOV ≈ 70°, `pd_joint_delta_pos`, `obs_mode=rgb` |
| Seeds (local dev) | Generate with `select_eval_seeds.py` — **not** the hidden competition set |

### T3 — `toggle_switch_table`

| | |
|---|---|
| Env | `toggle_switch_table` |
| Stages | 1 (flip the switch of a specific color) |
| Variants | `red` / `blue` (specific color in the instruction) |
| Horizon | from env / eval config |
| Sensor | 512×512, FOV ≈ 70°, `pd_joint_delta_pos`, `obs_mode=rgb` |
| Seeds (local dev) | Generate with `select_eval_seeds.py` — **not** the hidden competition set |

### T4 — `put_blocks_into_boxes`

| | |
|---|---|
| Env | `put_blocks_into_boxes` |
| Stages | 3 (place special cube in left box, then remaining cubes in right) |
| Variants | `red` / `blue` / `green` (special cube color in the instruction) |
| Horizon | 800 steps |
| Sensor | 512×512, FOV ≈ 57°, `pd_joint_delta_pos`, `obs_mode=rgb` |
| Seeds (local dev) | Generate with `select_eval_seeds.py` — **not** the hidden competition set |

### T5 — `insert_letter`

| | |
|---|---|
| Env | `insert_letter` |
| Stages | 1 (insert the instructed peg into its matching slot) |
| Variants | `C` / `o` / `R` / `L` |
| Horizon | 400 steps |
| Sensor | 512×512, FOV ≈ 70°, `pd_joint_delta_pos`, `obs_mode=rgb` |
| Seeds (local dev) | Generate with `select_eval_seeds.py` — **not** the hidden competition set |

T4 success uses `_is_in_box` + `agent.is_grasping`. T1 scores below are under the **strict contact criterion**. T2 policy eval uses `grasped_contact_fallback` (off by default for every other task).

---

## Success criterion (T1)

```
success = is_grasping(link)                      # agent.is_grasping — true contact
          and part_links[link] == part_name      # link_0=cap / link_1=body
          and not (free object knocked over / away)
          holds for grasp_hold_steps (=5) consecutive steps
```

`assets/3558/model_data.json` maps `part_links`: `cap → [link_0]`, `body → [link_1]`.

While grasping, only tilt (> 30°) counts as disturbance (lifting is allowed). When not grasping, both tilt and XY displacement (> 5 cm) gate success — so knocking the bottle over without holding it cannot pass.

---

## π0 baseline scores

π0 trained 30k steps per task · 20 seeds · strict criterion for T1.

| Task | Perc. AUSC (mean) | cam / light | Clean SR | Understanding | Behavior |
|---|---|---|---|---|---|
| T1 `grasp_part` | **0.296** | 0.233 / 0.358 | 0.25 | 0.35 | 0.35 |
| T2 `grasp_move_mug` | **0.892** | 0.825 / 0.958 | 1.00 | 0.95 | 0.95 |
| T3 `toggle_switch_table` | **0.346** | 0.292 / 0.400 | 0.40 | 0.60 | 0.60 |
| T4 `put_blocks_into_boxes` | **0.192** | 0.150 / 0.233 | 0.25 | 0.35 | 0.45 (mean stage progress) |
| T5 `insert_letter` | **0.025** | 0.033 / 0.017 | 0.10 | 0.05 | 0.05 |

Source: `eval_runs/*/metafine_report.json` (seed-free baseline reports shipped with this repo).

### Perception DR curves

| Profile | T1 | T2 | T3 | T4 | T5 |
|---|---|---|---|---|---|
| clean | 0.25 | 1.00 | 0.40 | 0.25 | 0.10 |
| cam_l1 | 0.40 | 1.00 | 0.55 | 0.25 | 0.05 |
| cam_l2 | 0.10 | 0.55 | 0.10 | 0.05 | 0.00 |
| cam_l3 | 0.15 | 0.85 | 0.05 | 0.05 | 0.00 |
| light_l1 | 0.35 | 1.00 | 0.40 | 0.35 | 0.00 |
| light_l2 | 0.40 | 0.90 | 0.50 | 0.15 | 0.00 |
| light_l3 | 0.40 | 0.95 | 0.20 | 0.15 | 0.00 |

### Understanding confusion

**T1** (rows = instructed part; cells = detected / none):

| instructed → | cap | body | none |
|---|---|---|---|
| cap | 7 | 4 | 9 |
| body | 0 | 7 | 13 |

**T2** (rows = instructed direction; cells = classified mug motion / none):

| instructed → | left | right | forward | none |
|---|---|---|---|---|
| left | 19 | 0 | 0 | 1 |
| right | 0 | 19 | 0 | 1 |
| forward | 3 | 1 | 16 | 0 |

Per-variant SR is 0.95 for every T2 variant.

**T3** (rows = instructed color; cells = which switch flipped / both / none):

| instructed → | red | blue | both | none |
|---|---|---|---|---|
| red | 4 | 5 | 7 | 4 |
| blue | 4 | 7 | 7 | 2 |

Per-variant SR: red 0.55, blue 0.65.

**T4** (rows = special-cube color; cells = which cube ended in left box / none):

| instructed → | red | blue | green | none |
|---|---|---|---|---|
| red | 10 | 2 | 0 | 8 |
| blue | 0 | 16 | 0 | 4 |
| green | 4 | 0 | 10 | 6 |

Per-variant SR is 0.35 for every T1/T4 variant.

**T5** (rows = instructed letter; cells = which peg seated / none):

| instructed → | C | o | R | L | none |
|---|---|---|---|---|---|
| C | 1 | 0 | 0 | 0 | 19 |
| o | 0 | 1 | 0 | 0 | 19 |
| R | 0 | 0 | 2 | 0 | 18 |
| L | 0 | 0 | 0 | 0 | 20 |

Per-variant SR: C 0.05, o 0.05, R 0.10, L 0.00.

---

## Reproduce locally

> **Troubleshooting:** if SAPIEN crashes with `ErrorDeviceLost` on a machine with more than one Vulkan driver, pin the NVIDIA ICD first: `export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`.

### 1) Implement your policy

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

### 2) Self-test the protocol

This uses fake observations and finishes without launching the simulator:

```bash
python submission/policy_server.py --self-test
```

### 3) Generate local dev seeds

The official seed set is hidden. Generate your own seeds for debugging:

```bash
python -m eval.select_eval_seeds --config eval/configs/toggle_switch_table.yaml \
  --n-seeds 5 --rng-seed 42 --out /tmp/toggle_dev.json
```

Use the matching YAML under `eval/configs/` for each task. Local dev seeds will not match the official set.

### 4) Start a policy server

```bash
# Your implementation
python submission/policy_server.py --port 8080 &

# Or the complete π0 example
python submission/pi0_policy_server.py --port 8080 \
  --ckpt outputs/pi0_toggle_mixed/checkpoints/030000/pretrained_model \
  --tokenizer-path /path/to/paligemma-3b-pt-224 &
```

### 5) Run evaluation

```bash
python -m eval.eval_toggle_switch \
  --policy-url http://127.0.0.1:8080 \
  --seeds /tmp/toggle_dev.json --mode both \
  --record-dir eval_runs/toggle
```

Replace the module with `eval.eval_grasp_part`, `eval.eval_grasp_move_mug`, `eval.eval_put_blocks`, or `eval.eval_insert_letter` to run the other tasks. Add `--save-video` where supported; `put_blocks` videos require `tools/record_video.py`.

### 6) Aggregate → `metafine_report.json`

```bash
python -m utils.eval_report --task-id grasp_part \
  --perception eval_runs/grasp_part/perception_summary.json \
  --understanding eval_runs/grasp_part/understanding_summary.json \
  --out eval_runs/grasp_part/metafine_report.json
# repeat for grasp_move_mug, toggle_switch_table, put_blocks_into_boxes, insert_letter
```

Useful flags on the T1–T5 eval scripts:

- `--n-seeds N` — use only the first N seeds (smoke); all tasks
- `--save-video` — write RGB mp4s under `<record-dir>/videos/`; T1 / T2 / T3 / T5 (not T4)
- `--max-videos N` — cap saved videos per sweep; T2 / T5 only
- `--perception-profiles clean,cam_l1,...` — subset of DR profiles (default: all); T1 / T3 only

---

## Seed protocol

RoboTwin-style selection (implemented in `select_eval_seeds.py`):

1. Load every `episode_seed` from the task's training demo JSONs (`train_demo_jsons` in the YAML).
2. Sample candidates outside that pool.
3. Run the motion-planning expert (`obs_mode=none`) for every instruction variant.
4. Keep a seed only if all required variants plan to success under the **same** `evaluate()` used at policy eval time.
5. Write the list to a JSON file you pass to `--seeds` at eval time.

### Competition vs local development

| | Local dev | Official competition eval |
|---|---|---|
| Seed file | You generate (e.g. `/tmp/*_dev.json`) | **Hidden** — held by organizers only |
| `--rng-seed` | Any value you choose (examples use `42`) | **Private** organizer salt — never published |
| Purpose | Smoke-test the eval harness | Final leaderboard scoring |

Because sampling is deterministic given `--rng-seed`, publishing a seed list **or** a known default salt would leak the evaluation scenes. This repository intentionally ships **zero** seed JSON files under `eval/seeds/`.

---

## Output directories

Shipped baseline reports (no seed lists):

```
eval_runs/
  grasp_part/metafine_report.json
  grasp_move_mug/metafine_report.json
  toggle_switch_table/metafine_report.json
  put_blocks_into_boxes/metafine_report.json
  insert_letter/metafine_report.json
```

When you run eval locally, each task also writes `perception_summary.json` and `understanding_summary.json` under your `--record-dir`. Use `utils/eval_report.py` to produce `metafine_report.json`.

---

## Layout

```
eval/
  configs/          # per-task YAML (sensor, instructions, demo jsons)
  select_eval_seeds.py
  eval_grasp_part.py
  eval_grasp_move_mug.py
  eval_toggle_switch.py
  eval_put_blocks.py
  eval_insert_letter.py
  README.md
utils/
  eval_common.py    # shared PI0 load / rollout / DR / AUSC
  eval_report.py
```
