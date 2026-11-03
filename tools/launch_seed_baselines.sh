#!/usr/bin/env bash
# Launch the pi0 LoRA baseline on T3 for one training seed, with Josh's recipe
# (tools/t3_lora_baseline.sh on branch t3-lora-baseline), then evaluate it.
#
# Usage (one GPU per seed):
#   SEED=1001 STEPS=20000 tools/launch_seed_baselines.sh          # about 12 h on an RTX 3090 (2.2 s/step at batch 4)
#   SEED=1002 STEPS=20000 tools/launch_seed_baselines.sh
#   EVAL_MODE=both SEED=1001 tools/launch_seed_baselines.sh eval   # official eval, understanding + perception
#
# Difference from Josh's run: he trained 10k steps and resumed to 20k (tools/resume_lora.py, which restarts the
# cosine schedule); this script trains the full STEPS in one run with the same flags, so the budget is equal and
# the schedule is a single cosine decay. Record this in the registry row.
#
# Requirements: the cs593 env from tools/setup_env.sh, peft installed, $PI0_BASE and
# $TOKENIZER_PATH set as in docs/SETUP.md, the T3 demos under demos/CoRL/toggle_switch_table/mixed/lerobot,
# and tools/run_eval.py from branch t3-lora-baseline (merge or copy it first).
set -euo pipefail
SEED="${SEED:-1001}"
STEPS="${STEPS:-20000}"
TASK=toggle_switch_table
RUN="pi0_${TASK}_s${SEED}"                     # run naming: <method>_<task_id>_s<seed> (issue #14)
OUT="outputs/${RUN}"
SEEDS_FILE="${SEEDS_FILE:-eval/seeds/dev/${TASK}_100.json}"
EVAL_MODE="${EVAL_MODE:-understanding}"       # or: both (adds the camera/light perception sweep)
N_ACTION_STEPS="${N_ACTION_STEPS:-25}"         # execute 25 of the 50-step chunk per query (Josh's setting)
TORCH_SEED="${TORCH_SEED:-0}"                  # eval-time torch seed, as in the seed-1000 baseline eval
mode="${1:-train}"

if [[ "$mode" == "train" ]]; then
  lerobot-train \
    --policy.path="${PI0_BASE:?set PI0_BASE (see docs/SETUP.md)}" \
    --dataset.repo_id="${TASK}_mixed" \
    --dataset.root="$(pwd)/demos/CoRL/${TASK}/mixed/lerobot" \
    --dataset.video_backend=pyav \
    --output_dir="$OUT" --job_name="$RUN" \
    --steps="$STEPS" --batch_size=4 --num_workers=8 --seed="$SEED" \
    --save_freq=5000 --log_freq=200 \
    --policy.push_to_hub=false --policy.device=cuda \
    --policy.gradient_checkpointing=true --policy.dtype=bfloat16 \
    --policy.train_expert_only=false \
    --policy.optimizer_lr=1e-4 --policy.scheduler_decay_lr=1e-5 \
    --peft.method_type=LORA --peft.r=16 --peft.target_modules=all-linear \
    --rename_map='{"observation.images.base_camera":"observation.images.base_0_rgb","observation.images.hand_camera":"observation.images.left_wrist_0_rgb"}'
  echo "trained: $OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
elif [[ "$mode" == "eval" ]]; then
  CKPT="$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
  MERGED="$(dirname "$CKPT")/merged_model"
  if [[ ! -f "$MERGED/config.json" ]]; then
    CUDA_VISIBLE_DEVICES= python tools/merge_lora.py "$CKPT" "$MERGED"   # CPU merge, branch t3-lora-baseline
  fi
  # tools/run_eval.py (branch t3-lora-baseline) needs --task and writes <runs-dir>/<timestamp>/{official/metafine_report.json,
  # episodes.csv, episodes/}; it sets --record-dir itself. --tokenizer-path and --torch-seed pass through to eval/eval_toggle_switch.py.
  python tools/run_eval.py --task "$TASK" \
    --policy-path "$MERGED" \
    --seeds "$SEEDS_FILE" --mode "$EVAL_MODE" --n-action-steps "$N_ACTION_STEPS" --torch-seed "$TORCH_SEED" \
    --runs-dir "eval_runs/${RUN}" --note "$RUN step $STEPS (LoRA whole model, one run), replan every $N_ACTION_STEPS" \
    --tokenizer-path "${TOKENIZER_PATH:?set TOKENIZER_PATH (see docs/SETUP.md)}"
  echo "report: $(ls -dt eval_runs/${RUN}/*/official/metafine_report.json | head -1)"
else
  echo "usage: [SEED=1001] [STEPS=20000] $0 [train|eval]" >&2; exit 2
fi
