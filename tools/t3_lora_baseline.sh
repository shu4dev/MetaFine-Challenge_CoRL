#!/usr/bin/env bash
# Reproduce our T3 (toggle_switch_table) pi0 baseline on one 12 GB GPU:
#   1. LoRA on the whole model (vision + language + action expert), 10k steps, batch 4
#   2. continue to 20k steps (tools/resume_lora.py; the LR warm-restarts mid-cosine)
#   3. merge the adapters into a plain checkpoint (tools/merge_lora.py)
#   4. evaluate on the 100 dev scenes, Understanding only, replanning every 25 actions
# Result on an RTX 5070 (2026-10-11): Understanding SR 0.62; about 12 h of training.
#
#   bash tools/t3_lora_baseline.sh [seed]        # default seed 1000
#
# Run from an activated venv with lerobot 0.4.4 and peft installed, and the T3 demos under
# demos/CoRL/toggle_switch_table/mixed/lerobot.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
SEED=${1:-1000}
NAME=lora_pi0_toggle_switch_table_s$SEED
OUT=$REPO/outputs/$NAME
SEEDS=$REPO/eval/seeds/dev/toggle_switch_table_100.json
cd "$REPO"

PI0_BASE=$(python -c "from huggingface_hub import snapshot_download as d; \
print(d('lerobot/pi0_base', revision='26b99b9439acb1e352439e34ee9c67af0d76efa3'))")

lerobot-train \
  --policy.path="$PI0_BASE" \
  --dataset.repo_id=toggle_switch_table_mixed \
  --dataset.root="$REPO/demos/CoRL/toggle_switch_table/mixed/lerobot" \
  --dataset.video_backend=pyav \
  --output_dir="$OUT" --job_name="$NAME" \
  --steps=10000 --batch_size=4 --num_workers=8 --seed="$SEED" \
  --save_freq=5000 --log_freq=200 \
  --policy.push_to_hub=false --policy.device=cuda \
  --policy.gradient_checkpointing=true --policy.dtype=bfloat16 \
  --policy.train_expert_only=false \
  --policy.optimizer_lr=1e-4 --policy.scheduler_decay_lr=1e-5 \
  --peft.method_type=LORA --peft.r=16 --peft.target_modules=all-linear \
  '--rename_map={"observation.images.base_camera":"observation.images.base_0_rgb","observation.images.hand_camera":"observation.images.left_wrist_0_rgb"}'

python tools/resume_lora.py \
  --config_path="$OUT/checkpoints/010000/pretrained_model/train_config.json" \
  --resume=true --steps=20000

CUDA_VISIBLE_DEVICES= python tools/merge_lora.py \
  "$OUT/checkpoints/020000/pretrained_model" "$OUT/checkpoints/020000/merged_model"

python tools/run_eval.py --task toggle_switch_table \
  --policy-path "$OUT/checkpoints/020000/merged_model" \
  --seeds "$SEEDS" --mode understanding --torch-seed 0 --n-action-steps 25 \
  --note "$NAME step 20000 (LoRA whole model), replan every 25; 100 dev seeds, understanding only"
