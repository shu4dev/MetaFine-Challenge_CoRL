#!/usr/bin/env bash
# Keep training on the smooth demos: from smooth_demos_2k (LoRA 20k + 2000 smooth steps, merged) train a
# fresh LoRA 4000 more steps, checkpoints at +2000 and +4000 (= 4k and 6k total), eval each like before.
set -u
REPO=/home/lab250-4/MetaFine-Challenge_CoRL
D=$REPO/gmm_test/smooth_demos
LOG=$D/run_smooth_more.log
SEEDS=$REPO/eval/seeds/dev/toggle_switch_table_100.json
START=$REPO/outputs/smooth_demos_2k/checkpoints/002000
NAME=smooth_demos_more
OUT=$REPO/outputs/$NAME
source ~/venvs/metafine/env.sh
cd "$REPO"
[ -d "$START/merged_model" ] || CUDA_VISIBLE_DEVICES= python tools/merge_lora.py "$START/pretrained_model" "$START/merged_model" >> "$LOG" 2>&1
echo "== $(date) training $NAME" >> "$LOG"
lerobot-train \
  --policy.path="$START/merged_model" \
  --dataset.repo_id=lerobot_smooth \
  --dataset.root="$D/lerobot_smooth" --dataset.video_backend=pyav \
  --output_dir="$OUT" --job_name="$NAME" --steps=4000 --batch_size=4 --num_workers=8 --seed=1000 \
  --save_freq=2000 --log_freq=100 --policy.push_to_hub=false --policy.device=cuda \
  --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --policy.train_expert_only=false \
  --policy.optimizer_lr=1e-4 --policy.scheduler_decay_lr=1e-5 \
  --peft.method_type=LORA --peft.r=16 --peft.target_modules=all-linear \
  '--rename_map={"observation.images.base_camera":"observation.images.base_0_rgb","observation.images.hand_camera":"observation.images.left_wrist_0_rgb"}' \
  > "$D/train_$NAME.log" 2>&1
echo "== $(date) training exit $?" >> "$LOG"
for S in 002000 004000; do
  CK=$OUT/checkpoints/$S
  [ -d "$CK/pretrained_model" ] || { echo "== no checkpoint $S" >> "$LOG"; continue; }
  CUDA_VISIBLE_DEVICES= python tools/merge_lora.py "$CK/pretrained_model" "$CK/merged_model" >> "$LOG" 2>&1
  python tools/run_eval.py --task toggle_switch_table --policy-path "$CK/merged_model" \
      --seeds "$SEEDS" --mode understanding --torch-seed 0 --n-action-steps 25 \
      --note "smooth demos, $((10#$S + 2000)) steps total (smooth_demos_2k + fresh LoRA $((10#$S)) steps); 100 dev seeds, replan 25" 2>&1 \
      | grep -E "SR=|per-variant|confusion|run folder" >> "$LOG"
  rm -rf "$CK/merged_model"
done
rm -rf "$START/merged_model"   # base of the new LoRA: delete only after merging
echo "== $(date) more done" >> "$LOG"
