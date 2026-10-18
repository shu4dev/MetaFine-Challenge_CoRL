#!/usr/bin/env bash
# Target view in pi0's third image slot (tools/target_mask.py). From smooth 4k (merged), train a fresh
# LoRA 6000 steps on the smooth demos WITH the target view, checkpoints every 2000, eval each with
# --target-mask (100 dev scenes x red/blue, replan 25, torch seed 0). Then a control: same 6000 steps
# WITHOUT the view, so "more training" can be told apart from "the view helped".
set -u
REPO=/home/lab250-4/MetaFine-Challenge_CoRL
D=$REPO/gmm_test/color_mask
LOG=$D/run_mask_test.log
SEEDS=$REPO/eval/seeds/dev/toggle_switch_table_100.json
START=$REPO/outputs/smooth_demos_more/checkpoints/002000/merged_model   # smooth 4k, full weights
DATA=$REPO/gmm_test/smooth_demos/lerobot_smooth
source ~/venvs/metafine/env.sh
cd "$REPO"
[ -d "$START" ] || { echo "== missing $START" >> "$LOG"; exit 1; }

for ARM in mask nomask; do
  NAME=smooth4k_$ARM
  OUT=$REPO/outputs/$NAME
  if [ $ARM = mask ]; then TRAIN="python $D/train_mask.py"; EVAL=--target-mask; else TRAIN=lerobot-train; EVAL=; fi
  echo "== $(date) training $NAME" >> "$LOG"
  $TRAIN \
    --policy.path="$START" \
    --dataset.repo_id=lerobot_smooth --dataset.root="$DATA" --dataset.video_backend=pyav \
    --output_dir="$OUT" --job_name="$NAME" --steps=6000 --batch_size=4 --num_workers=8 --seed=1000 \
    --save_freq=2000 --log_freq=100 --policy.push_to_hub=false --policy.device=cuda \
    --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --policy.train_expert_only=false \
    --policy.optimizer_lr=1e-4 --policy.scheduler_decay_lr=1e-5 \
    --peft.method_type=LORA --peft.r=16 --peft.target_modules=all-linear \
    '--rename_map={"observation.images.base_camera":"observation.images.base_0_rgb","observation.images.hand_camera":"observation.images.left_wrist_0_rgb"}' \
    > "$D/train_$NAME.log" 2>&1
  echo "== $(date) training exit $?" >> "$LOG"
  for S in 002000 004000 006000; do
    CK=$OUT/checkpoints/$S
    [ -d "$CK/pretrained_model" ] || { echo "== no checkpoint $NAME $S" >> "$LOG"; continue; }
    CUDA_VISIBLE_DEVICES= python tools/merge_lora.py "$CK/pretrained_model" "$CK/merged_model" >> "$LOG" 2>&1 \
      || { echo "== merge failed $NAME $S" >> "$LOG"; continue; }
    python tools/run_eval.py --task toggle_switch_table --policy-path "$CK/merged_model" $EVAL \
        --seeds "$SEEDS" --mode understanding --torch-seed 0 --n-action-steps 25 \
        --note "$NAME step $S: smooth 4k + fresh LoRA $((10#$S)) steps on smooth demos ($ARM); 100 dev seeds, replan 25" 2>&1 \
        | grep -E "SR=|per-variant|confusion|run folder" >> "$LOG"
    echo "RUN $NAME $S $(ls -d "$REPO"/runs/*/ | sort | tail -1)" >> "$LOG"
    rm -rf "$CK/merged_model"
  done
  echo "== $(date) $ARM done" >> "$LOG"
  python $D/analyze_mask.py >> "$LOG" 2>&1
done
echo "== $(date) all done" >> "$LOG"   # START is kept: base of both LoRAs (re-merge needs it)
