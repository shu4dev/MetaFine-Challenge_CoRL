#!/usr/bin/env bash
# Does seeing the SAME scene with both instructions fix the red/blue confusion?
# From the merged LoRA 20k baseline, train a fresh LoRA for $STEPS steps on
#   A) original 200 demos + 193 paired demos (other colour on the original scenes)
#   B) original 200 demos + 193 demos on new scenes (same count and colour split; control for "more data")
# then evaluate each on the 100 dev scenes (Understanding, replan 25, torch seed 0) and report.
set -u
REPO=/home/lab250-4/MetaFine-Challenge_CoRL
BASE=$REPO/outputs/lora_pi0_toggle_switch_table_s1000/checkpoints/020000/merged_model
SEEDS=$REPO/eval/seeds/dev/toggle_switch_table_100.json
BASELINE_RUN=$REPO/runs/2026-10-11_041055
D=$REPO/gmm_test/paired
LOG=$D/run_paired_test.log
STEPS=${STEPS:-6000}
source ~/venvs/metafine/env.sh
cd "$REPO"

RUNS=("$BASELINE_RUN")
for ARM in paired fresh; do
  NAME=pairtest_$ARM
  OUT=$REPO/outputs/$NAME
  echo "== $(date) training $NAME ($STEPS steps)" >> "$LOG"
  lerobot-train \
    --policy.path="$BASE" \
    --dataset.repo_id=toggle_mixed_$ARM \
    --dataset.root="$D/mixed_$ARM" --dataset.video_backend=pyav \
    --output_dir="$OUT" --job_name="$NAME" --steps=$STEPS --batch_size=4 --num_workers=8 --seed=1000 \
    --save_freq=$STEPS --log_freq=100 --policy.push_to_hub=false --policy.device=cuda \
    --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --policy.train_expert_only=false \
    --policy.optimizer_lr=1e-4 --policy.scheduler_decay_lr=1e-5 \
    --peft.method_type=LORA --peft.r=16 --peft.target_modules=all-linear \
    '--rename_map={"observation.images.base_camera":"observation.images.base_0_rgb","observation.images.hand_camera":"observation.images.left_wrist_0_rgb"}' \
    > "$D/train_$NAME.log" 2>&1
  echo "== $(date) training exit $?" >> "$LOG"
  CK=$OUT/checkpoints/$(printf %06d $STEPS)
  [ -d "$CK/pretrained_model" ] || { echo "== no checkpoint for $NAME; stopping" >> "$LOG"; exit 1; }
  CUDA_VISIBLE_DEVICES= python tools/merge_lora.py "$CK/pretrained_model" "$CK/merged_model" >> "$LOG" 2>&1
  python tools/run_eval.py --task toggle_switch_table --policy-path "$CK/merged_model" \
      --seeds "$SEEDS" --mode understanding --torch-seed 0 --n-action-steps 25 \
      --note "$NAME: LoRA 20k + $STEPS steps on original 200 + 193 $ARM demos; 100 dev seeds, replan 25" 2>&1 \
      | grep -E "SR=|per-variant|confusion|run folder" >> "$LOG"
  RUNS+=("$(ls -d "$REPO"/runs/*/ | sort | tail -1)")
  rm -rf "$CK/merged_model"   # 6.6 GB; recreate with tools/merge_lora.py if needed
done
python tools/make_report.py "${RUNS[@]}" --out "$REPO/reports/t3_paired_demos_test.pdf" >> "$LOG" 2>&1
echo "== $(date) compare exit $?" >> "$LOG"
