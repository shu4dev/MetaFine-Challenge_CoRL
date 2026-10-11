#!/usr/bin/env bash
# Feasibility test of the motion-prior loss on T3. Starting from the LoRA 20k baseline
# (merged), train a fresh LoRA for 2000 steps with PRIOR_LAMBDA = 0 (control), 1e-4 and 1e-3,
# merge, evaluate each on the 100 dev scenes (Understanding, replan 25, torch seed 0), report.
set -u
REPO=/home/lab250-4/MetaFine-Challenge_CoRL
BASE=$REPO/outputs/lora_pi0_toggle_switch_table_s1000/checkpoints/020000/merged_model
SEEDS=$REPO/eval/seeds/dev/toggle_switch_table_100.json
BASELINE_RUN=$REPO/runs/2026-10-11_041055
LOG=$REPO/gmm_test/run_prior_test.log
source ~/venvs/metafine/env.sh
cd "$REPO"
export ACTION_STATS=$REPO/demos/CoRL/toggle_switch_table/mixed/lerobot/meta/stats.json

RUNS=("$BASELINE_RUN")
for L in 0 1e-4 1e-3; do
  NAME=gmmtest_lambda$L
  OUT=$REPO/outputs/$NAME
  echo "== $(date) training $NAME" >> "$LOG"
  AUX_LOG=$REPO/gmm_test/aux_$NAME.csv PRIOR_LAMBDA=$L python gmm_test/train_prior.py \
    --policy.path="$BASE" \
    --dataset.repo_id=toggle_switch_table_mixed \
    --dataset.root="$REPO/demos/CoRL/toggle_switch_table/mixed/lerobot" --dataset.video_backend=pyav \
    --output_dir="$OUT" --job_name="$NAME" --steps=2000 --batch_size=4 --num_workers=8 --seed=1000 \
    --save_freq=2000 --log_freq=100 --policy.push_to_hub=false --policy.device=cuda \
    --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --policy.train_expert_only=false \
    --policy.optimizer_lr=1e-4 --policy.scheduler_decay_lr=1e-5 \
    --peft.method_type=LORA --peft.r=16 --peft.target_modules=all-linear \
    '--rename_map={"observation.images.base_camera":"observation.images.base_0_rgb","observation.images.hand_camera":"observation.images.left_wrist_0_rgb"}' \
    > "$REPO/gmm_test/train_$NAME.log" 2>&1
  echo "== $(date) training exit $?" >> "$LOG"
  CK=$OUT/checkpoints/002000
  [ -d "$CK/pretrained_model" ] || { echo "== no checkpoint for $NAME; stopping" >> "$LOG"; exit 1; }
  CUDA_VISIBLE_DEVICES= python tools/merge_lora.py "$CK/pretrained_model" "$CK/merged_model" >> "$LOG" 2>&1
  python tools/run_eval.py --task toggle_switch_table --policy-path "$CK/merged_model" \
      --seeds "$SEEDS" --mode understanding --torch-seed 0 --n-action-steps 25 \
      --note "$NAME: LoRA 20k + 2000 steps, prior lambda $L; 100 dev seeds, replan 25" 2>&1 \
      | grep -E "SR=|per-variant|confusion|run folder" >> "$LOG"
  RUNS+=("$(ls -d "$REPO"/runs/*/ | sort | tail -1)")
  rm -rf "$CK/merged_model"   # 6.6 GB; recreate with tools/merge_lora.py if needed
done
python tools/make_report.py "${RUNS[@]}" --out "$REPO/reports/t3_gmm_prior_test.pdf" >> "$LOG" 2>&1
echo "== $(date) compare exit $?" >> "$LOG"
