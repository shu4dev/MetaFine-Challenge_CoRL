#!/usr/bin/env bash
# "Wrong colour word" contrastive loss. Waits for the guidance test, then from the merged LoRA 20k
# trains a fresh LoRA 2000 steps on the original 200 demos (same setup as the GMM-test control,
# runs/2026-10-11_133203, SR 0.575) with margin 0.05, then 0.15 if there is time before the
# 9:30 AM deadline (the user needs the machine at 10). Eval: 100 dev scenes, replan 25, seed 0.
set -u
REPO=/home/lab250-4/MetaFine-Challenge_CoRL
BASE=$REPO/outputs/lora_pi0_toggle_switch_table_s1000/checkpoints/020000/merged_model
SEEDS=$REPO/eval/seeds/dev/toggle_switch_table_100.json
D=$REPO/gmm_test/contrast
LOG=$D/run_contrast_test.log
DEADLINE=$(date -d "2026-10-12 09:30" +%s)
source ~/venvs/metafine/env.sh
cd "$REPO"
while pgrep -f run_paired_test.sh > /dev/null || pgrep -f run_guidance_test.sh > /dev/null; do sleep 60; done

RUNS=("$REPO/runs/2026-10-11_041055" "$REPO/runs/2026-10-11_133203")
for M in 0.05 0.15; do
  LEFT=$(( DEADLINE - $(date +%s) ))
  if [ $LEFT -lt 10800 ]; then echo "== $(date) skipping margin $M: only $((LEFT/60)) min left" >> "$LOG"; break; fi
  NAME=contrast_m$M
  OUT=$REPO/outputs/$NAME
  echo "== $(date) training $NAME" >> "$LOG"
  CONTRAST_MARGIN=$M CONTRAST_LAMBDA=1 CONTRAST_LOG=$D/contrast_$NAME.csv timeout $(( LEFT - 1800 )) \
  python gmm_test/contrast/train_contrast.py \
    --policy.path="$BASE" \
    --dataset.repo_id=toggle_switch_table_mixed \
    --dataset.root="$REPO/demos/CoRL/toggle_switch_table/mixed/lerobot" --dataset.video_backend=pyav \
    --output_dir="$OUT" --job_name="$NAME" --steps=2000 --batch_size=4 --num_workers=8 --seed=1000 \
    --save_freq=2000 --log_freq=100 --policy.push_to_hub=false --policy.device=cuda \
    --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --policy.train_expert_only=false \
    --policy.optimizer_lr=1e-4 --policy.scheduler_decay_lr=1e-5 \
    --peft.method_type=LORA --peft.r=16 --peft.target_modules=all-linear \
    '--rename_map={"observation.images.base_camera":"observation.images.base_0_rgb","observation.images.hand_camera":"observation.images.left_wrist_0_rgb"}' \
    > "$D/train_$NAME.log" 2>&1
  echo "== $(date) training exit $?" >> "$LOG"
  CK=$OUT/checkpoints/002000
  [ -d "$CK/pretrained_model" ] || { echo "== no checkpoint for $NAME; stopping" >> "$LOG"; break; }
  CUDA_VISIBLE_DEVICES= python tools/merge_lora.py "$CK/pretrained_model" "$CK/merged_model" >> "$LOG" 2>&1
  timeout $(( DEADLINE - $(date +%s) )) python tools/run_eval.py --task toggle_switch_table --policy-path "$CK/merged_model" \
      --seeds "$SEEDS" --mode understanding --torch-seed 0 --n-action-steps 25 \
      --note "$NAME: LoRA 20k + 2000 steps with wrong-colour contrastive loss, margin $M; 100 dev seeds, replan 25" 2>&1 \
      | grep -E "SR=|per-variant|confusion|run folder" >> "$LOG"
  RUNS+=("$(ls -d "$REPO"/runs/*/ | sort | tail -1)")
  rm -rf "$CK/merged_model"
done
python tools/make_report.py "${RUNS[@]}" --out "$REPO/reports/t3_contrast_test.pdf" >> "$LOG" 2>&1
echo "== $(date) contrast done $?" >> "$LOG"
