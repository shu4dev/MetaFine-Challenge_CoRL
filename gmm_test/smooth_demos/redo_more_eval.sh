#!/usr/bin/env bash
# Redo merge + eval of smooth_demos_more (first try merged onto a deleted base -> untrained model, SR 0).
set -u
REPO=/home/lab250-4/MetaFine-Challenge_CoRL
D=$REPO/gmm_test/smooth_demos; LOG=$D/run_smooth_more.log
SEEDS=$REPO/eval/seeds/dev/toggle_switch_table_100.json
START=$REPO/outputs/smooth_demos_2k/checkpoints/002000
source ~/venvs/metafine/env.sh; cd "$REPO"
echo "== $(date) REDO merge+eval" >> "$LOG"
CUDA_VISIBLE_DEVICES= python tools/merge_lora.py "$START/pretrained_model" "$START/merged_model" >> "$LOG" 2>&1 || exit 1
for S in 002000 004000; do
  CK=$REPO/outputs/smooth_demos_more/checkpoints/$S
  rm -rf "$CK/merged_model"
  CUDA_VISIBLE_DEVICES= python tools/merge_lora.py "$CK/pretrained_model" "$CK/merged_model" >> "$LOG" 2>&1 || exit 1
  python tools/run_eval.py --task toggle_switch_table --policy-path "$CK/merged_model" \
      --seeds "$SEEDS" --mode understanding --torch-seed 0 --n-action-steps 25 \
      --note "smooth demos, $((10#$S + 2000)) steps total (smooth_demos_2k + fresh LoRA $((10#$S)) steps); 100 dev seeds, replan 25" 2>&1 \
      | grep -E "SR=|per-variant|confusion|run folder" >> "$LOG"
  rm -rf "$CK/merged_model"
done
rm -rf "$START/merged_model"
echo "== $(date) redo done" >> "$LOG"
