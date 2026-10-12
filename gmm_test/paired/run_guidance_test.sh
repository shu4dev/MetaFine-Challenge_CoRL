#!/usr/bin/env bash
# After the paired-demo test finishes: contrastive red<->blue guidance on the LoRA 20k baseline,
# w = 0.5, 1, 2, 4; 100 dev scenes, Understanding, replan 25, torch seed 0; then a report.
set -u
REPO=/home/lab250-4/MetaFine-Challenge_CoRL
BASE=$REPO/outputs/lora_pi0_toggle_switch_table_s1000/checkpoints/020000/merged_model
SEEDS=$REPO/eval/seeds/dev/toggle_switch_table_100.json
LOG=$REPO/gmm_test/paired/run_guidance_test.log
source ~/venvs/metafine/env.sh
cd "$REPO"
while pgrep -f run_paired_test.sh > /dev/null; do sleep 60; done
RUNS=("$REPO/runs/2026-10-11_041055")
for W in 0.5 1 2 4; do
  echo "== $(date) guidance w=$W" >> "$LOG"
  python tools/run_eval.py --task toggle_switch_table --policy-path "$BASE" \
      --seeds "$SEEDS" --mode understanding --torch-seed 0 --n-action-steps 25 --guidance $W \
      --note "LoRA 20k + red/blue contrastive guidance w=$W; 100 dev seeds, replan 25" 2>&1 \
      | grep -E "SR=|per-variant|confusion|run folder|Traceback|Error:" >> "$LOG"
  RUNS+=("$(ls -d "$REPO"/runs/*/ | sort | tail -1)")
done
python tools/make_report.py "${RUNS[@]}" --out "$REPO/reports/t3_color_guidance_test.pdf" >> "$LOG" 2>&1
echo "== $(date) guidance done $?" >> "$LOG"
