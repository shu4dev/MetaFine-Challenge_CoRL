#!/usr/bin/env bash
# Do smoother demos give a smoother policy? Remake the 200 T3 training demos (same scenes and colours)
# as one continuous joint motion (gen_smooth.py), then from the merged LoRA 20k train a fresh LoRA
# 2000 steps on them, same setup as the GMM-test control (runs/2026-10-11_133203: 2000 steps on the
# original demos, SR 0.575). Eval: 100 dev scenes, replan 25, torch seed 0.
set -u
REPO=/home/lab250-4/MetaFine-Challenge_CoRL
BASE=$REPO/outputs/lora_pi0_toggle_switch_table_s1000/checkpoints/020000/merged_model
SEEDS=$REPO/eval/seeds/dev/toggle_switch_table_100.json
D=$REPO/gmm_test/smooth_demos
LOG=$D/run_smooth_test.log
NAME=smooth_demos_2k
OUT=$REPO/outputs/$NAME
source ~/venvs/metafine/env.sh
cd "$REPO"

echo "== $(date) generating" >> "$LOG"
python $D/gen_smooth.py $D/jobs200.txt $D/src200 --procs 16 2>&1 | grep -E "ok$|FAILED" > $D/gen200.log
echo "generated $(grep -c 'ok$' $D/gen200.log)/200" >> "$LOG"

echo "== $(date) replaying" >> "$LOG"
ls -d $D/src200/*/ | xargs -P 16 -I{} sh -c 'python utils/replay_trajectory.py --traj-path {}trajectory.h5 -o rgb -c pd_joint_delta_pos -b physx_cpu --save-traj --no-verbose > {}replay.log 2>&1'
DIRS=$(ls -d $D/src200/*/ | while read d; do grep -q "1/1=100" "${d}replay.log" 2>/dev/null && echo "${d%/}"; done)
echo "replayed $(echo "$DIRS" | wc -l)" >> "$LOG"

echo "== $(date) converting" >> "$LOG"
rm -rf $D/lerobot_smooth
python gmm_test/paired/to_lerobot.py $D/lerobot_smooth $DIRS > $D/convert.log 2>&1 || { echo "convert failed" >> "$LOG"; exit 1; }
python $D/fix_meta.py $D/lerobot_smooth >> "$LOG" 2>&1 || { echo "fix_meta failed" >> "$LOG"; exit 1; }
rm -f $D/src200/*/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5   # big; recreate with the replay line above

echo "== $(date) training $NAME" >> "$LOG"
lerobot-train \
  --policy.path="$BASE" \
  --dataset.repo_id=lerobot_smooth \
  --dataset.root="$D/lerobot_smooth" --dataset.video_backend=pyav \
  --output_dir="$OUT" --job_name="$NAME" --steps=2000 --batch_size=4 --num_workers=8 --seed=1000 \
  --save_freq=2000 --log_freq=100 --policy.push_to_hub=false --policy.device=cuda \
  --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --policy.train_expert_only=false \
  --policy.optimizer_lr=1e-4 --policy.scheduler_decay_lr=1e-5 \
  --peft.method_type=LORA --peft.r=16 --peft.target_modules=all-linear \
  '--rename_map={"observation.images.base_camera":"observation.images.base_0_rgb","observation.images.hand_camera":"observation.images.left_wrist_0_rgb"}' \
  > "$D/train_$NAME.log" 2>&1
echo "== $(date) training exit $?" >> "$LOG"
CK=$OUT/checkpoints/002000
[ -d "$CK/pretrained_model" ] || { echo "== no checkpoint; stopping" >> "$LOG"; exit 1; }
CUDA_VISIBLE_DEVICES= python tools/merge_lora.py "$CK/pretrained_model" "$CK/merged_model" >> "$LOG" 2>&1
python tools/run_eval.py --task toggle_switch_table --policy-path "$CK/merged_model" \
    --seeds "$SEEDS" --mode understanding --torch-seed 0 --n-action-steps 25 \
    --note "$NAME: LoRA 20k + 2000 steps on the 200 training scenes remade as continuous-motion demos; 100 dev seeds, replan 25" 2>&1 \
    | grep -E "SR=|per-variant|confusion|run folder" >> "$LOG"
NEW=$(ls -d "$REPO"/runs/*/ | sort | tail -1)
rm -rf "$CK/merged_model"
python tools/make_report.py "$REPO/runs/2026-10-11_041055" "$REPO/runs/2026-10-11_133203" "$NEW" \
    --out "$REPO/reports/t3_smooth_demos_test.pdf" >> "$LOG" 2>&1
echo "== $(date) smooth done $?" >> "$LOG"
