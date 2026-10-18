#!/usr/bin/env bash
# Opposite-colour twin of every smooth demo (same scene, other slider), so scene layout alone can't tell the target.
set -u
REPO=/home/lab250-4/MetaFine-Challenge_CoRL
D=$REPO/gmm_test/smooth_demos
source ~/venvs/metafine/env.sh
cd "$REPO"
python $D/gen_smooth.py $D/jobs200_flip.txt $D/src200_flip --procs 8 2>&1 | grep -E "ok$|FAILED" > $D/gen200_flip.log
ls -d $D/src200_flip/*/ | xargs -P 8 -I{} sh -c 'python utils/replay_trajectory.py --traj-path {}trajectory.h5 -o rgb -c pd_joint_delta_pos -b physx_cpu --save-traj --no-verbose > {}replay.log 2>&1'
for d in $D/src200/*/; do [ -f "${d}trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5" ] || python utils/replay_trajectory.py --traj-path ${d}trajectory.h5 -o rgb -c pd_joint_delta_pos -b physx_cpu --save-traj --no-verbose > ${d}replay.log 2>&1 & while [ $(jobs -r | wc -l) -ge 8 ]; do sleep 1; done; done; wait
DIRS=$(ls -d $D/src200/*/ $D/src200_flip/*/ | while read d; do grep -q "1/1=100" "${d}replay.log" 2>/dev/null && echo "${d%/}"; done)
echo "replayed $(echo "$DIRS" | wc -l)"
rm -rf $D/lerobot_smooth_paired
python gmm_test/paired/to_lerobot.py $D/lerobot_smooth_paired $DIRS > $D/convert_paired.log 2>&1 || { echo "convert failed"; exit 1; }
python $D/fix_meta.py $D/lerobot_smooth_paired && echo "paired done"
