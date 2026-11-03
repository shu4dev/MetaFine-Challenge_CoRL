#!/usr/bin/env bash
# Read-only report of a GPU machine before launching CS593 runs: GPU, CPU, RAM, disk, env,
# repo clone, demos, checkpoints, running jobs. Changes nothing.
#
#   bash tools/inspect_machine.sh                 # on the machine itself
#   ssh corallab4 'bash -s' < tools/inspect_machine.sh   # from a laptop
#
# Optional: REPO=/path/to/MetaFine-Challenge_CoRL ENV_NAME=cs593 bash tools/inspect_machine.sh
set +e
ENV_NAME="${ENV_NAME:-cs593}"
hr() { printf '\n== %s\n' "$1"; }

hr "host"
hostname; uname -srm; uptime 2>/dev/null | sed 's/.*load/load/'
date

hr "gpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv
  echo "-- compute apps:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
else
  echo "nvidia-smi: not found"
fi

hr "cpu / ram"
echo "cpus: $(nproc 2>/dev/null)"; free -h 2>/dev/null | head -2

hr "disk"
df -h "$HOME" /tmp 2>/dev/null | awk 'NR==1 || !seen[$0]++'
for d in /data /scratch /mnt /workspace /ephemeral /home; do [ -d "$d" ] && df -h "$d" 2>/dev/null | tail -1 | sed "s|^|$d: |"; done
command -v quota >/dev/null 2>&1 && quota -s 2>/dev/null | head -6
echo "-- largest in \$HOME:"; du -sh "$HOME"/* "$HOME"/.cache 2>/dev/null | sort -rh | head -10

hr "tools"
for t in git tmux screen conda mamba micromamba python python3 pip nvcc huggingface-cli hf uv brev; do
  printf '%-16s %s\n' "$t" "$(command -v "$t" 2>/dev/null || echo -)"
done
command -v tmux >/dev/null 2>&1 && { echo "-- tmux sessions:"; tmux ls 2>/dev/null || echo "(none)"; }
command -v screen >/dev/null 2>&1 && screen -ls 2>/dev/null | head -5

hr "conda envs"
if command -v conda >/dev/null 2>&1; then
  conda env list 2>/dev/null
else
  for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda"; do
    [ -d "$base/envs" ] && { echo "$base/envs:"; ls "$base/envs"; }
  done
fi

hr "python stack in env $ENV_NAME"
PY=""
for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda"; do
  [ -x "$base/envs/$ENV_NAME/bin/python" ] && PY="$base/envs/$ENV_NAME/bin/python" && break
done
if [ -z "$PY" ] && command -v conda >/dev/null 2>&1; then
  PY="$(conda run -n "$ENV_NAME" python -c 'import sys;print(sys.executable)' 2>/dev/null)"
fi
if [ -n "$PY" ]; then
  echo "python: $PY"
  "$PY" - <<'PY'
import importlib, sys
print("version", sys.version.split()[0])
for m in ["torch", "lerobot", "peft", "sklearn", "mani_skill", "sapien", "transformers", "numpy", "pandas", "scipy"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:14s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"{m:14s} MISSING ({type(e).__name__})")
try:
    import torch
    print("cuda available:", torch.cuda.is_available(), "| torch cuda:", torch.version.cuda,
          "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
except Exception as e:
    print("torch check failed:", e)
PY
  command -v conda >/dev/null 2>&1 && { echo "-- conda env config vars:"; conda env config vars list -n "$ENV_NAME" 2>/dev/null | sed 's/\(HF_TOKEN *= *\).*/\1<set>/'; }
else
  echo "env $ENV_NAME: not found"
fi

hr "env vars (login shell)"
env | grep -E '^(DATA_ROOT|PI0_BASE|TOKENIZER_PATH|CKPT_ROOT|HF_STORE|HF_HOME|HF_TOKEN|CUDA_VISIBLE_DEVICES|HF_HUB_OFFLINE)=' | sed 's/\(HF_TOKEN=\).*/\1<set>/'

hr "repo clones"
found=0
for d in "${REPO:-}" "$HOME/MetaFine-Challenge_CoRL" "$HOME"/*/MetaFine-Challenge_CoRL "$HOME"/*/*/MetaFine-Challenge_CoRL; do
  [ -n "$d" ] && [ -d "$d/.git" ] || continue
  found=1
  echo "$d"; git -C "$d" status -sb 2>/dev/null | head -3; git -C "$d" log --oneline -1 2>/dev/null
  git -C "$d" branch -r 2>/dev/null | grep -E 't3-lora-baseline|clever-lovelace' | sed 's/^/  remote: /'
  for sub in demos/CoRL outputs runs eval_runs gmm_test; do
    [ -e "$d/$sub" ] && printf '  %-12s %s\n' "$sub" "$(du -sh "$d/$sub" 2>/dev/null | cut -f1)  $(ls "$d/$sub" 2>/dev/null | head -6 | tr '\n' ' ')"
  done
  for t in toggle_switch_table grasp_move_mug; do
    [ -d "$d/demos/CoRL/$t/mixed/lerobot" ] && printf '  demos %-20s %s\n' "$t" "$(du -sh "$d/demos/CoRL/$t" 2>/dev/null | cut -f1)"
  done
done
[ "$found" = 0 ] && echo "(no clone found under \$HOME; set REPO=...)"

hr "hugging face cache / checkpoints"
HFC="${HF_HOME:-$HOME/.cache/huggingface}"
[ -d "$HFC" ] && { du -sh "$HFC" 2>/dev/null; ls "$HFC/hub" 2>/dev/null | head -12; }
for d in "${PI0_BASE:-}" "${TOKENIZER_PATH:-}" "${CKPT_ROOT:-}" "${DATA_ROOT:-}"; do
  [ -n "$d" ] && printf '%-60s %s\n' "$d" "$([ -e "$d" ] && du -sh "$d" 2>/dev/null | cut -f1 || echo MISSING)"
done

hr "busiest processes"
ps -eo pid,user,etime,%cpu,%mem,comm --sort=-%cpu 2>/dev/null | head -8
