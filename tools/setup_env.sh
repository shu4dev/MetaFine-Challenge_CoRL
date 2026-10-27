#!/usr/bin/env bash
# Create the pinned conda env from requirements/lock-linux-cu124.txt, then run the
# smoke tests and the README verify snippet. Linux x86_64 only; elsewhere follow the
# staged install in docs/SETUP.md.
#
# Usage: tools/setup_env.sh [env_name]      (default: cs593)
set -euo pipefail

ENV_NAME="${1:-cs593}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$REPO_ROOT/requirements/lock-linux-cu124.txt"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

if [[ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]]; then
  echo "The lock file targets Linux x86_64; follow the staged install in docs/SETUP.md." >&2
  exit 1
fi

# conda is a shell function that non-interactive shells often lack.
if ! command -v conda >/dev/null 2>&1; then
  for base in ~/miniconda3 ~/anaconda3 ~/miniforge3 /opt/conda; do
    if [[ -f "$base/etc/profile.d/conda.sh" ]]; then source "$base/etc/profile.d/conda.sh"; break; fi
  done
fi
command -v conda >/dev/null 2>&1 || { echo "conda not found" >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env '$ENV_NAME' already exists; remove it (conda env remove -n $ENV_NAME) or pass another name." >&2
  exit 1
fi

echo "==> Creating conda env '$ENV_NAME' (Python 3.10)"
conda create -y -q -n "$ENV_NAME" python=3.10
# Keep ~/.local packages from satisfying or shadowing requirements inside the env.
conda env config vars set -n "$ENV_NAME" PYTHONNOUSERSITE=1 >/dev/null
conda activate "$ENV_NAME"

echo "==> Installing the locked package set (no resolver)"
python -m pip install -q --no-deps -r "$LOCK" --extra-index-url "$TORCH_INDEX"
python -m pip install -q --no-deps -e "$REPO_ROOT"

echo "==> pip check (one rerun-sdk/numpy note is expected)"
unexpected="$(python -m pip check 2>&1 | grep -v -E '^rerun-sdk .* numpy>=2' || true)"
if [[ -n "$unexpected" && "$unexpected" != "No broken requirements found." ]]; then
  echo "$unexpected" >&2
  exit 1
fi

cd "$REPO_ROOT"
echo "==> Smoke tests"
python -m pytest -q tests/test_smoke.py

echo "==> README verify snippet (needs Vulkan; tier B and up)"
python -c "import core.env, core.skill; import gymnasium as gym; \
env = gym.make('grasp_part', object_name='3558', part_name='cap'); \
print('Ready:', type(env.unwrapped).__name__); env.close()" 2>&1 | grep -E 'Ready:|Error' \
  || echo "WARN: verify snippet failed; see the Vulkan section of docs/SETUP.md"

cat <<EOF

Done. Next (docs/SETUP.md):
  conda activate $ENV_NAME
  conda env config vars set -n $ENV_NAME DATA_ROOT=... PI0_BASE=... TOKENIZER_PATH=... CKPT_ROOT=... HF_STORE=cs593-metafine
  hf auth login, download the demos, python tools/check_demos.py
  add your row to docs/SETUP.md and pip freeze --exclude-editable > docs/freeze/<github-user>-<gpu>.txt
EOF
