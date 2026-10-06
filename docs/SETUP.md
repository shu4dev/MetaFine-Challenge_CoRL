# Setup

How each team member installs the pinned stack, downloads the T2 and T3 demos
and checks the result. Issue: #10 [C0.1].

## Machine tiers

| Tier | Hardware | Runs |
|---|---|---|
| A | any laptop, no GPU | `tools/check_demos.py`, `pytest tests/test_smoke.py`, analysis |
| B | Linux with a Vulkan-capable GPU | SAPIEN scenes: README verify snippet, demo replay, stage labelling |
| C | NVIDIA GPU with at least 16 GB and Vulkan | pi0 inference and official evaluation |
| D | training GPU | pi0 fine-tuning; VRAM need is measured in [C3.1] (#12) |

## Members

| Member | Tier | GPU | Env | Freeze |
|---|---|---|---|---|
| Dongheon97 | B, C (D pending #12) | RTX 3090 24 GB, driver 580.178.04, Ubuntu 22.04 | conda `cs593` | [`freeze/dongheon97-rtx3090.txt`](freeze/dongheon97-rtx3090.txt) |

Add one row and one `docs/freeze/<github-user>-<gpu>.txt` (`pip freeze --exclude-editable`) per machine.

Revisions in use (record yours if they differ):

| Artifact | Hugging Face id | Revision |
|---|---|---|
| Demos | [`hiangx/MetaFine_CoRL26`](https://huggingface.co/datasets/hiangx/MetaFine_CoRL26) (dataset) | `0e86f4a420cde600324a64568949489ac31b11c7` |
| pi0 base | [`lerobot/pi0_base`](https://huggingface.co/lerobot/pi0_base) | `25c379b52ba2ff8788cab921758a3cc3fe3f77f2` |
| Tokenizer | [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224) (gated) | `35e4f46485b4d07967e7e9935bc3786aad50687c` |

## Install

Reference versions (README.md, Installation): Python 3.10, torch 2.6 (cu124),
lerobot 0.4.4, transformers 4.57.0, sapien 3.0.3, mani_skill 3.0.1, numpy 1.26.4.

### Quick install (Linux x86_64, tier B to D)

```bash
tools/setup_env.sh            # creates conda env "cs593"; pass another name as $1
```

The script creates the env, installs the exact package set of
`requirements/lock-linux-cu124.txt` with `pip install --no-deps` (no resolver, so the
conflicts below do not apply), then runs `pip check`, the smoke tests and the verify
snippet. About 3 minutes with a warm pip cache.

### Tier A laptop (macOS, Windows, no GPU)

The simulator stack is not needed for the tests, `tools/check_demos.py` or offline analysis:

```bash
python3.10 -m venv .venv && source .venv/bin/activate   # or a conda env
pip install -r requirements/tier-a.txt
pytest tests/test_smoke.py
```

On macOS the full stack does not work regardless: mani_skill's motion planner `mplib` ships
Linux wheels only (`core.skill`, `record.py`, `eval/select_eval_seeds.py` need it) and there is
no CUDA. Not tested on a Mac yet.

### Staged install (manual)

Use this when the lock does not fit (another CUDA version, a different Python). The reference set
cannot be installed in one `pip install`: lerobot 0.4.4 requires
`rerun-sdk>=0.24`, and every such release requires `numpy>=2`, while the repo
pins `numpy<2`. pip 26 also fails to resolve `imageio[ffmpeg]` when the whole
set is requested at once. Install in stages and force numpy back to 1.26.4:

```bash
conda create -n cs593 python=3.10 -y
# Keep ~/.local packages out of the env
conda env config vars set -n cs593 PYTHONNOUSERSITE=1
conda activate cs593

# 1. simulator
pip install "sapien==3.0.3" "mani_skill==3.0.1"
# 2. pi0 stack on CUDA 12.4 wheels
pip install "torch==2.6.0" "lerobot==0.4.4" "transformers==4.57.0" \
    --extra-index-url https://download.pytorch.org/whl/cu124
# 3. this repo (dependencies are already in place)
pip install --no-deps -e . "pytest>=7.0" iniconfig pluggy
# 4. versions the baseline was run with
pip install --no-deps "numpy==1.26.4" "opencv-python==4.11.0.86" \
    "opencv-python-headless==4.11.0.86" "torchcodec==0.2.1" "setuptools==80.10.2"
```

`pip check` then reports only `rerun-sdk 0.26.2 has requirement numpy>=2`.
rerun is lerobot's visualizer; training, dataset loading and evaluation do not use it.
pip also warns that transformers 4.57.0 is yanked on PyPI; it is the README's reference
version and installs and runs fine here.

Verify (tier B and up):

```bash
python -c "import core.env, core.skill; import gymnasium as gym; \
           env = gym.make('grasp_part', object_name='3558', part_name='cap'); \
           print('Ready:', type(env.unwrapped).__name__); env.close()"
# → Ready: GraspPartEnv
```

### Known pitfalls

- **Video backend.** Always pass `--dataset.video_backend=pyav` to `lerobot-train`, as the README does.
  torchcodec 0.2.1 is the release built for torch 2.6, but lerobot 0.4.4 hands it a file object it
  cannot open (`TypeError: Unknown source type`). Newer torchcodec releases fail to load against torch 2.6
  (`undefined symbol`).
- **`~/.local` leaking in.** Without `PYTHONNOUSERSITE=1`, packages under `~/.local/lib/python3.10`
  satisfy requirements inside the env and hide missing ones.
- **Shell init.** A `~/.bashrc` that returns early for non-interactive shells skips `conda init`
  and any `export` below it; set env vars with `conda env config vars set` instead (next section).

### Vulkan

SAPIEN renders the RGB observations through Vulkan. Check that the NVIDIA ICD is present:

```bash
ls /usr/share/vulkan/icd.d/nvidia_icd.json
```

If it is missing, install the NVIDIA driver's Vulkan package (`libnvidia-gl-<version>` on Ubuntu)
and `libvulkan1`. A T3 reset with RGB sensors confirms rendering:

```bash
python -c "
import gymnasium as gym, core.env
from utils.eval_common import load_task_config, build_base_env_kwargs
cfg = load_task_config('eval/configs/toggle_switch_table.yaml')
env = gym.make(cfg['env_id'], **build_base_env_kwargs(cfg)); obs, _ = env.reset(seed=0)
print(obs['sensor_data']['base_camera']['rgb'].shape)"
# → torch.Size([1, 512, 512, 3])
```

## Env vars

Five per-machine variables (stub names; [C0.3] (#14) finalizes them). Set them on the conda env
so they apply wherever it is activated:

```bash
conda env config vars set -n cs593 \
  DATA_ROOT=... PI0_BASE=... TOKENIZER_PATH=... CKPT_ROOT=... HF_STORE=...
conda activate cs593   # re-activate to load them
```

| Variable | Meaning | Dongheon97 |
|---|---|---|
| `DATA_ROOT` | demo root, the `demos/CoRL` layout | `~/workspace/cs593/MetaFine-Challenge_CoRL/demos/CoRL` |
| `PI0_BASE` | local copy of `lerobot/pi0_base`, passed as `--policy.path` | `~/workspace/cs593/models/pi0_base` |
| `TOKENIZER_PATH` | local PaliGemma tokenizer, passed as `--tokenizer-path` | `~/workspace/cs593/models/paligemma-3b-pt-224` |
| `CKPT_ROOT` | training output root | `~/workspace/cs593/checkpoints` |
| `HF_STORE` | team's Hugging Face organization | `cs593-metafine` |

### Team store

The private repos under [`cs593-metafine`](https://huggingface.co/cs593-metafine) hold what one member
produces and another consumes; folder naming is finalized in #14.

| Repo | Type | Holds |
|---|---|---|
| [`cs593-metafine/checkpoints`](https://huggingface.co/cs593-metafine/checkpoints) | model, private | `lerobot-train` checkpoints, `<run_name>/pretrained_model/` |
| [`cs593-metafine/artifacts`](https://huggingface.co/datasets/cs593-metafine/artifacts) | dataset, private | rollouts (C4), priors (C3), stage side-tables (C1), eval reports |

To upload, ask an org admin for a write invite, then give your Hugging Face token access to the org:
at <https://huggingface.co/settings/tokens>, edit the token, add `cs593-metafine` under
**Org permissions** with read and write access to repo contents. A fine-grained token scoped only to
your user fails with `403 Forbidden ... under the namespace "cs593-metafine"`.

```bash
hf upload cs593-metafine/checkpoints "$CKPT_ROOT/<run_name>/checkpoints/last/pretrained_model" \
  <run_name>/pretrained_model
hf download cs593-metafine/artifacts --repo-type dataset --include "rollouts/<run_name>/*" --local-dir artifacts
```

## Data

Log in first; anonymous downloads hit the Hub rate limit within minutes, and the tokenizer repo is gated.

```bash
hf auth login
# accept the licence at https://huggingface.co/google/paligemma-3b-pt-224
```

T2 and T3 demos, without the RGB replay `.h5` files (3 to 5 GB per variant, about 25 GB in total,
only needed to rebuild the LeRobot datasets):

```bash
hf download hiangx/MetaFine_CoRL26 --repo-type dataset --local-dir demos/CoRL \
  --include "grasp_move_mug/*/*" "toggle_switch_table/*/*" \
  --exclude "*.rgb.pd_joint_delta_pos.physx_cpu.h5"
```

The include patterns skip the task-level README files, which differ from the copies in git.
About 3 GB on disk. Even when logged in, the Hub allows 1000 requests per 5 minutes; if the
download stops with HTTP 429, wait 5 minutes and run the same command again (finished files are skipped).

The full dataset (all five tasks) is needed only on the competition submission machine ([CX.2] (#35)).

pi0 base and the tokenizer files (the PaliGemma weights are not needed):

```bash
hf download lerobot/pi0_base --local-dir "$PI0_BASE"
hf download google/paligemma-3b-pt-224 --local-dir "$TOKENIZER_PATH" \
  --include "tokenizer*" "special_tokens_map.json" "added_tokens.json" \
            "preprocessor_config.json" "config.json"
```

Check the demos and run the tests:

```bash
python tools/check_demos.py          # CHECK_DEMOS: PASS, exit 0
pytest tests/test_smoke.py           # CPU only, about 1 s
git status --short -- demos/CoRL     # prints nothing
```

### What the demos look like

| Task | Episodes | Frames | Variants |
|---|---|---|---|
| T2 `grasp_move_mug` | 300 | 39,395 | left / right / forward, 100 each |
| T3 `toggle_switch_table` | 200 | 20,567 | red / blue, 100 each |

- Control runs at 20 Hz (`control_freq`, dt = 0.05 s). The `fps: 30` in the LeRobot metadata is a
  label only; use dt = 0.05 s for velocities and jerk (decision D1, confirmed in #14).
- Actions are 8-dim (`pd_joint_delta_pos`), state is 9-dim, images are 512×512 from `base_camera`
  and `hand_camera`.
- Known upstream defect: in `toggle_switch_table_blue.rgb.pd_joint_delta_pos.physx_cpu.json`,
  episodes 98 and 99 both carry seed 729406 and source seed 723909 is absent. The two LeRobot
  trajectories differ, so only the seed label is wrong, but seed 723909 is not excluded from the
  eval seed pool and seed-keyed joins (#16) must use the episode index. `check_demos.py`
  reports it as a warning.
