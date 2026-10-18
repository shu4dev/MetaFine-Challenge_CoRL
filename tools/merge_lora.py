#!/usr/bin/env python3
"""Fold a LoRA checkpoint from ``lerobot-train --peft.*`` into a plain pi0 checkpoint.

The eval scripts load ``PI0Policy.from_pretrained`` directly, which ignores LoRA adapters.
This merges the adapters into the base weights and writes a normal ``pretrained_model``
folder that ``tools/run_eval.py --policy-path`` can use::

    python tools/merge_lora.py outputs/<run>/checkpoints/010000/pretrained_model \\
        outputs/<run>/checkpoints/010000/merged_model

Runs on CPU (about 8 GB of RAM); no GPU needed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def merge(adapter_dir: Path, out_dir: Path) -> Path:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from peft import PeftModel

    base = json.loads((adapter_dir / "adapter_config.json").read_text())["base_model_name_or_path"]
    if not Path(base).is_dir():  # from_pretrained would silently return an untrained model
        raise FileNotFoundError(f"base model {base} not found; recreate it before merging")
    cfg = PreTrainedConfig.from_pretrained(adapter_dir)
    cfg.use_peft = False
    cfg.device = "cpu"
    policy = PI0Policy.from_pretrained(base, config=cfg)
    merged = PeftModel.from_pretrained(policy, adapter_dir).merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    for f in adapter_dir.iterdir():  # processors (normalization stats, camera renames) + train config
        if f.name.startswith("policy_") or f.name == "train_config.json":
            shutil.copy(f, out_dir / f.name)
    (out_dir / "merged_from.txt").write_text(f"adapter: {adapter_dir.resolve()}\nbase: {base}\n")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("adapter_dir", type=Path, help="checkpoint pretrained_model folder with adapter_*")
    p.add_argument("out_dir", type=Path)
    args = p.parse_args()
    print(f"wrote {merge(args.adapter_dir, args.out_dir)}")


if __name__ == "__main__":
    main()
