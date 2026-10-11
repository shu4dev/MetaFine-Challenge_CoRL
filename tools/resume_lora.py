#!/usr/bin/env python3
"""Resume a LoRA (``--peft.*``) lerobot-train run. Same arguments as ``lerobot-train``::

    python tools/resume_lora.py \\
        --config_path=outputs/<run>/checkpoints/010000/pretrained_model/train_config.json \\
        --resume=true --steps=20000

LeRobot 0.4.4's own resume breaks for LoRA runs: it reloads the saved adapters frozen
(``PeftModel.from_pretrained`` defaults to ``is_trainable=False``) and then wraps the model
with a second, fresh set of adapters. This loads the saved adapters as trainable and skips
the second wrap, so training continues from the checkpoint's adapters and optimizer state.
"""

import peft

_from_pretrained = peft.PeftModel.from_pretrained.__func__


@classmethod
def _from_pretrained_trainable(cls, *args, **kwargs):
    kwargs["is_trainable"] = True
    return _from_pretrained(cls, *args, **kwargs)


peft.PeftModel.from_pretrained = _from_pretrained_trainable
peft.PeftModel.wrap_with_peft = lambda self, *args, **kwargs: self  # already wrapped

if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import main

    main()
