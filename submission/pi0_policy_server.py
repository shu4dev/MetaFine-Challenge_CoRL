#!/usr/bin/env python3
"""π0 reference policy server — a worked example of ``policy_server.py``.

This is the organizers' own submission-shaped wrapper around the π0 baseline:
it subclasses :class:`submission.policy_server.Policy` and fills in the two
methods you have to implement. Read it as a template for wiring a real
checkpoint into the protocol; the transport code is untouched.

Unlike ``policy_server.py`` (a single standalone file), this example imports
``utils.eval_common`` from the repository so the batch it feeds π0 is built by
exactly the same code as the in-process evaluator. Your own submission has no
such dependency — build your model's inputs however you like.

Run it (what the eval harness talks to)::

    python submission/pi0_policy_server.py \\
        --ckpt outputs/pi0_toggle_mixed/checkpoints/030000/pretrained_model \\
        --tokenizer-path /path/to/paligemma-3b-pt-224 --port 8080

Check the protocol end-to-end with fake observations (loads the checkpoint,
runs three real forward passes, never touches the simulator)::

    python submission/pi0_policy_server.py --ckpt <...> --self-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from submission.policy_server import Policy, self_test, serve  # noqa: E402


class Pi0Policy(Policy):
    """π0 (LeRobot) behind the competition wire protocol."""

    name = "pi0-reference"

    def __init__(
        self,
        ckpt: str,
        tokenizer_path: str | None = None,
        device: str = "cuda",
    ) -> None:
        from utils.eval_common import load_pi0_policy

        self.device = device
        self.policy, self.preprocessor, self.postprocessor = load_pi0_policy(
            ckpt, device=device, tokenizer_path=tokenizer_path,
        )
        self.action_dim = 8

    def reset(self, task_id: str, instruction: str, action_dim: int) -> None:
        # π0 buffers a 50-step action chunk; a stale chunk would leak actions
        # from the previous episode into this one.
        self.action_dim = int(action_dim)
        self.policy.reset()

    def act(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        instruction: str,
        step: int,
    ) -> np.ndarray:
        import torch

        from utils.eval_common import build_pi0_batch

        # Mirrors utils.eval_common.run_episode exactly, including where the
        # inference_mode block starts and ends.
        batch = build_pi0_batch(state, images, instruction)
        batch = self.preprocessor(batch)
        with torch.inference_mode():
            action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        return np.asarray(action, dtype=np.float32).reshape(-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="π0 pretrained_model directory")
    p.add_argument("--tokenizer-path", default=None,
                   help="Local PaliGemma tokenizer (required on offline hosts)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--torch-seed", type=int, default=None,
                   help="Seed torch RNG. π0 samples flow-matching noise every "
                        "chunk, so this is what makes a run reproducible — "
                        "used for harness parity checks, off by default.")
    p.add_argument("--self-test", action="store_true",
                   help="Run one fake episode against this policy and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    policy = Pi0Policy(args.ckpt, tokenizer_path=args.tokenizer_path,
                       device=args.device)
    # Seed *after* loading: from_pretrained draws from the RNG, so seeding
    # first would leave the stream at a load-dependent offset.
    if args.torch_seed is not None:
        import torch

        torch.manual_seed(int(args.torch_seed))
    if args.self_test:
        self_test(args.port, policy)
    else:
        serve(args.port, policy)


if __name__ == "__main__":
    main()
