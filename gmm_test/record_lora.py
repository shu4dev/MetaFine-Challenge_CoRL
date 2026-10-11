"""tools/record_video.py with the policy replanning every N actions (default 25)."""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import torch
import tools.record_video as rv
N = int(os.environ.get("N_ACTION_STEPS", "25"))
_load = rv.load_pi0_policy
def load(*a, **kw):
    policy, pre, post = _load(*a, **kw)
    policy.config.n_action_steps = N
    policy.reset()
    return policy, pre, post
rv.load_pi0_policy = load
torch.manual_seed(0)
rv.main()
