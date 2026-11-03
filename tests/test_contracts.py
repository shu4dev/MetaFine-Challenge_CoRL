"""Round-trip tests for the C1 / C3 / C4 contract stubs in utils/contracts.py (issue #14). CPU only."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("contracts", ROOT / "utils" / "contracts.py")
contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contracts)


def test_c1_round_trip(tmp_path):
    table = contracts.make_synthetic_stage_table(n_episodes=3, frames_per_episode=60)
    p = contracts.save_stage_table(tmp_path / "stages" / "toggle_switch_table.npz", table)
    back = contracts.load_stage_table(p)
    assert set(back) >= set(contracts.C1_KEYS)
    assert back["offsets"][-1] == len(back["stage"])
    assert back["stage"].dtype == np.int8 and back["episode_index"].dtype == np.int64
    assert json.loads(back["codes"]) == {str(k): v for k, v in contracts.STAGE_CODES.items()}
    assert back["provisional"] is False
    assert contracts.stage_of_frame(back, 1, 0) == 1  # approach first
    with pytest.raises(IndexError):
        contracts.stage_of_frame(back, 1, 10_000)


def test_c1_provisional_is_all_other():
    table = contracts.make_synthetic_stage_table(provisional=True)
    assert table["stage"].max() == 0 and bool(table["provisional"]) is True


def test_c3_random_prior_log_prob_and_file(tmp_path):
    prior = contracts.RandomPrior(n_features=21, K=3)
    x = np.random.default_rng(1).normal(size=(16, 21)).astype(np.float32)
    lp = prior.log_prob(x, context={"instr_onehot": np.zeros((16, 14)), "stage_onehot": np.zeros((16, 4))})
    assert lp.shape == (16,) and np.all(np.isfinite(lp)) and lp.dtype == np.float32
    p = prior.save(tmp_path / "priors" / "prior_random.npz")
    with np.load(p, allow_pickle=False) as z:
        assert set(contracts.C3_KEYS) <= set(z.files)
        assert int(z["K"]) == 3 and int(z["window"]) == 10 and float(z["dt"]) == pytest.approx(0.05)
        assert json.loads(str(z["context_layout"])) == {"instr_onehot": 14, "stage_onehot": 4}


def test_c4_round_trip_and_check(tmp_path):
    for task in ("toggle_switch_table", "grasp_move_mug"):
        ro = contracts.make_synthetic_rollout(task_id=task, T=50, seed=3)
        p = contracts.save_rollout(tmp_path / "runs" / "r" / "eval" / task / "episodes" / "seed000003_red.npz", ro)
        assert contracts.check_rollout(p) == []
    bad = contracts.make_synthetic_rollout(T=40)
    bad["qpos"] = bad["qpos"][:-1]
    p = contracts.save_rollout(tmp_path / "bad.npz", bad)
    assert any("qpos" in m for m in contracts.check_rollout(p))
