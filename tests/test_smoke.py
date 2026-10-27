"""CPU-only smoke tests: no simulator, no GPU, no downloaded demos.

Run with ``pytest tests/test_smoke.py``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.eval_common import compute_ausc, summarise_perception  # noqa: E402
from utils.eval_report import build_report  # noqa: E402


def _load_check_demos():
    spec = importlib.util.spec_from_file_location("check_demos", REPO_ROOT / "tools" / "check_demos.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_demos = _load_check_demos()


# --- compute_ausc ---------------------------------------------------------


def test_compute_ausc_known_curves():
    assert compute_ausc([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert compute_ausc([0.0, 0.0]) == pytest.approx(0.0)
    assert compute_ausc([1.0, 0.0]) == pytest.approx(0.5)
    # Trapezoid over x = 0, 1/3, 2/3, 1.
    assert compute_ausc([0.4, 0.55, 0.1, 0.05]) == pytest.approx((0.4 + 2 * 0.55 + 2 * 0.1 + 0.05) / 6)


def test_compute_ausc_needs_two_points():
    assert math.isnan(compute_ausc([0.7]))
    assert math.isnan(compute_ausc([]))


# --- build_report ---------------------------------------------------------


def _profiles(srs: dict[str, float]) -> dict:
    return {name: {"success_rate": sr} for name, sr in srs.items()}


def test_build_report_from_summaries():
    perception = summarise_perception(
        _profiles(
            {
                "clean": 0.4,
                "cam_l1": 0.55, "cam_l2": 0.1, "cam_l3": 0.05,
                "light_l1": 0.4, "light_l2": 0.5, "light_l3": 0.2,
            }
        )
    )
    understanding = {
        "success_rate": 0.6,
        "per_variant_success_rate": {"red": 0.7, "blue": 0.5},
        "confusion": {"red": {"red": 14, "blue": 6}, "blue": {"red": 10, "blue": 10}},
        "behavior": {"success_rate": 0.6},
    }
    report = build_report(task_id="toggle_switch_table", perception=perception, understanding=understanding)

    perc = report["metrics"]["perception"]
    assert report["task_id"] == "toggle_switch_table"
    assert perc["ausc_camera"] == pytest.approx(compute_ausc([0.4, 0.55, 0.1, 0.05]))
    assert perc["ausc_light"] == pytest.approx(compute_ausc([0.4, 0.4, 0.5, 0.2]))
    assert perc["ausc_mean"] == pytest.approx((perc["ausc_camera"] + perc["ausc_light"]) / 2)
    assert perc["clean_success_rate"] == 0.4
    assert report["metrics"]["understanding"]["success_rate"] == 0.6
    assert report["metrics"]["behavior"]["success_rate"] == 0.6


def test_build_report_without_inputs():
    assert build_report(task_id="grasp_part", perception=None, understanding=None) == {
        "task_id": "grasp_part",
        "metrics": {},
    }


# --- instruction strings --------------------------------------------------


# The 14 strings the demos were recorded with; a policy trained on them sees
# exactly these, so any edit to eval/configs/*.yaml must be deliberate.
INSTRUCTIONS = {
    "grasp_part": {
        "cap": "Grasp the cap of the bottle",
        "body": "Grasp the body of the bottle",
    },
    "grasp_move_mug": {
        "left": "Grasp the mug by the handle and move it to the left",
        "right": "Grasp the mug by the handle and move it to the right",
        "forward": "Grasp the mug by the handle and move it forward",
    },
    "toggle_switch_table": {
        "red": "Toggle the red switch on the table",
        "blue": "Toggle the blue switch on the table",
    },
    "put_blocks_into_boxes": {
        color: f"Place the {color} cube in the left-hand box and the remaining cubes in the right-hand box."
        for color in ("red", "blue", "green")
    },
    "insert_letter": {
        letter: f"Insert the letter {letter} into its slot on the board" for letter in ("C", "o", "R", "L")
    },
}


def test_eval_configs_define_14_instructions():
    assert check_demos.check_instruction_total() == []
    assert sum(len(v) for v in INSTRUCTIONS.values()) == check_demos.N_INSTRUCTIONS


@pytest.mark.parametrize("task", sorted(INSTRUCTIONS))
def test_eval_config_instructions_unchanged(task):
    cfg = check_demos.parse_task_config(check_demos.CONFIG_DIR / f"{task}.yaml")
    assert cfg["variants"] == INSTRUCTIONS[task]
    assert check_demos.compare_instructions(cfg["variants"], INSTRUCTIONS[task].values()) == []
    assert len(cfg["train_demo_jsons"]) == len(INSTRUCTIONS[task])


def test_compare_instructions_exact_match():
    cfg = check_demos.parse_task_config(check_demos.CONFIG_DIR / "grasp_move_mug.yaml")["variants"]
    assert check_demos.compare_instructions(cfg, list(cfg.values())) == []


def test_compare_instructions_rejects_altered_string():
    cfg = check_demos.parse_task_config(check_demos.CONFIG_DIR / "toggle_switch_table.yaml")["variants"]
    found = [cfg["red"], cfg["blue"].replace("switch", "Switch")]
    errors = check_demos.compare_instructions(cfg, found)
    assert len(errors) == 2  # config string missing + unexpected demo string
    assert any("'blue'" in e for e in errors)
