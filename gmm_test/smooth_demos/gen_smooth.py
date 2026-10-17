"""T3 planner demos as ONE continuous joint motion instead of stop-and-go segments.

    python gmm_test/smooth_demos/gen_smooth.py jobs.txt out_dir [--procs 10]

Same waypoints and segment durations as core/skill.py:toggle_switch (pre-grasp via RRTConnect,
4 cm approach + 2 cm push via screw), but only their end joint angles are kept. One quintic spline
runs all 7 joints through start -> pre-grasp -> grasp -> push (zero velocity/acceleration only at the
two ends, no stops in between), and the gripper closes during the last 40% of the first leg.
Output like gen_demos.py: out_dir/<target>_<seed>/trajectory.h5 for successful demos.
"""
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def smooth_toggle(planner, env):
    import sapien
    import transforms3d as t3d
    from scipy.interpolate import make_interp_spline
    from core.skill import get_grasp_pose_from_config

    base = env.unwrapped
    pose1 = get_grasp_pose_from_config(base, "button", grasp_id=int(base.target_switch_id))
    p1, q1 = np.array(pose1.p), np.array(pose1.q)
    R = t3d.quaternions.quat2mat(q1)
    poses = [p1 - R[:, 2] * 0.04, p1, p1 + R[:, 0] * 0.022]  # same constants as toggle_switch

    mp_ = planner.planner
    dt = base.control_timestep
    q = base.agent.robot.get_qpos().cpu().numpy()[0]
    knots, n_steps = [q[:7]], []
    for i, p in enumerate(poses):
        goal = np.concatenate([p, q1])
        if i == 0:
            r = mp_.plan_qpos_to_pose(goal, q, time_step=dt, wrt_world=True)
        else:
            r = mp_.plan_screw(goal, q, time_step=dt)
        if r["status"] != "Success":
            return False
        knots.append(r["position"][-1])
        n_steps.append(len(r["position"]))
        q = np.concatenate([r["position"][-1], q[7:]])

    t = np.concatenate([[0], np.cumsum(n_steps)]) * dt
    z = np.zeros(7)
    spline = make_interp_spline(t, np.array(knots), k=5, bc_type=([(1, z), (2, z)], [(1, z), (2, z)]))
    close_at = 0.6 * t[1]
    info = {}
    for ti in np.arange(1, round(t[-1] / dt) + 1) * dt:
        g = planner.CLOSED if ti >= close_at else planner.OPEN
        *_, info = env.step(np.hstack([spline(ti), g]))
    hold = np.hstack([knots[-1], planner.CLOSED])
    for _ in range(int(getattr(base, "toggle_hold_steps", 20)) + 10):  # same hold as toggle_switch
        *_, info = env.step(hold)
    return bool(info.get("success", False))


def run(job):
    seed, target, out = job
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    import gymnasium as gym
    import core.env  # noqa: F401  registers envs
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from mani_skill.utils.wrappers.record import RecordEpisode

    d = Path(out) / f"{target}_{seed}"
    if (d / "trajectory.h5").exists():
        return seed, target, True
    d.mkdir(parents=True, exist_ok=True)
    env = gym.make("toggle_switch_table", obs_mode="none", control_mode="pd_joint_pos", render_mode="rgb_array",
                   sensor_configs=dict(shader_pack="default", width=512, height=512, base_camera={"fov": 1.2217304763960306}),
                   human_render_camera_configs=dict(shader_pack="default"), viewer_camera_configs=dict(shader_pack="default"),
                   sim_backend="physx_cpu", object_name="100920", part_name="button", target_switch=target)
    env = RecordEpisode(env, output_dir=str(d), trajectory_name="trajectory", save_video=False,
                        record_reward=False, video_fps=30, save_on_reset=False)
    ok = False
    try:
        env.reset(seed=seed)
        planner = PandaArmMotionPlanningSolver(env, vis=False, base_pose=env.unwrapped.agent.robot.pose,
                                               visualize_target_grasp_pose=False, print_env_info=False)
        ok = smooth_toggle(planner, env)
    except Exception as e:  # planner failures
        print(f"seed {seed} {target}: {e}", flush=True)
    env.flush_trajectory(save=ok)
    env.close()
    if not ok:
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    return seed, target, ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("jobs"); p.add_argument("out"); p.add_argument("--procs", type=int, default=10)
    a = p.parse_args()
    jobs = [(int(s), t, a.out) for s, t in (l.split() for l in open(a.jobs) if l.strip())]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    mp.set_start_method("spawn")
    with mp.Pool(a.procs, maxtasksperchild=1) as pool:
        for seed, target, ok in pool.imap_unordered(run, jobs):
            print(f"{target} {seed} {'ok' if ok else 'FAILED'}", flush=True)
