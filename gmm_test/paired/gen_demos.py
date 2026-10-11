"""Generate T3 planner demos for given (seed, target) pairs, the same way record.py does
(pd_joint_pos, obs none, 512@fov70, new env per trial), one h5 per demo.

    python gmm_test/paired/gen_demos.py jobs.txt out_dir [--procs 16]
jobs.txt: one "seed target" per line. Output: out_dir/<target>_<seed>/trajectory.h5 (+.json), only
for successful plans; failures are listed in out_dir/failed.txt.
"""
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def run(job):
    seed, target, out = job
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    import gymnasium as gym
    import core.env  # noqa: F401  registers envs
    from core.common import MP_SOLUTIONS
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
        res = MP_SOLUTIONS["toggle_switch"](planner, part_name="button", verbose=False)
        ok = res != -1 and bool(res[4].get("success", False))
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
    p.add_argument("jobs"); p.add_argument("out"); p.add_argument("--procs", type=int, default=16)
    a = p.parse_args()
    jobs = [(int(s), t, a.out) for s, t in (l.split() for l in open(a.jobs) if l.strip())]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    mp.set_start_method("spawn")
    n_ok = 0
    with mp.Pool(a.procs, maxtasksperchild=1) as pool, open(Path(a.out) / "failed.txt", "a") as fail:
        for i, (seed, target, ok) in enumerate(pool.imap_unordered(run, jobs), 1):
            n_ok += ok
            if not ok:
                fail.write(f"{seed} {target}\n"); fail.flush()
            print(f"[{i}/{len(jobs)}] {target} {seed} {'ok' if ok else 'FAILED'}  (ok so far {n_ok})", flush=True)
