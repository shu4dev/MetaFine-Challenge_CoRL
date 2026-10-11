"""Convert replayed demos (trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5) to a LeRobot dataset in
the same format as demos/CoRL/toggle_switch_table/mixed/lerobot (frames = obs[:-1] + actions,
h264 yuv444p crf 12, task = official instruction).

    python gmm_test/paired/to_lerobot.py out_root dir1 dir2 ...   (dirs named <target>_<seed>)
"""
import sys
from pathlib import Path

import h5py
import numpy as np

import lerobot.datasets.video_utils as VU
from lerobot.datasets.lerobot_dataset import LeRobotDataset

_enc = VU.encode_video_frames
def enc(imgs_dir, video_path, fps, vcodec="h264", pix_fmt="yuv444p", g=None, crf=12, **kw):
    return _enc(imgs_dir, video_path, fps, vcodec="h264", pix_fmt="yuv444p", g=None, crf=12, **kw)
VU.encode_video_frames = enc
import lerobot.datasets.lerobot_dataset as LD
LD.encode_video_frames = enc

TASK = {"red": "Toggle the red switch on the table", "blue": "Toggle the blue switch on the table"}
img = {"dtype": "video", "shape": (512, 512, 3), "names": ["height", "width", "channels"]}
FEATURES = {
    "action": {"dtype": "float32", "shape": (8,), "names": [f"action_{i}" for i in range(8)]},
    "observation.state": {"dtype": "float32", "shape": (9,), "names": [f"joint_{i}" for i in range(9)]},
    "observation.images.base_camera": img, "observation.images.hand_camera": img,
}

out, dirs = Path(sys.argv[1]), [Path(d) for d in sys.argv[2:]]
ds = LeRobotDataset.create(repo_id=out.name, fps=30, features=FEATURES, root=out, use_videos=True,
                           vcodec="h264", image_writer_threads=8)
for i, d in enumerate(dirs):
    target = d.name.split("_")[0]
    with h5py.File(d / "trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5") as f:
        t = f["traj_0"]
        A = t["actions"][:]; Q = t["obs/agent/qpos"][:-1]
        B = t["obs/sensor_data/base_camera/rgb"][:-1]; H = t["obs/sensor_data/hand_camera/rgb"][:-1]
    for k in range(len(A)):
        ds.add_frame({"action": A[k].astype(np.float32), "observation.state": Q[k].astype(np.float32),
                      "observation.images.base_camera": B[k], "observation.images.hand_camera": H[k],
                      "task": TASK[target]})
    ds.save_episode()
    print(f"[{i+1}/{len(dirs)}] {d.name} {len(A)} frames", flush=True)
ds.finalize()
