"""Contract stubs and synthetic generators for the C1 (stage side-table), C3 (prior file) and C4
(rollout npz) formats defined in issue #14 (contract version 1).

Pure numpy so the tests run on any laptop; ``RandomPrior.log_prob`` also accepts torch tensors when
torch is installed. Consumers (#23 loss, #31 metrics, #32 statistics) can be developed against these
before the real files from #18, #20 and #22 exist.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CONTRACT_VERSION = 1
STAGE_CODES = {0: "other", 1: "approach", 2: "transport", 3: "contact"}
N_INSTRUCTIONS = 14
N_STAGES = 4
C1_KEYS = ("episode_index", "offsets", "stage", "codes", "provisional")
C3_KEYS = ("weights", "means", "covariances", "feature_names", "window", "feat_mean", "feat_std",
           "condition_on", "git_sha", "dt", "context_layout", "K")
C4_KEYS = ("qpos", "action", "tcp_pose", "gripper_width", "target_pose", "success_step", "success_at_break",
           "success", "episode_length", "dt", "seed", "instruction", "task_id", "variant", "run_name", "git_sha")
C4_TASK_SIGNALS = {
    "grasp_move_mug": ("stage_grasped", "stage_moved", "moved_tcp", "moved_object"),
    "toggle_switch_table": ("had_contact", "is_toggled", "toggle_hold_count"),
}


def make_synthetic_stage_table(n_episodes: int = 4, frames_per_episode: int = 120, seed: int = 0,
                               provisional: bool = False) -> dict:
    """C1 side-table with an approach / transport / contact / other cycle per episode."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(frames_per_episode // 2, frames_per_episode, size=n_episodes)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    stage = np.zeros(int(offsets[-1]), dtype=np.int8)
    for i in range(n_episodes):
        n = int(lengths[i])
        cuts = (np.array([0.0, 0.35, 0.65, 0.9, 1.0]) * n).astype(int)
        for code, (a, b) in zip((1, 2, 3, 0), zip(cuts[:-1], cuts[1:])):
            stage[offsets[i] + a: offsets[i] + b] = 0 if provisional else code
    return {
        "episode_index": np.arange(n_episodes, dtype=np.int64),
        "offsets": offsets,
        "stage": stage,
        "codes": json.dumps(STAGE_CODES),
        "provisional": np.bool_(provisional),
    }


def save_stage_table(path: str | Path, table: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **table)
    return path


def load_stage_table(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in C1_KEYS if k not in z.files]
        if missing:
            raise KeyError(f"C1 file {path} lacks {missing}")
        out = {k: z[k] for k in z.files}
    out["codes"] = str(out["codes"])
    out["provisional"] = bool(out["provisional"])
    return out


def stage_of_frame(table: dict, episode_index: int, frame_index: int) -> int:
    """Stage code of one frame; the lookup #23 performs per chunk."""
    e = int(np.searchsorted(table["episode_index"], episode_index))
    if e >= len(table["episode_index"]) or table["episode_index"][e] != episode_index:
        raise KeyError(f"episode {episode_index} not in table")
    start, end = int(table["offsets"][e]), int(table["offsets"][e + 1])
    if not 0 <= frame_index < end - start:
        raise IndexError(f"frame {frame_index} outside episode {episode_index} ({end - start} frames)")
    return int(table["stage"][start + frame_index])


class RandomPrior:
    """C3-shaped stand-in: a diagonal Gaussian mixture with random parameters.

    ``log_prob(features, context)`` returns one finite value per row and ignores ``context`` when
    ``condition_on`` is empty, exactly as ``prior_ctxfree`` does. Use it to develop #23 before #22 ships.
    """

    def __init__(self, n_features: int = 21, K: int = 2, window: int = 10, dt: float = 0.05,
                 condition_on: tuple[str, ...] = (), seed: int = 0):
        rng = np.random.default_rng(seed)
        self.K, self.D, self.window, self.dt = int(K), int(n_features), int(window), float(dt)
        self.weights = np.full(self.K, 1.0 / self.K, dtype=np.float32)
        self.means = rng.normal(size=(self.K, self.D)).astype(np.float32)
        self.covariances = np.stack([np.diag(rng.uniform(0.5, 1.5, size=self.D)) for _ in range(self.K)]).astype(np.float32)
        self.feature_names = [f"f{i}" for i in range(self.D)]
        self.feat_mean = np.zeros(self.D, dtype=np.float32)
        self.feat_std = np.ones(self.D, dtype=np.float32)
        self.condition_on = tuple(condition_on)
        self.context_layout = {"instr_onehot": N_INSTRUCTIONS, "stage_onehot": N_STAGES}
        self.git_sha = "synthetic"

    def _log_prob_np(self, x: np.ndarray) -> np.ndarray:
        x = (np.asarray(x, dtype=np.float32) - self.feat_mean) / self.feat_std
        var = np.stack([np.diag(c) for c in self.covariances])  # (K, D)
        d = x[:, None, :] - self.means[None]                       # (B, K, D)
        ll = -0.5 * (np.sum(d * d / var[None], axis=-1) + np.sum(np.log(2 * np.pi * var), axis=-1))
        ll = ll + np.log(self.weights)[None]
        m = ll.max(axis=1, keepdims=True)
        return (m[:, 0] + np.log(np.exp(ll - m).sum(axis=1))).astype(np.float32)

    def log_prob(self, features, context: dict | None = None):
        """(B, D) -> (B,). Accepts numpy arrays or torch tensors (differentiable in torch)."""
        try:
            import torch  # noqa: F401
            if isinstance(features, torch.Tensor):
                x = (features - torch.as_tensor(self.feat_mean, device=features.device)) / torch.as_tensor(self.feat_std, device=features.device)
                var = torch.as_tensor(np.stack([np.diag(c) for c in self.covariances]), device=features.device)
                means = torch.as_tensor(self.means, device=features.device)
                d = x[:, None, :] - means[None]
                ll = -0.5 * ((d * d / var[None]).sum(-1) + torch.log(2 * torch.pi * var).sum(-1))
                ll = ll + torch.log(torch.as_tensor(self.weights, device=features.device))[None]
                return torch.logsumexp(ll, dim=1)
        except ImportError:
            pass
        return self._log_prob_np(features)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights=self.weights, means=self.means, covariances=self.covariances,
                 feature_names=np.array(self.feature_names), window=np.int64(self.window),
                 feat_mean=self.feat_mean, feat_std=self.feat_std, condition_on=np.array(self.condition_on, dtype=str),
                 git_sha=np.array(self.git_sha), dt=np.float32(self.dt),
                 context_layout=np.array(json.dumps(self.context_layout)), K=np.int64(self.K))
        return path


def make_synthetic_rollout(task_id: str = "toggle_switch_table", T: int = 80, seed: int = 0,
                           variant: str = "red", run_name: str = "synthetic", success: bool = True) -> dict:
    """C4 rollout npz contents with smooth random joint motion and the per-task signals."""
    if task_id not in C4_TASK_SIGNALS:
        raise ValueError(f"unknown task_id {task_id}; known: {sorted(C4_TASK_SIGNALS)}")
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T, dtype=np.float32)[:, None]
    qpos = (0.3 * np.sin(2 * np.pi * t * rng.uniform(0.2, 0.6, size=9)) + rng.normal(scale=0.01, size=(T, 9))).astype(np.float32)
    qpos[:, 7:] = 0.04
    action = np.zeros((T, 8), dtype=np.float32)
    action[1:, :7] = np.diff(qpos[:, :7], axis=0) / 0.1
    action[:, 7] = 1.0
    tcp = np.concatenate([0.5 + 0.1 * np.sin(2 * np.pi * t * np.array([0.3, 0.4, 0.5])), np.tile([1.0, 0, 0, 0], (T, 1))], axis=1).astype(np.float32)
    out = {
        "qpos": qpos, "action": action, "tcp_pose": tcp,
        "gripper_width": np.full(T, 0.08, dtype=np.float32),
        "target_pose": np.tile(np.array([0.45, 0.0, 0.2, 1, 0, 0, 0], dtype=np.float32), (T, 1)),
        "success_step": np.zeros(T, dtype=bool), "success_at_break": np.bool_(success), "success": np.bool_(success),
        "episode_length": np.int64(T), "dt": np.float32(0.05), "seed": np.int64(seed),
        "instruction": np.array(f"Toggle the {variant} switch on the table" if task_id == "toggle_switch_table" else f"Grasp the mug by the handle and move it {variant}"),
        "task_id": np.array(task_id), "variant": np.array(variant), "run_name": np.array(run_name), "git_sha": np.array("synthetic"),
    }
    if success:
        out["success_step"][-20:] = True
    for key in C4_TASK_SIGNALS[task_id]:
        out[key] = np.arange(T, dtype=np.int64) if key == "toggle_hold_count" else (np.arange(T) > T // 2)
    return out


def save_rollout(path: str | Path, rollout: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **rollout)
    return path


def check_rollout(path: str | Path) -> list[str]:
    """Return the list of C4 violations (empty when the file conforms)."""
    problems = []
    with np.load(path, allow_pickle=False) as z:
        for k in C4_KEYS:
            if k not in z.files:
                problems.append(f"missing key {k}")
        if problems:
            return problems
        T = int(z["episode_length"])
        for k, shape in (("qpos", (T, 9)), ("action", (T, 8)), ("tcp_pose", (T, 7)), ("gripper_width", (T,)), ("target_pose", (T, 7)), ("success_step", (T,))):
            if tuple(z[k].shape) != shape:
                problems.append(f"{k} has shape {z[k].shape}, expected {shape}")
        task = str(z["task_id"])
        for k in C4_TASK_SIGNALS.get(task, ()):
            if k not in z.files:
                problems.append(f"missing {task} signal {k}")
            elif tuple(z[k].shape) != (T,):
                problems.append(f"{k} has shape {z[k].shape}, expected ({T},)")
        if abs(float(z["dt"]) - 0.05) > 1e-6:
            problems.append(f"dt is {float(z['dt'])}, expected 0.05")
    return problems
