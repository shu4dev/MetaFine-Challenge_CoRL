"""MetaFine competition policy server — participant template.

Your submission exposes your policy over a tiny HTTP interface. The organizer
harness owns the simulator and the hidden seeds; your code never sees either.
Per episode the harness calls ``/reset`` once, then ``/act`` every control step.

To submit, implement the two methods of :class:`Policy` below (everything else
in this file is protocol plumbing — leave it alone), put your weights next to
this file, and push to your team's Hugging Face repo.

Wire protocol (JSON over HTTP, all endpoints under ``http://0.0.0.0:PORT``):

  GET  /health -> {"ok": true, "name": "<policy name>"}
  POST /reset  {"task_id": "grasp_part", "instruction": "Grasp the cap of the bottle",
                "action_dim": 8}
            -> {"ok": true}
  POST /act    {"instruction": "...", "step": 0,
                "state": [<9 floats — robot qpos>],
                "images": {"base_camera": {"shape": [512, 512, 3], "dtype": "uint8",
                                           "data": "<base64 raw bytes>"},
                           "hand_camera": {...}}}
            -> {"action": [<action_dim floats — pd_joint_delta_pos target>]}

Self-test without the simulator (checks your policy end-to-end with fake obs):

    python policy_server.py --self-test

Run as a server (what the eval harness does):

    python policy_server.py --port 8080
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


# --------------------------------------------------------------------------- #
# Implement this class.                                                       #
# --------------------------------------------------------------------------- #

class Policy:
    """Your policy. Replace the bodies of ``reset`` and ``act``.

    The template returns zero actions (the robot holds still) so the file runs
    out of the box; replace it with your model.
    """

    name = "zero-policy-template"

    def __init__(self) -> None:
        # Load your model ONCE here (weights live next to this file).
        # e.g. self.model = MyVLA.from_pretrained("weights/")
        self.action_dim = 8

    def reset(self, task_id: str, instruction: str, action_dim: int) -> None:
        """Called once at the start of every episode.

        ``task_id`` is one of: grasp_part, grasp_move_mug, toggle_switch_table,
        put_blocks_into_boxes, insert_letter. ``instruction`` is the language
        command for this episode. ``action_dim`` is the env action dimension
        your ``act`` must return.
        """
        self.action_dim = int(action_dim)
        # e.g. clear your action chunk buffer / recurrent state here.

    def act(self, state: np.ndarray, images: dict[str, np.ndarray],
            instruction: str, step: int) -> np.ndarray:
        """Called every control step. Return one action.

        Args:
            state: float32 ``(9,)`` — robot qpos.
            images: ``{"base_camera": uint8 (512, 512, 3), "hand_camera": ...}``.
            instruction: same string as in ``reset``.
            step: 0-based step index within the episode.

        Returns:
            float array ``(action_dim,)`` — ``pd_joint_delta_pos`` action.
        """
        return np.zeros(self.action_dim, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Protocol plumbing below — no need to edit.                                  #
# --------------------------------------------------------------------------- #

def _decode_image(spec: dict) -> np.ndarray:
    # bytearray (not bytes) so the array is writable — torch.from_numpy warns
    # loudly on read-only buffers.
    raw = bytearray(base64.b64decode(spec["data"]))
    return np.frombuffer(raw, dtype=np.dtype(spec["dtype"])).reshape(spec["shape"])


def _encode_image(arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype),
            "data": base64.b64encode(arr.tobytes()).decode("ascii")}


class _Handler(BaseHTTPRequestHandler):
    policy: Policy  # set by serve()
    _lock = threading.Lock()

    def _reply(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:  # silence per-request stderr noise
        pass

    def do_GET(self) -> None:
        if self.path == "/health":
            self._reply(200, {"ok": True, "name": self.policy.name})
        else:
            self._reply(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            with self._lock:  # episodes are strictly sequential
                if self.path == "/reset":
                    self.policy.reset(req["task_id"], req["instruction"],
                                      req.get("action_dim", 8))
                    self._reply(200, {"ok": True})
                elif self.path == "/act":
                    state = np.asarray(req["state"], dtype=np.float32)
                    images = {k: _decode_image(v) for k, v in req["images"].items()}
                    action = self.policy.act(state, images,
                                             req.get("instruction", ""),
                                             int(req.get("step", 0)))
                    action = np.asarray(action, dtype=np.float32).reshape(-1)
                    self._reply(200, {"action": [float(x) for x in action]})
                else:
                    self._reply(404, {"error": f"unknown path {self.path}"})
        except Exception as exc:  # surface the error to the harness, keep serving
            self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})


def serve(port: int, policy: "Policy | None" = None) -> None:
    _Handler.policy = policy if policy is not None else Policy()
    httpd = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    print(f"[policy_server] {_Handler.policy.name} listening on :{port}", flush=True)
    httpd.serve_forever()


# --------------------------------------------------------------------------- #
# Client + self-test (the eval harness uses the same client).                 #
# --------------------------------------------------------------------------- #

class PolicyClient:
    """Minimal client for the protocol above (stdlib only)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, obj: dict) -> dict:
        import urllib.error
        import urllib.request
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            # The handler puts the real traceback message in the body; without
            # this the caller only ever sees "HTTP Error 500".
            try:
                detail = json.loads(exc.read()).get("error", "")
            except Exception:
                detail = ""
            raise RuntimeError(
                f"policy server error on {path}: {detail or exc}") from None
        if "error" in out:
            raise RuntimeError(f"policy server error on {path}: {out['error']}")
        return out

    def health(self) -> dict:
        import urllib.request
        with urllib.request.urlopen(self.base_url + "/health", timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def reset(self, task_id: str, instruction: str, action_dim: int) -> None:
        self._post("/reset", {"task_id": task_id, "instruction": instruction,
                              "action_dim": action_dim})

    def act(self, state: np.ndarray, images: dict[str, np.ndarray],
            instruction: str, step: int) -> np.ndarray:
        out = self._post("/act", {
            "instruction": instruction, "step": step,
            "state": [float(x) for x in np.asarray(state).reshape(-1)],
            "images": {k: _encode_image(v) for k, v in images.items()},
        })
        return np.asarray(out["action"], dtype=np.float32)


def self_test(port: int, policy: "Policy | None" = None) -> None:
    """Spin up the server in-process and drive one fake episode against it."""
    _Handler.policy = policy if policy is not None else Policy()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        client = PolicyClient(f"http://127.0.0.1:{port}")
        info = client.health()
        print(f"[self-test] /health ok — policy: {info['name']}")

        action_dim = 8
        client.reset("grasp_part", "Grasp the cap of the bottle", action_dim)
        print("[self-test] /reset ok")

        rng = np.random.default_rng(0)
        for step in range(3):
            state = rng.standard_normal(9).astype(np.float32)
            images = {
                "base_camera": rng.integers(0, 256, (512, 512, 3), dtype=np.uint8),
                "hand_camera": rng.integers(0, 256, (512, 512, 3), dtype=np.uint8),
            }
            action = client.act(state, images, "Grasp the cap of the bottle", step)
            assert action.shape == (action_dim,), (
                f"action shape {action.shape} != ({action_dim},)")
            assert np.all(np.isfinite(action)), "action contains NaN/Inf"
            print(f"[self-test] /act step {step} ok — action[:3]={action[:3]}")
        print("[self-test] PASSED — your policy speaks the protocol correctly.")
    finally:
        httpd.shutdown()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--self-test", action="store_true",
                   help="run one fake episode against your Policy and exit")
    args = p.parse_args()
    if args.self_test:
        self_test(args.port)
    else:
        serve(args.port)


if __name__ == "__main__":
    main()
