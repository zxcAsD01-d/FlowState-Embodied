"""Frozen ARDY Core8 runtime for the visual boundary demo."""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The frozen ARDY Core8 generator lives outside this repo (research code from
# the AL project tree).  Point FLOWSTATE_AL0A / FLOWSTATE_CIP_RUNS at your
# own checkouts; see README "External dependencies".
AL0A = Path(os.environ.get("FLOWSTATE_AL0A", "external/al-0a"))
CIP = Path(os.environ.get("FLOWSTATE_CIP_RUNS", "external/CIP-01/runs"))
for p in (str(AL0A), str(CIP / "R07_ardy_precision"), str(CIP / "R10_ardy_inverse")):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)


class ArdyV:
    """One stateful ARDY actor driven by root-boundary paths."""

    def __init__(self, device="cuda", steps=6, seed=0, cfg=(2.0, 2.0)):
        import torch
        import run_r07_ardy_precision as r07
        import run_r10_ardy_inverse as r10
        self._torch = torch
        self._r10 = r10
        self.steps = int(steps)
        self.lock = threading.Lock()
        self.gen = r07.build(cfg=cfg, seed=seed, steps=steps, hist=8)
        self.gen.cfg = tuple(float(v) for v in cfg)
        banks = r10.bank(self.gen.model.device)
        self.e_walk = banks["walk"]
        self.e_stand = banks["stand"]
        self.origin = np.zeros(2, np.float64)
        self.seed_frames = []
        self.frames = 0
        self.stopped = False
        self.reset(np.zeros(2, np.float64))

    def reset(self, spawn):
        """Restart the actor and pin its local frame to a world spawn."""
        self._r10.set_e(self.gen, self.e_walk)
        chunk = self.gen.warm_start_free(frames=8)
        root = self.gen.joints[-1, 0].detach().cpu().numpy()[[0, 2]]
        self.origin = np.asarray(spawn, np.float64) - root
        self.seed_frames = self._chunk(chunk, float(self.gen._pelvis_heading()))
        self.frames = 0
        self.stopped = False
        return self.seed_frames

    # ------------------------------------------------------------- helpers
    def state(self):
        j = self.gen.joints[-1, 0]
        return {
            "root": [float(j[0]), float(j[2])],
            "yaw": float(self.gen._pelvis_heading()),
            "frames": int(self.gen.history_len),
        }

    def target_vector(self, target_local):
        root = self.gen.joints[-1, 0].detach().cpu().numpy()[[0, 2]]
        yaw = float(self.gen._pelvis_heading())
        d = np.asarray(target_local, np.float64) - root
        fwd = float(d[0] * np.sin(yaw) + d[1] * np.cos(yaw))
        lat = float(d[0] * np.cos(yaw) - d[1] * np.sin(yaw))
        dist = float(np.linalg.norm(d))
        bearing = float(np.arctan2(lat, fwd))
        return np.array([fwd / 4.0, lat / 4.0, dist / 10.0,
                         np.sin(bearing), np.cos(bearing), 1.0], np.float32), dist

    def _chunk(self, chunk, yaw):
        rot = np.asarray(chunk.rotations, np.float32)
        roots = np.asarray(chunk.roots, np.float32)
        out = []
        for i in range(rot.shape[0]):
            out.append({
                "rot": rot[i].tolist(),
                "root": roots[i].tolist(),
                "yaw": float(yaw),
                "t": i / 20.0,
            })
        return out

    # --------------------------------------------------------------- step
    def step(self, target_local, path_body=None, stop=False, constraint_frames=None):
        """One 8-frame window.  path_body is (8,3) in the current pelvis frame."""
        import torch
        with self.lock:
            t0 = time.perf_counter()
            if stop or np.linalg.norm(np.asarray(target_local) - self.state()["root"]) < 0.45:
                # Stand, not the hand-hold proxy: use the stand e and pin the
                # root to a zero path so the model does not keep walking.
                self._r10.set_e(self.gen, self.e_stand)
                zpath = np.zeros((8, 3), np.float32)
                obs, mask, _r2d, cons = self.gen.path_conditions(
                    zpath, root_height=0.95, constrain_heading=True,
                    constraint_stride=4, return_constraints=True)
                chunk = self.gen._finish(obs, mask, "stay", None, t0, None,
                                         constraints=cons)
                action = "stay"
            else:
                self._r10.set_e(self.gen, self.e_walk)
                path = np.asarray(path_body, np.float32)
                # constraint_stride=4: keying the root/heading/height on every
                # frame overspecifies the window and the model progressively
                # folds the upper body over it (head ends up below the pelvis
                # within ~12 windows).  Sparse keyframes track the same path
                # (constraint error 0.032 m) without the posture collapse.
                chunk = self.gen.generate_path(
                    path, root_height=0.95, exec_frames=8,
                    constrain_heading=True, constraint_stride=4,
                    constraint_frames=constraint_frames)
                action = "walk"
            yaw = float(self.gen._pelvis_heading())
            self.frames += 1
            return self._chunk(chunk, yaw), action, float(time.perf_counter() - t0)
