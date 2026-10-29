"""ARDY with a differentiable no-go barrier on the final clean root prediction.

ARDY's sampler is deterministic DDIM.  At the last step (t=0) the sampler
returns the denoiser's clean prediction x0 exactly.  We hook the CFG output at
that step, optimize the 8-frame root positions against
    goal(target) + hinge_barrier(no-go SDF) + smoothness + prior,
and write the corrected root back.  The body latent is untouched; the motion
therefore stays on ARDY's manifold instead of being a separately planned path.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .region import Region, region_from_depth
from .runtime import ArdyV


class GuidedArdy(ArdyV):
    def __init__(self, device="cuda", steps=6, seed=0, cfg=(2.0, 2.0),
                 margin: float = 0.32, iters: int = 18, lr: float = 0.05):
        super().__init__(device=device, steps=steps, seed=seed, cfg=cfg)
        self.margin = float(margin)
        self.iters = int(iters)
        self.lr = float(lr)
        self._region: Region | None = None
        self._target = None
        self._guide_root = None
        self._guide_on = False
        self.guide_stats = {}
        self._last = {"barrier": 0.0, "goal": 0.0, "prior": 0.0}
        self._sdf_t = None
        self.gen.model.denoiser.register_forward_hook(self._x0_hook)

    # --------------------------------------------------------------- hook
    def _x0_hook(self, module, inputs, output):
        if not self._guide_on or self._region is None:
            return output
        try:
            t = inputs[14]
            if int(t[0]) != 0:
                return output
        except Exception:
            return output
        # denoising_step runs under inference_mode; leave it for the small
        # root optimization and clone the output so autograd can see pos.
        with torch.inference_mode(False), torch.enable_grad():
            out = output.clone()
            ng = self.gen.gen_len // self.gen.model.num_frames_per_token
            g = out[:, -ng:, :]
            root_norm = g[..., :20].reshape(out.shape[0], -1, 5)
            stats = self.gen.model.motion_rep.global_root_stats
            root_m = stats.unnormalize(root_norm).clone()
            x0 = root_m[0, :, [0, 2]].detach().clone()
            # The model x0 lives in the recentered frame; the actor root is the
            # last history frame, i.e. the recenter translation.  global = x0 +
            # guide_root.  (Subtracting x0[0] here made the barrier sample a
            # frame that was ~1.5 m away and silently did nothing.)
            center = torch.as_tensor(self._guide_root, device=x0.device,
                                     dtype=x0.dtype)
            pos0 = x0 + center[None, :]
            target = torch.as_tensor(self._target, device=x0.device, dtype=x0.dtype)
            pos = self._optimize(pos0, target)
            new_xz = pos - center[None, :]
            y = root_m[0, :, 1]
            d = torch.zeros_like(new_xz)
            d[:-1] = new_xz[1:] - new_xz[:-1]
            d[-1] = target - new_xz[-1]
            yaw = torch.atan2(d[:, 0], d[:, 1])
            old = torch.atan2(root_m[0, :, 4], root_m[0, :, 3])
            speed = torch.linalg.norm(d, dim=-1)
            yaw = torch.where(speed > 0.004, yaw, old)
            root_new = torch.stack([new_xz[:, 0], y, new_xz[:, 1],
                                    torch.cos(yaw), torch.sin(yaw)], dim=-1)
            root_m[0] = root_new
            new_norm = stats.normalize(root_m)
            g[..., :20] = new_norm.reshape(1, ng, 20).detach()
            self.guide_stats = {
                "barrier": float(self._last["barrier"]),
                "goal": float(self._last["goal"]),
                "prior": float(self._last["prior"]),
                "max_shift": float(torch.linalg.norm(new_xz - x0, dim=-1).max()),
            }
        return out

    def _sdf_at(self, pos: torch.Tensor) -> torch.Tensor:
        """Differentiable bilinear SDF sample at (N,2); outside grid = free."""
        r = self._region
        p = (pos - torch.as_tensor(r.origin, device=pos.device, dtype=pos.dtype)) / r.res - 0.5
        h, w = r.sdf.shape
        x, y = p[:, 0], p[:, 1]
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        tx, ty = x - x0, y - y0
        acc = torch.zeros_like(x)
        for dy, dx, wgt in ((0, 0, (1 - tx) * (1 - ty)), (0, 1, tx * (1 - ty)),
                            (1, 0, (1 - tx) * ty), (1, 1, tx * ty)):
            xi = torch.clamp(x0 + dx, 0, w - 1)
            yi = torch.clamp(y0 + dy, 0, h - 1)
            acc = acc + self._sdf_t[yi, xi] * wgt
        inside = (x >= -0.5) & (x <= w - 0.5) & (y >= -0.5) & (y <= h - 0.5)
        return torch.where(inside, acc, torch.full_like(acc, 10.0))

    def _optimize(self, pos0: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos = pos0.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([pos], lr=self.lr)
        last = {}
        for _ in range(self.iters):
            opt.zero_grad(set_to_none=True)
            sdf = self._sdf_at(pos)
            barrier = torch.relu(self.margin - sdf).pow(2).mean()
            goal = ((pos[-1] - target) ** 2).sum()
            traj = ((pos - target) ** 2).sum(-1).mean() * 0.05
            smooth = ((pos[2:] - 2 * pos[1:-1] + pos[:-2]) ** 2).mean()
            prior = ((pos - pos0) ** 2).mean()
            # Do not let the goal term teleport the root: ARDY's walk is
            # ~0.4 m/window, so the barrier may steer but not lengthen the step.
            k = torch.arange(1, pos.shape[0] + 1, device=pos.device, dtype=pos.dtype)
            limit = 0.055 * k + 0.02
            over = torch.relu(torch.linalg.norm(pos - pos0[0], dim=-1) - limit)
            loss = (20.0 * barrier + 0.2 * goal + 0.01 * traj
                    + 2.0 * smooth + 2.0 * prior + 20.0 * over.pow(2).mean())
            loss.backward()
            opt.step()
            last = {"barrier": float(barrier), "goal": float(goal),
                    "prior": float(prior)}
        # Hard speed clamp: the optimizer may not satisfy the hinge, and a
        # root teleport is worse than an imperfect barrier.
        k = torch.arange(1, pos.shape[0] + 1, device=pos.device, dtype=pos.dtype)
        limit = 0.055 * k + 0.02
        v = pos - pos0[0]
        n = torch.linalg.norm(v, dim=-1).clamp_min(1e-6)
        scale = torch.minimum(torch.ones_like(n), limit / n)
        pos = pos0[0] + v * scale[:, None]
        self._last = last
        return pos.detach()

    # --------------------------------------------------------------- step
    def step_guided(self, target_local, depth: np.ndarray | None = None,
                    region: Region | None = None, use_region: bool = True):
        """One window from the current state, with depth->region barrier."""
        t0 = time.perf_counter()
        root = self.state()["root"]
        if region is None and depth is not None and use_region:
            region = region_from_depth(depth, root, self.state()["yaw"])
        elif not use_region:
            region = None
        if np.linalg.norm(np.asarray(target_local) - np.asarray(root)) < 0.45:
            frames, action, dt = self.step(target_local, stop=True)
            self._guide_on = False
            return frames, action, dt, None
        self._r10.set_e(self.gen, self.e_walk)
        if region is not None:
            self._region = region
            self._sdf_t = torch.as_tensor(region.sdf, device=self.gen.model.device,
                                          dtype=torch.float32)
            self._target = np.asarray(target_local, np.float64)
            self._guide_root = np.asarray(root, np.float64)
            self._guide_on = True
        else:
            self._guide_on = False
        with self.lock:
            chunk = self.gen._finish(None, None, "large_motion", None, t0, None)
            yaw = float(self.gen._pelvis_heading())
            self.frames += 1
        self._guide_on = False
        return self._chunk(chunk, yaw), "walk", float(time.perf_counter() - t0), region
