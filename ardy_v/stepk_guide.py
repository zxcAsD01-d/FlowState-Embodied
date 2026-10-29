"""Step-k latent guidance: goal + no-go SDF barrier through the denoiser.

Unlike the rejected x0 post-hoc root edit, this injects a trainable latent at
step k (default: the last denoising step) and lets ARDY's root/body stages
re-run, so the body is regenerated consistently with the constrained root.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CIP = Path(os.environ.get("FLOWSTATE_CIP_RUNS", "external/CIP-01/runs"))
for p in (str(CIP / "R07_ardy_precision"), str(CIP / "R10_ardy_inverse")):
    if p not in sys.path:
        sys.path.insert(0, p)
import run_r07_ardy_precision as r07  # noqa: E402
from app.ardy_runtime import _to_joints  # noqa: E402

from .region import Region  # noqa: E402
from .runtime import ArdyV  # noqa: E402


class Injector:
    def __init__(self, k, hist_tokens, leaf=None, record=None):
        self.k = k; self.hist = hist_tokens
        self.leaf = leaf; self.record = record; self.count = 0

    def __call__(self, x, next_fn):
        if self.record is not None:
            self.record.append(x[:, self.hist:].detach().clone())
        if self.k is not None and self.count == self.k and self.leaf is not None:
            x = torch.cat([x[:, :self.hist], self.leaf.to(x.dtype)], dim=1)
        self.count += 1
        return next_fn(x)


class StepKGuidedArdy(ArdyV):
    def __init__(self, device="cuda", steps=6, seed=0, cfg=(2.0, 2.0),
                 k=None, iters=4, lr=0.05, margin=0.32,
                 goal_w=5.0, prior_w=0.2):
        super().__init__(device=device, steps=steps, seed=seed, cfg=cfg)
        self.k = int(steps - 1 if k is None else k)
        self.iters = int(iters); self.lr = float(lr); self.margin = float(margin)
        self.goal_w = float(goal_w); self.prior_w = float(prior_w)
        self._region = None; self._target = None; self._sdf_t = None
        self.stats = {}

    def _sdf_at(self, pos):
        r = self._region
        p = (pos - torch.as_tensor(r.origin, device=pos.device, dtype=pos.dtype)) / r.res - 0.5
        h, w = r.sdf.shape
        x, y = p[:, 0], p[:, 1]
        x0 = torch.floor(x).long(); y0 = torch.floor(y).long()
        tx, ty = x - x0, y - y0
        acc = torch.zeros_like(x)
        for dy, dx, wgy in ((0, 0, (1 - tx) * (1 - ty)), (0, 1, tx * (1 - ty)),
                            (1, 0, (1 - tx) * ty), (1, 1, tx * ty)):
            xi = torch.clamp(x0 + dx, 0, w - 1); yi = torch.clamp(y0 + dy, 0, h - 1)
            acc = acc + self._sdf_t[yi, xi] * wgy
        inside = (x >= -0.5) & (x <= w - 0.5) & (y >= -0.5) & (y <= h - 0.5)
        return torch.where(inside, acc, torch.full_like(acc, 10.0))

    def _run(self, injector):
        model = self.gen.model
        orig = type(model).denoising_step.__get__(model)
        def step(x, *a, **kw):
            return injector(x, lambda xx: orig(xx, *a, **kw))
        model.denoising_step = step
        try:
            with r07._Patches(injector.leaf):
                return self.gen._sample(self.gen.history_len + self.gen.gen_len)
        finally:
            del model.denoising_step

    def step_stepk(self, target_local, region: Region, iters=None, margin=None):
        t0 = time.perf_counter()
        target = torch.as_tensor(np.asarray(target_local, np.float64),
                                 device=self.gen.model.device, dtype=torch.float32)
        self._region = region
        self._target = target
        self._sdf_t = torch.as_tensor(region.sdf, device=self.gen.model.device,
                                      dtype=torch.float32)
        self._r10.set_e(self.gen, self.e_walk)
        nfpt = int(self.gen.model.num_frames_per_token)
        hist_tokens = int(self.gen.history_len // nfpt)
        rec = []
        self._run(Injector(self.k, hist_tokens, record=rec))
        leaf0 = rec[self.k].clone()
        leaf = leaf0.clone().requires_grad_(True)
        opt = torch.optim.Adam([leaf], lr=self.lr)
        it = int(self.iters if iters is None else iters)
        mg = float(self.margin if margin is None else margin)
        curve = []
        for _ in range(it):
            opt.zero_grad(set_to_none=True)
            out = self._run(Injector(self.k, hist_tokens, leaf=leaf))
            j = r07.to_joints_grad(self.gen.model, out)["posed_joints"][0]
            root = j[:, 0, [0, 2]]
            sdf = self._sdf_at(root)
            barrier = torch.relu(mg - sdf).pow(2).mean()
            goal = ((root[-1] - target) ** 2).sum()
            traj = ((root - target) ** 2).sum(-1).mean() * 0.02
            smooth = ((root[2:] - 2 * root[1:-1] + root[:-2]) ** 2).mean()
            prior = ((leaf - leaf0) ** 2).mean()
            loss = (20.0 * barrier + self.goal_w * goal + 0.01 * traj
                    + 2.0 * smooth + self.prior_w * prior)
            loss.backward()
            opt.step()
            curve.append({"barrier": float(barrier), "goal": float(goal),
                          "prior": float(prior), "sdf": float(sdf.min())})
        with torch.no_grad():
            out = self._run(Injector(self.k, hist_tokens, leaf=leaf.detach()))
            inverse = _to_joints(self.gen.model, out)
            chunk = self.gen._pack(out, inverse, 0, self.gen.gen_len,
                                   action="walk", started=t0, self_state=None,
                                   note="stepk", text_conditioned=False,
                                   target_hand=None, constraints=None)
            yaw = float(self.gen._pelvis_heading())
            self.frames += 1
        self.stats = {"k": self.k, "iters": it, "curve": curve,
                      "seconds": time.perf_counter() - t0}
        return self._chunk(chunk, yaw), "walk", float(time.perf_counter() - t0)
