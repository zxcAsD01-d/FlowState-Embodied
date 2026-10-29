"""Inference policy: RGB -> DINOv2 depth -> boundary path -> ARDY root conditions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import (BND_H, BND_W, DEPTH_MAX, DEPTH_MIN, IMG_H, IMG_W, LOG_MAX,
                   MEAN, RES_SCALE, STD, base_path)
from .models import BoundaryHead, DinoDepth, load_dino

ROOT = Path(__file__).resolve().parents[1]


def colorize_depth(d: np.ndarray) -> np.ndarray:
    """(H,W) metres -> (H,W,3) uint8 turbo map."""
    import cv2
    x = np.clip((np.log(np.clip(d, DEPTH_MIN, DEPTH_MAX)) - np.log(DEPTH_MIN))
                / (np.log(DEPTH_MAX) - np.log(DEPTH_MIN)), 0, 1)
    u = (x * 255).astype(np.uint8)
    return cv2.applyColorMap(u, cv2.COLORMAP_TURBO)


class VisualPolicy:
    def __init__(self, depth_ckpt=None, boundary_ckpt=None, device="cuda"):
        self.device = device
        self.depth_ckpt = Path(depth_ckpt) if depth_ckpt else ROOT / "checkpoints" / "depth.pt"
        self.boundary_ckpt = Path(boundary_ckpt) if boundary_ckpt else ROOT / "checkpoints" / "boundary.pt"
        self.mode = "oracle"
        self.dino = None
        self.depth = None
        self.boundary = None
        if self.depth_ckpt.exists():
            self.dino = load_dino(device=device, dtype=torch.float16)
            self.depth = DinoDepth(self.dino).to(device).eval()
            sd = torch.load(self.depth_ckpt, map_location="cpu")
            self.depth.head.load_state_dict(sd["head"])
            for p in self.depth.parameters():
                p.requires_grad_(False)
        if self.boundary_ckpt.exists():
            self.boundary = BoundaryHead().to(device).eval()
            sd = torch.load(self.boundary_ckpt, map_location="cpu")
            self.boundary.load_state_dict(sd["head"])
            for p in self.boundary.parameters():
                p.requires_grad_(False)
        if self.depth is not None and self.boundary is not None:
            self.mode = "visual"

    def _depth_from_rgb(self, rgb: np.ndarray):
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(self.device)
        with torch.no_grad():
            logd = self.depth(x, in_hw=(IMG_H, IMG_W))
            d = torch.exp(logd).clamp(DEPTH_MIN, DEPTH_MAX)
        return d[0].float().cpu().numpy()

    def infer(self, rgb: np.ndarray, target_vec: np.ndarray, fallback=None,
              use_residual: bool = True):
        """RGB (H,W,3 uint8) + normalized target -> path (8,3) m/rad, stop, depth."""
        depth = None
        if self.depth is not None:
            depth = self._depth_from_rgb(rgb)
        if self.mode == "visual":
            dn = np.log(np.clip(depth, DEPTH_MIN, DEPTH_MAX)) / LOG_MAX
            x = torch.from_numpy(dn[None, None].astype(np.float32)).to(self.device)
            t = torch.from_numpy(target_vec[None].astype(np.float32)).to(self.device)
            with torch.no_grad():
                res, logit = self.boundary(x, t)
            base = base_path(target_vec)
            if use_residual:
                p = base + res[0].float().cpu().numpy() * RES_SCALE
                mode = "visual"
            else:
                p = base
                mode = "base"
            stop = bool(torch.sigmoid(logit)[0].item() > 0.5)
            return p.astype(np.float32), stop, depth, mode
        if fallback is None:
            raise RuntimeError("no visual checkpoints and no fallback policy")
        p, stop = fallback
        return p, stop, depth, "oracle"
