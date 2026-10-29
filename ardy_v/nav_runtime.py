"""DINOv2 depth + collision inference for the online navigator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import IMG_H, IMG_W, MEAN, STD
from .models import NavDepth, load_dino

ROOT = Path(__file__).resolve().parents[1]


class NavEncoder:
    def __init__(self, ckpt=None, device="cuda"):
        self.device = device
        self.dino = load_dino(device=device, dtype=torch.float16)
        self.model = NavDepth(self.dino).to(device).eval()
        ckpt = Path(ckpt) if ckpt else ROOT / "checkpoints" / "nav.pt"
        sd = torch.load(ckpt, map_location="cpu")
        self.model.depth.load_state_dict(sd["depth"])
        self.model.collision.load_state_dict(sd["collision"])
        for p in self.model.parameters():
            p.requires_grad_(False)

    def predict(self, rgb: np.ndarray):
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(self.device)
        with torch.no_grad():
            d, m = self.model.predict(x, in_hw=(IMG_H, IMG_W))
        return d[0].float().cpu().numpy(), m[0].float().cpu().numpy()
