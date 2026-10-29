"""Frozen DINOv2 encoder + trainable absolute-depth and boundary heads."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DINO_HUB = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
DINO_WEIGHTS = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vitb14_reg4_pretrain.pth"
DINO_LAYERS = (2, 5, 8, 11)


def load_dino(device="cuda", dtype=torch.float16):
    """Load the local ViT-B/14 register checkpoint; frozen and eval."""
    if not DINO_WEIGHTS.exists():
        raise FileNotFoundError(DINO_WEIGHTS)
    model = torch.hub.load(str(DINO_HUB), "dinov2_vitb14_reg",
                           source="local", pretrained=False)
    sd = torch.load(DINO_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval().to(device=device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class DPTHead(nn.Module):
    """Four-level DPT-style depth readout at 1/4 input resolution."""

    def __init__(self, in_dim: int = 768, feat: int = 256, out_hw=(168, 294)):
        super().__init__()
        self.out_hw = tuple(out_hw)
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_dim, feat, 1), nn.GroupNorm(8, feat), nn.SiLU())
            for _ in DINO_LAYERS
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(feat * 4, feat, 3, padding=1), nn.GroupNorm(8, feat), nn.SiLU(),
            nn.Conv2d(feat, feat, 3, padding=1), nn.GroupNorm(8, feat), nn.SiLU(),
        )
        self.out = nn.Conv2d(feat, 1, 1)

    def forward(self, feats, grid):
        h, w = grid
        xs = []
        for proj, f in zip(self.proj, feats):
            x = f.reshape(f.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
            x = proj(x)
            x = F.interpolate(x, size=(h * 2, w * 2), mode="bilinear", align_corners=False)
            xs.append(x)
        x = self.fuse(torch.cat(xs, dim=1))
        x = self.out(x)
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        return x.squeeze(1)  # log-depth at 128x224


class DinoDepth(nn.Module):
    """Frozen DINOv2 + trainable absolute depth head.  Input is normalized RGB."""

    def __init__(self, dino, feat=256):
        super().__init__()
        self.dino = dino
        self.head = DPTHead(feat=feat)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, x, in_hw=(168, 294)):
        dtype = next(self.dino.parameters()).dtype
        with torch.no_grad():
            feats = self.dino.get_intermediate_layers(
                x.to(dtype), n=list(DINO_LAYERS), reshape=False,
                return_class_token=False)
        feats = [f.float() for f in feats]
        h = in_hw[0] // 14
        w = in_hw[1] // 14
        return self.head(feats, (h, w))

    def predict_meters(self, x, in_hw=(168, 294), depth_range=(0.2, 40.0)):
        logd = self.forward(x, in_hw)
        return torch.exp(logd).clamp(*depth_range)


class BoundaryHead(nn.Module):
    """Depth + body-frame target -> 8x3 root path + stop logit.

    The depth map is treated as a 2-D visual field; eight learned queries read
    it with the target token.  Outputs are normalized (fwd/0.6, lat/0.6,
    dyaw/0.7) and a stop logit.
    """

    def __init__(self, width=128, depth_hw=(168, 294), nq=8, layers=2, heads=4):
        super().__init__()
        self.depth_hw = tuple(depth_hw)
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.SiLU(),
            nn.Conv2d(96, width, 3, stride=2, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
        )
        gh = (depth_hw[0] - 1) // 16 + 1
        gw = (depth_hw[1] - 1) // 16 + 1
        self.gh, self.gw = gh, gw
        self.pos = nn.Parameter(torch.zeros(1, gh * gw, width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.target = nn.Sequential(nn.Linear(6, width), nn.SiLU(), nn.Linear(width, width))
        self.queries = nn.Parameter(torch.zeros(1, nq, width))
        nn.init.trunc_normal_(self.queries, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.0,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.tr = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
        self.stop = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, depth, target):
        # depth: (B,1,H,W) in log-metres; target: (B,6) normalized
        x = self.enc(depth).flatten(2).transpose(1, 2)  # B,N,C
        x = x + self.pos[:, : x.shape[1]]
        t = self.target(target).unsqueeze(1)
        q = self.queries.expand(x.shape[0], -1, -1)
        seq = torch.cat([t, q, x], dim=1)
        y = self.tr(seq)
        q = y[:, 1:1 + self.queries.shape[1]]
        return self.out(q), self.stop(q.mean(1)).squeeze(-1)


def count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class NavDepth(nn.Module):
    """Frozen DINOv2 with depth + per-pixel collision/obstacle heads."""

    def __init__(self, dino, depth_feat=256, mask_feat=128):
        super().__init__()
        self.dino = dino
        self.depth = DPTHead(feat=depth_feat)
        self.collision = DPTHead(feat=mask_feat)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, x, in_hw=(168, 294)):
        dtype = next(self.dino.parameters()).dtype
        with torch.no_grad():
            feats = self.dino.get_intermediate_layers(
                x.to(dtype), n=list(DINO_LAYERS), reshape=False,
                return_class_token=False)
        feats = [f.float() for f in feats]
        grid = (in_hw[0] // 14, in_hw[1] // 14)
        return self.depth(feats, grid), self.collision(feats, grid)

    def predict(self, x, in_hw=(168, 294)):
        logd, logit = self.forward(x, in_hw)
        return torch.exp(logd).clamp(0.2, 40.0), torch.sigmoid(logit)
