"""YOLOv8 pretrained backbone + trainable absolute-depth head.

The backbone is loaded from the local Ultralytics checkpoint and frozen.  Only
the DPT-style head is trained; this is the "YOLO post-trained depth" branch
used to compare against DINOv2.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = Path(os.environ.get("FLOWSTATE_YOLO_WEIGHTS", "weights/yolov8s.pt"))


class YoloDPTHead(nn.Module):
    def __init__(self, channels=(128, 256, 512), feat=128, out_hw=(192, 320)):
        super().__init__()
        self.out_hw = tuple(out_hw)
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, feat, 1), nn.GroupNorm(8, feat), nn.SiLU())
            for c in channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(feat * 3, feat, 3, padding=1), nn.GroupNorm(8, feat), nn.SiLU(),
            nn.Conv2d(feat, feat, 3, padding=1), nn.GroupNorm(8, feat), nn.SiLU(),
        )
        self.out = nn.Conv2d(feat, 1, 1)

    def forward(self, feats, target_hw=(96, 160)):
        xs = []
        for proj, f in zip(self.proj, feats):
            x = proj(f)
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            xs.append(x)
        x = self.fuse(torch.cat(xs, 1))
        x = self.out(x)
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        return x.squeeze(1)


class YoloDepth(nn.Module):
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, feat=128):
        super().__init__()
        from ultralytics import YOLO
        det = YOLO(str(weights)).model
        self.backbone = det.model[:10]
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        # yolov8n/s P3/P4/P5 channel counts; read them from a dummy forward.
        with torch.no_grad():
            x = torch.zeros(1, 3, 192, 320)
            feats = []
            for i, layer in enumerate(self.backbone):
                x = layer(x)
                if i in (4, 6, 9):
                    feats.append(x)
        channels = tuple(f.shape[1] for f in feats)
        self.head = YoloDPTHead(channels=channels, feat=feat)
        self.in_hw = (192, 320)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        feats = []
        with torch.no_grad():
            y = x
            for i, layer in enumerate(self.backbone):
                y = layer(y)
                if i in (4, 6, 9):
                    feats.append(y)
        feats = [f.float() for f in feats]
        h = self.in_hw[0] // 8
        w = self.in_hw[1] // 8
        return self.head(feats, target_hw=(h, w))

    def predict_meters(self, x, depth_range=(0.2, 40.0)):
        return torch.exp(self.forward(x)).clamp(*depth_range)


def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
