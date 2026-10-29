"""Datasets for the DINOv2 depth head and the boundary head."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMG_W, IMG_H = 294, 168
DEPTH_W, DEPTH_H = 294, 168
BND_W, BND_H = 294, 168
DEPTH_MIN, DEPTH_MAX = 0.2, 40.0
LOG_MAX = float(np.log(DEPTH_MAX))
# The boundary head predicts a residual around an analytic target-seeking arc.
# This keeps the goal direction out of the learned part; the network only has
# to learn how depth boundaries bend that arc.
RES_SCALE = np.array([0.35, 0.35, 0.5], np.float32)


def base_path(tvec: np.ndarray, horizon: int = 8) -> np.ndarray:
    """Target-seeking arc in the current pelvis frame (8x3, metres/rad)."""
    t = np.asarray(tvec, np.float32)
    fwd, lat, dist = float(t[0]) * 4.0, float(t[1]) * 4.0, float(t[2]) * 10.0
    bearing = float(np.arctan2(t[3], t[4]))
    out = np.zeros((horizon, 3), np.float32)
    if dist < 0.45:
        return out
    unit = np.array([fwd, lat], np.float64) / max(dist, 1e-6)
    step = min(0.46, dist)
    end = unit * step
    if end[0] < 0.0:
        end[0] = 0.05
        end[1] = float(np.sign(end[1]) * min(0.20, abs(end[1])))
    dyaw = float(np.clip(bearing, -0.7, 0.7))
    for k in range(horizon):
        a = (k + 1) / horizon
        out[k, 0] = end[0] * a
        out[k, 1] = end[1] * a
        out[k, 2] = dyaw * a
    return out


def list_episodes(root: str | Path):
    root = Path(root)
    return sorted([p for p in root.glob("ep_*") if (p / "labels.npz").exists()])


def split_episodes(eps, val_frac=0.2, seed=0):
    idx = np.arange(len(eps))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    nval = max(1, int(round(len(eps) * val_frac))) if len(eps) > 1 else 0
    val = set(int(i) for i in idx[:nval])
    train = [e for i, e in enumerate(eps) if i not in val]
    val_eps = [e for i, e in enumerate(eps) if i in val]
    return train, val_eps


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
        return np.asarray(im, np.float32) / 255.0


def _normalize_rgb(rgb: np.ndarray) -> torch.Tensor:
    x = (rgb - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32))


def _resize_depth(d: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


_DEPTH_CACHE: dict[str, np.ndarray | None] = {}


def _depth_episode(ep: Path):
    key = str(ep)
    if key not in _DEPTH_CACHE:
        f = ep / "depth_mm.npy"
        # Load once per episode.  Per-item mmap handles exhausted the Windows
        # mapping limit around epoch 10 of a 40-epoch run.
        _DEPTH_CACHE[key] = np.load(f) if f.exists() else None
    return _DEPTH_CACHE[key]


def _read_depth(ep: Path, z, i: int) -> np.ndarray:
    d = _depth_episode(ep)
    if d is not None:
        return d[i].astype(np.float32) / 1000.0
    return np.asarray(z["depth_m"][i], np.float32)


def target_norm(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, np.float32).copy()
    out = np.array([t[0] / 4.0, t[1] / 4.0, t[2] / 10.0,
                    t[3], t[4], 1.0], np.float32)
    return np.clip(out, -3.0, 3.0)


def path_norm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, np.float32).copy()
    p[:, 0] /= 0.6
    p[:, 1] /= 0.6
    p[:, 2] /= 0.7
    return np.clip(p, -3.0, 3.0)


def depth_norm(d: np.ndarray) -> np.ndarray:
    d = np.clip(np.asarray(d, np.float32), DEPTH_MIN, DEPTH_MAX)
    return np.log(d) / LOG_MAX


YOLO_W, YOLO_H = 320, 192


class ArenaDepthNav(Dataset):
    """RGB + metric depth + per-pixel obstacle mask (floor/ceiling excluded)."""
    def __init__(self, eps, augment=False):
        self.eps = list(eps)
        self.index = []
        for ep_i, ep in enumerate(self.eps):
            z = np.load(ep / "labels.npz")
            n = len(z["stop"])
            self.index.extend((ep_i, i) for i in range(n))
        self.augment = augment

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        ep_i, i = self.index[k]
        ep = self.eps[ep_i]
        z = np.load(ep / "labels.npz")
        rgb = _read_rgb(ep / f"frame_{i:03d}.jpg")
        d_native = _read_depth(ep, z, i)
        root = np.asarray(z["root"][i], np.float64)
        yaw = float(z["yaw"][i])
        from .pointcloud import obstacle_mask
        m_native = obstacle_mask(d_native, root, yaw)
        mask = cv2.resize(m_native.astype(np.uint8), (DEPTH_W, DEPTH_H),
                          interpolation=cv2.INTER_NEAREST)
        depth = _resize_depth(d_native, DEPTH_W, DEPTH_H)
        if self.augment and np.random.rand() < 0.5:
            rgb = rgb[:, ::-1].copy()
            depth = depth[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        return {
            "image": _normalize_rgb(rgb),
            "depth": torch.from_numpy(depth.astype(np.float32)),
            "mask": torch.from_numpy(mask.astype(np.float32)),
            "ep": ep_i, "i": i,
        }


class ArenaDepthYolo(Dataset):
    """RGB 0-1 (YOLO preprocessing) + metric depth at YOLO resolution."""
    def __init__(self, eps, augment=False):
        self.eps = list(eps)
        self.index = []
        for ep_i, ep in enumerate(self.eps):
            z = np.load(ep / "labels.npz")
            n = len(z["stop"])
            self.index.extend((ep_i, i) for i in range(n))
        self.augment = augment

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        ep_i, i = self.index[k]
        ep = self.eps[ep_i]
        z = np.load(ep / "labels.npz")
        with Image.open(ep / f"frame_{i:03d}.jpg") as im:
            im = im.convert("RGB").resize((YOLO_W, YOLO_H), Image.BILINEAR)
            rgb = np.asarray(im, np.float32) / 255.0
        depth = _read_depth(ep, z, i)
        depth = cv2.resize(depth.astype(np.float32), (YOLO_W, YOLO_H),
                           interpolation=cv2.INTER_LINEAR)
        if self.augment and np.random.rand() < 0.5:
            rgb = rgb[:, ::-1].copy()
            depth = depth[:, ::-1].copy()
        return {
            "image": torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32)),
            "depth": torch.from_numpy(depth.astype(np.float32)),
            "ep": ep_i, "i": i,
        }


class ArenaDepth(Dataset):
    def __init__(self, eps, augment=False):
        self.eps = list(eps)
        self.index = []
        for ep_i, ep in enumerate(self.eps):
            z = np.load(ep / "labels.npz")
            n = len(z["stop"])
            self.index.extend((ep_i, i) for i in range(n))
        self.augment = augment

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        ep_i, i = self.index[k]
        ep = self.eps[ep_i]
        z = np.load(ep / "labels.npz")
        rgb = _read_rgb(ep / f"frame_{i:03d}.jpg")
        depth = _resize_depth(_read_depth(ep, z, i), DEPTH_W, DEPTH_H)
        if self.augment:
            if np.random.rand() < 0.5:
                rgb = rgb[:, ::-1].copy()
                depth = depth[:, ::-1].copy()
        return {
            "image": _normalize_rgb(rgb),
            "depth": torch.from_numpy(depth.astype(np.float32)),
            "ep": ep_i,
            "i": i,
        }


class ArenaBoundary(Dataset):
    def __init__(self, eps, pred_depth: dict | None = None, use_pred=0.5,
                 augment=False, seed=0):
        self.eps = list(eps)
        self.pred = pred_depth or {}
        self.use_pred = float(use_pred)
        self.rng = np.random.default_rng(seed)
        self.augment = augment
        self.index = []
        for ep_i, ep in enumerate(self.eps):
            z = np.load(ep / "labels.npz")
            kind = z["kind"] if "kind" in z else np.ones(len(z["stop"]), np.float32)
            n = len(z["stop"])
            self.index.extend((ep_i, i) for i in range(n) if float(kind[i]) > 0.5)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        ep_i, i = self.index[k]
        ep = self.eps[ep_i]
        z = np.load(ep / "labels.npz")
        use_pred = self.pred and (self.rng.random() < self.use_pred)
        if use_pred:
            d = self.pred[str(ep)][i]
            d = _resize_depth(d, BND_W, BND_H)
            if self.rng.random() < 0.5:
                d = d * np.exp(float(self.rng.normal(0, 0.05))) + float(self.rng.uniform(-0.02, 0.02))
        else:
            d = _resize_depth(_read_depth(ep, z, i), BND_W, BND_H)
        x = depth_norm(d)[None]
        tgt = target_norm(z["target"][i])
        path = z["path"][i]
        base = base_path(tgt)
        stop = float(z["stop"][i])
        if self.augment and self.rng.random() < 0.5:
            # Horizontal mirror: lateral/dyaw and target lateral/bearing flip.
            x = x[:, :, ::-1].copy()
            path = path.copy()
            path[:, 1] *= -1
            path[:, 2] *= -1
            tgt = tgt.copy()
            tgt[1] *= -1
            tgt[3] *= -1
            base = base.copy()
            base[:, 1] *= -1
            base[:, 2] *= -1
        return {
            "depth": torch.from_numpy(x.astype(np.float32)),
            "target": torch.from_numpy(tgt.astype(np.float32)),
            "path": torch.from_numpy(path.astype(np.float32)),
            "base": torch.from_numpy(base.astype(np.float32)),
            "stop": torch.tensor(stop, dtype=torch.float32),
            "ep": ep_i,
            "i": i,
        }
