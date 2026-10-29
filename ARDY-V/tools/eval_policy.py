# -*- coding: utf-8 -*-
"""Evaluate the trained depth + boundary heads on held-out episodes.

Open-loop metrics are on the episode frames.  The optional closed-loop mode
rolls the boundary head with ground-truth depth in the same arena (no ARDY),
which isolates the visual boundary from the motion generator.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ardy_v.arena import Arena, Box  # noqa: E402
from ardy_v.data import (ArenaBoundary, ArenaDepth, BND_H, BND_W, DEPTH_MAX,  # noqa: E402
                         DEPTH_MIN, IMG_H, IMG_W, LOG_MAX, RES_SCALE, base_path,
                         list_episodes, split_episodes, target_norm)
from ardy_v.models import BoundaryHead, DinoDepth, load_dino  # noqa: E402


def depth_metrics(pred, gt):
    p = pred.clamp(0.2, 60.0)
    g = gt.clamp(0.2, 60.0)
    absrel = ((p - g).abs() / g).mean().item()
    rmse = torch.sqrt(((p - g) ** 2).mean()).item()
    ratio = torch.maximum(p / g, g / p)
    d1 = (ratio < 1.25).float().mean().item()
    return absrel, rmse, d1


def path_metrics(pred, gt):
    p = pred.clone().float()
    g = gt.clone().float()
    l1 = (p - g).abs().mean().item()
    ep = torch.linalg.norm(p[:, -1, :2] - g[:, -1, :2], dim=-1).mean().item()
    yaw = (p[:, -1, 2] - g[:, -1, 2]).abs().mean().item()
    return l1, ep, yaw


def load_models(device, depth_ckpt, boundary_ckpt):
    dino = load_dino(device=device, dtype=torch.float16)
    dm = DinoDepth(dino).to(device).eval()
    dm.head.load_state_dict(torch.load(depth_ckpt, map_location="cpu")["head"])
    bm = BoundaryHead().to(device).eval()
    bm.load_state_dict(torch.load(boundary_ckpt, map_location="cpu")["head"])
    for p in dm.parameters():
        p.requires_grad_(False)
    for p in bm.parameters():
        p.requires_grad_(False)
    return dm, bm


def open_loop(args, dm, bm, va, device):
    ds = ArenaDepth(va, augment=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
    depth_tot = np.zeros(3)
    n = 0
    with torch.no_grad():
        for b in dl:
            x = b["image"].to(device)
            gt = b["depth"].to(device)
            p = dm.predict_meters(x, in_hw=(IMG_H, IMG_W))
            a, r, d = depth_metrics(p, gt)
            depth_tot += np.array([a, r, d]) * x.shape[0]
            n += x.shape[0]
    depth_tot /= max(1, n)

    # cache predicted depth at boundary resolution
    pred = {}
    for ep in va:
        dse = ArenaDepth([ep], augment=False)
        dle = torch.utils.data.DataLoader(dse, batch_size=16, shuffle=False)
        rows = []
        with torch.no_grad():
            for b in dle:
                x = b["image"].to(device)
                p = dm.predict_meters(x, in_hw=(IMG_H, IMG_W))
                p = F.interpolate(p[:, None], size=(BND_H, BND_W), mode="bilinear",
                                  align_corners=False)[:, 0]
                rows.append(p.float().cpu().numpy().astype(np.float16))
        pred[str(ep)] = np.concatenate(rows, 0)

    dsb = ArenaBoundary(va, pred_depth=pred, use_pred=1.0, augment=False)
    dlb = torch.utils.data.DataLoader(dsb, batch_size=16, shuffle=False)
    res_tot = np.zeros(3)
    base_tot = np.zeros(3)
    stop_hit = 0
    stop_n = 0
    with torch.no_grad():
        for b in dlb:
            d = b["depth"].to(device)
            t = b["target"].to(device)
            g = b["path"].to(device)
            base = b["base"].to(device)
            stop = b["stop"].to(device)
            r, logit = bm(d, t)
            scale = torch.as_tensor(RES_SCALE, device=device)
            full = base + r * scale
            l1, ep, yaw = path_metrics(full.cpu(), g.cpu())
            bl1, bep, byaw = path_metrics(base.cpu(), g.cpu())
            k = d.shape[0]
            res_tot += np.array([l1, ep, yaw]) * k
            base_tot += np.array([bl1, bep, byaw]) * k
            hit = ((torch.sigmoid(logit) > 0.5).float() == stop).float().sum().item()
            stop_hit += hit
            stop_n += k
    res_tot /= max(1, stop_n)
    base_tot /= max(1, stop_n)
    return {
        "depth": {"absrel": depth_tot[0], "rmse": depth_tot[1], "d1": depth_tot[2],
                  "frames": n},
        "boundary": {"residual": {"path_l1_m": res_tot[0], "endpoint_m": res_tot[1],
                                  "yaw_rad": res_tot[2]},
                     "base_only": {"path_l1_m": base_tot[0], "endpoint_m": base_tot[1],
                                   "yaw_rad": base_tot[2]},
                     "stop_acc": stop_hit / max(1, stop_n), "frames": stop_n},
    }


def closed_loop(args, bm, va, device):
    """Roll the boundary head with GT depth in each held-out arena."""
    rows = []
    for ep in va:
        spec = json.loads((ep / "spec.json").read_text("utf-8"))
        arena = Arena(half=spec["half"], ceiling=spec["ceiling"],
                      boxes=[Box(**b) for b in spec["boxes"]])
        z = np.load(ep / "labels.npz")
        start = z["start"].astype(np.float64)
        target = z["target_world"].astype(np.float64)
        root = start.copy()
        yaw = float(z["yaw0"][0]) if "yaw0" in z else 0.0
        arrived = False
        collided = False
        for step in range(40):
            d = np.asarray(root, np.float32)
            dist = float(np.linalg.norm(target - root))
            if dist < 0.45:
                arrived = True
                break
            depth = arena.raycast_depth([root[0], 1.55, root[1]], yaw, -0.06,
                                        BND_W, BND_H)
            dx, dz = target[0] - root[0], target[1] - root[1]
            fwd = dx * math.sin(yaw) + dz * math.cos(yaw)
            lat = dx * math.cos(yaw) - dz * math.sin(yaw)
            bearing = math.atan2(lat, fwd)
            tvec = np.array([fwd / 4, lat / 4, dist / 10,
                             math.sin(bearing), math.cos(bearing), 1], np.float32)
            dn = np.log(np.clip(depth, DEPTH_MIN, DEPTH_MAX)) / LOG_MAX
            with torch.no_grad():
                p, logit = bm(torch.from_numpy(dn[None, None]).to(device),
                              torch.from_numpy(tvec[None]).to(device))
            base = base_path(tvec)
            path = base + p[0].float().cpu().numpy() * RES_SCALE
            if float(torch.sigmoid(logit)) > 0.5:
                arrived = True
                break
            fw = np.array([math.sin(yaw), math.cos(yaw)])
            rt = np.array([math.cos(yaw), -math.sin(yaw)])
            nxt = root + path[-1, 0] * fw + path[-1, 1] * rt
            if not arena.point_free(nxt, clearance=0.24):
                collided = True
                root = nxt
                break
            root = nxt
            yaw = float(yaw + path[-1, 2])
        rows.append({"ep": ep.name, "arrived": arrived, "collided": collided,
                     "dist": float(np.linalg.norm(target - root)),
                     "steps": step + 1})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data_full"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "eval_policy.json"))
    ap.add_argument("--closed-loop", action="store_true")
    ap.add_argument("--depth", default=str(ROOT / "checkpoints" / "depth.pt"))
    ap.add_argument("--boundary", default=str(ROOT / "checkpoints" / "boundary.pt"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eps = list_episodes(args.data)
    _, va = split_episodes(eps, val_frac=0.2, seed=0)
    print(f"held-out episodes {len(va)}", flush=True)
    dm, bm = load_models(device, args.depth, args.boundary)
    out = open_loop(args, dm, bm, va, device)
    print(json.dumps(out, indent=1), flush=True)
    if args.closed_loop:
        rows = closed_loop(args, bm, va, device)
        out["closed_loop"] = rows
        out["closed_loop_summary"] = {
            "arrived": sum(r["arrived"] for r in rows),
            "collided": sum(r["collided"] for r in rows),
            "n": len(rows),
        }
        print("closed loop", out["closed_loop_summary"], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), "utf-8")
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
