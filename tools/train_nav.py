# -*- coding: utf-8 -*-
"""Depth + collision multi-task head (DINOv2 frozen).

Training-method correction over the earlier depth-only run:
  * obstacle pixels are weighted 4x in the depth loss;
  * a per-pixel obstacle head is supervised directly (BCE + Dice), so
    navigation does not have to infer collision from a dense metric map;
  * data from both the random arena set and the demo scene;
  * mask/edge-weighted depth, not uniform floor-dominated L1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ardy_v.data import ArenaDepthNav, IMG_H, IMG_W, list_episodes, split_episodes  # noqa: E402
from ardy_v.models import NavDepth, count_trainable, load_dino  # noqa: E402


def metrics(pred, gt):
    p = pred.clamp(0.2, 60.0); g = gt.clamp(0.2, 60.0)
    absrel = ((p - g).abs() / g).mean().item()
    rmse = torch.sqrt(((p - g) ** 2).mean()).item()
    ratio = torch.maximum(p / g, g / p)
    return absrel, rmse, (ratio < 1.25).float().mean().item()


def mask_metrics(prob, gt, thr=0.5):
    p = prob > thr; g = gt > 0.5
    inter = (p & g).sum().item(); union = (p | g).sum().item()
    iou = inter / max(union, 1)
    dice = 2 * inter / max(p.sum().item() + g.sum().item(), 1)
    prec = inter / max(p.sum().item(), 1)
    rec = inter / max(g.sum().item(), 1)
    return iou, dice, prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[str(ROOT / "data_full"), str(ROOT / "data_demo")])
    ap.add_argument("--init", default=str(ROOT / "checkpoints" / "depth_demo.pt"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "nav.pt"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1; args.batch = min(args.batch, 2)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eps = []
    for d in args.data:
        eps += list_episodes(d)
    tr, va = split_episodes(eps, val_frac=0.2, seed=args.seed)
    ds_tr = ArenaDepthNav(tr, augment=True)
    ds_va = ArenaDepthNav(va, augment=False) if va else None
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0)
    dl_va = (torch.utils.data.DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                                         num_workers=0) if ds_va else None)
    dino = load_dino(device=device, dtype=torch.float16)
    model = NavDepth(dino).to(device)
    init = Path(args.init)
    if init.exists():
        sd = torch.load(init, map_location="cpu")
        try:
            model.depth.load_state_dict(sd["head"])
            print("depth head init", init, flush=True)
        except Exception as e:
            print("depth init skipped", repr(e), flush=True)
    print(f"trainable {count_trainable(model)/1e6:.2f}M; "
          f"depth+mask params; {len(tr)} train / {len(va)} val episodes", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    best = 1e9; steps = 0; t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train(); model.dino.eval()
        run = 0.0; n = 0
        for b in dl_tr:
            x = b["image"].to(device); gt = b["depth"].to(device)
            mk = b["mask"].to(device)
            logd, logit = model(x, in_hw=(IMG_H, IMG_W))
            gt = gt.clamp(0.2, 60.0); logg = torch.log(gt)
            valid = (gt > 0.2).float()
            gx = torch.zeros_like(gt); gy = torch.zeros_like(gt)
            gx[:, :, 1:] = (gt[:, :, 1:] - gt[:, :, :-1]).abs()
            gy[:, 1:, :] = (gt[:, 1:, :] - gt[:, :-1, :]).abs()
            edge = ((gx + gy) > 0.25).float()
            w = valid * (1.0 + 4.0 * mk + 2.0 * edge)
            w = w / w.mean().clamp_min(1e-6)
            l_log = (F.l1_loss(logd, logg, reduction="none") * w).mean()
            p = torch.exp(logd).clamp(0.2, 60.0)
            l_abs = (F.l1_loss(p, gt, reduction="none") * w).mean()
            prob = torch.sigmoid(logit)
            bce = F.binary_cross_entropy_with_logits(logit, mk, pos_weight=torch.tensor(2.0, device=device))
            dice = 1.0 - (2.0 * (prob * mk).sum() + 1.0) / (prob.sum() + mk.sum() + 1.0)
            loss = l_log + 0.5 * l_abs + 0.8 * bce + 0.3 * dice
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); steps += 1; run += float(loss); n += 1
            if args.smoke:
                print(f"smoke {steps} loss {float(loss):.3f} depth {float(l_log):.3f} "
                      f"bce {float(bce):.3f} dice {float(dice):.3f} "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
                if steps >= 2:
                    break
        sched.step()
        if dl_va is not None:
            model.eval(); tot = np.zeros(7); cnt = 0
            with torch.no_grad():
                for b in dl_va:
                    x = b["image"].to(device); gt = b["depth"].to(device)
                    mk = b["mask"].to(device)
                    logd, logit = model(x, in_hw=(IMG_H, IMG_W))
                    p = torch.exp(logd).clamp(0.2, 60.0)
                    a, r, d = metrics(p, gt)
                    iou, dice, prec, rec = mask_metrics(torch.sigmoid(logit), mk)
                    k = x.shape[0]
                    tot += np.array([a, r, d, iou, dice, prec, rec]) * k; cnt += k
                    if args.smoke:
                        break
            tot /= max(1, cnt)
            print(f"epoch {ep+1}/{args.epochs} loss {run/max(1,n):.3f} "
                  f"absrel {tot[0]:.3f} rmse {tot[1]:.3f} d1 {tot[2]:.3f} "
                  f"mask IoU {tot[3]:.3f} dice {tot[4]:.3f} P {tot[5]:.3f} R {tot[6]:.3f} "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)
            if tot[0] < best:
                best = float(tot[0])
                if not args.smoke:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"depth": model.depth.state_dict(),
                                "collision": model.collision.state_dict(),
                                "args": vars(args), "val": tot.tolist()}, args.out)
        if args.smoke:
            break
    if args.smoke:
        out = Path(args.out).with_name("nav_smoke.pt")
        torch.save({"depth": model.depth.state_dict(),
                    "collision": model.collision.state_dict(), "args": vars(args)}, out)
        print("smoke saved", out, flush=True)
    else:
        print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
