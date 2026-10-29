# -*- coding: utf-8 -*-
"""Post-train a YOLOv8-backbone absolute-depth head; compare with DINOv2."""
from __future__ import annotations

import argparse
import os
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

from ardy_v.data import ArenaDepthYolo, split_episodes, list_episodes  # noqa: E402
from ardy_v.yolo_depth import YoloDepth, count_trainable  # noqa: E402

YOLO_H, YOLO_W = 192, 320


def metrics(pred, gt):
    p = pred.clamp(0.2, 60.0); g = gt.clamp(0.2, 60.0)
    absrel = ((p - g).abs() / g).mean().item()
    rmse = torch.sqrt(((p - g) ** 2).mean()).item()
    ratio = torch.maximum(p / g, g / p)
    return absrel, rmse, (ratio < 1.25).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data_demo"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "yolo_depth_demo.pt"))
    ap.add_argument("--weights", default=os.environ.get("FLOWSTATE_YOLO_WEIGHTS", "weights/yolov8s.pt"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1; args.batch = min(args.batch, 2)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eps = list_episodes(args.data)
    tr, va = split_episodes(eps, val_frac=0.2, seed=args.seed)
    ds_tr = ArenaDepthYolo(tr, augment=True)
    ds_va = ArenaDepthYolo(va, augment=False) if va else None
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0)
    dl_va = (torch.utils.data.DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                                         num_workers=0) if ds_va else None)
    model = YoloDepth(args.weights).to(device)
    print(f"head params {count_trainable(model)/1e6:.2f}M; backbone frozen", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    best = 1e9; steps = 0; t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train(); model.backbone.eval()
        run = 0.0; n = 0
        for b in dl_tr:
            x = b["image"].to(device); gt = b["depth"].to(device)
            logp = model(x)
            gt = gt.clamp(0.2, 60.0); logg = torch.log(gt)
            l_log = F.l1_loss(logp, logg)
            p = torch.exp(logp).clamp(0.2, 60.0)
            l_abs = F.l1_loss(p, gt)
            diff = logp - logg
            silog = torch.sqrt((diff ** 2).mean() - diff.mean() ** 2 + 1e-6)
            loss = l_log + 0.5 * l_abs + 0.05 * silog
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); steps += 1; run += float(loss); n += 1
            if args.smoke:
                print(f"smoke step {steps} loss {float(loss):.4f} "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
                if steps >= 2: break
        sched.step()
        if dl_va is not None:
            model.eval(); tot = np.zeros(3); cnt = 0
            with torch.no_grad():
                for b in dl_va:
                    x = b["image"].to(device); gt = b["depth"].to(device)
                    p = model.predict_meters(x)
                    a, r, d = metrics(p, gt)
                    k = x.shape[0]; tot += np.array([a, r, d]) * k; cnt += k
                    if args.smoke: break
            tot /= max(1, cnt)
            print(f"epoch {ep+1}/{args.epochs} loss {run/max(1,n):.4f} "
                  f"val absrel {tot[0]:.4f} rmse {tot[1]:.3f} d1 {tot[2]:.3f} "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)
            if tot[1] < best:
                best = float(tot[1])
                if not args.smoke:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"head": model.head.state_dict(), "args": vars(args),
                                "channels": [128, 256, 512]}, args.out)
        if args.smoke:
            break
    if args.smoke:
        out = Path(args.out).with_name("yolo_depth_smoke.pt")
        torch.save({"head": model.head.state_dict(), "args": vars(args)}, out)
        print("smoke saved", out, flush=True)
    else:
        print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
