# -*- coding: utf-8 -*-
"""Train the DINOv2 absolute-depth readout.  ARDY and DINOv2 stay frozen."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ardy_v.data import ArenaDepth, IMG_H, IMG_W, split_episodes, list_episodes  # noqa: E402
from ardy_v.models import DinoDepth, count_trainable, load_dino  # noqa: E402


def metrics(pred, gt):
    p = pred.clamp(0.2, 60.0)
    g = gt.clamp(0.2, 60.0)
    absrel = ((p - g).abs() / g).mean().item()
    rmse = torch.sqrt(((p - g) ** 2).mean()).item()
    ratio = torch.maximum(p / g, g / p)
    d1 = (ratio < 1.25).float().mean().item()
    return absrel, rmse, d1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data_raw"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "depth.pt"))
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="", help="optional depth head checkpoint to fine-tune")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1
        args.batch = min(args.batch, 2)

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eps = list_episodes(args.data)
    if not eps:
        raise SystemExit(f"no episodes under {args.data}; run tools/generate_data.py first")
    tr, va = split_episodes(eps, val_frac=args.val_frac, seed=args.seed)
    nframes = sum(int(np.load(e / "labels.npz")["stop"].shape[0]) for e in eps)
    print(f"episodes train={len(tr)} val={len(va)} frames={nframes}", flush=True)
    ds_tr = ArenaDepth(tr, augment=True)
    ds_va = ArenaDepth(va, augment=False) if va else None
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=False)
    dl_va = (torch.utils.data.DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)
             if ds_va else None)

    dino = load_dino(device=device, dtype=torch.float16)
    model = DinoDepth(dino).to(device)
    if args.init and Path(args.init).exists():
        sd0 = torch.load(args.init, map_location="cpu")
        model.head.load_state_dict(sd0["head"])
        print("init depth head from", args.init, flush=True)
    print(f"trainable head params {count_trainable(model)/1e6:.2f}M; "
          f"dino {sum(p.numel() for p in model.dino.parameters())/1e6:.1f}M frozen", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    best = 1e9
    hist = []
    steps = 0
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        model.dino.eval()
        run = 0.0
        n = 0
        for batch in dl_tr:
            x = batch["image"].to(device, non_blocking=True)
            gt = batch["depth"].to(device, non_blocking=True)
            logp = model(x, in_hw=(IMG_H, IMG_W))
            gt = gt.clamp(0.2, 60.0)
            logg = torch.log(gt)
            l_log = F.l1_loss(logp, logg)
            p = torch.exp(logp).clamp(0.2, 60.0)
            l_abs = F.l1_loss(p, gt)
            diff = logp - logg
            silog = torch.sqrt((diff ** 2).mean() - diff.mean() ** 2 + 1e-6)
            loss = l_log + 0.5 * l_abs + 0.05 * silog
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            steps += 1
            run += float(loss)
            n += 1
            if args.smoke:
                print(f"smoke step {steps} loss={float(loss):.4f} "
                      f"log={float(l_log):.4f} abs={float(l_abs):.4f} "
                      f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
                if steps >= 2:
                    break
        sched.step()
        va_res = None
        if dl_va is not None:
            model.eval()
            tot = [0.0, 0.0, 0.0]
            cnt = 0
            with torch.no_grad():
                for batch in dl_va:
                    x = batch["image"].to(device)
                    gt = batch["depth"].to(device)
                    p = model.predict_meters(x, in_hw=(IMG_H, IMG_W))
                    a, r, d = metrics(p, gt)
                    tot[0] += a * x.shape[0]
                    tot[1] += r * x.shape[0]
                    tot[2] += d * x.shape[0]
                    cnt += x.shape[0]
                    if args.smoke:
                        break
            va_res = [v / max(1, cnt) for v in tot]
            print(f"epoch {ep+1}/{args.epochs} loss={run/max(1,n):.4f} "
                  f"val absrel={va_res[0]:.4f} rmse={va_res[1]:.3f} d1={va_res[2]:.3f} "
                  f"{(time.perf_counter()-t0):.1f}s", flush=True)
            if va_res[1] < best:
                best = va_res[1]
                if not args.smoke:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"head": model.head.state_dict(), "args": vars(args)}, args.out)
        if args.smoke:
            break
    if args.smoke:
        out = Path(args.out).with_name("depth_smoke.pt")
        torch.save({"head": model.head.state_dict(), "args": vars(args)}, out)
        print("smoke saved", out, flush=True)
    else:
        meta = {"args": vars(args), "best_rmse": best, "metrics": hist}
        Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1), "utf-8")
        print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
