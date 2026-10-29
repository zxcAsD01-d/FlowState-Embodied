# -*- coding: utf-8 -*-
"""Train the depth+target -> root-boundary head with ARDY and DINOv2 frozen."""
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

from ardy_v.data import (ArenaBoundary, ArenaDepth, BND_H, BND_W, IMG_H, IMG_W,  # noqa: E402
                         RES_SCALE, list_episodes, split_episodes)
from ardy_v.models import BoundaryHead, DinoDepth, count_trainable, load_dino  # noqa: E402


def predict_depth_maps(eps, depth_ckpt, device, batch=8):
    """Cache the trained depth head's prediction at boundary resolution."""
    dino = load_dino(device=device, dtype=torch.float16)
    model = DinoDepth(dino).to(device)
    sd = torch.load(depth_ckpt, map_location="cpu")
    model.head.load_state_dict(sd["head"])
    model.eval()
    out = {}
    for ep in eps:
        ds = ArenaDepth([ep], augment=False)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
        rows = []
        with torch.no_grad():
            for b in dl:
                x = b["image"].to(device)
                p = model.predict_meters(x, in_hw=(IMG_H, IMG_W))
                p = F.interpolate(p[:, None], size=(BND_H, BND_W), mode="bilinear",
                                  align_corners=False)[:, 0]
                rows.append(p.float().cpu().numpy().astype(np.float16))
        out[str(ep)] = np.concatenate(rows, 0)
        print(f"depth cache {ep.name}: {out[str(ep)].shape}", flush=True)
    return out


def path_metrics(pred, gt):
    # normalized tensors -> metres / radians
    p = pred.float()
    p[..., 0] *= 0.6
    p[..., 1] *= 0.6
    p[..., 2] *= 0.7
    g = gt.float()
    g[..., 0] *= 0.6
    g[..., 1] *= 0.6
    g[..., 2] *= 0.7
    l1 = (p - g).abs().mean().item()
    ep_err = torch.linalg.norm(p[:, -1, :2] - g[:, -1, :2], dim=-1).mean().item()
    yaw = (p[:, -1, 2] - g[:, -1, 2]).abs().mean().item()
    return l1, ep_err, yaw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data_raw"))
    ap.add_argument("--depth", default=str(ROOT / "checkpoints" / "depth.pt"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "boundary.pt"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--use-pred", type=float, default=0.6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1
        args.batch = min(args.batch, 4)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eps = list_episodes(args.data)
    tr, va = split_episodes(eps, val_frac=args.val_frac, seed=args.seed)
    print(f"episodes train={len(tr)} val={len(va)}", flush=True)

    pred_tr = pred_va = None
    if not args.smoke and Path(args.depth).exists():
        pred_tr = predict_depth_maps(tr, args.depth, device)
        pred_va = predict_depth_maps(va, args.depth, device) if va else None
    elif not args.smoke:
        print("no depth checkpoint; training on GT depth only", flush=True)

    ds_tr = ArenaBoundary(tr, pred_depth=pred_tr, use_pred=args.use_pred, augment=True,
                          seed=args.seed)
    ds_va = (ArenaBoundary(va, pred_depth=pred_va, use_pred=1.0, augment=False,
                           seed=args.seed + 1) if va else None)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0)
    dl_va = (torch.utils.data.DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                                         num_workers=0) if ds_va else None)
    model = BoundaryHead().to(device)
    print(f"boundary params {count_trainable(model)/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    best = 1e9
    t0 = time.perf_counter()
    steps = 0
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        n = 0
        for b in dl_tr:
            d = b["depth"].to(device)
            t = b["target"].to(device)
            g = b["path"].to(device)
            base = b["base"].to(device)
            stop = b["stop"].to(device)
            res, logit = model(d, t)
            scale = torch.as_tensor(RES_SCALE, device=device)
            p = base + res * scale
            per = F.l1_loss(p, g, reduction="none").mean(-1)
            # Hard-case weighting: frames where the depth must bend the base
            # arc are rare; without this the head averages them away and the
            # closed loop collides at the first box.
            hard = (torch.linalg.norm(g[:, -1, :2] - base[:, -1, :2], dim=-1) / 0.35
                    + (g[:, -1, 2] - base[:, -1, 2]).abs() / 0.5).clamp(0, 3)
            w = (1.0 - 0.6 * stop) * (1.0 + 2.0 * hard)
            path_loss = (per * w[:, None]).mean()
            end_loss = (F.l1_loss(p[:, -1], g[:, -1], reduction="none").mean(-1)
                        * w).mean()
            stop_loss = F.binary_cross_entropy_with_logits(logit, stop)
            smooth = ((p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) ** 2).mean()
            loss = path_loss + 0.8 * end_loss + 0.6 * stop_loss + 0.05 * smooth
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
            run += float(loss)
            n += 1
            if args.smoke:
                print(f"smoke step {steps} loss={float(loss):.4f} path={float(path_loss):.4f} "
                      f"stop={float(stop_loss):.4f} peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB",
                      flush=True)
                if steps >= 2:
                    break
        sched.step()
        if dl_va is not None:
            model.eval()
            tot = np.zeros(4)
            cnt = 0
            with torch.no_grad():
                for b in dl_va:
                    d = b["depth"].to(device)
                    t = b["target"].to(device)
                    g = b["path"].to(device)
                    base = b["base"].to(device)
                    stop = b["stop"].to(device)
                    res, logit = model(d, t)
                    scale = torch.as_tensor(RES_SCALE, device=device)
                    p = base + res * scale
                    l1, ep_err, yaw = path_metrics(p.cpu(), g.cpu())
                    acc = ((logit > 0).float() == stop).float().mean().item()
                    n = d.shape[0]
                    tot += np.array([l1, ep_err, yaw, acc]) * n
                    cnt += n
                    if args.smoke:
                        break
            tot /= max(1, cnt)
            print(f"epoch {ep+1}/{args.epochs} loss={run/max(1,n):.4f} "
                  f"val pathL1={tot[0]:.3f}m end={tot[1]:.3f}m yaw={tot[2]:.3f}rad "
                  f"stopAcc={tot[3]:.3f} {(time.perf_counter()-t0):.1f}s", flush=True)
            if tot[1] < best:
                best = float(tot[1])
                if not args.smoke:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"head": model.state_dict(), "args": vars(args),
                                "val": tot.tolist()}, args.out)
        if args.smoke:
            break
    if args.smoke:
        out = Path(args.out).with_name("boundary_smoke.pt")
        torch.save({"head": model.state_dict(), "args": vars(args)}, out)
        print("smoke saved", out, flush=True)
    else:
        meta = {"args": vars(args), "best_endpoint_m": best}
        Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1), "utf-8")
        print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
