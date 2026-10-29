"""Fit a monotone distance calibration on the train split, test on val + live.

The nav depth head has a distance-dependent bias (too far <2 m, too near >8 m).
A monotone piecewise-linear map pred -> corrected is fitted on train pixels and
validated on held-out episodes and on the live browser frames.  No retraining.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ardy_v.data import ArenaDepthNav, IMG_H, IMG_W, list_episodes, split_episodes  # noqa: E402
from ardy_v.models import NavDepth, load_dino  # noqa: E402

OUT = ROOT / "checkpoints" / "depth_calib.json"


def collect(ds, model, device, limit=None):
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    P, G = [], []
    with torch.no_grad():
        for i, b in enumerate(dl):
            d, _ = model.predict(b["image"].to(device), in_hw=(IMG_H, IMG_W))
            P.append(d.float().cpu().numpy().reshape(-1))
            G.append(b["depth"].numpy().reshape(-1))
            if limit and i + 1 >= limit:
                break
    return np.concatenate(P), np.concatenate(G)


def stats(g, p, tag):
    e = p - g
    print("%-6s N=%d absrel %.3f MAE %.3f bias %+.3f medAE %.3f d1 %.3f" % (
        tag, len(g), np.mean(np.abs(e) / g), np.mean(np.abs(e)), np.mean(e),
        np.median(np.abs(e)), np.mean(np.maximum(p / g, g / p) < 1.25)), flush=True)
    for a, b in [(0.2, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 40)]:
        s = (g >= a) & (g < b)
        if s.any():
            ee = e[s]
            print("   %4.1f-%4.1f n=%7d MAE %.3f bias %+.3f absrel %.3f" % (
                a, b, s.sum(), np.mean(np.abs(ee)), np.mean(ee),
                np.mean(np.abs(ee) / g[s])), flush=True)


def fit_knots(p, g, n=17):
    qs = np.linspace(0.005, 0.995, n)
    edges = np.quantile(p, qs)
    edges = np.unique(np.round(edges, 4))
    ks, vs = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (p >= a) & (p < b)
        if s.sum() < 200:
            continue
        ks.append(0.5 * (a + b))
        vs.append(float(np.median(g[s])))
    ks = np.asarray(ks); vs = np.asarray(vs)
    # enforce monotone increasing corrected depth
    vs = np.maximum.accumulate(vs)
    for i in range(1, len(vs)):
        if vs[i] <= vs[i - 1]:
            vs[i] = vs[i - 1] + 1e-3
    return ks, vs


def apply_calib(p, ks, vs):
    return np.interp(p, ks, vs).astype(np.float32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eps = list_episodes(ROOT / "data_full") + list_episodes(ROOT / "data_demo")
    tr, va = split_episodes(eps, val_frac=0.2, seed=0)
    dino = load_dino(device=device, dtype=torch.float16)
    model = NavDepth(dino).to(device)
    ck = torch.load(ROOT / "checkpoints" / "nav.pt", map_location="cpu")
    model.depth.load_state_dict(ck["depth"])
    model.collision.load_state_dict(ck["collision"])
    model.eval()

    print("train episodes", len(tr), "val", len(va), flush=True)
    p, g = collect(ArenaDepthNav(tr, augment=False), model, device)
    ok = (g > 0.2) & (g < 40) & np.isfinite(p)
    p, g = p[ok], g[ok]
    # pixel subsample for fitting only
    idx = np.random.default_rng(0).choice(len(p), size=min(len(p), 3_000_000), replace=False)
    ks, vs = fit_knots(p[idx], g[idx])
    print("knots pred", np.round(ks, 2).tolist(), flush=True)
    print("knots gt  ", np.round(vs, 2).tolist(), flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"pred": ks.tolist(), "gt": vs.tolist()}), "utf-8")

    stats(g, p, "train")
    stats(g, apply_calib(p, ks, vs), "trainC")

    pv, gv = collect(ArenaDepthNav(va, augment=False), model, device)
    ok = (gv > 0.2) & (gv < 40) & np.isfinite(pv)
    pv, gv = pv[ok], gv[ok]
    stats(gv, pv, "val")
    stats(gv, apply_calib(pv, ks, vs), "valC")

    live = np.load(ROOT / "runs" / "live_depth_pred.npz")
    pl, gl = live["pred"], live["gt"]
    stats(gl, pl, "live")
    stats(gl, apply_calib(pl, ks, vs), "liveC")


if __name__ == "__main__":
    main()
