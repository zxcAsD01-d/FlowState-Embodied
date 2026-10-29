"""Capture live eye frames + exact pose, rebuild analytic GT depth, compare.

The live renderer is buildArena() (same as the training renderer), so GT can be
recomputed from the arena and the pose the eye was captured at.  The boxes are
at their reset positions (the avatar stays away from them in this capture).
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "runs"
IMG_W, IMG_H = 294, 168


def bins_of(g, p, tag=""):
    e = p - g
    rows = []
    for a, b in [(0.2, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 40)]:
        s = (g >= a) & (g < b)
        if not s.any():
            continue
        ee = e[s]
        rows.append((a, b, int(s.sum()), float(np.mean(np.abs(ee))),
                     float(np.mean(ee)), float(np.mean(np.abs(ee) / g[s]))))
    print(tag, "N", len(g), "absrel %.3f MAE %.3f bias %+.3f medAE %.3f d1 %.3f" % (
        np.mean(np.abs(e) / g), np.mean(np.abs(e)), np.mean(e),
        np.median(np.abs(e)), np.mean(np.maximum(p / g, g / p) < 1.25)))
    for a, b, n, mae, bias, rel in rows:
        print("  %4.1f-%4.1f m n=%7d MAE %.3f bias %+.3f absrel %.3f" %
              (a, b, n, mae, bias, rel))
    return rows


def main():
    from playwright.sync_api import sync_playwright
    frames = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="msedge", headless=True, args=["--use-angle=default"])
        page = b.new_page(viewport={"width": 1280, "height": 760})
        page.goto("http://127.0.0.1:8830/", wait_until="domcontentloaded")
        page.wait_for_function(
            "() => window.ardyDemo && window.ardyDemo.state().ready && window.ardyDemo.state().frames>0",
            timeout=90000)
        page.evaluate("() => window.ardyDemo.reset()")
        time.sleep(2.0)
        for k in range(8):
            ims = page.evaluate("() => window.ardyDemo.images()")
            st = page.evaluate("() => window.ardyDemo.state()")
            meta = {"eyePose": ims.get("eyePose"), "boxes": st.get("boxes"),
                    "root": st.get("root"), "yaw": st.get("yaw")}
            eye = ims.get("eye")
            if eye and meta["eyePose"]:
                raw = base64.b64decode(eye.split(",", 1)[1])
                (OUT / f"live_depth_{k:02d}.jpg").write_bytes(raw)
                meta["file"] = f"live_depth_{k:02d}.jpg"
                frames.append(meta)
            time.sleep(1.4)
        b.close()
    (OUT / "live_depth_frames.json").write_text(json.dumps(frames, indent=1), "utf-8")

    from ardy_v.arena import demo_arena
    from ardy_v.nav_runtime import NavEncoder
    arena = demo_arena()
    enc = NavEncoder(ROOT / "checkpoints" / "nav.pt")
    P, G = [], []
    for i, f in enumerate(frames):
        pose = f["eyePose"]
        im = Image.open(OUT / f["file"]).convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
        rgb = np.asarray(im, np.uint8)
        gt = arena.raycast_depth([pose["root"][0], 1.55, pose["root"][1]],
                                 float(pose["yaw"]), float(pose["pitch"]),
                                 width=IMG_W, height=IMG_H)
        pred, mask = enc.predict(rgb)
        valid = np.isfinite(gt) & (gt > 0.2) & (gt < 40)
        print(f"frame {i} root {np.round(pose['root'], 2)} yaw {pose['yaw']:.2f} "
              f"gtmean {gt[valid].mean():.2f} predmean {pred[valid].mean():.2f} "
              f"mask {mask.mean():.2f}", flush=True)
        P.append(pred[valid]); G.append(gt[valid])
    p = np.concatenate(P); g = np.concatenate(G)
    rows = bins_of(g, p, "[live]")
    np.savez_compressed(OUT / "live_depth_pred.npz",
                        pred=np.concatenate([x for x in P]),
                        gt=np.concatenate([x for x in G]))
    (OUT / "live_depth_bins.json").write_text(json.dumps(rows), "utf-8")


if __name__ == "__main__":
    main()
