# -*- coding: utf-8 -*-
"""Render arena episodes and store RGB / absolute depth / A* boundary labels.

Each episode is a real rendered scene (three.js in headless Edge) with analytic
ground-truth depth and an A* expert root path.  The RGB is what the learned
encoder sees at inference; depth labels are never fed to the demo policy.
"""
from __future__ import annotations

import argparse
import base64
import functools
import http.server
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WEB = ROOT / "web"
OUT = ROOT / "data_raw"
EYE_H = 1.55


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(port: int):
    handler = functools.partial(Quiet, directory=str(WEB))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def spec_of(arena):
    return {
        "half": float(arena.half),
        "ceiling": float(arena.ceiling),
        "boxes": [{"x": float(b.x), "z": float(b.z), "hx": float(b.hx),
                   "hz": float(b.hz), "h": float(b.h), "yaw": float(b.yaw)}
                  for b in arena.boxes],
    }


def rollout(arena, rng, windows: int, look_frames: int = 4):
    from ardy_v.arena import random_start_target
    start, target = random_start_target(arena, rng, min_dist=4.0)
    # Force one obstacle near the start-target line: without it most A* paths
    # are straight and the residual head has nothing depth-specific to learn.
    if np.linalg.norm(target - start) > 4.5:
        from ardy_v.arena import Box
        mid = 0.5 * (start + target)
        d = target - start
        d = d / max(np.linalg.norm(d), 1e-6)
        perp = np.array([-d[1], d[0]])
        off = float(rng.uniform(-0.55, 0.55))
        bx, bz = mid + perp * off
        hx = float(rng.uniform(0.55, 1.05))
        hz = float(rng.uniform(0.55, 1.05))
        if (np.linalg.norm(np.array([bx, bz]) - start) > 2.2
                and np.linalg.norm(np.array([bx, bz]) - target) > 2.2
                and arena.point_free([bx, bz], hx + 0.2)):
            arena.boxes.append(Box(float(bx), float(bz), hx, hz,
                                   float(rng.uniform(1.5, 2.2)),
                                   float(rng.uniform(-0.3, 0.3))))
    # Random initial facing: the policy must learn to turn toward the target,
    # not inherit a planner that always starts aimed at it.
    yaw = float(rng.uniform(-np.pi, np.pi))
    root = start.astype(np.float64).copy()
    rows = []

    # look-around frames: depth coverage without a navigation target
    for _ in range(look_frames):
        y = yaw + float(rng.uniform(-1.2, 1.2))
        pitch = float(rng.uniform(-0.25, 0.15))
        depth = arena.raycast_depth([root[0], EYE_H, root[1]], y, pitch)
        rows.append({
            "eye": [float(root[0]), EYE_H, float(root[1])],
            "yaw": y, "pitch": pitch,
            "depth": depth,
            "path": np.zeros((8, 3), np.float32),
            "target": arena.target_vector(root, y, target),
            "stop": 0.0,
            "kind": 0.0,
            "root": root.astype(np.float32),
            "yaw0": float(yaw),
        })

    for _ in range(windows):
        pitch = -0.06 + float(rng.normal(0, 0.012))
        yaw_view = yaw + float(rng.normal(0, 0.018))
        body, stop, world = arena.expert_window(root, yaw_view, target)
        depth = arena.raycast_depth([root[0], EYE_H, root[1]], yaw_view, pitch)
        rows.append({
            "eye": [float(root[0]), EYE_H, float(root[1])],
            "yaw": yaw_view, "pitch": pitch,
            "depth": depth,
            "path": body.astype(np.float32),
            "target": arena.target_vector(root, yaw_view, target),
            "stop": float(stop),
            "kind": 1.0,
            "root": root.astype(np.float32),
            "yaw0": float(yaw),
        })
        if stop:
            break
        nxt = world[-1] + rng.normal(0, 0.035, 2)
        if arena.point_free(nxt, 0.30):
            root = nxt
        else:
            root = world[-1]
        yaw = float(yaw + float(body[-1, 2]) + rng.normal(0, 0.015))
    return start, target, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--windows", type=int, default=28)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--demo", action="store_true", help="fixed demo arena (central box)")
    args = ap.parse_args()
    if args.smoke:
        args.episodes = min(args.episodes, 3)
        args.windows = min(args.windows, 6)

    from ardy_v.arena import demo_arena, random_arena
    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    serve(port)
    rng = np.random.default_rng(args.seed)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            channel="msedge", headless=not args.headed,
            args=["--use-angle=default"])
        page = browser.new_page(viewport={"width": 640, "height": 360})
        page.goto(f"http://127.0.0.1:{port}/data.html")
        page.wait_for_function("window.__ready === true", timeout=20000)

        t0 = time.perf_counter()
        total = 0
        for ep in range(args.episodes):
            arena = demo_arena() if args.demo else random_arena(rng, half=8.0)
            start, target, rows = rollout(arena, rng, args.windows)
            spec = spec_of(arena)
            frames = [{"eye": r["eye"], "yaw": r["yaw"], "pitch": r["pitch"]}
                      for r in rows]
            urls = page.evaluate(
                "([spec, frames]) => { window.setScene(spec); return window.renderFrames(frames); }",
                [spec, frames],
            )
            if len(urls) != len(rows):
                raise RuntimeError(f"render returned {len(urls)} for {len(rows)} frames")
            ep_dir = out / f"ep_{ep:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            (ep_dir / "spec.json").write_text(json.dumps(spec), "utf-8")
            jpg = []
            for i, u in enumerate(urls):
                b = base64.b64decode(u.split(",", 1)[1])
                (ep_dir / f"frame_{i:03d}.jpg").write_bytes(b)
                jpg.append(len(b))
            depth = np.stack([r["depth"] for r in rows]).astype(np.float32)
            np.save(ep_dir / "depth_mm.npy",
                    np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))
            np.savez_compressed(
                ep_dir / "labels.npz",
                path=np.stack([r["path"] for r in rows]).astype(np.float32),
                target=np.stack([r["target"] for r in rows]).astype(np.float32),
                stop=np.asarray([r["stop"] for r in rows], np.float32),
                kind=np.asarray([r["kind"] for r in rows], np.float32),
                root=np.stack([r["root"] for r in rows]).astype(np.float32),
                yaw=np.asarray([r["yaw"] for r in rows], np.float32),
                yaw0=np.asarray([r["yaw0"] for r in rows], np.float32),
                start=start.astype(np.float32), target_world=target.astype(np.float32),
            )
            total += len(rows)
            print(f"[{ep+1}/{args.episodes}] frames={len(rows)} "
                  f"boxes={len(arena.boxes)} jpg={np.mean(jpg)/1024:.1f}KB "
                  f"stop={int(sum(r['stop'] for r in rows))} "
                  f"{(time.perf_counter()-t0):.1f}s", flush=True)
        browser.close()
    print(f"done {args.episodes} episodes, {total} frames -> {out}")


if __name__ == "__main__":
    main()
