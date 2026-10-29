"""ARDY-v demo: DINOv2 depth + collision -> online occupancy -> planned route.

Each window the browser sends the rendered eye view.  The multi-task head
predicts metric depth and an obstacle mask; the mask + depth build/refresh a
world log-odds map.  A route is planned once from the current map, monitored
every window, and re-planned only when it becomes infeasible (confirmed over
two windows).  ARDY follows the route as a root boundary.
"""
from __future__ import annotations

import base64
import os
import io
import sys
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ardy_v.arena import demo_arena  # noqa: E402
from ardy_v.data import IMG_H, IMG_W  # noqa: E402
from ardy_v.nav_runtime import NavEncoder  # noqa: E402
from ardy_v.online_nav import VisualNavigator  # noqa: E402
from ardy_v.policy import colorize_depth  # noqa: E402
from ardy_v.region import region_png  # noqa: E402
from ardy_v.runtime import ArdyV  # noqa: E402

WEB = ROOT / "web"
VRM = Path(os.environ.get("FLOWSTATE_VRM", "assets/avatar.vrm"))

DEMO = demo_arena()
SPAWN = (-5.0, -5.0)
START_TARGET = (5.0, 5.0)

state = {"ready": False, "error": None, "actor": None, "nav": None,
         "navigator": None, "target": START_TARGET, "last": None}


def spec():
    return {
        "half": DEMO.half, "ceiling": DEMO.ceiling,
        "boxes": [{"x": b.x, "z": b.z, "hx": b.hx, "hz": b.hz,
                   "h": b.h, "yaw": b.yaw} for b in DEMO.boxes],
        "spawn": list(SPAWN), "target": list(state["target"]),
    }


def world_frames(frames, origin):
    out = []
    for f in frames:
        g = dict(f)
        r = list(g["root"])
        r[0] += float(origin[0])
        r[2] += float(origin[1])
        g["root"] = r
        out.append(g)
    return out


def jpeg_data_url(bgr: np.ndarray, quality: int = 84):
    import cv2
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def boot():
    try:
        t0 = time.perf_counter()
        nav = NavEncoder(ROOT / "checkpoints" / "nav.pt")
        actor = ArdyV(steps=6, seed=0, cfg=(2.0, 2.0))
        actor.reset(np.asarray(SPAWN, np.float64))
        nav.predict(np.full((IMG_H, IMG_W, 3), 120, np.uint8))
        state["nav"] = nav
        state["actor"] = actor
        state["navigator"] = VisualNavigator(half=12.0, speed=0.70)
        state["navigator"].reset(np.asarray(START_TARGET, np.float64))
        state["ready"] = True
        print(f"[ardy-v] visual-nav ready in {time.perf_counter()-t0:.1f}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        state["error"] = repr(exc)
        print("[ardy-v] boot failed", repr(exc), flush=True)


app = FastAPI(title="ARDY-v")

nav_cmd_log = []  # per-window ARDY command dump (QC replay / ablation)


@app.get("/api/cmdlog")
def cmdlog(n: int = 200):
    return {"log": nav_cmd_log[-int(n):]}


@app.get("/api/health")
def health():
    return {"ready": bool(state["ready"]), "error": state["error"],
            "mode": "visual" if state["ready"] else None}


@app.get("/api/state")
def get_state():
    if not state["ready"]:
        raise HTTPException(503, state["error"] or "loading")
    actor = state["actor"]
    s = actor.state()
    origin = actor.origin
    return {
        "spec": spec(), "origin": [float(origin[0]), float(origin[1])],
        "spawn": list(SPAWN), "target": list(state["target"]),
        "mode": "visual",
        "root": [float(s["root"][0] + origin[0]), float(s["root"][1] + origin[1])],
        "yaw": s["yaw"],
    }


@app.post("/api/reset")
def reset():
    if not state["ready"]:
        raise HTTPException(503, state["error"] or "loading")
    actor = state["actor"]
    state["target"] = START_TARGET
    state["navigator"].reset(np.asarray(START_TARGET, np.float64))
    seed = actor.reset(np.asarray(SPAWN, np.float64))
    return {
        "seed": world_frames(seed, actor.origin),
        "origin": [float(actor.origin[0]), float(actor.origin[1])],
        "spawn": list(SPAWN), "target": list(state["target"]),
        "mode": "visual",
    }


def decode_rgb(data_url: str) -> np.ndarray:
    raw = data_url.split(",", 1)[-1]
    im = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
    im = im.resize((IMG_W, IMG_H), Image.BILINEAR)
    return np.asarray(im, np.uint8)


@app.post("/api/window")
def window(req: dict):
    if not state["ready"]:
        raise HTTPException(503, state["error"] or "loading")
    actor = state["actor"]
    nav_model = state["nav"]
    nav_planner = state["navigator"]
    t0 = time.perf_counter()
    rgb = decode_rgb(req.get("image", ""))
    target_world = np.asarray(req.get("target", state["target"]), np.float64)
    state["target"] = (float(target_world[0]), float(target_world[1]))
    target_local = target_world - actor.origin
    constraint = bool(req.get("constraint", True))

    before = actor.state()
    root_before = np.asarray(before["root"], np.float64)
    yaw_before = float(before["yaw"])
    depth, mask = nav_model.predict(rgb)
    root_world = root_before + actor.origin
    region = None
    route = None
    route_sdf = None
    route_min_sdf = None
    route_bad_i = None
    info = {"replanned": False, "invalid": False, "reason": None,
            "replans": 0, "remaining": None}
    if constraint:
        eye_spec = req.get("eye") or {}
        try:
            eye_root = np.asarray(eye_spec.get("root") or root_world, np.float64)
            eye_h = float(eye_spec.get("y", 1.55))
            eye_yaw = float(eye_spec.get("yaw", yaw_before))
            eye_pitch = float(eye_spec.get("pitch", -0.06))
            eye = (eye_root, eye_h, eye_yaw, eye_pitch)
        except (TypeError, ValueError):
            eye = None
        path, info = nav_planner.update(depth, mask, root_world, yaw_before,
                                        target_world, eye=eye)
        route = nav_planner.route
        region = nav_planner.map.region_at(root_world, half=6.0, clearance=0.25)
        remaining = info.get("remaining")
        print(f"[nav] root=({root_world[0]:.2f},{root_world[1]:.2f}) yaw={yaw_before:.2f} "
              f"repl={nav_planner.replans} reason={info.get('reason')} "
              f"streak={nav_planner.invalid_streak} brake={info.get('brake')} "
              f"remaining={remaining} path={'None' if path is None else len(path)}",
              flush=True)
        # debug: where does the current route fail its clearance check?
        route_min_sdf = None
        route_bad_i = None
        route_sdf = None
        if route is not None and len(route) > 1:
            rr = nav_planner.map.sample_sdf(np.asarray(route, np.float64))
            route_min_sdf = float(np.min(rr))
            route_bad_i = int(np.argmin(rr))
            route_sdf = [round(float(v), 2) for v in rr]
        stop_path = (float(np.linalg.norm(target_local - root_before)) < 0.45
                     or (remaining is not None and remaining < 0.45))
        if path is None:
            path = np.zeros((8, 3), np.float32)
            stop_path = True
    else:
        d = target_local - root_before
        dist0 = float(np.linalg.norm(d))
        step = min(0.28, dist0)
        u = d / max(dist0, 1e-6)
        bearing = float(np.arctan2(u[0] * np.cos(yaw_before) - u[1] * np.sin(yaw_before),
                                   u[0] * np.sin(yaw_before) + u[1] * np.cos(yaw_before)))
        path = np.zeros((8, 3), np.float32)
        for k in range(8):
            a = (k + 1) / 8
            path[k, 0] = (u[0] * np.sin(yaw_before) + u[1] * np.cos(yaw_before)) * step * a
            path[k, 1] = (u[0] * np.cos(yaw_before) - u[1] * np.sin(yaw_before)) * step * a
            path[k, 2] = float(np.clip(bearing, -0.7, 0.7)) * a
        stop_path = dist0 < 0.45
    # Viewer-side collision (a box overlapping the avatar): keep the map and
    # monitoring alive but hold the actor standing until the obstacle clears.
    if bool(req.get("halt")) and path is not None:
        path = np.zeros((8, 3), np.float32)
        stop_path = True
        info = dict(info, stop=True, halt=True)
    t1 = time.perf_counter()
    frames, action, gen_dt = actor.step(target_local, path_body=path, stop=stop_path)
    # per-window command dump for offline replay / QC ablation
    try:
        _p = np.asarray(path if path is not None else np.zeros((8, 3)), np.float32)
        nav_cmd_log.append({
            "stop": bool(stop_path), "halt": bool(req.get("halt")),
            "path": _p.tolist(),
            "fwd": [float(_p[:, 0].min()), float(_p[:, 0].max())],
            "lat": [float(_p[:, 1].min()), float(_p[:, 1].max())],
            "dyaw": [float(_p[:, 2].min()), float(_p[:, 2].max())],
        })
        if len(nav_cmd_log) > 4000:
            del nav_cmd_log[:2000]
    except Exception:
        pass
    frames_w = world_frames(frames, actor.origin)
    local_roots = np.asarray([[f["root"][0], f["root"][2]] for f in frames], np.float64)
    violation = False
    min_sdf = None
    if region is not None:
        world_roots = local_roots + actor.origin
        sdf = nav_planner.map.sample_sdf(world_roots)
        min_sdf = float(np.min(sdf))
        violation = bool(min_sdf < 0.0)
    s_after = actor.state()
    dist_after = float(np.linalg.norm(target_local - np.asarray(s_after["root"], np.float64)))
    world_root = np.asarray(s_after["root"], np.float64) + actor.origin
    collision = bool(nav_planner.map.sample_sdf(world_root[None, :])[0] < 0.0) if constraint else False
    stop = action == "stay"
    arrived = bool(stop and dist_after < 0.45 and not info.get("invalid"))

    depth_img = None
    mask_img = None
    mask_bin = None
    try:
        import cv2
        col = colorize_depth(depth)
        col = cv2.resize(col, (320, 180), interpolation=cv2.INTER_NEAREST)
        depth_img = jpeg_data_url(col)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(mask.astype(np.float32), (320, 180), interpolation=cv2.INTER_LINEAR)
        red = np.zeros_like(bgr)
        red[..., 2] = 255
        a = np.clip(m, 0.0, 1.0)[..., None] * 0.55
        gray = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        vis = (gray.astype(np.float32) * (1 - a) + red.astype(np.float32) * a).astype(np.uint8)
        mask_img = jpeg_data_url(vis, 88)
        mb = cv2.resize(mask.astype(np.float32), (320, 180), interpolation=cv2.INTER_LINEAR)
        bin_img = np.zeros((180, 320, 3), np.uint8)
        bin_img[mb > 0.5] = (255, 255, 255)
        mask_bin = jpeg_data_url(bin_img, 88)
    except Exception:
        mask_bin = None
    region_img = None
    if region is not None:
        try:
            rimg = region_png(region, root_world, target_world,
                              path_xz=(route if route else None))
            region_img = jpeg_data_url(rimg, 90)
        except Exception:
            pass

    total = time.perf_counter() - t0
    traj = [[float(f["root"][0]), float(f["root"][2])] for f in frames_w]
    route_out = ([[float(p[0]), float(p[1])] for p in route]
                 if route is not None and len(route) else None)
    occ_cells = (int(np.count_nonzero(nav_planner.map.occupied()))
                 if constraint else 0)
    map_updates = int(nav_planner.map.updates) if constraint else 0
    out = {
        "frames": frames_w,
        "origin": [float(actor.origin[0]), float(actor.origin[1])],
        "traj": traj, "route": route_out,
        "stop": bool(stop), "arrived": arrived, "action": action,
        "collision": bool(collision), "violation": bool(violation),
        "min_sdf": min_sdf,
        "depth": depth_img, "mask": mask_img, "mask_bin": mask_bin,
        "region": region_img,
        "mode": "visual" if constraint else "free",
        "dist": dist_after,
        "root": [float(s_after["root"][0] + actor.origin[0]),
                 float(s_after["root"][1] + actor.origin[1])],
        "yaw": s_after["yaw"],
        "guide": {"route_remaining": info.get("remaining"),
                  "replans": info.get("replans"),
                  "replanned": bool(info.get("replanned")),
                  "invalid": bool(info.get("invalid")),
                  "invalid_streak": int(nav_planner.invalid_streak) if constraint else 0,
                  "reason": info.get("reason"),
                  "route_min_sdf": route_min_sdf,
                  "route_sdf": route_sdf,
                  "route_bad_i": route_bad_i,
                  "occupied_cells": occ_cells, "map_updates": map_updates,
                  "mask_frac": float(mask.mean()) if constraint else 0.0},
        "latency": {"total": total, "infer": t1 - t0, "ardy": gen_dt},
    }
    state["last"] = out
    return JSONResponse(out)


@app.get("/api/mapprobe")
def mapprobe(x: float = 0.0, z: float = 0.0):
    if not state["ready"]:
        raise HTTPException(503, state["error"] or "loading")
    nav_planner = state["navigator"]
    occ, sdf = nav_planner.map.sdf(0.0)
    v = nav_planner.map.sample_sdf(np.array([[x, z]], np.float64))[0]
    ix = int(round((x - nav_planner.map.origin[0]) / nav_planner.map.res - 0.5))
    iz = int(round((z - nav_planner.map.origin[1]) / nav_planner.map.res - 0.5))
    n = nav_planner.map.n
    logv = float(nav_planner.map.log[iz, ix]) if (0 <= ix < n and 0 <= iz < n) else None
    return {"x": x, "z": z, "log": logv, "occupied": bool(occ[iz, ix]) if logv is not None else None,
            "sdf": float(v), "updates": int(nav_planner.map.updates)}


@app.get("/api/avatars/default/file")
def avatar_file():
    if not VRM.exists():
        raise HTTPException(404, str(VRM))
    return FileResponse(str(VRM), media_type="model/gltf-binary")


@app.get("/")
def index():
    return HTMLResponse((WEB / "index.html").read_text("utf-8"))


app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
threading.Thread(target=boot, daemon=True).start()
