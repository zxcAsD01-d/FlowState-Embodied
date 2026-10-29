# -*- coding: utf-8 -*-
"""Vision loop for the engine.

Two perception backends feed the SAME FSM observation:

* ``geometric`` -- distance / bearing / visibility from the world state.
  No renderer, no GPU vision: this is the fast default for offline
  generation and is labelled as world-truth in the episode meta.
* ``render``    -- the browser page captures each agent's egocentric view
  (``window.duet.capture()``: own body hidden, 3x3 depth rays against the
  other bodies), then YOLOv8-m detects the nearest person and the box
  height + torso depth give metric distance.  This is the full visual
  closure used for live runs.

Both return, per agent: {visible, conf, distance, bearing_deg, depth_m, box}.
"""
from __future__ import annotations

import base64
import io
import math

FOCAL_PX = (360.0 / 2.0) / math.tan(math.radians(72.0) / 2.0)
VISION_BODY_M = 1.75
VISION_DIST_K = 1.12
BODY_RADIUS_M = 0.32
MIN_BOX_PX = 36.0
VISIBLE_RANGE_M = 9.0


def nearest_other(agent, agents):
    best, bd = None, None
    for other in agents:
        if other is agent:
            continue
        dx = float(other.pos[0] - agent.pos[0])
        dz = float(other.pos[2] - agent.pos[2])
        d = math.hypot(dx, dz)
        if bd is None or d < bd:
            best, bd = other, d
    return best, (bd if bd is not None else 99.0)


def geometric(agent, agents, t_sim: float) -> dict:
    """World-truth perception of the nearest other agent."""
    other, d = nearest_other(agent, agents)
    if other is None:
        return {
            "visible": False, "conf": 0.0, "distance": 99.0,
            "bearing_deg": 0.0, "depth_m": 99.0, "box": None,
            "facing_me": 0.0, "mutual": 0.0, "other": None, "mode": "geo",
        }
    dx = float(other.pos[0] - agent.pos[0])
    dz = float(other.pos[2] - agent.pos[2])
    to_other = math.atan2(dx, dz)
    bearing = math.degrees(agent.heading - to_other)
    facing_me = math.cos(math.atan2(-dx, -dz) - other.heading)
    mutual = max(0.0, math.cos(agent.heading - to_other)) * max(
        0.0, math.cos(math.atan2(-dx, -dz) - other.heading))
    return {
        "visible": bool(d < VISIBLE_RANGE_M),
        "conf": 0.9,
        "distance": float(d),
        "bearing_deg": float(bearing),
        "depth_m": float(d),
        "box": None,
        "facing_me": float(facing_me),
        "mutual": float(mutual),
        "other": other,
        "mode": "geo",
    }


# --------------------------------------------------------------------- YOLO
def load_yolo(root):
    from ultralytics import YOLO
    weights = root / "yolov8m.pt"
    return YOLO(str(weights) if weights.exists() else "yolov8m.pt")


def detect(model, image_b64: str, width: int = 640, height: int = 360) -> dict:
    from PIL import Image
    raw = base64.b64decode(image_b64.split(",", 1)[-1])
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    res = model.predict(img, classes=[0], conf=0.15, imgsz=640, verbose=False,
                        max_det=5, iou=0.5)[0]
    best = None
    for b in res.boxes:
        conf = float(b.conf[0])
        x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        if h < MIN_BOX_PX or h / w < 1.15:
            continue
        score = h * (0.5 + conf)
        if best is None or score > best[0]:
            best = (score, conf, (x0, y0, x1, y1))
    if best is None:
        return {"visible": False, "conf": 0.0, "box": None,
                "distance": None, "bearing_deg": 0.0, "img": img}
    _, conf, box = best
    x0, y0, x1, y1 = box
    h = max(1.0, y1 - y0)
    cx = (x0 + x1) / 2.0
    dist = VISION_DIST_K * FOCAL_PX * VISION_BODY_M / h
    bearing = math.degrees(math.atan2(cx - width / 2.0, FOCAL_PX))
    return {"visible": True, "conf": conf, "box": box,
            "distance": float(max(0.35, min(9.0, dist))),
            "bearing_deg": float(bearing), "img": img}


def torso_depth(cap: dict, box, width: int = 640, height: int = 360):
    samples = cap.get("depth_samples") or []
    if not samples:
        return None
    picked = samples
    if box is not None:
        x0, y0, x1, y1 = box
        nx0 = x0 / width * 2.0 - 1.0
        nx1 = x1 / width * 2.0 - 1.0
        ny0 = 1.0 - y1 / height * 2.0
        ny1 = 1.0 - y0 / height * 2.0
        inside = [s for s in samples
                  if nx0 - 0.04 <= s["x"] <= nx1 + 0.04
                  and ny0 - 0.04 <= s["y"] <= ny1 + 0.04]
        if inside:
            picked = inside
    ds = sorted(float(s["d"]) for s in picked)
    return ds[len(ds) // 2] + BODY_RADIUS_M


class RenderVision:
    """Drives the platform page for egocentric captures + YOLO detection.

    The page must expose ``window.duet.capture()`` returning
    ``{actors: {key: {image, depth_samples, ...}}, ...}`` for every live
    actor currently rendered.
    """

    def __init__(self, page, yolo, width: int = 640):
        self.page = page
        self.yolo = yolo
        self.width = width
        self.height = round(width / 1.7778)

    def capture_all(self) -> dict:
        return self.page.evaluate(
            "(w) => window.duet.capture(w)", {"w": self.width})

    def perceive(self, agent, agents, t_sim: float) -> dict:
        caps = self.capture_all()
        cap = caps.get("actors", {}).get(agent.key)
        if not cap:
            return geometric(agent, agents, t_sim)
        vis = detect(self.yolo, cap["image"], self.width, self.height)
        depth = torso_depth(cap, vis.get("box"), self.width, self.height)
        if depth is not None:
            vis["depth_m"] = float(depth)
            if not vis["visible"]:
                vis["visible"] = True
                vis["conf"] = max(float(vis.get("conf", 0.0)), 0.25)
            vis["distance"] = float(depth)
        other, _ = nearest_other(agent, agents)
        dx = float(other.pos[0] - agent.pos[0])
        dz = float(other.pos[2] - agent.pos[2])
        to_other = math.atan2(dx, dz)
        vis["bearing_deg"] = float(math.degrees(agent.heading - to_other))
        vis["facing_me"] = float(math.cos(
            math.atan2(-dx, -dz) - other.heading))
        vis["mutual"] = float(max(0.0, math.cos(agent.heading - to_other))
                              * vis["facing_me"])
        vis["other"] = other
        vis["mode"] = "rgb" if vis.get("box") else "depth"
        vis["img"] = vis.get("img")
        return vis
