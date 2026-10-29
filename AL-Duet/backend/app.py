"""AL-Duet platform backend.

Serves the web player and keeps in-memory live frame buffers for N agents:

* static: the web player (default page), three.js vendor + scene modules and
  VRM/scene assets live in assets/ and web/ inside this folder;
* ``/api/live/*``: agents push ARDY frames as they are generated
  (realtime mode); the page polls them at ~0.4 s;
* ``/api/episode``: the pre-generated episode (offline infer-then-play).

No simulation, no MuJoCo, no model code here -- this server is intentionally
thin; all intelligence lives in ``engine/``.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DATA = WEB / "data"
ASSETS = ROOT / "assets"

app = FastAPI(title="AL-Duet", version="1.0",
              description="多 AI 同场：状态机 + ARDY 持续生成 + 视觉回路")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost)(:\d+)?",
    allow_methods=["*"], allow_headers=["*"],
)

app.mount("/vendor", StaticFiles(directory=WEB / "vendor"), name="vendor")
app.mount("/scenes", StaticFiles(directory=WEB / "scenes"), name="scenes")
app.mount("/data", StaticFiles(directory=DATA, check_dir=False), name="data")


# ------------------------------------------------------------------ assets
def _read_json(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _scene_manifest(scene_id: str) -> dict:
    path = ASSETS / "scenes" / scene_id / "scene.json"
    if not path.is_file():
        raise HTTPException(404, f"unknown scene {scene_id}")
    return _read_json(path)


def _avatar_manifest(avatar_id: str) -> dict:
    path = ASSETS / "avatars" / f"{avatar_id}.json"
    if not path.is_file():
        raise HTTPException(404, f"unknown avatar {avatar_id}")
    data = _read_json(path)
    data.setdefault("id", avatar_id)
    return data


def _avatar_file(avatar_id: str) -> Path:
    man = _avatar_manifest(avatar_id)
    candidates = []
    if man.get("file"):
        candidates.append(Path(man["file"]))
    candidates.append(ASSETS / "avatars" / f"{avatar_id}.vrm")
    for p in candidates:
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    raise HTTPException(409, f"avatar {avatar_id} has no model file")


@app.get("/api/health")
def health() -> dict:
    ep = DATA / "episode.json"
    return {"ok": True, "version": "1.0", "episode": ep.is_file(),
            "live": {k: len(v["frames"]) for k, v in _LIVE.items()}}


@app.get("/api/episode")
def episode() -> dict:
    ep = DATA / "episode.json"
    if not ep.is_file():
        raise HTTPException(404, "episode.json not found")
    data = _read_json(ep)
    return {
        "schema": data.get("schema"),
        "created": data.get("created"),
        "fps": data.get("fps", 20.0),
        "sim_s": data.get("sim_s"),
        "joints": len(data.get("joint_names", [])),
        "actors": {k: {"label": v.get("label"), "frames": len(v.get("frames", []))}
                   for k, v in data.get("actors", {}).items()},
        "timeline": len(data.get("timeline", [])),
    }


@app.get("/api/scene/{scene_id}")
def scene(scene_id: str) -> dict:
    return _scene_manifest(scene_id)


@app.get("/api/avatars")
def avatars() -> list[dict]:
    out = []
    for p in sorted((ASSETS / "avatars").glob("*.json")):
        man = _read_json(p)
        out.append({"id": man.get("id", p.stem), "name": man.get("name", p.stem),
                    "kind": man.get("kind", "vrm"),
                    "height_m": man.get("height_m", 1.62),
                    "credit": man.get("credit", "")})
    return out


@app.get("/api/avatars/{avatar_id}")
def avatar(avatar_id: str) -> dict:
    man = _avatar_manifest(avatar_id)
    man["available"] = True
    return man


@app.get("/api/avatars/{avatar_id}/file")
def avatar_file(avatar_id: str) -> FileResponse:
    path = _avatar_file(avatar_id)
    return FileResponse(path, media_type="model/gltf-binary",
                        filename=path.name,
                        headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------- live buffers
_LOCK = threading.Lock()
_LIVE: dict[str, dict] = {}


class LiveStart(BaseModel):
    actor: str = Field(min_length=1, max_length=8)
    joint_names: list[str]
    fps: float = 20.0
    title: str = ""


class LiveAppend(BaseModel):
    actor: str = Field(min_length=1, max_length=8)
    frames: list[dict]


@app.post("/api/live/start")
def live_start(req: LiveStart) -> dict:
    with _LOCK:
        _LIVE[req.actor] = {
            "joint_names": req.joint_names, "fps": float(req.fps),
            "title": req.title, "frames": [], "t0": time.time(),
        }
    return {"ok": True, "actor": req.actor}


@app.post("/api/live/append")
def live_append(req: LiveAppend) -> dict:
    with _LOCK:
        buf = _LIVE.get(req.actor)
        if buf is None:
            raise HTTPException(404, f"actor {req.actor} has no live stream")
        buf["frames"].extend(req.frames)
        buf["last_append"] = time.time()
        if len(buf["frames"]) > 60000:      # ~50 min @20fps, then trim
            del buf["frames"][:len(buf["frames"]) - 20000]
    return {"ok": True, "total": len(buf["frames"])}


@app.get("/api/live/actors")
def live_actors() -> dict:
    now = time.time()
    with _LOCK:
        # dead engine runs must not haunt the page: drop streams whose last
        # append is older than the realtime tick by a wide margin
        stale = [k for k, v in _LIVE.items()
                 if now - v.get("last_append", v["t0"]) > 10.0]
        for k in stale:
            _LIVE.pop(k, None)
            _HUD.pop(k, None)
        return {"actors": [
            {"key": k, "title": v["title"], "fps": v["fps"],
             "joint_names": v["joint_names"], "frames": len(v["frames"])}
            for k, v in _LIVE.items()]}


@app.get("/api/live/frames")
def live_frames(actor: str, offset: int = 0) -> dict:
    with _LOCK:
        buf = _LIVE.get(actor)
        if buf is None:
            raise HTTPException(404, f"actor {actor} has no live stream")
        frames = buf["frames"][offset:]
        return {"actor": actor, "offset": offset,
                "total": len(buf["frames"]), "frames": frames}


@app.post("/api/live/stop")
def live_stop(body: dict) -> dict:
    actor = str(body.get("actor", ""))
    with _LOCK:
        _LIVE.pop(actor, None)
        # keep the last HUD state so the page shows the agents' final state
    return {"ok": True}


# --------------------------------------------------------------- live HUD
_HUD: dict[str, dict] = {}


@app.post("/api/live/hud")
def live_hud_push(body: dict) -> dict:
    hud = body.get("hud")
    if not isinstance(hud, dict):
        raise HTTPException(400, "body.hud must be an object keyed by actor")
    with _LOCK:
        _HUD.update(hud)
    return {"ok": True}


@app.get("/api/live/hud")
def live_hud_get() -> dict:
    with _LOCK:
        return {"hud": dict(_HUD)}


# Keep this last: the SPA + its assets.
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
