# -*- coding: utf-8 -*-
"""Agent: one AI body in the shared world.

Self-contained port of the audited clean path (no live_common dependency):

* ``world_frames`` maps an ARDY local chunk to world frames -- only the
  pelvis/Hips quaternion carries the fixed stage yaw; child joints are
  parent-relative;
* ``apply_controller`` drives the world root externally (raw ARDY supplies
  rotations only): speed comes from the raw chunk root speed, capped so the
  agent stops ~1.85 m from the nearest other agent;
* ``reset_stand`` re-seeds a drifted stand history from the model's own
  stand snapshot (never hand-keyed animation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

STOP_AT_M = 0.45       # hard stop: bodies may stand nearly touching
WALK_MIN_START = 0.50
STAND_HEIGHT = 0.947
TICK_S = 0.4           # one ARDY window = 8 frames @ 20 fps
TURN_RATE = 2.0        # rad/s max heading change (a 180 deg turn in ~1.6 s)
# walkable rect (kept in sync with engine.run.WORLD_BOUNDS; applies to ALL
# social walking -- the navigators handle avoidance, this is the fence)
WORLD_BOUNDS = (-4.5, 4.5, -3.5, 3.5)


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def qy(a: float) -> list[float]:
    return [0.0, math.sin(a / 2.0), 0.0, math.cos(a / 2.0)]


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def qslerp(a, b, t: float) -> list[float]:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    if dot < 0.0:
        b = [-float(v) for v in b]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        out = [float(x) + (float(y) - float(x)) * t for x, y in zip(a, b)]
        n = math.sqrt(sum(v * v for v in out)) or 1.0
        return [v / n for v in out]
    theta = math.acos(dot)
    s = math.sin(theta)
    wa = math.sin((1.0 - t) * theta) / s
    wb = math.sin(t * theta) / s
    return [float(a[i]) * wa + float(b[i]) * wb for i in range(4)]


def pelvis_yaw(joints_row, joint_names) -> float:
    ri = joint_names.index("RightUpLeg")
    li = joint_names.index("LeftUpLeg")
    d = np.asarray(joints_row, np.float64)[ri] - np.asarray(joints_row, np.float64)[li]
    return float(np.arctan2(d[2], -d[0]))


def chunk_frames(chunk, joint_names, t0: float = 0.0) -> list[dict]:
    """ARDY MotionChunk -> frame rows (rot 27x4, root xyz, pose x/z/yaw)."""
    rot = np.asarray(chunk.rotations, np.float32)
    roots = np.asarray(chunk.roots, np.float32)
    joints = np.asarray(chunk.joints, np.float32)
    fps = float(chunk.fps)
    rows = []
    for i in range(rot.shape[0]):
        if i + 1 < roots.shape[0]:
            step = float(np.linalg.norm((roots[i + 1] - roots[i])[[0, 2]]) * fps)
        else:
            step = (float(np.linalg.norm((roots[i] - roots[i - 1])[[0, 2]]) * fps)
                    if i else 0.0)
        rows.append({
            "rot": np.round(rot[i], 6).tolist(),
            "root": np.round(roots[i], 5).tolist(),
            "pose": [round(float(roots[i][0]), 5), round(float(roots[i][2]), 5),
                     round(pelvis_yaw(joints[i], joint_names), 5)],
            "speed": round(step, 4),
            "t": round(t0 + i / fps, 5),
        })
    return rows


@dataclass
class Agent:
    key: str                    # "A", "B", "C", ...
    label: str
    origin: tuple[float, float]  # world x, z of the generator origin
    yaw0: float                  # world yaw of the generator's +z forward
    gen: object
    fsm: object
    names: list[str]
    phase: float = 0.0

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    heading: float = 0.0
    face: float = 0.0        # world yaw the body/motion currently faces
    face_prev: float = 0.0   # yaw at the previous window end (in-window lerp)
    last_rot: list | None = None   # last displayed 27x4 quats (crossfade src)
    speed: float = 0.0
    local_root: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    stand_snap: dict | None = None       # {"snap": {...}, "stand_ref": chunk}
    last_pose_reset_t: float = -99.0
    pulse_acc: float = 0.0               # pulsed-approach accumulator
    walk_streak: int = 0                 # consecutive walk windows

    # FSM bookkeeping (sparse updates)
    last_state: str = "solo_idle"
    last_emotion: str = "calm"
    state_entry: float = 0.0
    last_dec: object = None
    last_step: object = None
    last_decision_t: float = -99.0
    next_decision_t: float = 0.0
    last_d_vis: float | None = None
    last_other_speaking: bool = False
    last_other_visible: bool = False
    decision_count: int = 0
    wave_windows: int = 0

    # speech / floor
    speech: str = ""
    speech_turns: int = 0
    speaking_until: float = 0.0
    addressed_until: float = 0.0
    events: dict[str, list[float]] = field(default_factory=dict)

    # perception (filled by the vision module)
    vision: dict = field(default_factory=dict)
    dist_ema: float | None = None
    last_seen_t: float = -99.0

    all_frames: list = field(default_factory=list)

    def add_event(self, name: str, t: float) -> None:
        self.events.setdefault(name, []).append(float(t))

    def recent(self, name: str, t: float, window: float) -> bool:
        ts = self.events.get(name) or []
        return bool(ts) and t - ts[-1] <= window

    def to_local(self, p) -> tuple[float, float]:
        dx = float(p[0]) - self.origin[0]
        dz = float(p[2]) - self.origin[1]
        c, s = math.cos(self.yaw0), math.sin(self.yaw0)
        return c * dx - s * dz, s * dx + c * dz

    def to_world(self, lx: float, lz: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw0), math.sin(self.yaw0)
        return (self.origin[0] + c * lx + s * lz,
                self.origin[1] - s * lx + c * lz)

    # ------------------------------------------------------------------ pose
    def steer(self, target: float, dt: float = TICK_S) -> float:
        """Turn `face` toward `target` at TURN_RATE; returns the turn applied.
        Social agents never steer (face == yaw0); explorers use this to
        navigate.  The turn is spread across the next window's 8 frames
        (``face_prev`` -> ``face``), so heading changes never snap."""
        delta = wrap(target - self.face)
        step = max(-TURN_RATE * dt, min(TURN_RATE * dt, delta))
        self.face = wrap(self.face + step)
        return step

    def _yaw_at(self, i: int, n: int) -> float:
        """Interpolated world yaw for frame i of n: morphs face_prev -> face
        smoothly across the window instead of snapping between windows."""
        return wrap(self.face_prev + wrap(self.face - self.face_prev)
                    * ((i + 1) / max(1, n)))

    def world_frames(self, chunk, t0: float,
                     blend_from: list | None = None) -> list[dict]:
        """Only the Hips quaternion carries the world yaw (interpolated).
        ``blend_from`` crossfades the 27 quats from the previous displayed
        pose over this window -- used after a stand-history reset, whose
        otherwise-instant pose change read as a teleport every few seconds."""
        frames: list[dict] = []
        hip = self.names.index("Hips")
        rows = chunk_frames(chunk, self.names, t0=t0)
        n = len(rows)
        for i, f in enumerate(rows):
            yaw_i = self._yaw_at(i, n)
            lx, ly, lz = f["root"]
            wx, wz = self.to_world(lx, lz)
            rot = [list(r) for r in f["rot"]]
            q = qy(yaw_i)
            rot[hip] = qmul(q, rot[hip])
            if blend_from is not None:
                alpha = (i + 1) / n
                for j in range(len(rot)):
                    rot[j] = qslerp(blend_from[j], rot[j], alpha)
            yaw_world = wrap(float(f["pose"][2]) + yaw_i)
            frames.append({
                "rot": rot,
                "root": [round(wx, 5), round(float(ly), 5), round(wz, 5)],
                "pose": [round(wx, 5), round(wz, 5), round(yaw_world, 5)],
                "speed": f["speed"],
                "t": f["t"],
            })
        return frames

    # ------------------------------------------------------------ controller
    RAW_WALK_V = 0.55     # ARDY walk embedding cadence, m/s (median measured)

    def plan_motion(self, intent: str, d_vis: float | None,
                    away: bool = False, stop_at: float | None = None) -> bool:
        """Decide whether THIS window walks.  Two mechanisms keep the raw
        gait honest and collision-free:

        * the world root follows the raw chunk's own per-frame displacement
          (``controller_motion``), so the feet can never skate -- including
          the slow gait-initiation windows right after a stand;
        * near a partner we cap consecutive walk windows and force a stand
          window, which re-initialises the gait: the next walk window only
          creeps its initiation distance (~0.1-0.3 m).  That gives
          sub-window granularity exactly where collisions could happen.

        ``away`` (retreat intents) bypasses the stop zone: backing off must
        stay possible when someone is already inside the personal space.
        ``stop_at`` overrides the stop distance -- the social approach stops
        at conversational range (1.0 m) so the greet transition fires well
        before the 0.6 m personal-space safety line is ever reached; the old
        0.55 m stop put the creep INSIDE the greet band's shadow and caused
        endless approach<->too_close limit cycles."""
        stop_dist = (stop_at if stop_at is not None else STOP_AT_M) + 0.10
        moving = intent in ("walk", "walk_fwd") and (
            away or d_vis is None or d_vis > WALK_MIN_START)
        if not moving:
            self.walk_streak = 0
            return False
        if not away and d_vis is not None and d_vis <= stop_dist:
            self.walk_streak = 0
            return False
        if away:
            # retreat: walk CONTINUOUSLY (no gait breaks) -- the gait warms
            # up and the agent actually leaves the personal space within the
            # state's dwell; the turn-around already costs ~1.6 s
            self.walk_streak += 1
            return True
        # near a partner break the gait often: the first walk window after a
        # stand only creeps its initiation distance (~0.1-0.3 m), which gives
        # the FSM sub-window granularity exactly where it matters.  The creep
        # zone starts at 2.0 m so the approach LANDS inside the greet band
        # (0.7-1.1 m) instead of overshooting through it at warm-gait speed.
        if d_vis is not None and d_vis < 2.0 and self.walk_streak >= 1:
            self.walk_streak = 0
            return False
        if d_vis is not None and d_vis < 2.5 and self.walk_streak >= 2:
            self.walk_streak = 0
            return False
        self.walk_streak += 1
        return True

    def controller_motion(self, chunk, walk: bool) -> np.ndarray:
        """Per-frame cumulative world offsets.

        The raw chunk's own root displacement is rotated so that its net
        travel direction maps onto the per-frame interpolated yaw: without
        this, a raw walk that travels local +z renders as a moonwalk for any
        agent whose stage yaw is not 0 -- feet animate forward while the
        root slides backward.  Per-frame following makes skating impossible
        by construction."""
        if not walk:
            return np.zeros((1, 2), np.float32)
        roots = np.asarray(chunk.roots, np.float32)
        rel = roots[:, [0, 2]] - roots[0][[0, 2]]
        net = rel[-1] - rel[0]
        if float(np.linalg.norm(net)) > 1e-4:
            travel = math.atan2(float(net[0]), float(net[1]))
        else:
            travel = 0.0
        n = len(rel)
        out = np.empty((n, 2), np.float64)
        for i in range(n):
            psi = self._yaw_at(i, n) - travel
            c, s = math.cos(psi), math.sin(psi)
            out[i, 0] = c * rel[i, 0] + s * rel[i, 1]
            out[i, 1] = -s * rel[i, 0] + c * rel[i, 1]
        return out

    def apply_motion(self, frames: list[dict], offsets: np.ndarray,
                     walk: bool) -> None:
        x0, z0 = float(self.pos[0]), float(self.pos[2])
        n = len(frames)
        bx0, bx1, bz0, bz1 = WORLD_BOUNDS
        m = 0.3
        for i, f in enumerate(frames):
            k = min(i, len(offsets) - 1)
            # social walks have no navigator: hard-clamp every body inside
            # the walkable rect so no state can wander off the world
            wx = max(bx0 - m, min(bx1 + m, x0 + float(offsets[k][0])))
            wz = max(bz0 - m, min(bz1 + m, z0 + float(offsets[k][1])))
            ry = float(f["root"][1])
            ry = (max(0.86, min(1.00, ry)) if walk else STAND_HEIGHT)
            yaw_i = self._yaw_at(i, n)
            f["root"] = [round(wx, 5), round(ry, 5), round(wz, 5)]
            f["pose"] = [round(wx, 5), round(wz, 5), round(yaw_i, 5)]

    def apply_controller(self, frames: list[dict], v: float) -> None:
        x0, z0 = float(self.pos[0]), float(self.pos[2])
        dx, dz = math.sin(self.face), math.cos(self.face)
        moving = v > 1e-4
        for i, f in enumerate(frames):
            wx = x0 + dx * v * (i + 1) / 20.0
            wz = z0 + dz * v * (i + 1) / 20.0
            ry = float(f["root"][1])
            ry = (max(0.86, min(1.00, ry)) if moving else STAND_HEIGHT)
            f["root"] = [round(wx, 5), round(ry, 5), round(wz, 5)]
            f["pose"] = [round(wx, 5), round(wz, 5), round(self.face, 5)]

    def commit(self, frames: list[dict], t_sim: float) -> None:
        if not frames:
            return
        last = frames[-1]
        prev = self.pos.copy()
        self.pos = np.asarray(last["root"], np.float32)
        self.heading = float(last["pose"][2])
        self.face = self.heading
        self.face_prev = self.heading
        self.last_rot = [list(r) for r in last["rot"]]
        self.speed = float(np.linalg.norm((self.pos - prev)[[0, 2]])) / TICK_S
        self.local_root = np.asarray(
            self.gen.joints[-1, self.gen.layout.root_idx].detach().cpu().numpy(),
            np.float32)
        self.all_frames.extend(frames)
