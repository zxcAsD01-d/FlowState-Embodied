"""Continuous route tracking for the live ARDY demo.

Planning a fresh 8-frame path every 0.4 s was the main source of window-boundary
velocity jumps.  Here a longer world route (A* on the static scene) is followed
by arclength; each window takes the next 0.28 m at 0.70 m/s.  The route is
replanned only when the target changes, the actor leaves it, or it is consumed.
"""
from __future__ import annotations

import math

import numpy as np


def smooth_route(route, arena, clearance: float = 0.32, iterations: int = 2):
    """Chaikin corner cutting with a hard free-space check.

    A* returns a polyline with sharp corners; those corners are root-velocity
    jumps for ARDY.  Smooth only if every new point stays outside the inflated
    obstacle footprint.
    """
    pts = [np.asarray(p, np.float64) for p in route]
    if len(pts) < 3:
        return np.asarray(route, np.float64)
    cur = pts
    for _ in range(iterations):
        nxt = [cur[0]]
        for a, b in zip(cur[:-1], cur[1:]):
            q = 0.75 * a + 0.25 * b
            r = 0.25 * a + 0.75 * b
            nxt.extend([q, r])
        nxt.append(cur[-1])
        if all(arena.point_free(p, clearance) for p in nxt):
            cur = nxt
        else:
            break
    return np.asarray(cur, np.float64)


def _cumlen(route):
    route = np.asarray(route, np.float64)
    if len(route) < 2:
        return np.zeros(len(route)), 0.0
    seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return cum, float(cum[-1])


class RouteTracker:
    def __init__(self, route, target):
        self.route = np.asarray(route, np.float64)
        self.target = np.asarray(target, np.float64)
        self.cum, self.length = _cumlen(self.route)
        self.progress = 0.0

    def project(self, root):
        """Nearest arclength on the route and distance from it."""
        p = np.asarray(root, np.float64)
        best_s, best_d = 0.0, float("inf")
        for k in range(len(self.route) - 1):
            a, b = self.route[k], self.route[k + 1]
            ab = b - a
            L2 = float(ab @ ab)
            t = 0.0 if L2 < 1e-12 else float(np.clip((p - a) @ ab / L2, 0, 1))
            q = a + t * ab
            d = float(np.linalg.norm(p - q))
            if d < best_d:
                best_d = d
                best_s = self.cum[k] + t * math.sqrt(L2)
        return best_s, best_d

    def point_at(self, s):
        s = float(np.clip(s, 0.0, self.length))
        k = int(np.searchsorted(self.cum, s, side="right") - 1)
        k = max(0, min(k, len(self.route) - 2))
        a, b = self.route[k], self.route[k + 1]
        d = self.cum[k + 1] - self.cum[k]
        t = 0.0 if d < 1e-12 else (s - self.cum[k]) / d
        return a + t * (b - a)

    def needs_replan(self, root, target, speed=0.70, horizon=8):
        target = np.asarray(target, np.float64)
        if np.linalg.norm(target - self.target) > 0.3:
            return True
        s, d = self.project(root)
        if d > 0.55:
            return True
        remaining = self.length - s
        if remaining < speed * horizon / 20.0 * 0.8:
            return True
        return False

    def heading_at(self, s):
        a = self.point_at(max(0.0, s - 0.25))
        b = self.point_at(min(self.length, s + 0.35))
        return math.atan2(b[0] - a[0], b[1] - a[1])

    def window(self, root, yaw, speed=0.70, horizon=8, fps=20.0):
        """8x3 body-frame path from the current root along the route."""
        root = np.asarray(root, np.float64)
        s0, d = self.project(root)
        # slow down through corners: the route tangent changes there, and a
        # constant 0.70 m/s forces ARDY into a fast turn.
        h0 = self.heading_at(s0)
        h1 = self.heading_at(s0 + speed * horizon / fps)
        turn = abs((h1 - h0 + math.pi) % (2 * math.pi) - math.pi)
        v = float(np.clip(speed * (1.0 - 0.45 * min(1.0, turn / 0.8)), 0.42, speed))
        pts = [root]
        for k in range(horizon):
            s = s0 + v * (k + 1) / fps
            pts.append(self.point_at(s))
        # Heading shaping: per-frame route tangent headings, with the gap to
        # the actor's current yaw closed at a bounded rate.  A replan that
        # flips the route direction used to command a +-0.7 rad heading jump
        # every window with up to 0.26 m of sideways drift (replay-verified to
        # fold the walk into a crouch: head drops from y0.67 to y0.30 in ~10
        # windows).  0.35 rad/window still turns 90 deg in under 2 s, and the
        # positions below are *integrated from the shaped headings*, so the
        # root-position and heading constraints stay self-consistent.
        hd = [self.heading_at(s0 + v * (k + 1) / fps) for k in range(horizon)]
        gap = (hd[-1] - yaw + math.pi) % (2 * math.pi) - math.pi
        shift = float(np.clip(gap, -0.35, 0.35)) - gap
        # Per-frame heading offsets stay inside the +-0.7 rad the generator
        # was trained on (the pre-shaping tracker clipped them too); the gap
        # closure above bounds the window-to-window change, the clip bounds
        # the in-window absolute offset on sharp corners.
        hd = [yaw + float(np.clip(
            (h + shift - yaw + math.pi) % (2 * math.pi) - math.pi, -0.7, 0.7))
            for h in hd]
        pts = [np.asarray(root, np.float64)]
        step = v / fps
        for k in range(horizon):
            pts.append(pts[-1] + step * np.array(
                [math.sin(hd[k]), math.cos(hd[k])]))
        pts = np.asarray(pts)
        body = np.zeros((horizon, 3), np.float32)
        fwd_acc = 0.0
        for k in range(horizon):
            p = pts[k + 1]
            dx, dz = p[0] - root[0], p[1] - root[1]
            f = max(dx * math.sin(yaw) + dz * math.cos(yaw), fwd_acc + 0.004)
            fwd_acc = f
            body[k, 0] = f
            body[k, 1] = dx * math.cos(yaw) - dz * math.sin(yaw)
            body[k, 2] = (hd[k] - yaw + math.pi) % (2 * math.pi) - math.pi
        return body, float(self.length - s0)
