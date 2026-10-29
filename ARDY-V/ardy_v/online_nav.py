"""Plan once, monitor continuously, replan only when the plan becomes infeasible.

The only inputs are the current pose and the goal.  The occupancy map is built
online from DINOv2 depth + collision mask; the route is re-planned only when a
new/changed obstacle invalidates the remaining route (confirmed over two
windows).
"""
from __future__ import annotations

import math

import numpy as np

from .occupancy_map import WorldMap
from .route_tracker import RouteTracker


def _polyline_len(route):
    route = np.asarray(route, np.float64)
    if len(route) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())


class VisualNavigator:
    def __init__(self, half: float = 12.0, speed: float = 0.62,
                 plan_clearance: float = 0.55, monitor_clearance: float = 0.45,
                 improve_every: int = 8, improve_frac: float = 0.15,
                 improve_min: float = 0.6, brake_dist: float = 1.6):
        self.map = WorldMap(half=half, res=0.1)
        self.speed = float(speed)
        self.plan_clearance = float(plan_clearance)
        self.monitor_clearance = float(monitor_clearance)
        self.target = None
        self.route = None
        self.tracker = None
        self.invalid_streak = 0
        self.replans = 0
        self.last_reason = None
        self.last_valid = True
        self.just_replanned = False
        self.improve_every = int(improve_every)
        self.improve_frac = float(improve_frac)
        self.improve_min = float(improve_min)
        self.brake_dist = float(brake_dist)
        self.window_i = 0
        self.last_better = None

    def reset(self, target):
        self.map.clear()
        self.target = np.asarray(target, np.float64)
        self.route = None
        self.tracker = None
        self.invalid_streak = 0
        self.replans = 0
        self.last_reason = None
        self.last_valid = True
        self.just_replanned = False
        self.window_i = 0
        self.last_better = None

    def _plan(self, root, reason):
        route = self.map.plan_global(root, self.target,
                                     clearance=self.plan_clearance)
        self.route = route if route else None
        self.just_replanned = True
        self.last_reason = reason if self.route else "unreachable"
        if self.route:
            self.tracker = RouteTracker(self.route, self.target)
            self.replans += 1
        else:
            self.tracker = None
        self.invalid_streak = 0

    def _brake_path(self, root, yaw, speed, frames=8, fps=20.0):
        """Short decelerating path along the route tangent.

        The safety stop used to cut the root boundary to zero in one window,
        which ARDY renders as a hard freeze.  A linear velocity ramp over
        ~0.4 s (total travel ~0.12 m) keeps the stand transition coherent.
        """
        root = np.asarray(root, np.float64)
        v0 = float(np.clip(speed, 0.3, 0.7))
        body = np.zeros((frames, 3), np.float32)
        s = 0.0
        for k in range(frames):
            vk = v0 * (1.0 - (k + 1) / frames)   # velocity ramp -> 0
            s += vk / fps
            body[k, 0] = s                       # straight forward, no turn
        return body

    def _maybe_improve(self, root, remaining):
        """Re-optimize when a clearly better route exists (not a failure)."""
        if self.window_i < self.improve_every or self.just_replanned:
            return
        if remaining is None or remaining < 1.2:
            return
        alt = self.map.plan_global(root, self.target,
                                   clearance=self.plan_clearance)
        if alt is None:
            return
        alt_len = _polyline_len(alt)
        gain = remaining - alt_len
        if gain < max(self.improve_min, self.improve_frac * remaining):
            return
        if not self.map.route_valid(alt, root, clearance=self.monitor_clearance):
            return
        self.route = alt
        self.tracker = RouteTracker(alt, self.target)
        self.replans += 1
        self.just_replanned = True
        self.last_reason = "better"
        self.last_better = float(gain)

    def _violation_dist(self, root, clearance):
        """Arclength from the root to the first point that breaks clearance.

        None when the remaining route is clean; 0.0 when the actor itself is
        too far off the route to reason about distances (treat as imminent).
        """
        if self.tracker is None:
            return 0.0
        s0, d = self.tracker.project(root)
        if d > 0.75:
            return 0.0
        s = s0
        while s <= self.tracker.length:
            p = self.tracker.point_at(s)
            if self.map.sample_sdf(np.asarray([[p[0], p[1]]]))[0] \
                    <= clearance + 0.02:
                return float(s - s0)
            s += 0.15
        return None

    def update(self, depth, mask, root_world, yaw, target_world,
               eye=None):
        target = np.asarray(target_world, np.float64)
        target_changed = self.target is None or np.linalg.norm(target - self.target) > 0.3
        if target_changed:
            self.target = target
            self.route = None
            self.tracker = None
            self.invalid_streak = 0
        self.just_replanned = False
        self.window_i += 1
        # The map is built from the first-person view; when the eye is bound
        # to the posed head, its position/height/orientation differ slightly
        # from the pelvis frame and are passed through here.
        if eye is not None:
            eye_root, eye_h, eye_yaw, eye_pitch = eye
            self.map.update(depth, eye_root, eye_yaw, pitch=eye_pitch,
                            eye_h=eye_h, mask=mask)
        else:
            self.map.update(depth, root_world, yaw, mask=mask)
        root = np.asarray(root_world, np.float64)
        braking = False
        if self.route is None:
            self._plan(root, "target" if target_changed else "initial")
        else:
            valid = self.map.route_valid(self.route, root,
                                         clearance=self.monitor_clearance)
            self.last_valid = valid
            if not valid:
                self.invalid_streak += 1
                vd = self._violation_dist(root, self.monitor_clearance)
                if self.invalid_streak >= 2:
                    self._plan(root, "blocked")
                    self.last_valid = self.route is not None
                elif vd is not None and vd < self.brake_dist:
                    # Obstacle confirmed close ahead: brake smoothly now
                    # (linear ramp, ~0.4 s) instead of freezing, and keep
                    # braking while the second confirmation window runs.
                    braking = True
                # else: the violation is far ahead — keep walking at route
                # speed while the next window confirms; braking on every
                # transient mask flicker was the main stop-and-go source.
            else:
                self.invalid_streak = 0
                self._maybe_improve(root, self.tracker_window_remaining(root))
        if self.tracker is None:
            return None, {"replanned": bool(self.just_replanned),
                          "invalid": not self.last_valid,
                          "brake": braking,
                          "reason": self.last_reason, "replans": self.replans,
                          "remaining": None}
        if braking:
            return self._brake_path(root, yaw, self.speed), {
                "replanned": False, "invalid": True, "brake": True,
                "reason": self.last_reason, "replans": self.replans,
                "remaining": None}
        path, remaining = self.tracker.window(root, yaw, speed=self.speed)
        return path, {"replanned": bool(self.just_replanned),
                      "invalid": not self.last_valid,
                      "brake": False,
                      "reason": self.last_reason, "replans": self.replans,
                      "remaining": remaining}

    def tracker_window_remaining(self, root):
        if self.tracker is None:
            return None
        s, _ = self.tracker.project(root)
        return float(self.tracker.length - s)
