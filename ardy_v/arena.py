"""Synthetic-but-real arena geometry for ARDY-v training and evaluation.

The arena is a closed room (floor, ceiling, four walls, axis-aligned boxes).
Everything here is analytic: ground-truth absolute depth comes from ray casting,
and the expert root path comes from an inflated 2-D occupancy A*.  No learned
labels and no hand-keyed motion are used.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

FPS = 20.0
HORIZON = 8
DEFAULT_SPEED = 1.15  # m/s, ARDY walking speed measured in R07/R10 runs


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_dir(yaw: float) -> np.ndarray:
    """World XZ forward for a pelvis yaw (matches ardy_runtime)."""
    return np.array([math.sin(yaw), math.cos(yaw)], np.float64)


def right_dir(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), -math.sin(yaw)], np.float64)


@dataclass
class Box:
    x: float
    z: float
    hx: float
    hz: float
    h: float
    yaw: float = 0.0

    def corners(self) -> np.ndarray:
        """(4,2) world XZ corners."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        local = np.array([[-self.hx, -self.hz], [self.hx, -self.hz],
                          [self.hx, self.hz], [-self.hx, self.hz]])
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + np.array([self.x, self.z])


@dataclass
class Arena:
    half: float = 8.0
    ceiling: float = 4.0
    boxes: list[Box] = field(default_factory=list)

    # ------------------------------------------------------------- occupancy
    def occupancy(self, res: float = 0.1, clearance: float = 0.35):
        n = int(round(2 * self.half / res))
        xs = -self.half + (np.arange(n) + 0.5) * res
        zs = xs.copy()
        X, Z = np.meshgrid(xs, zs, indexing="xy")
        free = (np.abs(X) < self.half - clearance) & (np.abs(Z) < self.half - clearance)
        for b in self.boxes:
            if abs(b.yaw) < 1e-6:
                free &= ~((np.abs(X - b.x) < b.hx + clearance)
                          & (np.abs(Z - b.z) < b.hz + clearance))
            else:
                c, s = math.cos(b.yaw), math.sin(b.yaw)
                dx, dz = X - b.x, Z - b.z
                lx = c * dx + s * dz
                lz = -s * dx + c * dz
                free &= ~((np.abs(lx) < b.hx + clearance)
                          & (np.abs(lz) < b.hz + clearance))
        return free, xs, zs

    def point_free(self, p: Iterable[float], clearance: float = 0.2) -> bool:
        x, z = float(p[0]), float(p[1])
        if abs(x) >= self.half - clearance or abs(z) >= self.half - clearance:
            return False
        for b in self.boxes:
            c, s = math.cos(b.yaw), math.sin(b.yaw)
            dx, dz = x - b.x, z - b.z
            lx = c * dx + s * dz
            lz = -s * dx + c * dz
            if abs(lx) < b.hx + clearance and abs(lz) < b.hz + clearance:
                return False
        return True

    # ------------------------------------------------------------------ A*
    def plan(self, start, goal, res: float = 0.1, clearance: float = 0.35,
             smooth: bool = True):
        """8-connected A* over the inflated grid; returns [(x,z), ...] or []."""
        free, xs, zs = self.occupancy(res, clearance)
        n = len(xs)

        def to_idx(p):
            i = int(round((p[0] + self.half) / res - 0.5))
            j = int(round((p[1] + self.half) / res - 0.5))
            return max(0, min(n - 1, i)), max(0, min(n - 1, j))

        def nearest_free(i, j, radius=12):
            if free[j, i]:
                return i, j
            for r in range(1, radius + 1):
                for dj in range(-r, r + 1):
                    for di in range(-r, r + 1):
                        if max(abs(di), abs(dj)) != r:
                            continue
                        ii, jj = i + di, j + dj
                        if 0 <= ii < n and 0 <= jj < n and free[jj, ii]:
                            return ii, jj
            return None

        si, sj = to_idx(start)
        gi, gj = to_idx(goal)
        s = nearest_free(si, sj)
        g = nearest_free(gi, gj)
        if s is None or g is None:
            return []
        si, sj = s
        gi, gj = g
        if (si, sj) == (gi, gj):
            return [(float(goal[0]), float(goal[1]))]

        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
                (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
        INF = float("inf")
        gscore = np.full((n, n), INF, np.float64)
        parent = np.full((n, n, 2), -1, np.int32)
        gscore[sj, si] = 0.0
        h0 = math.hypot(si - gi, sj - gj) * res
        heap = [(h0, 0.0, si, sj)]
        closed = np.zeros((n, n), bool)
        while heap:
            f, gs, i, j = heapq.heappop(heap)
            if closed[j, i]:
                continue
            closed[j, i] = True
            if (i, j) == (gi, gj):
                break
            for di, dj, c in nbrs:
                ii, jj = i + di, j + dj
                if not (0 <= ii < n and 0 <= jj < n) or closed[jj, ii] or not free[jj, ii]:
                    continue
                if di and dj:
                    if not free[j, ii] or not free[jj, i]:
                        continue
                ng = gs + c * res
                if ng + 1e-12 < gscore[jj, ii]:
                    gscore[jj, ii] = ng
                    parent[jj, ii] = (i, j)
                    h = math.hypot(ii - gi, jj - gj) * res
                    heapq.heappush(heap, (ng + h, ng, ii, jj))
        if not np.isfinite(gscore[gj, gi]):
            return []
        cells = []
        i, j = gi, gj
        while (i, j) != (si, sj):
            cells.append((float(xs[i]), float(zs[j])))
            pi, pj = parent[j, i]
            if pi < 0:
                return []
            i, j = int(pi), int(pj)
        cells.append((float(xs[si]), float(zs[sj])))
        cells.reverse()
        # pin the true endpoints (grid centres are up to half a cell off)
        cells[0] = (float(start[0]), float(start[1]))
        cells[-1] = (float(goal[0]), float(goal[1]))
        if smooth:
            cells = self._string_pull(cells, free, xs, zs, clearance)
        return cells

    def _line_free(self, a, b, free, xs, zs, clearance, step=0.06):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(d / step))
        for k in range(n + 1):
            t = k / n
            p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if not self.point_free(p, clearance=clearance):
                return False
        return True

    def _string_pull(self, pts, free, xs, zs, clearance):
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._line_free(pts[i], pts[j], free, xs, zs, clearance):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    # ------------------------------------------------------------- depth ray
    def raycast_depth(self, eye, yaw: float, pitch: float,
                      width: int = 320, height: int = 180,
                      fov_v_deg: float = 72.0) -> np.ndarray:
        """Camera z-depth (metres) for every pixel.  Unnormalized ray directions
        are built so the slab parameter t *is* z-depth."""
        W, H = int(width), int(height)
        fwd = np.array([math.sin(yaw) * math.cos(pitch),
                        math.sin(pitch),
                        math.cos(yaw) * math.cos(pitch)], np.float64)
        right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], np.float64)
        up = np.cross(right, fwd)
        tv = math.tan(math.radians(fov_v_deg) * 0.5)
        th = tv * (W / H)
        x = (np.arange(W) + 0.5) / W * 2.0 - 1.0
        y = 1.0 - (np.arange(H) + 0.5) / H * 2.0
        X, Y = np.meshgrid(x, y, indexing="xy")
        dirs = (fwd[None, None, :]
                + (X * th)[..., None] * right[None, None, :]
                + (Y * tv)[..., None] * up[None, None, :])
        ox, oy, oz = float(eye[0]), float(eye[1]), float(eye[2])
        dx, dy, dz = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        depth = np.full((H, W), np.inf, np.float64)

        def update(t, valid):
            nonlocal depth
            good = valid & (t > 1e-4) & (t < depth)
            depth[good] = t[good]

        with np.errstate(divide="ignore", invalid="ignore"):
            # floor y=0
            t = -oy / dy
            px = ox + t * dx
            pz = oz + t * dz
            update(t, (np.abs(px) <= self.half) & (np.abs(pz) <= self.half) & (dy < 0))
            # ceiling
            t = (self.ceiling - oy) / dy
            px = ox + t * dx
            pz = oz + t * dz
            update(t, (np.abs(px) <= self.half) & (np.abs(pz) <= self.half) & (dy > 0))
            # four walls
            for plane, axis in ((self.half, "x"), (-self.half, "x"),
                                (self.half, "z"), (-self.half, "z")):
                if axis == "x":
                    t = (plane - ox) / dx
                    py = oy + t * dy
                    pz = oz + t * dz
                    ok = (np.abs(pz) <= self.half) & (py >= 0) & (py <= self.ceiling)
                else:
                    t = (plane - oz) / dz
                    px = ox + t * dx
                    py = oy + t * dy
                    ok = (np.abs(px) <= self.half) & (py >= 0) & (py <= self.ceiling)
                update(t, ok)
            # boxes (AABB slab)
            for b in self.boxes:
                c, s = math.cos(b.yaw), math.sin(b.yaw)
                # rotate ray into box frame
                lox, loz = ox - b.x, oz - b.z
                lox, loz = c * lox + s * loz, -s * lox + c * loz
                ldx, ldz = c * dx + s * dz, -s * dx + c * dz
                with np.errstate(divide="ignore", invalid="ignore"):
                    tx1 = (-b.hx - lox) / ldx
                    tx2 = (b.hx - lox) / ldx
                    tz1 = (-b.hz - loz) / ldz
                    tz2 = (b.hz - loz) / ldz
                tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(tz1, tz2))
                tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(tz1, tz2))
                ty1 = (0.0 - oy) / dy
                ty2 = (b.h - oy) / dy
                tmin = np.maximum(tmin, np.minimum(ty1, ty2))
                tmax = np.minimum(tmax, np.maximum(ty1, ty2))
                update(tmin, (tmax >= tmin) & (tmin > 0))
        depth[~np.isfinite(depth)] = 0.0
        return depth.astype(np.float32)

    # ----------------------------------------------------------- expert path
    def resample_path(self, pts, speed: float = DEFAULT_SPEED,
                      horizon: int = HORIZON, fps: float = FPS):
        """Return horizon points at speed*(k+1)/fps metres along a polyline."""
        if not pts:
            return np.zeros((horizon, 2), np.float64), False
        seg = []
        total = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            if d > 1e-9:
                seg.append((a, b, d, total))
                total += d
        if total < 1e-9:
            return np.repeat(np.asarray(pts[0])[None], horizon, 0), True
        out = []
        for k in range(horizon):
            s = min(total, speed * (k + 1) / fps)
            for a, b, d, base in seg:
                if s <= base + d or (a, b, d, base) is seg[-1]:
                    t = 0.0 if d < 1e-9 else (s - base) / d
                    out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
                    break
        return np.asarray(out, np.float64), False

    def expert_window(self, root, yaw: float, target, speed: float = DEFAULT_SPEED,
                      clearance: float = 0.35):
        """Return (path 8x3 body-frame, stop, world_points 8x2)."""
        root = np.asarray(root, np.float64)[:2]
        target = np.asarray(target, np.float64)[:2]
        dist = float(np.linalg.norm(target - root))
        if dist < 0.45:
            return np.zeros((HORIZON, 3), np.float32), True, np.repeat(target[None], HORIZON, 0)
        pts = self.plan(root, target, clearance=clearance, smooth=True)
        if not pts:
            return np.zeros((HORIZON, 3), np.float32), True, np.repeat(root[None], HORIZON, 0)
        world, stop = self.resample_path(pts, speed=speed)
        body = np.zeros((HORIZON, 3), np.float32)
        prev = root
        for k, p in enumerate(world):
            d = p - root
            body[k, 0] = d[0] * math.sin(yaw) + d[1] * math.cos(yaw)
            body[k, 1] = d[0] * math.cos(yaw) - d[1] * math.sin(yaw)
            tangent = p - prev
            if np.linalg.norm(tangent) < 1e-5:
                tangent = target - root
            heading = math.atan2(tangent[0], tangent[1])
            body[k, 2] = float(np.clip(wrap(heading - yaw), -0.7, 0.7))
            prev = p
        return body, stop, world

    def target_vector(self, root, yaw: float, target):
        """Body-frame target features for the boundary head."""
        root = np.asarray(root, np.float64)[:2]
        target = np.asarray(target, np.float64)[:2]
        d = target - root
        fwd = float(d[0] * math.sin(yaw) + d[1] * math.cos(yaw))
        lat = float(d[0] * math.cos(yaw) - d[1] * math.sin(yaw))
        dist = float(np.linalg.norm(d))
        bearing = math.atan2(lat, fwd)
        return np.array([fwd, lat, dist, math.sin(bearing), math.cos(bearing), 1.0], np.float32)


def random_arena(rng: np.random.Generator, n_boxes: int | None = None,
                 half: float = 8.0) -> Arena:
    if n_boxes is None:
        n_boxes = int(rng.integers(3, 8))
    boxes = []
    tries = 0
    while len(boxes) < n_boxes and tries < 200:
        tries += 1
        hx = float(rng.uniform(0.35, 1.5))
        hz = float(rng.uniform(0.35, 1.5))
        x = float(rng.uniform(-half + 1.2 + hx, half - 1.2 - hx))
        z = float(rng.uniform(-half + 1.2 + hz, half - 1.2 - hz))
        if math.hypot(x, z) < 2.0:
            continue
        ok = True
        for b in boxes:
            if abs(x - b.x) < hx + b.hx + 1.0 and abs(z - b.z) < hz + b.hz + 1.0:
                ok = False
                break
        if ok:
            boxes.append(Box(x, z, hx, hz, float(rng.uniform(0.8, 2.4)),
                             float(rng.uniform(-0.6, 0.6))))
    return Arena(half=half, ceiling=4.0, boxes=boxes)


def random_start_target(arena: Arena, rng: np.random.Generator, min_dist=3.0):
    for _ in range(200):
        s = rng.uniform(-arena.half + 1.0, arena.half - 1.0, 2)
        t = rng.uniform(-arena.half + 1.0, arena.half - 1.0, 2)
        if np.linalg.norm(t - s) < min_dist:
            continue
        if arena.point_free(s, 0.45) and arena.point_free(t, 0.45):
            if arena.plan(s, t, clearance=0.35, smooth=False):
                return s, t
    return np.array([-4.0, -4.0]), np.array([4.0, 4.0])


def demo_arena() -> Arena:
    """Fixed arena for the avatar demo: one box on the spawn-target line."""
    return Arena(half=8.0, ceiling=4.0, boxes=[
        Box(0.0, 0.0, 0.55, 0.55, 1.6, 0.15),
        Box(-3.8, 3.8, 0.7, 0.7, 1.6, 0.0),
        Box(4.5, -4.0, 0.7, 0.7, 1.6, 0.0),
    ])
