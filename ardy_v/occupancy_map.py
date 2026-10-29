"""Temporal world occupancy from depth + pose; the point-cloud region layer."""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .pointcloud import clean_depth, unproject

FOV_V_DEG = 72.0
EYE_H = 1.55
PITCH = -0.06


class WorldMap:
    """Log-odds occupancy in world XZ.  Depth endpoints are occupied; cells
    along each ray up to the endpoint are carved free.  Static boxes accumulate
    across windows, which suppresses single-frame depth errors."""

    def __init__(self, half: float = 8.0, res: float = 0.1,
                 l_occ: float = 0.85, l_free: float = 0.35,
                 occ_thresh: float = 0.45, clamp: float = 4.0):
        self.half = float(half); self.res = float(res)
        self.n = int(round(2 * half / res))
        self.log = np.zeros((self.n, self.n), np.float32)
        self.origin = np.array([-half, -half], np.float64)
        self.l_occ = float(l_occ); self.l_free = float(l_free)
        self.occ_thresh = float(occ_thresh); self.clamp = float(clamp)
        self.updates = 0
        self._sdf_cache = {}

    def clear(self):
        self.log.fill(0.0); self.updates = 0
        self._sdf_cache = {}

    def _idx(self, x, z):
        ix = np.floor((x - self.origin[0]) / self.res).astype(np.int64)
        iz = np.floor((z - self.origin[1]) / self.res).astype(np.int64)
        ok = (ix >= 0) & (ix < self.n) & (iz >= 0) & (iz < self.n)
        return ix, iz, ok

    def update(self, depth: np.ndarray, root_world, yaw: float,
               pitch: float = PITCH, eye_h: float = EYE_H,
               mask: np.ndarray | None = None, floor_guard: bool = True,
               min_h: float = 0.20, max_h: float = 3.0, conf: float = 0.5,
               step_px: int = 3, t_step: float = 0.25, decay: float = 0.995,
               max_range: float = 8.0):
        H, W = depth.shape
        d, valid = clean_depth(depth, median=3, edge=0.30)
        if decay != 1.0:
            self.log *= float(decay)
        fwd = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                        math.cos(yaw) * math.cos(pitch)], np.float64)
        right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], np.float64)
        up = np.cross(right, fwd)
        tv = math.tan(math.radians(FOV_V_DEG) * 0.5); th = tv * (W / H)
        xs = (np.arange(W) + 0.5) / W * 2.0 - 1.0
        ys = 1.0 - (np.arange(H) + 0.5) / H * 2.0
        X, Y = np.meshgrid(xs, ys)
        dirs = (fwd[None, None, :] + (X * th)[..., None] * right[None, None, :]
                + (Y * tv)[..., None] * up[None, None, :])
        eye = np.array([root_world[0], eye_h, root_world[1]], np.float64)
        # endpoints
        pt = eye[None, None, :] + d[..., None] * dirs
        y = pt[..., 1]
        occ = valid & (y > min_h) & (y < max_h)
        if mask is not None:
            occ &= (mask > conf)
        if max_range is not None:
            # Far depth is biased by ~1 m and turns floor/wall pixels into a
            # phantom occupied shell right where distant goals sit; true far
            # obstacles enter the map once the actor is within range anyway.
            occ &= (d < max_range)
        if floor_guard:
            # A pixel whose measured range equals the ground-plane hit range is
            # floor, even if the mask head calls it an obstacle.  Without this,
            # a floor band near the horizon maps to points 0.3-0.5 m high and
            # becomes a false occupied shell right where the actor walks.
            dy = dirs[..., 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                d_floor = -eye_h / dy
            tol = np.maximum(0.30, 0.10 * np.abs(d_floor))
            floor_like = (dy < -0.02) & np.isfinite(d_floor) & (d_floor > 0) \
                & (np.abs(d - d_floor) < tol)
            occ &= ~floor_like
        ix, iz, ok = self._idx(pt[..., 0], pt[..., 2])
        m = occ & ok
        np.add.at(self.log, (iz[m], ix[m]), self.l_occ)
        # free carving: sample t before the endpoint, only for valid depth
        tmax = np.minimum(d, 8.0)
        for t in np.arange(0.35, 8.0, t_step):
            m = valid & (t < d - 0.15)
            if not m.any():
                break
            q = eye[None, None, :] + t * dirs
            yq = q[..., 1]
            m &= (yq > 0.05) & (yq < 2.8)
            ix, iz, ok = self._idx(q[..., 0], q[..., 2])
            m &= ok
            np.add.at(self.log, (iz[m], ix[m]), -self.l_free)
        np.clip(self.log, -self.clamp, self.clamp, out=self.log)
        self.updates += 1
        self._sdf_cache = {}

    def occupied(self):
        return self.log > self.occ_thresh

    def sdf(self, clearance: float = 0.25):
        key = round(float(clearance), 3)
        hit = self._sdf_cache.get(key)
        if hit is not None:
            return hit
        occ = self.occupied()
        if clearance > 0:
            r = max(1, int(round(clearance / self.res)))
            occ = ndimage.binary_dilation(occ, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
        free = ~occ
        dist = ndimage.distance_transform_edt(free) * self.res
        sdf = np.where(occ, -ndimage.distance_transform_edt(occ) * self.res, dist)
        out = (occ, sdf.astype(np.float32))
        if len(self._sdf_cache) > 6:
            self._sdf_cache.clear()
        self._sdf_cache[key] = out
        return out

    def region_at(self, root_world, half: float = 6.0, clearance: float = 0.25):
        from .region import Region
        occ_full, sdf_full = self.sdf(clearance)
        n = int(round(2 * half / self.res))
        ox = float(root_world[0]) - half
        oz = float(root_world[1]) - half
        i0 = int(round((ox - self.origin[0]) / self.res))
        j0 = int(round((oz - self.origin[1]) / self.res))
        # crop with edge padding (unknown = free)
        occ = np.zeros((n, n), bool); sdf = np.full((n, n), 5.0, np.float32)
        si0 = max(0, i0); sj0 = max(0, j0)
        si1 = min(self.n, i0 + n); sj1 = min(self.n, j0 + n)
        if si1 > si0 and sj1 > sj0:
            di0 = si0 - i0; dj0 = sj0 - j0
            occ[dj0:dj0 + (sj1 - sj0), di0:di0 + (si1 - si0)] = occ_full[sj0:sj1, si0:si1]
            sdf[dj0:dj0 + (sj1 - sj0), di0:di0 + (si1 - si0)] = sdf_full[sj0:sj1, si0:si1]
        return Region(origin=np.array([ox, oz]), res=self.res, occ=occ, sdf=sdf,
                      points=np.zeros((0, 3)))

    def plan(self, start_world, target_world, yaw: float, speed: float = 0.70,
             horizon: int = 8, fps: float = 20.0, clearance: float = 0.25,
             margin: float = 0.05):
        """A* on the fused map; returns an 8x3 body-frame path (not keyframes)."""
        from .region import plan_region_path
        reg = self.region_at(start_world, half=6.0, clearance=clearance)
        return plan_region_path(reg, start_world, target_world, yaw,
                                margin=margin, speed=speed, horizon=horizon, fps=fps)

    def sample_sdf(self, xz):
        """Bilinear occupancy SDF (raw, metres) at world xz; outside = free."""
        _, sdf = self.sdf(0.0)
        p = (np.asarray(xz, np.float64) - self.origin) / self.res - 0.5
        h, w = sdf.shape
        x, y = p[..., 0], p[..., 1]
        x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
        tx, ty = x - x0, y - y0
        acc = np.zeros(x.shape, np.float64)
        for dy, dx, wgt in ((0, 0, (1 - tx) * (1 - ty)), (0, 1, tx * (1 - ty)),
                            (1, 0, (1 - tx) * ty), (1, 1, tx * ty)):
            xi = np.clip(x0 + dx, 0, w - 1); yi = np.clip(y0 + dy, 0, h - 1)
            acc += sdf[yi, xi] * wgt
        inside = (x >= -0.5) & (x <= w - 0.5) & (y >= -0.5) & (y <= h - 0.5)
        return np.where(inside, acc, 10.0)

    def plan_global(self, start_world, target_world, clearance: float = 0.35,
                    margin: float = 0.05):
        """A* over the entire world map; unknown cells are traversable."""
        import heapq
        occ, sdf_raw = self.sdf(0.0)
        free = ~occ
        n = self.n
        start = np.asarray(start_world, np.float64)
        goal = np.asarray(target_world, np.float64)
        if np.linalg.norm(goal - start) < 1e-6:
            return [tuple(start)]
        lim = self.half - 0.3
        goal = np.clip(goal, -lim, lim)

        def idx(p):
            i = int(round((p[0] - self.origin[0]) / self.res - 0.5))
            j = int(round((p[1] - self.origin[1]) / self.res - 0.5))
            return max(0, min(n - 1, i)), max(0, min(n - 1, j))

        si, sj = idx(start); gi, gj = idx(goal)
        r = max(1, int(round(clearance / self.res)))
        inflated = ndimage.binary_dilation(occ, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
        travel = ~inflated
        ys, xs = np.where(travel)
        if len(xs) == 0:
            return []
        if not travel[sj, si]:
            k = int(np.argmin((xs - si) ** 2 + (ys - sj) ** 2)); si, sj = int(xs[k]), int(ys[k])
        if not travel[gj, gi]:
            k = int(np.argmin((xs - gi) ** 2 + (ys - gj) ** 2)); gi, gj = int(xs[k]), int(ys[k])
        nbrs = [(-1, 0, 1.), (1, 0, 1.), (0, -1, 1.), (0, 1, 1.),
                (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
        INF = float("inf")
        g = np.full((n, n), INF); par = np.full((n, n, 2), -1, np.int32)
        g[sj, si] = 0.0
        heap = [(np.hypot(si - gi, sj - gj), 0.0, si, sj)]
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
                if not (0 <= ii < n and 0 <= jj < n) or closed[jj, ii] or not travel[jj, ii]:
                    continue
                if di and dj and (not travel[j, ii] or not travel[jj, i]):
                    continue
                ng = gs + c
                if ng < g[jj, ii]:
                    g[jj, ii] = ng; par[jj, ii] = (i, j)
                    heapq.heappush(heap, (ng + np.hypot(ii - gi, jj - gj), ng, ii, jj))
        if not np.isfinite(g[gj, gi]):
            return []
        cells = []
        i, j = gi, gj
        while (i, j) != (si, sj):
            cells.append((self.origin[0] + (i + .5) * self.res,
                          self.origin[1] + (j + .5) * self.res))
            pi, pj = par[j, i]
            if pi < 0:
                return []
            i, j = int(pi), int(pj)
        cells.append((self.origin[0] + (si + .5) * self.res,
                      self.origin[1] + (sj + .5) * self.res))
        cells.reverse()
        cells[0] = tuple(start); cells[-1] = tuple(goal)

        def los(a, b):
            L = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            m = max(2, int(L / (self.res * 0.5)))
            pts = np.array([[a[0] + (b[0] - a[0]) * k / m,
                             a[1] + (b[1] - a[1]) * k / m] for k in range(m + 1)])
            return bool((self.sample_sdf(pts) > clearance + margin).all())

        out = [cells[0]]; a = 0
        while a < len(cells) - 1:
            b = len(cells) - 1
            while b > a + 1 and not los(cells[a], cells[b]):
                b -= 1
            out.append(cells[b]); a = b
        return out

    def route_valid(self, route, root_world, clearance: float = 0.28,
                    margin: float = 0.02, tail_ignore: float = 0.30):
        """Is the remaining route still outside the inflated obstacles?

        The final tail_ignore metres are exempt: the goal itself is a chosen
        world point, and rejecting a route because its endpoint hugs a
        (possibly phantom) occupied cell caused endless replan loops.
        """
        from .route_tracker import RouteTracker
        if not route:
            return False
        tr = RouteTracker(route, route[-1])
        s0, d = tr.project(root_world)
        if d > 0.75:
            return False
        s = s0; end = max(s0, tr.length - tail_ignore)
        pts = []
        while s <= end:
            pts.append(tr.point_at(s)); s += 0.10
        if not pts:
            return True
        return bool((self.sample_sdf(np.asarray(pts)) > clearance + margin).all())
