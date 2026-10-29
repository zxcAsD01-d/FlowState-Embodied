"""Depth -> physical no-go region (occupancy + signed distance) in ARDY local XZ.

The depth head predicts camera z-depth.  Pixels above the floor are obstacle
surface points; their XZ footprints form the forbidden region.  The region is a
hard constraint in the demo, and its SDF is what the differentiable barrier
reads inside ARDY's sampling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

FOV_V_DEG = 72.0
EYE_H = 1.55
PITCH = -0.06


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Region:
    origin: np.ndarray      # (2,) world/local x,z of grid cell (0,0)
    res: float
    occ: np.ndarray         # (H,W) bool
    sdf: np.ndarray         # (H,W) float32, metres; negative inside
    points: np.ndarray      # (N,3) sampled obstacle points (for debug)

    @property
    def shape(self):
        return self.occ.shape

    def sample_sdf(self, xz: np.ndarray) -> np.ndarray:
        """Bilinear SDF at (...,2) xz; outside the grid is free (large)."""
        p = (np.asarray(xz, np.float64) - self.origin) / self.res - 0.5
        h, w = self.sdf.shape
        x, y = p[..., 0], p[..., 1]
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        tx, ty = x - x0, y - y0
        acc = np.zeros(x.shape, np.float64)
        for dy, dx, wgt in ((0, 0, (1 - tx) * (1 - ty)), (0, 1, tx * (1 - ty)),
                            (1, 0, (1 - tx) * ty), (1, 1, tx * ty)):
            xi = np.clip(x0 + dx, 0, w - 1)
            yi = np.clip(y0 + dy, 0, h - 1)
            acc += self.sdf[yi, xi] * wgt
        inside = (x >= -0.5) & (x <= w - 0.5) & (y >= -0.5) & (y <= h - 0.5)
        return np.where(inside, acc, 10.0)

def depth_to_points(depth: np.ndarray, root_xz, yaw: float, pitch: float = PITCH,
                    eye_h: float = EYE_H) -> np.ndarray:
    """Unproject z-depth (H,W) to local world points (H*W,3).  root_xz is ARDY
    local XZ; the viewer's eye is at floor+eye_h, not pelvis+eye_h."""
    H, W = depth.shape
    fwd = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                    math.cos(yaw) * math.cos(pitch)], np.float64)
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], np.float64)
    up = np.cross(right, fwd)
    tv = math.tan(math.radians(FOV_V_DEG) * 0.5)
    th = tv * (W / H)
    xs = (np.arange(W) + 0.5) / W * 2.0 - 1.0
    ys = 1.0 - (np.arange(H) + 0.5) / H * 2.0
    X, Y = np.meshgrid(xs, ys)
    d = np.clip(depth, 0.2, 30.0)
    cam = (X * th * d)[..., None] * right + (Y * tv * d)[..., None] * up + d[..., None] * fwd
    eye = np.array([root_xz[0], eye_h, root_xz[1]], np.float64)
    return (eye[None, None, :] + cam).reshape(-1, 3)


def floor_obstacle_mask(depth: np.ndarray, yaw: float, pitch: float = PITCH,
                        eye_h: float = EYE_H, tol: float = 0.22) -> np.ndarray:
    """Pixels whose depth is closer than the expected floor hit.  This catches
    boxes even when the depth head's absolute y is biased."""
    H, W = depth.shape
    fwd = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                    math.cos(yaw) * math.cos(pitch)], np.float64)
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], np.float64)
    up = np.cross(right, fwd)
    tv = math.tan(math.radians(FOV_V_DEG) * 0.5)
    ys = 1.0 - (np.arange(H) + 0.5) / H * 2.0
    dir_y = fwd[1] + (ys * tv)[:, None] * up[1]
    floor_t = np.where(dir_y < -0.03, -eye_h / dir_y, np.inf)
    return (dir_y < -0.03) & (depth < floor_t - tol)


def region_from_depth(depth: np.ndarray, root_xz, yaw: float,
                      pitch: float = PITCH, eye_h: float = EYE_H,
                      half: float = 6.0, res: float = 0.1,
                      clearance: float = 0.30,
                      occ_low: float = 0.22, occ_high: float = 3.0,
                      step_px: int = 2, extend: float = 2.0) -> Region:
    """Build the forbidden region from a depth map.

    Depth only sees surfaces, so marking surface points alone leaves the box
    interior free (and the barrier walks through it).  We extrude each ray
    behind the first hit by ``extend`` metres; obstacles become solid
    footprints, while the floor/walls stay outside the local map.  The result
    is inflated by ``clearance``; SDF is negative inside.
    """
    pts = depth_to_points(depth, root_xz, yaw, pitch, eye_h)
    # Obstacle points by the floor-plane test (robust to a depth bias in y).
    om = floor_obstacle_mask(depth, yaw, pitch, eye_h)
    obs_pts = pts.reshape(depth.shape[0], depth.shape[1], 3)[om]
    if extend > 0:
        ext = []
        for s in np.arange(0.2, extend + 1e-6, 0.2):
            ext.append(depth_to_points(np.minimum(depth + s, 30.0), root_xz, yaw, pitch, eye_h))
        pts = np.concatenate([pts] + ext, 0)
    pts = pts[::step_px]
    n = int(round(2 * half / res))
    ox = float(root_xz[0]) - half
    oz = float(root_xz[1]) - half
    occ = np.zeros((n, n), bool)
    sel = (pts[:, 1] > occ_low) & (pts[:, 1] < occ_high)
    p = np.concatenate([pts[sel], obs_pts], 0) if len(obs_pts) else pts[sel]
    ix = np.floor((p[:, 0] - ox) / res).astype(np.int64)
    iz = np.floor((p[:, 2] - oz) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < n) & (iz >= 0) & (iz < n)
    occ[iz[ok], ix[ok]] = True
    # inflate by clearance
    r = max(0, int(round(clearance / res)))
    if r > 0:
        occ = ndimage.binary_dilation(occ, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    # SDF in metres: positive in free space, negative inside
    free = ~occ
    dist = ndimage.distance_transform_edt(free) * res
    sdf = np.where(occ, -ndimage.distance_transform_edt(occ) * res, dist)
    return Region(origin=np.array([ox, oz], np.float64), res=res,
                  occ=occ, sdf=sdf.astype(np.float32), points=pts)


def region_png(region: Region, root_xz, target_xz, path_xz=None,
               size: int = 220) -> np.ndarray:
    """BGR top-down constraint map for the browser inset."""
    import cv2
    h, w = region.occ.shape
    img = np.zeros((size, size, 3), np.uint8)
    # free dark, occupied red
    img[:] = (22, 28, 36)
    occ_u8 = (region.occ.astype(np.uint8) * 255)
    occ_u8 = cv2.resize(occ_u8, (size, size), interpolation=cv2.INTER_NEAREST)
    red = np.zeros_like(img)
    red[:, :, 2] = occ_u8
    img = cv2.addWeighted(img, 1.0, red, 0.55, 0)
    def to_px(xz):
        return (int(round((xz[0] - region.origin[0]) / region.res / w * size)),
                int(round((xz[1] - region.origin[1]) / region.res / h * size)))
    if path_xz is not None:
        pts = np.array([to_px(p) for p in path_xz], np.int32)
        cv2.polylines(img, [pts], False, (90, 255, 150), 2)
    tx, ty = to_px(target_xz)
    cv2.drawMarker(img, (tx, ty), (80, 255, 120), cv2.MARKER_CROSS, 14, 2)
    ax, ay = to_px(root_xz)
    cv2.circle(img, (ax, ay), 4, (255, 220, 90), -1)
    cv2.circle(img, (ax, ay), 6, (0, 0, 0), 1)
    # 1 m grid
    for k in range(0, int(region.res * w) + 1, 10):
        x0 = int(k / (region.res * w) * size)
        cv2.line(img, (x0, 0), (x0, size), (40, 48, 58), 1)
        y0 = int(k / (region.res * h) * size)
        cv2.line(img, (0, y0), (size, y0), (40, 48, 58), 1)
    return img


def plan_region_path(region: "Region", start, goal, yaw: float,
                     margin: float = 0.05, speed: float = 0.70,
                     horizon: int = 8, fps: float = 20.0):
    """A* on the depth-derived free space; return a body-frame 8x3 path.

    The goal may be outside the local grid; in that case a subgoal along the
    start-goal line (inside the grid) is used and the next window re-plans.
    """
    import heapq
    h, w = region.sdf.shape
    free = region.sdf > margin
    start = np.asarray(start, np.float64)
    goal = np.asarray(goal, np.float64)
    d = goal - start
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return np.zeros((horizon, 3), np.float32), True
    unit = d / dist
    # subgoal inside the local grid
    lim = min(dist, region.res * w * 0.42)
    sub = start + unit * lim
    for _ in range(20):
        gi = int(round((sub[0] - region.origin[0]) / region.res - 0.5))
        gj = int(round((sub[1] - region.origin[1]) / region.res - 0.5))
        if 0 <= gi < w and 0 <= gj < h and free[gj, gi]:
            break
        lim *= 0.9
        sub = start + unit * lim
    else:
        return np.zeros((horizon, 3), np.float32), True
    si = int(round((start[0] - region.origin[0]) / region.res - 0.5))
    sj = int(round((start[1] - region.origin[1]) / region.res - 0.5))
    si = max(0, min(w - 1, si)); sj = max(0, min(h - 1, sj))
    if not free[sj, si]:
        # nearest free cell to the start
        ys, xs = np.where(free)
        if len(xs) == 0:
            return np.zeros((horizon, 3), np.float32), True
        k = int(np.argmin((xs - si) ** 2 + (ys - sj) ** 2))
        si, sj = int(xs[k]), int(ys[k])
    gi = max(0, min(w - 1, gi)); gj = max(0, min(h - 1, gj))
    nbrs = [(-1, 0, 1.), (1, 0, 1.), (0, -1, 1.), (0, 1, 1.),
            (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
    INF = float("inf")
    g = np.full((h, w), INF)
    par = np.full((h, w, 2), -1, np.int32)
    g[sj, si] = 0.0
    heap = [(np.hypot(si - gi, sj - gj), 0.0, si, sj)]
    closed = np.zeros((h, w), bool)
    while heap:
        f, gs, i, j = heapq.heappop(heap)
        if closed[j, i]:
            continue
        closed[j, i] = True
        if (i, j) == (gi, gj):
            break
        for di, dj, c in nbrs:
            ii, jj = i + di, j + dj
            if not (0 <= ii < w and 0 <= jj < h) or closed[jj, ii] or not free[jj, ii]:
                continue
            if di and dj and (not free[j, ii] or not free[jj, i]):
                continue
            ng = gs + c
            if ng < g[jj, ii]:
                g[jj, ii] = ng
                par[jj, ii] = (i, j)
                heapq.heappush(heap, (ng + np.hypot(ii - gi, jj - gj), ng, ii, jj))
    if not np.isfinite(g[gj, gi]):
        return np.zeros((horizon, 3), np.float32), True
    cells = []
    i, j = gi, gj
    while (i, j) != (si, sj):
        cells.append((region.origin[0] + (i + .5) * region.res,
                      region.origin[1] + (j + .5) * region.res))
        pi, pj = par[j, i]
        if pi < 0:
            return np.zeros((horizon, 3), np.float32), True
        i, j = int(pi), int(pj)
    cells.append((region.origin[0] + (si + .5) * region.res,
                  region.origin[1] + (sj + .5) * region.res))
    cells.reverse()
    cells[0] = tuple(start)
    # line-of-sight string pull
    def los(a, b):
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / (region.res * 0.5)))
        for k in range(n + 1):
            t = k / n
            p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if region.sample_sdf(np.array(p)) <= margin:
                return False
        return True
    out = [cells[0]]
    a = 0
    while a < len(cells) - 1:
        b = len(cells) - 1
        while b > a + 1 and not los(cells[a], cells[b]):
            b -= 1
        out.append(cells[b]); a = b
    # resample horizon points by arclength
    seg = []
    total = 0.0
    for a, b in zip(out[:-1], out[1:]):
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if d > 1e-9:
            seg.append((a, b, d, total)); total += d
    if total < 1e-9:
        return np.zeros((horizon, 3), np.float32), True
    pts = []
    for k in range(horizon):
        sdist = min(total, speed * (k + 1) / fps)
        for a, b, d, base in seg:
            if sdist <= base + d or (a, b, d, base) is seg[-1]:
                t = 0.0 if d < 1e-9 else (sdist - base) / d
                pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
                break
    body = np.zeros((horizon, 3), np.float32)
    prev = start
    for k, p in enumerate(pts):
        dx, dz = p[0] - start[0], p[1] - start[1]
        body[k, 0] = dx * math.sin(yaw) + dz * math.cos(yaw)
        body[k, 1] = dx * math.cos(yaw) - dz * math.sin(yaw)
        tan = np.array(p) - np.array(prev)
        if np.linalg.norm(tan) < 1e-5:
            tan = goal - start
        heading = math.atan2(tan[0], tan[1])
        body[k, 2] = float(np.clip(wrap(heading - yaw), -0.7, 0.7))
        prev = p
    return body, False


def region_from_arena(arena, root_xz, offset=(0.0, 0.0), half: float = 6.0,
                      res: float = 0.1, clearance: float = 0.30) -> Region:
    """Physical no-go region from the scene geometry (demo hard constraint)."""
    n = int(round(2 * half / res))
    ox = float(root_xz[0]) - half
    oz = float(root_xz[1]) - half
    xs = ox + (np.arange(n) + 0.5) * res
    zs = oz + (np.arange(n) + 0.5) * res
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    occ = np.zeros((n, n), bool)
    ox0, oz0 = float(offset[0]), float(offset[1])
    for b in arena.boxes:
        c, s = math.cos(b.yaw), math.sin(b.yaw)
        dx, dz = X - (b.x - ox0), Z - (b.z - oz0)
        lx = c * dx + s * dz
        lz = -s * dx + c * dz
        occ |= (np.abs(lx) < b.hx + clearance) & (np.abs(lz) < b.hz + clearance)
    sdf = np.where(occ, -ndimage.distance_transform_edt(occ) * res,
                   ndimage.distance_transform_edt(~occ) * res).astype(np.float32)
    return Region(origin=np.array([ox, oz], np.float64), res=res, occ=occ,
                  sdf=sdf, points=np.zeros((0, 3)))
