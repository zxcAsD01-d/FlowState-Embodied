"""Point-cloud reconstruction and error audit for depth -> physical region."""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def unproject(depth: np.ndarray, root_xz, yaw: float, pitch: float = -0.06,
              eye_h: float = 1.55, fov_v_deg: float = 72.0) -> np.ndarray:
    """z-depth (H,W) -> local world points (H,W,3)."""
    H, W = depth.shape
    fwd = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                    math.cos(yaw) * math.cos(pitch)], np.float64)
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], np.float64)
    up = np.cross(right, fwd)
    tv = math.tan(math.radians(fov_v_deg) * 0.5)
    th = tv * (W / H)
    xs = (np.arange(W) + 0.5) / W * 2.0 - 1.0
    ys = 1.0 - (np.arange(H) + 0.5) / H * 2.0
    X, Y = np.meshgrid(xs, ys)
    d = np.clip(depth, 0.2, 30.0)
    cam = ((X * th * d)[..., None] * right
           + (Y * tv * d)[..., None] * up
           + d[..., None] * fwd)
    eye = np.array([root_xz[0], eye_h, root_xz[1]], np.float64)
    return (eye[None, None, :] + cam).reshape(-1, 3)


def clean_depth(depth: np.ndarray, median: int = 3, edge: float = 0.30):
    """Median filter + drop depth-discontinuity pixels (mixed box edges)."""
    d = np.asarray(depth, np.float32)
    if median and median >= 3:
        d = cv2.medianBlur(d, median)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    bad = np.maximum(np.abs(gx), np.abs(gy)) > edge
    bad = ndimage.binary_dilation(bad, np.ones((3, 3), bool))
    return d, ~bad


def region_from_points(points: np.ndarray, root_xz, half=6.0, res=0.1,
                       clearance=0.30, closing=0.6, occ_low=0.20,
                       occ_high=3.0):
    from .region import Region
    n = int(round(2 * half / res))
    ox = float(root_xz[0]) - half
    oz = float(root_xz[1]) - half
    occ = np.zeros((n, n), bool)
    if len(points):
        p = points
        sel = (p[:, 1] > occ_low) & (p[:, 1] < occ_high)
        p = p[sel]
        ix = np.floor((p[:, 0] - ox) / res).astype(np.int64)
        iz = np.floor((p[:, 2] - oz) / res).astype(np.int64)
        ok = (ix >= 0) & (ix < n) & (iz >= 0) & (iz < n)
        occ[iz[ok], ix[ok]] = True
    if closing > 0:
        r = max(1, int(round(closing / res)))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        st = (xx * xx + yy * yy) <= r * r
        occ = ndimage.binary_closing(occ, structure=st)
    if clearance > 0:
        r = max(1, int(round(clearance / res)))
        occ = ndimage.binary_dilation(occ, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    sdf = np.where(occ, -ndimage.distance_transform_edt(occ) * res,
                   ndimage.distance_transform_edt(~occ) * res).astype(np.float32)
    return Region(origin=np.array([ox, oz], np.float64), res=res, occ=occ, sdf=sdf,
                  points=points if len(points) else np.zeros((0, 3)))


def cloud_region(depth, root_xz, yaw, pitch=-0.06, eye_h=1.55,
                 half=6.0, res=0.1, clearance=0.30, closing=0.6,
                 median=3, edge=0.30):
    d, valid = clean_depth(depth, median=median, edge=edge)
    p = unproject(d, root_xz, yaw, pitch, eye_h).reshape(depth.shape[0], depth.shape[1], 3)
    p = p[valid]
    return region_from_points(p, root_xz, half, res, clearance, closing)


def obstacle_mask(depth, root_xz, yaw, pitch=-0.06, eye_h=1.55,
                  floor=0.22, high=3.0):
    """Per-pixel obstacle mask (above floor, below ceiling) from GT depth."""
    H, W = depth.shape
    p = unproject(depth, root_xz, yaw, pitch, eye_h).reshape(H, W, 3)
    y = p[..., 1]
    m = (y > floor) & (y < high) & (depth > 0.2)
    return m


def obstacle_points(depth, root_xz, yaw, pitch=-0.06, eye_h=1.55,
                    floor=0.20, high=3.0):
    p = unproject(depth, root_xz, yaw, pitch, eye_h)
    y = p[:, 1]
    return p[(y > floor) & (y < high)]


def cloud_metrics(pred_depth, gt_depth, root_xz, yaw, pitch=-0.06, eye_h=1.55,
                  sample=6000):
    pred = obstacle_points(pred_depth, root_xz, yaw, pitch, eye_h)
    gt = obstacle_points(gt_depth, root_xz, yaw, pitch, eye_h)
    rng = np.random.default_rng(0)
    if len(pred) > sample:
        pred = pred[rng.choice(len(pred), sample, replace=False)]
    if len(gt) > sample:
        gt = gt[rng.choice(len(gt), sample, replace=False)]
    if len(pred) == 0 or len(gt) == 0:
        return {"chamfer": None, "pred_n": len(pred), "gt_n": len(gt)}
    tp = cKDTree(pred); tg = cKDTree(gt)
    d_pg, _ = tg.query(pred)
    d_gp, _ = tp.query(gt)
    return {
        "chamfer": float(0.5 * (d_pg.mean() + d_gp.mean())),
        "p95": float(0.5 * (np.percentile(d_pg, 95) + np.percentile(d_gp, 95))),
        "pred_n": int(len(pred)), "gt_n": int(len(gt)),
    }


def depth_affine(pred_depth, gt_depth, mask=None):
    p = np.asarray(pred_depth, np.float64).reshape(-1)
    g = np.asarray(gt_depth, np.float64).reshape(-1)
    if mask is not None:
        m = np.asarray(mask, bool).reshape(-1)
        p, g = p[m], g[m]
    ok = np.isfinite(p) & np.isfinite(g) & (g > 0.2) & (g < 30) & (p > 0.2) & (p < 30)
    p, g = p[ok], g[ok]
    if len(p) < 10:
        return None
    A = np.stack([g, np.ones_like(g)], 1)
    coef, *_ = np.linalg.lstsq(A, p, rcond=None)  # pred = a*gt + b
    ratio = float(np.median(p / g))
    return {"a": float(coef[0]), "b": float(coef[1]), "median_ratio": ratio,
            "median_abs": float(np.median(np.abs(p - g)))}


def occupancy_metrics(pred_depth, gt_depth, root_xz, yaw, pitch=-0.06,
                      eye_h=1.55, half=6.0, res=0.1, clearance=0.15):
    """Compare the solid occupancy footprints from predicted vs GT depth."""
    from .region import region_from_depth
    a = region_from_depth(pred_depth, root_xz, yaw, pitch, eye_h,
                          half=half, res=res, clearance=clearance)
    b = region_from_depth(gt_depth, root_xz, yaw, pitch, eye_h,
                          half=half, res=res, clearance=clearance)
    inter = np.logical_and(a.occ, b.occ).sum()
    union = np.logical_or(a.occ, b.occ).sum()
    missed = np.logical_and(b.occ, ~a.occ).sum()
    false = np.logical_and(a.occ, ~b.occ).sum()
    return {
        "iou": float(inter / max(union, 1)),
        "miss_rate": float(missed / max(int(b.occ.sum()), 1)),
        "false_rate": float(false / max(int(b.occ.sum()), 1)),
        "pred_cells": int(a.occ.sum()), "gt_cells": int(b.occ.sum()),
    }


def distance_bins(pred_depth, gt_depth, bins=(0.5, 1, 2, 4, 8, 16, 30)):
    p = np.asarray(pred_depth, np.float64)
    g = np.asarray(gt_depth, np.float64)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (g >= lo) & (g < hi)
        if m.sum() < 20:
            continue
        rows.append({"range": [lo, hi], "n": int(m.sum()),
                     "abs": float(np.abs(p[m] - g[m]).mean()),
                     "rel": float((np.abs(p[m] - g[m]) / g[m]).mean()),
                     "bias": float((p[m] - g[m]).mean())})
    return rows


def boxes_from_depth(depth, root_xz, yaw, pitch=-0.06, eye_h=1.55,
                     min_points=18, eps=0.35, max_extent=4.0,
                     clearance=0.30, max_height=3.0):
    """Cluster obstacle points and fit 2-D oriented boxes.

    Returns a list of (cx, cz, hx, hz, yaw, height) in the depth camera frame.
    The boxes are the physical no-go footprint; they fill the interior that a
    raw surface point cloud leaves empty.
    """
    from sklearn.cluster import DBSCAN
    from .arena import Box
    d, valid = clean_depth(depth, median=3, edge=0.30)
    p = unproject(d, root_xz, yaw, pitch, eye_h).reshape(depth.shape[0],
                                                         depth.shape[1], 3)
    p = p[valid]
    p = p[(p[:, 1] > 0.22) & (p[:, 1] < max_height)]
    if len(p) < min_points:
        return [], []
    lab = DBSCAN(eps=eps, min_samples=6).fit_predict(p[:, [0, 2]])
    boxes = []
    for k in np.unique(lab):
        if k < 0:
            continue
        q = p[lab == k]
        if len(q) < min_points:
            continue
        xy = q[:, [0, 2]].astype(np.float32)
        rect = cv2.minAreaRect(xy)
        (cx, cz), (w, h), ang = rect
        if w < 0.25 or h < 0.25:
            continue
        if w > max_extent or h > max_extent:
            continue
        boxes.append(Box(float(cx), float(cz), float(w) / 2, float(h) / 2,
                         float(q[:, 1].max() - q[:, 1].min()),
                         float(np.deg2rad(ang))))
    return boxes, p


def region_from_boxes(boxes, root_xz, half=6.0, res=0.1, clearance=0.30):
    from .region import Region
    n = int(round(2 * half / res))
    ox = float(root_xz[0]) - half; oz = float(root_xz[1]) - half
    xs = ox + (np.arange(n) + 0.5) * res
    zs = oz + (np.arange(n) + 0.5) * res
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    occ = np.zeros((n, n), bool)
    for b in boxes:
        c, s = math.cos(b.yaw), math.sin(b.yaw)
        dx, dz = X - b.x, Z - b.z
        lx = c * dx + s * dz; lz = -s * dx + c * dz
        occ |= (np.abs(lx) < b.hx + clearance) & (np.abs(lz) < b.hz + clearance)
    sdf = np.where(occ, -ndimage.distance_transform_edt(occ) * res,
                   ndimage.distance_transform_edt(~occ) * res).astype(np.float32)
    return Region(origin=np.array([ox, oz], np.float64), res=res, occ=occ,
                  sdf=sdf, points=np.zeros((0, 3)))


def compact_region(depth, root_xz, yaw, pitch=-0.06, eye_h=1.55,
                   half=6.0, res=0.1, clearance=0.30, closing=0.4,
                   max_extent=3.5, fill=1.0, fill_step=0.25):
    """Fill only compact obstacle clusters (boxes), not large walls/planes.

    For each DBSCAN cluster with a compact 3-D extent, extend its points along
    the camera ray into the object by ``fill`` metres, then build the footprint.
    """
    from sklearn.cluster import DBSCAN
    d, valid = clean_depth(depth, median=3, edge=0.30)
    p = unproject(d, root_xz, yaw, pitch, eye_h).reshape(depth.shape[0],
                                                         depth.shape[1], 3)
    p = p[valid]
    keep = (p[:, 1] > 0.22) & (p[:, 1] < 3.0)
    q = p[keep]
    if len(q) < 12:
        return region_from_points(q, root_xz, half, res, clearance, closing)
    lab = DBSCAN(eps=0.35, min_samples=6).fit_predict(q[:, [0, 2]])
    eye = np.array([root_xz[0], eye_h, root_xz[1]], np.float64)
    filled = [q]
    for k in np.unique(lab):
        if k < 0:
            continue
        c = q[lab == k]
        if len(c) < 12:
            continue
        ext = c.max(0) - c.min(0)
        if max(ext[0], ext[2]) > max_extent or ext[1] < 0.15:
            continue
        vec = c - eye[None, :]
        n = np.linalg.norm(vec[:, [0, 2]], axis=1, keepdims=True).clip(min=1e-6)
        unit = np.zeros_like(vec); unit[:, 0] = vec[:, 0] / n[:, 0]; unit[:, 2] = vec[:, 2] / n[:, 0]
        for s in np.arange(fill_step, fill + 1e-6, fill_step):
            filled.append(c + unit * s)
    pts = np.concatenate(filled, 0)
    return region_from_points(pts, root_xz, half, res, clearance, closing)
