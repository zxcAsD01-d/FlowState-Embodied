# -*- coding: utf-8 -*-
"""Offline ablation: where does the forward spine collapse come from?

Runs the frozen ARDY generator (same build as ardy_v.runtime) in three modes
and measures the sagittal profile of the spine joints relative to the pelvis:

  free   : text-conditioned free rollout, no path constraints
  path   : generate_path with a straight 0.70 m/s body path (demo conditions)
  zpath  : generate_path with a zero path (brake/stop conditions)

If `free` hunches too, the walk e / model is the source; if only `path`,
the path conditioning is the source.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ardy_v.runtime import ArdyV  # noqa: E402  (loads r07/r10 stack)
import run_r07_ardy_precision as r07  # noqa: E402

IDX = {"Spine": 1, "Spine1": 2, "Spine2": 3, "Spine3": 4, "Neck": 5, "Head": 6}


def spine_profile(chunk, gen, yaw):
    """Head/neck sagittal offsets relative to the pelvis, in the body frame."""
    j = np.asarray(chunk.joints, np.float64)  # (frames, bones, 3)
    c, s = np.cos(-yaw), np.sin(-yaw)
    out = []
    for f in range(j.shape[0]):
        root = j[f, gen.layout.root_idx]
        row = {}
        for name, bi in IDX.items():
            d = j[f, bi] - root
            row[name] = (float(d[1]), float(c * d[2] + s * d[0]))
        out.append(row)
    return out


def report(name, rows):
    print(f"\n[{name}]")
    print("  frame  " + "  ".join("%14s" % n for n in IDX))
    for i, row in enumerate(rows):
        if i % 4 and i != len(rows) - 1:
            continue
        print("  %5d  " % i + "  ".join(
            "%14s" % ("y%+.2f z%+.3f" % row[n]) for n in IDX))


def main():
    av = ArdyV(device="cuda", steps=6, seed=0)
    gen = av.gen

    # ---- 1. free rollout, walk e, no path constraints ------------------
    av._r10.set_e(gen, av.e_walk)
    rows = []
    for w in range(4):
        ch = gen.warm_start_free(frames=8) if w == 0 else gen.generate(
            type("Cmd", (), {"action": "large_motion", "target": None,
                             "hint": None, "root_height": 0.95})())
        rows.extend(spine_profile(ch, gen, float(gen._pelvis_heading())))
    report("free walk e", rows)

    # ---- 2. path walk, straight 0.70 m/s -------------------------------
    av.reset(np.zeros(2))
    rows = []
    for w in range(4):
        ch = gen.generate_path(
            np.array([[0.035, 0, 0]] * 8, np.float32),
            root_height=0.95, exec_frames=8, constrain_heading=True)
        rows.extend(spine_profile(ch, gen, float(gen._pelvis_heading())))
    report("path walk 0.70m/s", rows)

    # ---- 3. zero path (stop) -------------------------------------------
    av.reset(np.zeros(2))
    rows = []
    for w in range(2):
        ch = gen.generate_path(
            np.zeros((8, 3), np.float32),
            root_height=0.95, exec_frames=8, constrain_heading=True)
        rows.extend(spine_profile(ch, gen, float(gen._pelvis_heading())))
    report("zero path (stop)", rows)


if __name__ == "__main__":
    main()
