"""Closed-loop QC for the online visual navigator (live browser, real ARDY).

A: first plan from the initial view, walk to the default target; static scene
   must not keep re-planning.
B: put a box on the remaining route, in view; one confirmed invalidation must
   produce a new route, and the walk must still finish without collision.
C: real pointer drag; it must move the box and not be treated as a target click.
Saves screenshots + JSON under runs/.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs"


def state(page):
    return page.evaluate("() => window.ardyDemo.state()")


def snap(page, name):
    try:
        page.screenshot(path=str(OUT / name))
    except Exception as exc:  # noqa: BLE001
        print("[qc] screenshot failed", name, repr(exc), flush=True)


def wait_ready(page, timeout=90):
    page.wait_for_function(
        "() => window.ardyDemo && window.ardyDemo.state().ready"
        " && window.ardyDemo.state().frames > 0", timeout=timeout * 1000)


def slim(s):
    g = s.get("guide") or {}
    return {
        "root": s.get("root"), "yaw": s.get("yaw"), "dist": s.get("dist"),
        "status": s.get("status"), "collision": s.get("collision"),
        "violation": s.get("violation"), "minSdf": s.get("minSdf"),
        "replans": g.get("replans"), "invalid": g.get("invalid"),
        "streak": g.get("invalid_streak"), "reason": g.get("reason"),
        "remaining": g.get("route_remaining"), "occ": g.get("occupied_cells"),
        "mask": g.get("mask_frac"), "route": s.get("route"),
    }


def route_ahead(route, root, dist):
    """Point at `dist` metres along the route, measured from the projection of root."""
    if not route or len(route) < 2:
        return None
    best = (1e9, 0, 0.0)
    for i, (a, b) in enumerate(zip(route[:-1], route[1:])):
        ax, az = a; bx, bz = b
        abx, abz = bx - ax, bz - az
        L2 = abx * abx + abz * abz
        t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0,
              ((root[0] - ax) * abx + (root[1] - az) * abz) / L2))
        qx, qz = ax + abx * t, az + abz * t
        d = math.hypot(root[0] - qx, root[1] - qz)
        if d < best[0]:
            best = (d, i, t)
    _, i, t = best
    a, b = route[i], route[i + 1]
    sx, sz = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
    rem = dist
    for j in range(i, len(route) - 1):
        a, b = route[j], route[j + 1]
        if j == i:
            a = (sx, sz)
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg <= 1e-9:
            continue
        if rem <= seg:
            u = rem / seg
            return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
        rem -= seg
    return tuple(route[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8830/")
    ap.add_argument("--walk", type=float, default=120.0)
    ap.add_argument("--block-walk", type=float, default=80.0)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    log = {"samples": [], "phases": {}}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            channel="msedge", headless=not args.headed,
            args=["--use-angle=default"])
        page = browser.new_page(viewport={"width": 1280, "height": 760})
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(args.url, wait_until="domcontentloaded")
        wait_ready(page)

        # ---------------- A: initial plan, static scene, walk
        page.evaluate("() => window.ardyDemo.reset()")
        time.sleep(2.0)
        s0 = state(page)
        print("[qc] A start", json.dumps(slim(s0), ensure_ascii=False), flush=True)
        snap(page, "nav_live_start.png")
        t0 = time.time(); samples = []; last = s0; arrived = False
        while time.time() - t0 < args.walk:
            s = state(page)
            samples.append({"t": round(time.time() - t0, 2), **slim(s)})
            last = s
            if s["dist"] is not None and s["dist"] < 0.65:
                arrived = True; break
            if s["collision"] or s["violation"]:
                break
            time.sleep(1.0)
        log["samples"] = samples
        log["phases"]["A_walk"] = {
            "arrived": arrived, "seconds": round(time.time() - t0, 1),
            "replans": (last.get("guide") or {}).get("replans"),
            "collision": last["collision"], "violation": last["violation"],
            "dist": last["dist"], "minSdf": last["minSdf"],
        }
        print("[qc] A done", json.dumps(log["phases"]["A_walk"], ensure_ascii=False), flush=True)
        snap(page, "nav_live_arrive.png")
        time.sleep(4.0)
        s_after = state(page)
        log["phases"]["A_static"] = {"replans": (s_after.get("guide") or {}).get("replans"),
                                     "status": s_after["status"]}
        snap(page, "nav_live_static.png")

        # ---------------- C: real pointer drag on a fresh scene
        page.evaluate("() => window.ardyDemo.reset()")
        time.sleep(2.5)
        tgt_before = state(page)["target"]
        pos = page.evaluate("() => window.ardyDemo.screenOfBox(0)")
        drag = {"projected": pos, "target_before": tgt_before}
        if pos:
            x0, y0 = pos
            page.mouse.move(x0, y0)
            page.mouse.down()
            for k in range(1, 9):
                page.mouse.move(x0 + 16 * k, y0 - 5 * k, steps=2)
                time.sleep(0.05)
            time.sleep(0.4)
            page.mouse.up()
            time.sleep(1.2)
            s = state(page)
            b = s["boxes"][0]
            drag.update({"box_after": [b["x"], b["z"]],
                         "target_after": s["target"], "status": s["status"]})
        log["phases"]["C_drag"] = drag
        print("[qc] C drag", json.dumps(drag, ensure_ascii=False), flush=True)
        snap(page, "nav_live_drag.png")

        # ---------------- B: block the remaining route in view
        page.evaluate("() => window.ardyDemo.reset()")
        time.sleep(4.0)
        s = state(page)
        pt = route_ahead(s.get("route"), s["root"], 2.5)
        # wait (a little) until the box point is roughly in front of the eye
        tend = time.time() + 12
        while pt is not None and time.time() < tend:
            dx, dz = pt[0] - s["root"][0], pt[1] - s["root"][1]
            ang = abs((math.atan2(dx, dz) - (s.get("yaw") or 0) + math.pi)
                      % (2 * math.pi) - math.pi)
            if ang < 0.55 and math.hypot(dx, dz) > 1.6:
                break
            time.sleep(1.0)
            s = state(page)
            pt = route_ahead(s.get("route"), s["root"], 2.5)
        if pt is None:
            pt = (s["root"][0] + 2.0 * math.sin(s.get("yaw") or 0),
                  s["root"][1] + 2.0 * math.cos(s.get("yaw") or 0))
        before = (s.get("guide") or {}).get("replans")
        page.evaluate("([x, z]) => window.ardyDemo.moveBox(0, x, z, 0.8)", [pt[0], pt[1]])
        print("[qc] B place box at", pt, "replans before", before, flush=True)

        tb = time.time(); detected = None; sb = None
        while time.time() - tb < 25:
            sb = state(page)
            g = sb.get("guide") or {}
            if (g.get("invalid") or (g.get("replans") or 0) > (before or 0)):
                detected = sb
                break
            time.sleep(0.6)
        log["phases"]["B_detect"] = {
            "box_at": list(pt), "replans_before": before,
            "replans": (sb.get("guide") or {}).get("replans"),
            "invalid": (sb.get("guide") or {}).get("invalid"),
            "reason": (sb.get("guide") or {}).get("reason"),
            "seconds": round(time.time() - tb, 1),
            "collision": sb["collision"], "dist": sb["dist"],
            "nav": sb.get("nav"), "route_before": s.get("route"),
            "route_after": sb.get("route"),
        }
        print("[qc] B detect", json.dumps(log["phases"]["B_detect"], ensure_ascii=False), flush=True)
        snap(page, "nav_live_blocked.png")

        # continue the walk with the new box in place; must not collide
        tb2 = time.time(); sa = sb; arrived_b = False
        while time.time() - tb2 < args.block_walk:
            sa = state(page)
            if sa["dist"] is not None and sa["dist"] < 0.65:
                arrived_b = True; break
            if sa["collision"] or sa["violation"]:
                break
            time.sleep(1.0)
        log["phases"]["B_walk"] = {
            "arrived": arrived_b, "seconds": round(time.time() - tb2, 1),
            "dist": sa["dist"], "collision": sa["collision"],
            "violation": sa["violation"], "minSdf": sa["minSdf"],
            "replans": (sa.get("guide") or {}).get("replans"),
            "nav": sa.get("nav"),
        }
        print("[qc] B walk", json.dumps(log["phases"]["B_walk"], ensure_ascii=False), flush=True)
        snap(page, "nav_live_after_block.png")

        log["console_errors"] = errors[-12:]
        page.evaluate("() => window.ardyDemo.reset()")
        browser.close()

    OUT.mkdir(exist_ok=True)
    (OUT / "qc_nav_live.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), "utf-8")
    print("[qc] wrote runs/qc_nav_live.json", flush=True)


if __name__ == "__main__":
    main()
