"""Run a closed-loop test route against the live demo service.

A route is a small JSON file (see routes/): where the avatar starts, where
the goal is, where the boxes are placed, and what counts as success.  The
runner drives the same page API a user does — setTarget / moveBox / reset —
and records what the frozen ARDY + online navigator actually do.

    python tools/run_route.py routes/direct.json
    python tools/run_route.py routes/blocked_midway.json --headed

Results go to runs/routes/<name>.json (+ screenshot).  This is a real
closed loop: the service must be running and a GPU is required.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "routes"


def state(page):
    return page.evaluate("() => window.ardyDemo.state()")


def wait_ready(page, timeout=90):
    page.wait_for_function(
        "() => window.ardyDemo && window.ardyDemo.state().ready"
        " && window.ardyDemo.state().frames > 0", timeout=timeout * 1000)


def route_ahead(route, root, dist):
    """Point `dist` metres along the route from the projection of root."""
    if not route or len(route) < 2:
        return None
    best = (1e9, 0, 0.0)
    for i, (a, b) in enumerate(zip(route[:-1], route[1:])):
        abx, abz = b[0] - a[0], b[1] - a[1]
        L2 = abx * abx + abz * abz
        t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0,
              ((root[0] - a[0]) * abx + (root[1] - a[1]) * abz) / L2))
        qx, qz = a[0] + abx * t, a[1] + abz * t
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


def slim(s):
    g = s.get("guide") or {}
    return {
        "root": s.get("root"), "yaw": s.get("yaw"), "dist": s.get("dist"),
        "status": s.get("status"), "collision": s.get("collision"),
        "violation": s.get("violation"), "minSdf": s.get("minSdf"),
        "replans": g.get("replans"), "invalid": g.get("invalid"),
        "reason": g.get("reason"), "remaining": g.get("route_remaining"),
        "occ": g.get("occupied_cells"), "route": s.get("route"),
    }


def walk_until_done(page, timeout_s, walk_timeout):
    """Poll until arrival / collision / violation / timeout."""
    t0 = time.time()
    last = state(page)
    arrived = False
    samples = []
    while time.time() - t0 < walk_timeout:
        s = state(page)
        last = s
        samples.append({"t": round(time.time() - t0, 2), **slim(s)})
        if s["dist"] is not None and s["dist"] < 0.65:
            arrived = True
            break
        if s["collision"] or s["violation"]:
            break
        if time.time() - t0 > timeout_s:
            break
        time.sleep(1.0)
    return arrived, last, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("route", type=str)
    ap.add_argument("--url", default="http://127.0.0.1:8830/")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = json.loads(Path(args.route).read_text("utf-8"))
    name = spec.get("name") or Path(args.route).stem
    kind = spec.get("kind", "walk")
    timeout_s = float(spec.get("timeout_s", 90))
    result = {"route": name, "kind": kind, "spec": spec, "phases": {}}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            channel="msedge", headless=not args.headed,
            args=["--use-angle=default"])
        page = browser.new_page(viewport={"width": 1280, "height": 760})
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(args.url, wait_until="domcontentloaded")
        wait_ready(page)

        page.evaluate("() => window.ardyDemo.reset()")
        time.sleep(1.5)
        for i, b in enumerate(spec.get("boxes", [])):
            page.evaluate("([i, x, z, y]) => window.ardyDemo.moveBox(i, x, z, y)",
                          [int(b.get("slot", i)), b["x"], b["z"], b.get("y", 0.8)])
        if spec.get("goal"):
            page.evaluate("([x, z]) => window.ardyDemo.setTarget(x, z)",
                          [spec["goal"][0], spec["goal"][1]])
        time.sleep(2.0)

        s0 = state(page)
        result["start"] = slim(s0)
        print(f"[route:{name}] start", json.dumps(slim(s0), ensure_ascii=False), flush=True)

        if kind == "block":
            # walk first, then drop a box on the remaining route in view
            block_dist = float(spec.get("block_dist", 2.5))
            t_end = time.time() + 20
            pt = None
            while time.time() < t_end:
                s = state(page)
                pt = route_ahead(s.get("route"), s["root"], block_dist)
                if pt is None:
                    break
                dx, dz = pt[0] - s["root"][0], pt[1] - s["root"][1]
                ang = abs((math.atan2(dx, dz) - (s.get("yaw") or 0) + math.pi)
                          % (2 * math.pi) - math.pi)
                if ang < 0.55 and math.hypot(dx, dz) > 1.6:
                    break
                time.sleep(1.0)
            if pt is None:
                s = state(page)
                pt = (s["root"][0] + 2.0 * math.sin(s.get("yaw") or 0),
                      s["root"][1] + 2.0 * math.cos(s.get("yaw") or 0))
            before = (s.get("guide") or {}).get("replans")
            page.evaluate("([i, x, z]) => window.ardyDemo.moveBox(i, x, z, 0.8)",
                          [int(spec.get("block_slot", 0)), pt[0], pt[1]])
            tb = time.time()
            detected = None
            while time.time() - tb < 25:
                sb = state(page)
                g = sb.get("guide") or {}
                if g.get("invalid") or (g.get("replans") or 0) > (before or 0):
                    detected = sb
                    break
                time.sleep(0.6)
            result["phases"]["block"] = {
                "box_at": list(pt), "replans_before": before,
                "detected": detected is not None,
                "seconds": round(time.time() - tb, 1),
                "replans": (detected or sb).get("guide", {}).get("replans"),
            }
            print(f"[route:{name}] block", json.dumps(result["phases"]["block"], ensure_ascii=False), flush=True)

        arrived, last, samples = walk_until_done(page, timeout_s, timeout_s + 30)
        result["phases"]["walk"] = {
            "arrived": arrived, "dist": last["dist"],
            "collision": last["collision"], "violation": last["violation"],
            "minSdf": last["minSdf"], "replans": (last.get("guide") or {}).get("replans"),
            "seconds": samples[-1]["t"] if samples else 0.0,
        }
        result["samples"] = samples
        result["console_errors"] = errors[-8:]

        expect = spec.get("expect") or {}
        ok = (arrived == expect.get("arrived", arrived)
              and (not last["collision"]) == (not expect.get("collision", False))
              and (not last["violation"]) == (not expect.get("violation", False)))
        result["pass"] = bool(ok)
        print(f"[route:{name}] {'PASS' if ok else 'FAIL'}",
              json.dumps(result["phases"]["walk"], ensure_ascii=False), flush=True)

        OUT.mkdir(parents=True, exist_ok=True)
        try:
            page.screenshot(path=str(OUT / f"{name}.png"))
        except Exception:  # noqa: BLE001
            pass
        page.evaluate("() => window.ardyDemo.reset()")
        browser.close()

    out = Path(args.out) if args.out else OUT / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), "utf-8")
    print(f"[route:{name}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
