# -*- coding: utf-8 -*-
"""Closed-loop QC for the ARDY-v browser demo.

Requires the service and viewer already running at http://127.0.0.1:8830/.
Runs each target from the same spawn, waits for arrival/collision/timeout, and
compares the visual residual against the base-only path.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [("A", 5.0, 5.0), ("B", -5.0, 5.0), ("C", 5.0, -5.0), ("D", 0.0, 6.0)]


def wait_done(page, timeout=40.0):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        last = page.evaluate("window.ardyDemo.state()")
        st = last.get("status", "")
        if "已到达" in st or "碰撞" in st:
            return last, time.time() - t0
        time.sleep(0.5)
    return last, time.time() - t0


def run_one(page, x, z, visual=True, timeout=40.0):
    page.evaluate("window.ardyDemo.reset()")
    time.sleep(1.0)
    page.evaluate(f"document.getElementById('useVisual').checked = {str(bool(visual)).lower()}")
    page.evaluate(f"window.ardyDemo.setTarget({x}, {z})")
    state, dt = wait_done(page, timeout)
    return {"target": [x, z], "visual": visual, "elapsed_s": round(dt, 1), **state}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8830/")
    ap.add_argument("--out", default=str(ROOT / "runs" / "qc_demo.json"))
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="msedge", headless=not args.headed,
                               args=["--use-angle=default"])
        p = b.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
        p.goto(args.url, wait_until="load")
        p.wait_for_function("window.ardyDemo !== undefined", timeout=120000)
        # Wait until the viewer has booted and the first window is buffered.
        for _ in range(120):
            st = p.evaluate("window.ardyDemo.state()")
            if st.get("ready") and st.get("frames", 0) > 0:
                break
            time.sleep(1)
        rows = []
        for name, x, z in TARGETS:
            st = run_one(p, x, z, visual=True)
            st["name"] = name
            rows.append(st)
            print(name, st, flush=True)
            p.screenshot(path=str(ROOT / "runs" / f"qc_{name}.png"))
        base = run_one(p, 5.0, 5.0, visual=False)
        base["name"] = "A_base_only"
        rows.append(base)
        print("base", base, flush=True)
        p.screenshot(path=str(ROOT / "runs" / "qc_A_base.png"))
        b.close()
    out = {"rows": rows,
           "summary": {
               "arrived": sum("已到达" in r.get("status", "") for r in rows),
               "collided": sum("碰撞" in r.get("status", "") for r in rows),
               "n": len(rows),
           }}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), "utf-8")
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
