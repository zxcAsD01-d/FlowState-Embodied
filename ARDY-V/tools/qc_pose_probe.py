"""Live pose probe: quantify hunching during real closed-loop walking.

Samples window.ardyDemo (VRM world bones + the raw ARDY Core27 frame currently
on screen) while the avatar walks, and reports:
  - VRM spine forward lean (hips->spine/chest/neck/head vs vertical, in yaw frame)
  - knee bend, foot clearance, root bob, walk speed
  - the raw Core27 quats saved for an offline model-vs-retarget ablation
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs"


def main():
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="msedge", headless=True,
                               args=["--use-angle=default"])
        p = b.new_page(viewport={"width": 1280, "height": 760})
        p.goto("http://127.0.0.1:8830/", wait_until="domcontentloaded")
        p.wait_for_function(
            "() => window.ardyDemo && window.ardyDemo.state().ready"
            " && window.ardyDemo.state().frames > 0", timeout=90_000)
        p.evaluate("() => window.ardyDemo.reset()")
        time.sleep(3.0)

        samples = []
        t0 = time.time()
        prev_hips = None
        while time.time() - t0 < 10.0:
            s = p.evaluate(
                "() => { const st = window.ardyDemo.state();"
                " const b = window.ardyDemo.bones();"
                " return {root: st.root, yaw: st.yaw, dist: st.dist,"
                " status: st.status, rawLast: st.rawLast, bones: b}; }")
            bo = s["bones"]
            row = {"t": round(time.time() - t0, 2),
                   "dist": s.get("dist"), "yaw": s.get("yaw")}
            if bo and bo.get("hips"):
                hips = bo["hips"]
                yaw = s.get("yaw") or 0.0
                fwd = (math.sin(yaw), math.cos(yaw))

                def lean(a, bb):
                    v = [bb[i] - hips[i] for i in range(3)]
                    h = v[0] * fwd[0] + v[2] * fwd[1]
                    return round(math.degrees(math.atan2(h, v[1])), 1)

                row["hipsY"] = round(hips[1], 3)
                if bo.get("spine"):
                    row["leanSpine"] = lean(hips, bo["spine"])
                if bo.get("chest"):
                    row["leanChest"] = lean(hips, bo["chest"])
                if bo.get("neck"):
                    row["leanNeck"] = lean(hips, bo["neck"])
                if bo.get("head"):
                    row["leanHead"] = lean(hips, bo["head"])
                fy = min(bo["rightFoot"][1], bo["leftFoot"][1]) \
                    if bo.get("rightFoot") and bo.get("leftFoot") else None
                row["footMinY"] = round(fy, 3) if fy is not None else None
                if prev_hips is not None:
                    row["speed"] = round(
                        math.hypot(hips[0] - prev_hips[0],
                                   hips[2] - prev_hips[1]) / 0.25, 2)
                prev_hips = hips
                row["pos"] = {k: [round(c, 4) for c in bo[k]]
                              for k in ("hips", "spine", "chest", "neck", "head")
                              if bo.get(k)}
            if s.get("rawLast"):
                row["raw"] = {"root": s["rawLast"]["root"],
                              "yaw": s["rawLast"]["yaw"],
                              "rot": s["rawLast"]["rot"]}
            samples.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "raw"},
                             ensure_ascii=False), flush=True)
            time.sleep(0.25)

        (OUT / "pose_probe.json").write_text(
            json.dumps(samples, ensure_ascii=False), "utf-8")
        p.screenshot(path=str(OUT / "pose_probe_final.png"))
        b.close()
    print("saved", OUT / "pose_probe.json", flush=True)


if __name__ == "__main__":
    main()
