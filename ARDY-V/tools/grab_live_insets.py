import base64, json, time
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT = Path("runs")
with sync_playwright() as pw:
    b = pw.chromium.launch(channel="msedge", headless=True, args=["--use-angle=default"])
    p = b.new_page(viewport={"width": 1280, "height": 760})
    p.goto("http://127.0.0.1:8830/", wait_until="domcontentloaded")
    p.wait_for_function("() => window.ardyDemo && window.ardyDemo.state().ready && window.ardyDemo.state().frames>0", timeout=90000)
    p.evaluate("() => window.ardyDemo.reset()")
    for k in range(4):
        time.sleep(1.5)
        ims = p.evaluate("() => window.ardyDemo.images()")
        st = p.evaluate("() => window.ardyDemo.state()")
        for key in ("eye", "depth", "mask", "maskBin", "region"):
            u = ims.get(key)
            if u:
                (OUT / f"grab_{k}_{key}.jpg").write_bytes(base64.b64decode(u.split(",",1)[1]))
        print(k, st["root"], st["yaw"], (st.get("guide") or {}).get("mask_frac"), (st.get("guide") or {}).get("occupied_cells"))
    b.close()
