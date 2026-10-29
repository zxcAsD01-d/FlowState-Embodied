"""Retarget neck-chain experiment: localise where the +56 deg chest->neck
pitch error is introduced, using synthetic Core27 poses applied live.
"""
from __future__ import annotations

import json
import math
import time

from playwright.sync_api import sync_playwright

JS = r"""
async () => {
  const THREE = await import('three');
  const d = window.ardyDemo;
  const avatar = d.avatar();
  const ret = d.retarget();
  const ident = () => Array.from({length: 27}, () => [0, 0, 0, 1]);

  const NODES = ['hips','spine','chest','neck','head'];
  const IDX = {Hips:0, Spine:1, Spine1:2, Spine2:3, Spine3:4, Neck:5, Head:6};
  const REST = {
    Hips:[0,0,0], Spine:[0,0.07099,-0.04733], Spine1:[0,0.1642,-0.06376],
    Spine2:[0,0.2585,-0.07201], Spine3:[0,0.35315,-0.07201],
    Neck:[0,0.60161,-0.03652], Head:[0,0.72978,-0.01392]};
  const P = {Spine:'Hips', Spine1:'Spine', Spine2:'Spine1', Spine3:'Spine2',
             Neck:'Spine3', Head:'Neck'};

  function measure(tag) {
    avatar.root.updateMatrixWorld(true);
    const pos = {};
    for (const n of NODES) {
      const node = avatar.getNode(n);
      if (node) pos[n] = node.getWorldPosition(new THREE.Vector3()).toArray();
    }
    const out = {tag, pos};
    return out;
  }

  const savedPos = avatar.root.position.clone();
  const savedRot = avatar.root.rotation.clone();
  avatar.root.position.set(0, 0, 0);
  avatar.root.rotation.set(0, 0, 0);
  const results = [];

  // 1. full identity Core27 -> T-pose
  avatar.applyPose(ret.apply(ident()));
  avatar.update(0);
  results.push(measure('tpose'));

  // 2. identity + Neck local pitched forward 45 deg about +X
  const q = ident();
  const h = Math.PI / 4;
  q[IDX.Neck] = [Math.sin(h / 2), 0, 0, Math.cos(h / 2)];
  avatar.applyPose(ret.apply(q));
  avatar.update(0);
  results.push(measure('neck45'));

  // 3. identity + Spine3 pitched forward 45 deg
  const q2 = ident();
  q2[IDX.Spine3] = [Math.sin(h / 2), 0, 0, Math.cos(h / 2)];
  avatar.applyPose(ret.apply(q2));
  avatar.update(0);
  results.push(measure('spine3_45'));

  // 4. identity + Spine and Spine1 pitched forward 20 deg each
  const q3 = ident();
  const h2 = Math.PI / 9;
  q3[IDX.Spine] = [Math.sin(h2 / 2), 0, 0, Math.cos(h2 / 2)];
  q3[IDX.Spine1] = [Math.sin(h2 / 2), 0, 0, Math.cos(h2 / 2)];
  avatar.applyPose(ret.apply(q3));
  avatar.update(0);
  results.push(measure('spine12_20'));

  // raw upperChest inventory
  let bones = null;
  try {
    const raw = avatar.vrm.humanoid.rawHumanoid;
    bones = Object.keys(raw.humanBones);
  } catch (e) { bones = ['ERR ' + e.message]; }

  avatar.root.position.copy(savedPos);
  avatar.root.rotation.copy(savedRot);
  avatar.update(0);
  return { results, rawBones: bones,
           dir: avatar.direction, yawOffset: avatar.yawOffset };
}
"""


def pitch_between(a, b, fwd_sign):
    v = [b[i] - a[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in v)) or 1e-9
    # sagittal plane: use horizontal magnitude along the skeleton's own facing.
    # For these experiments the root yaw is 0 and the VRM faces ±z.
    h = fwd_sign * v[2]
    return math.degrees(math.atan2(h, v[1]))


def core_pitch(ca, cb):
    a, b = {'Hips':[0,0,0],'Spine':[0,0.07099,-0.04733],'Spine1':[0,0.1642,-0.06376],
            'Spine2':[0,0.2585,-0.07201],'Spine3':[0,0.35315,-0.07201],
            'Neck':[0,0.60161,-0.03652],'Head':[0,0.72978,-0.01392]}[ca], \
           {'Hips':[0,0,0],'Spine':[0,0.07099,-0.04733],'Spine1':[0,0.1642,-0.06376],
            'Spine2':[0,0.2585,-0.07201],'Spine3':[0,0.35315,-0.07201],
            'Neck':[0,0.60161,-0.03652],'Head':[0,0.72978,-0.01392]}[cb]
    v = [b[i] - a[i] for i in range(3)]
    return math.degrees(math.atan2(v[2], v[1]))


def main():
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="msedge", headless=True,
                               args=["--use-angle=default"])
        p = b.new_page(viewport={"width": 1280, "height": 760})
        p.goto("http://127.0.0.1:8830/", wait_until="domcontentloaded")
        p.wait_for_function(
            "() => window.ardyDemo && window.ardyDemo.state().ready", timeout=90_000)
        out = p.evaluate(JS)
        time.sleep(0.2)

    fwd = out.get("dir") or 1
    print("avatar.direction =", fwd, " yawOffset =", out.get("yawOffset"))
    print("raw human bones:", out.get("rawBones"))
    core_segs = [('hips','spine','Hips','Spine'), ('spine','chest','Spine','Spine1'),
                 ('chest','neck','Spine3','Neck'), ('neck','head','Neck','Head')]
    for res in out["results"]:
        pos = res["pos"]
        print(f"\n[{res['tag']}]")
        for a, bnode, ca, cb in core_segs:
            if a not in pos or bnode not in pos:
                print(f"  {a}->{bnode}: MISSING")
                continue
            vrm_p = pitch_between(pos[a], pos[bnode], fwd)
            print(f"  {a}->{bnode}: vrm {vrm_p:+7.1f}  coreRest {core_pitch(ca,cb):+6.1f}"
                  f"  diff {vrm_p - core_pitch(ca,cb):+7.1f}")
    b.close()


if __name__ == "__main__":
    main()
