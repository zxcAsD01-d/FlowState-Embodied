"""Model-vs-retarget spine ablation (offline, uses runs/pose_probe.json).

The retarget solver promises: world segment direction of the avatar equals
R_model * coreRestDir.  This script computes the promised direction for every
spine segment from the raw Core27 quats and compares it with the actual VRM
node-to-node directions captured live in the browser.
"""
from __future__ import annotations

import json
import math

import numpy as np

REST = np.array([
 [0,0,0],[0,0.07099,-0.04733],[0,0.1642,-0.06376],[0,0.2585,-0.07201],
 [0,0.35315,-0.07201],[0,0.60161,-0.03652],[0,0.72978,-0.01392],
 [-0.03199,0.52592,-0.01869],[-0.1909,0.52592,-0.01869],[-0.48634,0.52592,-0.01869],
 [-0.71899,0.52592,-0.01869],[-0.7886,0.52592,-0.01869],[-0.74684,0.50736,0.02772],
 [0.03199,0.52592,-0.01869],[0.1909,0.52592,-0.01869],[0.48634,0.52592,-0.01869],
 [0.71899,0.52592,-0.01869],[0.7886,0.52592,-0.01869],[0.74684,0.50736,0.02772],
 [-0.09492,-0.02773,0],[-0.09492,-0.43985,0],[-0.09492,-0.89594,0],[-0.09492,-0.95441,0.16066],
 [0.09492,-0.02773,0],[0.09492,-0.43985,0],[0.09492,-0.89594,0],[0.09492,-0.95441,0.16066]])
PARENTS = [-1,0,1,2,3,4,5,4,7,8,9,10,10,4,13,14,15,16,16,0,19,20,21,0,23,24,25]
IDX = {n: i for i, n in enumerate(
    ['Hips','Spine','Spine1','Spine2','Spine3','Neck','Head'])}

SEGMENTS = [  # (vrm a, vrm b, core a, core b)
    ('hips','spine','Hips','Spine'),
    ('spine','chest','Spine','Spine1'),
    ('chest','neck','Spine3','Neck'),
    ('neck','head','Neck','Head'),
]


def qmat(q):
    x, y, z, w = q
    return np.array([
      [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def fk(rot):
    R = [qmat(rot[i]) for i in range(27)]
    pos = [REST[0].copy()]
    for i in range(1, 27):
        pos.append(pos[PARENTS[i]] + R[PARENTS[i]] @ (REST[i]-REST[PARENTS[i]]))
    return R, np.array(pos)


def pitch(v):
    return math.degrees(math.atan2(v[0], v[1]))  # sagittal in +z-up frame


samples = json.load(open('runs/pose_probe.json', encoding='utf-8'))
print(' t    segment      model_pitch  vrm_pitch  d_pitch   d_3d_deg')
agg = {}
for s in samples:
    raw = s.get('raw')
    if not raw:
        continue
    R, J = fk(raw['rot'])
    for a, b, ca, cb in SEGMENTS:
        va, vb = s.get('pos', {}).get(a), s.get('pos', {}).get(b)
        if va is None or vb is None:
            continue
        v_vrm = np.array(vb) - np.array(va)
        v_vrm = v_vrm / np.linalg.norm(v_vrm)
        # expected: R_model(world of core joint) @ coreRestDir
        i, j = IDX[ca], IDX[cb]
        d = REST[j] - REST[i]
        d = d / np.linalg.norm(d)
        v_exp = R[i] @ d
        ang3d = math.degrees(math.acos(
            float(np.clip(v_vrm @ v_exp, -1, 1))))
        dp = pitch(v_vrm) - pitch(v_exp)
        while dp > 180: dp -= 360
        while dp < -180: dp += 360
        agg.setdefault((a, b), []).append((s['t'], dp, ang3d))
        if abs(s['t'] - round(s['t'])) < 0.01 and round(s['t']*2) % 2 == 0:
            print('%5.1f  %-11s  %+8.1f  %+8.1f  %+7.1f  %7.1f' % (
                s['t'], f'{a}->{b}', pitch(v_exp), pitch(v_vrm), dp, ang3d))

print('\nsummary over walk:')
for k, v in agg.items():
    dps = np.array([x[1] for x in v])
    a3 = np.array([x[2] for x in v])
    print('  %-11s  d_pitch mean %+6.1f  p5 %+6.1f  p95 %+6.1f | 3D err mean %5.1f max %5.1f'
          % (f'{k[0]}->{k[1]}', dps.mean(), np.percentile(dps, 5),
             np.percentile(dps, 95), a3.mean(), a3.max()))
