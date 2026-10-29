import json, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from ardy_v.arena import demo_arena
from ardy_v.nav_runtime import NavEncoder
from ardy_v.occupancy_map import WorldMap
from ardy_v.data import IMG_W, IMG_H

frames=json.load(open('runs/live_depth_frames.json'))
arena=demo_arena(); enc=NavEncoder('checkpoints/nav.pt')
cache=[]
for f in frames:
    pose=f['eyePose']
    im=Image.open(Path('runs')/f['file']).convert('RGB').resize((IMG_W,IMG_H),Image.BILINEAR)
    d,m=enc.predict(np.asarray(im,np.uint8))
    cache.append((d,m,np.array(pose['root'],float),float(pose['yaw']),float(pose['pitch'])))

half=12.0; res=.1
xs=np.arange(-half+res/2,half,res); zs=xs
X,Z=np.meshgrid(xs,zs)
free=np.ones_like(X,bool)
for b in arena.boxes:
    c,s=np.cos(b.yaw),np.sin(b.yaw); dx=X-b.x; dz=Z-b.z
    lx=c*dx+s*dz; lz=-s*dx+c*dz
    free &= ~((np.abs(lx)<b.hx+0.5)&(np.abs(lz)<b.hz+0.5))
free &= (np.abs(X)<half-0.9)&(np.abs(Z)<half-0.9)
near=np.zeros_like(free)
for f in frames:
    rx,rz=f['eyePose']['root']; near |= (X-rx)**2+(Z-rz)**2 < 6**2
free &= near
boxmsk=[]
for b in arena.boxes:
    c,s=np.cos(b.yaw),np.sin(b.yaw); dx=X-b.x; dz=Z-b.z
    lx=c*dx+s*dz; lz=-s*dx+c*dz
    boxmsk.append((np.abs(lx)<=b.hx)&(np.abs(lz)<=b.hz))

variants=[('h20_c50_fg',0.20,0.5,True),('h45_c50',0.45,0.5,False),('h60_c50',0.60,0.5,False),
          ('h45_c70',0.45,0.7,False),('h60_c70',0.60,0.7,False),('h75_c70',0.75,0.7,False)]
for name,mh,cf,fg in variants:
    m0=WorldMap(half=half,res=res)
    for d,m,root,yaw,pitch in cache:
        m0.update(d,root,yaw,pitch=pitch,mask=m,floor_guard=fg,min_h=mh,conf=cf)
    occ=m0.occupied(); n=int(occ.sum()); ff=int((occ&free).sum())
    ib=int(sum((occ&bm).sum() for bm in boxmsk))
    # box footprint at (0,0) specifically
    ib0=int((occ&boxmsk[0]).sum())
    print(f'{name:12s} occ {n:5d} false_free {ff:4d} in_boxes {ib:4d} central {ib0:4d} updates {m0.updates}')
    img=np.full((m0.n,m0.n,3),18,np.uint8)
    img[occ]=(0,0,220)
    for bm in boxmsk: img[bm]=(70,70,70)
    for f in frames:
        rx,rz=f['eyePose']['root']; ix=int((rx+half)/res); iz=int((rz+half)/res)
        img[max(0,iz-2):iz+3,max(0,ix-2):ix+3]=(255,180,0)
    cv2.imwrite(f'runs/map_{name}.png',cv2.resize(img,(720,720),interpolation=cv2.INTER_NEAREST))
