import argparse, json, math, time
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT=Path('runs')
def state(p): return p.evaluate("() => window.ardyDemo.state()")
def slim(s):
    g=s.get('guide') or {}
    return {'t':None,'root':s.get('root'),'dist':s.get('dist'),'status':s.get('status'),
            'stop':s.get('stopped'),'collision':s.get('collision'),'violation':s.get('violation'),
            'minSdf':s.get('minSdf'),'replans':g.get('replans'),'invalid':g.get('invalid'),
            'replanned':g.get('replanned'),'streak':g.get('invalid_streak'),
            'reason':g.get('reason'),'remaining':g.get('route_remaining'),
            'rmsdf':g.get('route_min_sdf'),'rbad':g.get('route_bad_i'),
            'occ':g.get('occupied_cells'),'route':s.get('route')}
def route_ahead(route,root,dist):
    if not route or len(route)<2: return None
    best=(1e9,0,0.0)
    for i,(a,b) in enumerate(zip(route[:-1],route[1:])):
        ax,az=a; bx,bz=b; abx,abz=bx-ax,bz-az; L2=abx*abx+abz*abz
        t=0.0 if L2<1e-12 else max(0.,min(1.,((root[0]-ax)*abx+(root[1]-az)*abz)/L2))
        qx,qz=ax+abx*t,az+abz*t; d=math.hypot(root[0]-qx,root[1]-qz)
        if d<best[0]: best=(d,i,t)
    _,i,t=best; a,b=route[i],route[i+1]; sx,sz=a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t
    rem=dist
    for j in range(i,len(route)-1):
        a,b=route[j],route[j+1]
        if j==i: a=(sx,sz)
        seg=math.hypot(b[0]-a[0],b[1]-a[1])
        if seg<=1e-9: continue
        if rem<=seg:
            u=rem/seg; return (a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u)
        rem-=seg
    return tuple(route[-1])
ap=argparse.ArgumentParser(); ap.add_argument('--max',type=float,default=70.0); args=ap.parse_args()
with sync_playwright() as pw:
    b=pw.chromium.launch(channel='msedge',headless=True,args=['--use-angle=default'])
    p=b.new_page(viewport={'width':1280,'height':760})
    p.goto('http://127.0.0.1:8830/',wait_until='domcontentloaded')
    p.wait_for_function("() => window.ardyDemo && window.ardyDemo.state().ready && window.ardyDemo.state().frames>0",timeout=90000)
    p.evaluate("() => window.ardyDemo.reset()"); time.sleep(4.0)
    s=state(p); pt=route_ahead(s.get('route'),s['root'],2.5)
    tend=time.time()+12
    while pt is not None and time.time()<tend:
        dx,dz=pt[0]-s['root'][0],pt[1]-s['root'][1]
        ang=abs((math.atan2(dx,dz)-(s.get('yaw') or 0)+math.pi)%(2*math.pi)-math.pi)
        if ang<0.55 and math.hypot(dx,dz)>1.6: break
        time.sleep(1.0); s=state(p); pt=route_ahead(s.get('route'),s['root'],2.5)
    if pt is None:
        pt=(s['root'][0]+2.0*math.sin(s.get('yaw') or 0),s['root'][1]+2.0*math.cos(s.get('yaw') or 0))
    before=(s.get('guide') or {}).get('replans')
    p.evaluate("([x,z]) => window.ardyDemo.moveBox(0,x,z,0.8)",[pt[0],pt[1]])
    out={'box_at':list(pt),'replans_before':before,'samples':[]}
    print('place',pt,'before',before,flush=True)
    t0=time.time(); stop_seen=False; replan_seen=False; arrived=False
    while time.time()-t0<args.max:
        s=state(p); d=slim(s); d['t']=round(time.time()-t0,2); out['samples'].append(d)
        g=s.get('guide') or {}
        if s.get('stopped') and not s.get('stopped') is None: stop_seen = stop_seen or bool(g.get('invalid'))
        if (g.get('replans') or 0)>(before or 0): replan_seen=True
        if s['dist'] is not None and s['dist']<0.65: arrived=True; break
        if s['collision'] or s['violation']: break
        time.sleep(0.45)
    out.update({'stop_seen':stop_seen,'replan_seen':replan_seen,'arrived':arrived,
                'collision':s['collision'],'violation':s['violation'],'dist':s['dist'],
                'replans':(s.get('guide') or {}).get('replans'),
                'seconds':round(time.time()-t0,1),'nav':s.get('nav')})
    print(json.dumps({k:v for k,v in out.items() if k!='samples'},ensure_ascii=False),flush=True)
    for d in out['samples']:
        rr=d['route']
        print(d['t'],'d=%.2f'%(d['dist'] or -1),d['status'],'stop',d['stop'],'repl',d['replans'],'inv',d['invalid'],'st',d['streak'],d['reason'],
              'msdf',None if d['rmsdf'] is None else round(d['rmsdf'],2),'bad',d['rbad'],'occ',d['occ'],
              'root',(round(d['root'][0],1),round(d['root'][1],1)),'col',d['collision'])
    p.screenshot(path='runs/nav_block_final.png')
    b.close()
Path('runs/qc_nav_block.json').write_text(json.dumps(out,ensure_ascii=False,indent=1),'utf-8')
