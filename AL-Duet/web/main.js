import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadAvatar, fitLegs, buildRetarget, updateFpCamera } from '/avatar.js';
import { getSceneFactory } from '/scenes/index.js';

const $ = (id) => document.getElementById(id);
const canvas = $('stage');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
renderer.setSize(window.innerWidth, window.innerHeight, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.06;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color('#080d16');
scene.fog = new THREE.Fog('#080d16', 22, 120);
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.05, 260);
const capCamera = new THREE.PerspectiveCamera(72, 16 / 9, 0.08, 220);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enablePan = false;
controls.minDistance = 1.8;
controls.maxDistance = 26;
controls.maxPolarAngle = Math.PI * 0.495;

const LIVE = new URLSearchParams(location.search).has('live');
document.body.dataset.mode = LIVE ? 'live' : 'play';

const S = {
  ready: false, playing: true, baseTime: 0, baseAt: performance.now(),
  duration: 0, fps: 20, episode: null, timeline: null, timelineIdx: -1,
  rigs: new Map(),           // key -> rig {adapter, retarget, ground, frames, label, yaw}
  sceneManifest: null, sceneModule: null, lastHud: 0,
  frameCount: 0, fpsAt: performance.now(), fpsValue: 0, follow: true,
  // live mode
  live: { actors: new Map(), clockStart: 0, pollTimer: null, offsets: new Map() },
  captureRT: null, captureCanvas: null,
};

function setBoot(msg, error = false) {
  const boot = $('boot');
  boot.classList.remove('hidden');
  $('bootMsg').textContent = msg;
  if (error) $('bootMsg').style.color = '#ff9a9a';
}
function hideBoot() { $('boot').classList.add('hidden'); }

function applyEnvironment(manifest) {
  const env = manifest.environment || {};
  if (env.background) scene.background = new THREE.Color(env.background);
  if (env.fog) scene.fog = new THREE.Fog(env.fog.color || '#080d16',
    Number(env.fog.near ?? 22), Number(env.fog.far ?? 120));
  const cam = manifest.camera || {};
  return {
    dist: Number(cam.distance ?? 5.4),
    yaw: THREE.MathUtils.degToRad(Number(cam.yaw_deg ?? 72)),
    pitch: THREE.MathUtils.degToRad(Number(cam.pitch_deg ?? 9)),
  };
}

async function loadScene(sceneId) {
  const id = sceneId || 'sakura_crossing';
  const manifest = await fetch(`/api/scene/${encodeURIComponent(id)}`).then(r => {
    if (!r.ok) throw new Error(`scene ${id} ${r.status}`);
    return r.json();
  });
  S.sceneManifest = manifest;
  const factory = await getSceneFactory(manifest.module || manifest.id);
  const module = factory({ THREE, manifest, scene, renderer, camera, fpCamera: capCamera });
  S.sceneModule = module;
  scene.add(module.root);
  const cam = applyEnvironment(manifest);
  $('infoScene').textContent = manifest.name || manifest.id;
  return cam;
}

function measureGround(adapter) {
  adapter.root.position.set(0, 0, 0);
  adapter.root.rotation.y = 0;
  adapter.resetPose();
  adapter.update(0);
  adapter.root.updateMatrixWorld(true);
  const m = adapter.groundMetrics();
  return { hipY: m.hipY, groundOffset: -m.hipY };
}

function makeRetarget(adapter, jointNames) {
  return buildRetarget(adapter, jointNames, {
    pelvisMatch: false,
    restBones: ['spine', 'leftFoot', 'rightFoot'],
  });
}

async function loadAvatarManifest() {
  return fetch('/api/avatars/default').then(r => r.json());
}

async function ensureRig(key, label, names, man) {
  if (S.rigs.has(key)) return S.rigs.get(key);
  // Concurrent polls must not load the same avatar twice (the polling timer
  // and the initial poll can race before the first rig lands in S.rigs).
  S.pendingRigs = S.pendingRigs || new Map();
  if (S.pendingRigs.has(key)) return S.pendingRigs.get(key);
  const p = (async () => {
    const adapter = await loadAvatar(man);
    scene.add(adapter.root);
    const retarget = makeRetarget(adapter, names);
    fitLegs(adapter, retarget);
    const ground = measureGround(adapter);
    adapter.root.visible = true;
    const rig = { key, adapter, retarget, ground, frames: [], label: label || key, yaw: 0 };
    S.rigs.set(key, rig);
    buildHud(key, rig.label);
    return rig;
  })();
  S.pendingRigs.set(key, p);
  try {
    return await p;
  } finally {
    S.pendingRigs.delete(key);
  }
}

// ------------------------------------------------------------------ pose
function samplePose(rig, t) {
  const frames = rig.frames;
  const n = frames.length;
  if (!n) return null;
  const x = Math.max(0, Math.min(t * S.fps, n - 1));
  const i = Math.floor(x);
  const j = Math.min(i + 1, n - 1);
  const a = Math.min(Math.max(x - i, 0), 1);
  const A = frames[i], B = frames[j];
  const quats = new Array(27);
  const qa = new THREE.Quaternion(), qb = new THREE.Quaternion(), qo = new THREE.Quaternion();
  for (let k = 0; k < 27; k++) {
    qa.fromArray(A.rot[k]); qb.fromArray(B.rot[k]);
    qo.copy(qa).slerp(qb, a);
    quats[k] = qo.toArray();
  }
  rig.adapter.applyPose(rig.retarget.apply(quats));
  const rx = A.root[0] + (B.root[0] - A.root[0]) * a;
  const ry = A.root[1] + (B.root[1] - A.root[1]) * a;
  const rz = A.root[2] + (B.root[2] - A.root[2]) * a;
  rig.adapter.root.position.set(rx, ry + rig.ground.groundOffset, rz);
  rig.adapter.root.rotation.y = rig.adapter.yawOffset || 0;
  rig.yaw = B.pose[2];
  rig.adapter.update(1 / 60);
  rig.adapter.root.updateMatrixWorld(true);
  return rig.adapter.root.position;
}

function groupTarget() {
  let x = 0, z = 0, n = 0;
  for (const rig of S.rigs.values()) {
    const p = rig.adapter.root.position;
    if (p.lengthSq() > 0) { x += p.x; z += p.z; n++; }
  }
  if (!n) return new THREE.Vector3(0, 1.03, 0);
  return new THREE.Vector3(x / n, 1.03, z / n);
}

function clampCam() {
  // keep the camera out of the building blocks flanking the street: an
  // inside-building camera renders a solid-black interior
  const LIM = { x: 5.4, z: 4.4 };
  camera.position.x = Math.max(-LIM.x, Math.min(LIM.x, camera.position.x));
  camera.position.z = Math.max(-LIM.z, Math.min(LIM.z, camera.position.z));
}

function frameCamera(cam) {
  if (!S.rigs.size) return;
  const target = groupTarget();
  let rmax = 1.2;
  for (const rig of S.rigs.values()) {
    rmax = Math.max(rmax, rig.adapter.root.position.distanceTo(target) + 0.8);
  }
  const dist = Math.min(6.0, Math.max(cam?.dist ?? 4.6, rmax * 1.7));
  const yaw = cam?.yaw ?? THREE.MathUtils.degToRad(72);
  const pitch = cam?.pitch ?? THREE.MathUtils.degToRad(9);
  camera.position.set(
    target.x + Math.sin(yaw) * Math.cos(pitch) * dist,
    target.y + Math.sin(pitch) * dist,
    target.z + Math.cos(yaw) * Math.cos(pitch) * dist,
  );
  clampCam();
  controls.target.copy(target);
  camera.lookAt(target);
  controls.update();
}

function updateCamera() {
  if (!S.follow || !S.rigs.size) { controls.update(); return; }
  const target = groupTarget();
  const delta = target.clone().sub(controls.target);
  if (delta.lengthSq() > 1e-8) {
    controls.target.copy(target);
    camera.position.add(delta);
    clampCam();
  }
  controls.update();
}

// -------------------------------------------------------------------- HUD
function buildHud(key, label) {
  const hud = $('duetHud');
  if ($(`hudCard${key}`)) return;
  const card = document.createElement('div');
  card.className = 'duet-card';
  card.id = `hudCard${key}`;
  card.innerHTML = `
    <div class="duet-head"><span>${key} · ${label}</span><b id="hud${key}State">—</b></div>
    <img id="hud${key}Img" alt="${key} 第一人称" style="display:none" />
    <div class="duet-meta" id="hud${key}Meta">…</div>`;
  hud.appendChild(card);
}

function setHud(key, d) {
  const st = $(`hud${key}State`);
  if (!st || !d) return;
  st.textContent = d.state || '—';
  const bits = [];
  if (d.emotion) bits.push(d.emotion);
  if (d.speech) bits.push(`「${d.speech}」`);
  if (Number.isFinite(d.distance)) bits.push(`${Number(d.distance).toFixed(2)} m`);
  bits.push(d.visible ? '看见对方' : '视野丢失');
  if (d.mode && d.mode !== 'geo') bits.push(d.mode);
  else if (Number.isFinite(d.conf)) bits.push(`conf ${Number(d.conf).toFixed(2)}`);
  $(`hud${key}Meta`).textContent = bits.join(' · ');
}

function updateHudPlay() {
  const list = S.timeline;
  if (!list || !list.length) return;
  const t = currentTime();
  while (S.timelineIdx + 1 < list.length && list[S.timelineIdx + 1].t <= t + 1e-3) {
    S.timelineIdx += 1;
  }
  const e = list[Math.max(0, S.timelineIdx)];
  if (!e) return;
  for (const [key] of S.rigs) setHud(key, e[key.toLowerCase()]);
}

// ------------------------------------------------------------------ clock
function currentTime() {
  if (LIVE) {
    return S.live.playReady ? S.live.playT : 0;
  }
  if (!S.duration) return 0;
  const raw = S.playing
    ? S.baseTime + (performance.now() - S.baseAt) / 1000
    : S.baseTime;
  return Math.max(0, Math.min(raw, S.duration));
}

// Live playhead: advance in real time but stay anchored to the FRAME
// timestamps (never run past the buffer, never jump).  This is what keeps
// live playback smooth when the engine's tick jitter delivers frames in
// bursts -- without it every poll shows the newest frame and the body
// teleports 0.4-0.8 s worth of motion at once.
function advanceLiveClock(dt) {
  let maxT = 0;
  for (const rig of S.rigs.values()) {
    const n = rig.frames.length;
    if (n) maxT = Math.max(maxT, (rig.t0 || 0) + n / S.fps);
  }
  if (!maxT) return;
  const target = Math.max(0, maxT - 0.2);
  if (!S.live.playReady) { S.live.playT = target; S.live.playReady = true; return; }
  if (target < S.live.playT - 0.5) { S.live.playT = target; return; } // buffer reset
  const rate = (S.live.playT < target - 1.0) ? 1.5 : 1.0;             // gentle catch-up
  S.live.playT = Math.min(S.live.playT + dt * rate, Math.max(S.live.playT, target));
}

function setPlaying(on) {
  const now = performance.now();
  const t = currentTime();
  if (on) {
    S.baseTime = (!LIVE && t >= S.duration - 1e-4) ? 0 : t;
    S.baseAt = now;
  } else {
    S.baseTime = t;
  }
  S.playing = on;
  $('playBtn').textContent = on ? '暂停' : '播放';
}

function seek(t) {
  S.baseTime = Math.max(0, Math.min(Number(t) || 0, S.duration));
  S.baseAt = performance.now();
  S.timelineIdx = -1;
}

function updatePlayerUi() {
  const t = currentTime();
  const dur = LIVE ? t : S.duration;
  const last = Math.max(0, Math.round(dur * S.fps) || 0);
  const frame = Math.min(Math.floor(t * S.fps), last);
  $('timeRead').textContent = `${t.toFixed(2)} / ${dur.toFixed(2)} s · 帧 ${frame}`;
  const seekEl = $('seek');
  if (document.activeElement !== seekEl && !LIVE) seekEl.value = String(t);
}

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

let last = performance.now();
function raf(now) {
  requestAnimationFrame(raf);
  const dt = Math.min(Math.max((now - last) / 1000, 0), 0.1);
  last = now;
  if (!S.ready) return;
  const dtLive = dt;
  if (LIVE) advanceLiveClock(dtLive);
  const t = currentTime();
  if (!LIVE && S.playing && t >= S.duration - 1e-4) setPlaying(false);
  for (const rig of S.rigs.values()) {
    const p = samplePose(rig, t - (rig.t0 || 0));
    if (p !== null) rig.lastPos = p;
  }
  updateCamera();
  S.sceneModule?.update?.(dt, {});
  if (now - S.lastHud >= 120) {
    S.lastHud = now;
    if (LIVE) updatePlayerUi(); else { updateHudPlay(); updatePlayerUi(); }
  }
  renderer.render(scene, camera);
  S.frameCount++;
  if (now - S.fpsAt >= 700) {
    S.fpsValue = Math.round(S.frameCount * 1000 / (now - S.fpsAt));
    S.frameCount = 0; S.fpsAt = now;
    $('fpsRead').textContent = `${S.fpsValue} fps`;
  }
}

// ------------------------------------------------------- egocentric capture
function captureActor(key, width = 640) {
  const rig = S.rigs.get(key);
  if (!rig || !rig.frames.length) return null;
  const w = Math.max(160, Math.min(1024, Math.round(Number(width) || 640)));
  const h = Math.round(w / 1.7778);
  capCamera.aspect = w / h;
  capCamera.updateProjectionMatrix();
  const eye = updateFpCamera(capCamera, rig.adapter, rig.yaw, true, 0, null);
  if (!eye) return null;
  capCamera.updateMatrixWorld(true);
  if (!S.captureRT) {
    S.captureRT = new THREE.WebGLRenderTarget(w, h, {
      depthBuffer: true, stencilBuffer: false });
    S.captureRT.texture.colorSpace = THREE.SRGBColorSpace;
  } else if (S.captureRT.width !== w || S.captureRT.height !== h) {
    S.captureRT.setSize(w, h);
  }
  const wasVisible = rig.adapter.root.visible;
  rig.adapter.root.visible = false;
  const prevTarget = renderer.getRenderTarget();
  const prevViewport = renderer.getViewport(new THREE.Vector4());
  renderer.setRenderTarget(S.captureRT);
  renderer.setViewport(0, 0, w, h);
  renderer.setScissorTest(false);
  renderer.clear();
  renderer.render(scene, capCamera);
  const buf = new Uint8Array(w * h * 4);
  renderer.readRenderTargetPixels(S.captureRT, 0, 0, w, h, buf);
  renderer.setRenderTarget(prevTarget);
  renderer.setViewport(prevViewport);
  rig.adapter.root.visible = wasVisible;

  // 3x3 depth rays against the other bodies (same rendered geometry).
  const depthSamples = [];
  const rc = new THREE.Raycaster();
  for (const [otherKey, other] of S.rigs) {
    if (otherKey === key || !other.adapter.root.visible) continue;
    for (const ny of [0.20, 0.0, -0.20]) {
      for (const nx of [-0.10, 0.0, 0.10]) {
        rc.setFromCamera(new THREE.Vector2(nx, ny), capCamera);
        const hits = rc.intersectObject(other.adapter.root, true);
        if (hits.length) {
          depthSamples.push({ x: nx, y: ny, d: Number(hits[0].distance.toFixed(4)) });
        }
      }
    }
  }
  depthSamples.sort((a, b) => a.d - b.d);

  if (!S.captureCanvas) S.captureCanvas = document.createElement('canvas');
  const cv = S.captureCanvas;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext('2d');
  const imgData = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    const src = (h - 1 - y) * w * 4;
    imgData.data.set(buf.subarray(src, src + w * 4), y * w * 4);
  }
  ctx.putImageData(imgData, 0, 0);
  return {
    image: cv.toDataURL('image/jpeg', 0.84),
    width: w, height: h, actor: key,
    depth_samples: depthSamples,
    pose: eye,
  };
}

function captureAll(width) {
  const actors = {};
  for (const [key] of S.rigs) actors[key] = captureActor(key, width);
  return { actors, roots: Object.fromEntries(
    [...S.rigs].map(([k, r]) => [k, r.adapter.root.position.toArray()])) };
}

// ------------------------------------------------------------------- boot
async function bootPlay() {
  const ep = await fetch('/data/episode.json').then(r => {
    if (!r.ok) throw new Error(`episode.json ${r.status}`);
    return r.json();
  });
  S.episode = ep;
  S.fps = Number(ep.fps) || 20;
  const man = await loadAvatarManifest();
  const cam = await loadScene(ep.scene);
  for (const [key, act] of Object.entries(ep.actors)) {
    const rig = await ensureRig(key, act.label || key, ep.joint_names, man);
    rig.frames = act.frames;
  }
  const maxFrames = Math.max(...[...S.rigs.values()].map(r => r.frames.length));
  S.duration = maxFrames / S.fps;
  S.timeline = Array.isArray(ep.timeline) ? ep.timeline : null;
  $('infoDuration').textContent = `${S.duration.toFixed(1)} s · ${maxFrames} 帧`;
  $('infoSource').textContent = 'episode.json（推理后播放）';
  $('seek').max = String(S.duration);
  frameCamera(cam);
  S.ready = true;
  setPlaying(true);
  hideBoot();
}

async function pollLive() {
  if (S.live.polling) return;   // overlapping polls duplicate frames
  S.live.polling = true;
  try {
    await pollLiveInner();
  } finally {
    S.live.polling = false;
  }
}

async function pollLiveInner() {
  try {
    const { actors } = await fetch('/api/live/actors').then(r => r.json());
    if (!actors.length) return;
    const man = S.live.manifest;
    for (const a of actors) {
      if (!S.rigs.has(a.key)) {
        setBoot(`接入实时流 ${a.key}…`);
        await ensureRig(a.key, a.title || a.key, a.joint_names, man);
        if (!S.live.clockStart && S.rigs.size === actors.length) {
          S.live.clockStart = performance.now();
          S.ready = true;
          hideBoot();
          frameCamera({});
        }
        $('infoSource').textContent = 'live（实时推理）';
      }
      const off = S.live.offsets.get(a.key) || 0;
      const res = await fetch(`/api/live/frames?actor=${encodeURIComponent(a.key)}&offset=${off}`)
        .then(r => r.json());
      if (res.total < off) {
        // backend restarted / buffer trimmed: re-align from zero
        S.live.offsets.set(a.key, 0);
        const rig0 = S.rigs.get(a.key);
        if (rig0) { rig0.frames = []; rig0.t0 = 0; }
        continue;
      }
      if (res.frames?.length) {
        const rig = S.rigs.get(a.key);
        if (rig && !rig.t0) rig.t0 = res.frames[0].t;  // late-join alignment
        rig.frames.push(...res.frames);
        S.live.offsets.set(a.key, res.total);
      }
    }
    try {
      const { hud } = await fetch('/api/live/hud').then(r => r.json());
      for (const [key, d] of Object.entries(hud || {})) setHud(key, d);
    } catch (err) { /* hud is optional */ }
  } catch (err) { /* server restarting — keep polling */ }
}

async function bootLive() {
  S.live.manifest = await loadAvatarManifest();
  await loadScene(null);
  $('infoDuration').textContent = '实时';
  S.live.pollTimer = setInterval(pollLive, 400);
  pollLive();
}

window.addEventListener('resize', resize);
window.addEventListener('webglcontextlost', (e) => { e.preventDefault(); });
$('playBtn').addEventListener('click', () => { if (!LIVE) setPlaying(!S.playing); });
$('restartBtn').addEventListener('click', () => { if (!LIVE) { seek(0); setPlaying(true); } });
$('seek').addEventListener('input', (e) => { if (!LIVE) seek(Number(e.target.value)); });
$('followChk').addEventListener('change', (e) => { S.follow = e.target.checked; });

window.duet = {
  get ready() { return S.ready; },
  get mode() { return LIVE ? 'live' : 'play'; },
  get rigs() { return [...S.rigs.keys()]; },
  capture: captureAll,
  frameCamera: () => frameCamera({}),
};

// QC hook: rendered playhead + per-rig world positions (continuity checks)
window.__duetState = () => ({
  t: LIVE && S.live.playReady ? S.live.playT : 0,
  pos: Object.fromEntries([...S.rigs].map(([k, r]) => {
    const p = r.adapter.root.position;
    return [k, [Number(p.x.toFixed(4)), Number(p.z.toFixed(4))]];
  })),
  cam: camera.position.toArray().map(v => Number(v.toFixed(2))),
  tgt: controls.target.toArray().map(v => Number(v.toFixed(2))),
  sceneKids: scene.children.length,
});

(async () => {
  try {
    setBoot(LIVE ? '等待实时推理流…' : '载入 episode…');
    if (LIVE) await bootLive(); else await bootPlay();
  } catch (err) {
    console.error(err);
    setBoot(`启动失败：${err.message}`, true);
  }
})();
raf(performance.now());
