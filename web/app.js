import * as THREE from 'three';
import { buildArena, makeTargetMarker } from './arena.js';
import { loadAvatar, buildRetarget, fitLegs, CORE_SKELETON } from './avatar.js';
import { createPhysics } from './physics.js';

const canvas = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({
  canvas, antialias: true, powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(52, window.innerWidth / window.innerHeight, 0.05, 120);
camera.layers.enableAll();
const eyeCam = new THREE.PerspectiveCamera(72, 320 / 180, 0.05, 80);
eyeCam.layers.set(0);
const eyeRT = new THREE.WebGLRenderTarget(320, 180, { depthBuffer: true, stencilBuffer: false });
eyeRT.texture.colorSpace = THREE.SRGBColorSpace;
const eyeCanvas = document.createElement('canvas');
eyeCanvas.width = 320;
eyeCanvas.height = 180;
const eyeCtx = eyeCanvas.getContext('2d');
const eyePixels = eyeCtx.createImageData(320, 180);
const clock = new THREE.Clock();

let arena = null;
let arenaSpec = null;
let physics = null;
let dragMode = false;
let targetMarker = null;
let avatar = null;
let retarget = null;
let frames = [];
let origin = [0, 0];
let target = new THREE.Vector2(5, 5);
let playT = 0;
let pending = false;
let stopCount = 0;
let ready = false;
let currentYaw = 0;
let lastResp = null;
let lastEyeUrl = null;
let lastEyePose = null;
let latEma = 0.3;           // seconds; EMA of window latency drives prefetch
let footDropSm = 0.05;      // slow ground drop so the stance foot touches
let eyeRel0 = null;         // head-vs-hips rest offset, calibrated on frame 1
const footProbe = new THREE.Vector3();
const _qHead = new THREE.Quaternion();
const _qHips = new THREE.Quaternion();
const _qRel = new THREE.Quaternion();
const _vFwd = new THREE.Vector3();
const _vEye = new THREE.Vector3();
const _vTmp = new THREE.Vector3();

// First-person anchor: eye at the real head bone (mid-eye when the VRM has
// eye bones), orientation as body heading plus the head's turn *relative to
// the hips*.  The posed head bone alone is too noisy (the retargeted neck
// chain swings tens of degrees), so the relative offset is rest-calibrated
// and clamped.
function headEyePose() {
  if (!avatar) return null;
  const head = avatar.getNode('head');
  const hips = avatar.getNode('hips');
  if (!head || !hips) return null;
  avatar.root.updateMatrixWorld(true);
  head.getWorldQuaternion(_qHead);
  hips.getWorldQuaternion(_qHips);
  const dir = avatar.direction || 1;
  _qRel.copy(_qHips).invert().multiply(_qHead);
  _vFwd.set(0, 0, dir).applyQuaternion(_qRel);
  const yawRel = Math.atan2(_vFwd.x, _vFwd.z);
  const pitchRel = Math.asin(clamp(_vFwd.y, -1, 1));
  const le = avatar.getNode('leftEye');
  const re = avatar.getNode('rightEye');
  if (le && re) {
    le.getWorldPosition(_vEye);
    re.getWorldPosition(_vTmp);
    _vEye.add(_vTmp).multiplyScalar(0.5);
  } else {
    head.getWorldPosition(_vEye);
    _vEye.y += 0.05;
  }
  return { eye: _vEye, yawRel, pitchRel };
}
let groundOffset = 0;
let viewerCollision = false;
let colFreeT = 0;

const pathPos = new Float32Array(8 * 3);
const pathGeo = new THREE.BufferGeometry();
pathGeo.setAttribute('position', new THREE.BufferAttribute(pathPos, 3));
const pathLine = new THREE.Line(pathGeo, new THREE.LineBasicMaterial({
  color: 0x2bff9a, transparent: true, opacity: 0.95,
}));
pathLine.frustumCulled = false;
pathLine.visible = false;
pathLine.layers.set(2);
scene.add(pathLine);

const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function setStatus(s) { $('status').textContent = s; }
function setMode(m) {
  $('mode').textContent = m;
  $('modeNote').textContent = m === 'visual' ? '在线占据 → 规划路线 → ARDY root'
    : '直走先验 · 无碰撞检查';
}

async function poll() {
  try {
    const h = await api('/api/health');
    if (h.ready) { await boot(); return; }
    if (h.error) { $('loadingText').textContent = '启动失败: ' + h.error; }
  } catch (e) {
    $('loadingText').textContent = '等待服务…';
  }
  setTimeout(poll, 700);
}

function setTargetMarker() {
  if (!targetMarker) return;
  targetMarker.position.set(target.x, 0, target.y);
}

async function boot() {
  try {
    const st = await api('/api/state');
    arena = buildArena(scene, st.spec);
    arenaSpec = st.spec;
    physics = createPhysics(scene, st.spec, arena.boxes, camera, canvas);
    targetMarker = makeTargetMarker(scene);
    target.set(st.target[0], st.target[1]);
    setTargetMarker();
    setMode(st.mode);

    const q = new URLSearchParams(location.search);
    const who = q.get('avatar') || 'default';
    const manifest = who === 'mannequin'
      ? { id: 'mannequin', kind: 'procedural', height_m: 1.62, yaw_offset_deg: 0 }
      : { id: 'default', kind: 'vrm', height_m: 1.62, yaw_offset_deg: 0 };
    avatar = await loadAvatar(manifest);
    avatar.root.traverse(o => o.layers.set(1));
    scene.add(avatar.root);
    retarget = buildRetarget(avatar, CORE_SKELETON.names);
    fitLegs(avatar, retarget);
    // loadVrm puts the avatar root at the feet; ARDY roots are pelvis positions.
    groundOffset = -avatar.groundMetrics().hipY;
    physics.setAvatar(avatar);
    origin = st.origin.slice();
    ready = true;
    await reset();
    $('loading').classList.add('hide');
    requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    $('loadingText').textContent = '初始化失败: ' + e.message;
    setTimeout(boot, 1500);
  }
}

async function reset() {
  setStatus('重置 ARDY…');
  const r = await api('/api/reset', { method: 'POST' });
  frames = r.seed || [];
  origin = r.origin.slice();
  target.set(r.target[0], r.target[1]);
  setTargetMarker();
  playT = 0;
  pending = false;
  stopCount = 0;
  viewerCollision = false;
  if (physics) physics.reset();
  pathLine.visible = false;
  setMode(r.mode);
  setStatus('运行中');
  if (frames.length) applyFrame(frames.length - 1, frames.length - 1, 0);
  requestWindow();
}

function captureEye(last) {
  let yaw = Number(last?.yaw || 0);
  let pitch = -0.06;
  let ex = last.root[0];
  let ez = last.root[2];
  let ey = 1.55;
  const hp = headEyePose();
  if (hp) {
    if (eyeRel0 == null) eyeRel0 = { yaw: hp.yawRel, pitch: hp.pitchRel };
    yaw = currentYaw + clamp(hp.yawRel - eyeRel0.yaw, -0.5, 0.5);
    pitch = clamp(-0.06 + (hp.pitchRel - eyeRel0.pitch), -0.5, 0.35);
    ex = hp.eye.x;
    ey = hp.eye.y;
    ez = hp.eye.z;
  }
  eyeCam.position.set(ex, ey, ez);
  eyeCam.up.set(0, 1, 0);
  eyeCam.lookAt(
    ex + Math.sin(yaw) * Math.cos(pitch),
    ey + Math.sin(pitch),
    ez + Math.cos(yaw) * Math.cos(pitch),
  );
  eyeCam.updateMatrixWorld(true);
  const prev = renderer.getRenderTarget();
  renderer.setRenderTarget(eyeRT);
  renderer.clear();
  renderer.render(scene, eyeCam);
  const buf = new Uint8Array(320 * 180 * 4);
  renderer.readRenderTargetPixels(eyeRT, 0, 0, 320, 180, buf);
  renderer.setRenderTarget(prev);
  const px = eyePixels.data;
  for (let y = 0; y < 180; y++) {
    const src = (179 - y) * 320 * 4;
    px.set(buf.subarray(src, src + 320 * 4), y * 320 * 4);
  }
  eyeCtx.putImageData(eyePixels, 0, 0);
  const url = eyeCanvas.toDataURL('image/jpeg', 0.86);
  $('eyeImg').src = url;
  lastEyeUrl = url;
  lastEyePose = { root: [ex, ez], y: ey, yaw, pitch };
  return url;
}

async function requestWindow() {
  if (!ready || pending || !frames.length || stopCount >= 2) return;
  pending = true;
  setStatus('视觉窗口…');
  try {
    const last = frames[frames.length - 1];
    const eye = captureEye(last);
    const t0 = performance.now();
    const res = await api('/api/window', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: eye, target: [target.x, target.y],
                             eye: lastEyePose,
                             halt: viewerCollision,
                             constraint: $('useVisual').checked }),
    });
    const dt = (performance.now() - t0) / 1000;
    latEma = latEma * 0.75 + dt * 0.25;
    lastResp = res;
    if (res.frames?.length) frames.push(...res.frames);
    origin = res.origin.slice();
    if (res.depth) $('depthImg').src = res.depth;
    if (res.mask) $('maskImg').src = res.mask;
    if (res.region) $('regionImg').src = res.region;
    if (res.traj?.length) {
      for (let i = 0; i < 8; i++) {
        const p = res.traj[Math.min(i, res.traj.length - 1)];
        pathPos[i * 3 + 0] = p[0];
        pathPos[i * 3 + 1] = 0.04;
        pathPos[i * 3 + 2] = p[1];
      }
      pathGeo.attributes.position.needsUpdate = true;
      pathGeo.computeBoundingSphere();
      pathLine.visible = true;
    }
    setMode(res.mode);
    $('action').textContent = `${res.action || '—'} · sdf ${res.min_sdf == null ? '—' : res.min_sdf.toFixed(2)}`;
    $('lat').textContent = `infer ${(res.latency.infer*1000).toFixed(0)} · ardy ${(res.latency.ardy*1000).toFixed(0)} · total ${(dt*1000).toFixed(0)} ms`;
    const g = res.guide || {};
    $('nav').textContent = res.mode === 'visual'
      ? `路线 ${g.route_remaining == null ? '—' : g.route_remaining.toFixed(1)} m · 重规划 ${g.replans ?? 0}`
        + ` · 占据 ${g.occupied_cells ?? 0} · ${g.invalid ? '原路线失效' : '路线有效'}`
        + `${g.replanned ? ' · 已重规划' : ''}`
      : '直走先验 · 无重规划';
    if (res.violation) { setStatus('约束违反'); }
    else if (viewerCollision) { setStatus('碰撞 · 等待障碍移开'); }
    else if (res.stop && !res.arrived) setStatus('避障停车 · 重规划中');
    else setStatus(res.arrived ? '已到达 · 站立' : '运行中');
    // Only arrival ends the loop; a safety stop or a viewer collision must
    // keep requesting windows so the route can recover when the obstacle
    // (e.g. a dragged box) is moved away.
    if (res.arrived) stopCount += 1; else stopCount = 0;
  } catch (e) {
    console.warn(e);
    setStatus('窗口失败：' + e.message);
    await sleep(1000);
  } finally {
    pending = false;
  }
}

function applyFrame(i, j, alpha) {
  if (!frames.length || !avatar || !retarget) return;
  const A = frames[clamp(i, 0, frames.length - 1)];
  const B = frames[clamp(j, 0, frames.length - 1)];
  const quats = new Array(27);
  const qa = new THREE.Quaternion();
  const qb = new THREE.Quaternion();
  const qo = new THREE.Quaternion();
  for (let k = 0; k < 27; k++) {
    qa.fromArray(A.rot[k]);
    qb.fromArray(B.rot[k]);
    qo.copy(qa).slerp(qb, alpha);
    quats[k] = qo.toArray();
  }
  avatar.applyPose(retarget.apply(quats));
  const rx = A.root[0] + (B.root[0] - A.root[0]) * alpha;
  const ry = A.root[1] + (B.root[1] - A.root[1]) * alpha;
  const rz = A.root[2] + (B.root[2] - A.root[2]) * alpha;
  avatar.root.position.set(rx, ry + groundOffset - footDropSm, rz);
  avatar.root.rotation.y = avatar.yawOffset;
  avatar.update(1 / 60);
  avatar.root.updateMatrixWorld(true);
  // Ground contact compensation: the retargeted walk keeps both feet a few
  // cm above the floor line of the root trajectory; measure the lower foot
  // and sink the whole avatar so the stance foot lands (smoothed over time
  // so it doesn't turn into vertical jitter).
  let footMin = null;
  for (const n of ['rightFoot', 'leftFoot']) {
    const nd = avatar.getNode(n);
    if (!nd) continue;
    nd.getWorldPosition(footProbe);
    footMin = footMin == null ? footProbe.y : Math.min(footMin, footProbe.y);
  }
  if (footMin != null) {
    // Slow EMA (rate 0.02): compensates the constant retarget rig bias but
    // ignores the window-to-window root-height re-anchoring noise, which a
    // fast follower turned into visible sinking.
    const want = Math.min(0.08, Math.max(0.0, footMin - 0.008));
    footDropSm = footDropSm + (want - footDropSm) * 0.02;
    avatar.root.position.y = ry + groundOffset - footDropSm;
    avatar.root.updateMatrixWorld(true);
  }
  currentYaw = Number(A.yaw || 0) + (Number(B.yaw || 0) - Number(A.yaw || 0)) * alpha;
  if (!viewerCollision && !freeAt(rx, rz)) {
    viewerCollision = true;
    setStatus('碰撞');
    stopCount = 5;
  }
}

// Collision uses the *current* physics AABB, so a dragged box is judged where
// it is now, not where it spawned.
function freeAt(x, z, clr = 0.22) {
  if (!arenaSpec) return true;
  const half = Number(arenaSpec.half) - clr;
  if (Math.abs(x) >= half || Math.abs(z) >= half) return false;
  if (!physics) return true;
  for (const b of physics.boxStates()) {
    if (b.minY > 1.55 || b.maxY < 0.12) continue;   // above head / below floor
    const nx = clamp(x, b.minX, b.maxX);
    const nz = clamp(z, b.minZ, b.maxZ);
    if (Math.hypot(x - nx, z - nz) < clr) return false;
  }
  return true;
}

function loop() {
  if (!ready) return;
  const dt = Math.min(clock.getDelta(), 0.1);
  if (physics) { physics.step(dt); if (avatar) physics.setAvatarPosition(avatar.root.position); }
  // Collision recovery: while a box overlaps the avatar the windows keep
  // flowing (halt=true keeps the actor standing); once the obstacle is
  // dragged clear for ~1.5 s the walk resumes automatically.
  if (viewerCollision) {
    const r = avatar.root.position;
    if (freeAt(r.x, r.z)) {
      colFreeT += dt;
      if (colFreeT > 1.5) { viewerCollision = false; colFreeT = 0; }
    } else {
      colFreeT = 0;
    }
  }
  if (frames.length) {
    playT += dt;
    const x = playT * 20;
    let i = Math.floor(x);
    let alpha = x - i;
    let j = i + 1;
    if (i >= frames.length - 1) { i = frames.length - 1; j = i; alpha = 0; }
    applyFrame(i, j, clamp(alpha, 0, 1));
    const remain = frames.length - 1 - i;
    // Adaptive prefetch: keep enough buffered frames to cover the measured
    // window latency (plus a small margin), so playback never starves, but
    // don't hoard stale route after the environment changes.
    const need = Math.min(40, Math.max(12, Math.ceil(latEma * 20) + 6));
    if (remain < need && !pending) requestWindow();
    const r = avatar.root.position;
    $('dist').textContent = `目标 ${Math.hypot(target.x-r.x, target.y-r.z).toFixed(2)} m`;
  }
  // follow camera: behind the pelvis heading
  if (avatar) {
    const r = avatar.root.position;
    const fx = Math.sin(currentYaw), fz = Math.cos(currentYaw);
    const desired = new THREE.Vector3(r.x - fx * 4.0, r.y + 2.8, r.z - fz * 4.0);
    camera.position.lerp(desired, 1 - Math.exp(-6 * dt));
    camera.lookAt(r.x, r.y + 1.0, r.z);
  }
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}

// click-to-target on the floor
let down = null;
canvas.addEventListener('pointerdown', e => {
  if (physics && physics.dragStart(e)) { dragMode = true; down = null; return; }
  down = [e.clientX, e.clientY];
});
canvas.addEventListener('pointermove', e => {
  if (dragMode && physics) physics.dragMove(e);
});
canvas.addEventListener('pointerup', e => {
  if (dragMode) { if (physics) physics.dragEnd(); dragMode = false; return; }
  if (!down) return;
  const moved = Math.hypot(e.clientX - down[0], e.clientY - down[1]);
  down = null;
  if (moved > 6 || !arena) return;
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2(
    (e.clientX / window.innerWidth) * 2 - 1,
    -(e.clientY / window.innerHeight) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hit = new THREE.Vector3();
  if (!ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), hit)) return;
  const half = arena.half - 0.4;
  target.set(clamp(hit.x, -half, half), clamp(hit.z, -half, half));
  setTargetMarker();
  stopCount = 0;
  setStatus('新目标');
  requestWindow();
});
canvas.addEventListener('pointercancel', () => {
  if (dragMode && physics) physics.dragEnd();
  dragMode = false; down = null;
});
canvas.addEventListener('contextmenu', e => e.preventDefault());

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight, false);
});

$('resetBtn').addEventListener('click', () => { if (ready) reset().catch(console.warn); });
$('useVisual').addEventListener('change', () => { stopCount = 0; viewerCollision = false; requestWindow(); });

// Small hook for automated closed-loop QC (same target path as a click).
window.ardyDemo = {
  setTarget(x, z) {
    const half = (arena?.half ?? 8) - 0.4;
    target.set(clamp(Number(x), -half, half), clamp(Number(z), -half, half));
    setTargetMarker();
    stopCount = 0;
    viewerCollision = false;
    setStatus('新目标');
    requestWindow();
  },
  state() {
    const r = avatar?.root?.position;
    return {
      ready, frames: frames.length, pending, stopCount,
      // Raw ARDY Core27 output of the frame currently on screen, for
      // model-vs-retarget ablation (QC only; small enough to ship).
      rawLast: frames.length
        ? { rot: frames[clamp(playT * 20, 0, frames.length - 1) | 0].rot,
            root: frames[clamp(playT * 20, 0, frames.length - 1) | 0].root,
            yaw: currentYaw }
        : null,
      root: r ? [r.x, r.z] : null,
      target: [target.x, target.y],
      dist: r ? Math.hypot(target.x - r.x, target.y - r.z) : null,
      status: $('status').textContent,
      yaw: currentYaw,
      mode: $('mode').textContent,
      modeNote: $('modeNote').textContent,
      lat: $('lat').textContent,
      nav: $('nav').textContent,
      collision: Boolean(viewerCollision),
      backendCollision: Boolean(lastResp?.collision),
      violation: Boolean(lastResp?.violation),
      minSdf: lastResp?.min_sdf ?? null,
      stopped: Boolean(lastResp?.stop),
      guide: lastResp?.guide ?? null,
      route: lastResp?.route ?? null,
      boxes: physics ? physics.boxStates() : null,
    };
  },
  images() {
    return { eye: lastEyeUrl, eyePose: lastEyePose,
             depth: lastResp?.depth || null,
             mask: lastResp?.mask || null, maskBin: lastResp?.mask_bin || null,
             region: lastResp?.region || null,
             guide: lastResp?.guide || null, route: lastResp?.route || null };
  },
  screenOfBox(i) {
    if (!physics) return null;
    const b = physics.bodies[i];
    if (!b) return null;
    const v = new THREE.Vector3(b.position.x, b.position.y, b.position.z).project(camera);
    if (v.z > 1) return null;
    return [(v.x + 1) * 0.5 * window.innerWidth, (1 - v.y) * 0.5 * window.innerHeight];
  },
  // Environment edit for QC (the same path as dragging, minus the pointer).
  moveBox(i, x, z, y) {
    if (!physics) return null;
    const body = physics.bodies[i];
    if (!body) return null;
    body.position.set(Number(x), y == null ? body.position.y : Number(y), Number(z));
    body.velocity.setZero();
    body.angularVelocity.setZero();
    body.wakeUp();
    return physics.boxStates();
  },
  reset,
  retarget: () => retarget,
  avatar: () => avatar,
  bones() {
    const names = ['hips','spine','chest','neck','head','rightHand','leftHand',
                   'rightFoot','leftFoot',
                   'rightUpperArm','rightLowerArm','leftUpperArm','leftLowerArm',
                   'rightUpperLeg','rightLowerLeg','leftUpperLeg','leftLowerLeg'];
    const out = {};
    for (const n of names) {
      const node = avatar?.getNode(n);
      if (node) out[n] = node.getWorldPosition(new THREE.Vector3()).toArray();
    }
    if (avatar) {
      const q = (n) => {
        const node = avatar.getNode(n);
        return node ? node.getWorldQuaternion(new THREE.Quaternion()).toArray() : null;
      };
      out._hipsQ = q('hips');
      out._headQ = q('head');
      out._neckQ = q('neck');
      out._direction = avatar.direction ?? null;
    }
    return out;
  },
};

poll();
