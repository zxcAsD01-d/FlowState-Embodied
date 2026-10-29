// Quiz alignment hall: Chinese question boards, glowing red letter buttons and
// a right-hand contact test.  The scene itself owns the trial bookkeeping and
// exposes it through window.vrmHub.quiz; the brain/hand live outside the page.
//
// Contract (same as the other scene plugins):
//   createScene({ manifest, scene, camera, fpCamera, renderer })
//   -> { root, update(dt, serverState), setOverlay(group), dispose(), meta }
//
// Button surfaces live in a tilted console frame.  A press is a hand point that
// enters the button's local box while the avatar stands at that station; the
// letters are the only labels on the buttons (the question board carries the
// option text), which is what the black-on-red requirement asks for.

import * as THREE from 'three';

const TILT = THREE.MathUtils.degToRad(22);
const STATION_X = [-2.5, 0.0, 2.5];
const STAND_Z = -0.62;
const ACTIVE_RADIUS = 2.1;
const AUTO_NEXT_MS = 3000;   // a correct answer advances to the next question
const RETRY_MS = 2200;       // a wrong answer re-arms the same question

const CONFIRM_KEY = 'OK';
const PRESS_HALF = { x: 0.125, z: 0.08, ymin: -0.025, ymax: 0.17 };
const BANK_URL = '/quiz/bank.json';

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function canvas2d(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

function textureOf(canvas) {
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

function letterTexture(label) {
  const c = canvas2d(256, 150);
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  const wide = String(label).length > 1;
  const grad = ctx.createLinearGradient(0, 0, 0, c.height);
  grad.addColorStop(0, '#ffffff');
  grad.addColorStop(1, '#d8d8d8');
  ctx.fillStyle = grad;
  roundRect(ctx, 8, 8, c.width - 16, c.height - 16, 26);
  ctx.fill();
  ctx.fillStyle = '#080808';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `900 ${wide ? 66 : 108}px "Microsoft YaHei", "Noto Sans SC", Arial, sans-serif`;
  ctx.fillText(String(label), c.width / 2, c.height / 2 + (wide ? 2 : 6));
  return textureOf(c);
}

function glowTexture() {
  const c = canvas2d(128, 128);
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(64, 64, 6, 64, 64, 62);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.42, 'rgba(255,255,255,0.5)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  return textureOf(c);
}

function titleTexture() {
  const c = canvas2d(1400, 220);
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#dce8ff';
  ctx.font = '900 86px "Microsoft YaHei", "Noto Sans SC", Arial, sans-serif';
  ctx.fillText('问答对齐厅', c.width / 2, 86);
  ctx.fillStyle = '#7fa6d8';
  ctx.font = '600 40px "Segoe UI", Arial, sans-serif';
  ctx.fillText('QUIZ  ALIGNMENT  HALL', c.width / 2, 160);
  return textureOf(c);
}

function drawBoard(canvas, q, n) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0a1122';
  roundRect(ctx, 4, 4, w - 8, h - 8, 26);
  ctx.fill();
  ctx.strokeStyle = '#38527a';
  ctx.lineWidth = 6;
  roundRect(ctx, 14, 14, w - 28, h - 28, 20);
  ctx.stroke();

  // header
  ctx.fillStyle = '#8fb3e6';
  ctx.font = '600 34px "Microsoft YaHei", "Noto Sans SC", Arial, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(`第 ${n} 题`, 56, 78);
  if (q.multi) {
    ctx.fillStyle = '#ffb347';
    ctx.textAlign = 'right';
    ctx.font = '700 34px "Microsoft YaHei", "Noto Sans SC", Arial, sans-serif';
    ctx.fillText('多选题', w - 56, 78);
  }

  // stem, wrapped to at most two lines
  ctx.fillStyle = '#f2f6ff';
  ctx.textAlign = 'left';
  ctx.font = '800 58px "Microsoft YaHei", "Noto Sans SC", Arial, sans-serif';
  const maxW = w - 130;
  const words = String(q.stem).split('');
  const lines = [];
  let line = '';
  for (const ch of words) {
    if (ctx.measureText(line + ch).width > maxW && line) {
      lines.push(line);
      line = ch;
      if (lines.length === 2) break;
    } else {
      line += ch;
    }
  }
  if (lines.length < 2 && line) lines.push(line);
  lines.forEach((ln, i) => ctx.fillText(ln, 66, 160 + i * 72));

  let y = lines.length > 1 ? 330 : 300;
  if (q.hint) {
    ctx.fillStyle = '#ffc35c';
    ctx.font = '600 33px "Microsoft YaHei", "Noto Sans SC", Arial, sans-serif';
    ctx.fillText(q.hint, 66, y - 34);
  }

  const opts = q.options || [];
  const twoCol = opts.length > 4;
  // Four options get a larger face: the board is read from the stand spot,
  // ~2 m away, so the first-person crop needs the extra weight.
  const big = !twoCol;
  const colX = twoCol ? [66, 640] : [66];
  const perCol = twoCol ? Math.ceil(opts.length / 2) : opts.length;
  const rowH = twoCol ? 78 : 92;
  const optFont = big ? 48 : 40;
  const badge = big ? 60 : 54;
  // one-line stems start higher than two-line ones
  if (big) y = Math.min(y, lines.length > 1 ? 330 : 310);
  opts.forEach((opt, i) => {
    const col = twoCol ? Math.floor(i / perCol) : 0;
    const row = twoCol ? i % perCol : i;
    const x = colX[col];
    const yy = y + row * rowH;
    ctx.fillStyle = '#ff3b3b';
    roundRect(ctx, x, yy - badge + 12, badge, badge, 14);
    ctx.fill();
    ctx.fillStyle = '#050505';
    ctx.font = `900 ${big ? 42 : 38}px "Segoe UI", Arial, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(opt.key, x + badge / 2, yy - badge / 2 + 13);
    ctx.fillStyle = '#dfe8ff';
    ctx.textAlign = 'left';
    ctx.font = `600 ${optFont}px "Microsoft YaHei", "Noto Sans SC", Arial, sans-serif`;
    ctx.fillText(opt.text, x + badge + 22, yy - 13);
  });
  ctx.textBaseline = 'alphabetic';
}

function drawIdle(canvas) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = '#0a1122';
  roundRect(ctx, 4, 4, w - 8, h - 8, 26);
  ctx.fill();
  ctx.strokeStyle = '#38527a';
  ctx.lineWidth = 6;
  roundRect(ctx, 14, 14, w - 28, h - 28, 20);
  ctx.stroke();
  ctx.fillStyle = '#8fb3e6';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = '600 44px "Microsoft YaHei", "Noto Sans SC", Arial, sans-serif';
  ctx.fillText('题库加载中…', w / 2, h / 2);
}

export function createScene({ manifest, scene, camera, fpCamera, renderer }) {
  const root = new THREE.Group();
  root.name = 'quiz_hall';
  const owned = [];
  const stationGroups = [];
  const events = [];
  const trials = [];
  const rngSeed = { s: 12345 };
  const rand = () => {
    rngSeed.s = (rngSeed.s * 1103515245 + 12345) & 0x7fffffff;
    return rngSeed.s / 0x7fffffff;
  };

  // ------------------------------------------------------------------ lights
  const hemi = new THREE.HemisphereLight(0x9db8ff, 0x16121c, 0.9);
  const key = new THREE.DirectionalLight(0xfff0dd, 1.9);
  key.position.set(7, 10, 3);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -9;
  key.shadow.camera.right = 9;
  key.shadow.camera.top = 9;
  key.shadow.camera.bottom = -9;
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 40;
  key.shadow.bias = -0.0004;
  key.shadow.normalBias = 0.03;
  const rim = new THREE.DirectionalLight(0x5a7dff, 1.1);
  rim.position.set(-6, 7, -7);
  root.add(hemi, key, key.target, rim);
  owned.push(hemi, key, rim);

  // --------------------------------------------------------------- materials
  const floorMat = new THREE.MeshStandardMaterial({ color: 0x121a28, roughness: 0.88, metalness: 0.08 });
  const gridMat = new THREE.MeshBasicMaterial({ color: 0x233452, transparent: true, opacity: 0.55 });
  const wallMat = new THREE.MeshStandardMaterial({ color: 0x141c2c, roughness: 0.92 });
  const trimMat = new THREE.MeshBasicMaterial({ color: 0x3b5f9e });
  const pedestalMat = new THREE.MeshStandardMaterial({ color: 0x232d40, roughness: 0.55, metalness: 0.35 });
  const slabMat = new THREE.MeshStandardMaterial({ color: 0x1a2233, roughness: 0.42, metalness: 0.55 });
  const frameMat = new THREE.MeshStandardMaterial({ color: 0x2b3852, roughness: 0.4, metalness: 0.6 });
  const markerMat = new THREE.MeshBasicMaterial({ color: 0x54d7ff, transparent: true, opacity: 0.55 });
  owned.push(floorMat, gridMat, wallMat, trimMat, pedestalMat, slabMat, frameMat, markerMat);

  const box = (w, h, d, mat, x, y, z, parent = root) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z);
    m.castShadow = true;
    m.receiveShadow = true;
    parent.add(m);
    return m;
  };
  const plane = (w, d, mat, x, y, z, parent = root) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), mat);
    m.rotation.x = -Math.PI / 2;
    m.position.set(x, y, z);
    m.receiveShadow = true;
    parent.add(m);
    return m;
  };

  // ------------------------------------------------------------------- hall
  plane(18, 15, floorMat, 0, 0, 0);
  for (let i = -8; i <= 8; i++) {
    const line = plane(0.02, 15, gridMat, i, 0.004, 0);
    line.castShadow = false;
    line.receiveShadow = false;
  }
  for (let j = -6; j <= 6; j++) {
    const line = plane(18, 0.02, gridMat, 0, 0.004, j);
    line.castShadow = false;
    line.receiveShadow = false;
  }
  // back + side walls
  box(18, 4.6, 0.16, wallMat, 0, 2.3, 1.72);
  box(0.16, 4.6, 12, wallMat, -8.4, 2.3, -2.2);
  box(0.16, 4.6, 12, wallMat, 8.4, 2.3, -2.2);
  box(18, 0.08, 0.1, trimMat, 0, 4.1, 1.62);
  box(0.1, 0.08, 11, trimMat, -8.3, 4.1, -2.2);
  box(0.1, 0.08, 11, trimMat, 8.3, 4.1, -2.2);

  const titleMat = new THREE.MeshBasicMaterial({ map: titleTexture() });
  const title = new THREE.Mesh(new THREE.PlaneGeometry(5.6, 0.88), titleMat);
  title.position.set(0, 3.35, 1.6);
  title.rotation.y = Math.PI;
  root.add(title);
  owned.push(titleMat);

  const bannerGlow = new THREE.PointLight(0x7fa6ff, 6.0, 8.0, 2.0);
  bannerGlow.position.set(0, 3.2, 1.1);
  root.add(bannerGlow);
  owned.push(bannerGlow);

  // ------------------------------------------------------------ station bits
  const glowTex = glowTexture();
  const glowGeo = new THREE.PlaneGeometry(1, 1);
  const buttonGeo = new THREE.PlaneGeometry(0.24, 0.14);
  const confirmGeo = new THREE.PlaneGeometry(0.36, 0.08);
  const boardCanvas = new Map();

  function makeButton(st, slot, label, size, local) {
    const faceMat = new THREE.MeshBasicMaterial({
      map: letterTexture(label), color: 0xff3030, transparent: true,
    });
    const face = new THREE.Mesh(size === 'ok' ? confirmGeo : buttonGeo, faceMat);
    face.position.set(local.x, 0.012, local.z);
    // The plane lies on the tilted slab; -90deg about X makes its +Z normal
    // point up, but that also maps the texture's top edge toward the avatar,
    // so the letters read upside down from the stand spot.  Rotate the texture
    // half a turn in-plane: the top edge goes to +Z (the far side of the key).
    face.rotation.set(-Math.PI / 2, 0, Math.PI);
    const glowMat = new THREE.MeshBasicMaterial({
      map: glowTex, color: 0xff2020, transparent: true, opacity: 0.55,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    glow.scale.set(size === 'ok' ? 0.62 : 0.42, size === 'ok' ? 0.28 : 0.28, 1);
    glow.position.set(local.x, 0.008, local.z);
    glow.rotation.x = -Math.PI / 2;
    st.slab.add(face, glow);
    const b = {
      key: label, slot, face, glow, faceMat, glowMat,
      local: { x: local.x, z: local.z },
      world: new THREE.Vector3(),
      normal: new THREE.Vector3(),
      armed: true, flashUntil: 0, baseScale: 1,
      kind: size,
    };
    st.buttons.push(b);
    return b;
  }

  function buildStation(index) {
    const x = STATION_X[index];
    const g = new THREE.Group();
    g.position.set(x, 0, 0);
    root.add(g);
    stationGroups.push(g);

    // pedestal + tilted console top.  The console sits ~0.7 m in front of the
    // stand mark: ARDY's produced right hand rests at z≈-0.09, so a console
    // at the body (old z=-0.02) put the resting hand inside the key volume
    // and the scene fired presses before the agent ever moved.
    box(1.5, 0.92, 0.52, pedestalMat, 0, 0.46, 0.34, g);
    box(1.54, 0.06, 0.54, frameMat, 0, 0.94, 0.34, g);
    const slab = new THREE.Group();
    slab.position.set(0, 0.98, 0.34);
    slab.rotation.x = -TILT;
    g.add(slab);
    const top = box(1.5, 0.05, 0.52, slabMat, 0, -0.026, 0, slab);
    top.castShadow = true;

    // question board: a large wall screen behind the console.  At the stand
    // spot the eye is ~2.2 m from it, so all options fit; mounted on the
    // console it was so close that the first-person view clipped C/D.
    const canvas = canvas2d(1200, 700);
    drawIdle(canvas);
    const tex = textureOf(canvas);
    const boardMat = new THREE.MeshBasicMaterial({ map: tex });
    const board = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 1.52), boardMat);
    board.position.set(0, 1.95, 1.35);
    board.rotation.y = Math.PI;
    g.add(board);
    box(2.7, 1.62, 0.06, frameMat, 0, 1.95, 1.385, g);
    board.renderOrder = 2;
    boardCanvas.set(index, { canvas, tex, mat: boardMat, mesh: board });

    // status lamp on the console lip so it stays visible while standing close
    const lampMat = new THREE.MeshStandardMaterial({
      color: 0x20304a, emissive: 0x385a88, emissiveIntensity: 1.2, roughness: 0.4,
    });
    const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.055, 16, 12), lampMat);
    lamp.position.set(0, 1.02, 0.16);
    g.add(lamp);

    // station label above the console, under the wall screen
    const labelCanvas = canvas2d(256, 96);
    const lctx = labelCanvas.getContext('2d');
    lctx.clearRect(0, 0, 256, 96);
    lctx.fillStyle = '#8fb3e6';
    lctx.font = '700 52px "Segoe UI", Arial, sans-serif';
    lctx.textAlign = 'center';
    lctx.textBaseline = 'middle';
    lctx.fillText(`STATION ${index + 1}`, 128, 48);
    const labelMat = new THREE.MeshBasicMaterial({ map: textureOf(labelCanvas), transparent: true });
    const labelMesh = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 0.34), labelMat);
    labelMesh.position.set(0, 1.1, 1.36);
    labelMesh.rotation.y = Math.PI;
    g.add(labelMesh);
    owned.push(labelMat);

    // floor stand marker
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.34, 0.42, 40), markerMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(0, 0.012, STAND_Z);
    g.add(ring);

    // warm spot above the console
    const spot = new THREE.PointLight(0xffd9b0, 14.0, 5.2, 2.0);
    spot.position.set(0, 2.5, 0.35);
    g.add(spot);
    owned.push(spot);

    const st = {
      index, x, group: g, slab, board, boardMat, canvas, tex, lamp, lampMat,
      ring, buttons: [], question: null, qIndex: index * 5, cycle: 0,
      selections: new Set(), pressed: [], done: false, correct: null,
      autoNextAt: 0, retryAt: 0, lastResult: null, multi: false,
      hover: null, lastPressAt: 0,
      // Wrong keys picked in an earlier attempt are excluded from every later
      // attempt of the same question: they stay visible (dark) but can no
      // longer be pressed, and the agent masks them before scoring letters.
      eliminated: new Set(), elimQid: null,
    };

    // option slots: up to 4 columns x 2 rows
    for (let slot = 0; slot < 8; slot++) {
      const btn = makeButton(st, slot, 'A', 'opt', { x: 0, z: 0 });
      btn.face.visible = false;
      btn.glow.visible = false;
    }
    // confirm button on the front lip
    st.confirm = makeButton(st, 8, CONFIRM_KEY, 'ok', { x: 0, z: -0.215 });
    st.confirm.face.visible = false;
    st.confirm.glow.visible = false;

    return st;
  }

  const stations = [0, 1, 2].map(buildStation);

  // ------------------------------------------------------------------- quiz
  let bank = null;
  let active = -1;
  let forcedActive = null;
  let ready = false;
  const tmpA = new THREE.Vector3();
  const tmpB = new THREE.Vector3();
  const tmpQ = new THREE.Quaternion();
  const now = () => performance.now();

  function stationPublic(st) {
    return {
      index: st.index,
      stand: [st.x, 0, STAND_Z],
      board: [st.x, 0, 1.6],
      active: active === st.index,
    };
  }

  function layout(st) {
    const q = st.question;
    if (!q) return;
    const opts = q.options;
    const n = opts.length;
    const twoRow = n > 4;
    const cols = twoRow ? Math.ceil(n / 2) : n;
    const pitch = 0.30;
    st.buttons.forEach((b) => {
      if (b.kind === 'ok') return;
      b.face.visible = false;
      b.glow.visible = false;
    });
    opts.forEach((opt, i) => {
      const b = st.buttons[i];
      if (!b) return;
      const col = twoRow ? i % cols : i;
      const row = twoRow ? Math.floor(i / cols) : 0;
      const rowCount = twoRow ? Math.min(n - row * cols, cols) : n;
      const x = (col - (rowCount - 1) / 2) * pitch;
      const z = twoRow ? (row === 0 ? -0.10 : 0.10) : -0.02;
      b.local.x = x;
      b.local.z = z;
      b.key = opt.key;
      b.face.geometry = buttonGeo;
      b.faceMat.map?.dispose?.();
      b.faceMat.map = letterTexture(opt.key);
      b.faceMat.color.setHex(0xff3030);
      b.faceMat.needsUpdate = true;
      b.face.position.set(x, 0.012, z);
      b.glow.position.set(x, 0.008, z);
      b.face.visible = true;
      b.glow.visible = true;
      b.face.scale.setScalar(1);
      b.glowMat.color.setHex(0xff2020);
      b.glowMat.opacity = 0.55;
      b.armed = true;
      b.flashUntil = 0;
      if (st.eliminated && st.eliminated.has(opt.key)) {
        setButtonFeedback(st, opt.key, 'eliminated');
      } else {
        b.eliminated = false;
      }
      b.face.updateMatrix();
      b.glow.updateMatrix();
      // world target: button centre lifted along the slab normal
      st.slab.updateWorldMatrix(true, true);
      b.world.set(x, 0.012, z).applyMatrix4(st.slab.matrixWorld);
      b.normal.set(0, 1, 0).applyQuaternion(
        st.slab.getWorldQuaternion(tmpQ));
    });
    const c = st.confirm;
    c.face.visible = !!q.multi;
    c.glow.visible = !!q.multi;
    c.key = CONFIRM_KEY;
    c.faceMat.map?.dispose?.();
    c.faceMat.map = letterTexture('确认');
    c.faceMat.color.setHex(0xff3030);
    c.faceMat.needsUpdate = true;
    c.face.scale.setScalar(1);
    c.glowMat.color.setHex(0xff2020);
    c.glowMat.opacity = 0.55;
    c.armed = true;
    c.flashUntil = 0;
    st.slab.updateWorldMatrix(true, true);
    c.world.set(0, 0.012, c.local.z).applyMatrix4(st.slab.matrixWorld);
    c.normal.set(0, 1, 0).applyQuaternion(st.slab.getWorldQuaternion(tmpQ));
    // refresh local coords of confirm
    c.local.x = 0;
  }

  function drawQuestionBoard(st) {
    const rec = boardCanvas.get(st.index);
    drawBoard(rec.canvas, st.question, st.cycle);
    rec.tex.needsUpdate = true;
  }

  // The active question board's screen-space rectangle in the first-person
  // view, normalised to [0,1].  The eye sees the neighbouring station's board
  // at the edge of its 72-degree FOV; feeding that strip to the answer policy
  // adds a second question (and, for a multi-choice neighbour, a "多选题"
  // badge) that is not in the quiz's own board.  Cropping to this rectangle
  // keeps the observation on the board the avatar is actually answering.
  function boardRectNorm(st) {
    if (!st || !st.board || !fpCamera) return null;
    st.board.updateWorldMatrix(true, false);
    fpCamera.updateMatrixWorld(true);
    const hw = 1.3, hh = 0.76;
    const corners = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    const v = new THREE.Vector3();
    for (const [x, y] of corners) {
      v.set(x, y, 0).applyMatrix4(st.board.matrixWorld).project(fpCamera);
      if (!Number.isFinite(v.x) || !Number.isFinite(v.y) ||
          v.z < -1 || v.z > 1) return null;
      const nx = v.x * 0.5 + 0.5;
      const ny = 0.5 - v.y * 0.5;
      x0 = Math.min(x0, nx); x1 = Math.max(x1, nx);
      y0 = Math.min(y0, ny); y1 = Math.max(y1, ny);
    }
    const m = 0.015;
    x0 = Math.max(0, x0 - m); y0 = Math.max(0, y0 - m);
    x1 = Math.min(1, x1 + m); y1 = Math.min(1, y1 + m);
    if (x1 - x0 < 0.05 || y1 - y0 < 0.05) return null;
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  }

  function setQuestion(st, idx) {
    if (!bank) return;
    const q = bank[idx % bank.length];
    const prevQid = st.question ? st.question.id : null;
    if (!st.eliminated) st.eliminated = new Set();
    if (prevQid !== q.id) {
      // A fresh question starts with no exclusions; retrying the same one
      // keeps every key that was already proven wrong.
      st.eliminated = new Set();
      st.elimQid = q.id;
    }
    st.question = q;
    st.qIndex = idx;
    st.cycle = Math.floor(idx / bank.length) + 1;
    st.selections = new Set();
    st.pressed = [];
    st.done = false;
    st.correct = null;
    st.multi = !!q.multi;
    st.lastResult = null;
    st.autoNextAt = 0;
    st.retryAt = 0;
    st.hover = null;
    st.lampMat.color.setHex(0x20304a);
    st.lampMat.emissive.setHex(0x385a88);
    st.lampMat.emissiveIntensity = 1.2;
    layout(st);
    drawQuestionBoard(st);
    // A question switch can happen while the hand still sits in a key volume
    // (auto-next right after a press).  A fresh question must not fire on that
    // hand: start every key disarmed; updateHand re-arms each key once the
    // hand is outside its near box.
    st.buttons.concat([st.confirm]).forEach((b) => { b.armed = false; });
    events.push({
      type: 'question', station: st.index, qid: q.id, stem: q.stem,
      multi: !!q.multi, n_options: q.options.length,
      buttons: st.buttons.slice(0, q.options.length).map((b) => ({
        key: b.key, world: b.world.toArray(), normal: b.normal.toArray(),
      })),
      confirm: q.multi ? st.confirm.world.toArray() : null,
      t: now(),
    });
  }

  function retryQuestion(st) {
    const qid = st.question ? st.question.id : null;
    setQuestion(st, st.qIndex);
    // the hand may still sit on the key that was just pressed: it has to
    // leave the key volume before that key can count again
    st.buttons.concat([st.confirm]).forEach((b) => { b.armed = false; });
    events.push({ type: 'retry', station: st.index, qid, t: now() });
  }

  function setButtonFeedback(st, key, state) {    const b = key === CONFIRM_KEY ? st.confirm
      : st.buttons.find((x) => x.key === key);
    if (!b) return;
    b.eliminated = false;
    const isElim = key !== CONFIRM_KEY && st.eliminated && st.eliminated.has(key);
    if (state === 'eliminated' || isElim) {
      // Excluded, not hidden: the option and its letter stay readable, but the
      // key is dead for contact tests and carries no glow.  The set is the
      // source of truth; every feedback call re-derives the flag from it so a
      // later "dim" pass cannot accidentally re-arm an excluded key.
      b.eliminated = true;
      b.faceMat.color.setHex(0x2a2f38);
      b.glowMat.color.setHex(0x10141a);
      b.glowMat.opacity = 0.0;
    } else if (state === 'correct') {
      b.faceMat.color.setHex(0x22ff66);
      b.glowMat.color.setHex(0x22ff66);
      b.glowMat.opacity = 0.95;
    } else if (state === 'wrong') {
      b.faceMat.color.setHex(0x5a0f0f);
      b.glowMat.color.setHex(0x330000);
      b.glowMat.opacity = 0.25;
    } else if (state === 'dim') {
      b.faceMat.color.setHex(0x6b1515);
      b.glowMat.color.setHex(0x5a0a0a);
      b.glowMat.opacity = 0.28;
    } else {
      b.faceMat.color.setHex(0xff3030);
      b.glowMat.color.setHex(0xff2020);
      b.glowMat.opacity = 0.55;
    }
  }

  function publicState(st) {
    if (!st || !st.question) {
      return { ready, active: -1, stations: stations.map(stationPublic) };
    }
    const q = st.question;
    return {
      ready,
      active: st.index,
      station: st.index,
      stations: stations.map(stationPublic),
      question: {
        id: q.id, stem: q.stem, hint: q.hint || '',
        multi: !!q.multi,
        options: q.options.map((o) => ({ key: o.key, text: o.text })),
      },
      selections: [...st.selections],
      eliminated: [...st.eliminated],
      done: st.done,
      correct: st.correct,
      presses: st.pressed.length,
      buttons: st.buttons.slice(0, q.options.length).map((b) => ({
        key: b.key, world: b.world.toArray(), normal: b.normal.toArray(),
        eliminated: !!b.eliminated,
      })),
      confirm: q.multi ? st.confirm.world.toArray() : null,
      stand: [st.x, 0, STAND_Z],
      t: now(),
    };
  }

  function eliminateKeys(st, keys) {
    const add = keys.filter((k) => !st.eliminated.has(k));
    if (!add.length) return add;
    add.forEach((k) => st.eliminated.add(k));
    events.push({
      type: 'eliminate', station: st.index,
      qid: st.question ? st.question.id : null, keys: add, t: now(),
    });
    return add;
  }

  function finalize(st, reason) {
    if (!st.question || st.done) return null;
    const answer = st.question.answer;
    const picked = new Set(st.selections);
    const correct = picked.size === answer.length
      && answer.every((k) => picked.has(k));
    st.done = true;
    st.correct = correct;
    // Every picked key that is not in the answer is now excluded.  The red
    // feedback is shown first; the next attempt lays the key out as dead.
    if (!correct) eliminateKeys(st, [...picked].filter((k) => !answer.includes(k)));
    // correct -> next question; wrong -> re-arm the same one after the feedback
    st.autoNextAt = correct ? now() + AUTO_NEXT_MS : 0;
    st.retryAt = correct ? 0 : now() + RETRY_MS;
    st.lampMat.color.setHex(correct ? 0x22ff66 : 0xff3030);
    st.lampMat.emissive.setHex(correct ? 0x22ff66 : 0xff2020);
    st.lampMat.emissiveIntensity = 2.2;
    for (const opt of st.question.options) {
      if (answer.includes(opt.key)) setButtonFeedback(st, opt.key, 'correct');
      else if (picked.has(opt.key)) setButtonFeedback(st, opt.key, 'wrong');
      else setButtonFeedback(st, opt.key, 'dim');
    }
    if (st.question.multi) setButtonFeedback(st, CONFIRM_KEY, correct ? 'correct' : 'wrong');
    const rec = {
      type: 'result', station: st.index, qid: st.question.id, reason,
      answer: [...answer], picked: [...picked], correct,
      eliminated: [...st.eliminated],
      reward: correct ? 1 : 0, presses: st.pressed.slice(),
      t: now(), t_rel: now() - (st.startedAt || now()),
    };
    st.lastResult = rec;
    st.reward = correct ? 1 : 0;
    trials.push(rec);
    events.push(rec);
    return rec;
  }

  function press(st, key, handWorld, source) {
    if (!st.question || st.done) return null;
    if (key !== CONFIRM_KEY && st.eliminated.has(key)) {
      events.push({ type: 'blocked', station: st.index, qid: st.question.id,
        key, source, t: now() });
      return null;
    }
    const q = st.question;
    const t = now();
    const hand = handWorld ? [handWorld[0], handWorld[1], handWorld[2]]
      : [st.x, 1.05, STAND_Z];
    if (key === CONFIRM_KEY) {
      if (!q.multi) return null;
      const rec = finalize(st, 'confirm');
      st.pressed.push({ key, t, source, hand, correct: rec ? rec.correct : null });
      events.push({ type: 'press', station: st.index, qid: q.id, key,
        correct: rec ? rec.correct : null, source, hand, t, confirm: true });
      return rec;
    }
    if (!q.options.some((o) => o.key === key)) return null;
    const isCorrect = q.answer.includes(key);
    const wasSelected = st.selections.has(key);
    if (q.multi) {
      if (wasSelected) st.selections.delete(key);
      else st.selections.add(key);
      if (st.selections.has(key)) setButtonFeedback(st, key, isCorrect ? 'correct' : 'wrong');
      else setButtonFeedback(st, key, 'normal');
      st.lampMat.emissive.setHex(isCorrect ? 0x22ff66 : 0xff2020);
      st.lampMat.emissiveIntensity = 1.6;
    } else {
      st.selections = new Set([key]);
      finalize(st, 'press');
    }
    const row = { key, t, source, hand, correct: isCorrect, selected: st.selections.has(key) };
    st.pressed.push(row);
    events.push({ type: 'press', station: st.index, qid: q.id, key,
      correct: isCorrect, selected: row.selected, source, hand, t });
    return row;
  }

  function pressAt(key, stationIndex) {
    const st = stationIndex === null || stationIndex === undefined
      ? stations[active >= 0 ? active : 0] : stations[stationIndex];
    if (!st) return null;
    const b = key === CONFIRM_KEY ? st.confirm
      : st.buttons.find((x) => x.key === key);
    return press(st, key, b ? b.world.toArray() : null, 'inject');
  }

  // hand contact, once per update
  let handPrev = null;
  function updateHand() {
    const hub = window.vrmHub;
    if (!hub) { handPrev = null; return; }
    const node = hub.avatar?.getNode ? hub.avatar.getNode('rightHand') : null;
    if (!node) { handPrev = null; return; }
    node.getWorldPosition(tmpA);
    const p = tmpA.toArray();
    // keep the sample in world coords for the event log
    stations.forEach((st) => {
      const local = st.slab.worldToLocal(tmpB.copy(tmpA));
      let hit = null;
      const all = st.buttons.concat([st.confirm]);
      for (const b of all) {
        if (!b.face.visible || b.eliminated) continue;
        const dx = Math.abs(local.x - b.local.x);
        const dz = Math.abs(local.z - b.local.z);
        const inside = dx <= PRESS_HALF.x && dz <= PRESS_HALF.z
          && local.y >= PRESS_HALF.ymin && local.y <= PRESS_HALF.ymax;
        const near = dx <= PRESS_HALF.x + 0.08 && dz <= PRESS_HALF.z + 0.08
          && local.y >= PRESS_HALF.ymin - 0.06 && local.y <= PRESS_HALF.ymax + 0.10;
        if (inside && near) {
          if (b.armed) { hit = b; }
        } else if (!near) {
          b.armed = true;
        }
      }
      if (hit && active === st.index && !st.done) {
        press(st, hit.key, p, 'hand');
        hit.armed = false;
        hit.flashUntil = now() + 260;
        hit.face.scale.setScalar(0.9);
      }
      // hover glow, only for the active station
      for (const b of all) {
        if (!b.face.visible) continue;
        if (b.eliminated) {
          if (b.glowMat.opacity !== 0) b.glowMat.opacity = 0;
          continue;
        }
        const d = Math.abs(local.x - b.local.x) <= PRESS_HALF.x + 0.03
          && Math.abs(local.z - b.local.z) <= PRESS_HALF.z + 0.03;
        const want = (active === st.index && d && !st.done) ? 1.0 : 0.55;
        if (Math.abs(b.glowMat.opacity - want) > 0.01) {
          if (b.glowMat.opacity > 0.4 || want > 0.4) b.glowMat.opacity = want;
        }
      }
    });
    handPrev = p;
  }

  // Head attention: look at the question while standing at a station, and drop
  // the eyes to the hand once it comes down to the key plane.  This is a gaze
  // controller (three-vrm lookAt), not a generated body motion.
  const lookHand = new THREE.Vector3();
  function updateLook() {
    const hub = window.vrmHub;
    if (!hub?.setLookAt) return;
    if (active < 0) { hub.setLookAt(null); return; }
    const st = stations[active];
    if (st.done) { hub.setLookAt([st.x, 1.05, 0.34]); return; }
    const hand = hub.avatar?.getNode ? hub.avatar.getNode('rightHand') : null;
    if (hand) {
      hand.getWorldPosition(lookHand);
      // Only look down at the hand once it is actually over the slab; a hand
      // hanging at the body's side is 0.3 m away and must not steal the gaze
      // from the question.
      if (Math.abs(lookHand.z - 0.34) < 0.22
          && Math.abs(lookHand.x - st.x) < 0.6 && lookHand.y > 0.85) {
        hub.setLookAt([lookHand.x, lookHand.y, lookHand.z]);
        return;
      }
    }
    hub.setLookAt([st.x, 1.95, 1.35]);
  }

  function update(dt, serverState) {
    const t = now();
    const hub = window.vrmHub;
    if (hub && hub.avatar) {
      const pos = hub.avatar.root.position;
      let best = -1;
      let bestD = 1e9;
      stations.forEach((st) => {
        const d = Math.abs(pos.x - st.x) + Math.abs(pos.z - STAND_Z) * 0.15;
        if (d < bestD) { bestD = d; best = st.index; }
      });
      const near = Math.abs(pos.x - stations[best].x) < ACTIVE_RADIUS
        && Math.abs(pos.z - STAND_Z) < 1.3;
      const want = forcedActive === null ? (near ? best : -1) : forcedActive;
      if (want !== active) {
        active = want;
        stations.forEach((st) => {
          const on = st.index === active;
          st.boardMat.color.setHex(on ? 0xffffff : 0x717a8c);
        });
        events.push({ type: 'active', station: active, t });
      }
    }
    updateHand();
    updateLook();
    for (const st of stations) {
      if (st.done && st.autoNextAt && t >= st.autoNextAt) {
        setQuestion(st, st.qIndex + 1);
      } else if (st.done && st.retryAt && t >= st.retryAt) {
        retryQuestion(st);
      }
      const pulse = 1 + 0.06 * Math.sin(t / 220 + st.index);
      st.ring.material = markerMat;
      st.ring.scale.setScalar(active === st.index ? pulse : 1);
      // flash scale back
      for (const b of st.buttons.concat([st.confirm])) {
        if (b.flashUntil && t > b.flashUntil) {
          b.face.scale.setScalar(1);
          b.flashUntil = 0;
        }
      }
    }
  }

  function setOverlay(group) {
    // The hub passes a boolean (research overlay on/off); this hall has no
    // extra overlay, so only accept an actual Object3D.
    if (group && group.isObject3D) root.add(group);
  }

  function dispose() {
    root.removeFromParent();
    if (window.vrmHub && window.vrmHub.quiz === api) delete window.vrmHub.quiz;
    for (const rec of boardCanvas.values()) {
      rec.tex.dispose();
      rec.mat.dispose();
    }
    glowTex.dispose();
    glowGeo.dispose();
    buttonGeo.dispose();
    confirmGeo.dispose();
    for (const st of stations) {
      for (const b of st.buttons.concat([st.confirm])) {
        b.faceMat.map?.dispose?.();
        b.faceMat.dispose();
        b.glowMat.dispose();
      }
    }
    for (const o of owned) {
      if (o.isMaterial) o.dispose?.();
      if (o.map) o.map.dispose?.();
    }
    root.traverse((o) => {
      if (o.isMesh) {
        o.geometry?.dispose?.();
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach((m) => {
          m?.map?.dispose?.();
          m?.dispose?.();
        });
      }
    });
  }

  // ------------------------------------------------------------------ public
  const api = {
    version: '1.0',
    ready: () => ready,
    state: () => {
      if (!ready) return { ready: false, active: -1, stations: stations.map(stationPublic) };
      const st = stations[active >= 0 ? active : 0];
      return publicState(st);
    },
    stationState: (i) => {
      const st = stations[i];
      return st ? publicState(st) : null;
    },
    events: () => events.slice(),
    trials: () => trials.slice(),
    handTarget: (key, stationIndex) => {
      const st = stations[stationIndex === undefined || stationIndex === null
        ? (active >= 0 ? active : 0) : stationIndex];
      if (!st || !st.question) return null;
      const b = key === CONFIRM_KEY ? st.confirm
        : st.buttons.find((x) => x.key === key);
      if (!b) return null;
      const p = b.world.clone().addScaledVector(b.normal, 0.06);
      return {
        key, station: st.index, position: p.toArray(),
        surface: b.world.toArray(), normal: b.normal.toArray(),
      };
    },
    setActive: (i) => {
      forcedActive = (i === null || i === undefined || i < 0) ? null : Number(i);
      active = forcedActive === null ? -1 : forcedActive;
      stations.forEach((st) => {
        st.boardMat.color.setHex(st.index === active ? 0xffffff : 0x717a8c);
      });
      return active;
    },
    press: (key, stationIndex) => pressAt(key, stationIndex),
    debug: {
      snapshot: () => ({
        ready, active, forcedActive,
        stations: stations.map((st) => ({
          index: st.index, stand: [st.x, 0, STAND_Z],
          question: st.question ? {
            ...st.question,
            buttons: st.buttons.slice(0, st.question.options.length).map((b) => ({
              key: b.key, local: b.local, world: b.world.toArray(),
              normal: b.normal.toArray(),
            })),
          } : null,
          selections: [...st.selections], eliminated: [...st.eliminated],
          done: st.done, correct: st.correct,
          pressed: st.pressed.slice(), lastResult: st.lastResult,
          autoNextAt: st.autoNextAt, retryAt: st.retryAt,
        })),
        global: typeof window !== 'undefined' && window.vrmHub
          ? {
            avatar: window.vrmHub.avatar?.root.position.toArray() ?? null,
            hand: window.vrmHub.avatar?.getNode?.('rightHand')
              ? window.vrmHub.avatar.getNode('rightHand').getWorldPosition(new THREE.Vector3()).toArray()
              : null,
            heading: window.vrmHub.state?.avatar?.heading ?? null,
          } : null,
        events: events.slice(-40),
      }),
      layout: () => stations.map((st) => ({
        index: st.index, x: st.x, stand: [st.x, 0, STAND_Z],
        buttons: st.buttons.map((b) => ({
          key: b.key, local: b.local, world: b.world.toArray(),
          visible: b.face.visible, eliminated: !!b.eliminated,
        })),
        eliminated: [...st.eliminated],
        confirm: { local: st.confirm.local, world: st.confirm.world.toArray(),
          visible: st.confirm.face.visible },
      })),
      boardRect: (stationIndex) => {
        const i = stationIndex === undefined || stationIndex === null
          ? (active >= 0 ? active : 0) : stationIndex;
        return boardRectNorm(stations[i]);
      },
      // Evaluation hook: redraw one station's board with a specific option
      // order.  The RL training set needs the same live-rendered board under
      // balanced letter permutations; no game state is finalized here.
      setOptions: (stationIndex, options) => {
        const i = stationIndex === undefined || stationIndex === null
          ? (active >= 0 ? active : 0) : stationIndex;
        const st = stations[i];
        if (!st || !st.question) return null;
        st.question = {
          ...st.question,
          options: options.map((o) => ({ key: String(o.key), text: String(o.text) })),
        };
        layout(st);
        drawQuestionBoard(st);
        return st.question.options.map((o) => ({ key: o.key, text: o.text }));
      },
      pressAt,
      boardPng: (stationIndex) => {
        const i = stationIndex === undefined || stationIndex === null
          ? (active >= 0 ? active : 0) : stationIndex;
        const rec = boardCanvas.get(i);
        return rec ? rec.canvas.toDataURL('image/png') : null;
      },
      next: (stationIndex) => {
        const i = stationIndex === undefined || stationIndex === null
          ? (active >= 0 ? active : 0) : stationIndex;
        setQuestion(stations[i], stations[i].qIndex + 1);
        return stations[i].qIndex;
      },
      reset: () => {
        events.length = 0;
        trials.length = 0;
        stations.forEach((st, i) => {
          st.eliminated = new Set();
          st.elimQid = null;
          setQuestion(st, i * 5);
        });
      },
    },
  };

  fetch(BANK_URL).then((r) => r.json()).then((data) => {
    bank = data.questions || [];
    ready = true;
    stations.forEach((st, i) => setQuestion(st, st.qIndex));
    stations.forEach((st) => { st.startedAt = now(); });
    events.push({ type: 'bank', n: bank.length, t: now() });
  }).catch((err) => {
    console.warn('quiz bank failed', err);
    events.push({ type: 'error', message: String(err), t: now() });
  });

  if (window.vrmHub) window.vrmHub.quiz = api;

  return {
    root,
    update,
    setOverlay,
    dispose,
    meta: { name: manifest.name, kind: 'quiz', source: 'vrm-hub' },
  };
}
