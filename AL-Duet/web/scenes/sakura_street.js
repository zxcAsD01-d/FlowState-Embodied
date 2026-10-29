import * as THREE from 'three';

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function radialTexture(color) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 128;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(64, 64, 4, 64, 64, 64);
  grad.addColorStop(0, color);
  grad.addColorStop(0.45, color.replace('1)', '0.55)'));
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export function createScene({ manifest, scene }) {
  const visual = manifest.visual || {};
  const rng = mulberry32(visual.seed ?? 7);
  const rand = (a, b) => a + (b - a) * rng();
  const root = new THREE.Group();
  root.name = 'sakura_street';
  const overlay = new THREE.Group();
  overlay.name = 'research_overlay';
  overlay.visible = false;
  root.add(overlay);

  const owned = [];

  // ------------------------------------------------------------------ lights
  const hemi = new THREE.HemisphereLight(0xa8c2ff, 0x241a26, 1.15);
  const key = new THREE.DirectionalLight(0xffe7f0, 2.4);
  key.position.set(14, 22, 9);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -26;
  key.shadow.camera.right = 26;
  key.shadow.camera.top = 26;
  key.shadow.camera.bottom = -26;
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 70;
  key.shadow.bias = -0.00035;
  key.shadow.normalBias = 0.03;
  key.target.position.set(0, 0, 0);
  const rim = new THREE.DirectionalLight(0x7fb4ff, 1.35);
  rim.position.set(-16, 9, -18);
  root.add(hemi, key, key.target, rim);

  // ---------------------------------------------------------------- materials
  const groundMat = new THREE.MeshStandardMaterial({ color: 0x0a101b, roughness: 0.96, metalness: 0.02 });
  const roadMat = new THREE.MeshStandardMaterial({ color: 0x151c29, roughness: 0.9, metalness: 0.03 });
  const walkMat = new THREE.MeshStandardMaterial({ color: 0x1b2434, roughness: 0.94 });
  const curbMat = new THREE.MeshStandardMaterial({ color: 0x2a3547, roughness: 0.9 });
  const barkMat = new THREE.MeshStandardMaterial({ color: 0x3b2b2e, roughness: 0.95 });
  const buildingMat = new THREE.MeshStandardMaterial({ color: 0x151d2d, roughness: 0.85, metalness: 0.05 });
  const vermilion = new THREE.MeshStandardMaterial({ color: 0xc9503f, roughness: 0.72 });
  const woodMat = new THREE.MeshStandardMaterial({ color: 0x5a4030, roughness: 0.85 });
  const lampMat = new THREE.MeshStandardMaterial({
    color: 0xffd9a8, emissive: 0xffa54f, emissiveIntensity: 2.4, roughness: 0.4,
  });
  const windowMat = new THREE.MeshBasicMaterial({
    color: 0xffd7a0, transparent: true, opacity: 0.55,
  });
  const dashMat = new THREE.MeshBasicMaterial({
    color: 0x9fb6ce, transparent: true, opacity: 0.3,
  });
  owned.push(groundMat, roadMat, walkMat, curbMat, barkMat, buildingMat,
    vermilion, woodMat, lampMat, windowMat, dashMat);

  const box = (w, h, d, mat, x, y, z, parent = root) => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    mesh.position.set(x, y, z);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
  };
  const plane = (w, d, mat, x, y, z, parent = root) => {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, d), mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(x, y, z);
    mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
  };

  // ------------------------------------------------------------------ ground
  plane(260, 260, groundMat, 0, 0, 0);
  plane(150, 7.0, roadMat, 0, 0.012, 0);
  plane(150, 2.8, walkMat, 0, 0.014, 5.1);
  plane(150, 2.8, walkMat, 0, 0.014, -5.1);
  box(150, 0.12, 0.16, curbMat, 0, 0.06, 3.55);
  box(150, 0.12, 0.16, curbMat, 0, 0.06, -3.55);

  for (let x = -72; x <= 72; x += 6.5) {
    const dash = plane(2.4, 0.11, dashMat, x, 0.022, 0);
    dash.castShadow = false;
  }
  // crosswalk
  for (let i = 0; i < 7; i++) {
    const stripe = box(0.55, 0.014, 5.4, dashMat, 3.4 + i * 0.82, 0.022, 0);
    stripe.castShadow = false;
  }

  // ------------------------------------------------------------------- trees
  const trees = [];
  const treeCfg = visual.trees || {};
  const treeCount = treeCfg.count ?? 9;
  const spacing = treeCfg.spacing ?? 8.5;
  const treeZ = treeCfg.z ?? 5.6;
  const xStart = treeCfg.x_start ?? -34;
  for (const side of [1, -1]) {
    for (let i = 0; i < treeCount; i++) {
      trees.push({
        x: xStart + i * spacing + rand(-1.2, 1.2),
        z: side * (treeZ + rand(-0.35, 0.35)),
        scale: rand(0.86, 1.16),
        seed: rng(),
      });
    }
  }

  const trunkGeo = new THREE.CylinderGeometry(0.11, 0.2, 2.8, 8);
  const branchGeo = new THREE.CylinderGeometry(0.035, 0.085, 1.5, 6);
  owned.push(trunkGeo, branchGeo);
  for (const t of trees) {
    const g = new THREE.Group();
    g.position.set(t.x, 0, t.z);
    g.scale.setScalar(t.scale);
    const trunk = new THREE.Mesh(trunkGeo, barkMat);
    trunk.position.y = 1.4;
    trunk.rotation.z = (t.seed - 0.5) * 0.16;
    trunk.castShadow = true;
    trunk.receiveShadow = true;
    g.add(trunk);
    for (let b = 0; b < 3; b++) {
      const br = new THREE.Mesh(branchGeo, barkMat);
      const ang = (b / 3) * Math.PI * 2 + t.seed * 3.1;
      br.position.set(Math.cos(ang) * 0.32, 2.35 + b * 0.1, Math.sin(ang) * 0.32);
      br.rotation.set(Math.sin(ang) * 0.6, ang, Math.cos(ang) * 0.6);
      br.castShadow = true;
      g.add(br);
    }
    root.add(g);
  }

  // blossoms, one instanced draw call
  const blossomGeo = new THREE.IcosahedronGeometry(1, 1);
  const blossomMat = new THREE.MeshStandardMaterial({
    color: 0xffffff, roughness: 0.88, metalness: 0.0,
    emissive: 0x3a1220, emissiveIntensity: 0.35,
  });
  owned.push(blossomGeo, blossomMat);
  const clustersPerTree = 7;
  const blossom = new THREE.InstancedMesh(blossomGeo, blossomMat, trees.length * clustersPerTree);
  blossom.castShadow = true;
  blossom.receiveShadow = true;
  const dummy = new THREE.Object3D();
  const shades = [0xf6b3cb, 0xf3a0be, 0xfbd0df, 0xee8fb2, 0xffdbe8].map(c => new THREE.Color(c));
  let bi = 0;
  for (const t of trees) {
    for (let k = 0; k < clustersPerTree; k++) {
      const ang = rand(0, Math.PI * 2);
      const rad = rand(0.15, 1.35);
      dummy.position.set(
        t.x + Math.cos(ang) * rad * t.scale,
        (2.75 + rand(0, 1.15)) * t.scale,
        t.z + Math.sin(ang) * rad * t.scale,
      );
      const s = rand(0.42, 0.78) * t.scale;
      dummy.scale.set(s * rand(0.9, 1.3), s * rand(0.7, 1.05), s * rand(0.9, 1.3));
      dummy.rotation.set(rand(0, 3.14), rand(0, 3.14), rand(0, 3.14));
      dummy.updateMatrix();
      blossom.setMatrixAt(bi, dummy.matrix);
      blossom.setColorAt(bi, shades[Math.floor(rand(0, shades.length))]);
      bi++;
    }
  }
  blossom.instanceMatrix.needsUpdate = true;
  if (blossom.instanceColor) blossom.instanceColor.needsUpdate = true;
  root.add(blossom);

  // fallen petals
  const fallGeo = new THREE.CircleGeometry(0.06, 6);
  const fallMat = new THREE.MeshStandardMaterial({
    color: 0xf2a8c4, roughness: 0.9, transparent: true, opacity: 0.72, depthWrite: false,
  });
  owned.push(fallGeo, fallMat);
  const fallCount = 240;
  const fallen = new THREE.InstancedMesh(fallGeo, fallMat, fallCount);
  for (let i = 0; i < fallCount; i++) {
    dummy.position.set(rand(-48, 26), 0.02, rand(-6.6, 6.6));
    dummy.rotation.set(-Math.PI / 2, 0, rand(0, 6.28));
    dummy.scale.setScalar(rand(0.6, 1.4));
    dummy.updateMatrix();
    fallen.setMatrixAt(i, dummy.matrix);
  }
  fallen.instanceMatrix.needsUpdate = true;
  fallen.receiveShadow = true;
  root.add(fallen);

  // --------------------------------------------------------------- buildings
  const bCfg = visual.buildings || {};
  const bCount = bCfg.count ?? 9;
  const bSpacing = bCfg.spacing ?? 15;
  const bZ = bCfg.z ?? 12.5;
  const bStart = bCfg.x_start ?? -60;
  for (const side of [1, -1]) {
    for (let i = 0; i < bCount; i++) {
      const w = rand(6.5, 11.5);
      const h = rand(5.0, 11.5);
      const d = rand(6, 9);
      const x = bStart + i * bSpacing + rand(-1.5, 1.5);
      const z = side * (bZ + rand(-1.2, 2.4));
      box(w, h, d, buildingMat, x, h / 2, z);
      const faceZ = z - side * (d / 2 + 0.02);
      const rows = Math.max(2, Math.floor(h / 2.4));
      for (let r = 0; r < rows; r++) {
        const win = new THREE.Mesh(new THREE.PlaneGeometry(w * rand(0.35, 0.65), 0.5), windowMat);
        win.position.set(x + rand(-w * 0.18, w * 0.18), 1.4 + r * 2.1, faceZ);
        win.rotation.y = side > 0 ? Math.PI : 0;
        root.add(win);
      }
      if (rng() > 0.55) {
        const roof = box(w * 0.4, 0.5, 0.5, windowMat, x, h + 0.25, z);
        roof.castShadow = false;
      }
    }
  }

  // ----------------------------------------------------------------- lanterns
  const lCfg = visual.lanterns || {};
  const lCount = lCfg.count ?? 7;
  const lSpacing = lCfg.spacing ?? 16;
  const lampPositions = [];
  for (const side of [1, -1]) {
    for (let i = 0; i < lCount; i++) {
      const x = -48 + i * lSpacing;
      const z = side * (lCfg.z ?? 4.1);
      lampPositions.push([x, z, side]);
      const g = new THREE.Group();
      g.position.set(x, 0, z);
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.075, 3.0, 8), curbMat);
      pole.position.y = 1.5;
      pole.castShadow = true;
      g.add(pole);
      const arm = box(0.55, 0.06, 0.06, curbMat, -side * 0.0, 2.95, 0, g);
      arm.position.x = -side * 0.25;
      const lantern = box(0.24, 0.32, 0.24, lampMat, -side * 0.48, 2.82, 0, g);
      lantern.castShadow = false;
      const cap = box(0.32, 0.05, 0.32, curbMat, -side * 0.48, 3.0, 0, g);
      cap.castShadow = false;
      root.add(g);
    }
  }
  const lightSpots = lampPositions
    .map(p => ({ p, d: Math.hypot(p[0], p[1]) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, 4);
  for (const { p } of lightSpots) {
    const light = new THREE.PointLight(0xffb46b, 22, 17, 2);
    light.position.set(p[0], 2.8, p[1]);
    root.add(light);
  }

  // -------------------------------------------------------------------- torii
  const toriiX = visual.torii_x ?? -22;
  const gate = new THREE.Group();
  gate.position.set(toriiX, 0, 0);
  for (const side of [1, -1]) {
    const pillar = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.3, 5.2, 12), vermilion);
    pillar.position.set(0, 2.6, side * 3.9);
    pillar.castShadow = true;
    pillar.receiveShadow = true;
    gate.add(pillar);
    const base = box(0.8, 0.25, 0.8, curbMat, 0, 0.12, side * 3.9, gate);
    base.castShadow = false;
  }
  const topBeam = box(0.5, 0.42, 8.8, vermilion, 0, 5.0, 0, gate);
  topBeam.rotation.y = Math.PI / 2;
  const midBeam = box(0.34, 0.28, 7.3, vermilion, 0, 4.25, 0, gate);
  midBeam.rotation.y = Math.PI / 2;
  const plaque = box(0.1, 0.8, 0.9, curbMat, 0, 4.62, 0, gate);
  plaque.castShadow = false;
  root.add(gate);

  // ------------------------------------------------------------------- props
  const bench = (x, z, flip) => {
    const g = new THREE.Group();
    g.position.set(x, 0, z);
    g.rotation.y = flip ? Math.PI : 0;
    box(1.9, 0.1, 0.46, woodMat, 0, 0.46, 0, g);
    box(1.9, 0.5, 0.08, woodMat, 0, 0.72, -0.2, g);
    for (const s of [-0.75, 0.75]) {
      box(0.1, 0.44, 0.4, curbMat, s, 0.22, 0, g);
    }
    root.add(g);
  };
  bench(6.8, -4.45, false);
  bench(-11.5, 4.45, true);

  const vending = new THREE.Group();
  vending.position.set(9.6, 0, 4.55);
  box(1.0, 1.95, 0.75, new THREE.MeshStandardMaterial({ color: 0x223349, roughness: 0.6 }), 0, 0.98, 0, vending);
  const vface = new THREE.Mesh(new THREE.PlaneGeometry(0.78, 1.35),
    new THREE.MeshBasicMaterial({ color: 0x9fe9ff, transparent: true, opacity: 0.55 }));
  vface.position.set(0, 1.12, -0.381);
  vface.rotation.y = Math.PI;
  vending.add(vface);
  root.add(vending);

  // ------------------------------------------------------------------- sky
  const starGeo = new THREE.BufferGeometry();
  const starPos = new Float32Array(720 * 3);
  for (let i = 0; i < 720; i++) {
    const theta = rand(0, Math.PI * 2);
    const phi = rand(0.05, 0.62) * Math.PI;
    const r = rand(60, 95);
    starPos[i * 3] = Math.cos(theta) * Math.sin(phi) * r;
    starPos[i * 3 + 1] = Math.cos(phi) * r + 10;
    starPos[i * 3 + 2] = Math.sin(theta) * Math.sin(phi) * r;
  }
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
  const starMat = new THREE.PointsMaterial({
    color: 0xdce8ff, size: 0.45, sizeAttenuation: true, transparent: true, opacity: 0.75,
    depthWrite: false,
  });
  owned.push(starGeo, starMat);
  root.add(new THREE.Points(starGeo, starMat));

  if (visual.moon !== false) {
    const moonTex = radialTexture('rgba(255,232,244,1)');
    const moonMat = new THREE.SpriteMaterial({ map: moonTex, transparent: true, depthWrite: false });
    const moon = new THREE.Sprite(moonMat);
    moon.position.set(-44, 34, 58);
    moon.scale.set(16, 16, 1);
    root.add(moon);
    owned.push(moonTex, moonMat);
  }

  // --------------------------------------------------------- floating petals
  const pCfg = visual.petals || {};
  const petalCount = pCfg.count ?? 380;
  const petalGeo = new THREE.PlaneGeometry(0.075, 0.05);
  const petalMat = new THREE.MeshStandardMaterial({
    color: 0xf7b6d0, side: THREE.DoubleSide, roughness: 0.85,
    transparent: true, opacity: 0.94, depthWrite: false,
  });
  owned.push(petalGeo, petalMat);
  const petals = new THREE.InstancedMesh(petalGeo, petalMat, petalCount);
  petals.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  petals.frustumCulled = false;
  root.add(petals);
  const petalData = [];
  for (let i = 0; i < petalCount; i++) {
    petalData.push({
      x: rand(-48, 24),
      y: rand(0, 8.4),
      z: rand(-8.5, 8.5),
      phase: rand(0, Math.PI * 2),
      fall: rand(0.16, 0.46),
      sway: rand(0.25, 0.9),
      spin: rand(-1.1, 1.1),
      size: rand(0.7, 1.5),
    });
  }
  let petalTime = 0;

  // -------------------------------------------------------------- research FX
  const grid = new THREE.GridHelper(80, 80, 0x5b86b8, 0x22364f);
  grid.position.y = 0.035;
  grid.material.transparent = true;
  grid.material.opacity = 0.45;
  overlay.add(grid);
  const axes = new THREE.AxesHelper(2.0);
  axes.position.y = 0.04;
  overlay.add(axes);

  const trailCount = 600;
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(trailCount * 3), 3));
  trailGeo.setDrawRange(0, 0);
  const trailMat = new THREE.LineBasicMaterial({ color: 0x7ce7ff, transparent: true, opacity: 0.85 });
  owned.push(trailGeo, trailMat);
  const trail = new THREE.Line(trailGeo, trailMat);
  trail.frustumCulled = false;
  overlay.add(trail);
  const trailPts = [];

  const ringGeo = new THREE.RingGeometry(0.2, 0.28, 24);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0xffc36b, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false,
  });
  owned.push(ringGeo, ringMat);
  const rings = [];
  for (let i = 0; i < 10; i++) {
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.visible = false;
    overlay.add(ring);
    rings.push(ring);
  }

  function updatePetals(dt) {
    petalTime += dt;
    for (let i = 0; i < petalCount; i++) {
      const p = petalData[i];
      const y = 8.6 - ((p.y + petalTime * p.fall) % 8.6);
      dummy.position.set(
        p.x + Math.sin(petalTime * 0.55 + p.phase) * p.sway,
        y,
        p.z + Math.cos(petalTime * 0.37 + p.phase * 1.3) * p.sway * 0.7,
      );
      dummy.rotation.set(
        petalTime * p.spin + p.phase,
        p.phase * 2.0,
        Math.sin(petalTime * 0.6 + p.phase) * 0.9,
      );
      dummy.scale.setScalar(p.size);
      dummy.updateMatrix();
      petals.setMatrixAt(i, dummy.matrix);
    }
    petals.instanceMatrix.needsUpdate = true;
  }

  function updateOverlay(state) {
    if (!overlay.visible || !state) return;
    const rootPos = state.avatar?.root;
    if (rootPos) {
      const last = trailPts[trailPts.length - 1];
      if (!last || Math.hypot(last[0] - rootPos[0], last[2] - rootPos[2]) > 0.08) {
        trailPts.push([rootPos[0], 0.045, rootPos[2]]);
        if (trailPts.length > trailCount) trailPts.shift();
        const arr = trailGeo.attributes.position;
        trailPts.forEach((p, i) => arr.setXYZ(i, p[0], p[1], p[2]));
        arr.needsUpdate = true;
        trailGeo.setDrawRange(0, trailPts.length);
      }
    }
    const contacts = state.physics?.contacts || [];
    rings.forEach((ring, i) => {
      const c = contacts[i];
      if (!c) { ring.visible = false; return; }
      ring.visible = true;
      ring.position.set(c.pos[0], 0.05, c.pos[2]);
      const s = 1 + Math.min((c.force || 0) / 40, 2.4);
      ring.scale.setScalar(s);
    });
  }

  function setOverlay(on) {
    overlay.visible = !!on;
    if (!on) trailPts.length = 0;
  }

  function update(dt, state) {
    updatePetals(dt);
    updateOverlay(state);
  }

  function dispose() {
    root.removeFromParent();
    root.traverse(o => {
      if (o.isMesh || o.isLine || o.isPoints) {
        o.geometry?.dispose?.();
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach(m => m?.dispose?.());
      }
    });
    owned.forEach(x => x.dispose?.());
  }

  return {
    root,
    update,
    setOverlay,
    dispose,
    lights: { hemi, key, rim },
    meta: { name: manifest.name, petals: petalCount, trees: trees.length },
  };
}
