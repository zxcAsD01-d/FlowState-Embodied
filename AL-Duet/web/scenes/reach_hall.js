// Reach hall: one green cursor in front of the avatar.  The cursor is where a
// demonstrated ARDY walk ended (see tools/cursor_demos.py); the page only owns
// the visual marker and the head gaze, the motion is the live ARDY stream.
//
// Contract: createScene({ manifest, scene, camera, fpCamera, renderer })
//   -> { root, update(dt, serverState), setOverlay(group), dispose(), meta }

import * as THREE from 'three';

export function createScene({ manifest, scene, camera, renderer }) {
  const root = new THREE.Group();
  root.name = 'reach_hall';
  const owned = [];

  const hemi = new THREE.HemisphereLight(0x9db8ff, 0x10141c, 0.9);
  const key = new THREE.DirectionalLight(0xfff0dd, 1.7);
  key.position.set(6, 9, 4);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -12;
  key.shadow.camera.right = 12;
  key.shadow.camera.top = 12;
  key.shadow.camera.bottom = -12;
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 40;
  key.shadow.bias = -0.0004;
  const rim = new THREE.DirectionalLight(0x5a7dff, 1.0);
  rim.position.set(-6, 6, -7);
  root.add(hemi, key, key.target, rim);
  owned.push(hemi, key, rim);

  const floorMat = new THREE.MeshStandardMaterial({ color: 0x0f1622, roughness: 0.9 });
  const gridMat = new THREE.MeshBasicMaterial({ color: 0x1f3048, transparent: true, opacity: 0.55 });
  const wallMat = new THREE.MeshStandardMaterial({ color: 0x121a28, roughness: 0.95 });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(26, 26), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  root.add(floor);
  const grid = new THREE.GridHelper(26, 52, 0x2a4166, 0x1a283c);
  grid.position.y = 0.002;
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  grid.material.depthWrite = false;
  root.add(grid);
  const wall = new THREE.Mesh(new THREE.PlaneGeometry(26, 6), wallMat);
  wall.position.set(0, 3, 12);
  wall.rotation.y = Math.PI;
  root.add(wall);
  owned.push(floorMat, gridMat, wallMat);

  // --------------------------------------------------------------- cursor
  const cursor = new THREE.Group();
  const orbMat = new THREE.MeshStandardMaterial({
    color: 0x22ff88, emissive: 0x16cc66, emissiveIntensity: 2.2,
    roughness: 0.25, metalness: 0.1,
  });
  const orb = new THREE.Mesh(new THREE.SphereGeometry(0.11, 24, 18), orbMat);
  orb.castShadow = true;
  const haloMat = new THREE.MeshBasicMaterial({
    color: 0x22ff88, transparent: true, opacity: 0.18,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const halo = new THREE.Mesh(new THREE.SphereGeometry(0.24, 24, 18), haloMat);
  const light = new THREE.PointLight(0x22ff88, 0.6, 1.4, 2.0);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x22ff88, transparent: true, opacity: 0.5,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.18, 0.24, 40), ringMat);
  ring.rotation.x = -Math.PI / 2;
  cursor.add(orb, halo, light, ring);
  cursor.position.set(0, 1.0, 2.0);
  root.add(cursor);
  owned.push(orbMat, haloMat, ringMat);

  let target = new THREE.Vector3(0, 1.0, 2.0);
  let t0 = performance.now();
  let cursorHidden = false;
  const setCursorVisible = (on) => {
    cursorHidden = !on;
    cursor.visible = !!on;
    return cursor.visible;
  };
  const api = {
    setCursor: (p) => {
      if (!p || p.length < 3) return null;
      target.set(Number(p[0]), Number(p[1]), Number(p[2]));
      setCursorVisible(true);
      return target.toArray();
    },
    setVisible: setCursorVisible,
    state: () => ({
      cursor: target.toArray(),
      distance: Math.hypot(target.x, target.z),
      y: target.y,
    }),
    clear: () => { target.set(0, 1.0, 2.0); return target.toArray(); },
  };
  if (window.vrmHub) window.vrmHub.reach = api;

  // ------------------------------------------------- vision pattern wall
  // Nine colored/shape blocks on a wall at z=6.  The prompt names a block; the
  // head lookAt turns to it and the test measures the face-target angle.
  const visionDefs = [
    { x: -2.2, y: 2.45, color: '#ff4d4d', shape: 'circle' },
    { x: -1.1, y: 2.45, color: '#4dff88', shape: 'square' },
    { x: 0.0, y: 2.45, color: '#4da6ff', shape: 'triangle' },
    { x: 1.1, y: 2.45, color: '#ffd24d', shape: 'diamond' },
    { x: 2.2, y: 2.45, color: '#c34dff', shape: 'circle' },
    { x: -1.65, y: 1.55, color: '#ff8a4d', shape: 'square' },
    { x: -0.55, y: 1.55, color: '#4dfff0', shape: 'triangle' },
    { x: 0.55, y: 1.55, color: '#ff4dc4', shape: 'diamond' },
    { x: 1.65, y: 1.55, color: '#b8ff4d', shape: 'circle' },
  ];
  const visionBoard = new THREE.Mesh(
    new THREE.PlaneGeometry(6.4, 3.2),
    new THREE.MeshStandardMaterial({ color: 0x0c1422, roughness: 0.9 }));
  visionBoard.position.set(0, 1.95, 6);
  visionBoard.rotation.y = Math.PI;
  root.add(visionBoard);
  const visionBlocks = [];
  visionDefs.forEach((d, i) => {
    const c = document.createElement('canvas');
    c.width = c.height = 256;
    const g = c.getContext('2d');
    g.clearRect(0, 0, 256, 256);
    g.fillStyle = d.color;
    g.strokeStyle = '#f4f8ff';
    g.lineWidth = 10;
    g.beginPath();
    if (d.shape === 'circle') g.arc(128, 128, 96, 0, Math.PI * 2);
    else if (d.shape === 'square') g.rect(32, 32, 192, 192);
    else if (d.shape === 'diamond') {
      g.moveTo(128, 20); g.lineTo(236, 128); g.lineTo(128, 236); g.lineTo(20, 128);
      g.closePath();
    } else {
      g.moveTo(128, 24); g.lineTo(236, 226); g.lineTo(20, 226); g.closePath();
    }
    g.fill();
    g.stroke();
    const mat = new THREE.MeshBasicMaterial({
      map: new THREE.CanvasTexture(c), transparent: true,
    });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 0.62), mat);
    mesh.position.set(d.x, d.y, 5.96);
    mesh.rotation.y = Math.PI;
    root.add(mesh);
    visionBlocks.push({
      id: i, color: d.color, shape: d.shape,
      world: new THREE.Vector3(d.x, d.y, 5.96), mesh,
    });
  });
  const vision = { target: null };
  const visionApi = {
    targets: visionBlocks.map((b) => ({
      id: b.id, color: b.color, shape: b.shape, world: b.world.toArray(),
    })),
    setTarget: (id) => {
      const b = visionBlocks.find((x) => x.id === Number(id));
      vision.target = b || null;
      return b ? { id: b.id, color: b.color, shape: b.shape, world: b.world.toArray() } : null;
    },
    clear: () => { vision.target = null; return null; },
    state: () => (vision.target
      ? { id: vision.target.id, color: vision.target.color,
          shape: vision.target.shape, world: vision.target.world.toArray() }
      : null),
  };
  if (window.vrmHub) window.vrmHub.vision = visionApi;


  function update(dt, state) {
    if (window.vrmHub) {
      if (window.vrmHub.reach !== api) window.vrmHub.reach = api;
      if (window.vrmHub.vision !== visionApi) window.vrmHub.vision = visionApi;
    }
    const t = performance.now();
    // ease the marker to the requested point (the demo cursor is set once)
    cursor.position.lerp(target, 1 - Math.exp(-dt * 10));
    // Once the body is on the cursor, the green orb/light sits inside it and
    // washes the model / FP eye green (and the additive halo pops through the
    // swaying body).  Hide the guide on arrival; show it again when a new
    // target is set far away.
    const root = state?.avatar?.root;
    if (root) {
      const d = Math.hypot(root[0] - cursor.position.x,
                           root[2] - cursor.position.z);
      if (!cursorHidden && d < 0.45) setCursorVisible(false);
      else if (cursorHidden && d > 0.8) setCursorVisible(true);
    }
    const pulse = 1 + 0.08 * Math.sin((t - t0) / 260);
    orb.scale.setScalar(pulse);
    halo.material.opacity = 0.14 + 0.06 * Math.sin((t - t0) / 420);
    const hub = window.vrmHub;
    const obs = hub?.observer?.state ? hub.observer.state() : null;
    let look = (obs && obs.look)
      ? { x: obs.position[0], y: obs.position[1], z: obs.position[2] }
      : (vision.target ? vision.target.world : cursor.position);
    if (!vision.target && cursorHidden) {
      // Standing on the cursor: looking at it would aim the head at its own
      // pelvis (the FP eye showed the model's skirt/hair).  Look at the
      // pattern wall ahead instead.
      look = { x: cursor.position.x, y: 1.7, z: cursor.position.z + 4.0 };
    }
    if (hub?.setLookAt) hub.setLookAt([look.x, look.y, look.z]);
  }

  return {
    root,
    update,
    setOverlay() {},
    dispose() {
      if (window.vrmHub?.reach === api) delete window.vrmHub.reach;
      if (window.vrmHub?.setLookAt) window.vrmHub.setLookAt(null);
      if (window.vrmHub?.testCam === camApi) delete window.vrmHub.testCam;
      grid.geometry.dispose();
      grid.material.dispose();
      owned.forEach((m) => m.dispose?.());
      root.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material && !owned.includes(o.material)) o.material.dispose?.();
      });
      scene.remove(root);
    },
    meta: { name: manifest.name, kind: 'reach', source: 'vrm-hub' },
  };
}
