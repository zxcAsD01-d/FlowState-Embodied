import * as THREE from 'three';
import { PAL } from './sakura_crossing/core/palette.js';
import { buildSky } from './sakura_crossing/core/sky.js';
import { buildWorld } from './sakura_crossing/world/index.js';
import { batchStaticWorld, thinInstances } from './sakura_crossing/batch.js';

/* Sakura Crossing (MIT, AL-0B/repos/sakura-crossing) as a pluggable hub scene.
 *
 * The source world is authored on a flat XZ plane and then wrapped onto a
 * 160 m planet.  The hub only needs the street around the level crossing, so
 * the planet bake is skipped (flat mode) and the whole block is turned a
 * quarter turn: the authored street runs north along +z, while the SMPL clip
 * walks along its own -x.  After the turn the avatar walks up the street.
 * The +8 m offset parks the crossing behind the start of the clip.
 */
export function createScene({ manifest, scene, camera, fpCamera, renderer }) {
  const root = new THREE.Group();
  root.name = 'sakura_crossing';
  scene.add(root);

  const stage = new THREE.Group();
  stage.name = 'sakura_crossing_stage';
  stage.rotation.y = -Math.PI / 2;
  stage.position.x = 8;
  root.add(stage);

  const sky = buildSky(scene, 500);
  const world = buildWorld(scene, { planet: false });
  stage.add(world.root);
  const thin = thinInstances(world.root);
  const batch = batchStaticWorld(world.root);
  if (batch.merged) {
    console.info(`[sakura] instances ${thin.before} -> ${thin.after}, ` +
      `static batch ${batch.merged} meshes -> ${batch.draws} draws`);
  }

  // Two-light cel setup lifted from the source scene's entry point.
  const sun = new THREE.DirectionalLight(PAL.sun, 2.25);
  sun.position.set(-52, 62, 56);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  // Follow the avatar instead of shadowing a fixed 68 x 68 m square: the far
  // half of that square was 1270 draw calls / 5.4 M triangles per update for
  // shadows nobody can see.  36 x 36 m around the body also doubles the
  // texel density at the same map size.
  const SHADOW_HALF = 18;
  sun.shadow.camera.left = -SHADOW_HALF;
  sun.shadow.camera.right = SHADOW_HALF;
  sun.shadow.camera.top = SHADOW_HALF;
  sun.shadow.camera.bottom = -SHADOW_HALF;
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 160;
  sun.shadow.camera.updateProjectionMatrix();
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.035;
  root.add(sun, sun.target);

  const shadowFocus = new THREE.Vector3(0, 0, 0);
  let shadowFocusDirty = false;
  const SUN_OFFSET = new THREE.Vector3(-52, 62, 56);
  function placeSun() {
    sun.target.position.set(shadowFocus.x, 0, shadowFocus.z);
    sun.position.copy(sun.target.position).add(SUN_OFFSET);
    sun.target.updateMatrixWorld();
    shadowFocusDirty = false;
  }
  function setShadowFocus(p) {
    if (!p) return;
    const x = Number(p.x ?? p[0]) || 0;
    const z = Number(p.z ?? p[2]) || 0;
    if (Math.abs(x - shadowFocus.x) < 1e-4 && Math.abs(z - shadowFocus.z) < 1e-4) return;
    shadowFocus.set(x, 0, z);
    shadowFocusDirty = true;
  }

  const fill = new THREE.DirectionalLight(PAL.fill, 1.08);
  fill.position.set(48, 26, -44);
  root.add(fill, fill.target);

  const bounce = new THREE.DirectionalLight(0xd8cbe8, 0.34);
  bounce.position.set(10, -18, 40);
  root.add(bounce, bounce.target);

  const hemi = new THREE.HemisphereLight(PAL.hemiSky, PAL.hemiGround, 1.12);
  root.add(hemi);

  // The source look is untonemapped; ACES would flatten the cel bands.
  const prevTone = renderer.toneMapping;
  const prevExposure = renderer.toneMappingExposure;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.toneMappingExposure = 1;

  const prevFar = [camera.far, fpCamera.far];
  for (const cam of [camera, fpCamera]) {
    cam.far = 700;
    cam.updateProjectionMatrix();
  }

  // Research overlay lives in world space, outside the rotated stage.
  const overlay = new THREE.Group();
  overlay.name = 'research_overlay';
  overlay.visible = false;
  scene.add(overlay);

  const rings = [];
  for (let i = 0; i < 8; i++) {
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.18, 0.24, 28),
      new THREE.MeshBasicMaterial({
        color: 0xffcf6a, transparent: true, opacity: 0.85,
        side: THREE.DoubleSide, depthWrite: false,
      }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.visible = false;
    overlay.add(ring);
    rings.push(ring);
  }

  const trailCount = 240;
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(trailCount * 3), 3));
  trailGeo.setDrawRange(0, 0);
  const trail = new THREE.Line(
    trailGeo,
    new THREE.LineBasicMaterial({ color: 0x7ce7ff, transparent: true, opacity: 0.8 }),
  );
  overlay.add(trail);
  const trailPts = [];

  function setOverlay(on) {    overlay.visible = !!on;
    if (!on) {
      trailPts.length = 0;
      trailGeo.setDrawRange(0, 0);
    }
  }

  function updateOverlay(state) {
    if (!overlay.visible || !state) return;
    const p = state.avatar?.root;
    if (p) {
      const last = trailPts[trailPts.length - 1];
      if (!last || Math.hypot(last[0] - p[0], last[2] - p[2]) > 0.08) {
        trailPts.push([p[0], 0.05, p[2]]);
        if (trailPts.length > trailCount) trailPts.shift();
        const arr = trailGeo.attributes.position;
        trailPts.forEach((q, i) => arr.setXYZ(i, q[0], q[1], q[2]));
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
      ring.scale.setScalar(1 + Math.min((c.force || 0) / 40, 2.4));
    });
  }

  function update(dt, state) {
    if (shadowFocusDirty) placeSun();
    world.update(Math.min(dt, 1 / 20));
    sky.dome.position.copy(camera.position);
    sky.clouds.position.copy(camera.position);
    updateOverlay(state);
  }

  function dispose() {
    root.removeFromParent();
    scene.remove(sky.dome, sky.clouds, overlay);
    for (const obj of [root, sky.dome, sky.clouds, overlay]) {
      obj.traverse?.(o => {
        if (o.isMesh || o.isLine || o.isPoints) {
          o.geometry?.dispose?.();
          const mats = Array.isArray(o.material) ? o.material : [o.material];
          mats.forEach(m => {
            m?.map?.dispose?.();
            m?.gradientMap?.dispose?.();
            m?.dispose?.();
          });
        }
      });
    }
    renderer.toneMapping = prevTone;
    renderer.toneMappingExposure = prevExposure;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    [camera.far, fpCamera.far] = prevFar;
    camera.updateProjectionMatrix();
    fpCamera.updateProjectionMatrix();
  }

  return {
    root,
    update,
    setOverlay,
    setShadowFocus,
    dispose,
    meta: { name: manifest.name, source: 'sakura-crossing', flat: true },
  };
}
