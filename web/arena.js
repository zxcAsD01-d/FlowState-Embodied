// Shared closed-room arena used by the data renderer and the live demo.
// Geometry matches ardy_v/arena.py: floor y=0, ceiling, four walls, AABB boxes.
import * as THREE from 'three';

export function buildArena(scene, spec, opts = {}) {
  const half = Number(spec.half ?? 8);
  const ceiling = Number(spec.ceiling ?? 4);
  const root = new THREE.Group();
  root.name = 'arena';
  scene.add(root);

  scene.background = new THREE.Color(opts.background ?? '#111a26');
  scene.fog = new THREE.Fog(opts.fog ?? '#111a26', half * 2.4, half * 6.0);

  const hemi = new THREE.HemisphereLight(0xd6e6ff, 0x263242, 1.85);
  const key = new THREE.DirectionalLight(0xfff4e2, 2.35);
  key.position.set(5, 9, 3);
  const fill = new THREE.DirectionalLight(0x9dbcff, 0.75);
  fill.position.set(-6, 5, -4);
  root.add(hemi, key, fill, key.target, fill.target);

  const floorMat = new THREE.MeshStandardMaterial({
    color: 0x37485c, roughness: 0.82, metalness: 0.02,
  });
  const wallMat = new THREE.MeshStandardMaterial({
    color: 0x465c74, roughness: 0.9, metalness: 0.0,
  });
  const ceilMat = new THREE.MeshStandardMaterial({
    color: 0x283442, roughness: 0.95, metalness: 0.0,
  });

  const floor = new THREE.Mesh(new THREE.PlaneGeometry(half * 2, half * 2), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  root.add(floor);

  // Grid gives the depth/RGB encoder metric-scale cues on the ground plane.
  const grid = new THREE.GridHelper(half * 2, half * 4, 0x7f9ec0, 0x4a6078);
  grid.position.y = 0.003;
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  root.add(grid);

  const ceil = new THREE.Mesh(new THREE.PlaneGeometry(half * 2, half * 2), ceilMat);
  ceil.rotation.x = Math.PI / 2;
  ceil.position.y = ceiling;
  root.add(ceil);

  const wallGeoX = new THREE.PlaneGeometry(half * 2, ceiling);
  for (const [x, z, ry] of [
    [0, -half, 0], [0, half, Math.PI], [-half, 0, Math.PI / 2], [half, 0, -Math.PI / 2],
  ]) {
    const w = new THREE.Mesh(wallGeoX, wallMat);
    w.position.set(x, ceiling / 2, z);
    w.rotation.y = ry;
    root.add(w);
  }

  const palette = [0x8aa3bd, 0xa08ab4, 0x86a995, 0xb39a7c, 0x7f95ad, 0xa87f8b];
  const boxes = [];
  (spec.boxes || []).forEach((b, i) => {
    const geo = new THREE.BoxGeometry(b.hx * 2, b.h, b.hz * 2);
    const mat = new THREE.MeshStandardMaterial({
      color: palette[i % palette.length], roughness: 0.78, metalness: 0.04,
    });
    const m = new THREE.Mesh(geo, mat);
    m.position.set(b.x, b.h / 2, b.z);
    m.rotation.y = Number(b.yaw || 0);
    m.castShadow = true;
    m.receiveShadow = true;
    root.add(m);
    boxes.push(m);
  });

  return { root, half, ceiling, boxes };
}

export function makeTargetMarker(scene) {
  const g = new THREE.Group();
  g.name = 'target';
  const mat = new THREE.MeshStandardMaterial({
    color: 0x2bff9a, emissive: 0x18d977, emissiveIntensity: 2.2,
    roughness: 0.2, metalness: 0.0,
  });
  const orb = new THREE.Mesh(new THREE.SphereGeometry(0.10, 20, 14), mat);
  orb.position.y = 0.12;
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.18, 0.26, 36),
    new THREE.MeshBasicMaterial({
      color: 0x2bff9a, transparent: true, opacity: 0.6,
      side: THREE.DoubleSide, depthWrite: false,
    }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.006;
  const halo = new THREE.Mesh(
    new THREE.SphereGeometry(0.22, 20, 14),
    new THREE.MeshBasicMaterial({
      color: 0x2bff9a, transparent: true, opacity: 0.10,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }),
  );
  halo.position.y = 0.16;
  g.add(orb, ring, halo);
  g.traverse(o => { o.layers.set(2); });
  scene.add(g);
  return g;
}
