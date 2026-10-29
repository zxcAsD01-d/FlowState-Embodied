import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/* The district is authored as ~20k individual meshes.  That is fine for a
 * single-viewport walking demo on a fast GPU, but the hub draws the same scene
 * three times a frame (observer, first person, sun shadow), and every mesh is a
 * draw call: ~18k calls, ~10 fps.  This pass merges static opaque surfaces into
 * a handful of draws by material signature, with the material colour baked into
 * a vertex colour so that a few hundred single-colour materials collapse into
 * one.  Animated rigs keep their pivots: each `planetRigid` subtree is merged
 * within itself, and any object referenced from `userData` (cat head, dispensed
 * can, cloth pivots) is left alone.
 *
 * Instanced meshes, transparent surfaces, textured materials and multi-material
 * meshes keep their individual draws.  They are what remains after this pass.
 */

function isUnder(o, set) {
  for (let p = o; p; p = p.parent) if (set.has(p)) return true;
  return false;
}

function collectReferenced(scene) {
  const set = new Set();
  const take = (v) => {
    if (v && v.isObject3D) set.add(v);
    else if (Array.isArray(v)) for (const x of v) take(x);
  };
  scene.traverse((o) => {
    const u = o.userData;
    if (!u) return;
    for (const k of Object.keys(u)) take(u[k]);
  });
  return set;
}

function colorAttribute(geo, color) {
  const n = geo.attributes.position.count;
  const data = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    data[i * 3] = color.r;
    data[i * 3 + 1] = color.g;
    data[i * 3 + 2] = color.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(data, 3));
  return geo;
}

function cleanGeometry(src, keepUV) {
  const g = src.index ? src.toNonIndexed() : src.clone();
  for (const name of Object.keys(g.attributes)) {
    if (name !== 'position' && name !== 'normal' && !(keepUV && name === 'uv')) g.deleteAttribute(name);
  }
  if (!g.attributes.normal) g.computeVertexNormals();
  return g;
}

/* One `geometry.groups` range as a standalone triangle soup, so that each
 * material of a multi-material mesh can join the bucket its signature belongs
 * to.  `start`/`count` address the index buffer when the source is indexed and
 * the vertex buffer otherwise, which is how the renderer reads groups. */
function groupGeometry(src, start, count, keepUV) {
  const idx = src.index;
  const pos = src.attributes.position;
  const nor = src.attributes.normal;
  const uv = keepUV ? src.attributes.uv : null;
  const total = (idx ? idx.count : pos.count) - start;
  const n = Math.max(0, Math.min(count, total));
  const P = new Float32Array(n * 3);
  const N = nor ? new Float32Array(n * 3) : null;
  const U = uv ? new Float32Array(n * 2) : null;
  for (let i = 0; i < n; i++) {
    const v = idx ? idx.getX(start + i) : start + i;
    P[i * 3] = pos.getX(v);
    P[i * 3 + 1] = pos.getY(v);
    P[i * 3 + 2] = pos.getZ(v);
    if (N) {
      N[i * 3] = nor.getX(v);
      N[i * 3 + 1] = nor.getY(v);
      N[i * 3 + 2] = nor.getZ(v);
    }
    if (U) {
      U[i * 2] = uv.getX(v);
      U[i * 2 + 1] = uv.getY(v);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(P, 3));
  if (N) g.setAttribute('normal', new THREE.BufferAttribute(N, 3));
  if (U) g.setAttribute('uv', new THREE.BufferAttribute(U, 2));
  if (!N) g.computeVertexNormals();
  return g;
}

function isInkShell(o) {
  const m = o.material;
  return !!m && m.isShaderMaterial && !!m.uniforms && !!m.uniforms.uThickness && !!m.uniforms.uColor;
}

function solidKey(m) {
  return [
    m.type, m.side, m.fog ? 1 : 0,
    m.gradientMap ? m.gradientMap.uuid : '-',
    m.emissive ? m.emissive.getHexString() : '-', m.emissiveIntensity,
    m.alphaTest, m.depthWrite ? 1 : 0,
    m.map ? 'map:' + m.map.uuid : '-',
    m.map ? [m.map.offset.x, m.map.offset.y, m.map.repeat.x, m.map.repeat.y,
      m.map.rotation, m.map.center.x, m.map.center.y].join(',') : '-',
    m.toneMapped ? 1 : 0,
  ].join('|');
}

/* Foliage is instanced, but one InstancedMesh per canopy type covers the whole
 * 160 m district.  Most of those instances sit >110 m from the street and are
 * barely a screen pixel wide, yet every pass submits all of them (4.1 M
 * triangles a pass).  Split each big field into near / mid / far by distance
 * from the walking street, drop the far field to every third instance, and stop
 * distant instances casting sun shadows they can never reach. */
export function thinInstances(root, { near = 60, mid = 110, farStride = 3 } = {}) {
  root.updateMatrixWorld(true);
  const inv = new THREE.Matrix4().copy(root.matrixWorld).invert();
  const rel = new THREE.Matrix4();
  const m4 = new THREE.Matrix4();
  const p = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  const ANCHOR = { x: 0, z: 11 };

  const rigids = new Set();
  root.traverse((o) => {
    if (o.userData && (o.userData.planetRigid || o.userData.planetSpin)) rigids.add(o);
  });

  const jobs = [];
  root.traverse((o) => {
    if (!o.isInstancedMesh || o.count < 64) return;
    if (isUnder(o, rigids)) return;
    const g = o.geometry;
    const tris = ((g.index ? g.index.count : g.attributes.position.count) / 3) * o.count;
    if (tris < 4000) return;
    jobs.push(o);
  });

  const stats = { fields: 0, before: 0, after: 0 };
  for (const o of jobs) {
    rel.multiplyMatrices(inv, o.matrixWorld);
    const arr = o.instanceMatrix.array;
    const nearI = [];
    const midI = [];
    const farI = [];
    for (let i = 0; i < o.count; i++) {
      m4.fromArray(arr, i * 16);
      p.setFromMatrixPosition(m4);
      p.applyMatrix4(rel);
      const d = Math.hypot(p.x - ANCHOR.x, p.z - ANCHOR.z);
      if (d <= near) nearI.push(i);
      else if (d <= mid) midI.push(i);
      else farI.push(i);
    }
    const keptFar = farI.filter((_, k) => k % farStride === 0);
    if (!midI.length && keptFar.length === farI.length) continue;

    const cloneField = (indices, tag) => {
      if (!indices.length) return null;
      const im = new THREE.InstancedMesh(o.geometry, o.material, indices.length);
      im.name = (o.name || 'instanced') + ':' + tag;
      im.position.copy(o.position);
      im.quaternion.copy(o.quaternion);
      im.scale.copy(o.scale);
      for (let k = 0; k < indices.length; k++) im.setMatrixAt(k, m4.fromArray(arr, indices[k] * 16));
      im.instanceMatrix.needsUpdate = true;
      if (o.instanceColor) {
        im.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(indices.length * 3), 3);
        for (let k = 0; k < indices.length; k++) {
          im.instanceColor.setXYZ(k,
            o.instanceColor.getX(indices[k]),
            o.instanceColor.getY(indices[k]),
            o.instanceColor.getZ(indices[k]));
        }
      }
      im.castShadow = false;
      im.receiveShadow = o.receiveShadow;
      im.computeBoundingSphere();
      im.frustumCulled = true;
      o.parent.add(im);
      return im;
    };

    const before = o.count;
    cloneField(midI, 'mid');
    cloneField(keptFar, 'far');
    for (let k = 0; k < nearI.length; k++) {
      m4.fromArray(arr, nearI[k] * 16);
      m4.toArray(o.instanceMatrix.array, k * 16);
    }
    o.count = nearI.length;
    o.instanceMatrix.needsUpdate = true;
    o.computeBoundingSphere();
    if (!nearI.length) o.visible = false;

    stats.fields++;
    stats.before += before;
    stats.after += o.count + midI.length + keptFar.length;
  }
  return stats;
}

export function batchStaticWorld(root) {
  root.updateMatrixWorld(true);
  const referenced = collectReferenced(root);

  const rigids = [];
  root.traverse((o) => {
    if (o.userData && (o.userData.planetRigid || o.userData.planetSpin)) rigids.push(o);
  });
  const rigidSet = new Set(rigids);

  const solid = new Map();
  const glass = new Map();
  const ink = new Map();
  const victims = [];
  const detach = [];

  const keyWorld = new THREE.Matrix4();

  const bucketFor = (owner, m) => {
    const target = m.transparent ? glass : solid;
    let g = target.get(owner);
    if (!g) { g = new Map(); target.set(owner, g); }
    const key = solidKey(m) + (m.transparent
      ? '|t' + m.opacity + '|b' + m.blending + '|d' + (m.depthWrite ? 1 : 0) + '|a' + m.alphaHash
      : '');
    let bucket = g.get(key);
    if (!bucket) {
      const mat = m.clone();
      mat.color.setRGB(1, 1, 1);
      mat.vertexColors = true;
      mat.needsUpdate = true;
      bucket = { mat, geos: [], cast: false, receive: false };
      g.set(key, bucket);
    }
    return bucket;
  };

  /* geo is already in the owner's space. */
  const addSolid = (owner, m, geo, cast, receive) => {
    if (!m || o_bad(m)) return false;
    if (m.map && !geo.attributes.uv) return false;
    colorAttribute(geo, m.color);
    const bucket = bucketFor(owner, m);
    bucket.geos.push(geo);
    bucket.cast = bucket.cast || cast;
    bucket.receive = bucket.receive || receive;
    return true;
  };
  function o_bad(m) {
    return !m.isMeshToonMaterial && !m.isMeshBasicMaterial;
  }

  /* One pass over a subtree, merging into `bucketOwner` (root for the static
   * world, a rigid pivot for animated props).  Coordinates are baked relative
   * to the owner, so the owner keeps its own transform and animation. */
  const sweep = (owner, { staticWorld }) => {
    const inv = new THREE.Matrix4().copy(owner.matrixWorld).invert();
    const insideNestedRigid = (o) => {
      for (let p = o.parent; p && p !== owner; p = p.parent) if (rigidSet.has(p)) return true;
      return false;
    };
    const blocked = (o) => {
      if (isUnder(o, referenced)) return true;
      if (staticWorld ? isUnder(o, rigidSet) : insideNestedRigid(o)) return true;
      return false;
    };
    owner.traverse((o) => {
      if (o === owner) return;
      if (!o.isMesh || o.isInstancedMesh || o.isSkinnedMesh) return;
      if (o.userData.batchSkip || !o.visible) return;
      if (blocked(o)) return;

      if (Array.isArray(o.material)) {
        const groups = o.geometry.groups;
        if (!groups || !groups.length) return;
        const parts = [];
        for (const gr of groups) {
          const m = o.material[gr.materialIndex];
          if (!m || m.transparent || o_bad(m) || m.map && !o.geometry.attributes.uv) { parts.length = 0; break; }
          parts.push({ m, geo: groupGeometry(o.geometry, gr.start, gr.count, !!m.map) });
        }
        if (!parts.length) return;
        keyWorld.multiplyMatrices(inv, o.matrixWorld);
        let ok = true;
        for (const part of parts) {
          part.geo.applyMatrix4(keyWorld);
          if (!addSolid(owner, part.m, part.geo, o.castShadow, o.receiveShadow)) ok = false;
        }
        if (ok) victims.push(o);
        return;
      }

      const m = o.material;
      if (!m || o_bad(m)) return;
      const geo = cleanGeometry(o.geometry, !!m.map);
      keyWorld.multiplyMatrices(inv, o.matrixWorld);
      geo.applyMatrix4(keyWorld);
      if (addSolid(owner, m, geo, o.castShadow, o.receiveShadow)) victims.push(o);
    });

    owner.traverse((o) => {
      if (o === owner) return;
      if (!o.isMesh || o.isInstancedMesh || o.isSkinnedMesh) return;
      if (Array.isArray(o.material)) return;
      const m = o.material;
      if (!m || o.userData.batchSkip || !isInkShell(o)) return;
      if (isUnder(o, referenced)) return;
      if (staticWorld ? isUnder(o, rigidSet) : insideNestedRigid(o)) return;
      const u = m.uniforms;
      if (!u || !u.uThickness || !u.uColor) return;
      const key = [u.uThickness.value, u.uColor.value.getHexString(), u.uOpacity.value].join('|');
      let bucket = ink.get(owner) || new Map();
      ink.set(owner, bucket);
      let g = bucket.get(key);
      if (!g) {
        g = {
          mat: new THREE.ShaderMaterial({
            uniforms: {
              uThickness: { value: u.uThickness.value },
              uColor: { value: u.uColor.value.clone() },
              uOpacity: { value: u.uOpacity.value },
              uResolution: { value: u.uResolution.value.clone() },
            },
            vertexShader: m.vertexShader,
            fragmentShader: m.fragmentShader,
            side: m.side,
            transparent: m.transparent,
            depthWrite: m.depthWrite,
            fog: false,
          }),
          geos: [],
        };
        bucket.set(key, g);
      }
      const geo = cleanGeometry(o.geometry);
      keyWorld.multiplyMatrices(inv, o.matrixWorld);
      geo.applyMatrix4(keyWorld);
      g.geos.push(geo);
      detach.push(o);
    });
  };

  // Static world, then each rigid pivot internally.
  sweep(root, { staticWorld: true });
  for (const pivot of rigids) {
    if (pivot.parent && isUnder(pivot.parent, rigidSet)) continue;
    sweep(pivot, { staticWorld: false });
  }

  // Detach shells first: they live under the meshes about to be removed.
  for (const o of detach) o.removeFromParent();

  // Remove batched meshes; keep any real children by lifting them a level.
  for (const o of victims) {
    if (!o.parent) continue;
    for (const child of [...o.children]) o.parent.attach(child);
    o.removeFromParent();
  }

  let merged = 0;
  const addMerged = (owner, key, group, mode) => {
    let geo = null;
    try {
      geo = mergeGeometries(group.geos, false);
    } catch (e) {
      console.warn('sakura batch: merge failed for', key, e);
    }
    const cast = mode === 'solid' ? group.cast : false;
    const receive = mode === 'ink' ? false : group.receive;
    if (!geo) {
      for (const g of group.geos) {
        const mesh = new THREE.Mesh(g, group.mat);
        mesh.name = 'batch:' + key;
        mesh.castShadow = cast;
        mesh.receiveShadow = receive;
        if (mode === 'ink') mesh.renderOrder = -1;
        owner.add(mesh);
      }
      return;
    }
    for (const g of group.geos) g.dispose();
    const mesh = new THREE.Mesh(geo, group.mat);
    mesh.name = 'batch:' + key;
    mesh.castShadow = cast;
    mesh.receiveShadow = receive;
    if (mode === 'ink') mesh.renderOrder = -1;
    owner.add(mesh);
    merged += group.geos.length;
  };

  let draws = 0;
  for (const [owner, groups] of solid) {
    for (const [key, group] of groups) { addMerged(owner, key, group, 'solid'); draws++; }
  }
  for (const [owner, groups] of glass) {
    for (const [key, group] of groups) { addMerged(owner, key, group, 'glass'); draws++; }
  }
  for (const [owner, groups] of ink) {
    for (const [key, group] of groups) { addMerged(owner, key, group, 'ink'); draws++; }
  }

  return { merged, draws, rigids: rigids.length };
}
