import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { createMannequin } from '/mannequin.js';

// --------------------------------------------------------------- Core27 rest
// ARDY CoreSkeleton27: Y-up, metres, pelvis at the origin, T-pose.  The clip's
// local rotations are only meaningful against this skeleton, so the solve needs
// it in the browser.  Mirrors ardy/skeleton/definitions.py.
const CORE_NAMES = [
  'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
  'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandEnd',
  'RightHandThumb1', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
  'LeftHandEnd', 'LeftHandThumb1',
  'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
  'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
];
const CORE_PARENTS = [
  -1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 10, 10, 4, 13, 14, 15, 16, 16,
  0, 19, 20, 21, 0, 23, 24, 25,
];
const CORE_REST = [
  [0, 0, 0],
  [0, 0.07099, -0.04733],
  [0, 0.1642, -0.06376],
  [0, 0.2585, -0.07201],
  [0, 0.35315, -0.07201],
  [0, 0.60161, -0.03652],
  [0, 0.72978, -0.01392],
  [-0.03199, 0.52592, -0.01869],
  [-0.1909, 0.52592, -0.01869],
  [-0.48634, 0.52592, -0.01869],
  [-0.71899, 0.52592, -0.01869],
  [-0.7886, 0.52592, -0.01869],
  [-0.74684, 0.50736, 0.02772],
  [0.03199, 0.52592, -0.01869],
  [0.1909, 0.52592, -0.01869],
  [0.48634, 0.52592, -0.01869],
  [0.71899, 0.52592, -0.01869],
  [0.7886, 0.52592, -0.01869],
  [0.74684, 0.50736, 0.02772],
  [-0.09492, -0.02773, 0],
  [-0.09492, -0.43985, 0],
  [-0.09492, -0.89594, 0],
  [-0.09492, -0.95441, 0.16066],
  [0.09492, -0.02773, 0],
  [0.09492, -0.43985, 0],
  [0.09492, -0.89594, 0],
  [0.09492, -0.95441, 0.16066],
];
const CORE_INDEX = new Map(CORE_NAMES.map((name, i) => [name, i]));

// Pelvis -> lowest toe joint in the rest skeleton.  Any avatar of a different
// build is scaled onto this leg length so its pelvis and its sole both sit where
// the model puts them; without it one of the two has to be off by the mismatch.
export const CORE_LEG = 0.95441;

// Single source of truth for QC tools (tools/retarget_probe.cjs reads this off
// window.vrmHub instead of copying the numbers).
export const CORE_SKELETON = {
  names: CORE_NAMES,
  parents: CORE_PARENTS,
  rest: CORE_REST,
  leg: CORE_LEG,
};

// Avatar bone -> Core27 joint.  `tail` names the avatar bone that defines this
// bone's direction (the segment it actually points along) and the matching Core
// joint; `inherit` is used for bones with no segment of their own (head, hand,
// toes), which keep their authored relation to the corrected parent instead of
// picking up an arbitrary twist.  `hips` is the frame bone: it is driven by the
// model's pelvis rotation alone, never direction-matched.
const RETARGET = [
  // The hips segment is measured as well, but only used when the pelvis is
  // explicitly matched to the Core27 template (see buildRetarget options).
  // Without it the avatar keeps its own rest pelvis frame and the constant
  // Hips->Spine template angle (34.35 deg for the VRM rigs here) stays visible.
  { dst: 'hips', src: 'Hips', tail: 'spine', coreTail: 'Spine' },
  { dst: 'spine', src: 'Spine', tail: 'chest', coreTail: 'Spine1' },
  { dst: 'chest', src: 'Spine3', tail: 'neck', coreTail: 'Neck' },
  { dst: 'neck', src: 'Neck', tail: 'head', coreTail: 'Head' },
  { dst: 'head', src: 'Head', inherit: 'neck' },
  { dst: 'leftShoulder', src: 'LeftShoulder', tail: 'leftUpperArm', coreTail: 'LeftArm' },
  { dst: 'leftUpperArm', src: 'LeftArm', tail: 'leftLowerArm', coreTail: 'LeftForeArm' },
  { dst: 'leftLowerArm', src: 'LeftForeArm', tail: 'leftHand', coreTail: 'LeftHand' },
  { dst: 'leftHand', src: 'LeftHand', inherit: 'leftLowerArm' },
  { dst: 'rightShoulder', src: 'RightShoulder', tail: 'rightUpperArm', coreTail: 'RightArm' },
  { dst: 'rightUpperArm', src: 'RightArm', tail: 'rightLowerArm', coreTail: 'RightForeArm' },
  { dst: 'rightLowerArm', src: 'RightForeArm', tail: 'rightHand', coreTail: 'RightHand' },
  { dst: 'rightHand', src: 'RightHand', inherit: 'rightLowerArm' },
  { dst: 'leftUpperLeg', src: 'LeftUpLeg', tail: 'leftLowerLeg', coreTail: 'LeftLeg' },
  { dst: 'leftLowerLeg', src: 'LeftLeg', tail: 'leftFoot', coreTail: 'LeftFoot' },
  { dst: 'leftFoot', src: 'LeftFoot', tail: 'leftToes', coreTail: 'LeftToeBase' },
  { dst: 'leftToes', src: 'LeftToeBase', inherit: 'leftFoot' },
  { dst: 'rightUpperLeg', src: 'RightUpLeg', tail: 'rightLowerLeg', coreTail: 'RightLeg' },
  { dst: 'rightLowerLeg', src: 'RightLeg', tail: 'rightFoot', coreTail: 'RightFoot' },
  { dst: 'rightFoot', src: 'RightFoot', tail: 'rightToes', coreTail: 'RightToeBase' },
  { dst: 'rightToes', src: 'RightToeBase', inherit: 'rightFoot' },
];

const IDENT = [0, 0, 0, 1];

function qmul(a, b) {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

function qinv(q) {
  return [-q[0], -q[1], -q[2], q[3]];
}

function qnorm(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  return n > 1e-8 ? [q[0] / n, q[1] / n, q[2] / n, q[3] / n] : IDENT;
}

function unitDir(from, to) {
  const v = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
  const n = Math.hypot(v[0], v[1], v[2]);
  return n > 1e-8 ? [v[0] / n, v[1] / n, v[2] / n] : [0, 1, 0];
}

function roundTo(value, digits = 2) {
  const f = 10 ** digits;
  return Math.round(value * f) / f;
}

function swing(from, to) {
  return new THREE.Quaternion()
    .setFromUnitVectors(new THREE.Vector3(...from), new THREE.Vector3(...to))
    .toArray();
}

// Rest frame of the avatar in its own character space (root at the origin, no
// root rotation): world quats, world bone positions, and each mapped bone's
// relation to its nearest mapped ancestor.
function measureRestFrame(adapter) {
  const root = adapter.root;
  const savedPos = root.position.clone();
  const savedRot = root.rotation.clone();
  root.position.set(0, 0, 0);
  root.rotation.set(0, 0, 0);
  adapter.resetPose();
  adapter.update(0);
  root.updateMatrixWorld(true);

  const rest = new Map();
  for (const entry of RETARGET) {
    const node = adapter.getNode(entry.dst);
    if (!node) continue;
    const q = node.getWorldQuaternion(new THREE.Quaternion());
    const p = node.getWorldPosition(new THREE.Vector3());
    rest.set(entry.dst, { q: q.toArray(), p: p.toArray(), node });
  }

  const dstOf = new Map();
  for (const [dst, rec] of rest) dstOf.set(rec.node, dst);
  const ancestorOf = new Map();
  for (const [dst, rec] of rest) {
    let p = rec.node.parent;
    let found = null;
    while (p && !found) {
      found = dstOf.get(p) ?? null;
      p = p.parent;
    }
    ancestorOf.set(dst, found);
  }

  root.position.copy(savedPos);
  root.rotation.copy(savedRot);
  root.updateMatrixWorld(true);
  return { rest, ancestorOf };
}

/**
 * Build the solver that turns the clip's Core27 local rotations into loco-
 * rotations for the avatar.
 *
 * Direction-based retargeting: for every bone that has a tail, the constant
 * rest swing Cw maps the avatar's rest segment onto the Core27 rest segment;
 * the model's global rotation is then applied on top, so the produced world
 * direction equals R_model * coreRestDir exactly.  `hips` keeps its own rest
 * frame (the pelvis is the character frame, not a bone to swing), and bones
 * without a tail inherit their parent's correction.  All constants are measured
 * once at load; nothing here is frame-dependent.
 */
export function buildRetarget(adapter, jointNames, options = {}) {
  const source = new Map(jointNames.map((name, index) => [name, index]));
  const { rest, ancestorOf } = measureRestFrame(adapter);
  // `pelvisMatch` swings the hips frame as well, so an all-identity Core27 pose
  // reproduces the template's Hips->Spine direction (and therefore a T-pose
  // whose every segment is 0.00 deg off).  Off keeps the avatar's own pelvis
  // frame and leaves the constant template angle at Hips->Spine.
  const pelvisMatch = options.pelvisMatch !== false;
  // Preserved-rest bones skip the absolute template swing: identity Core27
  // then reproduces the avatar's own rest pose.  The direction-based solver is
  // correct for limbs, but the template's Hips->Spine rest angle and its foot
  // segment differ from this rig (34.4 deg back-lean, toes raised ~23 deg), so
  // forcing those two directions distorted the T-pose and made the body sit
  // behind the pelvis.  Rotations still come from the model; only the constant
  // rest offset is kept native.
  const restBones = new Set(options.restBones || []);
  let hipsTemplateDeg = 0;
  const bones = [];
  const byDst = new Map();

  for (const entry of RETARGET) {
    const rec = rest.get(entry.dst);
    const k = source.get(entry.src);
    if (!rec || k === undefined) continue;
    const parent = ancestorOf.get(entry.dst) ?? null;
    const anchor = parent ? rest.get(parent) : null;
    let cw;
    if (entry.coreTail && rest.has(entry.tail)) {
      const tailRec = rest.get(entry.tail);
      const avatarDir = unitDir(rec.p, tailRec.p);
      const coreDir = unitDir(CORE_REST[CORE_INDEX.get(entry.src)],
        CORE_REST[CORE_INDEX.get(entry.coreTail)]);
      const dot = avatarDir[0] * coreDir[0] + avatarDir[1] * coreDir[1]
        + avatarDir[2] * coreDir[2];
      const deg = Math.acos(Math.max(-1, Math.min(1, dot))) * 180 / Math.PI;
      if (entry.dst === 'hips') {
        hipsTemplateDeg = roundTo(deg, 2);
        cw = pelvisMatch ? swing(avatarDir, coreDir) : IDENT.slice();
      } else if (restBones.has(entry.dst)) {
        cw = IDENT.slice();
      } else {
        cw = swing(avatarDir, coreDir);
      }
    } else if (entry.inherit && byDst.has(entry.inherit)) {
      cw = byDst.get(entry.inherit).cw;
    } else {
      cw = IDENT.slice();
    }
    const bone = {
      dst: entry.dst,
      k,
      parent,
      cw,
      qRest: rec.q,
      rel: anchor ? qmul(qinv(anchor.q), rec.q) : rec.q.slice(),
      restDir: entry.coreTail && rest.has(entry.tail)
        ? unitDir(rec.p, rest.get(entry.tail).p) : null,
      coreDir: entry.coreTail
        ? unitDir(CORE_REST[CORE_INDEX.get(entry.src)],
          CORE_REST[CORE_INDEX.get(entry.coreTail)]) : null,
    };
    bones.push(bone);
    byDst.set(entry.dst, bone);
  }

  return {
    bones,
    pelvisMatch,
    restBones: [...restBones],
    hipsTemplateDeg,
    apply(localQuats) {
      // The clip stores local rotations; the solve needs each source joint's
      // world rotation, i.e. the accumulation along the Core27 parent chain.
      const world0 = new Array(CORE_NAMES.length).fill(null);
      for (let k = 0; k < CORE_NAMES.length; k++) {
        const p = CORE_PARENTS[k];
        world0[k] = p < 0 ? localQuats[k] : qmul(world0[p], localQuats[k]);
      }
      const pose = {};
      const world = new Map();
      for (const bone of bones) {
        // W = R_model * Cw * Q_rest: the avatar's rest frame carried by the
        // model's rotation, with the rig difference swung out first.
        const target = qmul(qmul(world0[bone.k], bone.cw), bone.qRest);
        const parentWorld = bone.parent ? (world.get(bone.parent) ?? IDENT) : IDENT;
        // W_b = W_parent * rel * L_b  ->  L_b = rel^-1 * W_parent^-1 * W_b
        const local = qmul(qmul(qinv(bone.rel), qinv(parentWorld)), target);
        world.set(bone.dst, target);
        pose[bone.dst] = { rotation: qnorm(local) };
      }
      return pose;
    },
  };
}

// Avatar bone pair -> Core27 bone pair, for the T-pose calibration readout.
export const TPOSE_SEGMENTS = [
  ['hips', 'spine', 'Hips', 'Spine'],
  ['spine', 'chest', 'Spine', 'Spine1'],
  ['chest', 'neck', 'Spine3', 'Neck'],
  ['neck', 'head', 'Neck', 'Head'],
  ['leftShoulder', 'leftUpperArm', 'LeftShoulder', 'LeftArm'],
  ['leftUpperArm', 'leftLowerArm', 'LeftArm', 'LeftForeArm'],
  ['leftLowerArm', 'leftHand', 'LeftForeArm', 'LeftHand'],
  ['rightShoulder', 'rightUpperArm', 'RightShoulder', 'RightArm'],
  ['rightUpperArm', 'rightLowerArm', 'RightArm', 'RightForeArm'],
  ['rightLowerArm', 'rightHand', 'RightForeArm', 'RightHand'],
  ['leftUpperLeg', 'leftLowerLeg', 'LeftUpLeg', 'LeftLeg'],
  ['leftLowerLeg', 'leftFoot', 'LeftLeg', 'LeftFoot'],
  ['leftFoot', 'leftToes', 'LeftFoot', 'LeftToeBase'],
  ['rightUpperLeg', 'rightLowerLeg', 'RightUpLeg', 'RightLeg'],
  ['rightLowerLeg', 'rightFoot', 'RightLeg', 'RightFoot'],
  ['rightFoot', 'rightToes', 'RightFoot', 'RightToeBase'],
];

/**
 * Freeze the avatar in the Core27 identity pose and measure every segment's
 * world direction against the template.  The pose is left applied; callers
 * re-apply the clip pose afterwards.  Root position/rotation are restored.
 */
export function measureTpose(adapter, retarget) {
  if (!adapter || !retarget) return null;
  const root = adapter.root;
  const savedPos = root.position.clone();
  const savedRot = root.rotation.clone();
  root.position.set(0, 0, 0);
  root.rotation.set(0, 0, 0);
  adapter.applyPose(retarget.apply(CORE_NAMES.map(() => IDENT.slice())));
  adapter.update(0);
  root.updateMatrixWorld(true);

  const pos = new Map();
  const names = new Set(TPOSE_SEGMENTS.flatMap(s => [s[0], s[1]]));
  for (const name of names) {
    const node = adapter.getNode(name);
    if (node) pos.set(name, node.getWorldPosition(new THREE.Vector3()).toArray());
  }

  const preservedBones = new Set(retarget.restBones || []);
  const rows = [];
  let sum = 0;
  let max = 0;
  for (const [pa, ca, ps, cs] of TPOSE_SEGMENTS) {
    const a = pos.get(pa);
    const b = pos.get(ca);
    if (!a || !b) continue;
    const d = unitDir(a, b);
    const e = unitDir(CORE_REST[CORE_INDEX.get(ps)], CORE_REST[CORE_INDEX.get(cs)]);
    const dot = d[0] * e[0] + d[1] * e[1] + d[2] * e[2];
    const deg = Math.acos(Math.max(-1, Math.min(1, dot))) * 180 / Math.PI;
    const preserved =
      (preservedBones.has('spine')
       && ((ps === 'Hips' && cs === 'Spine')
           || (ps === 'Spine' && cs === 'Spine1')))
      || (ps === 'LeftFoot' && cs === 'LeftToeBase'
          && preservedBones.has('leftFoot'))
      || (ps === 'RightFoot' && cs === 'RightToeBase'
          && preservedBones.has('rightFoot'));
    if (!preserved) {
      sum += deg;
      max = Math.max(max, deg);
    }
    rows.push({
      segment: `${ps}->${cs}`,
      avatar: `${pa}->${ca}`,
      angle_deg: roundTo(deg, 2),
      preserved,
      tpose_dir: d.map(v => roundTo(v, 3)),
      core_dir: e.map(v => roundTo(v, 3)),
    });
  }

  const hip = pos.get('hips');
  const ltoe = pos.get('leftToes');
  const rtoe = pos.get('rightToes');
  root.position.copy(savedPos);
  root.rotation.copy(savedRot);
  root.updateMatrixWorld(true);

  const sorted = rows.slice().sort((a, b) => b.angle_deg - a.angle_deg);
  return {
    pelvis_match: Boolean(retarget.pelvisMatch),
    hips_template_deg: retarget.hipsTemplateDeg ?? null,
    mean_deg: rows.length ? roundTo(sum / rows.length, 2) : null,
    max_deg: roundTo(max, 2),
    rows,
    worst: sorted.slice(0, 5),
    hip_y: hip ? roundTo(hip[1], 4) : null,
    toe_y: {
      left: ltoe ? roundTo(ltoe[1], 4) : null,
      right: rtoe ? roundTo(rtoe[1], 4) : null,
    },
  };
}

async function loadVrm(manifest) {
  const loader = new GLTFLoader();
  loader.register(parser => new VRMLoaderPlugin(parser));
  const url = `/api/avatars/${encodeURIComponent(manifest.id)}/file`;
  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error('VRM humanoid 数据缺失');

  VRMUtils.rotateVRM0(vrm);
  VRMUtils.combineSkeletons(vrm.scene);
  vrm.scene.traverse(node => {
    if (node.isMesh) {
      node.castShadow = true;
      node.receiveShadow = true;
      node.frustumCulled = false;
    }
  });

  const root = new THREE.Group();
  root.name = `vrm:${manifest.id}`;
  root.add(vrm.scene);
  root.updateMatrixWorld(true);

  const target = Number(manifest.height_m) || 1.62;
  let bounds = new THREE.Box3().setFromObject(vrm.scene);
  const height = Math.max(bounds.max.y - bounds.min.y, 0.01);
  vrm.scene.scale.setScalar(target / height);
  root.updateMatrixWorld(true);

  bounds = new THREE.Box3().setFromObject(vrm.scene);
  vrm.scene.position.y -= bounds.min.y;
  vrm.humanoid.resetNormalizedPose();
  root.updateMatrixWorld(true);
  // The clip's root *is* the pelvis, so the rig is put on the hips bone, not on
  // the mesh centre: otherwise the pelvis would sit sideways/backwards of the
  // root and the character body would be offset from the model it is solving.
  const hipsNode = vrm.humanoid.getNormalizedBoneNode('hips');
  if (hipsNode) {
    const hp = hipsNode.getWorldPosition(new THREE.Vector3());
    vrm.scene.position.x -= hp.x - root.position.x;
    vrm.scene.position.z -= hp.z - root.position.z;
    root.updateMatrixWorld(true);
  }
  vrm.update(0);

  const metaVersion = vrm.meta?.metaVersion ?? '1';
  return {
    kind: 'vrm',
    root,
    vrm,
    height: target,
    direction: metaVersion === '0' ? -1 : 1,
    yawOffset: THREE.MathUtils.degToRad(Number(manifest.yaw_offset_deg) || 0),
    getNode: name => vrm.humanoid.getNormalizedBoneNode(name),
    applyPose(pose) { vrm.humanoid.setNormalizedPose(pose); },
    resetPose() { vrm.humanoid.resetNormalizedPose(); },
    update(dt) { vrm.update(dt); },
    groundMetrics() {
      root.updateMatrixWorld(true);
      const hips = vrm.humanoid.getNormalizedBoneNode('hips');
      const hp = hips ? hips.getWorldPosition(new THREE.Vector3()) : new THREE.Vector3();
      const box = new THREE.Box3().setFromObject(root);
      let toeY = Infinity;
      for (const name of ['leftToes', 'rightToes']) {
        const node = vrm.humanoid.getNormalizedBoneNode(name);
        if (node) toeY = Math.min(toeY, node.getWorldPosition(new THREE.Vector3()).y);
      }
      return {
        hipY: hp.y - root.position.y,
        soleY: box.min.y - root.position.y,
        toeY: Number.isFinite(toeY) ? toeY - root.position.y : box.min.y - root.position.y,
      };
    },
    dispose() {
      root.removeFromParent();
      vrm.scene.traverse(node => {
        if (node.isMesh) {
          node.geometry?.dispose();
          const mats = Array.isArray(node.material) ? node.material : [node.material];
          mats.forEach(m => {
            if (!m) return;
            for (const key of Object.keys(m)) {
              const value = m[key];
              if (value && value.isTexture) value.dispose();
            }
            m.dispose();
          });
        }
      });
    },
  };
}

/**
 * Scale the avatar onto the model's pelvis -> toe span (CORE_LEG).
 *
 * The measurement must be taken in the *Core27 rest pose*, not the rig's native
 * T-pose: the solve re-aims every segment (the foot bone included) at Core27's
 * rest direction, so hip height and toe height only mean something once that
 * pose is applied.  Measuring the native pose instead mixes two different foot
 * angles into the scale and leaves the toes ~6 cm short of the model's.
 */
export function fitLegs(adapter, retarget) {
  const root = adapter.root;
  const savedPos = root.position.clone();
  const savedRot = root.rotation.clone();
  const base = adapter.baseScale ?? root.scale.clone();
  adapter.baseScale = base;
  root.position.set(0, 0, 0);
  root.rotation.set(0, 0, 0);
  root.scale.set(1, 1, 1);
  if (retarget) {
    adapter.applyPose(retarget.apply(CORE_NAMES.map(() => [0, 0, 0, 1])));
  } else {
    adapter.resetPose();
  }
  adapter.update(0);
  root.updateMatrixWorld(true);
  const m = adapter.groundMetrics();
  const leg = m.hipY - m.toeY;
  const fit = leg > 1e-3 ? CORE_LEG / leg : 1;
  adapter.legFit = fit;
  adapter.fittedWithRetarget = Boolean(retarget);
  adapter.baseHeight ??= adapter.height || 1.62;
  adapter.height = adapter.baseHeight * fit;
  root.scale.copy(base).multiplyScalar(fit);
  adapter.resetPose();
  adapter.update(0);
  root.updateMatrixWorld(true);
  root.position.copy(savedPos);
  root.rotation.copy(savedRot);
  return fit;
}

export async function loadAvatar(manifest) {
  let adapter;
  if (!manifest || manifest.kind === 'procedural') {
    adapter = createMannequin();
    adapter.yawOffset = THREE.MathUtils.degToRad(Number(manifest?.yaw_offset_deg) || 0);
  } else {
    adapter = await loadVrm(manifest);
  }
  return adapter;
}

const fpTarget = new THREE.Vector3();
const fpEye = new THREE.Vector3();
const fpEyeL = new THREE.Vector3();
const fpEyeR = new THREE.Vector3();
const fpDir = new THREE.Vector3();
const fpFlat = new THREE.Vector3();
const GAZE_PITCH_MIN = -1.20;   // -69°: look at a prop at your feet
const GAZE_PITCH_MAX = 0.55;    // +31°
// Neutral FP follows the posed head; these only guard yaw extraction when the
// head is nearly vertical, they are not a "natural neck" clamp.
const HEAD_PITCH_MIN = -1.50;
const HEAD_PITCH_MAX = 1.50;

export function updateFpCamera(camera, adapter, heading, snap = false, dt = 1 / 60,
                               gaze = null) {
  if (!adapter) return;
  const head = adapter.getNode('head');
  if (!head) return;
  adapter.root.updateMatrixWorld(true);
  const q = head.getWorldQuaternion(new THREE.Quaternion());
  // VRM 0.x normalized bones carry a 180° Y rest rotation, so their +Z points
  // out of the back of the head.  VRM 1.0 +Z is the face.
  const forward = adapter.direction || 1;
  const headDir = new THREE.Vector3(0, 0, forward).applyQuaternion(q);
  if (headDir.lengthSq() < 1e-8) headDir.set(0, 0, forward);

  const eyeBase = head.getWorldPosition(fpEye);
  const leftEye = adapter.getNode('leftEye');
  const rightEye = adapter.getNode('rightEye');
  if (leftEye && rightEye) {
    leftEye.getWorldPosition(fpEyeL);
    rightEye.getWorldPosition(fpEyeR);
    eyeBase.copy(fpEyeL).add(fpEyeR).multiplyScalar(0.5);
  } else {
    eyeBase.y += 0.054;
  }

  // Attention overrides the head's own gaze: the brain aims the eye at the
  // prop it is about to act on, and may pitch far beyond a natural neck.
  const active = gaze && gaze.until > performance.now() && gaze.point;
  let dir;
  let pitch;
  if (active) {
    dir = new THREE.Vector3(gaze.point[0], gaze.point[1], gaze.point[2]).sub(eyeBase);
    if (dir.lengthSq() < 1e-6) dir = headDir.clone();
    dir.normalize();
    pitch = Math.asin(THREE.MathUtils.clamp(dir.y, -1, 1));
    pitch = THREE.MathUtils.clamp(pitch, GAZE_PITCH_MIN, GAZE_PITCH_MAX);
  } else {
    pitch = THREE.MathUtils.clamp(Math.asin(THREE.MathUtils.clamp(headDir.y, -1, 1)),
      HEAD_PITCH_MIN, HEAD_PITCH_MAX);
    const yaw = (Math.abs(headDir.x) < 1e-6 && Math.abs(headDir.z) < 1e-6)
      ? heading : Math.atan2(headDir.x, headDir.z);
    dir = new THREE.Vector3(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch),
      Math.cos(yaw) * Math.cos(pitch));
  }
  const yawFinal = Math.abs(dir.x) < 1e-6 && Math.abs(dir.z) < 1e-6
    ? heading : Math.atan2(dir.x, dir.z);
  dir.copy(fpDir.set(Math.sin(yawFinal) * Math.cos(pitch), Math.sin(pitch),
    Math.cos(yawFinal) * Math.cos(pitch)));
  // Step the eye in front of the face so the head shell is behind the near
  // plane; the rest of the body is behind it in depth and stays visible.
  // The step is *horizontal*: stepping along a steep downward gaze moved the
  // eye out past the chest, so at -69deg (looking at a prop at the feet) the
  // own body vanished from the first-person frame.
  const flat = fpFlat.set(Math.sin(yawFinal), 0, Math.cos(yawFinal));
  const eye = eyeBase.clone().addScaledVector(flat, 0.10)
    .addScaledVector(dir, 0.02);

  const target = eye.clone().addScaledVector(dir, 4.0);
  const k = snap ? 1 : 1 - Math.exp(-dt * 18);
  if (snap) {
    camera.position.copy(eye);
    fpTarget.copy(target);
  } else {
    camera.position.lerp(eye, k);
    fpTarget.lerp(target, k);
  }
  camera.lookAt(fpTarget);
  return { pitch, yaw: yawFinal, dir: [dir.x, dir.y, dir.z],
           eye: [eye.x, eye.y, eye.z], active: !!active };
}
