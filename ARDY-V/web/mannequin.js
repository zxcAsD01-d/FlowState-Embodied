import * as THREE from 'three';

const JOINTS = ['hips', 'spine', 'chest', 'neck', 'head',
  'leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand',
  'rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand',
  'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToes',
  'rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'rightToes'];

function mat(color, { metalness = 0.05, roughness = 0.6, emissive = 0x000000, glow = 0 } = {}) {
  return new THREE.MeshStandardMaterial({
    color, metalness, roughness, emissive, emissiveIntensity: glow,
  });
}

function segment(parent, a, b, radius, material, caps = 8) {
  const va = new THREE.Vector3(...a);
  const vb = new THREE.Vector3(...b);
  const dir = vb.clone().sub(va);
  const len = Math.max(dir.length() - radius * 1.4, 0.02);
  const mesh = new THREE.Mesh(new THREE.CapsuleGeometry(radius, len, caps, 14), material);
  mesh.position.copy(va).add(vb).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

function box(parent, size, pos, material) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material);
  mesh.position.set(...pos);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

/**
 * Built-in procedural avatar with the same normalized humanoid bone names as
 * a VRM.  Built in T-pose so the 27-joint CoreSkeleton rotations apply directly.
 */
export function createMannequin() {
  const root = new THREE.Group();
  root.name = 'mannequin';

  const shell = mat(0xe9e7ec, { roughness: 0.55 });
  const joint = mat(0x2a2f3a, { roughness: 0.4, metalness: 0.35 });
  const suit = mat(0x141b26, { roughness: 0.62 });
  const glow = mat(0x7ce7ff, { roughness: 0.25, metalness: 0.2, emissive: 0x37c8e8, glow: 2.5 });
  const ribbon = mat(0xf3a6c6, { roughness: 0.5, emissive: 0x5c1f34, glow: 0.5 });

  const nodes = {};
  const bone = (name, parent, pos) => {
    const g = new THREE.Group();
    g.name = name;
    g.position.set(...pos);
    parent.add(g);
    nodes[name] = g;
    return g;
  };

  const hips = bone('hips', root, [0, 0.92, 0]);
  box(hips, [0.27, 0.17, 0.19], [0, 0.015, 0], suit);

  const spine = bone('spine', hips, [0, 0.10, 0]);
  const chest = bone('chest', spine, [0, 0.12, 0]);
  segment(chest, [0, 0.01, 0], [0, 0.27, 0], 0.145, shell);
  box(chest, [0.34, 0.30, 0.21], [0, 0.14, 0], shell);
  box(chest, [0.36, 0.05, 0.22], [0, 0.24, 0], suit);
  box(chest, [0.10, 0.10, 0.012], [0, 0.17, 0.108], glow);

  const neck = bone('neck', chest, [0, 0.24, 0]);
  segment(neck, [0, 0, 0], [0, 0.08, 0], 0.045, joint);

  const head = bone('head', neck, [0, 0.09, 0]);
  const skull = new THREE.Mesh(new THREE.SphereGeometry(0.115, 26, 20), shell);
  skull.position.set(0, 0.075, 0);
  skull.scale.set(1, 1.06, 0.98);
  skull.castShadow = true;
  head.add(skull);
  const hair = new THREE.Mesh(new THREE.SphereGeometry(0.121, 24, 18,
    0, Math.PI * 2, 0, Math.PI * 0.62), mat(0x232936, { roughness: 0.7 }));
  hair.position.set(0, 0.09, -0.006);
  hair.castShadow = true;
  head.add(hair);
  box(head, [0.155, 0.036, 0.02], [0, 0.085, 0.106], glow);
  const tail = new THREE.Mesh(new THREE.CapsuleGeometry(0.035, 0.26, 6, 10), ribbon);
  tail.position.set(0, 0.06, -0.135);
  tail.rotation.x = -0.32;
  tail.castShadow = true;
  head.add(tail);

  for (const side of [1, -1]) {
    const s = side > 0 ? 'left' : 'right';
    const sh = bone(`${s}Shoulder`, chest, [0.055 * side, 0.20, 0]);
    const ua = bone(`${s}UpperArm`, sh, [0.115 * side, 0, 0]);
    segment(ua, [0, 0, 0], [0.255 * side, 0, 0], 0.046, shell);
    const la = bone(`${s}LowerArm`, ua, [0.255 * side, 0, 0]);
    segment(la, [0, 0, 0], [0.225 * side, 0, 0], 0.039, suit);
    const hand = bone(`${s}Hand`, la, [0.225 * side, 0, 0]);
    box(hand, [0.10, 0.055, 0.085], [0.045 * side, 0, 0], joint);

    const ul = bone(`${s}UpperLeg`, hips, [0.088 * side, -0.025, 0]);
    segment(ul, [0, 0, 0], [0, -0.40, 0], 0.062, suit);
    const ll = bone(`${s}LowerLeg`, ul, [0, -0.40, 0]);
    segment(ll, [0, 0, 0], [0, -0.40, 0], 0.052, shell);
    const foot = bone(`${s}Foot`, ll, [0, -0.40, 0]);
    box(foot, [0.105, 0.075, 0.225], [0, -0.022, 0.055], joint);
    bone(`${s}Toes`, foot, [0, -0.03, 0.13]);
  }

  root.traverse(o => {
    if (o.isMesh) {
      o.castShadow = true;
      o.receiveShadow = true;
      o.frustumCulled = false;
    }
  });

  return {
    kind: 'procedural',
    root,
    height: 1.62,
    yawOffset: 0,
    jointNames: JOINTS,
    getNode: name => nodes[name] || null,
    applyPose(pose) {
      for (const [name, value] of Object.entries(pose)) {
        const node = nodes[name];
        if (node && value && value.rotation) node.quaternion.fromArray(value.rotation);
      }
    },
    resetPose() {
      for (const node of Object.values(nodes)) node.quaternion.identity();
    },
    update() {},
    groundMetrics() {
      root.updateMatrixWorld(true);
      const hipsPos = nodes.hips.getWorldPosition(new THREE.Vector3());
      const bounds = new THREE.Box3().setFromObject(root);
      return {
        hipY: hipsPos.y - root.position.y,
        soleY: bounds.min.y - root.position.y,
        toeY: Math.min(
          nodes.leftToes.getWorldPosition(new THREE.Vector3()).y,
          nodes.rightToes.getWorldPosition(new THREE.Vector3()).y,
        ) - root.position.y,
      };
    },
    dispose() {
      root.removeFromParent();
      root.traverse(o => {
        if (o.isMesh) {
          o.geometry?.dispose();
          const m = o.material;
          if (Array.isArray(m)) m.forEach(x => x.dispose());
          else m?.dispose();
        }
      });
    },
  };
}
