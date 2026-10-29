// Rigid-body boxes + mouse drag for the ARDY-v demo (cannon-es).
import * as THREE from 'three';
import * as CANNON from 'cannon-es';

export function createPhysics(scene, spec, boxMeshes, camera, canvas) {
  const world = new CANNON.World({ gravity: new CANNON.Vec3(0, -9.82, 0) });
  world.allowSleep = true;
  world.broadphase = new CANNON.SAPBroadphase(world);
  world.solver.iterations = 12;
  const boxMat = new CANNON.Material('box');
  const groundMat = new CANNON.Material('ground');
  world.addContactMaterial(new CANNON.ContactMaterial(boxMat, groundMat, {
    friction: 0.55, restitution: 0.05,
  }));
  world.addContactMaterial(new CANNON.ContactMaterial(boxMat, boxMat, {
    friction: 0.45, restitution: 0.08,
  }));

  const half = Number(spec.half ?? 8);
  const ceiling = Number(spec.ceiling ?? 4);
  const floor = new CANNON.Body({ mass: 0, shape: new CANNON.Plane(), material: groundMat });
  floor.quaternion.setFromEuler(-Math.PI / 2, 0, 0);
  world.addBody(floor);

  // four static walls
  for (const [x, z, sx, sz] of [[0, -half, half, 0.25], [0, half, half, 0.25],
                                [-half, 0, 0.25, half], [half, 0, 0.25, half]]) {
    const b = new CANNON.Body({
      mass: 0, material: groundMat,
      shape: new CANNON.Box(new CANNON.Vec3(sx, ceiling / 2, sz)),
      position: new CANNON.Vec3(x, ceiling / 2, z),
    });
    world.addBody(b);
  }

  const bodies = [];
  const initial = [];
  (spec.boxes || []).forEach((b, i) => {
    const mass = Math.max(2.5, b.hx * b.hz * b.h * 2.4);
    const body = new CANNON.Body({
      mass, material: boxMat,
      shape: new CANNON.Box(new CANNON.Vec3(b.hx, b.h / 2, b.hz)),
      position: new CANNON.Vec3(b.x, b.h / 2, b.z),
      linearDamping: 0.12, angularDamping: 0.45,
    });
    body.quaternion.setFromEuler(0, b.yaw || 0, 0);
    world.addBody(body);
    const mesh = boxMeshes[i];
    if (mesh) {
      mesh.userData.body = body;
      mesh.userData.initial = {
        p: new CANNON.Vec3(b.x, b.h / 2, b.z),
        q: new CANNON.Quaternion().copy(body.quaternion),
      };
    }
    bodies.push(body);
    initial.push({ p: new CANNON.Vec3(b.x, b.h / 2, b.z),
                   q: new CANNON.Quaternion().copy(body.quaternion) });
  });

  let avatarBody = null;
  const cursor = new CANNON.Body({ type: CANNON.Body.KINEMATIC,
    shape: new CANNON.Sphere(0.05) });
  world.addBody(cursor);
  let constraint = null;
  let dragged = null;
  let planeY = 0.8;
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -planeY);

  function setAvatar(adapter) {
    if (avatarBody) world.removeBody(avatarBody);
    avatarBody = new CANNON.Body({
      type: CANNON.Body.KINEMATIC,
      shape: new CANNON.Sphere(0.30),
      position: new CANNON.Vec3(0, 0.9, 0),
    });
    world.addBody(avatarBody);
  }

  function setAvatarPosition(p) {
    if (!avatarBody) return;
    avatarBody.position.set(p.x, p.y + 0.55, p.z);
    avatarBody.velocity.set(0, 0, 0);
  }

  function sync() {
    for (const mesh of boxMeshes) {
      const body = mesh.userData.body;
      if (!body) continue;
      mesh.position.copy(body.position);
      mesh.quaternion.copy(body.quaternion);
    }
  }

  // Current world AABB of every box, for the viewer collision check and HUD.
  function boxStates() {
    return bodies.map((body, i) => {
      body.updateAABB();
      const a = body.aabb;
      return {
        x: body.position.x, y: body.position.y, z: body.position.z,
        minX: a.lowerBound.x, maxX: a.upperBound.x,
        minY: a.lowerBound.y, maxY: a.upperBound.y,
        minZ: a.lowerBound.z, maxZ: a.upperBound.z,
        asleep: body.sleepState === CANNON.Body.SLEEPING,
        source: i,
      };
    });
  }

  function step(dt) {
    world.step(1 / 60, Math.min(dt, 0.05), 3);
    sync();
  }

  function pointerNdc(e) {
    const r = canvas.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1,
            -((e.clientY - r.top) / r.height) * 2 + 1);
  }

  function dragStart(e) {
    pointerNdc(e);
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObjects(boxMeshes, false);
    if (!hits.length) return false;
    const hit = hits[0];
    const body = hit.object.userData.body;
    if (!body) return false;
    const local = body.pointToLocalFrame(
      new CANNON.Vec3(hit.point.x, hit.point.y, hit.point.z));
    cursor.position.set(hit.point.x, hit.point.y, hit.point.z);
    cursor.velocity.set(0, 0, 0);
    constraint = new CANNON.PointToPointConstraint(
      body, local, cursor, new CANNON.Vec3(), 350);
    world.addConstraint(constraint);
    dragged = body;
    planeY = hit.point.y;
    dragPlane.constant = -planeY;
    body.wakeUp();
    // The walking avatar is a kinematic sphere; with collision response on it
    // shoves the constrained box away from the cursor ("the box escapes").
    if (avatarBody) avatarBody.collisionResponse = false;
    return true;
  }

  function dragMove(e) {
    if (!dragged) return;
    pointerNdc(e);
    raycaster.setFromCamera(ndc, camera);
    const hit = new THREE.Vector3();
    if (raycaster.ray.intersectPlane(dragPlane, hit)) {
      const lim = half - 0.4;
      cursor.position.set(Math.max(-lim, Math.min(lim, hit.x)),
                          planeY,
                          Math.max(-lim, Math.min(lim, hit.z)));
    }
  }

  function dragEnd() {
    if (constraint) { world.removeConstraint(constraint); constraint = null; }
    if (dragged) {
      dragged.velocity.setZero();
      dragged.angularVelocity.setZero();
      dragged.wakeUp();
      dragged = null;
    }
    if (avatarBody) avatarBody.collisionResponse = true;
  }

  function reset() {
    dragEnd();
    bodies.forEach((body, i) => {
      body.position.copy(initial[i].p);
      body.quaternion.copy(initial[i].q);
      body.velocity.setZero();
      body.angularVelocity.setZero();
      body.wakeUp();
    });
    sync();
  }

  return { world, bodies, setAvatar, setAvatarPosition, step, reset, boxStates,
           dragStart, dragMove, dragEnd, get dragging() { return !!dragged; } };
}
