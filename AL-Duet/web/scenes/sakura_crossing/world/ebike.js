import * as THREE from 'three';
import { flat } from '../core/toon.js';
import { RIDE } from '../core/player.js';
import { makeScooter } from './streetprops.js';
import { basisAt, positionAt, wrapX, wrapDelta } from './planet.js';

/* ------------------------------------------------------------------ *
 * 電動バイク -- the one machine in this world you can get on.
 *
 * Everything else with wheels is scenery: `traffic.js` parks thirty-seven
 * vehicles and twelve scooters and not one of them moves.  This is the same
 * `makeScooter` those are built from -- a step-through 50, 1.65 m long, seat
 * at 0.62 -- summoned with `V`, ridden with `E`.
 *
 * It is built and placed **after** `bakeToPlanet` has run, which is the whole
 * reason this is a runtime module rather than a district.  The bake folds
 * every mesh's geometry into root space and clears the container transforms,
 * so anything that moves has to be marked `planetRigid` and re-seated by
 * hand; a machine that is summoned at an arbitrary point and then driven
 * around cannot be baked at all.  So it is added straight to the scene and
 * seated the way the bake seats a rigid rig: tangent frame from `basisAt` for
 * the orientation, `positionAt` for the position.  Everything below is in
 * flat authoring coordinates, exactly like every builder in the world.
 *
 * Three numbers are shared with `core/player.js` through `RIDE` and must stay
 * shared: where the rider sits on the machine, how far the nose reaches past
 * them, and how high their eye is.  See the note there.
 * ------------------------------------------------------------------ */

/** Half extents of the machine in plan, for the parked collider. */
const HALF_LEN = 0.88;
const HALF_WID = 0.34;
/** Axle to axle, which is what the ground pitch is measured over. */
const WHEELBASE = 1.17;
/** On the side stand.  `makeScooter`'s own default, restated because the
 *  ridden machine has to be given a lean explicitly on every frame. */
const PARK_LEAN = -0.09;
/** How hard it leans into a turn, against the camera's own bank. */
const LEAN_GAIN = 2.4;
/** The player's footprint, for testing whether a spot is free. */
const BODY_R = 0.34;
const STEP = 0.38;

/**
 * @param scene   the render scene -- *not* `world.root`, which is baked
 * @param world   the built world: colliders, interactables, `heightAt`
 * @param player  the walker, which this mounts and dismounts
 * @param hud     optional, for the toast line
 */
export function createEbike({ scene, world, player, hud }) {
  /* A warm body colour, and the reason is the same one `makeScooter`'s own
   * comment gives for not painting it cream: every other scooter in the world
   * is a blue-grey, and against pale paving under a violet shadow tint a cool
   * machine has nowhere to separate from the ground.  This one has to be
   * findable at forty metres. */
  const bike = makeScooter({ color: 0xcf7a62, lean: PARK_LEAN, cockpit: true });
  const inner = bike.userData.inner;
  bike.visible = false;
  scene.add(bike);

  /* The hitbox `E` picks, a child of the machine so it travels with it for
   * free.  Invisible rather than absent: `Raycaster` does not test `visible`,
   * which is what the cat's hitbox in `index.js` relies on too. */
  const hit = new THREE.Mesh(
    new THREE.BoxGeometry(1.9, 1.35, 0.95),
    flat({ color: 0xff0000, cache: false })
  );
  hit.position.set(-0.05, 0.68, 0);
  hit.visible = false;
  bike.add(hit);

  const entry = {
    hitbox: hit,
    label: '電動バイク  ·  ride it',
    action: () => mount(),
  };

  const state = { out: false, riding: false, x: 0, z: 0, heading: 0 };
  let collider = null;

  const _up = new THREE.Vector3();
  const _east = new THREE.Vector3();
  const _north = new THREE.Vector3();
  const _basis = new THREE.Matrix4();
  const _surfaceQ = new THREE.Quaternion();
  const _localQ = new THREE.Quaternion();
  const _localE = new THREE.Euler();

  /* --------------------------------- seating --------------------------------- */

  /**
   * Put the machine on the sphere at a flat point.
   *
   * `heading` is the direction the nose faces in the same convention the
   * player's yaw uses (0 looks along -z).  The machine is authored **along +x
   * with the nose at +x**, so it is a quarter turn off: a nose along -z is
   * `ry = PI/2`.  `pitch` is the ground's rake along its length, applied about
   * the local z **before** the heading -- Euler 'YXZ' is Ry·Rx·Rz, so the z
   * term is the innermost, and a positive one sends the nose up.
   */
  function seat(x, z, y, heading, pitch) {
    basisAt(x, z, _up, _east, _north);
    _basis.makeBasis(_east, _up, _north);
    _surfaceQ.setFromRotationMatrix(_basis);
    _localE.set(0, heading + Math.PI / 2, pitch, 'YXZ');
    _localQ.setFromEuler(_localE);
    bike.quaternion.copy(_surfaceQ).multiply(_localQ);
    positionAt(x, y, z, bike.position);
    bike.updateMatrixWorld(true);
  }

  /** The ground's rake along the machine, from a sample at each axle. */
  function groundPitch(x, z, heading, fromY) {
    const fx = -Math.sin(heading), fz = -Math.cos(heading);
    const h = WHEELBASE / 2;
    const front = world.heightAt(wrapX(x + fx * h), z + fz * h, fromY);
    const rear = world.heightAt(wrapX(x - fx * h), z - fz * h, fromY);
    return THREE.MathUtils.clamp(Math.atan2(front - rear, WHEELBASE), -0.45, 0.45);
  }

  /* -------------------------------- colliders -------------------------------- */

  /**
   * The parked machine blocks the ground it stands on, like every other
   * vehicle here -- and like them the box is the *rotated* AABB, because a
   * 1.76 m machine turned forty-five degrees is 1.5 m across in both axes and
   * the player's radius is added to every side of whatever we write down.
   */
  function setCollider(on) {
    if (collider) {
      const i = world.colliders.indexOf(collider);
      if (i >= 0) world.colliders.splice(i, 1);
      collider = null;
    }
    if (!on) return;
    const s = Math.abs(Math.sin(state.heading)), c = Math.abs(Math.cos(state.heading));
    const hx = HALF_LEN * s + HALF_WID * c;
    const hz = HALF_LEN * c + HALF_WID * s;
    const y = world.heightAt(state.x, state.z, player.pos.y);
    collider = {
      x0: state.x - hx, x1: state.x + hx,
      z0: state.z - hz, z1: state.z + hz,
      top: y + 1.02,
    };
    world.colliders.push(collider);
  }

  /** Is there room for a body of radius `r` at a flat point, feet at `feetY`? */
  function roomAt(x, z, r, feetY) {
    for (const c of world.colliders) {
      if (c === collider) continue;
      if (c.top !== undefined && c.top <= feetY + STEP) continue;
      if (c.bottom !== undefined && c.bottom > feetY + 1.9) continue;
      if (x > c.x0 - r && x < c.x1 + r && z > c.z0 - r && z < c.z1 + r) return false;
    }
    return true;
  }

  function setInteract(on) {
    const i = world.interactables.indexOf(entry);
    if (on && i < 0) world.interactables.push(entry);
    if (!on && i >= 0) world.interactables.splice(i, 1);
  }

  /** Park it where it is standing: stand lean, ground rake, collider, prompt. */
  function park() {
    const y = world.heightAt(state.x, state.z, player.pos.y);
    inner.rotation.x = PARK_LEAN;
    seat(state.x, state.z, y, state.heading, groundPitch(state.x, state.z, state.heading, y));
    bike.visible = true;
    state.out = true;
    setCollider(true);
    setInteract(true);
  }

  /* --------------------------------- summon --------------------------------- */

  /**
   * Stand it in front of the player.
   *
   * The spot is searched rather than assumed: two metres ahead is a wall about
   * as often as it is pavement, and a machine half inside a shopfront is the
   * same class of mistake as every prop in the trap table.  It needs 0.8 m of
   * clear ground -- less than the machine's own half-length, because it may
   * legitimately overhang a kerb or a verge -- and ground within half a metre
   * of the player's own, so it cannot be summoned onto a roof or into the
   * channel.
   *
   * Parked at a slight angle to the player, which is worth the twenty degrees
   * the view snaps on mounting: square on it is a silhouette of its own tail.
   */
  function summon() {
    const feet = player.pos.y;
    let spot = null;
    for (const d of [2.0, 1.65, 2.6, 1.3]) {
      for (const a of [0, 0.45, -0.45, 0.95, -0.95, 1.6, -1.6]) {
        const x = wrapX(player.pos.x - Math.sin(player.yaw + a) * d);
        const z = player.pos.z - Math.cos(player.yaw + a) * d;
        if (!roomAt(x, z, 0.8, feet)) continue;
        if (Math.abs(world.heightAt(x, z, feet) - feet) > 0.5) continue;
        spot = { x, z };
        break;
      }
      if (spot) break;
    }
    // nowhere clear: stand it right in front anyway rather than refusing
    if (!spot) {
      spot = {
        x: wrapX(player.pos.x - Math.sin(player.yaw) * 1.5),
        z: player.pos.z - Math.cos(player.yaw) * 1.5,
      };
    }
    state.x = spot.x;
    state.z = spot.z;
    state.heading = player.yaw - 0.35;
    park();
    hud?.flash('電動バイク  ·  E to ride', 1700);
  }

  /** Put it away. */
  function recall() {
    if (state.riding) return;
    bike.visible = false;
    state.out = false;
    setCollider(false);
    setInteract(false);
    hud?.flash('電動バイク  ·  put away', 1200);
  }

  /* ---------------------------- mount and dismount ---------------------------- */

  function mount() {
    if (!state.out || state.riding) return;
    /* Snap the view to the machine's heading.  It has to be a snap: the yaw
     * *is* the heading while riding, so anything else would have the rider
     * facing off the side of a machine that is pointing somewhere else. */
    player.yaw = state.heading;
    player.pos.x = wrapX(state.x + Math.sin(state.heading) * RIDE.seatFwd);
    player.pos.z = state.z + Math.cos(state.heading) * RIDE.seatFwd;
    setCollider(false);
    setInteract(false);
    state.riding = true;
    player.mount(entry);
    hud?.flash('電動バイク  ·  W to go, E to get off', 2000);
  }

  function dismount() {
    if (!state.riding) return;
    state.riding = false;
    player.unmount();

    /* Step off to the side, and check there is somewhere to step to.  Left
     * first, which is the side the stand is on and therefore the side a rider
     * gets off; then right; then back off the tail.  1.35 m clears the parked
     * machine's own collider, so the player is not shoved by the box that
     * appears under them a line later. */
    const rx = Math.cos(state.heading), rz = -Math.sin(state.heading);
    const fx = -Math.sin(state.heading), fz = -Math.cos(state.heading);
    const feet = player.pos.y;
    const tries = [
      [-rx * 1.35, -rz * 1.35],
      [rx * 1.35, rz * 1.35],
      [-fx * 1.9, -fz * 1.9],
    ];
    for (const [dx, dz] of tries) {
      const x = wrapX(player.pos.x + dx);
      const z = player.pos.z + dz;
      if (!roomAt(x, z, BODY_R, feet)) continue;
      if (Math.abs(world.heightAt(x, z, feet) - feet) > 0.6) continue;
      player.pos.x = x;
      player.pos.z = z;
      break;
    }
    park();
    hud?.flash('電動バイク  ·  parked', 1200);
  }

  /* ---------------------------------- keys ---------------------------------- */

  /**
   * One key does all three states, which is what makes it worth a key at all:
   * summon it, call it back to you if you have left it somewhere, put it away
   * if it is under your nose -- and get off it if you are on it.
   */
  function toggle() {
    if (state.riding) { dismount(); return; }
    if (!state.out) { summon(); return; }
    const d = Math.hypot(wrapDelta(state.x, player.pos.x), state.z - player.pos.z);
    if (d > 4.0) summon(); else recall();
  }

  /* --------------------------------- per frame --------------------------------- */

  function update() {
    if (!state.riding) return;   // parked, it is placed once and left alone
    const h = player.yaw;
    const fx = -Math.sin(h), fz = -Math.cos(h);
    const x = wrapX(player.pos.x + fx * RIDE.seatFwd);
    const z = player.pos.z + fz * RIDE.seatFwd;
    /* Seated on the player's *own* smoothed feet height rather than on a
     * fresh ground query, so the machine never pops a kerb the rider is still
     * easing up over -- they are the same body. */
    seat(x, z, player.pos.y, h, groundPitch(x, z, h, player.pos.y));
    // lean with the camera; see the note on `roll` in `player.js`
    inner.rotation.x = -player.roll * LEAN_GAIN;
    state.x = x;
    state.z = z;
    state.heading = h;
  }

  return {
    group: bike,
    toggle, summon, recall, mount, dismount, update,
    get riding() { return state.riding; },
    get summoned() { return state.out; },
  };
}
