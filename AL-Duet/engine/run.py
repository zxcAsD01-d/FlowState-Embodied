# -*- coding: utf-8 -*-
"""AL-Duet engine: the end-to-end script.

One script chains everything:

    vision -> social/emotion FSM (sparse) -> action plan -> ARDY continuous
    generation (N agents in ONE diffusion pass) -> external root controller
    -> world frames -> live push or episode file.

Modes:
* ``--mode offline``  infer-then-play: generate the whole episode for N
  agents (geometric perception, no browser), write ``web/data/episode.json``
  + ``timeline``; the frontend only plays it.
* ``--mode live``     realtime: every window is pushed to the backend live
  buffers and the page renders it as it is produced.  ``--vision render``
  here closes the loop through the actual rendered pixels (YOLOv8-m).
  ``--realtime`` paces the loop to the simulation clock (0.4 s / window);
  without it the loop runs as fast as the GPU allows.

Agents spawn on a circle facing its centre.  Spawn count is ``--actors N``.

Run:
    python -m engine.run --mode offline --actors 2 --windows 80
    python -m engine.run --mode live --actors 2 --windows 300 --realtime
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from . import ardy
from .actors import Agent, TICK_S, WALK_MIN_START
from .explore import Explorer
from .social.dual import DualStateMachine
from .social.schema import Observation, PersonState
from .social.speech import (ADDRESSED_S, DYAD_STATES, SPEAK_S, SPEECH_BY_STATE,
                            WAVE_STATES, speech_line)
from .vision import geometric, nearest_other

ROOT = Path(__file__).resolve().parents[1]
WEB_DATA = ROOT / "web" / "data"
SPAWN_R = 1.3
# FSM rhythm (tuned 2026-10-29: snappier dyad turns, faster solo reactions)
DYAD_INTERVAL = 1.2
SOLO_INTERVAL = 2.0
MIN_DECISION_GAP = 0.6
# World bounds for the explorer (sakura street walkable core; props avoided
# via the Explorer's prop circles -- the street scene has no hard walls in
# this playback world, the rect is the fence line).
# World bounds for the explorer -- keep wanderers on the visibly paved core
# of the street scene (beyond this the ground plane ends and the camera
# follows them into the void).
WORLD_BOUNDS = (-4.5, 4.5, -3.5, 3.5)
WORLD_PROPS = [(-3.6, -1.6, 1.4), (3.8, 1.2, 1.6), (0.0, 3.2, 1.2)]
# mode switching: solo for this long with nobody in sight -> explorer runner
SOLO_TIMEOUT = 6.0
SOLO_STATES = {"solo_idle", "solo_travel", "walk_through", "observe_near",
               "observe_far"}
# states whose walk moves AWAY from the partner: the travel-aligned walk
# always steps forward, so the agent turns around first (no back-pedal skate)
RETREAT_STATES = {"too_close_retreat", "intimate_withdraw"}
# states whose walk APPROACHES the partner: stop at conversational range so
# the greet transition fires before the 0.6 m personal-space line.  Transit
# walks (walk_through/avoid_person/yield_path) must NOT be distance-stopped --
# blocking them jammed two agents face-to-face in a transit loop.
APPROACH_WALK_STATES = {"approach_person", "reengage", "follow_person",
                        "join_circle", "seek_person"}


def spawn_agents(n: int) -> list[tuple[tuple[float, float], float]]:
    """N agents on a circle, each facing the centre."""
    out = []
    for i in range(n):
        a = math.pi / 2.0 + 2.0 * math.pi * i / n
        x, z = SPAWN_R * math.cos(a), SPAWN_R * math.sin(a)
        yaw = math.atan2(-x, -z)   # face the centre
        out.append(((x, z), yaw))
    return out


def observe_and_decide(agent: Agent, agents: list[Agent], t_sim: float,
                       vis: dict) -> None:
    """Sparse FSM update for one agent (same policy as the audited duet)."""
    visible = bool(vis["visible"])
    d_vis = float(vis.get("distance") or 9.0)
    other = vis.get("other")
    speaking_other = bool(other is not None and other.speaking_until > t_sim)
    addressed = speaking_other or t_sim < agent.addressed_until
    mutual = float(vis.get("mutual", 0.0))

    band = agent.last_dec.plan.distance_band if agent.last_dec else (0.0, 0.0)
    due = agent.last_dec is None or t_sim >= agent.next_decision_t
    # approach is exactly when decisions must be reactive: while the plan
    # walks and someone is close, decide EVERY window (rules are cheap)
    if (agent.last_dec is not None
            and agent.last_dec.plan.ardy_intent in ("walk", "walk_fwd")
            and visible and d_vis < 2.5):
        due = True
    if agent.last_other_speaking != speaking_other:
        due = True
    if agent.last_other_visible != visible:
        due = True
    if band[1] > 0 and agent.last_d_vis is not None:
        lo, hi = band[0] - 0.25, band[1] + 0.25
        if ((agent.last_d_vis < lo) != (d_vis < lo)
                or (agent.last_d_vis > hi) != (d_vis > hi)):
            due = True
    if (due and agent.last_dec is not None
            and t_sim - agent.last_decision_t < MIN_DECISION_GAP):
        due = False

    if due:
        pstate = PersonState(
            id="peer", distance=d_vis if visible else 99.0,
            bearing_deg=float(vis.get("bearing_deg", 0.0)) if visible else 0.0,
            facing_me=float(vis.get("facing_me", 0.0)),
            relationship="stranger", visible=visible,
            speaking=speaking_other,
            addressed_to_me=addressed,
            moving=bool(other is not None and other.speed > 0.06),
            speed=float(other.speed) if other is not None else 0.0,
            confidence=float(vis.get("conf", 0.0)),
            position=(tuple(float(v) for v in other.pos)
                      if other is not None else (0.0, 0.0, 0.0)),
            last_seen=agent.last_seen_t,
        )
        events = {k: list(v) for k, v in agent.events.items()}
        if speaking_other:
            events.setdefault("speak_other", []).append(t_sim)
            agent.add_event("speak_other", t_sim)
        obs = Observation(
            t=t_sim + agent.phase,
            self_pos=(float(agent.pos[0]), 0.0, float(agent.pos[2])),
            self_heading_deg=math.degrees(agent.heading),
            self_speed=agent.speed, self_moving=agent.speed > 0.06,
            scene_mode="public", scene_crowd=0.0,
            audio_activity=speaking_other, audio_speech=speaking_other,
            audio_energy=0.4 if speaking_other else 0.0,
            floor_holder="peer" if speaking_other else None,
            focus_id="peer" if visible else None,
            group_size=len(agents) if visible else 0,
            relationship_stage="stranger", consent_state="none",
            comfort=0.65, arousal=0.35, intimacy_level=0,
            mutual_gaze=mutual,
            reciprocity=0.5 if agent.recent("speak_other", t_sim, 8.0) else 0.0,
            proximity=max(0.0, min(1.0, 1.0 - d_vis / 3.5)),
            emotion_label="calm", emotion_valence=0.15,
            emotion_comfort=0.65,
            persons=[pstate] if visible else [],
            events=events,
        )
        step = agent.fsm.step(obs, intensity=1.0)
        dec = step.social
        agent.last_step = step
        agent.last_dec = dec
        agent.last_emotion = step.emotion.label
        agent.last_decision_t = t_sim
        agent.next_decision_t = t_sim + (DYAD_INTERVAL if dec.state in DYAD_STATES
                                         else SOLO_INTERVAL)
        agent.decision_count += 1
        agent.last_other_speaking = speaking_other
        agent.last_other_visible = visible
        if dec.state != agent.last_state:
            agent.state_entry = t_sim
            agent.last_state = dec.state
            agent.wave_windows = 0
    agent.last_d_vis = d_vis
    agent.vision = vis


def floor_policy(agents: list[Agent], t_sim: float) -> None:
    """One voice at a time, measured on the sim clock."""
    for agent in agents:
        if agent.last_state in SPEECH_BY_STATE and \
                abs(t_sim - agent.state_entry) < 1.0 and \
                agent.speaking_until <= t_sim:
            agent.speaking_until = t_sim + SPEAK_S
            agent.speech = speech_line(agent.last_state, agent.key,
                                       agent.speech_turns)
            agent.speech_turns += 1
            for other in agents:
                if other is agent:
                    continue
                other.add_event("speak_other", t_sim)
                other.addressed_until = t_sim + 3.5
            break
    for agent in agents:
        for other in agents:
            if other is agent:
                continue
            if other.speaking_until > t_sim:
                agent.addressed_until = max(agent.addressed_until,
                                            t_sim + ADDRESSED_S)
        if agent.last_state in ("greet_initiate", "greet_ack") and \
                abs(t_sim - agent.state_entry) < 0.5:
            for other in agents:
                if other is agent:
                    continue
                other.add_event("speak_other", t_sim)
                other.addressed_until = t_sim + 3.5
        if agent.last_state in WAVE_STATES:
            for other in agents:
                if other is agent:
                    continue
                other.add_event("wave", t_sim)


def run(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("offline", "live"), default="offline")
    ap.add_argument("--actors", type=int, default=2,
                    help="agents (each runs ONE machine that switches between "
                         "explorer and social mode by itself)")
    ap.add_argument("--join-after", type=float, default=0.0, dest="join_after",
                    help="live only: agent i joins at i*S seconds (agent 0 "
                         "starts alone and explores until others appear)")
    ap.add_argument("--windows", type=int, default=80,
                    help="0 = run forever (live mode, Ctrl+C to stop)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scene", default="sakura_crossing")
    ap.add_argument("--vision", choices=("geometric", "render"),
                    default="geometric")
    ap.add_argument("--realtime", action="store_true",
                    help="pace the live loop to the 0.4 s sim tick")
    ap.add_argument("--api", default="http://127.0.0.1:8840")
    ap.add_argument("--out", default="", help="episode.json dir (offline)")
    args = ap.parse_args(argv)
    if args.windows <= 0 and args.mode != "live":
        ap.error("--windows 0 (infinite) is only valid in --mode live")

    t_start = time.perf_counter()

    # ---------------------------------------------------------- engine boot
    n_total = args.actors
    gens, bank, names = ardy.build_agents(n_total, steps=args.steps)
    agents: list[Agent] = []
    spawn = spawn_agents(n_total)
    bx0, bx1, bz0, bz1 = WORLD_BOUNDS
    for i, gen in enumerate(gens):
        key = chr(ord("A") + i)
        fsm = DualStateMachine()
        if i == 0 or not args.join_after:
            origin, yaw0 = spawn[i]
        else:
            # late joiners spawn in a far corner, out of sight range
            origin = ((bx1 - 1.6) if i % 2 else (bx0 + 1.6),
                      (bz0 + 1.6) if i % 2 else (bz1 - 1.6))
            yaw0 = math.atan2(-origin[0], -origin[1])
        a = Agent(key=key, label=f"AI-{key}", origin=origin, yaw0=yaw0,
                  gen=gen, fsm=fsm, names=names, phase=0.05 * i)
        # ONE machine, two modes: the FSM itself judges the environment --
        # alone -> explorer runner; someone within sight -> social mode.
        a.social_active = False
        a.last_vis_t = -99.0
        a.nav = Explorer(bounds=WORLD_BOUNDS,
                         props=[(x, z, r) for x, z, r in WORLD_PROPS],
                         rng=__import__("random").Random(args.seed + i))
        a.nav_state = {"intent": "stand", "speed": 0.0}
        a.join_t = 0.0 if (i == 0 or not args.join_after) else i * args.join_after
        a.joined = a.join_t <= 0.0
        a.pos = np.asarray([origin[0], ardy.STAND_HEIGHT, origin[1]],
                           np.float32)
        a.heading = a.face = a.face_prev = yaw0
        a.stand_snap = ardy.seed_stand(gen, bank)
        a.local_root = np.asarray(
            gen.joints[-1, gen.layout.root_idx].detach().cpu().numpy(),
            np.float32)
        agents.append(a)

    # ------------------------------------------------------- vision backend
    yolo = page = None
    if args.vision == "render":
        from .vision import RenderVision, load_yolo
        yolo = load_yolo(ROOT)
        page = open_platform(args.api)
        vision_backend = RenderVision(page, yolo)
    else:
        vision_backend = None

    # ---------------------------------------------------------- live client
    import httpx
    client = httpx.Client(timeout=30.0)
    if args.mode == "live":
        # the backend may still be booting: wait for health before pushing
        for _ in range(60):
            try:
                if client.get(args.api + "/api/health", timeout=2.0).json().get("ok"):
                    break
            except Exception:
                pass
            time.sleep(1.0)
    # live streams are started when each agent joins (see the loop)

    rows: list[dict] = []
    timeline: list[dict] = []

    rows: list[dict] = []
    timeline: list[dict] = []
    started_streams: list[str] = []
    w = 0
    while args.windows <= 0 or w < args.windows:
        cyc = time.perf_counter()
        t_sim = w * TICK_S

        # ---- 0. late join (live demo: solo world first, others appear) ----
        for a in agents:
            if not a.joined and t_sim >= a.join_t:
                a.joined = True
                if args.mode == "live":
                    r = client.post(args.api + "/api/live/start", json={
                        "actor": a.key, "joint_names": names, "fps": 20.0,
                        "title": a.label})
                    r.raise_for_status()
                print(json.dumps({"event": "join", "actor": a.key,
                                  "t": round(t_sim, 1)}, ensure_ascii=False),
                      flush=True)
        active = [a for a in agents if a.joined]

        # ---- 1. perception + mode switch (one machine, two modes) ----
        for a in active:
            vis = (vision_backend.perceive(a, active, t_sim)
                   if vision_backend else geometric(a, active, t_sim))
            if vis["visible"]:
                a.last_seen_t = t_sim
                a.last_vis_t = t_sim
                d = float(vis["distance"])
                a.dist_ema = d if a.dist_ema is None else (
                    0.55 * a.dist_ema + 0.45 * d)
                vis["distance"] = a.dist_ema
            else:
                vis["distance"] = a.dist_ema or 9.0
            if a.social_active:
                observe_and_decide(a, active, t_sim, vis)
                # nobody in sight for a while and no live conversation ->
                # judge it a single-person environment, back to exploring.
                # (No state whitelist: a dyad state whose partner vanished
                # must not trap the agent in converse_* forever.)
                if (t_sim - a.last_vis_t > SOLO_TIMEOUT
                        and a.speaking_until <= t_sim
                        and a.addressed_until <= t_sim):
                    a.social_active = False
                    print(json.dumps(
                        {"event": "mode", "actor": a.key, "mode": "explore",
                         "t": round(t_sim, 1)}, ensure_ascii=False), flush=True)
                # social approach steering: keep the walk aimed at the
                # partner (rate-limited), otherwise a fixed heading walks
                # straight past them between FSM decisions.  Retreat states
                # invert this: turn away first, then walk forward.  Transit
                # walks keep their own heading (steering them toward the
                # partner made two agents orbit each other).
                if (vis["visible"] and a.last_dec is not None
                        and a.last_dec.plan.ardy_intent in ("walk", "walk_fwd")):
                    other = vis["other"]
                    if a.last_dec.state in RETREAT_STATES:
                        a.steer(math.atan2(float(a.pos[0] - other.pos[0]),
                                           float(a.pos[2] - other.pos[2])))
                    elif a.last_dec.state in APPROACH_WALK_STATES:
                        a.steer(math.atan2(float(other.pos[0] - a.pos[0]),
                                           float(other.pos[2] - a.pos[2])))
            else:
                near = None
                nd = None
                for o in active:
                    if o is a:
                        continue
                    d = math.hypot(float(o.pos[0] - a.pos[0]),
                                   float(o.pos[2] - a.pos[2]))
                    if nd is None or d < nd:
                        near, nd = (float(o.pos[0]), float(o.pos[2])), d
                if vis["visible"]:
                    # social environment triggered: enter the social FSM and
                    # turn toward the person (skip the nav tick this window)
                    a.social_active = True
                    a.state_entry = t_sim
                    a.steer(math.atan2(float(vis["other"].pos[0] - a.pos[0]),
                                       float(vis["other"].pos[2] - a.pos[2])))
                    print(json.dumps(
                        {"event": "mode", "actor": a.key, "mode": "social",
                         "t": round(t_sim, 1)}, ensure_ascii=False), flush=True)
                else:
                    a.nav_state = a.nav.update(a, t_sim, near)
                a.last_d_vis = nd
                a.vision = vis

        # ---- 2. plans -> one batched ARDY window for ALL active agents ----
        walk_flags: list[bool] = []
        intents: list[str] = []
        for a in active:
            if a.social_active:
                intent = a.last_dec.plan.ardy_intent if a.last_dec else "stand"
                d_vis = a.last_d_vis if a.last_d_vis is not None else 9.0
                approaching = (a.last_dec is not None
                               and a.last_dec.state in APPROACH_WALK_STATES)
                walk = a.plan_motion(
                    intent, d_vis,
                    away=(a.last_dec is not None
                          and a.last_dec.state in RETREAT_STATES),
                    # 1.0: walking stops when the tick-start d <= 1.1; with the
                    # 2.0 m creep zone the last windows close ~0.15 m each, so
                    # the landing lands inside the greet band (<=1.1) and far
                    # above the 0.6 m personal-space line
                    stop_at=1.0 if approaching else None)
            else:
                intent = a.nav_state["intent"]
                walk = intent == "walk"
            text = ("wave" if intent == "wave"
                    else "walk" if walk else "stand")
            # autoregressive stand drift: re-seed from the model's own stand
            raw_y = float(a.gen.joints[-1, a.gen.layout.root_idx][1])
            if (not walk
                    and ((raw_y < 0.88 and t_sim - a.last_pose_reset_t > 2.0)
                         or t_sim - a.last_pose_reset_t > 6.0)):
                ardy.reset_stand(a.gen, a.stand_snap, a.local_root,
                                 ardy.STAND_HEIGHT)
                a.last_pose_reset_t = t_sim
                a.pending_blend = True   # crossfade the pose, don't pop
            ardy.set_text(a.gen, bank[text])
            intents.append(intent)
            walk_flags.append(walk)
        chunks = ardy.window_batch([a.gen for a in active],
                                   seed=args.seed + w)

        # ---- 3. world transform + controller + push ----
        for a, chunk, walk in zip(active, chunks, walk_flags):
            blend = a.last_rot if getattr(a, "pending_blend", False) else None
            a.pending_blend = False
            frames = a.world_frames(chunk, t0=t_sim, blend_from=blend)
            offsets = a.controller_motion(chunk, walk)
            a.apply_motion(frames, offsets, walk)
            a.commit(frames, t_sim)
            if args.mode == "live" and frames:
                r = client.post(args.api + "/api/live/append", json={
                    "actor": a.key, "frames": frames})
                if r.status_code == 404:
                    # backend restarted mid-run: re-register and re-append
                    client.post(args.api + "/api/live/start", json={
                        "actor": a.key, "joint_names": names, "fps": 20.0,
                        "title": a.label})
                    client.post(args.api + "/api/live/append", json={
                        "actor": a.key, "frames": frames})
                else:
                    r.raise_for_status()

        # ---- 4. conversation floor (active agents) ----
        floor_policy(active, t_sim)

        # ---- 5. timeline (HUD) ----
        if w % 2 == 0:
            hud = {}
            entry = {"t": round(TICK_S + t_sim, 2)}
            for a in active:
                item = {
                    "state": (a.last_state if a.social_active else
                              ("explore" if a.nav_state["intent"] == "walk"
                               else "wander_rest")),
                    "emotion": a.last_emotion,
                    "speech": a.speech if a.speaking_until > t_sim else "",
                    "distance": round(float(a.dist_ema or 0.0), 2),
                    "visible": bool(a.vision.get("visible")),
                    "conf": round(float(a.vision.get("conf", 0.0)), 2),
                    "mode": a.vision.get("mode", "geo"),
                }
                entry[a.key.lower()] = item
                hud[a.key] = item
                # transition log: every state change, with distance context
                prev_state = getattr(a, "_log_state", None)
                if item["state"] != prev_state:
                    a._log_state = item["state"]
                    print(json.dumps(
                        {"w": w, "t": round(t_sim, 1), "actor": a.key,
                         "to": item["state"], "d": item["distance"],
                         "emo": item["emotion"]},
                        ensure_ascii=False), flush=True)
            timeline.append(entry)
            if args.mode == "live":
                client.post(args.api + "/api/live/hud",
                            json={"hud": hud})
        if w % 10 == 0:
            row = {"w": w, "t": round(t_sim, 1)}
            for a, intent in zip(active, intents):
                row[a.key] = {
                    "state": (a.last_state if a.social_active else "explore"),
                    "emo": a.last_emotion,
                    "intent": intent,
                    "d": round(float(a.dist_ema or 0.0), 2),
                    "pos": [round(float(v), 2) for v in a.pos],
                }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        w += 1

        if args.mode == "live" and args.realtime:
            elapsed = time.perf_counter() - cyc
            if elapsed < TICK_S:
                time.sleep(TICK_S - elapsed)

    wall = time.perf_counter() - t_start

    # ------------------------------------------------------------- outputs
    if args.mode == "offline":
        out_dir = Path(args.out) if args.out else WEB_DATA
        out_dir.mkdir(parents=True, exist_ok=True)
        episode = {
            "schema": "al_duet.episode.v1",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "scene": args.scene,
            "fps": 20.0,
            "joint_names": names,
            "sim_s": round(args.windows * TICK_S, 3),
            "wall_s": round(wall, 3),
            "steps": args.steps,
            "seed": args.seed,
            "perception": args.vision,
            "actors": {
                a.key: {
                    "label": a.label,
                    "base_root": [a.origin[0], 0.0, a.origin[1]],
                    "base_heading": a.yaw0,
                    "frames": a.all_frames,
                } for a in agents
            },
            "timeline": timeline,
        }
        ep_path = out_dir / "episode.json"
        ep_path.write_text(
            json.dumps(episode, ensure_ascii=False, separators=(",", ":")),
            "utf-8")
        print("episode", ep_path, "frames",
              {a.key: len(a.all_frames) for a in agents},
              "timeline", len(timeline), flush=True)
    else:
        for a in agents:
            client.post(args.api + "/api/live/stop", json={"actor": a.key})

    summary = {
        "wall_s": round(wall, 3),
        "sim_s": round(args.windows * TICK_S, 3),
        "steps": args.steps,
        "agents": args.actors,
        "vision_rate": {
            a.key: round(sum(1 for e in timeline
                             if e[a.key.lower()].get("visible"))
                         / max(1, len(timeline)), 3) for a in agents},
        "end": {a.key: {"state": (a.last_state if a.social_active
                                  else "explore"),
                        "pos": [round(float(v), 3) for v in a.pos]}
                for a in agents},
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "last_run.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False,
                   indent=1), "utf-8")
    return summary


def open_platform(api: str):
    """Open the platform page in Edge (used by --vision render)."""
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    br = pw.chromium.launch(headless=False, channel="msedge",
                            args=["--start-maximized"])
    page = br.new_page(viewport={"width": 1500, "height": 950})
    page.goto(api + "/?live=1", wait_until="domcontentloaded", timeout=60000)
    page.wait_for_function(
        "window.duet && window.duet.ready", timeout=90000)
    return page


if __name__ == "__main__":
    run()
