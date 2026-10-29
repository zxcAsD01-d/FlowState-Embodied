"""Explicit-text social state machine.

States live in ``data/states.yaml``; global safety overrides in
``data/rules.yaml``.  Selection is:

1. hard global rules (priority, ignores min-dwell)
2. eligible soft rules: ``prior + rule_bias + theta . context_features``
3. argmax / softmax over those, if the current state has served ``min_dwell``

The learnable part is intentionally small: a per-rule bias and a shared
context weight vector.  It can be fit later on labelled state episodes; the
machine is fully rule-driven with all weights zero.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .rules import eval_guard, extract_facts, guard_facts, FACT_NAMES
from .schema import (ActionPlan, ActionProfile, Decision, Observation,
                     StateSpec, Transition, heading_to, stand_point)

DATA = Path(__file__).resolve().parent / "data"
FAMILIES = {"solo", "bystander", "dyad", "group", "transit", "safety"}
INTENTS = {"stand", "walk", "walk_fwd", "wave"}
CONTEXT_FEATURES = (
    "bias", "dwell", "dist", "crowd", "group", "speech", "moving",
    "friend", "addressed", "goal", "cohesion", "time_family",
)


def _transition(d: dict[str, Any], source: str = "manual") -> Transition:
    return Transition(
        to=str(d["to"]),
        when=d.get("when") or {},
        prior=float(d.get("prior", 0.0)),
        why=str(d.get("why", "")),
        learnable=bool(d.get("learnable", True)),
        cooldown=float(d.get("cooldown", 0.0)),
        rule_id=str(d.get("id", "")),
        hard_priority=int(d.get("hard_priority", 0)),
        source=source,
    )


def _action(d: dict[str, Any] | None) -> ActionProfile:
    d = d or {}
    a = ActionProfile()
    for key in ("intent", "target", "gaze", "hand", "heading", "speech_act", "touch"):
        if key in d:
            setattr(a, key, str(d[key]))
    for key in ("distance_band", "speed_band"):
        if key in d and d[key] is not None:
            setattr(a, key, tuple(float(x) for x in d[key]))
    for key in ("stop_radius", "hold_s"):
        if key in d:
            setattr(a, key, float(d[key]))
    a.consent_required = bool(d.get("consent_required", False))
    a.intimacy_level = int(d.get("intimacy_level", 0))
    if d.get("hand_point_offset") is not None:
        a.hand_point_offset = tuple(float(x) for x in d["hand_point_offset"])
    return a


@lru_cache(maxsize=8)
def load_machine(data_dir: str | Path | None = None) -> tuple[dict[str, StateSpec],
                                                              list[Transition],
                                                              list[str]]:
    root = Path(data_dir) if data_dir else DATA
    raw = yaml.safe_load((root / "states.yaml").read_text("utf-8"))
    grows = yaml.safe_load((root / "rules.yaml").read_text("utf-8"))

    states: dict[str, StateSpec] = {}
    for s in raw["states"]:
        st = StateSpec(
            id=str(s["id"]), name_zh=str(s.get("name_zh", s["id"])),
            family=str(s["family"]), l1=(None if s.get("l1") is None else int(s["l1"])),
            text=str(s["text"]).strip(),
            invariants=[str(x) for x in s.get("invariants", [])],
            action=_action(s.get("action")),
            min_dwell=float(s.get("min_dwell", 0.5)),
            max_dwell=float(s.get("max_dwell", 20.0)),
            fallback=(str(s["fallback"]) if s.get("fallback") else None),
            priority=int(s.get("priority", 0)),
            tags=[str(x) for x in s.get("tags", [])],
        )
        for i, t in enumerate(s.get("transitions", [])):
            tr = _transition(t)
            tr.rule_id = tr.rule_id or f"{st.id}->{tr.to}:{i}"
            st.transitions.append(tr)
        states[st.id] = st

    glob: list[Transition] = []
    for i, r in enumerate(grows.get("global_rules", [])):
        tr = _transition(r, source="global")
        tr.rule_id = tr.rule_id or f"global:{i}"
        tr.when = dict(tr.when or {})
        tr.when.setdefault("_exclude", r.get("exclude_states", []))
        glob.append(tr)
    starts = [str(x) for x in raw.get("start_states", [])]

    # generated timeout exits: explicit text, not hidden behaviour
    for st in states.values():
        if st.fallback and st.fallback in states:
            st.transitions.append(Transition(
                to=st.fallback,
                when={"fact": "dwell", "op": ">=", "value": st.max_dwell},
                prior=-0.4,
                why=f"在「{st.name_zh}」停留超过 {st.max_dwell:g} 秒，回落到「{states[st.fallback].name_zh}」",
                rule_id=f"{st.id}:timeout",
                source="timeout",
            ))
    return states, glob, starts


def validate_machine(states: dict[str, StateSpec], glob: list[Transition],
                     starts: list[str]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    for sid, st in states.items():
        if st.family not in FAMILIES:
            errors.append(f"{sid}: bad family {st.family}")
        if st.l1 not in (None, 1, 2, 3, 4, 5):
            errors.append(f"{sid}: bad l1 {st.l1}")
        if st.action.intent not in INTENTS:
            errors.append(f"{sid}: bad intent {st.action.intent}")
        if len(st.text) < 10:
            warnings.append(f"{sid}: short text")
        for tr in st.transitions:
            if tr.to not in states:
                errors.append(f"{tr.rule_id}: target {tr.to} missing")
            for f in guard_facts(tr.when):
                if f.startswith(("recent.", "since.")):
                    continue
                if f not in FACT_NAMES:
                    errors.append(f"{tr.rule_id}: unknown fact {f}")
        if st.fallback and st.fallback not in states:
            errors.append(f"{sid}: fallback {st.fallback} missing")
    for tr in glob:
        if tr.to not in states:
            errors.append(f"{tr.rule_id}: target {tr.to} missing")
        for f in guard_facts(tr.when):
            if f.startswith(("recent.", "since.")):
                continue
            if f not in FACT_NAMES:
                errors.append(f"{tr.rule_id}: unknown fact {f}")

    # reachability over all explicit + global + timeout edges
    edges: dict[str, set[str]] = {sid: set() for sid in states}
    for sid, st in states.items():
        edges[sid].update(tr.to for tr in st.transitions)
    for tr in glob:
        for sid in states:
            if sid not in tr.when.get("_exclude", []):
                edges[sid].add(tr.to)
    seen: set[str] = set()
    stack = [s for s in starts if s in states]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(edges[cur] - seen)
    unreachable = sorted(set(states) - seen)
    if unreachable:
        warnings.append(f"unreachable: {unreachable}")
    no_exit = sorted(sid for sid, st in states.items()
                     if not st.transitions and not any(
                         sid not in tr.when.get("_exclude", []) for tr in glob))
    if no_exit:
        warnings.append(f"no exit: {no_exit}")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "states": len(states),
        "rules": sum(len(st.transitions) for st in states.values()) + len(glob),
        "global_rules": len(glob),
        "reachable": len(seen),
        "unreachable": unreachable,
        "by_family": {f: sum(1 for s in states.values() if s.family == f)
                      for f in sorted(FAMILIES)},
        "by_l1": {k: sum(1 for s in states.values() if s.l1 == k)
                  for k in (1, 2, 3, 4, 5)},
    }


class TransitionScorer:
    """Shared context weights + per-rule bias; all zero means pure rules."""

    def __init__(self) -> None:
        self.bias: dict[str, float] = {}
        self.theta: dict[str, float] = {k: 0.0 for k in CONTEXT_FEATURES}

    def features(self, tr: Transition, facts: dict[str, Any],
                 dwell: float, time_in_family: float) -> dict[str, float]:
        dist = float(facts.get("nearest.distance", 8.0))
        return {
            "bias": 1.0,
            "dwell": min(float(dwell), 10.0) / 10.0,
            "dist": max(0.0, min(dist, 8.0)) / 8.0,
            "crowd": float(facts.get("scene.crowd", 0.0)),
            "group": min(float(facts.get("group.size", 0)), 8.0) / 8.0,
            "speech": 1.0 if facts.get("audio.speech") else 0.0,
            "moving": 1.0 if facts.get("self.moving") else 0.0,
            "friend": 1.0 if facts.get("nearest.relationship") in
            ("friend", "close") else 0.0,
            "addressed": 1.0 if facts.get("nearest.addressed_to_me") else 0.0,
            "goal": 1.0 if facts.get("self.at_goal") else 0.0,
            "cohesion": float(facts.get("group.cohesion", 0.0)),
            "time_family": min(float(time_in_family), 60.0) / 60.0,
        }

    def score(self, tr: Transition, facts: dict[str, Any], dwell: float,
              time_in_family: float) -> float:
        f = self.features(tr, facts, dwell, time_in_family)
        return float(tr.prior + self.bias.get(tr.rule_id, 0.0) +
                     sum(self.theta[k] * f[k] for k in CONTEXT_FEATURES))


class SocialFSM:
    def __init__(self, data_dir: str | Path | None = None,
                 start: str | None = None, temperature: float = 0.35,
                 seed: int = 0, semantic_weight: float = 0.0,
                 semantic_confidence: float = 0.55) -> None:
        self.states, self.global_rules, starts = load_machine(data_dir)
        self.report = validate_machine(self.states, self.global_rules, starts)
        if not self.report["ok"]:
            raise ValueError("invalid state machine: " + "; ".join(self.report["errors"]))
        self.state = start or (starts[0] if starts else next(iter(self.states)))
        self.prev: str | None = None
        self.entry_t = 0.0
        self.family_entry_t = 0.0
        self.prev_family = self.states[self.state].family
        self.temperature = float(temperature)
        self.rng = __import__("random").Random(seed)
        self.scorer = TransitionScorer()
        self.semantic_weight = float(semantic_weight)
        self.semantic_confidence = float(semantic_confidence)
        self.last_fired: dict[str, float] = {}
        self.memory: dict[str, Any] = {}

    @property
    def spec(self) -> StateSpec:
        return self.states[self.state]

    def _eligible_global(self, obs: Observation, facts: dict[str, Any],
                         dwell: float) -> list[Transition]:
        out = []
        for tr in self.global_rules:
            if self.state in tr.when.get("_exclude", []):
                continue
            if tr.to == self.state:
                continue
            if obs.t - self.last_fired.get(tr.rule_id, -1e9) < tr.cooldown:
                continue
            if eval_guard(tr.when, facts, obs):
                out.append(tr)
        return out

    def _eligible_soft(self, obs: Observation, facts: dict[str, Any],
                       dwell: float) -> list[Transition]:
        out = []
        if dwell < self.spec.min_dwell:
            return out
        for tr in self.spec.transitions:
            if tr.to == self.state:
                continue
            if obs.t - self.last_fired.get(tr.rule_id, -1e9) < tr.cooldown:
                continue
            if eval_guard(tr.when, facts, obs):
                out.append(tr)
        return out

    def _plan(self, obs: Observation) -> ActionPlan:
        st = self.spec
        a = st.action
        target = None
        if a.target in ("nearest", "focus", "addressee", "away"):
            target = obs.focus() if a.target in ("focus", "addressee", "away") else obs.nearest()
            if a.target == "addressee" and target is not None and not target.addressed_to_me:
                target = obs.nearest()
        elif a.target == "speaker":
            target = obs.speaker() or obs.focus()
        elif a.target in ("group_center", "path", "exit"):
            target = None

        target_pos = None
        if target is not None:
            if target.position is not None:
                target_pos = tuple(float(v) for v in target.position)
            else:
                ang = math.radians(obs.self_heading_deg + target.bearing_deg)
                target_pos = (obs.self_pos[0] + math.sin(ang) * target.distance,
                              obs.self_pos[1],
                              obs.self_pos[2] + math.cos(ang) * target.distance)

        group_center = None
        pts = [p.position for p in obs.visible_persons() if p.position is not None]
        if pts:
            group_center = (sum(p[0] for p in pts) / len(pts), obs.self_pos[1],
                            sum(p[2] for p in pts) / len(pts))
        if a.target == "group_center" and group_center is not None:
            target_pos = group_center
            target = None

        stand = None
        if target_pos is not None and a.distance_band[1] > 0:
            desired = sum(a.distance_band) / 2.0
            stand = stand_point(target_pos, obs.self_pos, desired)

        heading = None
        if a.heading == "face_target" and target_pos is not None:
            heading = heading_to(obs.self_pos, target_pos)
        elif a.heading == "path":
            heading = float(obs.source.get("path_heading", obs.self_heading_deg))
        elif a.heading == "away" and target_pos is not None:
            heading = heading_to(target_pos, obs.self_pos)
        elif a.heading == "keep":
            heading = obs.self_heading_deg

        gaze = None
        if a.gaze == "target_face" and target_pos is not None:
            gaze = (target_pos[0], target_pos[1] + 1.45, target_pos[2])
        elif a.gaze == "speaker":
            sp = obs.speaker() or target
            if sp is not None and sp.position is not None:
                gaze = (sp.position[0], sp.position[1] + 1.45, sp.position[2])
            elif target_pos is not None:
                gaze = (target_pos[0], target_pos[1] + 1.45, target_pos[2])
        elif a.gaze == "shared":
            gaze = obs.source.get("shared_point") or (
                (target_pos[0], target_pos[1] + 1.2, target_pos[2])
                if target_pos is not None else None)
        elif a.gaze == "down":
            gaze = (obs.self_pos[0], obs.self_pos[1] + 0.6, obs.self_pos[2])

        hand = None
        if a.hand in ("wave", "point", "offer", "headpat", "pet", "hug"):
            base = target_pos or (obs.self_pos[0], obs.self_pos[1] + 1.2, obs.self_pos[2])
            if a.hand == "headpat":
                off = (0.0, 1.55, 0.0)
            elif a.hand == "pet":
                off = (0.16, 1.45, 0.05)
            elif a.hand == "hug":
                off = (0.0, 1.20, 0.0)
            elif a.hand == "wave":
                off = (0.35, 0.1, 0.0)
            elif a.hand == "point":
                off = (0.0, 1.45, 0.5)
            else:
                off = (0.0, 1.0, 0.55)
            if a.hand_point_offset is not None:
                off = a.hand_point_offset
            hand = (base[0] + off[0], base[1] + off[1], base[2] + off[2])

        keyframe = None
        if a.intent in ("walk", "walk_fwd"):
            if a.target in ("path", "exit") or target_pos is None:
                ang = math.radians(heading if heading is not None else obs.self_heading_deg)
                dist = 1.2 if a.intent == "walk" else 1.5
                keyframe = (obs.self_pos[0] + math.sin(ang) * dist,
                            obs.self_pos[1],
                            obs.self_pos[2] + math.cos(ang) * dist)
            elif a.target == "away" and target_pos is not None:
                ang = math.radians(heading_to(target_pos, obs.self_pos))
                keyframe = (obs.self_pos[0] + math.sin(ang) * 0.8,
                            obs.self_pos[1],
                            obs.self_pos[2] + math.cos(ang) * 0.8)
            else:
                keyframe = stand or target_pos

        return ActionPlan(
            state=st.id, state_text=st.text, family=st.family, t=obs.t,
            target_id=(target.id if target is not None else None),
            target_pos=target_pos,
            target_vel=(target.velocity if target is not None else None),
            target_confidence=(target.confidence if target is not None else 0.0),
            stand_point=stand, heading_deg=heading, gaze_point=gaze,
            gaze_target_id=(target.id if target is not None else None),
            hand_point=hand, distance_band=a.distance_band,
            stop_radius=a.stop_radius, speed_band=a.speed_band,
            hold_seconds=a.hold_s, ardy_intent=a.intent,
            hand_action=a.hand, speech_act=a.speech_act,
            touch_action=a.touch, consent_required=a.consent_required,
            intimacy_level=a.intimacy_level,
            keyframe_root=keyframe, note=st.name_zh,
        )

    def force_state(self, sid: str, t: float) -> None:
        """External estimator proposes a state; the FSM keeps the history."""
        if sid not in self.states or sid == self.state:
            return
        self.prev = self.state
        self.state = sid
        self.entry_t = float(t)
        fam = self.states[sid].family
        if fam != self.prev_family:
            self.family_entry_t = float(t)
        self.prev_family = fam

    def step(self, obs: Observation, *, sample: bool = False,
             state_prior: dict[str, float] | None = None) -> Decision:
        st = self.spec
        dwell = max(0.0, obs.t - self.entry_t)
        time_family = max(0.0, obs.t - self.family_entry_t)
        facts = extract_facts(obs, dwell=dwell, time_in_family=time_family,
                              memory=self.memory)
        facts["state.family"] = st.family
        facts["state.prev"] = self.prev or ""
        facts["state.prev_family"] = self.prev_family
        facts["state.intimate"] = "intimacy" in st.tags

        hard = self._eligible_global(obs, facts, dwell)
        soft = self._eligible_soft(obs, facts, dwell)
        candidates: list[tuple[str, str, float]] = []
        chosen: Transition | None = None
        hard_hit = False

        if hard:
            hard.sort(key=lambda tr: -tr.hard_priority)
            chosen = hard[0]
            hard_hit = True
            candidates = [(tr.rule_id, tr.to, float(tr.hard_priority))
                          for tr in hard]
        else:
            sem = None
            if state_prior and self.semantic_weight > 0:
                peak = max(state_prior.values()) if state_prior else 0.0
                if peak >= self.semantic_confidence:
                    sem = state_prior
            scored = []
            for tr in soft:
                sc = self.scorer.score(tr, facts, dwell, time_family)
                if sem is not None:
                    sc += self.semantic_weight * math.log(max(sem.get(tr.to, 1e-6), 1e-6))
                scored.append((sc, tr))
            scored.sort(key=lambda x: -x[0])
            candidates = [(tr.rule_id, tr.to, sc) for sc, tr in scored]
            if scored:
                if sample and len(scored) > 1:
                    mx = max(sc for sc, _ in scored)
                    wts = [math.exp((sc - mx) / max(self.temperature, 1e-3))
                           for sc, _ in scored]
                    r = self.rng.random() * sum(wts)
                    acc = 0.0
                    for w, (_, tr) in zip(wts, scored):
                        acc += w
                        if r <= acc:
                            chosen = tr
                            break
                else:
                    chosen = scored[0][1]

        prev = self.state
        if chosen is not None:
            self.last_fired[chosen.rule_id] = obs.t
            if chosen.to != self.state:
                self.prev = self.state
                self.state = chosen.to
                self.entry_t = obs.t
                new_family = self.states[self.state].family
                if new_family != self.prev_family:
                    self.family_entry_t = obs.t
                self.prev_family = new_family

        plan = self._plan(obs)
        return Decision(
            state=self.state, previous=prev,
            rule_id=(chosen.rule_id if chosen else None),
            rule_why=(chosen.why if chosen else "没有满足条件的转移，保持原状态"),
            hard=hard_hit, candidates=candidates, plan=plan, facts=facts,
            trace={"dwell": dwell, "time_in_family": time_family,
                   "soft_n": len(soft), "global_n": len(hard)},
        )
