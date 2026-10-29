"""Fact extraction and the tiny guard DSL used by ``states.yaml``.

Guards are data, not code.  Every atom is ``{fact, op, value}`` (optionally
``within`` for event recency) so the same atom id can later become a feature in
a trainable transition scorer.  Missing facts evaluate false instead of
raising, which keeps a live loop alive when a sensor drops out; the validator
separately rejects unknown fact names at load time.
"""
from __future__ import annotations

from typing import Any

from .schema import Observation

# Fact names are intentionally flat and stable: they are also feature keys.
FACT_NAMES = {
    "t", "dwell", "time_in_family", "state.prev", "state.prev_family",
    "state.family",
    "persons.count", "persons.visible", "persons.moving",
    "nearest.distance", "nearest.bearing", "nearest.facing_me",
    "nearest.relationship", "nearest.speaking", "nearest.addressed_to_me",
    "nearest.visible", "nearest.confidence", "nearest.moving", "nearest.speed",
    "nearest.id",
    "focus.distance", "focus.bearing", "focus.visible", "focus.relationship",
    "focus.speaking", "focus.addressed_to_me", "focus.id", "focus.facing_me",
    "focus.moving",
    "speaker.distance", "speaker.id", "speaker.relationship",
    "speaker.addressed_to_me", "speaker.group_size",
    "group.size", "group.cohesion",
    "self.speed", "self.moving", "self.heading", "self.menu_open",
    "self.collision", "self.stuck", "self.tracking_ok", "self.has_goal",
    "self.at_goal", "self.fallen",
    "scene.mode", "scene.crowd",
    "audio.activity", "audio.speech", "audio.energy", "gate.speak",
    "consent.state", "relationship.stage", "affect.comfort", "affect.arousal",
    "intimacy.level", "intent.touch", "state.intimate", "touch.recent",
    "interest.mutual_gaze", "interest.reciprocity", "interest.proximity",
    "initiative.score", "initiative.move", "initiative.ready",
    "initiative.rejection_count", "initiative.since_attempt",
    "emotion.label", "emotion.valence", "emotion.arousal",
    "emotion.dominance", "emotion.comfort", "emotion.intensity",
    "floor.holder", "floor.is_me",
    "memory.interactions", "memory.familiarity", "memory.last_family",
}

EVENT_PREFIXES = ("recent.", "since.")


def _get(obj, path: str, default=None):
    cur = obj
    for key in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
        if cur is None:
            return default
    return cur


def extract_facts(obs: Observation, *, dwell: float = 0.0,
                  time_in_family: float = 0.0,
                  memory: dict[str, Any] | None = None) -> dict[str, Any]:
    nearest = obs.nearest()
    focus = obs.focus()
    speaker = obs.speaker()
    visible = obs.visible_persons()
    facts: dict[str, Any] = {
        "t": obs.t,
        "dwell": dwell,
        "time_in_family": time_in_family,
        "persons.count": len(visible),
        "persons.visible": len(visible),
        "persons.moving": sum(1 for p in visible if p.moving),
        "group.size": int(obs.group_size or len(visible)),
        "group.cohesion": float(obs.group_cohesion),
        "self.speed": float(obs.self_speed),
        "self.moving": bool(obs.self_moving),
        "self.heading": float(obs.self_heading_deg),
        "self.menu_open": bool(obs.self_menu_open),
        "self.collision": bool(obs.self_collision),
        "self.stuck": bool(obs.self_stuck),
        "self.tracking_ok": bool(obs.self_tracking_ok),
        "self.has_goal": bool(obs.self_has_goal),
        "self.at_goal": bool(obs.self_at_goal),
        "self.fallen": bool(obs.source.get("fallen", False)),
        "scene.mode": obs.scene_mode,
        "scene.crowd": float(obs.scene_crowd),
        "audio.activity": bool(obs.audio_activity),
        "audio.speech": bool(obs.audio_speech),
        "audio.energy": float(obs.audio_energy),
        "gate.speak": float(obs.source.get("gate_speak", 0.0)),
        "consent.state": str(obs.consent_state),
        "relationship.stage": str(obs.relationship_stage),
        "affect.comfort": float(obs.comfort),
        "affect.arousal": float(obs.arousal),
        "intimacy.level": int(obs.intimacy_level),
        "intent.touch": str(obs.touch_intent),
        "state.intimate": False,
        "touch.recent": str(obs.source.get("last_touch", "")),
        "interest.mutual_gaze": float(obs.mutual_gaze),
        "interest.reciprocity": float(obs.reciprocity),
        "interest.proximity": float(obs.proximity),
        "initiative.score": float(obs.initiative_score),
        "initiative.move": str(obs.initiative_move),
        "initiative.ready": bool(obs.initiative_ready),
        "initiative.rejection_count": int(obs.rejection_count),
        "initiative.since_attempt": float(obs.since_last_attempt),
        "emotion.label": str(obs.emotion_label),
        "emotion.valence": float(obs.emotion_valence),
        "emotion.arousal": float(obs.emotion_arousal),
        "emotion.dominance": float(obs.emotion_dominance),
        "emotion.comfort": float(obs.emotion_comfort),
        "emotion.intensity": float(obs.emotion_intensity),
        "floor.holder": obs.floor_holder or "",
        "floor.is_me": obs.floor_holder == "me",
        "memory.interactions": 0,
        "memory.familiarity": 0.0,
        "memory.last_family": "",
    }
    if nearest is not None:
        facts.update({
            "nearest.distance": nearest.distance,
            "nearest.bearing": nearest.bearing_deg,
            "nearest.facing_me": nearest.facing_me,
            "nearest.relationship": nearest.relationship,
            "nearest.speaking": nearest.speaking,
            "nearest.addressed_to_me": nearest.addressed_to_me,
            "nearest.visible": nearest.visible,
            "nearest.confidence": nearest.confidence,
            "nearest.moving": nearest.moving,
            "nearest.speed": nearest.speed,
            "nearest.id": nearest.id,
        })
    if focus is not None:
        facts.update({
            "focus.distance": focus.distance,
            "focus.bearing": focus.bearing_deg,
            "focus.visible": focus.visible,
            "focus.relationship": focus.relationship,
            "focus.speaking": focus.speaking,
            "focus.addressed_to_me": focus.addressed_to_me,
            "focus.id": focus.id,
            "focus.facing_me": focus.facing_me,
        })
    if speaker is not None:
        facts.update({
            "speaker.distance": speaker.distance,
            "speaker.id": speaker.id,
            "speaker.relationship": speaker.relationship,
            "speaker.addressed_to_me": speaker.addressed_to_me,
            "speaker.group_size": int(obs.group_size or len(visible)),
        })
    if memory:
        facts.update(memory)
    return facts


def event_time(obs: Observation, name: str) -> float | None:
    ts = obs.events.get(name)
    return max(ts) if ts else None


def eval_atom(atom: dict[str, Any], facts: dict[str, Any],
              obs: Observation | None = None) -> bool:
    fact = str(atom.get("fact", ""))
    op = str(atom.get("op", "=="))
    want = atom.get("value")
    within = atom.get("within")

    if fact.startswith(("recent.", "since.")) and obs is not None:
        kind, name = fact.split(".", 1)
        ts = event_time(obs, name)
        if kind == "recent":
            window = 2.0 if within is None else float(within)
            have = ts is not None and (obs.t - ts) <= window
        else:
            have = 99.0 if ts is None else max(0.0, obs.t - ts)
        got = have
    else:
        got = facts.get(fact)

    if got is None:
        return False
    try:
        if op == "==":
            return got == want
        if op == "!=":
            return got != want
        if op == "<":
            return float(got) < float(want)
        if op == "<=":
            return float(got) <= float(want)
        if op == ">":
            return float(got) > float(want)
        if op == ">=":
            return float(got) >= float(want)
        if op == "in":
            return got in want
        if op == "not_in":
            return got not in want
    except (TypeError, ValueError):
        return False
    raise ValueError(f"unknown op {op!r}")


def eval_guard(when: dict[str, Any] | None, facts: dict[str, Any],
               obs: Observation | None = None) -> bool:
    if not when:
        return True
    if "all" in when:
        return all(eval_guard(w, facts, obs) for w in when["all"])
    if "any" in when:
        return any(eval_guard(w, facts, obs) for w in when["any"])
    if "not" in when:
        return not eval_guard(when["not"], facts, obs)
    if "fact" in when:
        return eval_atom(when, facts, obs)
    raise ValueError(f"bad guard node: {when}")


def guard_facts(when: dict[str, Any] | None) -> set[str]:
    """Collect fact names for validation / feature templates."""
    out: set[str] = set()
    if not when:
        return out
    if "fact" in when:
        out.add(str(when["fact"]))
    for key in ("all", "any"):
        for w in when.get(key, []):
            out |= guard_facts(w)
    if "not" in when:
        out |= guard_facts(when["not"])
    return out
