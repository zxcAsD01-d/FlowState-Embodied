"""Types shared by the social state machine, encoder and motion layer.

The state machine is deliberately text-first: every state carries a natural
language ``text`` that an encoder can read later.  Rules only reference facts
listed in :mod:`al_social.rules`, never raw tensors, so the same machine can be
driven by a video replay, a live VRM-HUB stream or a unit fixture.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable


# --------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------
@dataclass
class PersonState:
    id: str
    distance: float = 99.0
    bearing_deg: float = 0.0          # 0 = straight ahead, +right
    facing_me: float = 0.0            # cos(angle between their forward and me)
    relationship: str = "stranger"    # stranger | acquaintance | friend | close
    speaking: bool = False
    addressed_to_me: bool = False
    visible: bool = True
    moving: bool = False
    speed: float = 0.0                # m/s
    confidence: float = 1.0
    position: tuple[float, float, float] | None = None
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox: tuple[float, float, float, float] | None = None  # x0,y0,x1,y1 px
    last_seen: float = 0.0
    name: str | None = None


@dataclass
class Observation:
    t: float = 0.0
    self_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    self_heading_deg: float = 0.0
    self_speed: float = 0.0
    self_moving: bool = False
    self_menu_open: bool = False
    self_collision: bool = False
    self_stuck: bool = False
    self_tracking_ok: bool = True
    self_has_goal: bool = False
    self_at_goal: bool = False
    scene_mode: str = "public"        # public | private | event | menu
    scene_crowd: float = 0.0          # 0..1
    audio_activity: bool = False
    audio_speech: bool = False
    audio_energy: float = 0.0
    floor_holder: str | None = None    # person id, "me", or None
    focus_id: str | None = None
    group_size: int = 0
    group_cohesion: float = 0.0        # 0..1
    # interpersonal / psychological state (slow variables)
    relationship_stage: str = "stranger"  # stranger|acquaintance|friend|close|intimate|strained
    consent_state: str = "none"           # none|asked|granted|limited|denied|withdrawn
    comfort: float = 0.5                  # 0..1 felt safety/comfort
    arousal: float = 0.2                  # 0..1 arousal (not valence)
    intimacy_level: int = 0               # 0 none .. 3 high (affectionate, non-explicit)
    touch_intent: str = "none"            # none|headpat|pet|hug|kiss|handhold
    # proactive courtship signals (slow/fast, all observable or explicitly set)
    mutual_gaze: float = 0.0              # 0..1
    reciprocity: float = 0.0              # 0..1 (reply length/engagement)
    proximity: float = 0.0                # 0..1 (closer = higher)
    initiative_score: float = 0.0
    initiative_move: str = "none"         # none|compliment|self_disclose|invite|flirt
    initiative_ready: bool = False
    rejection_count: int = 0
    since_last_attempt: float = 999.0
    # emotion FSM (fast-ish, 0.5-2 Hz)
    emotion_label: str = "calm"
    emotion_valence: float = 0.10
    emotion_arousal: float = 0.30
    emotion_dominance: float = 0.00
    emotion_comfort: float = 0.60
    emotion_intensity: float = 0.20
    persons: list[PersonState] = field(default_factory=list)
    events: dict[str, list[float]] = field(default_factory=dict)
    # provenance for replay/debug: clip, frame index, asr segment id, ...
    source: dict[str, Any] = field(default_factory=dict)

    def nearest(self, max_distance: float = 99.0) -> PersonState | None:
        cand = [p for p in self.persons if p.visible and p.distance <= max_distance]
        return min(cand, key=lambda p: p.distance) if cand else None

    def focus(self) -> PersonState | None:
        if self.focus_id:
            for p in self.persons:
                if p.id == self.focus_id:
                    return p
        return self.nearest()

    def speaker(self) -> PersonState | None:
        if self.floor_holder in (None, "me"):
            return None
        for p in self.persons:
            if p.id == self.floor_holder:
                return p
        return next((p for p in self.persons if p.speaking), None)

    def visible_persons(self) -> list[PersonState]:
        return [p for p in self.persons if p.visible]


@dataclass
class ActionProfile:
    intent: str = "stand"             # stand | walk | walk_fwd | wave
    target: str = "none"              # none | nearest | speaker | addressee | focus | group_center | path | exit
    gaze: str = "scan"                # scan | target_face | speaker | shared | down | path
    hand: str = "none"                # none | wave | point | offer | rest
    heading: str = "keep"             # keep | face_target | path | away
    distance_band: tuple[float, float] = (0.0, 0.0)
    stop_radius: float = 0.0
    speed_band: tuple[float, float] = (0.0, 0.0)
    hold_s: float = 0.0
    speech_act: str = "none"          # none | greet | question | answer | backchannel | farewell | repair
    touch: str = "none"               # none | headpat | pet | hug | kiss | handhold
    consent_required: bool = False
    intimacy_level: int = 0
    hand_point_offset: tuple[float, float, float] | None = None


@dataclass
class Transition:
    to: str
    when: dict[str, Any]
    prior: float = 0.0
    why: str = ""
    learnable: bool = True
    cooldown: float = 0.0
    rule_id: str = ""
    hard_priority: int = 0            # >0 = deterministic override, bypasses min dwell
    source: str = "manual"


@dataclass
class StateSpec:
    id: str
    name_zh: str
    family: str
    l1: int | None
    text: str
    invariants: list[str] = field(default_factory=list)
    action: ActionProfile = field(default_factory=ActionProfile)
    min_dwell: float = 0.5
    max_dwell: float = 20.0
    fallback: str | None = None
    priority: int = 0
    transitions: list[Transition] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def action_dict(self) -> dict[str, Any]:
        return asdict(self.action)


@dataclass
class ActionPlan:
    state: str
    state_text: str
    family: str
    t: float
    target_id: str | None = None
    target_pos: tuple[float, float, float] | None = None
    target_vel: tuple[float, float, float] | None = None
    target_confidence: float = 0.0
    stand_point: tuple[float, float, float] | None = None
    heading_deg: float | None = None
    gaze_point: tuple[float, float, float] | None = None
    gaze_target_id: str | None = None
    hand_point: tuple[float, float, float] | None = None
    distance_band: tuple[float, float] = (0.0, 0.0)
    stop_radius: float = 0.0
    speed_band: tuple[float, float] = (0.0, 0.0)
    hold_seconds: float = 0.0
    ardy_intent: str = "stand"
    hand_action: str = "none"
    speech_act: str = "none"
    touch_action: str = "none"
    consent_required: bool = False
    intimacy_level: int = 0
    keyframe_root: tuple[float, float, float] | None = None
    note: str = ""


@dataclass
class Decision:
    state: str
    previous: str | None
    rule_id: str | None
    rule_why: str
    hard: bool
    candidates: list[tuple[str, str, float]]   # (rule_id, to, score)
    plan: ActionPlan
    facts: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# helper math (no numpy dependency at runtime)
# --------------------------------------------------------------------------
def _dist(a: Iterable[float], b: Iterable[float]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5


def stand_point(target: tuple[float, float, float], self_pos, desired: float,
                heading_deg: float | None = None) -> tuple[float, float, float]:
    """A point ``desired`` metres from ``target`` toward ``self_pos``."""
    dx = float(self_pos[0]) - float(target[0])
    dz = float(self_pos[2]) - float(target[2])
    n = (dx * dx + dz * dz) ** 0.5
    if n < 1e-6:
        dx, dz, n = 0.0, -1.0, 1.0
    return (float(target[0]) + dx / n * desired, float(target[1]),
            float(target[2]) + dz / n * desired)


def heading_to(a, b) -> float:
    """Degrees, 0 = +z, clockwise positive (VRM-HUB convention)."""
    import math
    return math.degrees(math.atan2(float(b[0]) - float(a[0]),
                                   float(b[2]) - float(a[2])))


def yaw_of(vec) -> float:
    import math
    return math.degrees(math.atan2(float(vec[0]), float(vec[2])))
