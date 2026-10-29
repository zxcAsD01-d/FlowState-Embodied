"""Two coupled state machines: social (data-driven) + emotion (theory-driven).

Each tick:
1. emotion appraises the event produced by the previous social action (or an
   external event such as rejection/consent)
2. social sees the updated emotion fields and chooses its state/action
3. the new social action becomes the next appraisal event

The social machine stays the data/rule-driven policy.  Emotion only modulates
thresholds and distance/initiative bias; it does not override consent/safety.
"""
from __future__ import annotations

from dataclasses import dataclass

from .emotion import EmotionFSM, EmotionState
from .fsm import SocialFSM
from .schema import Decision, Observation


@dataclass
class DualStep:
    social: Decision
    emotion: EmotionState
    event: str


EVENT_BY_STATE = {
    "greet_initiate": "greet", "greet_ack": "greet", "approach_person": "greet",
    "compliment": "compliment", "self_disclose": "self_disclose",
    "invite_activity": "invite", "flirt_signal": "compliment",
    "ask_consent": "consent_ask", "consent_wait": "consent_ask",
    "headpat_offer": "consent_ask", "headpat_give": "intimacy",
    "headpat_receive": "intimacy", "caress_give": "intimacy",
    "caress_receive": "intimacy", "hug_hold": "intimacy",
    "kiss_accept": "intimacy", "repair_misstep": "repair",
    "intimate_withdraw": "withdraw", "kiss_decline": "rejection",
    "disengage": "withdraw", "part_farewell": "withdraw",
    "group_leave": "withdraw", "avoid_person": "crowd",
    "crowd_navigate": "crowd", "too_close_retreat": "threat",
    "freeze_unknown": "threat",
}


def event_from_decision(dec: Decision) -> str:
    if dec.plan.touch_action and dec.plan.touch_action != "none":
        return "intimacy"
    return EVENT_BY_STATE.get(dec.state, "neutral")


class DualStateMachine:
    def __init__(self, social: SocialFSM | None = None,
                 emotion: EmotionFSM | None = None):
        self.social = social or SocialFSM()
        self.emotion = emotion or EmotionFSM()
        self.pending_event = "neutral"

    def step(self, obs: Observation, external_event: str | None = None,
             intensity: float = 1.0) -> DualStep:
        event = external_event or self.pending_event
        emo = self.emotion.step(event, intensity)
        for k, v in emo.as_observation().items():
            setattr(obs, k, v)
        dec = self.social.step(obs)
        self.pending_event = event_from_decision(dec)
        return DualStep(dec, emo, event)

    def reset(self) -> None:
        self.social = SocialFSM()
        self.emotion.reset()
        self.pending_event = "neutral"
