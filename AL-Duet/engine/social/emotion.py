"""Emotion state machine grounded in appraisal / dimensional emotion theory.

Theory anchors:
* Russell (1980) valence-arousal circumplex.
* Mehrabian & Russell (1974) PAD (pleasure-arousal-dominance).
* Plutchik (1980) basic-emotion families (labels only; no intensity wheel).
* Ortony, Clore & Collins (1988) OCC appraisal: events/agents/objects.
* Scherer (2001) component process: relevance, implication, coping, norm.
* Lazarus (1991) cognitive appraisal; Gross (1998) regulation strategies.

The emotion machine is explicit and slow-ish (0.5-2 Hz).  It is not trained on
VRChat (no emotion labels); its appraisal weights are hand priors and can be
fit later from real prosody/events.  It couples to the data-driven social FSM
by exposing continuous bias + a discrete label.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

# event -> (valence, arousal, dominance, comfort)
APPRAISAL = {
    "neutral": (0.0, 0.0, 0.0, 0.0),
    "greet": (0.10, 0.05, 0.02, 0.03),
    "compliment": (0.12, 0.04, 0.03, 0.02),
    "self_disclose": (0.10, 0.03, 0.02, 0.04),
    "invite": (0.10, 0.08, 0.03, 0.02),
    "accept_invite": (0.14, 0.08, 0.05, 0.06),
    "consent_ask": (0.02, 0.08, -0.02, 0.0),
    "consent_granted": (0.16, 0.06, 0.06, 0.12),
    "intimacy": (0.18, 0.12, 0.04, 0.10),
    "rejection": (-0.24, 0.16, -0.14, -0.14),
    "boundary": (-0.20, 0.12, -0.10, -0.20),
    "withdraw": (-0.08, 0.04, -0.05, -0.08),
    "crowd": (-0.06, 0.14, -0.05, -0.08),
    "repair": (0.15, -0.08, 0.03, 0.12),
    "ignore": (-0.10, 0.02, -0.04, -0.05),
    "threat": (-0.30, 0.26, -0.20, -0.22),
    "success": (0.15, 0.08, 0.12, 0.05),
    "conflict": (-0.18, 0.18, 0.05, -0.12),
}

EMOTION_TEXT = {
    "calm": "平静：没有明显趋近或回避，能继续观察和对话。",
    "content": "满足：轻微正效价、低唤醒，愿意维持当前互动。",
    "joyful": "愉快：正效价、较高唤醒，愿意分享和接近。",
    "playful": "玩闹：高兴奋、低威胁，适合玩笑和轻度暧昧。",
    "affectionate": "亲近：高舒适、正效价，愿意靠近和回应亲密。",
    "anxious": "焦虑：负效价、较高唤醒、低掌控，倾向确认安全。",
    "fearful": "害怕：威胁评估高、掌控低，倾向后退或冻结。",
    "angry": "生气：负效价、高唤醒、较高掌控，倾向设边界或修复。",
    "sad": "低落：负效价、低唤醒，倾向安静或寻求支持。",
    "embarrassed": "尴尬：自我归因的规范违背，倾向道歉/后退。",
    "overwhelmed": "过载：高唤醒、低舒适，需要停下和降刺激。",
    "shutdown": "关闭：威胁+低掌控的冻结，暂不社交。",
    "recovering": "恢复：情绪强度下降，重新观察环境。",
}

REGULATION = {
    "calm": "none", "content": "none", "joyful": "share", "playful": "share",
    "affectionate": "approach", "anxious": "reappraise", "fearful": "withdraw",
    "angry": "boundary", "sad": "seek_support", "embarrassed": "repair",
    "overwhelmed": "withdraw", "shutdown": "freeze", "recovering": "reappraise",
}


@dataclass
class EmotionState:
    label: str = "calm"
    valence: float = 0.10
    arousal: float = 0.30
    dominance: float = 0.00
    comfort: float = 0.60
    intensity: float = 0.20
    regulation: str = "none"
    text: str = ""
    action_tendency: str = "maintain"
    social_bias: dict = field(default_factory=dict)

    def as_observation(self) -> dict:
        return {"emotion_label": self.label, "emotion_valence": self.valence,
                "emotion_arousal": self.arousal, "emotion_dominance": self.dominance,
                "emotion_comfort": self.comfort, "emotion_intensity": self.intensity}


class EmotionFSM:
    """Appraisal-driven, dimensional state with discrete labels."""

    def __init__(self, baseline=(0.10, 0.30, 0.00, 0.60), decay: float = 0.86):
        self.bv, self.ba, self.bd, self.bc = baseline
        self.decay = float(decay)
        self.state = EmotionState()
        self._relabel()

    def _relabel(self) -> None:
        v, a, d, c = (self.state.valence, self.state.arousal,
                      self.state.dominance, self.state.comfort)
        if a > 0.82 and c < 0.32 and d < -0.15:
            lab = "shutdown"
        elif a > 0.72 and c < 0.45:
            lab = "overwhelmed"
        elif v < -0.35 and d < -0.15 and a > 0.45:
            lab = "fearful"
        elif v < -0.30 and d > 0.18:
            lab = "angry"
        elif a > 0.52 and c < 0.55 and v < 0.08:
            lab = "anxious"
        elif v < -0.18 and a < 0.42:
            lab = "sad"
        elif v < -0.10 and abs(d) < 0.2 and a >= 0.35:
            lab = "embarrassed"
        elif c > 0.72 and v > 0.25 and a > 0.25 and a < 0.7:
            lab = "affectionate"
        elif v > 0.30 and a > 0.55:
            lab = "playful"
        elif v > 0.28 and a >= 0.35:
            lab = "joyful"
        elif v > 0.18 and a < 0.38:
            lab = "content"
        elif abs(v) < 0.15 and abs(a - self.ba) < 0.15:
            lab = "calm"
        else:
            lab = "recovering" if self.state.intensity > 0.25 else "calm"
        self.state.label = lab
        self.state.text = EMOTION_TEXT.get(lab, "")
        self.state.regulation = REGULATION.get(lab, "none")
        self.state.action_tendency = {
            "calm": "maintain", "content": "maintain", "joyful": "approach",
            "playful": "approach", "affectionate": "approach", "anxious": "check",
            "fearful": "retreat", "angry": "boundary", "sad": "seek_support",
            "embarrassed": "repair", "overwhelmed": "withdraw",
            "shutdown": "freeze", "recovering": "observe",
        }.get(lab, "maintain")
        self.state.social_bias = {
            "approach": float(np.clip(v + c - 0.7 * max(0, -d), -1, 1)),
            "initiative": float(np.clip(v + c - 0.8 * (a if v < 0 else 0), -1, 1)),
            "distance": float(np.clip(-c + max(0, -v) + 0.4 * a, -1, 1)),
            "repair": float(np.clip(-v if d > 0 else 0, 0, 1)),
            "freeze": 1.0 if lab in ("overwhelmed", "shutdown") else 0.0,
        }

    def step(self, event: str = "neutral", intensity: float = 1.0):
        dv, da, dd, dc = APPRAISAL.get(event, APPRAISAL["neutral"])
        k = float(np.clip(intensity, 0.0, 2.0))
        s = self.state
        s.valence = float(np.clip(s.valence + dv * k, -1, 1))
        s.arousal = float(np.clip(s.arousal + da * k, 0, 1))
        s.dominance = float(np.clip(s.dominance + dd * k, -1, 1))
        s.comfort = float(np.clip(s.comfort + dc * k, 0, 1))
        s.intensity = float(np.clip(s.intensity + 0.35 * abs(dv + da + dc) * k, 0, 1))
        # homeostatic decay toward baseline
        s.valence = s.valence * self.decay + self.bv * (1 - self.decay)
        s.arousal = s.arousal * self.decay + self.ba * (1 - self.decay)
        s.dominance = s.dominance * self.decay + self.bd * (1 - self.decay)
        s.comfort = s.comfort * self.decay + self.bc * (1 - self.decay)
        s.intensity *= self.decay
        self._relabel()
        return self.state

    def reset(self) -> None:
        self.state = EmotionState()
        self._relabel()
