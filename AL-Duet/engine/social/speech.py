# -*- coding: utf-8 -*-
"""Speech lines and the conversation-floor policy (one voice at a time)."""
from __future__ import annotations

SPEECH_BY_STATE = {
    "greet_initiate": "你好。",
    "greet_ack": "你好。",
    "converse_speak": "今天这里人不多，你也是来看樱花的吗？",
    "backchannel": "嗯，我在听。",
    "disengage": "那我先去那边看看。",
    "part_farewell": "回头见。",
}
LINES = {
    "greet_initiate": {"A": "你好。", "B": "你好，很高兴见到你。"},
    "greet_ack": {"A": "你好呀。", "B": "你好，很高兴见到你。"},
    "backchannel": {"A": "嗯，我在听。", "B": "原来如此。"},
    "disengage": {"A": "那我先去那边看看。", "B": "好，改天再聊。"},
    "part_farewell": {"A": "回头见。", "B": "拜拜。"},
    "converse_speak": {
        "A": ["今天这里人不多，你也是来看樱花的吗？",
              "我偶尔来这条街，樱花开的这几天最好看。"],
        "B": ["算是吧，刚好路过。你常来这条街吗？",
              "确实，风一吹就像下雪。"],
    },
}
WAVE_STATES = {"greet_initiate", "greet_ack", "part_farewell"}
DYAD_STATES = {"approach_person", "greet_initiate", "greet_wait_reply",
               "greet_ack", "converse_listen", "converse_speak",
               "backchannel", "hold_distance", "disengage"}
SPEAK_S = 1.8          # one spoken turn on the sim clock
ADDRESSED_S = 2.5      # being spoken to keeps you "addressed" this long


def speech_line(state: str, key: str, turns: int) -> str:
    table = LINES.get(state)
    if isinstance(table, dict):
        val = table.get(key, SPEECH_BY_STATE.get(state, "…"))
        if isinstance(val, list):
            return val[turns % len(val)]
        return val
    return SPEECH_BY_STATE.get(state, "…")
