# -*- coding: utf-8 -*-
"""Non-social runner: the explorer.

A waypoint navigator for agents that are NOT driven by the social FSM.
It wanders the walkable area of the scene, steers smoothly (turn-rate
limited), and avoids:

* walls     -- waypoints and steering stay inside the walkable rect with a
               safety margin; a bounce rule picks a new waypoint when the
               agent is pushed near an edge;
* obstacles -- a small circle list (scene props) with a clearance radius;
* bodies    -- soft repulsion from every other agent so a wanderer never
               walks through a conversation.

The explorer still *moves* through the same ARDY batch pipeline (its text
is walk/stand chosen by this controller), it just never socialises.
"""
from __future__ import annotations

import math
import random

ARRIVE_R = 0.55       # waypoint reached
WAYPOINT_MARGIN = 0.8  # keep waypoints this far inside the walkable rect
BODY_CLEAR = 0.75     # soft repulsion radius around other agents
PROP_CLEAR = 0.9      # clearance around props
TURN_COS = 0.35       # only walk when roughly facing the target


class Explorer:
    def __init__(self, bounds: tuple[float, float, float, float] = (-7.0, 7.0, -5.0, 5.0),
                 props: list[tuple[float, float, float]] | None = None,
                 rng: random.Random | None = None):
        """bounds = (x_min, x_max, z_min, z_max); props = [(x, z, r), ...]"""
        self.bounds = bounds
        self.props = props or []
        self.rng = rng or random.Random(11)
        self.target = self._pick()
        self.paused_until = 0.0

    def _inside(self, margin: float = WAYPOINT_MARGIN) -> tuple[float, float]:
        x0, x1, z0, z1 = self.bounds
        return (self.rng.uniform(x0 + margin, x1 - margin),
                self.rng.uniform(z0 + margin, z1 - margin))

    def _pick(self) -> tuple[float, float]:
        for _ in range(24):
            x, z = self._inside()
            if all(math.hypot(x - px, z - pz) > pr + PROP_CLEAR
                   for px, pz, pr in self.props):
                return (x, z)
        return self._inside(1.4)

    def _clear_of_props(self, x: float, z: float) -> bool:
        return all(math.hypot(x - px, z - pz) > pr + 0.35
                   for px, pz, pr in self.props)

    def update(self, agent, t_sim: float, near_pos) -> dict:
        """One tick: steer + speed decision.  Returns the intent for ARDY.

        `agent` is an engine.actors.Agent (uses pos/face); `near_pos` is the
        (x, z) of the nearest other agent or None (soft repulsion).
        """
        px, pz = float(agent.pos[0]), float(agent.pos[2])
        x0, x1, z0, z1 = self.bounds

        # wall bounce: too close to the rect edge -> new waypoint inward
        m = 0.45
        if not (x0 + m < px < x1 - m and z0 + m < pz < z1 - m):
            self.target = self._pick()

        # body repulsion: if someone is inside the clear radius, sidestep
        if near_pos is not None:
            d_near = math.hypot(px - near_pos[0], pz - near_pos[1])
            if d_near < BODY_CLEAR:
                away = math.atan2(px - near_pos[0], pz - near_pos[1])
                self.target = (
                    max(x0 + WAYPOINT_MARGIN,
                        min(x1 - WAYPOINT_MARGIN,
                            px + 2.2 * math.sin(away))),
                    max(z0 + WAYPOINT_MARGIN,
                        min(z1 - WAYPOINT_MARGIN,
                            pz + 2.2 * math.cos(away))))

        # obstacle: current target swallowed by a prop -> repick
        if not self._clear_of_props(*self.target):
            self.target = self._pick()

        dx = self.target[0] - px
        dz = self.target[1] - pz
        dist = math.hypot(dx, dz)
        if dist < ARRIVE_R:
            self.target = self._pick()
            return {"intent": "stand", "speed": 0.0, "face": agent.face}

        # pause briefly at waypoints sometimes (look around)
        if t_sim < self.paused_until:
            return {"intent": "stand", "speed": 0.0, "face": agent.face}
        if self.rng.random() < 0.015:
            self.paused_until = t_sim + self.rng.uniform(1.2, 3.0)
            return {"intent": "stand", "speed": 0.0, "face": agent.face}

        target_yaw = math.atan2(dx, dz)
        agent.steer(target_yaw)
        facing = math.cos(agent.face - target_yaw)
        if facing < TURN_COS:
            return {"intent": "stand", "speed": 0.0, "face": agent.face}
        # walk speed: the raw walk cadence (~0.55 m/s), capped by the
        # controller as usual; the caller runs the same controller math.
        return {"intent": "walk", "speed": None, "face": agent.face}
