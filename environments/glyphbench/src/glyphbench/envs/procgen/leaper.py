"""Procgen Leaper (Frogger) environment.

Cross roads (avoid cars) and rivers (ride logs) to reach the goal at the top.

Gym ID: glyphbench/procgen-leaper-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase

_GRASS = ","
_ROAD = "\u00b7"
_WATER = "\u2248"
_LOG = "\u25ac"
_CAR = "V"
_GOAL = "G"


class LeaperEnv(ProcgenBase):
    """Procgen Leaper: Frogger-style lane crossing game."""

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
        ),
    )

    GRID_W = 14
    GRID_H = 14
    # Reward shaping (Pattern A): single terminal goal worth +1.0.
    _GOAL_REWARD = 1.0

    def env_id(self) -> str:
        return "glyphbench/procgen-leaper-v0"

    def _generate_level(self, seed: int) -> None:
        w, h = self.GRID_W, self.GRID_H
        self._init_world(w, h, fill=_GRASS)

        # Lane layout (bottom to top):
        # Row h-1: safe start (grass)
        # Rows h-2 to 1: alternating road/water lanes with safe rows between
        # Row 0: goal row

        self._lane_types: list[str] = ["safe"] * h  # per-row lane type
        self._lane_dirs: list[int] = [0] * h  # per-row scroll direction
        self._lane_speeds: list[int] = [0] * h  # per-row speed (steps per move)
        self._step_count = 0
        self._on_log = False
        self._log_dx = 0

        # Build lanes from bottom-1 upward
        # Pattern: safe, road, road, safe, water, water,
        # safe, road, road, safe, water, water, safe/goal
        lane_pattern = ["safe"]  # row h-1 (bottom, start)
        # Fill middle rows
        for i in range(1, h - 1):
            phase = (i - 1) % 6
            if phase < 2:
                lane_pattern.append("road")
            elif phase == 2:
                lane_pattern.append("safe")
            elif phase < 5:
                lane_pattern.append("water")
            else:
                lane_pattern.append("safe")
        lane_pattern.append("goal")  # row 0 (top)

        # Reverse so index 0 = top row
        lane_pattern.reverse()

        for row_y in range(h):
            lt = lane_pattern[row_y]
            self._lane_types[row_y] = lt

            if lt == "safe" or lt == "goal":
                fill = _GRASS if lt == "safe" else _GOAL
                for x in range(w):
                    self._set_cell(x, row_y, fill)
                self._lane_dirs[row_y] = 0
            elif lt == "road":
                direction = 1 if int(self.rng.integers(0, 2)) == 0 else -1
                self._lane_dirs[row_y] = direction
                self._lane_speeds[row_y] = int(self.rng.integers(1, 3))
                for x in range(w):
                    self._set_cell(x, row_y, _ROAD)
                # Place 2-3 cars
                num_cars = int(self.rng.integers(2, 4))
                spacing = w // (num_cars + 1)
                for ci in range(num_cars):
                    cx = (ci + 1) * spacing + int(self.rng.integers(-1, 2))
                    cx = max(0, min(w - 1, cx))
                    self._add_entity(
                        "car", _CAR, cx, row_y, dx=direction, data={"lane": row_y}
                    )
            elif lt == "water":
                direction = 1 if int(self.rng.integers(0, 2)) == 0 else -1
                self._lane_dirs[row_y] = direction
                self._lane_speeds[row_y] = int(self.rng.integers(1, 3))
                for x in range(w):
                    self._set_cell(x, row_y, _WATER)
                # Place 2-3 logs (3 cells wide each)
                num_logs = int(self.rng.integers(2, 4))
                spacing = w // (num_logs + 1)
                for li in range(num_logs):
                    lx = (li + 1) * spacing + int(self.rng.integers(-1, 2))
                    lx = max(0, min(w - 4, lx))
                    for dx in range(3):
                        if 0 <= lx + dx < w:
                            self._set_cell(lx + dx, row_y, _LOG)

        # Agent starts at bottom center
        self._agent_x = w // 2
        self._agent_y = h - 1

    def _advance_entities(self) -> float:
        """Move cars. Cars wrap around the grid instead of being removed."""
        self._step_count += 1
        for e in self._entities:
            if not e.alive:
                continue
            lane_y = e.data.get("lane", e.y)
            speed = self._lane_speeds[lane_y] if lane_y < len(self._lane_speeds) else 1
            if self._step_count % speed == 0:
                e.x += e.dx
                # Wrap around
                if e.x < 0:
                    e.x = self.GRID_W - 1
                elif e.x >= self.GRID_W:
                    e.x = 0

        # Scroll logs (modify world grid)
        for row_y in range(self.GRID_H):
            if self._lane_types[row_y] != "water":
                continue
            speed = self._lane_speeds[row_y]
            if self._step_count % speed != 0:
                continue
            direction = self._lane_dirs[row_y]
            old_row = [self._world_at(x, row_y) for x in range(self.GRID_W)]
            for x in range(self.GRID_W):
                src = (x - direction) % self.GRID_W
                self._set_cell(x, row_y, old_row[src])

        # Track if agent is on a log and carry them
        self._on_log = False
        self._log_dx = 0
        if 0 <= self._agent_y < self.GRID_H and self._lane_types[self._agent_y] == "water":
            speed = self._lane_speeds[self._agent_y]
            if self._step_count % speed == 0:
                direction = self._lane_dirs[self._agent_y]
                cell_at_agent = self._world_at(self._agent_x, self._agent_y)
                if cell_at_agent == _LOG:
                    self._on_log = True
                    self._log_dx = direction
                    # Carry agent with log
                    new_x = self._agent_x + direction
                    if 0 <= new_x < self.GRID_W:
                        self._agent_x = new_x
                    else:
                        # Carried off edge -> death
                        self._on_log = False

        # Bug 7 fix: check water death after log scroll carries (or fails
        # to carry) the agent.  Without this, an agent that ends up on
        # water after logs move survives for one extra turn.
        if (
            0 <= self._agent_y < self.GRID_H
            and self._lane_types[self._agent_y] == "water"
        ):
            cell = self._world_at(self._agent_x, self._agent_y)
            if cell != _LOG:
                self._message = "Fell in the water!"
                self._entity_terminated = True
            else:
                self._on_log = True
                self._log_dx = self._lane_dirs[self._agent_y]
        else:
            self._on_log = False
            self._log_dx = 0
        return 0.0

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        # Movement (don't use _try_move; water/road aren't solid)
        dx, dy = 0, 0
        if action_name == "LEFT":
            dx = -1
        elif action_name == "RIGHT":
            dx = 1
        elif action_name == "UP":
            dy = -1
        elif action_name == "DOWN":
            dy = 1

        if dx != 0 or dy != 0:
            self._agent_dir = (dx, dy)

        nx = self._agent_x + dx
        ny = self._agent_y + dy
        if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H:
            self._agent_x = nx
            self._agent_y = ny

        # Check goal (top row)
        if self._lane_types[self._agent_y] == "goal":
            reward = self._GOAL_REWARD
            terminated = True
            self._message = "You reached the goal!"
            return reward, terminated, info

        # Check car collision
        for e in self._entities:
            if e.alive and e.etype == "car" and e.x == self._agent_x and e.y == self._agent_y:
                terminated = True
                self._message = "Hit by a car!"
                return reward, terminated, info

        # Check water death (not on log)
        if (
            0 <= self._agent_y < self.GRID_H
            and self._lane_types[self._agent_y] == "water"
        ):
            cell = self._world_at(self._agent_x, self._agent_y)
            if cell != _LOG:
                terminated = True
                self._message = "Fell in the water!"

        return reward, terminated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        # Build compact lane info (bottom to top)
        parts: list[str] = []
        for row_y in range(self.GRID_H - 1, -1, -1):
            lt = self._lane_types[row_y]
            d = self._lane_dirs[row_y]
            if lt == "safe":
                parts.append("safe")
            elif lt == "goal":
                parts.append("GOAL")
            elif lt == "road":
                arr = "->" if d == 1 else "<-"
                parts.append(f"road{arr}")
            elif lt == "water":
                arr = "->" if d == 1 else "<-"
                parts.append(f"water{arr}")
        lane_str = "Lanes(bot->top): " + " | ".join(parts)
        on_log = "yes" if self._on_log else "no"
        extra = f"OnLog: {on_log}\n{lane_str}"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Cross roads (avoid cars V) and rivers (ride logs \u25ac) to reach "
            "the goal row (G) for +1 reward. Water (\u2248) is deadly "
            "unless you stand on a log."
        )

    def _symbol_meaning(self, ch: str) -> str:
        meanings = {
            ",": "grass (safe)",
            "\u00b7": "road",
            "\u2248": "water (deadly)",
            "\u25ac": "log (safe on water)",
            "V": "car (deadly)",
            "G": "goal",
        }
        return meanings.get(ch, super()._symbol_meaning(ch))
