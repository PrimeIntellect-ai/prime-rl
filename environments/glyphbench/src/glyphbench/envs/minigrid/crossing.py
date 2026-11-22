"""MiniGrid Crossing and SimpleCrossing environments.

Crossing: horizontal lava/water strips with gaps.
SimpleCrossing: horizontal wall strips with gaps.
All variants sample each strip gap away from the start column so a
FORWARD-only policy cannot win. SimpleCrossing-Easy uses fewer strips than
the matching non-easy variant and adds D1 wind drift: each step in which the
agent actually moves has a 25% chance of an additional 1-tile cardinal drift.
A stationary or wall-blocked agent does not drift.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minigrid.base import (
    DIR_TO_VEC,
    DIR_UP,
    MiniGridBase,
)
from glyphbench.envs.minigrid.objects import Goal, Lava, Wall, Water


class _CrossingBase(MiniGridBase):
    _num_strips: int = 1
    _obstacle_type: str = "lava"  # "lava", "water", or "wall"
    _drift_prob: float = 0.0  # per-step lateral drift probability

    def _generate_grid(self, seed: int) -> None:
        # Scale grid so strips never overlap the goal row (y=1)
        size = max(9, 2 * self._num_strips + 5)
        self._init_grid(size, size)

        # Agent start column — must NOT be a candidate for gap_x, otherwise
        # FORWARD-spam from the centre wins for free on easy mode.
        start_col = size // 2

        # Place obstacle strips at evenly spaced y positions
        strip_spacing = (size - 2) // (self._num_strips + 1)
        for i in range(self._num_strips):
            strip_y = strip_spacing * (i + 1)
            for x in range(1, size - 1):
                if self._obstacle_type == "lava":
                    self._place_obj(x, strip_y, Lava())
                elif self._obstacle_type == "water":
                    self._place_obj(x, strip_y, Water())
                else:
                    self._place_obj(x, strip_y, Wall())

            # gap_x sampled uniformly from interior columns, excluding the
            # agent's start column so FORWARD-spam can't trivially win.
            while True:
                gap_x = int(self.rng.integers(1, size - 1))
                if gap_x != start_col:
                    break
            self._grid[strip_y][gap_x] = None

        # Agent at bottom center
        self._place_agent(start_col, size - 2, DIR_UP)

        # Goal at top center
        self._place_obj(start_col, 1, Goal())

    # -- step: optional wind drift (D1) ---------------------------------

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        pre_pos = self._agent_pos
        obs, reward, terminated, truncated, info = super()._step(action)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info
        # Wind drift represents air-current during motion; it only fires
        # when the agent actually moved this step. A blocked / stationary
        # agent does not drift — this prevents FORWARD-spam from trivially
        # solving easy mode by random-walking through drifts while parked
        # against a wall.
        moved_this_step = self._agent_pos != pre_pos
        if (
            moved_this_step
            and self._drift_prob > 0.0
            and float(self.rng.random()) < self._drift_prob
        ):
            # Random cardinal drift; cancel if blocked.
            dirn = int(self.rng.integers(0, 4))
            ddx, ddy = DIR_TO_VEC[dirn]
            ax, ay = self._agent_pos
            nx, ny = ax + ddx, ay + ddy
            if 0 <= nx < self._grid_w and 0 <= ny < self._grid_h:
                cell = self._get_obj(nx, ny)
                if cell is None or cell.can_overlap:
                    # Walking into lava via drift terminates with 0 reward —
                    # consistent with the parent's lava handling.
                    self._agent_pos = (nx, ny)
                    info["agent_pos"] = self._agent_pos
                    if isinstance(cell, Lava):
                        info["lava"] = True
                        info["drifted"] = True
                        return (
                            self._render_current_observation(),
                            0.0,
                            True,
                            False,
                            info,
                        )
                    if (
                        self._goal_pos is not None
                        and self._agent_pos == self._goal_pos
                    ):
                        info["goal_reached"] = True
                        info["drifted"] = True
                        return (
                            self._render_current_observation(),
                            self._reward_on_goal(),
                            True,
                            False,
                            info,
                        )
                    info["drifted"] = True
                    return self._render_current_observation(), reward, False, False, info
        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        obs_name = {
            "lava": f"lava ({Lava().render_char()})",
            "water": f"water ({Water().render_char()})",
            "wall": f"walls ({Wall().render_char()})",
        }.get(self._obstacle_type, self._obstacle_type)
        danger = ""
        if self._obstacle_type == "lava":
            danger = " Stepping on lava ends the episode with zero reward."
        elif self._obstacle_type == "water":
            danger = " Water blocks movement; use the gaps to cross."
        goal = Goal().render_char()
        drift_block = ""
        if self._drift_prob > 0.0:
            pct = int(round(self._drift_prob * 100))
            drift_block = (
                f" Wind drift: when you actually move this turn, there is "
                f"a {pct}% chance of an additional 1-tile drift in a random "
                "cardinal direction (drifts that hit a wall or solid object "
                "are cancelled; standing still or being blocked never "
                "drifts). Plan with slack."
            )
        return (
            f"Navigate through {self._num_strips} horizontal strip(s) of "
            f"{obs_name} to reach the goal ({goal}). Each strip has one gap "
            f"you can pass through.{danger}{drift_block} "
            "PICKUP, DROP, TOGGLE, and DONE have no useful effect in this "
            "strip-crossing task. Reward = 1 - 0.9 * (steps / max_steps) "
            "on reaching the goal; timeout or other failure gives 0 reward."
        )


# Lava Crossing variants
class MiniGridCrossingN1Env(_CrossingBase):
    _num_strips = 1
    _obstacle_type = "lava"

    def env_id(self) -> str:
        return "glyphbench/minigrid-crossing-n1-v0"


class MiniGridCrossingN2Env(_CrossingBase):
    _num_strips = 2
    _obstacle_type = "lava"

    def env_id(self) -> str:
        return "glyphbench/minigrid-crossing-n2-v0"


class MiniGridCrossingN3Env(_CrossingBase):
    _num_strips = 3
    _obstacle_type = "lava"

    def env_id(self) -> str:
        return "glyphbench/minigrid-crossing-n3-v0"


# Safe (water) Crossing variants
class MiniGridCrossingN1SafeEnv(_CrossingBase):
    _num_strips = 1
    _obstacle_type = "water"

    def env_id(self) -> str:
        return "glyphbench/minigrid-crossing-n1-safe-v0"


class MiniGridCrossingN2SafeEnv(_CrossingBase):
    _num_strips = 2
    _obstacle_type = "water"

    def env_id(self) -> str:
        return "glyphbench/minigrid-crossing-n2-safe-v0"


class MiniGridCrossingN3SafeEnv(_CrossingBase):
    _num_strips = 3
    _obstacle_type = "water"

    def env_id(self) -> str:
        return "glyphbench/minigrid-crossing-n3-safe-v0"


# SimpleCrossing (wall) variants
class MiniGridSimpleCrossingN1Env(_CrossingBase):
    _num_strips = 1
    _obstacle_type = "wall"

    def env_id(self) -> str:
        return "glyphbench/minigrid-simplecrossing-n1-v0"


class MiniGridSimpleCrossingN2Env(_CrossingBase):
    _num_strips = 2
    _obstacle_type = "wall"

    def env_id(self) -> str:
        return "glyphbench/minigrid-simplecrossing-n2-v0"


class MiniGridSimpleCrossingN3Env(_CrossingBase):
    _num_strips = 3
    _obstacle_type = "wall"

    def env_id(self) -> str:
        return "glyphbench/minigrid-simplecrossing-n3-v0"


# SimpleCrossing Easy variants — random gaps + D1 wind drift.
class MiniGridSimpleCrossingEasyN1Env(_CrossingBase):
    _num_strips = 1
    _obstacle_type = "wall"
    _drift_prob = 0.25

    def env_id(self) -> str:
        return "glyphbench/minigrid-simplecrossing-easy-n1-v0"


class MiniGridSimpleCrossingEasyN2Env(_CrossingBase):
    _num_strips = 2
    _obstacle_type = "wall"
    _drift_prob = 0.25

    def env_id(self) -> str:
        return "glyphbench/minigrid-simplecrossing-easy-n2-v0"


class MiniGridSimpleCrossingEasyN3Env(_CrossingBase):
    _num_strips = 3
    _obstacle_type = "wall"
    _drift_prob = 0.25

    def env_id(self) -> str:
        return "glyphbench/minigrid-simplecrossing-easy-n3-v0"
