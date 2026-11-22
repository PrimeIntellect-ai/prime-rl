"""MiniGrid GoToDoor and GoToObject environments.

GoToDoor (F5 — constraint clue): drop the goal-star shortcut. Termination
fires when the agent stands directly in front of the door whose colour
matches the per-episode target colour named in the system prompt. The
agent MUST ground the colour token to win.

GoToObject (A8 — periodic refresh): drop the goal star. The target
description (e.g. ``ball (red)``) blinks visible in the HUD only on every
3rd step (turn % 3 == 0). The agent must time its lookups. Termination
fires on PICKUP — correct picks reward speed, wrong picks end with 0.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation
from glyphbench.envs.minigrid.base import DIR_RIGHT, DIR_TO_CHAR, MiniGridBase
from glyphbench.envs.minigrid.objects import Ball, Door, Key

_COLORS = ["red", "green", "blue", "yellow", "purple"]


class _GoToDoorBase(MiniGridBase):
    _room_size: int = 5  # interior size

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._target_color: str = "red"
        self._target_door_pos: tuple[int, int] = (0, 0)

    def _generate_grid(self, seed: int) -> None:
        size = self._room_size + 2
        self._init_grid(size, size)

        # Place 4 colored doors on the walls (one per wall)
        colors_used = list(_COLORS[:4])
        self.rng.shuffle(colors_used)

        door_positions = [
            (size // 2, 0),            # top wall
            (size // 2, size - 1),     # bottom wall
            (0, size // 2),            # left wall
            (size - 1, size // 2),     # right wall
        ]

        doors = []
        for i, (dx, dy) in enumerate(door_positions):
            color = colors_used[i]
            door = Door(color=color)
            self._grid[dy][dx] = door  # overwrite wall with door
            doors.append((dx, dy, color))

        # Pick target door — colour-only signal. NO Goal star is placed; the
        # win condition is "facing the door of matching colour".
        target_idx = int(self.rng.integers(0, len(doors)))
        self._target_color = doors[target_idx][2]
        self._target_door_pos = (doors[target_idx][0], doors[target_idx][1])

        # Agent at random interior position.
        while True:
            ax = int(self.rng.integers(1, size - 1))
            ay = int(self.rng.integers(1, size - 1))
            if self._get_obj(ax, ay) is None:
                break
        self._place_agent(ax, ay, DIR_RIGHT)

    # -- termination override -------------------------------------------

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Win when the cell directly in front of the agent is the target door.
        fx, fy = self._front_pos()
        if (fx, fy) == self._target_door_pos:
            front_obj = self._get_obj(fx, fy)
            if (
                isinstance(front_obj, Door)
                and front_obj.color == self._target_color
            ):
                reward = self._reward_on_goal()
                terminated = True
                info["door_reached"] = True
                info["goal_reached"] = True
        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        return (
            f"A room with four colored doors, one on each wall. Walk to the "
            f"door whose colour matches the target — the {self._target_color} "
            f"door — until you are standing directly in front of it (no need "
            f"to TOGGLE). Reward = 1 - 0.9 * (steps / max_steps)."
        )


class MiniGridGoToDoor5x5Env(_GoToDoorBase):
    _room_size = 5

    def env_id(self) -> str:
        return "glyphbench/minigrid-gotodoor-5x5-v0"


class MiniGridGoToDoor6x6Env(_GoToDoorBase):
    _room_size = 6

    def env_id(self) -> str:
        return "glyphbench/minigrid-gotodoor-6x6-v0"


class MiniGridGoToDoor8x8Env(_GoToDoorBase):
    _room_size = 8

    def env_id(self) -> str:
        return "glyphbench/minigrid-gotodoor-8x8-v0"


class MiniGridGoToObject6x6N2Env(MiniGridBase):
    """Room with 2 same-type, different-colour pickup objects — A8 (periodic refresh).

    Both placed objects share a single type (either two balls or two keys)
    with two distinct colours, so the per-episode target can only be
    identified by colour — the agent must ground the colour token from
    the periodic-refresh HUD.

    The target description blinks visible in the HUD on every 3rd step
    (visible at turn % 3 == 0; hidden otherwise). Termination on first
    PICKUP regardless of correctness; correct = +reward, wrong = 0.
    """

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._target_desc: str = ""
        self._target_obj_pos: tuple[int, int] = (0, 0)
        self._target_obj: Any = None

    def env_id(self) -> str:
        return "glyphbench/minigrid-gotoobject-6x6-n2-v0"

    def _generate_grid(self, seed: int) -> None:
        self._init_grid(8, 8)

        # Both placed objects share the SAME type, with DIFFERENT colours,
        # so type alone cannot disambiguate — the agent must ground the
        # per-episode target colour token. Pick one type per episode.
        obj_class = [Ball, Key][int(self.rng.integers(0, 2))]
        # Sample two distinct colours from the palette.
        palette = list(_COLORS)
        self.rng.shuffle(palette)
        colors = palette[:2]

        objects: list[tuple[int, int, Any]] = []
        occupied: set[tuple[int, int]] = set()
        for i in range(2):
            while True:
                ox = int(self.rng.integers(1, 7))
                oy = int(self.rng.integers(1, 7))
                if (ox, oy) not in occupied:
                    break
            occupied.add((ox, oy))
            obj = obj_class(color=colors[i])
            self._place_obj(ox, oy, obj)
            objects.append((ox, oy, obj))

        target_idx = int(self.rng.integers(0, 2))
        tx, ty, target_obj = objects[target_idx]
        self._target_desc = f"{target_obj.obj_type} ({target_obj.color})"
        self._target_obj_pos = (tx, ty)
        self._target_obj = target_obj

        # Agent at random interior position, not on any placed object.
        while True:
            ax = int(self.rng.integers(1, 7))
            ay = int(self.rng.integers(1, 7))
            if (ax, ay) not in occupied and self._get_obj(ax, ay) is None:
                break
        self._place_agent(ax, ay, DIR_RIGHT)

    # -- step: terminate on first PICKUP ---------------------------------

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        # Capture front-cell identity BEFORE the parent step (which may pick up).
        choice: str | None = None
        if name == "PICKUP":
            fx, fy = self._front_pos()
            if (fx, fy) == self._target_obj_pos:
                choice = "correct"
            else:
                front_obj = self._get_obj(fx, fy)
                if front_obj is not None and getattr(front_obj, "can_pickup", False):
                    choice = "wrong"

        obs, reward, terminated, truncated, info = super()._step(action)
        if choice == "correct":
            reward = self._reward_on_goal()
            terminated = True
            info["target_fetched"] = True
            info["goal_reached"] = True
        elif choice == "wrong":
            reward = 0.0
            terminated = True
            info["target_fetched"] = False
            info["wrong_pickup"] = True
        return obs, reward, terminated, truncated, info

    # -- HUD: hide target on most steps, refresh every 3rd step ---------

    def _render_current_observation(self) -> GridObservation:
        # Build the grid + base HUD via the same machinery as the parent,
        # then conditionally tack on the target description.
        render_grid = make_empty_grid(self._grid_w, self._grid_h)
        symbol_meanings: dict[str, str] = {"·": "floor"}
        for y in range(self._grid_h):
            for x in range(self._grid_w):
                obj = self._grid[y][x]
                if obj is not None:
                    ch = obj.render_char()
                    render_grid[y][x] = ch
                    if ch not in symbol_meanings:
                        symbol_meanings[ch] = obj.legend_name()

        agent_char = DIR_TO_CHAR[self._agent_dir]
        ax, ay = self._agent_pos
        render_grid[ay][ax] = agent_char
        _facing_name = {
            "→": "right", "↓": "down",
            "←": "left", "↑": "up",
        }[agent_char]
        symbol_meanings[agent_char] = f"you, facing {_facing_name}"
        legend = build_legend(symbol_meanings)

        carrying_str = (
            f"Carrying: {self._carrying.legend_name()}"
            if self._carrying is not None
            else "Carrying: nothing"
        )
        # Show the target every 3rd step; hide it otherwise. The agent must
        # plan around the periodic refresh rhythm.
        if self._turn % 3 == 0:
            target_block = f"    Target: {self._target_desc}"
        else:
            target_block = f"    Target: (refresh in {3 - (self._turn % 3)})"
        hud = (
            f"Step: {self._turn} / {self.max_turns}    {carrying_str}"
            f"{target_block}"
        )
        return GridObservation(
            grid=grid_to_string(render_grid),
            legend=legend,
            hud=hud,
            message="",
        )

    def _task_description(self) -> str:
        return (
            "A room with two pickup-able objects of the SAME type but "
            "different colours (so colour is the only thing that "
            "distinguishes them). The HUD shows the target description "
            "(e.g. \"ball (red)\") every 3rd step on a fixed cycle: visible "
            "at step 0, 3, 6, ...; hidden in between. PICKUP the correct "
            "object to win — the episode ends on the FIRST PICKUP "
            "regardless of correctness. Correct pickup rewards speed; "
            "wrong pickup ends with 0 reward."
        )
