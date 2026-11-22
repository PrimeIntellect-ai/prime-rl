"""Atari PrivateEye environment.

Detective driving game. Navigate city, find clues, solve case.

Gym ID: glyphbench/atari-privateeye-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase

_W = 30
_H = 16
_ROAD_Y = 12
_DIRS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

class PrivateEyeEnv(AtariBase):
    """PrivateEye: detective driving game.

    Drive through city streets, enter buildings, find clues (?),
    catch criminals (V), and avoid thugs (T).
    Examine (E) locations to discover hidden clues.

    Grid: 30x16.
    Pattern D: +1/_WIN_TARGET per case solved (full-scope = 5
    cases). -1.0 on death (caught by thug).
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "LEFT", "RIGHT",
            "UP", "DOWN", "EXAMINE",
        ),
        descriptions=(
            "do nothing",
            "drive/walk left",
            "drive/walk right",
            "enter building / move up",
            "exit building / move down",
            "examine current location for clues",
        ),
    )

    # Pattern D full-scope target: 5 cases solved.
    _WIN_TARGET: int = 5
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._clues_found: int = 0
        self._clues_total: int = 0
        self._in_building: bool = False
        self._building_idx: int = -1
        self._case_timer: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-privateeye-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        rng = np.random.default_rng(
            seed + self._level * 1327
        )
        self._init_grid(_W, _H)
        self._entities = []
        self._clues_found = 0
        self._clues_total = 0
        self._in_building = False
        self._building_idx = -1
        self._case_timer = 500 + self._level * 100

        # Sky
        for x in range(_W):
            self._set_cell(x, 0, "─")
            self._set_cell(x, _H - 1, "─")

        # Road
        for x in range(_W):
            self._set_cell(x, _ROAD_Y, "=")
            self._set_cell(x, _ROAD_Y + 1, "=")
        # Sidewalk
        for x in range(_W):
            self._set_cell(x, _ROAD_Y - 1, "·")

        # Buildings along the top
        bldg_positions: list[tuple[int, int]] = []
        for bx in range(2, _W - 5, 6):
            bw = 4
            bh = int(rng.integers(4, 7))
            by_top = _ROAD_Y - 1 - bh
            bldg_positions.append((bx, by_top))
            for yy in range(by_top, _ROAD_Y - 1):
                for xx in range(bx, bx + bw):
                    if xx < _W:
                        self._set_cell(xx, yy, "█")
            # Door
            door_x = bx + bw // 2
            if door_x < _W:
                self._set_cell(
                    door_x, _ROAD_Y - 2, "D",
                )
            # Roof
            for xx in range(bx, min(bx + bw, _W)):
                self._set_cell(xx, by_top, "↑")

        # Clues: some visible, some hidden in buildings
        clue_xs = list(
            rng.choice(
                range(3, _W - 3),
                size=min(5, _W // 6),
                replace=False,
            )
        )
        for cx in clue_xs:
            cy = _ROAD_Y - 1
            self._add_entity("clue", "?", int(cx), cy)
            self._clues_total += 1

        # Hidden clues near buildings
        n_hidden = 2 + self._level
        for i in range(min(n_hidden, len(bldg_positions))):
            bx, by_top = bldg_positions[
                i % len(bldg_positions)
            ]
            hx = bx + 2
            hy = by_top + 2
            if hy < _ROAD_Y - 2:
                e = self._add_entity(
                    "hidden_clue", " ", hx, hy,
                )
                e.data["revealed"] = False
                self._clues_total += 1

        # Thugs patrolling streets
        n_thugs = 2 + self._level
        for _ in range(n_thugs):
            tx = int(rng.integers(5, _W - 3))
            direction = 1 if rng.random() < 0.5 else -1
            e = self._add_entity(
                "thug", "T", tx, _ROAD_Y - 1, dx=direction,
            )
            e.data["timer"] = int(rng.integers(2, 5))

        # Criminal (target to catch)
        vx = int(rng.integers(_W // 2, _W - 3))
        v = self._add_entity(
            "criminal", "V", vx, _ROAD_Y - 1, dx=-1,
        )
        v.data["timer"] = int(rng.integers(3, 6))

        # Player starts on sidewalk
        self._player_x = 3
        self._player_y = _ROAD_Y - 1

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Case timer
        self._case_timer -= 1
        if self._case_timer <= 0:
            self._message = "Case went cold! Time up."
            return 0.0, True, info

        # Movement
        if action_name in _DIRS:
            d = _DIRS[action_name]
            self._player_dir = d
            nx = self._player_x + d[0]
            ny = self._player_y + d[1]
            # Enter building via door
            if action_name == "UP":
                cell = self._grid_at(
                    self._player_x, self._player_y - 1,
                )
                if cell == "D":
                    self._player_y -= 2
                elif not self._is_solid(nx, ny):
                    self._player_y = ny
            elif action_name == "DOWN":
                if not self._is_solid(nx, ny):
                    self._player_y = ny
                # Snap to sidewalk if going too low
                if self._player_y > _ROAD_Y - 1:
                    self._player_y = _ROAD_Y - 1
            else:
                if not self._is_solid(nx, ny):
                    self._player_x = nx
                    self._player_y = ny

        # Examine: reveal hidden clues nearby
        if action_name == "EXAMINE":
            for e in self._entities:
                if e.etype != "hidden_clue" or not e.alive:
                    continue
                if e.data.get("revealed"):
                    continue
                dist = (
                    abs(e.x - self._player_x)
                    + abs(e.y - self._player_y)
                )
                if dist <= 2:
                    e.data["revealed"] = True
                    e.char = "?"
                    self._message = "Clue discovered!"

        # Collect visible/revealed clues (no direct reward; only
        # solving the case yields progress)
        for e in self._entities:
            if not e.alive:
                continue
            if e.etype == "clue":
                if (
                    e.x == self._player_x
                    and e.y == self._player_y
                ):
                    e.alive = False
                    self._clues_found += 1
                    self._message = "Clue found!"
            elif e.etype == "hidden_clue" and (
                e.data.get("revealed")
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                e.alive = False
                self._clues_found += 1
                self._message = "Hidden clue!"

        # Thug AI
        for e in self._entities:
            if e.etype != "thug" or not e.alive:
                continue
            e.data["timer"] = (
                e.data.get("timer", 3) - 1
            )
            if e.data["timer"] > 0:
                continue
            e.data["timer"] = int(
                self.rng.integers(2, 4)
            )
            e.x += e.dx
            if e.x <= 1 or e.x >= _W - 2:
                e.dx = -e.dx

        # Criminal AI: flee from player
        for e in self._entities:
            if e.etype != "criminal" or not e.alive:
                continue
            e.data["timer"] = (
                e.data.get("timer", 4) - 1
            )
            if e.data["timer"] > 0:
                continue
            e.data["timer"] = int(
                self.rng.integers(2, 5)
            )
            if e.x > self._player_x:
                e.x += 1
            else:
                e.x -= 1
            e.x = max(1, min(e.x, _W - 2))

        # Thug collision (Pattern D death)
        for e in self._entities:
            if (
                e.etype == "thug"
                and e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                self._on_life_lost()
                reward = self._DEATH_PENALTY
                self._message = "Mugged by thug!"
                return reward, self._game_over, info

        # Criminal catch (no direct reward)
        for e in self._entities:
            if (
                e.etype == "criminal"
                and e.alive
                and e.x == self._player_x
                and abs(e.y - self._player_y) <= 1
            ):
                e.alive = False
                self._message = "Criminal caught!"

        self._entities = [
            e for e in self._entities if e.alive
        ]

        # Case solved: all clues found (Pattern D progress)
        if self._clues_found >= self._clues_total:
            if self._progress_count < self._WIN_TARGET:
                reward += 1.0 / self._WIN_TARGET
                self._progress_count += 1
            self._message = "Case solved!"
            self._level += 1
            if self._progress_count >= self._WIN_TARGET:
                self._game_over = True
                info["won"] = True
                self._message = "All cases solved!"
            else:
                self._generate_level(self._level * 5003)
            return reward, self._game_over, info

        info["clues"] = (
            f"{self._clues_found}/{self._clues_total}"
        )
        info["timer"] = self._case_timer
        return reward, self._game_over, info

    def _advance_entities(self) -> None:
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "border",
            "█": "building wall",
            "↑": "roof",
            "D": "door",
            "=": "road",
            "·": "sidewalk",
            "?": "clue",
            "T": "thug",
            "V": "criminal",
            " ": "empty",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        extra = (
            f"Timer: {self._case_timer}"
            f"  Clues: {self._clues_found}"
            f"/{self._clues_total}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Investigate as a private eye. Walk the streets, "
            "EXAMINE locations to find hidden clues (?). "
            "Catch the criminal (V). Avoid thugs (T). "
            "Find all clues to solve the case before time "
            "runs out."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Private Eye.\n\n"
            "TASK\n"
            "Play a detective walking a city street: find every "
            "clue (visible and hidden in buildings), catch the "
            "criminal, and solve the case before the timer "
            "(500 + 100*level) ticks runs out.\n\n"
            "BOARD\n"
            "30x16 street. Sky '-' tops the scene; the road '=' "
            "and sidewalk '.' run along the bottom. Buildings '#' "
            "with rooftops and doors 'D' line the upper part of "
            "the street. Visible clues '?' sit on the sidewalk. "
            "Hidden clues "
            "appear inside buildings and must be EXAMINEd. Thugs "
            "'T' patrol the sidewalk; the criminal 'V' flees "
            "sideways. You are an arrow glyph.\n\n"
            "MECHANICS\n"
            "UP / DOWN / LEFT / RIGHT move 1 cell; UP at a 'D' "
            "door teleports 2 rows up into the building. EXAMINE "
            "reveals any hidden clue within Manhattan distance 2, "
            "turning it into a visible '?'. Walking onto a visible "
            "clue collects it. Walking onto the criminal catches "
            "them. Thugs move every 2-4 steps horizontally, "
            "bouncing at walls; criminal flees from your column.\n\n"
            "SCORING\n"
            "+1/5 reward per case solved (Pattern D full-scope = 5 "
            "cases). Solving a case requires collecting every clue "
            "(visible + hidden in buildings) for that case. -1.0 "
            "on death (mugged by thug).\n\n"
            "TERMINATION\n"
            "Thug contact ends the episode with -1.0. Timer "
            "expiring ends the episode without penalty. Episode "
            "ends after 5 cases solved (cumulative reward plateaus "
            "at +1.0).\n\n"
            "HUD\n"
            "Shows score, level, timer, and clues found "
            "vs total.\n\n"
            + self.action_spec.render_for_prompt()
        )
