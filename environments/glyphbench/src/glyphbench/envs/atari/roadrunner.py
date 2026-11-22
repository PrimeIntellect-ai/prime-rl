"""Atari RoadRunner environment.

Chase platformer. Collect birdseed, avoid Coyote.

Gym ID: glyphbench/atari-roadrunner-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase

_W = 30
_H = 16
_GROUND_Y = 13
_ROAD_Y = 11

class RoadRunnerEnv(AtariBase):
    """RoadRunner: collect seeds, outrun the Coyote.

    The road scrolls left. Collect birdseed (o) and avoid
    the Coyote (W) who chases you. Jump over obstacles.
    Mines (*) can stun the Coyote if he hits them.

    Grid: 30x16.
    Pattern D: +1/_WIN_TARGET per progress unit (seed or
    coyote-stunned-by-mine). -1 on collision with rock or
    coyote.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "JUMP"),
        descriptions=(
            "do nothing",
            "run left",
            "run right",
            "jump over obstacle or gap",
        ),
    )

    # Pattern D full-scope: 50 progress events (seeds collected or
    # coyote-mine-stuns) along the scrolling road. Each yields
    # +1/50; collision with a rock or the coyote ends the episode
    # with -1.
    _WIN_TARGET: int = 50
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._scroll_timer: int = 0
        self._distance: int = 0
        self._jumping: bool = False
        self._jump_timer: int = 0
        self._coyote_stun: int = 0
        self._spawn_timer: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-roadrunner-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        rng = np.random.default_rng(
            seed + self._level * 1231
        )
        self._init_grid(_W, _H)
        self._entities = []
        self._scroll_timer = 0
        self._distance = 0
        self._jumping = False
        self._jump_timer = 0
        self._coyote_stun = 0
        self._spawn_timer = 0

        # Draw landscape
        for x in range(_W):
            self._set_cell(x, 0, "─")
            self._set_cell(x, _H - 1, "─")
            self._set_cell(x, _GROUND_Y, "=")
            # Desert floor below
            for y in range(_GROUND_Y + 1, _H - 1):
                self._set_cell(x, y, "·")

        # Road surface
        for x in range(_W):
            self._set_cell(x, _ROAD_Y, "_")
            self._set_cell(x, _ROAD_Y + 1, "_")

        # Initial seeds on road
        for _i in range(5):
            sx = int(rng.integers(10, _W - 2))
            self._add_entity("seed", "o", sx, _ROAD_Y)

        # Initial obstacles (rocks)
        for _ in range(2):
            ox = int(rng.integers(15, _W - 2))
            self._add_entity(
                "rock", "█", ox, _ROAD_Y, dx=-1,
            )

        # Mines
        for _ in range(2):
            mx = int(rng.integers(12, _W - 2))
            self._add_entity("mine", "*", mx, _ROAD_Y)

        # Coyote (chases from behind)
        self._add_entity(
            "coyote", "W", 1, _ROAD_Y, dx=0,
        )

        # Player (Road Runner) on the road
        self._player_x = 8
        self._player_y = _ROAD_Y

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Player movement
        if action_name == "LEFT":
            if self._player_x > 1:
                self._player_x -= 1
                self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            if self._player_x < _W - 2:
                self._player_x += 1
                self._player_dir = (1, 0)
        elif action_name == "JUMP" and not self._jumping:
            self._jumping = True
            self._jump_timer = 4
            self._player_y = _ROAD_Y - 2

        # Jump physics
        if self._jumping:
            self._jump_timer -= 1
            if self._jump_timer <= 0:
                self._jumping = False
                self._player_y = _ROAD_Y

        # Scroll world
        self._scroll_timer += 1
        if self._scroll_timer >= 2:
            self._scroll_timer = 0
            self._distance += 1
            for e in self._entities:
                if e.etype in ("seed", "rock", "mine") and e.alive:
                    e.x -= 1
                    if e.x < 1:
                        e.alive = False

        # Coyote AI
        coyote = None
        for e in self._entities:
            if e.etype == "coyote" and e.alive:
                coyote = e
                break

        if coyote and coyote.alive:
            if self._coyote_stun > 0:
                self._coyote_stun -= 1
            else:
                # Chase player
                speed = 1 if self.rng.random() < 0.6 else 0
                if coyote.x < self._player_x:
                    coyote.x += speed
                elif coyote.x > self._player_x:
                    coyote.x -= speed

        # Seed collection
        for e in self._entities:
            if (
                e.etype == "seed"
                and e.alive
                and e.x == self._player_x
                and abs(e.y - self._player_y) <= 1
            ):
                e.alive = False
                self._on_point_scored(1)
                if self._progress_count < self._WIN_TARGET:
                    reward += 1.0 / self._WIN_TARGET
                    self._progress_count += 1
                self._message = "Birdseed!"

        # Rock collision -- terminal failure
        for e in self._entities:
            if (
                e.etype == "rock"
                and e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                self._on_life_lost()
                self._message = "Hit a rock! Game Over."
                reward = self._DEATH_PENALTY
                return reward, self._game_over, info

        # Coyote-mine collision (counts as progress)
        if coyote and coyote.alive:
            for e in self._entities:
                if (
                    e.etype == "mine"
                    and e.alive
                    and e.x == coyote.x
                    and e.y == coyote.y
                ):
                    e.alive = False
                    self._coyote_stun = 20
                    coyote.x = max(1, coyote.x - 8)
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Coyote hit mine!"

        # Coyote catches player -- terminal failure
        if (
            coyote
            and coyote.alive
            and self._coyote_stun <= 0
            and coyote.x == self._player_x
            and abs(coyote.y - self._player_y) <= 1
        ):
            self._on_life_lost()
            self._message = "Caught by Coyote! Game Over."
            reward = self._DEATH_PENALTY
            return reward, self._game_over, info

        # Win check
        if (
            self._progress_count >= self._WIN_TARGET
            and not self._game_over
        ):
            self._game_over = True
            info["won"] = True
            self._message = "Run complete!"
            return reward, self._game_over, info

        self._entities = [
            e for e in self._entities if e.alive
        ]

        # Spawn new items
        self._spawn_timer += 1
        if self._spawn_timer >= max(6 - self._level, 3):
            self._spawn_timer = 0
            r = self.rng.random()
            if r < 0.4:
                self._add_entity(
                    "seed", "o", _W - 2, _ROAD_Y,
                )
            elif r < 0.7:
                self._add_entity(
                    "rock", "█", _W - 2, _ROAD_Y, dx=-1,
                )
            else:
                self._add_entity(
                    "mine", "*", _W - 2, _ROAD_Y,
                )

        # Level progression
        if self._distance >= 100 + self._level * 30:
            self._level += 1
            self._message = "Stage complete!"
            self._generate_level(self._level * 4007)

        info["distance"] = self._distance
        return reward, self._game_over, info

    def _advance_entities(self) -> None:
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "border",
            "=": "ground",
            "_": "road",
            "·": "desert",
            "o": "birdseed",
            "█": "rock obstacle",
            "*": "mine",
            "W": "Coyote",
            " ": "sky",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        seeds = sum(
            1 for e in self._entities
            if e.etype == "seed" and e.alive
        )
        coyote_state = "stunned" if self._coyote_stun > 0 else "chasing"
        extra = (
            f"Seeds: {seeds}  Coyote: {coyote_state}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Run along the road as the Road Runner. "
            "Collect birdseed (o), avoid rocks (#), and "
            "outrun the Coyote (W). JUMP over obstacles. "
            "Lure Coyote into mines (*) for bonus points."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Road Runner.\n\n"
            "TASK\n"
            "Run the Road Runner along a scrolling desert road: "
            "collect birdseed, dodge rocks, and outpace or stun "
            "the Coyote. Travel far enough to advance stages.\n\n"
            "BOARD\n"
            "30x16 desert. Sky '-' fills the top; ground '=' and "
            "road '_' run across the lower portion with desert '.' "
            "below. Birdseed "
            "'o' on the road, rocks '#' scroll toward you with "
            "dx=-1, mines '*' sit on the road. Coyote 'W' chases "
            "from the left. You are an arrow glyph.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT shift you 1 cell. JUMP leaps 2 rows up "
            "for 4 steps (lets you clear rocks). The world scrolls "
            "every 2 steps (rocks/seeds/mines shift left). Coyote "
            "moves toward you with 60 percent chance/step; if the "
            "coyote steps onto a mine, the mine explodes, stunning "
            "the coyote for 20 steps and pushing it 8 cells back.\n\n"
            "SCORING\n"
            "Pattern D: +1/50 reward per progress event "
            "(birdseed collected or coyote stunned by mine). -1 "
            "reward on collision with a rock or the coyote. "
            "Cumulative reward bound: [-1, +1].\n\n"
            "TERMINATION\n"
            "Single-life: hitting a rock or being caught by the "
            "coyote ends the episode with reward -1. Reaching 50 "
            "progress events ends with cumulative +1. Episode "
            "also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, seeds on screen, and "
            "coyote state (chasing or stunned).\n\n"
            + self.action_spec.render_for_prompt()
        )
