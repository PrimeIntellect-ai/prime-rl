"""Atari Gopher environment.

Defend a row of carrots from a tunnelling gopher. Multi-hole dig
mechanic mirrored from miniatari-gopher (Phase 3 redesign).

Gym ID: glyphbench/atari-gopher-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase


class GopherEnv(AtariBase):
    """Gopher: protect a row of carrots from a digging gopher.

    20x16 grid. The gopher tunnels at the bottom row, repositions
    every couple of ticks toward the nearest alive carrot, and digs
    one cell of dirt at its current column every few ticks. The
    agent walks across the surface row directly above the carrots
    and FILLs to restore one cell of dirt at its current column.

    A carrot is lost when its dirt column is fully eroded (the
    gopher reaches the carrot row at that column). The agent must
    defend the carrot row long enough for the time budget to
    expire; reward is proportional to the number of surviving
    carrots.

    Actions: NOOP, LEFT, RIGHT, FILL
    Pattern A: +1/_WIN_TARGET per surviving carrot at end of
    defense window. Failure (all carrots lost) -> -1.

    Audit BROKEN fix: previous impl had an auto-kill check
    (gopher at surface + no hole = dead) that fired the same step
    a gopher created its hole, with _WIN_TARGET = 8. Random play
    reliably maxed out reward. New impl removes auto-kill,
    requires multi-step digging through dirt, and ties reward to
    surviving the defense window.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FILL"),
        descriptions=(
            "do nothing",
            "shuffle left along the surface",
            "shuffle right along the surface",
            "shovel one dirt cell back into the column below you",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 16
    _N_CARROTS = 8
    _CARROT_COLS = (2, 4, 6, 9, 11, 14, 16, 18)
    _CARROT_Y = 4
    _DIRT_TOP = 5
    _DIRT_BOTTOM = 12  # rows 5..12 dirt by default
    _GOPHER_Y = 13
    _PLAYER_Y = 3
    _GOPHER_DIG_EVERY = 4
    _GOPHER_MOVE_EVERY = 2
    _DEFENSE_TURNS = 240

    _WIN_TARGET: int = _N_CARROTS
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        # dirt[col] = topmost dirt row (CARROT_Y+1 .. DIRT_BOTTOM).
        # When dirt[col] > DIRT_BOTTOM the column is fully eroded.
        self._dirt: dict[int, int] = {}
        self._carrots_alive: dict[int, bool] = {}
        self._gopher_x: int = 0
        self._tick_count: int = 0
        self._taken: int = 0  # number of carrots stolen so far

    def env_id(self) -> str:
        return "glyphbench/atari-gopher-v0"

    def _reset(self, seed: int) -> GridObservation:
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._tick_count = 0
        self._taken = 0
        # Border
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")
        # Init dirt (full) and carrots (alive).
        self._dirt = {col: self._DIRT_TOP for col in self._CARROT_COLS}
        self._carrots_alive = {col: True for col in self._CARROT_COLS}
        self._player_x = self._WIDTH // 2
        self._player_y = self._PLAYER_Y
        self._player_dir = (0, 1)
        # Gopher starts at a random alive-carrot column on tunnel row
        rng = self.rng
        self._gopher_x = self._CARROT_COLS[
            int(rng.integers(0, self._N_CARROTS))
        ]
        self._redraw()

    def _surviving_carrots(self) -> int:
        return sum(1 for v in self._carrots_alive.values() if v)

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Player move
        if action_name == "LEFT" and self._player_x > 1:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif (
            action_name == "RIGHT"
            and self._player_x < self._WIDTH - 2
        ):
            self._player_x += 1
            self._player_dir = (1, 0)

        # 2. FILL: restore one dirt cell at player's column (if it
        # is a carrot column and the carrot is still alive).
        if action_name == "FILL" and (
            self._player_x in self._dirt
            and self._carrots_alive.get(self._player_x, False)
        ):
            cur_top = self._dirt[self._player_x]
            if cur_top > self._DIRT_TOP:
                self._dirt[self._player_x] = cur_top - 1
                self._message = "Packed dirt back!"

        # 3. Gopher repositions toward nearest alive carrot column.
        if self._tick_count % self._GOPHER_MOVE_EVERY == 0:
            alive_cols = [
                c for c, alive in self._carrots_alive.items() if alive
            ]
            if alive_cols:
                target = min(
                    alive_cols,
                    key=lambda c: abs(c - self._gopher_x),
                )
                if self._gopher_x < target:
                    self._gopher_x += 1
                elif self._gopher_x > target:
                    self._gopher_x -= 1

        # 4. Gopher digs upward at current column every DIG_EVERY
        # ticks.
        if self._tick_count % self._GOPHER_DIG_EVERY == 0 and (
            self._gopher_x in self._dirt
            and self._carrots_alive.get(self._gopher_x, False)
        ):
            cur_top = self._dirt[self._gopher_x]
            if cur_top <= self._DIRT_BOTTOM:
                self._dirt[self._gopher_x] = cur_top + 1
            else:
                self._carrots_alive[self._gopher_x] = False
                self._taken += 1
                self._message = (
                    f"Gopher took a carrot! "
                    f"({self._surviving_carrots()} left)"
                )
                if self._surviving_carrots() == 0:
                    self._on_life_lost()
                    reward = self._DEATH_PENALTY
                    self._redraw()
                    return reward, True, info

        # 5. End-of-defense reward at DEFENSE_TURNS.
        if self._tick_count >= self._DEFENSE_TURNS:
            survivors = self._surviving_carrots()
            reward = survivors * (1.0 / self._WIN_TARGET)
            self._game_over = True
            self._message = (
                f"Time's up — {survivors}/{self._WIN_TARGET} "
                "carrots safe."
            )
            info["won"] = survivors > 0
            self._redraw()
            return reward, True, info

        info["taken"] = self._taken
        info["carrots_left"] = self._surviving_carrots()
        self._redraw()
        return reward, self._game_over, info

    def _redraw(self) -> None:
        # Clear interior except borders.
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        # Surface line just below carrots.
        for x in range(1, self._WIDTH - 1):
            self._set_cell(x, self._GOPHER_Y + 1, "─")
        # Carrots.
        for col, alive in self._carrots_alive.items():
            if alive:
                self._set_cell(col, self._CARROT_Y, "↑")
        # Dirt for each carrot column.
        for col in self._CARROT_COLS:
            if not self._carrots_alive.get(col, False):
                continue
            top = self._dirt[col]
            for y in range(top, self._DIRT_BOTTOM + 1):
                self._set_cell(col, y, "▓")
        # Gopher.
        if 0 < self._gopher_x < self._WIDTH - 1:
            self._set_cell(self._gopher_x, self._GOPHER_Y, "G")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall",
            "│": "wall",
            "↑": "carrot",
            "▓": "dirt",
            "G": "gopher",
            " ": "empty",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        carrots_left = self._surviving_carrots()
        extra = (
            f"Carrots: {carrots_left}/{self._N_CARROTS}  "
            f"Tick: {self._tick_count}/{self._DEFENSE_TURNS}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Defend the row of carrots (↑) from the digging "
            "gopher (G). The gopher tunnels upward through "
            "dirt (▓) toward the nearest alive carrot. "
            "FILL packs one cell of dirt back into the column "
            "below you. Survive the defense window."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Gopher.\n\n"
            "TASK\n"
            "Defend a row of 8 carrots from a digging gopher. "
            "The gopher tunnels upward through dirt toward the "
            "nearest alive carrot; you walk along the surface "
            "row above the carrots and FILL to restore dirt.\n\n"
            "BOARD\n"
            f"{self._WIDTH} columns by {self._HEIGHT} rows. "
            "Walls '-' / '|' on the borders. Carrots '↑' sit at "
            f"row {self._CARROT_Y} in 8 fixed columns. Dirt '▓' "
            f"fills the column below each carrot from row "
            f"{self._DIRT_TOP} down to row {self._DIRT_BOTTOM}. "
            f"The gopher 'G' tunnels at row {self._GOPHER_Y}. "
            f"You walk along row {self._PLAYER_Y} (just above "
            "the carrots) and appear as an arrow glyph.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT shuffle you along the surface row. "
            "FILL restores one cell of dirt directly below your "
            "column (if the column belongs to a still-alive "
            "carrot and there is dirt missing). Every "
            f"{self._GOPHER_MOVE_EVERY} ticks the gopher takes "
            "one step toward the nearest alive carrot column. "
            f"Every {self._GOPHER_DIG_EVERY} ticks the gopher "
            "removes one cell of dirt above it. When the dirt "
            "above a carrot is fully eroded, that carrot is "
            "stolen.\n\n"
            "SCORING\n"
            f"Pattern A: at the end of the {self._DEFENSE_TURNS}-"
            "tick defense window, +1/8 reward per surviving "
            "carrot (max +1 for 8 alive). -1 if all 8 carrots "
            "are lost (terminates early).\n\n"
            "TERMINATION\n"
            "Single-life: episode ends when all 8 carrots are "
            "stolen (-1) or when the defense window expires "
            "(reward = surviving / 8). The episode also ends "
            "after max_turns.\n\n"
            "HUD\n"
            "Shows score, carrots remaining out of 8, and tick "
            "out of the defense window.\n\n"
            + self.action_spec.render_for_prompt()
        )
