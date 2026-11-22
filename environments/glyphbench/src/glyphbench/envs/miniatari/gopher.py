"""miniatari Gopher.

Identity: Defend a row of carrots from a digging gopher by filling holes.
Win condition: 4 carrots survive K turns.
Reward: Pattern D, +1/4 per surviving carrot at K turns; -1 if all 4 lost early.

Tuning (Phase 3 hardening): grid widened to 4 carrot holes. The gopher
now picks a target hole UNIFORMLY AT RANDOM each dig cycle instead of
just the nearest, so the agent must actively monitor all 4 holes
rather than camping at one. This breaks the random-baseline +0.40 from
the audit, which came from the gopher being too predictable.

Gym ID: glyphbench/miniatari-gopher-v0
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniGopherEnv(MiniatariBase):
    """Mini Gopher: 16x8 grid; defend 4 carrots through 60 turns.

    Carrots (▼) sit at row 2 in columns 2, 6, 9, 13. Below each carrot
    is a 2-cell column of dirt (▓). The gopher (G) tunnels at row 6
    and digs the dirt above its current target column ONCE every 3
    ticks (the dig-cycle cooldown defined by _GOPHER_DIG_EVERY).
    Whenever the cooldown elapses while the gopher stands at its
    target, it digs one dirt cell and re-rolls its target uniformly
    among alive carrots' columns; otherwise it shuffles 1 cell per
    tick toward the current target. The player (Y, with shovel ↓)
    walks along row 1 (just above the carrots) and FILL packs dirt
    into the column directly below them, restoring 1 dirt cell
    (clamped to 2 max). If the gopher reaches row 2 (steals a
    carrot), that carrot is lost. After 60 ticks, +1/4 per surviving
    carrot. Lose all 4 -> -1 terminal.
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

    default_max_turns = 300

    _WIDTH = 16
    _HEIGHT = 8
    _N_CARROTS = 4
    _WIN_TARGET = _N_CARROTS
    _CARROT_COLS = (2, 6, 9, 13)
    _CARROT_Y = 2
    # Phase 3.5 fix: dirt budget reduced to 2 cells (rows 4, 5). With the
    # _GOPHER_DIG_EVERY cooldown the gopher only digs every 3 ticks, so 2
    # dirt cells = 6 ticks per carrot — keeps the random baseline near 0
    # and preserves discrimination against smarter play.
    _DIRT_TOP = 4
    _DIRT_BOTTOM = 5  # rows 4, 5 are dirt by default
    _GOPHER_Y = 6
    _PLAYER_Y = 1
    _GOPHER_DIG_EVERY = 3  # gopher digs every K ticks
    _GOPHER_MOVE_EVERY = 1  # gopher repositions every K ticks (faster)
    _DEFENSE_TURNS = 60

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        # dirt[col_idx] = top dirt row (CARROT_Y+1 down through DIRT_BOTTOM).
        # If a column has no dirt left and the gopher reaches CARROT_Y, the
        # carrot at that col is taken.
        self._dirt: dict[int, int] = {}
        self._carrots_alive: dict[int, bool] = {}
        self._gopher_x: int = 0
        self._gopher_target_col: int = 0
        self._tick_count: int = 0
        self._progress: int = 0  # carrots taken so far (informational)
        # Phase 3.5 fix: rate-limit dig+reroll to one per _GOPHER_DIG_EVERY
        # ticks. Without this the gopher dug every tick once it stood at
        # its target column, ignoring the dig-cycle budget.
        self._last_dig_tick: int = -10**9

    def env_id(self) -> str:
        return "glyphbench/miniatari-gopher-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._tick_count = 0
        self._progress = 0
        # Each carrot column: top of dirt block is row CARROT_Y+1 (=3).
        # Empty layer represented by top=DIRT_BOTTOM+1 means no dirt.
        self._dirt = {col: self._DIRT_TOP for col in self._CARROT_COLS}
        self._carrots_alive = {col: True for col in self._CARROT_COLS}
        self._player_x = self._CARROT_COLS[1]
        self._player_y = self._PLAYER_Y
        self._player_dir = (0, 1)
        # Gopher starts at a random hole; refresh target each dig cycle.
        rng = self.rng
        self._gopher_target_col = self._CARROT_COLS[
            int(rng.integers(0, self._N_CARROTS))
        ]
        self._gopher_x = self._gopher_target_col
        # Reset dig cooldown so the gopher's first dig can fire on tick 1.
        self._last_dig_tick = -self._GOPHER_DIG_EVERY

    def _surviving_carrots(self) -> int:
        return sum(1 for v in self._carrots_alive.values() if v)

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Player move
        if action_name == "LEFT" and self._player_x > 0:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 1:
            self._player_x += 1
            self._player_dir = (1, 0)

        # 2. FILL: pack dirt into player's column if it's a carrot column
        if action_name == "FILL" and self._player_x in self._dirt:
            cur_top = self._dirt[self._player_x]
            if cur_top > self._DIRT_TOP:
                # Restore one dirt row at top
                self._dirt[self._player_x] = cur_top - 1
                self._message = "Packed dirt back!"

        # 3. Gopher repositions toward its current target column.
        # If the current target is dead, pick a new random alive target.
        alive_cols = [
            c for c, alive in self._carrots_alive.items() if alive
        ]
        if alive_cols and not self._carrots_alive.get(
            self._gopher_target_col, False
        ):
            self._gopher_target_col = alive_cols[
                int(self.rng.integers(0, len(alive_cols)))
            ]
        target = self._gopher_target_col
        if self._gopher_x < target:
            self._gopher_x += 1
        elif self._gopher_x > target:
            self._gopher_x -= 1

        # 4. Gopher digs ONCE per _GOPHER_DIG_EVERY ticks while standing
        # at its target column. The cooldown means the agent has dig-cycle
        # ticks of warning between strikes — enough room to reach the
        # column and FILL. After each dig the gopher rerolls its target
        # ("multi-hole digging").
        cooldown_ready = (
            self._tick_count - self._last_dig_tick >= self._GOPHER_DIG_EVERY
        )
        if (cooldown_ready and
                self._gopher_x == self._gopher_target_col and
                self._gopher_x in self._dirt and
                self._carrots_alive.get(self._gopher_x, False)):
            self._last_dig_tick = self._tick_count
            cur_top = self._dirt[self._gopher_x]
            if cur_top <= self._DIRT_BOTTOM:
                # Remove top dirt row
                self._dirt[self._gopher_x] = cur_top + 1
            else:
                # No dirt left -> gopher steals carrot
                self._carrots_alive[self._gopher_x] = False
                self._progress += 1
                self._message = (
                    f"Gopher took a carrot! "
                    f"({self._surviving_carrots()} left)"
                )
                if self._surviving_carrots() == 0:
                    reward = self._death_reward()
                    self._on_life_lost()
                    return reward, True, info
            # Re-roll target after this dig.
            alive_cols = [
                c for c, alive in self._carrots_alive.items() if alive
            ]
            if alive_cols:
                self._gopher_target_col = alive_cols[
                    int(self.rng.integers(0, len(alive_cols)))
                ]

        # 5. End-of-defense check at DEFENSE_TURNS. Only a perfect
        # defense (all 4 alive) sets won=True; partial defenses give
        # partial credit but flag won=False so the agent has to actually
        # save EVERY carrot to claim the win.
        if self._tick_count >= self._DEFENSE_TURNS:
            survivors = self._surviving_carrots()
            self._message = (
                f"Time's up — {survivors}/{self._WIN_TARGET} carrots safe."
            )
            if survivors == self._WIN_TARGET:
                # Pattern A: +1.0 (all 4 carrots * +1/4).
                reward = survivors * self._progress_reward(self._WIN_TARGET)
                self._on_won()
            elif survivors > 0:
                # Partial credit, but not a "win".
                reward = survivors * self._progress_reward(self._WIN_TARGET)
                self._game_over = True
                self._won = False
            else:
                # No carrots left -> Pattern D terminal -1.
                reward = self._death_reward()
                self._on_life_lost()
            return reward, True, info

        info["progress"] = self._progress
        info["carrots_left"] = self._surviving_carrots()
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Surface line (player walks here at row 1; ground row at row 2 below carrots)
        for x in range(self._WIDTH):
            grid[self._GOPHER_Y + 1][x] = "─"
        # Carrots
        for col in self._CARROT_COLS:
            if self._carrots_alive[col]:
                grid[self._CARROT_Y][col] = "▼"
        # Dirt (rows DIRT_TOP..DIRT_BOTTOM, only if dirt[col] <= row)
        for col in self._CARROT_COLS:
            top = self._dirt[col]
            for y in range(top, self._DIRT_BOTTOM + 1):
                grid[y][col] = "▓"
        # Gopher
        if 0 <= self._gopher_x < self._WIDTH:
            grid[self._GOPHER_Y][self._gopher_x] = "G"
        # Player
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "open air",
            "─": "ground line",
            "▼": "carrot",
            "▓": "dirt",
            "G": "gopher",
            "Y": "you (with shovel)",
        }

        carrots_left = self._surviving_carrots()
        dirt_remaining = sum(self._DIRT_BOTTOM - self._dirt[c] + 1 for c in self._CARROT_COLS)
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Carrots: {carrots_left}/{self._WIN_TARGET}    "
            f"Defense: {self._tick_count}/{self._DEFENSE_TURNS}    "
            f"Score: {self._score:.3f}    "
            f"Dirt remaining: {dirt_remaining}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Gopher on a 16x8 field. Carrots (▼) sit at row 2 in "
            "columns 2, 6, 9, 13; below each is a 2-cell dirt column "
            "(▓). The gopher (G) tunnels at row 6. Once every 3 ticks "
            "(the dig cooldown), if the gopher is standing at its "
            "current target column it digs one dirt cell and picks a "
            "new random target among alive carrots; otherwise it "
            "shuffles 1 cell per tick toward its target. You (Y) walk "
            "row 1; LEFT/RIGHT moves you. FILL adds 1 dirt row back "
            "into the carrot column directly below you (clamped to 2 "
            "max). After 60 ticks the episode ends. Reward: +1/4 per "
            "surviving carrot. If all 4 are stolen before time runs "
            "out, -1 terminal."
        )
