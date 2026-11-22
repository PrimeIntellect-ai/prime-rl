"""miniatari Robot Tank.

Identity: Top-down tank combat; destroy enemy tanks before they hit you.
Win condition: destroy 4 enemy tanks.
Reward: Pattern A, +1/4 per enemy tank destroyed.
Loss: collision with an enemy tank ends the run (no -1; just terminates).

Gym ID: glyphbench/miniatari-robotank-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=12, mean_return=+0.000
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


def _sign(a: int, b: int) -> int:
    return 1 if a > b else (-1 if a < b else 0)


class MiniRobotankEnv(MiniatariBase):
    """Mini Robot Tank: 12x12 grid, 4 enemy tanks pursue you.

    Player tank (Y, arrow shows facing) starts at center. Enemies (E)
    spawn at corners and march one cell every 2 ticks toward the player.
    FIRE shoots in the player's facing direction up to 5 cells; first
    enemy hit is destroyed. Colliding with an enemy ends the run.
    Destroying 4 enemies wins.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"),
        descriptions=(
            "do nothing",
            "move left and face left",
            "move right and face right",
            "move up and face up",
            "move down and face down",
            "fire in your current facing direction",
        ),
    )

    default_max_turns = 200

    _WIDTH = 12
    _HEIGHT = 12
    _WIN_TARGET = 4
    _BULLET_RANGE = 5
    _ENEMY_MOVE_EVERY = 2

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        # enemies: [x, y]
        self._enemies: list[list[int]] = []
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-robotank-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT // 2
        self._player_dir = (0, -1)  # face up
        # 4 enemies at corners
        self._enemies = [
            [1, 1],
            [self._WIDTH - 2, 1],
            [1, self._HEIGHT - 2],
            [self._WIDTH - 2, self._HEIGHT - 2],
        ]

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Player move
        nx, ny = self._player_x, self._player_y
        if action_name == "LEFT":
            nx = max(0, nx - 1)
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx = min(self._WIDTH - 1, nx + 1)
            self._player_dir = (1, 0)
        elif action_name == "UP":
            ny = max(0, ny - 1)
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            ny = min(self._HEIGHT - 1, ny + 1)
            self._player_dir = (0, 1)
        # Don't allow stepping onto an enemy via movement (collision check below)
        self._player_x, self._player_y = nx, ny

        # 2. Fire (resolves same tick)
        if action_name == "FIRE":
            bdx, bdy = self._player_dir
            if bdx == 0 and bdy == 0:
                bdy = -1
            bx, by = self._player_x, self._player_y
            target: int | None = None
            for _ in range(self._BULLET_RANGE):
                bx += bdx
                by += bdy
                if bx < 0 or bx >= self._WIDTH or by < 0 or by >= self._HEIGHT:
                    break
                hit = False
                for i, (ex, ey) in enumerate(self._enemies):
                    if ex == bx and ey == by:
                        target = i
                        hit = True
                        break
                if hit:
                    break
            if target is not None:
                self._enemies.pop(target)
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = f"Enemy tank down! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Move enemies (every 2 ticks)
        if self._tick_count % self._ENEMY_MOVE_EVERY == 0:
            for e in self._enemies:
                # Step 1 cell toward player (axis with greater distance)
                dx = _sign(self._player_x, e[0])
                dy = _sign(self._player_y, e[1])
                if abs(self._player_x - e[0]) >= abs(self._player_y - e[1]):
                    e[0] += dx
                else:
                    e[1] += dy

        # 4. Collision check
        for ex, ey in self._enemies:
            if ex == self._player_x and ey == self._player_y:
                self._message = "Enemy tank rammed you!"
                self._game_over = True
                self._won = False
                return reward, True, info

        info["progress"] = self._progress
        info["enemies_left"] = len(self._enemies)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Enemies
        for ex, ey in self._enemies:
            if 0 <= ex < self._WIDTH and 0 <= ey < self._HEIGHT:
                grid[ey][ex] = "E"
        # Player (with facing)
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "ground",
            "E": "enemy tank",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"your tank (facing {self._DIR_NAMES.get(self._player_dir, 'up')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Killed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Enemies: {len(self._enemies)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Robot Tank on a 12x12 battlefield. You command a tank "
            "(arrow shows facing) starting at the center. 4 enemy tanks "
            "(E) start at the corners and pursue you, stepping 1 cell "
            "every 2 ticks toward your position (taking the axis with the "
            "greater distance). LEFT/RIGHT/UP/DOWN moves you 1 cell and "
            "sets your facing. FIRE shoots a bullet up to 5 cells in your "
            "facing direction; the first enemy hit is destroyed. Colliding "
            "with any enemy ends the run. Destroy all 4 enemies to win. "
            "Reward: +1/4 per enemy destroyed."
        )
