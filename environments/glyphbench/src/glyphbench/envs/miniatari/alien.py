"""miniatari Alien.

Identity: Hunt aliens through a small maze; aliens chase you.
Win condition: kill 4 aliens.
Reward: Pattern D, +1/4 per kill; -1 if an alien catches you.

Gym ID: glyphbench/miniatari-alien-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=18, mean_return=-0.961
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


def _sign(a: int, b: int) -> int:
    return 1 if a > b else (-1 if a < b else 0)


class MiniAlienEnv(MiniatariBase):
    """Mini Alien: 12x10 maze, 4 aliens, FIRE shoots in facing direction.

    The maze is a 12x10 grid with a border wall (█) and a row of interior
    pillars at odd columns of the middle rows. Player starts at the
    bottom-center, 4 aliens (a) at scattered open cells. Aliens step
    1 cell every 3 ticks toward the player on the dominant axis (skipped
    if blocked by a wall). FIRE shoots up to 4 cells in the facing
    direction; first alien in line dies. Pattern D: +1/4 per kill, -1 if
    caught.
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

    default_max_turns = 300

    _WIDTH = 12
    _HEIGHT = 10
    _WIN_TARGET = 4
    _BULLET_RANGE = 4
    _ALIEN_MOVE_EVERY = 3

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._aliens: list[list[int]] = []
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-alien-v0"

    def _is_wall(self, x: int, y: int) -> bool:
        if x <= 0 or x >= self._WIDTH - 1 or y <= 0 or y >= self._HEIGHT - 1:
            return True
        # Interior pillars at every other column on rows 3, 5, 7
        return bool(y in (3, 5, 7) and x % 2 == 1 and x not in (5, 7))

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        # Walls
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                if self._is_wall(x, y):
                    self._set_cell(x, y, "█")
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT - 2
        self._player_dir = (0, -1)
        # Place 4 aliens at distinct open cells far from the player
        rng = self.rng
        used: set[tuple[int, int]] = {(self._player_x, self._player_y)}
        self._aliens = []
        for _ in range(self._WIN_TARGET):
            for _attempt in range(60):
                x = int(rng.integers(1, self._WIDTH - 1))
                y = int(rng.integers(1, self._HEIGHT - 1))
                if (x, y) in used or self._is_wall(x, y):
                    continue
                if abs(x - self._player_x) + abs(y - self._player_y) < 4:
                    continue
                used.add((x, y))
                self._aliens.append([x, y])
                break

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Player move
        nx, ny = self._player_x, self._player_y
        if action_name == "LEFT":
            nx -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx += 1
            self._player_dir = (1, 0)
        elif action_name == "UP":
            ny -= 1
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            ny += 1
            self._player_dir = (0, 1)
        if not self._is_wall(nx, ny):
            self._player_x, self._player_y = nx, ny

        # 2. Fire
        if action_name == "FIRE":
            bdx, bdy = self._player_dir
            if bdx == 0 and bdy == 0:
                bdy = -1
            bx, by = self._player_x, self._player_y
            target: int | None = None
            for _ in range(self._BULLET_RANGE):
                bx += bdx
                by += bdy
                if self._is_wall(bx, by):
                    break
                hit = False
                for i, (ax, ay) in enumerate(self._aliens):
                    if ax == bx and ay == by:
                        target = i
                        hit = True
                        break
                if hit:
                    break
            if target is not None:
                self._aliens.pop(target)
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = f"Alien down! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Aliens move
        if self._tick_count % self._ALIEN_MOVE_EVERY == 0:
            for a in self._aliens:
                dx = _sign(self._player_x, a[0])
                dy = _sign(self._player_y, a[1])
                nax, nay = a[0], a[1]
                if abs(self._player_x - a[0]) >= abs(self._player_y - a[1]):
                    if not self._is_wall(a[0] + dx, a[1]):
                        nax = a[0] + dx
                    elif not self._is_wall(a[0], a[1] + dy):
                        nay = a[1] + dy
                else:
                    if not self._is_wall(a[0], a[1] + dy):
                        nay = a[1] + dy
                    elif not self._is_wall(a[0] + dx, a[1]):
                        nax = a[0] + dx
                a[0], a[1] = nax, nay

        # 4. Collision check (alien on player)
        for ax, ay in self._aliens:
            if ax == self._player_x and ay == self._player_y:
                self._message = "An alien caught you!"
                reward += self._death_reward()
                self._on_life_lost()
                return reward, True, info

        info["progress"] = self._progress
        info["aliens_left"] = len(self._aliens)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                if self._is_wall(x, y):
                    grid[y][x] = "█"
        for ax, ay in self._aliens:
            if 0 <= ax < self._WIDTH and 0 <= ay < self._HEIGHT:
                grid[ay][ax] = "a"
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "corridor",
            "█": "wall",
            "a": "alien",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'up')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Killed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Aliens: {len(self._aliens)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Alien on a 12x10 maze with border walls and a row of "
            "interior pillars (█). You (arrow shows facing) start at the "
            "bottom-center. 4 aliens (a) start scattered through the maze "
            "and step 1 cell every 3 ticks toward you on the dominant axis "
            "(blocked by walls). LEFT/RIGHT/UP/DOWN moves you 1 cell and "
            "sets your facing. FIRE shoots up to 4 cells in your facing "
            "direction; first alien in line is destroyed. Reward: +1/4 per "
            "kill. Being caught by an alien is -1 terminal."
        )
