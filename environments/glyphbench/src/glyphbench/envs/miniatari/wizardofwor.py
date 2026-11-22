"""miniatari Wizard of Wor.

Identity: Top-down dungeon shooter; clear monsters in a small maze.
Win condition: defeat all 5 monsters.
Reward: Pattern D, +1/5 per monster killed; -1 if a monster catches you.

Gym ID: glyphbench/miniatari-wizardofwor-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=14, mean_return=-0.980
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


def _sign(a: int, b: int) -> int:
    return 1 if a > b else (-1 if a < b else 0)


class MiniWizardOfWorEnv(MiniatariBase):
    """Mini Wizard of Wor: 14x10 dungeon with central pillars; 5 monsters.

    The dungeon is bordered by walls (█); a 4x2 block of pillars in the
    center (cols 5-8, rows 4-5) forces routing. Player (Y, arrow) starts
    at the bottom-center.
    5 monsters (m) start at scattered open cells. Monsters step 1 cell
    every 3 ticks toward the player on the dominant axis (skipped if
    blocked). FIRE shoots up to 5 cells in facing direction; first monster
    in line dies. Pattern D: +1/5 per kill, -1 if caught.
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

    _WIDTH = 14
    _HEIGHT = 10
    _WIN_TARGET = 5
    _BULLET_RANGE = 5
    _MONSTER_MOVE_EVERY = 3
    _PILLAR_RECT = (5, 4, 8, 5)  # x0,y0,x1,y1 inclusive

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._monsters: list[list[int]] = []
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-wizardofwor-v0"

    def _is_wall(self, x: int, y: int) -> bool:
        if x <= 0 or x >= self._WIDTH - 1 or y <= 0 or y >= self._HEIGHT - 1:
            return True
        x0, y0, x1, y1 = self._PILLAR_RECT
        return bool(x0 <= x <= x1 and y0 <= y <= y1)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._tick_count = 0
        self._progress = 0
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                if self._is_wall(x, y):
                    self._set_cell(x, y, "█")
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT - 2
        self._player_dir = (0, -1)

        rng = self.rng
        used: set[tuple[int, int]] = {(self._player_x, self._player_y)}
        self._monsters = []
        for _ in range(self._WIN_TARGET):
            for _attempt in range(60):
                x = int(rng.integers(1, self._WIDTH - 1))
                y = int(rng.integers(1, self._HEIGHT - 1))
                if (x, y) in used or self._is_wall(x, y):
                    continue
                if abs(x - self._player_x) + abs(y - self._player_y) < 4:
                    continue
                used.add((x, y))
                self._monsters.append([x, y])
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
                for i, (mx, my) in enumerate(self._monsters):
                    if mx == bx and my == by:
                        target = i
                        hit = True
                        break
                if hit:
                    break
            if target is not None:
                self._monsters.pop(target)
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = f"Monster down! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Monsters move
        if self._tick_count % self._MONSTER_MOVE_EVERY == 0:
            for m in self._monsters:
                dx = _sign(self._player_x, m[0])
                dy = _sign(self._player_y, m[1])
                nmx, nmy = m[0], m[1]
                if abs(self._player_x - m[0]) >= abs(self._player_y - m[1]):
                    if not self._is_wall(m[0] + dx, m[1]):
                        nmx = m[0] + dx
                    elif not self._is_wall(m[0], m[1] + dy):
                        nmy = m[1] + dy
                else:
                    if not self._is_wall(m[0], m[1] + dy):
                        nmy = m[1] + dy
                    elif not self._is_wall(m[0] + dx, m[1]):
                        nmx = m[0] + dx
                m[0], m[1] = nmx, nmy

        # 4. Catch?
        for mx, my in self._monsters:
            if mx == self._player_x and my == self._player_y:
                self._message = "A monster caught you!"
                reward += self._death_reward()
                self._on_life_lost()
                return reward, True, info

        info["progress"] = self._progress
        info["monsters_left"] = len(self._monsters)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                if self._is_wall(x, y):
                    grid[y][x] = "█"
        for mx, my in self._monsters:
            if 0 <= mx < self._WIDTH and 0 <= my < self._HEIGHT:
                grid[my][mx] = "m"
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "corridor",
            "█": "wall",
            "m": "monster",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'up')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Killed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Monsters: {len(self._monsters)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Wizard of Wor on a 14x10 dungeon with border walls and a "
            "4x2 central pillar block (cols 5-8, rows 4-5). You (arrow "
            "shows facing) start at the bottom-center. 5 monsters (m) sit "
            "at scattered open cells. Monsters step 1 cell every 3 ticks "
            "toward you on the dominant axis (skipping into walls). "
            "LEFT/RIGHT/UP/DOWN moves you 1 cell and sets facing. FIRE "
            "shoots up to 5 cells in your facing direction; first monster "
            "in line is destroyed. Reward: +1/5 per kill. Being caught is "
            "-1 terminal."
        )
