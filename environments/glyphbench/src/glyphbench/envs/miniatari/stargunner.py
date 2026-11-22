"""miniatari Star Gunner.

Identity: Side-scrolling shooter; enemies stream in from the right.
Win condition: destroy 5 enemy ships.
Reward: Pattern A, +1/5 per ship destroyed.
Loss: an enemy ship reaches your column ends the run (no -1; just terminates).

Gym ID: glyphbench/miniatari-stargunner-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=19, mean_return=+0.120
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniStarGunnerEnv(MiniatariBase):
    """Mini Star Gunner: 14x10 grid, scrolling space shooter.

    Player ship (Y) at column 0..3 (left side). Enemies (E) spawn at the
    right edge and march left 1 cell every 2 ticks. UP/DOWN moves the
    player vertically; LEFT/RIGHT moves horizontally within the player
    zone (columns 0..3). FIRE shoots a bullet rightward up to 10 cells;
    the first enemy in the same row is destroyed. If any enemy reaches
    column <= player's column, the run ends. Destroy 5 enemies to win.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move up one cell",
            "move down one cell",
            "move left within the player zone",
            "move right within the player zone",
            "fire a bullet rightward",
        ),
    )

    default_max_turns = 200

    _WIDTH = 14
    _HEIGHT = 10
    _WIN_TARGET = 5
    _PLAYER_X_MAX = 3
    _BULLET_RANGE = 10
    _ENEMY_MOVE_EVERY = 2
    _SPAWN_EVERY = 4

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._enemies: list[list[int]] = []  # [x, y]
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-stargunner-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        self._player_x = 1
        self._player_y = self._HEIGHT // 2
        self._player_dir = (1, 0)
        rng = self.rng
        # Pre-seed 3 enemies on the right side at distinct rows
        used_rows: set[int] = set()
        self._enemies = []
        for _ in range(3):
            for _attempt in range(20):
                ey = int(rng.integers(0, self._HEIGHT))
                if ey in used_rows:
                    continue
                used_rows.add(ey)
                ex = int(rng.integers(self._WIDTH - 4, self._WIDTH))
                self._enemies.append([ex, ey])
                break

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Player move
        if action_name == "UP" and self._player_y > 0:
            self._player_y -= 1
            self._player_dir = (0, -1)
        elif action_name == "DOWN" and self._player_y < self._HEIGHT - 1:
            self._player_y += 1
            self._player_dir = (0, 1)
        elif action_name == "LEFT" and self._player_x > 0:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._PLAYER_X_MAX:
            self._player_x += 1
            self._player_dir = (1, 0)

        # 2. Fire (rightward)
        if action_name == "FIRE":
            bx = self._player_x
            target: int | None = None
            for _ in range(self._BULLET_RANGE):
                bx += 1
                if bx >= self._WIDTH:
                    break
                hit = False
                for i, (ex, ey) in enumerate(self._enemies):
                    if ex == bx and ey == self._player_y:
                        target = i
                        hit = True
                        break
                if hit:
                    break
            if target is not None:
                self._enemies.pop(target)
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = f"Ship down! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Move enemies leftward
        if self._tick_count % self._ENEMY_MOVE_EVERY == 0:
            for e in self._enemies:
                e[0] -= 1
            # Check breakthrough
            for ex, _ey in self._enemies:
                if ex <= self._player_x:
                    self._message = "Enemy ship breached your zone!"
                    self._game_over = True
                    self._won = False
                    return reward, True, info

        # 4. Spawn new enemy periodically
        if self._tick_count % self._SPAWN_EVERY == 0:
            rng = self.rng
            occupied = {(e[0], e[1]) for e in self._enemies}
            for _attempt in range(20):
                ey = int(rng.integers(0, self._HEIGHT))
                if (self._WIDTH - 1, ey) not in occupied:
                    self._enemies.append([self._WIDTH - 1, ey])
                    break

        info["progress"] = self._progress
        info["enemies"] = len(self._enemies)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Player-zone divider
        for y in range(self._HEIGHT):
            grid[y][self._PLAYER_X_MAX + 1] = "│"
        # Enemies
        for ex, ey in self._enemies:
            if 0 <= ex < self._WIDTH and 0 <= ey < self._HEIGHT:
                grid[ey][ex] = "E"
        # Player
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "space",
            "│": "player-zone divider",
            "E": "enemy ship",
            "Y": "your ship",
        }
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
            "Mini Star Gunner on a 14x10 grid. Your ship (Y) holds the "
            "left-hand player zone (columns 0..3, divided by │). Enemies "
            "(E) march left 1 cell every 2 ticks; new enemies spawn at the "
            "right edge every 4 ticks. UP/DOWN moves vertically; LEFT/RIGHT "
            "moves within the player zone. FIRE shoots a bullet rightward "
            "up to 10 cells, destroying the first enemy in your row. If "
            "any enemy reaches your column or further left, the run ends. "
            "Destroy 5 ships to win. Reward: +1/5 per kill."
        )
