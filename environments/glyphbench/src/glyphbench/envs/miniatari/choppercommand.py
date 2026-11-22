"""miniatari Chopper Command.

Identity: Helicopter shooter; destroy enemy helis while protecting the
truck convoy on the ground.
Win condition: destroy all 5 enemy helicopters with at least one truck
still alive.
Reward: Pattern A, +1/5 per enemy destroyed; Pattern D, terminal -1 if
every truck is destroyed (convoy lost).
Loss: collision with an enemy helicopter ends the run (no -1, just
terminates); all trucks destroyed -> -1 terminal.

Gym ID: glyphbench/miniatari-choppercommand-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=59, mean_return=+0.160
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniChopperCommandEnv(MiniatariBase):
    """Mini Chopper Command: 16x10 grid, 5 enemy helis above a 3-truck convoy.

    Player heli (Y) starts mid-left. Enemy helis (E) drift horizontally on
    rows 2..6, bouncing off side walls. 3 trucks (T) sit stationary on the
    ground row (y=8). Each tick the player can move LEFT/RIGHT/UP/DOWN by
    1 cell or FIRE a bullet in their last horizontal facing direction. The
    bullet travels up to 6 cells in its direction that same tick,
    destroying the first enemy in line. Enemies drift one cell every 2
    ticks; an enemy that is directly above a truck has a small chance to
    dive once per tick and obliterate that truck. Colliding with an enemy
    ends the run with no further reward. Lose if every truck is destroyed
    (-1 terminal). Win by destroying all 5 enemies while at least one
    truck is still alive (+1/5 per kill, Pattern A).
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"),
        descriptions=(
            "do nothing",
            "move left one cell and face left",
            "move right one cell and face right",
            "move up one cell",
            "move down one cell",
            "fire a bullet horizontally in your facing direction",
        ),
    )

    default_max_turns = 200

    _WIDTH = 16
    _HEIGHT = 10
    _WIN_TARGET = 5
    _GROUND_Y = 9
    _TRUCK_Y = 8
    _BULLET_RANGE = 6
    _ENEMY_MOVE_EVERY = 2
    _N_TRUCKS = 3
    _DIVE_PROB = 0.02  # per-tick prob an enemy directly above a truck dives

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        # enemies as [x, y, dx]
        self._enemies: list[list[int]] = []
        # truck columns still alive (a set for O(1) hits)
        self._trucks: set[int] = set()
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-choppercommand-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        self._enemies = []
        self._player_x = 1
        self._player_y = 5
        self._player_dir = (1, 0)
        rng = self.rng
        # 5 enemy helis at rows 2..6, varying x and direction
        rows = [2, 3, 4, 5, 6]
        used: set[tuple[int, int]] = set()
        for r in rows:
            for _attempt in range(20):
                x = int(rng.integers(4, self._WIDTH - 1))
                if (x, r) in used:
                    continue
                used.add((x, r))
                dx = 1 if rng.random() < 0.5 else -1
                self._enemies.append([x, r, dx])
                break
        # Place 3 trucks at fixed columns evenly spaced across the convoy.
        self._trucks = {3, 7, 12}

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
        elif action_name == "UP" and self._player_y > 0:
            self._player_y -= 1
        elif action_name == "DOWN" and self._player_y < self._GROUND_Y - 1:
            self._player_y += 1

        # 2. Fire (resolves same tick, horizontal)
        if action_name == "FIRE":
            bdx = self._player_dir[0] if self._player_dir[0] != 0 else 1
            bx = self._player_x
            target: int | None = None
            for _ in range(self._BULLET_RANGE):
                bx += bdx
                if bx < 0 or bx >= self._WIDTH:
                    break
                hit = False
                for i, (ex, ey, _edx) in enumerate(self._enemies):
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
                self._message = f"Enemy down! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET and self._trucks:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Move enemies (every 2 ticks)
        if self._tick_count % self._ENEMY_MOVE_EVERY == 0:
            for e in self._enemies:
                e[0] += e[2]
                if e[0] <= 0 or e[0] >= self._WIDTH - 1:
                    e[2] *= -1
                    e[0] = max(0, min(self._WIDTH - 1, e[0]))

        # 4. Truck attacks: any enemy directly over a surviving truck has
        #    a small chance to dive and obliterate it.
        rng = self.rng
        for e in self._enemies:
            ex = e[0]
            if ex in self._trucks and rng.random() < self._DIVE_PROB:
                self._trucks.discard(ex)
                self._message = f"Truck at x={ex} destroyed!"
                if not self._trucks:
                    self._message = "All trucks lost — convoy destroyed!"
                    reward += self._death_reward()
                    self._on_life_lost()
                    return reward, True, info

        # 5. Collision check (player vs enemy)
        for ex, ey, _edx in self._enemies:
            if ex == self._player_x and ey == self._player_y:
                self._message = "Collision with enemy helicopter!"
                self._game_over = True
                self._won = False
                return reward, True, info

        info["progress"] = self._progress
        info["enemies_left"] = len(self._enemies)
        info["trucks_alive"] = len(self._trucks)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Ground
        for x in range(self._WIDTH):
            grid[self._GROUND_Y][x] = "="
        # Trucks (one row above the ground)
        for tx in self._trucks:
            if 0 <= tx < self._WIDTH:
                grid[self._TRUCK_Y][tx] = "T"
        # Enemies
        for ex, ey, _edx in self._enemies:
            if 0 <= ex < self._WIDTH and 0 <= ey < self._HEIGHT:
                grid[ey][ex] = "E"
        # Player
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = "Y"
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "sky",
            "=": "ground",
            "E": "enemy helicopter",
            "T": "truck (protect)",
            "Y": "your helicopter",
        }
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Killed: {self._progress}/{self._WIN_TARGET}    "
            f"Trucks: {len(self._trucks)}/{self._N_TRUCKS}    "
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
            "Mini Chopper Command on a 16x10 grid. Your helicopter (Y) flies "
            "above a desert floor (=). 5 enemy helis (E) drift on rows 2..6, "
            "bouncing off side walls every 2 ticks. 3 friendly trucks (T) "
            "sit on the convoy row at y=8 in fixed columns. LEFT/RIGHT "
            "moves you and sets your facing; UP/DOWN changes altitude. "
            "FIRE shoots a bullet up to 6 cells in your facing direction; "
            "the first enemy in your row is destroyed instantly. Each tick "
            "an enemy directly above a surviving truck has a small chance "
            "to dive and destroy that truck. Destroy all 5 enemies while "
            "at least one truck is still alive to win (+1/5 per kill). "
            "Lose with -1 if every truck is destroyed. Colliding with an "
            "enemy ends the run with no further reward."
        )
