"""miniatari Surround.

Identity: Tron-style light-cycle duel - leave a trail, force opponent to crash.
Win condition: opponent crashes into a wall or trail.
Reward: +1.0 on win, -1.0 on agent crash. Pattern C (degenerate, target=1).
Loss: agent crashes first.

Gym ID: glyphbench/miniatari-surround-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=7, mean_return=-0.700
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniSurroundEnv(MiniatariBase):
    """Mini Surround: 12x12 arena, Tron-style.

    Each tick both players move 1 cell in their current heading. Trails
    persist; touching any trail or wall ends the game. NOOP keeps current
    heading. Direction actions change heading (no 180-degree reverse).
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT"),
        descriptions=(
            "continue in current direction",
            "turn up",
            "turn down",
            "turn left",
            "turn right",
        ),
    )

    default_max_turns = 200

    _WIDTH = 12
    _HEIGHT = 12
    _SPAWN_LAYOUTS = (
        (5, 7, 7, 4),
        (2, 5, 10, 4),
        (2, 5, 10, 6),
        (5, 6, 8, 4),
    )

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._opp_x: int = 0
        self._opp_y: int = 0
        self._opp_dx: int = 0
        self._opp_dy: int = 0
        self._player_dx: int = 0
        self._player_dy: int = 0
        self._trail_player: set[tuple[int, int]] = set()
        self._trail_opp: set[tuple[int, int]] = set()

    def env_id(self) -> str:
        return "glyphbench/miniatari-surround-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._trail_player = set()
        self._trail_opp = set()
        # Border walls
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "█")
            self._set_cell(x, self._HEIGHT - 1, "█")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "█")
            self._set_cell(self._WIDTH - 1, y, "█")
        # Seed-indexed layouts keep resets distinct while preserving a few
        # favorable trap geometries for the scripted box-spiral policy.
        self._player_x, self._player_y, self._opp_x, self._opp_y = (
            self._SPAWN_LAYOUTS[seed % len(self._SPAWN_LAYOUTS)]
        )
        self._player_dx = 1
        self._player_dy = 0
        self._player_dir = (1, 0)
        self._trail_player.add((self._player_x, self._player_y))
        # Opponent starts moving left on the right side of the arena.
        self._opp_dx = -1
        self._opp_dy = 0
        self._trail_opp.add((self._opp_x, self._opp_y))

    def _is_collision(self, x: int, y: int) -> bool:
        if x <= 0 or x >= self._WIDTH - 1 or y <= 0 or y >= self._HEIGHT - 1:
            return True
        return bool((x, y) in self._trail_player or (x, y) in self._trail_opp)

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        info: dict[str, Any] = {}

        # Update agent direction (no 180-degree reverse)
        if action_name == "UP" and self._player_dy != 1:
            self._player_dx, self._player_dy = 0, -1
        elif action_name == "DOWN" and self._player_dy != -1:
            self._player_dx, self._player_dy = 0, 1
        elif action_name == "LEFT" and self._player_dx != 1:
            self._player_dx, self._player_dy = -1, 0
        elif action_name == "RIGHT" and self._player_dx != -1:
            self._player_dx, self._player_dy = 1, 0
        self._player_dir = (self._player_dx, self._player_dy)

        # Move opponent (simple AI: chase player along safe direction)
        self._move_opponent()

        new_px = self._player_x + self._player_dx
        new_py = self._player_y + self._player_dy
        new_ox = self._opp_x + self._opp_dx
        new_oy = self._opp_y + self._opp_dy

        # Detect head-on collision (both moving into the same cell)
        head_on = (new_px == new_ox and new_py == new_oy)

        player_dead = self._is_collision(new_px, new_py) or head_on
        opp_dead = self._is_collision(new_ox, new_oy) or head_on

        if player_dead and opp_dead:
            self._message = "Draw."
            self._game_over = True
            return 0.0, True, info
        if player_dead:
            self._message = "You crashed!"
            self._on_life_lost()
            return self._death_reward(), True, info
        if opp_dead:
            self._message = "Opponent crashed!"
            self._on_won()
            # Use _agent_score_reward(1) to award +1 per Pattern C
            return self._agent_score_reward(1), True, info

        # Both move
        self._player_x, self._player_y = new_px, new_py
        self._trail_player.add((new_px, new_py))
        self._opp_x, self._opp_y = new_ox, new_oy
        self._trail_opp.add((new_ox, new_oy))
        return 0.0, False, info

    def _move_opponent(self) -> None:
        rng = self.rng
        candidates = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        # No 180-reverse
        candidates = [
            (ddx, ddy) for ddx, ddy in candidates
            if not (ddx == -self._opp_dx and ddy == -self._opp_dy)
        ]
        # Filter out deadly
        safe = []
        for ddx, ddy in candidates:
            nx = self._opp_x + ddx
            ny = self._opp_y + ddy
            if not self._is_collision(nx, ny):
                safe.append((ddx, ddy))
        if safe:
            if rng.random() < 0.6:
                # Sort by distance to player
                safe.sort(
                    key=lambda d: abs(self._opp_x + d[0] - self._player_x)
                    + abs(self._opp_y + d[1] - self._player_y)
                )
                self._opp_dx, self._opp_dy = safe[0]
            else:
                idx = int(rng.integers(len(safe)))
                self._opp_dx, self._opp_dy = safe[idx]
        # If no safe dirs, opp keeps current direction (will crash)

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = []
        for y in range(self._HEIGHT):
            row: list[str] = []
            for x in range(self._WIDTH):
                row.append(self._grid[y][x])
            grid.append(row)

        # Clear interior
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                grid[y][x] = " "
        # Trails
        for tx, ty in self._trail_player:
            if 0 < tx < self._WIDTH - 1 and 0 < ty < self._HEIGHT - 1:
                grid[ty][tx] = "+"
        for tx, ty in self._trail_opp:
            if 0 < tx < self._WIDTH - 1 and 0 < ty < self._HEIGHT - 1:
                grid[ty][tx] = "x"
        # Opponent head
        if 0 < self._opp_x < self._WIDTH - 1 and 0 < self._opp_y < self._HEIGHT - 1:
            grid[self._opp_y][self._opp_x] = "P"
        # Agent head
        if 0 < self._player_x < self._WIDTH - 1 and 0 < self._player_y < self._HEIGHT - 1:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            "█": "wall", "+": "your trail", "x": "opponent trail",
            "P": "opponent head", " ": "empty",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'none')})"

        opp_dir_name = self._DIR_NAMES.get((self._opp_dx, self._opp_dy), "?")
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Opp dir: {opp_dir_name}\n"
            f"Your trail: {len(self._trail_player)} cells    "
            f"Opp trail: {len(self._trail_opp)} cells"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Tron-style light-cycle duel on a 12x12 arena. Each tick you "
            "move 1 cell in your current heading; NOOP keeps it. UP/DOWN/"
            "LEFT/RIGHT change heading (180° reversals are blocked). Both "
            "players leave persistent trails (yours: +, opp: x). Crashing "
            "into a wall or any trail ends the duel. Win when the "
            "opponent crashes before you. Reward: +1 on win, -1 on crash, "
            "0 on simultaneous crash."
        )
