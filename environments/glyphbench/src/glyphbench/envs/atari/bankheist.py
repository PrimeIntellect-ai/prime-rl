"""Atari Bank Heist environment.

Navigate a maze, rob banks, avoid police, escape before gas runs out.

Gym ID: glyphbench/atari-bankheist-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity

_DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

_W = 20
_H = 16
_BANK_CHAR = "$"
_POLICE_CHAR = "p"
_EXIT_CHAR = "E"
_GAS_START = 200
_DYNAMITE_CHAR = "d"

class BankHeistEnv(AtariBase):
    """Bank Heist: rob banks, avoid police, reach the exit.

    Actions: NOOP, UP, RIGHT, LEFT, DOWN, FIRE
    Pattern D: +1/_WIN_TARGET per bank robbed, -1 if caught by police
    or out of gas (single-life model).
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "RIGHT", "LEFT", "DOWN", "FIRE"),
        descriptions=(
            "do nothing this step",
            "move up one cell",
            "move right one cell",
            "move left one cell",
            "move down one cell",
            "drop dynamite to stun nearby police",
        ),
    )

    # Pattern D full-scope target: 30 banks robbed across multiple levels.
    _WIN_TARGET: int = 30
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._gas: int = _GAS_START
        self._banks_robbed: int = 0
        self._total_banks: int = 0
        self._dynamite_count: int = 3
        self._facing: tuple[int, int] = (1, 0)
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-bankheist-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._progress_count = 0
        return super()._reset(seed)

    def _task_description(self) -> str:
        return (
            "Rob banks ($) by walking over them. Avoid police (p). "
            "Reach the exit (E) before your gas runs out. "
            "Use FIRE to drop dynamite (d) that stuns nearby police."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Bank Heist.\n\n"
            "TASK\n"
            "Drive through a maze of streets, rob as many banks as you "
            "can, then escape through the exit before your gas runs out. "
            "Each level has more banks and more police.\n\n"
            "BOARD\n"
            "20x16 maze. Walls are '#', streets are ' ' (empty). Banks "
            "are '$', police cars are 'p', the exit is 'E', dynamite you "
            "drop is 'd'. Banks are placed in open cells.\n\n"
            "MECHANICS\n"
            "Move one cell per step in any cardinal direction; walls "
            "block you. Each step consumes 1 gas (starts at 200). "
            "Stepping on a bank robs it. Stepping on the exit clears the "
            "level. FIRE drops dynamite (limited to 3) that "
            "stuns all police within Manhattan distance 3 for 10 steps. "
            "Police move toward you with 40 percent probability per step "
            "and otherwise continue in their last direction.\n\n"
            "SCORING\n"
            "+2 reward for each bank robbed. Exiting adds a bonus of "
            "1 * (banks robbed this level). Getting caught by police "
            "gives -1 reward; running out of gas gives -1 reward.\n\n"
            "TERMINATION\n"
            "Single-life: getting caught by police or running out "
            "of gas ends the episode. The episode also ends after "
            "max_turns.\n\n"
            "HUD\n"
            "Shows score, level, gas remaining, dynamite count, "
            "and banks left this level.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "█": "wall",
            " ": "road",
            "$": "bank",
            "p": "police",
            "E": "exit",
            "d": "dynamite",
        }.get(ch, ch)

    def _generate_level(self, seed: int) -> None:
        rng = np.random.default_rng(seed + self._level * 1000)
        self._init_grid(_W, _H)
        self._entities = []
        self._banks_robbed = 0
        self._gas = _GAS_START
        self._dynamite_count = 3
        self._facing = (1, 0)

        # Build maze: walls with corridors
        for y in range(_H):
            for x in range(_W):
                self._set_cell(x, y, "█")

        # Carve corridors
        # Horizontal corridors
        h_rows = [1, 4, 7, 10, 13, _H - 2]
        for y in h_rows:
            for x in range(1, _W - 1):
                self._set_cell(x, y, " ")

        # Vertical corridors
        v_cols = [1, 5, 10, 14, _W - 2]
        for x in v_cols:
            for y in range(1, _H - 1):
                self._set_cell(x, y, " ")

        # Extra random connections
        n_extra = int(rng.integers(3, 7))
        for _ in range(n_extra):
            cx = int(rng.integers(2, _W - 2))
            cy = int(rng.integers(2, _H - 2))
            length = int(rng.integers(2, 5))
            if rng.random() < 0.5:
                for j in range(length):
                    if cx + j < _W - 1:
                        self._set_cell(cx + j, cy, " ")
            else:
                for j in range(length):
                    if cy + j < _H - 1:
                        self._set_cell(cx, cy + j, " ")

        # Place player at bottom-left
        self._player_x = 1
        self._player_y = _H - 2

        # Place exit at top-right
        self._set_cell(_W - 2, 1, _EXIT_CHAR)

        # Place banks
        n_banks = min(6, 2 + self._level)
        self._total_banks = 0
        for _ in range(n_banks):
            for _attempt in range(30):
                bx = int(rng.integers(2, _W - 3))
                by = int(rng.integers(2, _H - 3))
                if (
                    self._grid_at(bx, by) == " "
                    and abs(bx - self._player_x) + abs(by - self._player_y) > 3
                ):
                    self._set_cell(bx, by, _BANK_CHAR)
                    self._total_banks += 1
                    break

        # Place police
        n_police = min(4, 1 + self._level)
        for _ in range(n_police):
            for _attempt in range(30):
                px = int(rng.integers(3, _W - 3))
                py = int(rng.integers(2, _H - 3))
                if (
                    self._grid_at(px, py) == " "
                    and abs(px - self._player_x) + abs(py - self._player_y) > 4
                ):
                    cop = self._add_entity("police", _POLICE_CHAR, px, py)
                    cop.data["dir"] = (1, 0)
                    cop.data["stunned"] = 0
                    break

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Consume gas (Pattern D death penalty)
        self._gas -= 1
        if self._gas <= 0:
            self._message = "Out of gas!"
            self._on_life_lost()
            return self._DEATH_PENALTY, self._game_over, info

        # Parse action
        move_dir: tuple[int, int] | None = None
        fire = False
        if action_name in _DIRS:
            move_dir = _DIRS[action_name]
            self._facing = move_dir
            self._player_dir = move_dir
        elif action_name == "FIRE":
            fire = True

        # Move player
        if move_dir is not None:
            nx = self._player_x + move_dir[0]
            ny = self._player_y + move_dir[1]
            cell = self._grid_at(nx, ny)
            if not self._is_solid(nx, ny) or cell in (_EXIT_CHAR, _BANK_CHAR):
                self._player_x, self._player_y = nx, ny

        # Check cell content
        cell = self._grid_at(self._player_x, self._player_y)
        if cell == _BANK_CHAR:
            self._set_cell(self._player_x, self._player_y, " ")
            self._banks_robbed += 1
            self._on_point_scored(2)
            if self._progress_count < self._WIN_TARGET:
                reward += 1.0 / self._WIN_TARGET
                self._progress_count += 1
            self._message = "Bank robbed!"
            # Win check
            if self._progress_count >= self._WIN_TARGET:
                self._game_over = True
                info["won"] = True
                self._message = "All banks robbed!"
                return reward, self._game_over, info
        elif cell == _EXIT_CHAR:
            # Level complete (no bonus reward; cumulative is structural)
            self._level += 1
            self._message = "Escaped!"
            self._generate_level(self._level * 4231)
            info["level_cleared"] = True
            return reward, False, info

        # Drop dynamite
        if fire and self._dynamite_count > 0:
            self._dynamite_count -= 1
            # Stun nearby police
            for e in self._entities:
                if e.etype == "police" and e.alive:
                    dist = abs(e.x - self._player_x) + abs(e.y - self._player_y)
                    if dist <= 3:
                        e.data["stunned"] = 10

        # Move police
        for e in self._entities:
            if e.etype != "police" or not e.alive:
                continue
            if e.data.get("stunned", 0) > 0:
                e.data["stunned"] -= 1
                continue
            self._move_police(e)

        # Check police collision (Pattern D death penalty)
        for e in self._entities:
            if (
                e.etype == "police"
                and e.alive
                and e.x == self._player_x
                and e.y == self._player_y
                and e.data.get("stunned", 0) <= 0
            ):
                self._on_life_lost()
                reward = self._DEATH_PENALTY
                self._message = "Caught by police!"
                break

        info["gas"] = self._gas
        info["banks_robbed"] = self._banks_robbed
        info["dynamite"] = self._dynamite_count
        return reward, self._game_over, info

    def _move_police(self, cop: AtariEntity) -> None:
        """Police chase the player through corridors."""
        # Occasionally choose a direction toward player
        if self.rng.random() < 0.4:
            dx = 0
            dy = 0
            if self._player_x > cop.x:
                dx = 1
            elif self._player_x < cop.x:
                dx = -1
            if self._player_y > cop.y:
                dy = 1
            elif self._player_y < cop.y:
                dy = -1
            # Pick one axis
            if self.rng.random() < 0.5:
                dy = 0
            else:
                dx = 0
            nx, ny = cop.x + dx, cop.y + dy
            if not self._is_solid(nx, ny):
                cop.x, cop.y = nx, ny
                cop.data["dir"] = (dx, dy)
                return

        # Otherwise continue in current direction or pick random
        cur_dir = cop.data.get("dir", (1, 0))
        nx, ny = cop.x + cur_dir[0], cop.y + cur_dir[1]
        if not self._is_solid(nx, ny):
            cop.x, cop.y = nx, ny
        else:
            # Pick random open direction
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            self.rng.shuffle(dirs)
            for d in dirs:
                nx, ny = cop.x + d[0], cop.y + d[1]
                if not self._is_solid(nx, ny):
                    cop.x, cop.y = nx, ny
                    cop.data["dir"] = d
                    break

    def _render_current_observation(self) -> GridObservation:
        """Override to include gas and banks in HUD."""
        obs = super()._render_current_observation()
        remaining = self._total_banks - self._banks_robbed
        extra = (
            f"Gas: {self._gas}  "
            f"Dynamite: {self._dynamite_count}\n"
            f"Banks: {remaining}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid,
            legend=obs.legend,
            hud=new_hud,
            message=obs.message,
        )

    def _advance_entities(self) -> None:
        """Override: entities are moved in _game_step."""
        pass
