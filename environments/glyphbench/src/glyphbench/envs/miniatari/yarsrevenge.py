"""miniatari Yars' Revenge.

Identity: Two-phase win — break a 5-block shield, then strike the
Qotile when it appears.
Win condition: break the shield AND hit Qotile.
Reward: Pattern B milestones, +0.1 per shield block destroyed (5 total
= +0.5), +0.5 on Qotile hit. Total = +1.0 on full win. -1.0 on death.

Gym ID: glyphbench/miniatari-yarsrevenge-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=213, mean_return=-0.227
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniYarsRevengeEnv(MiniatariBase):
    """Mini Yars' Revenge: 14x10 arena; 5-block shield + Qotile.

    Right-side wall: column 11 holds 5 shield blocks (B) at rows 3-7.
    Behind the shield, Qotile (Q) sits at column 13 row 5. The agent
    (Y, arrow facing right) starts at the left at row 5.

    Phase 1 (shield): FIRE shoots a bullet from your facing direction.
    Each block hit = +0.1. After all 5 are gone, the shield is open.
    Phase 2 (Qotile): with the shield gone, FIRE that hits Qotile gives
    +0.5 and wins. Qotile fires a destroyer beam: every 6 ticks, a beam
    (b) launches from Qotile traveling left 1 cell/tick along row 5.
    If the beam hits the player, -1 terminal.
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
    _SHIELD_COL = 11
    _SHIELD_ROWS = (3, 4, 5, 6, 7)
    _QOTILE_X = 13
    _QOTILE_Y = 5
    _BEAM_EVERY = 6
    _BULLET_RANGE = 14  # full width is enough

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._blocks_alive: set[int] = set()  # row indices alive
        self._beams: list[list[int]] = []
        self._tick_count: int = 0
        self._shield_kills: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-yarsrevenge-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._tick_count = 0
        self._shield_kills = 0
        self._beams = []
        self._blocks_alive = set(self._SHIELD_ROWS)
        self._player_x = 1
        self._player_y = self._QOTILE_Y
        self._player_dir = (1, 0)

    def _shield_intact(self) -> bool:
        return len(self._blocks_alive) > 0

    def _bullet_hits(self, dx: int, dy: int) -> tuple[str, int] | None:
        """Trace a bullet, return ('block', row) or ('qotile', 0) or None."""
        bx, by = self._player_x + dx, self._player_y + dy
        for _ in range(self._BULLET_RANGE):
            if bx < 0 or bx >= self._WIDTH or by < 0 or by >= self._HEIGHT:
                return None
            # Block check
            if bx == self._SHIELD_COL and by in self._blocks_alive:
                return ("block", by)
            # Qotile check (only reachable if shield gone in this column path)
            if bx == self._QOTILE_X and by == self._QOTILE_Y:
                # If shield blocks the path on the same row at col 11 -> hit block first
                if by in self._blocks_alive:
                    return ("block", by)
                return ("qotile", 0)
            bx += dx
            by += dy
        return None

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
        # Player can't enter shield col when blocks present at that row,
        # nor Qotile cell.
        blocked = False
        if nx == self._SHIELD_COL and ny in self._blocks_alive:
            blocked = True
        if nx == self._QOTILE_X and ny == self._QOTILE_Y:
            blocked = True
        if not blocked:
            self._player_x, self._player_y = nx, ny

        # 2. Fire
        if action_name == "FIRE":
            bdx, bdy = self._player_dir
            if bdx == 0 and bdy == 0:
                bdx = 1
            hit = self._bullet_hits(bdx, bdy)
            if hit is not None:
                kind, row = hit
                if kind == "block":
                    self._blocks_alive.discard(row)
                    self._shield_kills += 1
                    reward += self._milestone_reward(0.1)
                    self._message = f"Shield block down! ({self._shield_kills}/5)"
                elif kind == "qotile":
                    if not self._shield_intact():
                        reward += self._milestone_reward(0.5)
                        self._message = "Qotile struck! Victory!"
                        self._on_won()
                        return reward, self._game_over, info

        # 3. Qotile fires destroyer beam every K ticks (only along row 5
        # where Qotile sits). Beam travels left 1 cell/tick.
        if self._tick_count % self._BEAM_EVERY == 0:
            self._beams.append([self._QOTILE_X - 1, self._QOTILE_Y])

        # 4. Move beams
        new_beams: list[list[int]] = []
        for bx, by in self._beams:
            if bx == self._player_x and by == self._player_y:
                self._message = "Hit by Qotile's destroyer beam!"
                reward += self._death_reward()
                self._on_life_lost()
                return reward, True, info
            # Beam blocked by intact shield in row by at col SHIELD_COL
            if bx == self._SHIELD_COL and by in self._blocks_alive:
                continue  # absorbed
            nbx = bx - 1
            if nbx < 0:
                continue
            new_beams.append([nbx, by])
        self._beams = new_beams

        info["shield_kills"] = self._shield_kills
        info["shield_intact"] = self._shield_intact()
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Shield blocks
        for r in self._blocks_alive:
            grid[r][self._SHIELD_COL] = "B"
        # Qotile
        grid[self._QOTILE_Y][self._QOTILE_X] = "Q"
        # Beams
        for bx, by in self._beams:
            if 0 <= bx < self._WIDTH and 0 <= by < self._HEIGHT:
                grid[by][bx] = "b"
        # Player
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "void",
            "B": "shield block",
            "Q": "Qotile",
            "b": "destroyer beam",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'right')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Shield: {5 - self._shield_kills}/5 left    "
            f"Score: {self._score:.3f}    "
            f"Beams: {len(self._beams)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Yars' Revenge on a 14x10 arena. A 5-block shield (B) "
            "stands at column 11, rows 3-7. Behind it, Qotile (Q) sits at "
            "(13, 5). You (Y, arrow facing) start at the left at row 5. "
            "LEFT/RIGHT/UP/DOWN moves you 1 cell and sets facing. FIRE "
            "shoots a bullet straight in your facing direction. Hitting a "
            "shield block destroys it (+0.1 each). After all 5 blocks are "
            "gone, shooting Qotile (with line of sight along your row) "
            "wins (+0.5). Every 6 ticks Qotile fires a destroyer beam (b) "
            "from its cell traveling left along row 5; intact shield "
            "blocks absorb beams. Being hit by a beam is -1 terminal."
        )
