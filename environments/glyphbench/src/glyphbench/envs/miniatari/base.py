"""MiniatariBase: shared base for short-horizon miniatari arcade envs.

Mirrors atari/base.py with two structural changes:
  - default max_turns is 200 (vs 10000 in the originals)
  - reward helpers force [-1, 1]-compliant per-step rewards by construction

Design contract per env (from spec §1):
  - smaller play field (12x8 to 16x16)
  - tight terminal win condition (clear N bricks, first to K, etc.)
  - max_turns in [100, 500]
  - reward = +1/N per progress unit (Pattern A) or
            +1/W per agent point, -1/W per opponent point (Pattern C) or
            structural milestones summing to 1.0 (Pattern B) or
            progress + terminal -1 on death (Pattern D)
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation


@dataclass
class MiniatariEntity:
    etype: str
    char: str
    x: int
    y: int
    dx: int = 0
    dy: int = 0
    alive: bool = True
    data: dict[str, Any] = field(default_factory=dict)


class MiniatariBase(BaseGlyphEnv):
    """Base class for miniatari games.

    Subclasses MUST implement _generate_level(seed) and _game_step(action_name).
    """

    noop_action_name: str = "NOOP"
    default_max_turns: int = 200  # subclasses may override

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns or self.default_max_turns)
        self._grid: list[list[str]] = []
        self._grid_w: int = 0
        self._grid_h: int = 0
        self._player_x: int = 0
        self._player_y: int = 0
        self._player_dir: tuple[int, int] = (0, 0)
        self._score: float = 0.0  # cumulative episodic return so far (for HUD)
        self._lives: int = 1
        self._level: int = 1
        self._game_over: bool = False
        self._won: bool = False
        self._entities: list[MiniatariEntity] = []
        self._message: str = ""

    # -- grid helpers ------------------------------------------------------
    def _init_grid(self, width: int, height: int, fill: str = " ") -> None:
        self._grid_w = width
        self._grid_h = height
        self._grid = [[fill for _ in range(width)] for _ in range(height)]

    def _set_cell(self, x: int, y: int, ch: str) -> None:
        if 0 <= x < self._grid_w and 0 <= y < self._grid_h:
            self._grid[y][x] = ch

    def _grid_at(self, x: int, y: int) -> str:
        if 0 <= x < self._grid_w and 0 <= y < self._grid_h:
            return self._grid[y][x]
        return "█"

    def _is_solid(self, x: int, y: int) -> bool:
        return self._grid_at(x, y) in ("█", "│", "─", "┼", "=")

    def _add_entity(self, etype: str, char: str, x: int, y: int, **kw: Any) -> MiniatariEntity:
        e = MiniatariEntity(etype=etype, char=char, x=x, y=y, **kw)
        self._entities.append(e)
        return e

    def _advance_entities(self) -> None:
        for e in self._entities:
            if not e.alive:
                continue
            e.x += e.dx
            e.y += e.dy
            if e.x < 0 or e.x >= self._grid_w or e.y < 0 or e.y >= self._grid_h:
                e.alive = False
        self._entities = [e for e in self._entities if e.alive]

    # -- reward helpers (enforce [-1, 1] structurally) ---------------------
    def _progress_reward(self, total_units: int) -> float:
        """Pattern A: +1/total_units per progress unit."""
        return 1.0 / max(1, total_units)

    def _agent_score_reward(self, target_score: int) -> float:
        """Pattern C: +1/target_score for an agent point."""
        return 1.0 / max(1, target_score)

    def _opp_score_reward(self, target_score: int) -> float:
        """Pattern C: -1/target_score for an opponent point."""
        return -1.0 / max(1, target_score)

    def _death_reward(self) -> float:
        """Pattern D: terminal -1.0 on death/loss."""
        return -1.0

    def _milestone_reward(self, share: float) -> float:
        """Pattern B: milestone share of the cumulative 1.0 budget.

        For multi-stage games where progress units have different weights;
        the implementer is responsible for ensuring all milestone shares
        sum to <= 1.0 across an episode.
        """
        return float(share)

    # -- lifecycle ----------------------------------------------------------
    def _on_life_lost(self) -> None:
        """Subclasses call this on death; sets game_over and won=False."""
        self._lives = 0
        self._game_over = True
        self._won = False
        self._message = "Game Over"

    def _on_won(self) -> None:
        """Subclasses call this when the win condition is reached."""
        self._game_over = True
        self._won = True
        self._message = "You Won!"

    def _reset(self, seed: int) -> GridObservation:
        self._score = 0.0
        self._lives = 1
        self._level = 1
        self._game_over = False
        self._won = False
        self._entities = []
        self._message = ""
        self._player_dir = (0, 0)
        self._generate_level(seed)
        return self._render_current_observation()

    @abstractmethod
    def _generate_level(self, seed: int) -> None: ...

    @abstractmethod
    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        """Returns (reward, terminated, info). Reward must be in a range
        such that the cumulative episodic return is in [-1, 1]."""

    def _step(self, action: int) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        self._message = ""
        reward, terminated, info = self._game_step(name)
        self._advance_entities()
        if self._game_over:
            terminated = True
        self._score += float(reward)
        info["score"] = self._score
        info["level"] = self._level
        info["won"] = self._won
        return self._render_current_observation(), reward, terminated, False, info

    # -- rendering ----------------------------------------------------------
    _DIR_CHARS: dict[tuple[int, int], str] = {
        (1, 0): "→", (-1, 0): "←", (0, -1): "↑", (0, 1): "↓", (0, 0): "@",
    }
    _DIR_NAMES: dict[tuple[int, int], str] = {
        (1, 0): "right", (-1, 0): "left", (0, -1): "up", (0, 1): "down", (0, 0): "none",
    }

    def _render_current_observation(self) -> GridObservation:
        render = [row[:] for row in self._grid]
        symbols: dict[str, str] = {}
        for y in range(self._grid_h):
            for x in range(self._grid_w):
                ch = render[y][x]
                if ch not in symbols:
                    symbols[ch] = self._symbol_meaning(ch)
        for e in self._entities:
            if e.alive and 0 <= e.x < self._grid_w and 0 <= e.y < self._grid_h:
                render[e.y][e.x] = e.char
                if e.char not in symbols:
                    symbols[e.char] = e.etype
        if 0 <= self._player_x < self._grid_w and 0 <= self._player_y < self._grid_h:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            render[self._player_y][self._player_x] = pch
            dname = self._DIR_NAMES.get(self._player_dir, "none")
            symbols[pch] = f"you (facing {dname})"
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Score: {self._score:.2f}    "
            f"Level: {self._level}"
        )
        return GridObservation(
            grid=grid_to_string(render),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "█": "wall", "│": "wall", "─": "wall", "┼": "corner",
            " ": "empty", "·": "pellet", "=": "ground",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return "Reach the win condition before time runs out."

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            f"TASK\n{self._task_description()}\n\n"
            + self.action_spec.render_for_prompt()
        )
