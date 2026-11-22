"""AtariBase: shared base for all Atari game environments."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation


@dataclass
class AtariEntity:
    etype: str
    char: str
    x: int
    y: int
    dx: int = 0
    dy: int = 0
    alive: bool = True
    data: dict[str, Any] = field(default_factory=dict)


class AtariBase(BaseGlyphEnv):
    noop_action_name: str = "NOOP"

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._grid: list[list[str]] = []
        self._grid_w: int = 0
        self._grid_h: int = 0
        self._player_x: int = 0
        self._player_y: int = 0
        self._player_dir: tuple[int, int] = (0, 0)
        self._score: int = 0
        # GlyphBench uses a single-life model across the whole benchmark —
        # we deliberately drop the legacy ALE multi-life mechanic. Death
        # ends the episode.
        self._lives: int = 1
        self._level: int = 1
        self._game_over: bool = False
        self._entities: list[AtariEntity] = []
        self._message: str = ""

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

    def _add_entity(
        self, etype: str, char: str, x: int, y: int, **kw: Any
    ) -> AtariEntity:
        e = AtariEntity(etype=etype, char=char, x=x, y=y, **kw)
        self._entities.append(e)
        return e

    def _on_point_scored(self, delta: int) -> None:
        self._score += delta

    def _on_life_lost(self) -> None:
        # Single-life model — any life-loss event ends the episode.
        self._lives = 0
        self._game_over = True
        self._message = "Game Over!"

    def _advance_entities(self) -> None:
        for e in self._entities:
            if not e.alive:
                continue
            e.x += e.dx
            e.y += e.dy
            if (
                e.x < 0
                or e.x >= self._grid_w
                or e.y < 0
                or e.y >= self._grid_h
            ):
                e.alive = False
        self._entities = [e for e in self._entities if e.alive]

    def _reset(self, seed: int) -> GridObservation:
        self._score = 0
        self._lives = 1
        self._level = 1
        self._game_over = False
        self._entities = []
        self._message = ""
        self._player_dir = (0, 0)
        self._generate_level(seed)
        return self._render_current_observation()

    @abstractmethod
    def _generate_level(self, seed: int) -> None: ...

    @abstractmethod
    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]: ...

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        self._message = ""
        reward, terminated, info = self._game_step(name)
        self._advance_entities()
        if self._game_over:
            terminated = True
        info["score"] = self._score
        info["level"] = self._level
        return self._render_current_observation(), reward, terminated, False, info

    _DIR_CHARS: dict[tuple[int, int], str] = {
        (1, 0): "→", (-1, 0): "←",
        (0, -1): "↑", (0, 1): "↓", (0, 0): "@",
    }
    _DIR_NAMES: dict[tuple[int, int], str] = {
        (1, 0): "right", (-1, 0): "left",
        (0, -1): "up", (0, 1): "down", (0, 0): "none",
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
        if (
            0 <= self._player_x < self._grid_w
            and 0 <= self._player_y < self._grid_h
        ):
            pch = self._DIR_CHARS.get(
                self._player_dir, "@"
            )
            render[self._player_y][self._player_x] = pch
            dname = self._DIR_NAMES.get(
                self._player_dir, "none"
            )
            symbols[pch] = f"you (facing {dname})"
        # HUD shown to the model on every turn. Single-life model: no
        # Lives field. ``Step: T / N`` is required by the prompt
        # convention (see OBSERVATION_CONVENTIONS_BLOCK in
        # verifiers_integration/prompting.py).
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Score: {self._score}    "
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
            "█": "wall",
            "│": "wall",
            "─": "wall",
            "┼": "corner",
            " ": "empty",
            "·": "pellet",
            "=": "ground",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return "Play the game and maximize your score."

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            f"TASK\n{self._task_description()}\n\n"
            + self.action_spec.render_for_prompt()
        )
