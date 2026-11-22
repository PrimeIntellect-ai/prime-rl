"""miniatari Skiing.

Identity: Slalom skier descending a course; pass through 10 gates to win.
Win condition: clear all 10 gates before the course ends.
Reward: Pattern A, +1/10 per gate cleared by passing between its flags.
Loss: course ends (last gate scrolls past the skier row) without all 10
gates cleared — episode terminates with whatever partial credit was
earned. No -1 penalty (Pattern A), so cumulative is always in [0, 1].

Gym ID: glyphbench/miniatari-skiing-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=200, mean_return=+0.157
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniSkiingEnv(MiniatariBase):
    """Mini Skiing: 14-wide x 16-tall slope, 10 gates from top to bottom.

    Skier (Y) sits at a fixed row near the top while the slope scrolls
    upward (i.e. the world rolls past). Each tick, all gate flags advance
    one row toward the skier; if a gate's flag-row matches the skier row,
    we check whether the skier is between the two flags. If yes, +1/10
    and the gate is consumed; if no, the gate scrolls past unscored.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT"),
        descriptions=(
            "ski straight (no edging)",
            "edge left (drift left one cell)",
            "edge right (drift right one cell)",
        ),
    )

    default_max_turns = 200

    _WIDTH = 14
    _HEIGHT = 16
    _SKIER_Y = 4
    _WIN_TARGET = 10
    _GATE_WIDTH = 4  # gap between flags (number of cells between them, exclusive)
    # Spacing in rows between successive gates. Skier moves at most 1
    # column/tick, so spacing must be wide enough that the skier can
    # cross the field between gates. 5 rows allows up to 5 column moves
    # between gates (more than the field width), making every random
    # gate sequence theoretically clearable.
    _GATE_SPACING = 5

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._gates: list[list[int]] = []  # per gate: [left_x, right_x, y, scored]
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-skiing-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._gates = []
        self._player_x = self._WIDTH // 2
        self._player_y = self._SKIER_Y
        rng = self.rng
        # Place gates at decreasing y values (so they spawn off-screen below
        # the skier and scroll upward). First gate just below skier.
        y = self._SKIER_Y + self._GATE_SPACING
        for _ in range(self._WIN_TARGET):
            # Random gate position: leftmost flag x in [0, _WIDTH-1-_GATE_WIDTH]
            lx = int(rng.integers(0, self._WIDTH - self._GATE_WIDTH))
            rx = lx + self._GATE_WIDTH - 1
            self._gates.append([lx, rx, y, 0])  # scored=0
            y += self._GATE_SPACING

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # 1. Skier edge
        if action_name == "LEFT" and self._player_x > 0:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 1:
            self._player_x += 1
            self._player_dir = (1, 0)

        # 2. Scroll: every gate moves up by 1
        for g in self._gates:
            g[2] -= 1

        # 3. Score gates that pass the skier row
        passed: list[int] = []
        for i, (lx, rx, y, scored) in enumerate(self._gates):
            if y == self._SKIER_Y and not scored:
                # Check if skier is between flags (exclusive)
                if lx < self._player_x < rx:
                    self._gates[i][3] = 1
                    reward += self._progress_reward(self._WIN_TARGET)
                    self._progress += 1
                    self._message = f"Gate cleared! ({self._progress}/{self._WIN_TARGET})"
                    if self._progress >= self._WIN_TARGET:
                        self._on_won()
                        return reward, self._game_over, info
                else:
                    self._message = "Missed gate."
            if y < 0:
                passed.append(i)

        # Remove gates that scrolled off screen
        self._gates = [g for i, g in enumerate(self._gates) if i not in passed]

        # 4. Course-end terminal: when the last gate has scrolled past the
        # skier row, the run is over. Without this the agent would spin
        # idle until max_turns truncation. Pattern A: no -1 penalty.
        if not self._gates and self._progress < self._WIN_TARGET:
            self._message = (
                f"Course ended — only {self._progress}/{self._WIN_TARGET} "
                "gates cleared."
            )
            self._game_over = True
            self._won = False
            return reward, True, info

        info["progress"] = self._progress
        info["gates_remaining"] = len(self._gates)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Skier row marker
        for x in range(self._WIDTH):
            grid[self._SKIER_Y][x] = "·"
        # Gates
        for lx, rx, y, scored in self._gates:
            if 0 <= y < self._HEIGHT:
                ch = "g" if scored else "F"
                if 0 <= lx < self._WIDTH:
                    grid[y][lx] = ch
                if 0 <= rx < self._WIDTH:
                    grid[y][rx] = ch
        # Skier
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "snow",
            "·": "skier line",
            "F": "active gate flag",
            "g": "scored gate flag",
            "Y": "you (skier)",
        }
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Gates cleared: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Skiing: a 14x16 slalom course. The skier (Y) sits at row "
            f"{self._SKIER_Y}, fixed; the slope scrolls upward (gate flag "
            "rows decrement by 1 each tick). 10 gates lie down the course, "
            f"each two flags ({self._GATE_WIDTH} cells apart) spaced "
            f"{self._GATE_SPACING} rows apart. LEFT/RIGHT edges your "
            "skier one cell. When a gate's flag row reaches your row, you "
            "score +1/10 if your column is strictly between the two flags "
            "(F). Otherwise the gate scrolls past unscored. The course "
            "ends when the last gate has scrolled past your row (~54 "
            "ticks). Clearing all 10 gates wins instantly; otherwise the "
            "course-end terminates the run with whatever partial credit "
            "you earned. Reward: +1/10 per cleared gate (no penalty)."
        )
