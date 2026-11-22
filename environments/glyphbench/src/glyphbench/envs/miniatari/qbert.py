"""miniatari Q*bert.

Identity: Hop diagonally on cubes of a 4-row pyramid; paint each cube once.
Win condition: paint all 10 cubes of the pyramid.
Reward: Pattern A, +1/10 per fresh cube painted.

Mechanic (B4 — one-way doors variant). At reset the apex (row=0,
col=0) is UNPAINTED. Standing on a cube does not paint it; only
HOPPING into a fresh cube earns the +1/10. The agent must therefore
plan a tour that returns to the apex at some point to claim that
tile. UP_* hops FROM the apex are no-ops (they cost the tick but
never end the run), removing the audit's "first-move death-trap"
bias. Off-pyramid hops in any other direction (down off the bottom,
or off the slanted edges) remain fatal — that is the discriminating
constraint that makes random play lose progress.

Gym ID: glyphbench/miniatari-qbert-v0
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniQbertEnv(MiniatariBase):
    """Mini Q*bert: 12x12 grid, 4-row pyramid (10 cubes total).

    Pyramid is laid out at logical coordinates (row, col) with row=0..3
    and col=0..row. Each cube maps to a grid cell using a fixed
    isometric-ish mapping: gx = 5 - row + 2*col, gy = 1 + 2*row. All 10
    cubes start UNPAINTED ('.'), including the apex. The start cube is
    seed-dependent and visible on reset; standing alone never paints
    anything. A cube paints (and grants +1/10) only when the agent
    actively HOPS onto it from another cube. Each tick the agent picks
    one of 4 diagonal directions: UP_LEFT, UP_RIGHT, DOWN_LEFT,
    DOWN_RIGHT, or NOOP. UP_* hops FROM the apex are no-ops (B4
    redesign: the first move is never structurally fatal). Any other
    off-pyramid hop — DOWN_* off the bottom row, or off the slanted
    edges — ends the run. To paint the apex the agent must hop down
    then return.
    Painting all 10 cubes wins (cumulative +1.0).
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT"),
        descriptions=(
            "do nothing",
            "hop diagonally up-left to the cube at (row-1, col-1)",
            "hop diagonally up-right to the cube at (row-1, col)",
            "hop diagonally down-left to the cube at (row+1, col)",
            "hop diagonally down-right to the cube at (row+1, col+1)",
        ),
    )

    default_max_turns = 50

    _WIDTH = 12
    _HEIGHT = 12
    _NUM_ROWS = 4
    _WIN_TARGET = 10  # 1+2+3+4 = 10
    _START_POSITIONS = ((0, 0), (1, 0), (1, 1), (2, 1))

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._row: int = 0
        self._col: int = 0
        # painted[row][col] for col in 0..row
        self._painted: list[list[bool]] = []
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-qbert-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._row, self._col = self._START_POSITIONS[seed % len(self._START_POSITIONS)]
        # All 10 cubes start unpainted including the apex. Standing on
        # the apex at reset does NOT auto-paint it — the agent must
        # actively HOP onto a cube to paint it. To paint the apex, the
        # agent must first hop down then return up.
        self._painted = [[False] * (r + 1) for r in range(self._NUM_ROWS)]
        self._sync_player_pos()

    def _grid_pos(self, row: int, col: int) -> tuple[int, int]:
        gx = 5 - row + 2 * col
        gy = 1 + 2 * row
        return gx, gy

    def _sync_player_pos(self) -> None:
        gx, gy = self._grid_pos(self._row, self._col)
        self._player_x = gx
        self._player_y = gy

    def _valid(self, row: int, col: int) -> bool:
        return 0 <= row < self._NUM_ROWS and 0 <= col <= row

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Compute hop destination.
        new_row, new_col = self._row, self._col
        if action_name == "UP_LEFT":
            new_row, new_col = self._row - 1, self._col - 1
        elif action_name == "UP_RIGHT":
            new_row, new_col = self._row - 1, self._col
        elif action_name == "DOWN_LEFT":
            new_row, new_col = self._row + 1, self._col
        elif action_name == "DOWN_RIGHT":
            new_row, new_col = self._row + 1, self._col + 1

        if action_name != "NOOP":
            if not self._valid(new_row, new_col):
                # Invalid hop is a no-op ONLY when it would have hopped
                # OFF the top of the pyramid (row < 0). Off the bottom
                # or sides remains fatal: the agent has to plan a real
                # tour rather than spam UP/DOWN. UP-from-apex stays
                # legal-no-op (B4: no fatal first move).
                if new_row < 0:
                    self._message = "No cube above — hop refused."
                else:
                    self._message = "Hopped off the pyramid!"
                    self._game_over = True
                    self._won = False
                    return reward, True, info
            else:
                self._row, self._col = new_row, new_col
                self._sync_player_pos()
                if not self._painted[self._row][self._col]:
                    self._painted[self._row][self._col] = True
                    self._progress += 1
                    reward += self._progress_reward(self._WIN_TARGET)
                    self._message = (
                        f"Cube painted! ({self._progress}/{self._WIN_TARGET})"
                    )
                    if self._progress >= self._WIN_TARGET:
                        self._on_won()
                        return reward, self._game_over, info
                else:
                    self._message = "Already painted."

        info["progress"] = self._progress
        info["row"] = self._row
        info["col"] = self._col
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        for r in range(self._NUM_ROWS):
            for c in range(r + 1):
                gx, gy = self._grid_pos(r, c)
                if 0 <= gx < self._WIDTH and 0 <= gy < self._HEIGHT:
                    grid[gy][gx] = "#" if self._painted[r][c] else "."
        # Player
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Q"

        symbols = {
            " ": "void",
            ".": "unpainted cube",
            "#": "painted cube",
            "Q": "you (Q*bert)",
        }
        unpainted = sum(
            1 for r in range(self._NUM_ROWS) for c in range(r + 1) if not self._painted[r][c]
        )
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Painted: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Unpainted: {unpainted}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Q*bert on a 12x12 grid. A 4-row pyramid has 10 cubes "
            "(. unpainted, # painted) at logical (row, col) positions "
            "with row=0..3 and col=0..row. You (Q) start on a visible, "
            "seed-selected cube near the top. Each tick choose one diagonal hop: "
            "UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT, or NOOP. The "
            "apex is unpainted at start; painting requires HOPPING "
            "onto an unpainted cube. Standing on the apex (NOOP) or "
            "revisiting an already-painted cube grants no reward. To "
            "paint the apex you must hop off it then return. UP_* "
            "hops from the apex are no-ops (you cannot hop above row "
            "0). Any OTHER off-pyramid hop is fatal — DOWN_* off the "
            "bottom row, or off the slanted edges, ENDS the run. "
            "Paint all 10 cubes to win. Reward: +1/10 per fresh cube "
            "painted (full clear sums to +1.0)."
        )
