"""Classic Frogger game.

Cross lanes of traffic and river to reach the safe zone at the top.

Gym IDs:
  glyphbench/classics-frogger-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIDTH = 13
HEIGHT = 10

FROGGER_ACTION_SPEC = ActionSpec(
    names=("UP", "DOWN", "LEFT", "RIGHT", "WAIT"),
    descriptions=(
        "move frog up one row",
        "move frog down one row",
        "move frog left one cell",
        "move frog right one cell",
        "stay in place for one step",
    ),
)

SYM_FROG = "@"
SYM_LOG = "\u25ac"    # ▬
SYM_CAR = "\u2588"    # █
SYM_WATER = "\u2248"  # ≈
SYM_ROAD = "\u2500"   # ─
SYM_SAFE = "\u00b7"   # ·
SYM_GOAL = "\u2605"   # ★

_DIR_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "WAIT": (0, 0),
}

# Lane configuration: rows 1-8 are hazard lanes (row 0 = goal, row 9 = start)
# Each lane: (type, direction, speed, object_pattern)
#   type is "road" or "river"
# direction: 1 = moving right, -1 = moving left
# speed: cells to shift per step
# object_pattern: list of booleans for initial placement (True = car/log present)


def _make_lane_configs() -> list[dict[str, Any]]:
    """Return lane configs for rows 1..8 (index 0 = row 1)."""
    return [
        # Row 1 (river, logs moving right)
        {"type": "river", "dir": 1, "length": 3, "gap": 3},
        # Row 2 (river, logs moving left)
        {"type": "river", "dir": -1, "length": 4, "gap": 2},
        # Row 3 (river, logs moving right)
        {"type": "river", "dir": 1, "length": 2, "gap": 4},
        # Row 4 (safe middle row)
        {"type": "safe"},
        # Row 5 (road, cars moving left)
        {"type": "road", "dir": -1, "length": 1, "gap": 3},
        # Row 6 (road, cars moving right)
        {"type": "road", "dir": 1, "length": 2, "gap": 3},
        # Row 7 (road, cars moving left)
        {"type": "road", "dir": -1, "length": 1, "gap": 2},
        # Row 8 (road, cars moving right)
        {"type": "road", "dir": 1, "length": 1, "gap": 4},
    ]


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class FroggerEnv(BaseGlyphEnv):
    """Frogger: cross roads and rivers to reach the top."""

    action_spec = FROGGER_ACTION_SPEC
    noop_action_name: str = "WAIT"

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)
        self._frog_x: int = 0
        self._frog_y: int = 0
        # Each lane stores a list of booleans of length WIDTH, True = object present
        self._lanes: list[list[bool]] = []
        self._lane_configs: list[dict[str, Any]] = []
        self._alive: bool = True
        self._won: bool = False

    def env_id(self) -> str:
        return "glyphbench/classics-frogger-v0"

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._frog_x = WIDTH // 2
        self._frog_y = HEIGHT - 1  # bottom row
        self._alive = True
        self._won = False
        self._lane_configs = _make_lane_configs()

        # Initialize lane object arrays
        self._lanes = []
        for cfg in self._lane_configs:
            if cfg["type"] == "safe":
                self._lanes.append([False] * WIDTH)
            else:
                lane = [False] * WIDTH
                length = cfg["length"]
                gap = cfg["gap"]
                pattern_len = length + gap
                # Randomize offset for each lane
                offset = int(self.rng.integers(0, WIDTH))
                for i in range(WIDTH):
                    pos_in_pattern = (i + offset) % pattern_len
                    lane[i] = pos_in_pattern < length
                self._lanes.append(lane)

        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]

        # Move frog
        dx, dy = _DIR_DELTAS[name]
        nx = self._frog_x + dx
        ny = self._frog_y + dy

        # Clamp to grid
        nx = max(0, min(WIDTH - 1, nx))
        ny = max(0, min(HEIGHT - 1, ny))

        self._frog_x = nx
        self._frog_y = ny

        # Scroll all lanes
        for i, cfg in enumerate(self._lane_configs):
            if cfg["type"] == "safe":
                continue
            direction = cfg["dir"]
            old_lane = self._lanes[i]
            new_lane = [False] * WIDTH
            for x in range(WIDTH):
                src = (x - direction) % WIDTH
                new_lane[x] = old_lane[src]
            self._lanes[i] = new_lane

            # If frog is on a river lane, it rides the log (moves with it)
            lane_row = i + 1  # lanes correspond to rows 1..8
            if cfg["type"] == "river" and self._frog_y == lane_row:
                self._frog_x = (self._frog_x + direction) % WIDTH

        # Check win condition
        if self._frog_y == 0:
            self._won = True
            return self._render_current_observation(), 1.0, True, False, info

        # Check collision with lane objects
        if 1 <= self._frog_y <= 8:
            lane_idx = self._frog_y - 1
            cfg = self._lane_configs[lane_idx]
            on_object = self._lanes[lane_idx][self._frog_x]

            if cfg["type"] == "road" and on_object:
                # Hit by car
                self._alive = False
                return self._render_current_observation(), -1.0, True, False, info
            elif cfg["type"] == "river" and not on_object:
                # Fell in water (not on a log)
                self._alive = False
                return self._render_current_observation(), -1.0, True, False, info

        return self._render_current_observation(), 0.0, False, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(WIDTH, HEIGHT, SYM_SAFE)

        # Row 0: goal row
        for x in range(WIDTH):
            grid[0][x] = SYM_GOAL

        # Rows 1-8: lanes
        for i, cfg in enumerate(self._lane_configs):
            row = i + 1
            if cfg["type"] == "safe":
                for x in range(WIDTH):
                    grid[row][x] = SYM_SAFE
            elif cfg["type"] == "road":
                for x in range(WIDTH):
                    grid[row][x] = SYM_CAR if self._lanes[i][x] else SYM_ROAD
            elif cfg["type"] == "river":
                for x in range(WIDTH):
                    grid[row][x] = SYM_LOG if self._lanes[i][x] else SYM_WATER

        # Row 9: start (safe zone) — already filled with SYM_SAFE

        # Stamp frog
        grid[self._frog_y][self._frog_x] = SYM_FROG

        legend = build_legend({
            SYM_FROG: "frog (you)",
            SYM_LOG: "log (safe on river)",
            SYM_CAR: "car (deadly)",
            SYM_WATER: "water (deadly without log)",
            SYM_ROAD: "road (safe if no car)",
            SYM_SAFE: "safe zone",
            SYM_GOAL: "goal (reach here to win)",
        })

        hud = f"Step: {self._turn} / {self.max_turns}"

        msg = ""
        if self._won:
            msg = "You reached the goal! You win!"
        elif not self._alive:
            msg = "You died!"

        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message=msg)

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Guide a frog from the bottom of the screen to the goal at the top.\n\n"
            "RULES\n"
            f"- The grid is {WIDTH}x{HEIGHT}.\n"
            "- Top three lanes are rivers, row 4 is a safe rest zone, and bottom four lanes are roads.\n"
            "- Road lanes have cars that scroll horizontally. Touching a car kills you (-1 reward).\n"
            "- River lanes have logs that scroll horizontally. You must land on a log to cross.\n"
            "  Falling into water (not on a log) kills you (-1 reward).\n"
            "- On a river lane, you ride the log and move with it each step.\n"
            "- Reaching the goal row gives +1 reward and ends the episode.\n"
            "- Cars and logs scroll one cell per step and wrap horizontally.\n"
            "- Your own left/right moves are clamped to the grid; you cannot step off either edge.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
