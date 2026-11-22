"""Downhill Skiing -- dodge obstacles and collect flag gates.

Gym IDs:
  glyphbench/classics-ski-v0
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

WIDTH = 15
HEIGHT = 20
SKIER_ROW = HEIGHT - 3  # row 17 (near bottom)
TOTAL_TERRAIN_ROWS = 100
OBSTACLE_DENSITY = 0.15

SKI_ACTION_SPEC = ActionSpec(
    names=("LEFT", "RIGHT", "NOOP"),
    descriptions=(
        "move skier one cell left",
        "move skier one cell right",
        "go straight down",
    ),
)

SYM_SKIER = "\u2193"   # ↓
SYM_TREE = "\u2663"    # ♣
SYM_ROCK = "\u25c6"    # ◆
SYM_FLAG = "\u2691"    # ⚑
SYM_SNOW = "\u00b7"    # ·
SYM_TRAIL = "\u2591"   # ░

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class SkiEnv(BaseGlyphEnv):
    """Downhill skiing: dodge trees/rocks, pass through flag gates."""

    action_spec = SKI_ACTION_SPEC
    noop_action_name: str = "NOOP"

    def __init__(self, max_turns: int = TOTAL_TERRAIN_ROWS) -> None:
        super().__init__(max_turns=max_turns)
        self._skier_col: int = WIDTH // 2
        self._terrain: list[list[str]] = []  # pre-generated terrain rows
        self._scroll_offset: int = 0  # how many rows have scrolled
        self._alive: bool = True
        self._score: float = 0.0
        self._flags_passed: int = 0
        self._target_gates: int = 1
        self._gate_ranges: dict[int, tuple[int, int]] = {}
        self._prev_col: int = WIDTH // 2  # for trail

    def env_id(self) -> str:
        return "glyphbench/classics-ski-v0"

    def _generate_terrain(self) -> list[list[str]]:
        """Pre-generate all terrain rows. Row 0 = first to appear at bottom."""
        rows: list[list[str]] = []
        self._gate_ranges = {}
        # The first collision/scoring row is SKIER_ROW + 1 after one step, so
        # start gates there or later rather than showing unreachable gates in
        # the initial viewport.
        next_gate = SKIER_ROW + 1 + int(self.rng.integers(0, 4))

        for i in range(TOTAL_TERRAIN_ROWS + HEIGHT):
            row = [SYM_SNOW] * WIDTH
            if i == next_gate:
                # Place flag gate: two flags with a gap between them
                gap_center = int(self.rng.integers(3, WIDTH - 3))
                gap_width = int(self.rng.integers(3, 6))
                left_flag = max(0, gap_center - gap_width // 2)
                right_flag = min(WIDTH - 1, gap_center + gap_width // 2)
                row[left_flag] = SYM_FLAG
                row[right_flag] = SYM_FLAG
                self._gate_ranges[i] = (left_flag, right_flag)
                next_gate = i + int(self.rng.integers(5, 9))
            else:
                # Scatter obstacles
                for col in range(WIDTH):
                    if self.rng.random() < OBSTACLE_DENSITY:
                        row[col] = SYM_TREE if self.rng.random() < 0.6 else SYM_ROCK
            rows.append(row)
        reachable_rows = range(SKIER_ROW + 1, SKIER_ROW + TOTAL_TERRAIN_ROWS + 1)
        self._target_gates = max(1, sum(1 for row in reachable_rows if row in self._gate_ranges))
        return rows

    def _reset(self, seed: int) -> GridObservation:
        self._terrain = self._generate_terrain()
        self._skier_col = WIDTH // 2
        self._prev_col = WIDTH // 2
        self._scroll_offset = 0
        self._alive = True
        self._score = 0.0
        self._flags_passed = 0
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]

        self._prev_col = self._skier_col

        if name == "LEFT":
            self._skier_col = max(0, self._skier_col - 1)
        elif name == "RIGHT":
            self._skier_col = min(WIDTH - 1, self._skier_col + 1)

        # Scroll terrain: advance one row
        self._scroll_offset += 1

        # Check what's at the skier's position in the new terrain
        terrain_row_idx = self._scroll_offset + SKIER_ROW
        reward = 0.0

        if terrain_row_idx < len(self._terrain):
            cell = self._terrain[terrain_row_idx][self._skier_col]
            if cell in (SYM_TREE, SYM_ROCK):
                self._alive = False
                return self._render_current_observation(), -1.0, True, False, info
            if (
                terrain_row_idx in self._gate_ranges
                and self._gate_ranges[terrain_row_idx][0]
                <= self._skier_col
                <= self._gate_ranges[terrain_row_idx][1]
            ):
                # Passing through a flag gate. Pattern A: each reachable gate
                # yields an equal slice, capped against remaining headroom so
                # cumulative reward never exceeds 1.0.
                self._flags_passed += 1
                already = (self._flags_passed - 1) / self._target_gates
                room = max(0.0, 1.0 - already)
                reward = min(1.0 / self._target_gates, room)
                self._score += reward

        # Check if terrain completed
        if self._scroll_offset >= TOTAL_TERRAIN_ROWS:
            info["flags_passed"] = self._flags_passed
            return self._render_current_observation(), reward, True, False, info

        info["flags_passed"] = self._flags_passed
        return self._render_current_observation(), reward, False, False, info

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(WIDTH, HEIGHT, SYM_SNOW)

        # Fill visible terrain
        for screen_row in range(HEIGHT):
            terrain_idx = self._scroll_offset + screen_row
            if 0 <= terrain_idx < len(self._terrain):
                for col in range(WIDTH):
                    grid[screen_row][col] = self._terrain[terrain_idx][col]
            # else stays as snow

        # Place trail marker at previous position
        if 0 <= self._prev_col < WIDTH and grid[SKIER_ROW][self._prev_col] == SYM_SNOW:
            grid[SKIER_ROW][self._prev_col] = SYM_TRAIL

        # Place skier
        if 0 <= self._skier_col < WIDTH:
            grid[SKIER_ROW][self._skier_col] = SYM_SKIER

        legend = build_legend({
            SYM_SKIER: "skier (you)",
            SYM_TREE: "tree (obstacle)",
            SYM_ROCK: "rock (obstacle)",
            SYM_FLAG: "flag gate (pass through for points)",
            SYM_SNOW: "snow (safe)",
            SYM_TRAIL: "your trail",
        })

        rows_left = max(0, TOTAL_TERRAIN_ROWS - self._scroll_offset)
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Score: {self._score:.2f}    "
            f"Flags: {self._flags_passed} / {self._target_gates}    "
            f"Rows left: {rows_left}"
        )

        msg = ""
        if not self._alive:
            msg = "Crashed into an obstacle!"
        elif self._scroll_offset >= TOTAL_TERRAIN_ROWS:
            msg = f"Run complete! Final score: {self._score:.2f}"

        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message=msg,
        )

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Ski downhill, dodging trees and rocks while passing through flag gates.\n\n"
            "RULES\n"
            f"- The grid is {WIDTH} wide x {HEIGHT} tall. You see a scrolling window.\n"
            f"- Your skier (\u2193) is fixed at row {SKIER_ROW} near the bottom.\n"
            "- Terrain scrolls upward 1 row per step (you move downhill automatically).\n"
            "- Trees (\u2663) and rocks (\u25c6) are obstacles: hitting one ends the game (-1).\n"
            "- Flag gates (\u2691) appear every 5-8 rows: pass between the two flags\n"
            "  to earn reward. The HUD shows passed gates / reachable gates;\n"
            "  passing every reachable gate gives cumulative reward = 1.0.\n"
            f"- Complete {TOTAL_TERRAIN_ROWS} rows of terrain to finish.\n"
            "- Move LEFT or RIGHT to dodge obstacles and aim for gates.\n"
            "- Look ahead at upcoming terrain to plan your path!\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
