"""Warehouse package delivery game.

Pick up colored packages and deliver them to matching destinations.

Gym IDs:
  glyphbench/classics-warehouse-v0
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

GRID_SIZE = 10

# Total packages -- used for [-1, 1] reward normalization.
_NUM_PACKAGES = 3

WAREHOUSE_ACTION_SPEC = ActionSpec(
    names=("UP", "DOWN", "LEFT", "RIGHT", "PICKUP", "DROP"),
    descriptions=(
        "move robot up one cell",
        "move robot down one cell",
        "move robot left one cell",
        "move robot right one cell",
        "pick up a package at your position",
        "drop the carried package at your position",
    ),
)

SYM_ROBOT = "@"
SYM_WALL = "\u2588"   # █
SYM_FLOOR = "\u00b7"  # ·

# Package/destination pairs: (package_sym, dest_sym, color_name)
PACKAGE_TYPES = [
    ("\u25cf", "\u25cb", "red"),     # ● ○
    ("\u25c6", "\u25c7", "blue"),    # ◆ ◇
    ("\u25a0", "\u25a1", "green"),   # ■ □
]

_DIR_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class WarehouseEnv(BaseGlyphEnv):
    """Warehouse: pick up packages and deliver to matching destinations."""

    action_spec = WAREHOUSE_ACTION_SPEC
    noop_action_name: str = "PICKUP"

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._robot_pos: tuple[int, int] = (0, 0)
        # Package positions: list of (x, y) or None if picked up/delivered
        self._packages: list[tuple[int, int] | None] = []
        # Destination positions
        self._destinations: list[tuple[int, int]] = []
        # Which package is carried (-1 = none)
        self._carrying: int = -1
        # Track deliveries
        self._delivered: list[bool] = []
        self._total_reward: float = 0.0
        # Walls on the grid
        self._walls: set[tuple[int, int]] = set()

    def env_id(self) -> str:
        return "glyphbench/classics-warehouse-v0"

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._carrying = -1
        self._delivered = [False, False, False]
        self._total_reward = 0.0

        # Place walls around the border
        self._walls = set()
        for x in range(GRID_SIZE):
            self._walls.add((x, 0))
            self._walls.add((x, GRID_SIZE - 1))
        for y in range(GRID_SIZE):
            self._walls.add((0, y))
            self._walls.add((GRID_SIZE - 1, y))

        # Add some interior walls for complexity
        interior_walls = [
            (3, 3), (3, 4), (3, 5),
            (6, 3), (6, 4), (6, 5),
        ]
        for wx, wy in interior_walls:
            self._walls.add((wx, wy))

        # Collect all free cells
        free_cells: list[tuple[int, int]] = []
        for y in range(1, GRID_SIZE - 1):
            for x in range(1, GRID_SIZE - 1):
                if (x, y) not in self._walls:
                    free_cells.append((x, y))

        # Shuffle and assign positions
        self.rng.shuffle(free_cells)  # type: ignore[arg-type]
        idx = 0

        self._robot_pos = free_cells[idx]
        idx += 1

        self._packages = []
        for _ in range(3):
            self._packages.append(free_cells[idx])
            idx += 1

        self._destinations = []
        for _ in range(3):
            self._destinations.append(free_cells[idx])
            idx += 1

        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]
        reward = 0.0

        if name in _DIR_DELTAS:
            dx, dy = _DIR_DELTAS[name]
            nx, ny = self._robot_pos[0] + dx, self._robot_pos[1] + dy
            if (nx, ny) not in self._walls:
                self._robot_pos = (nx, ny)
        elif name == "PICKUP":
            if self._carrying == -1:
                # Check if standing on a package
                for i, pkg_pos in enumerate(self._packages):
                    if pkg_pos is not None and pkg_pos == self._robot_pos:
                        self._carrying = i
                        self._packages[i] = None
                        break
        elif name == "DROP":
            if self._carrying >= 0:
                pkg_idx = self._carrying
                # Check if at the correct destination
                if self._robot_pos == self._destinations[pkg_idx]:
                    # Correct delivery. Pattern A: each correct delivery
                    # yields an equal share, with the final delivery absorbing
                    # any floating-point remainder so full completion lands
                    # exactly on 1.0.
                    self._delivered[pkg_idx] = True
                    self._carrying = -1
                    if all(self._delivered):
                        reward = 1.0 - self._total_reward
                    else:
                        reward = 1.0 / _NUM_PACKAGES
                    self._total_reward += reward
                else:
                    # Wrong destination or random spot: drop package back on floor
                    self._packages[pkg_idx] = self._robot_pos
                    self._carrying = -1

        # Check if all delivered. No completion bonus -- the per-delivery
        # rewards already sum to 1.0.
        terminated = False
        if all(self._delivered):
            terminated = True

        info["delivered"] = sum(self._delivered)
        info["carrying"] = self._carrying
        info["total_reward"] = self._total_reward

        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(GRID_SIZE, GRID_SIZE, SYM_FLOOR)

        # Walls
        for wx, wy in self._walls:
            grid[wy][wx] = SYM_WALL

        symbol_meanings: dict[str, str] = {
            SYM_ROBOT: "robot (you)",
            SYM_WALL: "wall",
            SYM_FLOOR: "floor",
        }

        # Destinations (render first so packages overlay if same cell)
        for i, (dx, dy) in enumerate(self._destinations):
            if not self._delivered[i]:
                _, dest_sym, color = PACKAGE_TYPES[i]
                grid[dy][dx] = dest_sym
                symbol_meanings[dest_sym] = f"{color} destination"

        # Packages on the ground
        for i, pkg_pos in enumerate(self._packages):
            if pkg_pos is not None:
                pkg_sym, _, color = PACKAGE_TYPES[i]
                px, py = pkg_pos
                grid[py][px] = pkg_sym
                symbol_meanings[pkg_sym] = f"{color} package"

        # Robot
        rx, ry = self._robot_pos
        grid[ry][rx] = SYM_ROBOT

        legend = build_legend(symbol_meanings)

        carrying_str = "nothing"
        if self._carrying >= 0:
            _, _, color = PACKAGE_TYPES[self._carrying]
            carrying_str = f"{color} package"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Carrying: {carrying_str}    "
            f"Delivered: {sum(self._delivered)}/3"
        )

        msg = ""
        if all(self._delivered):
            msg = "All packages delivered! Well done!"

        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message=msg)

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "You are a robot in a warehouse. Pick up colored packages and deliver them "
            "to their matching destinations.\n\n"
            "RULES\n"
            f"- The grid is {GRID_SIZE}x{GRID_SIZE} with walls around the border and some interior walls.\n"
            "- There are 3 packages and 3 matching destinations:\n"
            "  - Red: package = \u25cf, destination = \u25cb\n"
            "  - Blue: package = \u25c6, destination = \u25c7\n"
            "  - Green: package = \u25a0, destination = \u25a1\n"
            "- Walk onto a package and use PICKUP to carry it (one at a time).\n"
            f"- Walk to the matching destination and use DROP to deliver it (+{1.0/_NUM_PACKAGES:.4f} reward).\n"
            "- Dropping at the wrong location places the package on the ground.\n"
            "- Cumulative reward = 1.0 once all packages are delivered.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
