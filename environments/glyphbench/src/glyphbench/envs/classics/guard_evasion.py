"""Guard evasion stealth game: reach the exit while avoiding patrol guards.

Guards move on fixed patrol routes and watch a straight line ahead.
If the player enters a guard's vision line, the player is caught.

Gym IDs:
  glyphbench/classics-guardevasion-easy-v0    (10x10)
  glyphbench/classics-guardevasion-medium-v0  (12x12)
  glyphbench/classics-guardevasion-hard-v0    (15x15)
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

GUARD_ACTION_SPEC = ActionSpec(
    names=("UP", "DOWN", "LEFT", "RIGHT", "WAIT"),
    descriptions=(
        "move one cell up",
        "move one cell down",
        "move one cell left",
        "move one cell right",
        "stay in place while guards advance one turn",
    ),
)

SYM_PLAYER = "@"
SYM_GUARD = "\u2690"   # ⚐
SYM_EXIT = "\u2605"    # ★
SYM_WALL = "\u2588"    # █
SYM_FLOOR = "\u00b7"   # ·
SYM_VISION = "\u2591"  # ░

_DIR_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "WAIT": (0, 0),
}

# Direction indices for guards
_GUARD_DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT

DIFFICULTY = {
    "easy": {"size": 10, "guards": 2, "walls": 5, "max_turns": 150},
    "medium": {"size": 12, "guards": 4, "walls": 8, "max_turns": 200},
    "hard": {"size": 15, "guards": 6, "walls": 12, "max_turns": 300},
}

VISION_RANGE = 3

# ---------------------------------------------------------------------------
# Guard data
# ---------------------------------------------------------------------------


class _Guard:
    """A guard on a fixed patrol route."""

    __slots__ = ("route", "idx", "forward", "facing_dx", "facing_dy")

    def __init__(self, route: list[tuple[int, int]]) -> None:
        self.route = route
        self.idx = 0
        self.forward = True  # True = moving forward along route
        # Facing direction based on next step in route
        self.facing_dx = 0
        self.facing_dy = 1  # default: facing down
        self._update_facing()

    @property
    def pos(self) -> tuple[int, int]:
        return self.route[self.idx]

    def step(self) -> None:
        """Move guard one step along patrol route."""
        if len(self.route) <= 1:
            return
        if self.forward:
            if self.idx < len(self.route) - 1:
                self.idx += 1
            else:
                self.forward = False
                self.idx -= 1
        else:
            if self.idx > 0:
                self.idx -= 1
            else:
                self.forward = True
                self.idx += 1
        self._update_facing()

    def _update_facing(self) -> None:
        """Update facing direction based on movement."""
        if len(self.route) <= 1:
            return
        if self.forward and self.idx < len(self.route) - 1:
            nx, ny = self.route[self.idx + 1]
        elif not self.forward and self.idx > 0:
            nx, ny = self.route[self.idx - 1]
        else:
            return
        cx, cy = self.route[self.idx]
        self.facing_dx = nx - cx
        self.facing_dy = ny - cy

    def vision_cells(self, grid_size: int, walls: set[tuple[int, int]]) -> list[tuple[int, int]]:
        """Return cells in this guard's vision line."""
        cells: list[tuple[int, int]] = []
        cx, cy = self.pos
        for i in range(1, VISION_RANGE + 1):
            vx = cx + self.facing_dx * i
            vy = cy + self.facing_dy * i
            if vx <= 0 or vx >= grid_size - 1 or vy <= 0 or vy >= grid_size - 1:
                break
            if (vx, vy) in walls:
                break
            cells.append((vx, vy))
        return cells


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class _GuardEvasionBase(BaseGlyphEnv):
    """Reach the exit while avoiding guards with vision lines."""

    action_spec = GUARD_ACTION_SPEC
    noop_action_name: str = "WAIT"

    _grid_size: int = 10
    _num_guards: int = 2
    _num_walls: int = 5
    _difficulty: str = "easy"

    def __init__(self, max_turns: int = 150) -> None:
        super().__init__(max_turns=max_turns)
        self._player: tuple[int, int] = (0, 0)
        self._exit: tuple[int, int] = (0, 0)
        self._guards: list[_Guard] = []
        self._walls: set[tuple[int, int]] = set()

    def env_id(self) -> str:
        return f"glyphbench/classics-guardevasion-{self._difficulty}-v0"

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_level(self) -> None:
        """Generate a level with guards, walls, player, and exit."""
        s = self._grid_size
        self._walls = set()
        self._guards = []

        # Place some interior wall segments
        for _ in range(self._num_walls):
            wx = int(self.rng.integers(2, s - 2))
            wy = int(self.rng.integers(2, s - 2))
            self._walls.add((wx, wy))

        occupied = set(self._walls)

        # Place player near top-left
        self._player = (1, 1)
        occupied.add(self._player)

        # Place exit near bottom-right
        self._exit = (s - 2, s - 2)
        occupied.add(self._exit)

        # Place guards on patrol routes
        for _ in range(self._num_guards):
            # Create a patrol route (line of 3-5 cells)
            for _attempt in range(50):
                direction = int(self.rng.integers(0, 2))  # 0=horizontal, 1=vertical
                length = int(self.rng.integers(3, 6))
                sx = int(self.rng.integers(2, s - 2))
                sy = int(self.rng.integers(2, s - 2))

                route: list[tuple[int, int]] = []
                valid = True
                for i in range(length):
                    if direction == 0:
                        px, py = sx + i, sy
                    else:
                        px, py = sx, sy + i
                    if px <= 0 or px >= s - 1 or py <= 0 or py >= s - 1:
                        valid = False
                        break
                    if (px, py) in occupied:
                        valid = False
                        break
                    route.append((px, py))

                if valid and len(route) >= 2:
                    guard = _Guard(route)
                    self._guards.append(guard)
                    for cell in route:
                        occupied.add(cell)
                    break

    def _get_all_vision_cells(self) -> set[tuple[int, int]]:
        """Get all cells currently in any guard's vision."""
        vision: set[tuple[int, int]] = set()
        for guard in self._guards:
            vision.update(guard.vision_cells(self._grid_size, self._walls))
        return vision

    def _is_caught(self) -> bool:
        """Check if player is in any guard's vision or on a guard's position."""
        for guard in self._guards:
            if self._player == guard.pos:
                return True
            if self._player in guard.vision_cells(self._grid_size, self._walls):
                return True
        return False

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._generate_level()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]
        dx, dy = _DIR_DELTAS[name]

        # Move player
        px, py = self._player
        nx, ny = px + dx, py + dy
        s = self._grid_size
        if (
            1 <= nx < s - 1
            and 1 <= ny < s - 1
            and (nx, ny) not in self._walls
        ):
            self._player = (nx, ny)

        # Move all guards
        for guard in self._guards:
            guard.step()

        # Check conditions. Escaping takes precedence once the player reaches
        # the exit after this turn's movement.
        reward = 0.0
        terminated = False

        if self._player == self._exit:
            reward = 1.0
            terminated = True
            info["escaped"] = True
        elif self._is_caught():
            reward = -1.0
            terminated = True
            info["caught"] = True

        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        s = self._grid_size
        grid = make_empty_grid(s, s, SYM_FLOOR)

        # Border walls
        for x in range(s):
            grid[0][x] = SYM_WALL
            grid[s - 1][x] = SYM_WALL
        for y in range(s):
            grid[y][0] = SYM_WALL
            grid[y][s - 1] = SYM_WALL

        # Interior walls
        for wx, wy in self._walls:
            grid[wy][wx] = SYM_WALL

        # Guard vision cones
        vision_cells = self._get_all_vision_cells()
        for vx, vy in vision_cells:
            if grid[vy][vx] == SYM_FLOOR:
                grid[vy][vx] = SYM_VISION

        # Exit
        ex, ey = self._exit
        grid[ey][ex] = SYM_EXIT

        # Guards
        for guard in self._guards:
            gx, gy = guard.pos
            grid[gy][gx] = SYM_GUARD

        # Player (on top of everything)
        px, py = self._player
        grid[py][px] = SYM_PLAYER

        legend = build_legend({
            SYM_PLAYER: "you",
            SYM_GUARD: "guard (avoid their vision line!)",
            SYM_EXIT: "exit (reach this to win)",
            SYM_WALL: "wall",
            SYM_FLOOR: "floor",
            SYM_VISION: "guard vision line (entering = caught!)",
        })

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Guards: {len(self._guards)}"
        )

        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message="")

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Reach the exit while avoiding detection by patrol guards.\n\n"
            "RULES\n"
            f"- The grid is {self._grid_size}x{self._grid_size} with border walls and interior walls.\n"
            f"- There are {self._num_guards} guards patrolling fixed routes.\n"
            f"- Each guard watches a straight vision line up to {VISION_RANGE} cells in their facing direction.\n"
            "- Vision is blocked by walls.\n"
            "- Each turn: you move or WAIT, then guards move one patrol step, then detection is checked.\n"
            "- If you enter a guard's vision line or step on a guard, you are caught (-1 reward).\n"
            "- Reaching the exit gives +1 reward.\n"
            "- You can WAIT to let guards pass.\n"
            "- Plan your route to avoid guard vision lines.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class GuardEvasionEasyEnv(_GuardEvasionBase):
    _grid_size = 10
    _num_guards = 2
    _num_walls = 5
    _difficulty = "easy"

    def __init__(self, max_turns: int = 150) -> None:
        super().__init__(max_turns=max_turns)


class GuardEvasionMediumEnv(_GuardEvasionBase):
    _grid_size = 12
    _num_guards = 4
    _num_walls = 8
    _difficulty = "medium"

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)


class GuardEvasionHardEnv(_GuardEvasionBase):
    _grid_size = 15
    _num_guards = 6
    _num_walls = 12
    _difficulty = "hard"

    def __init__(self, max_turns: int = 300) -> None:
        super().__init__(max_turns=max_turns)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
