"""Pipe rotation puzzle: connect source to sink by rotating pipe pieces.

Gym IDs:
  glyphbench/classics-pipeconnect-easy-v0    (5x5)
  glyphbench/classics-pipeconnect-medium-v0  (7x7)
  glyphbench/classics-pipeconnect-hard-v0    (9x9)
"""

from __future__ import annotations

from collections import deque
from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Pipe connectivity definitions
# ---------------------------------------------------------------------------
# Directions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
# Each pipe type maps to a set of direction pairs it connects.

# Pipe types: each defined by which directions it connects.
# Stored as frozenset of directions it opens to.
# Rotation = shift all directions by +1 mod 4.

# Straight: connects two opposite dirs
# Elbow: connects two adjacent dirs
# T-junction: connects three dirs
# Cross: connects all four dirs

# Visual characters per (type, rotation):
# Straight: ═ (horizontal, connects L-R) or ║ (vertical, connects U-D)
# Elbow: ╔ (connects R,D), ╗ (connects L,D), ╚ (connects R,U), ╝ (connects L,U)
# T-junction: ╦ (connects L,R,D), ╩ (connects L,R,U), ╠ (connects U,R,D), ╣ (connects U,L,D)
# Cross: ╬ (all four)

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# Each pipe piece: (frozenset of open directions, char)
# We define "canonical" forms for each type, then rotate.

_STRAIGHT_H = (frozenset({LEFT, RIGHT}), "\u2550")     # ═
_STRAIGHT_V = (frozenset({UP, DOWN}), "\u2551")         # ║
_ELBOW_RD = (frozenset({RIGHT, DOWN}), "\u2554")        # ╔
_ELBOW_LD = (frozenset({LEFT, DOWN}), "\u2557")         # ╗
_ELBOW_RU = (frozenset({RIGHT, UP}), "\u255a")          # ╚
_ELBOW_LU = (frozenset({LEFT, UP}), "\u255d")           # ╝
_T_LRD = (frozenset({LEFT, RIGHT, DOWN}), "\u2566")     # ╦
_T_LRU = (frozenset({LEFT, RIGHT, UP}), "\u2569")       # ╩
_T_URD = (frozenset({UP, RIGHT, DOWN}), "\u2560")       # ╠
_T_ULD = (frozenset({UP, LEFT, DOWN}), "\u2563")        # ╣
_CROSS = (frozenset({UP, RIGHT, DOWN, LEFT}), "\u256c") # ╬

# Map from frozenset of directions -> char
_DIRS_TO_CHAR: dict[frozenset[int], str] = {
    frozenset({LEFT, RIGHT}): "\u2550",
    frozenset({UP, DOWN}): "\u2551",
    frozenset({RIGHT, DOWN}): "\u2554",
    frozenset({LEFT, DOWN}): "\u2557",
    frozenset({RIGHT, UP}): "\u255a",
    frozenset({LEFT, UP}): "\u255d",
    frozenset({LEFT, RIGHT, DOWN}): "\u2566",
    frozenset({LEFT, RIGHT, UP}): "\u2569",
    frozenset({UP, RIGHT, DOWN}): "\u2560",
    frozenset({UP, LEFT, DOWN}): "\u2563",
    frozenset({UP, RIGHT, DOWN, LEFT}): "\u256c",
}

SYM_SOURCE = "\u25a3"  # ▣
SYM_SINK = "\u25a2"    # ▢
SYM_EMPTY = "\u00b7"   # ·


def _rotate_dirs(dirs: frozenset[int], times: int = 1) -> frozenset[int]:
    """Rotate a set of directions clockwise by 90 degrees `times` times."""
    result = dirs
    for _ in range(times % 4):
        result = frozenset((d + 1) % 4 for d in result)
    return result


def _char_for_dirs(dirs: frozenset[int]) -> str:
    return _DIRS_TO_CHAR.get(dirs, "?")


# Pipe type templates: canonical direction sets
_PIPE_TYPES = [
    frozenset({LEFT, RIGHT}),           # straight horizontal
    frozenset({RIGHT, DOWN}),            # elbow
    frozenset({LEFT, RIGHT, DOWN}),      # T-junction
    frozenset({UP, RIGHT, DOWN, LEFT}),  # cross
]

DIFFICULTY = {
    "easy": {"size": 5, "max_turns": 100},
    "medium": {"size": 7, "max_turns": 200},
    "hard": {"size": 9, "max_turns": 300},
}

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


def _make_pipe_action_spec(grid_size: int) -> ActionSpec:
    """Build an ActionSpec for a pipe-connect grid of the given size."""
    n = grid_size * grid_size
    names = tuple(f"ROTATE_{i}" for i in range(n))
    descs = tuple(
        f"rotate pipe at row {i // grid_size}, col {i % grid_size} 90 degrees CW"
        for i in range(n)
    )
    return ActionSpec(names=names, descriptions=descs)


class _PipeConnectBase(BaseGlyphEnv):
    """Rotate pipe pieces to connect source to sink.

    Positions are stored as (x, y); the pipe grid is indexed as [y][x].
    """

    noop_action_name: str = "ROTATE_0"

    _grid_size: int = 5
    _difficulty: str = "easy"

    def __init__(self, max_turns: int = 100) -> None:
        self.action_spec = _make_pipe_action_spec(self._grid_size)
        super().__init__(max_turns=max_turns)
        # Grid of pipe directions: None means empty cell
        self._pipes: list[list[frozenset[int] | None]] = []
        self._source_pos: tuple[int, int] = (0, 0)
        self._sink_pos: tuple[int, int] = (0, 0)

    def env_id(self) -> str:
        return f"glyphbench/classics-pipeconnect-{self._difficulty}-v0"

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_path(self) -> list[tuple[int, int]]:
        """Generate a random path from source to sink using random walk."""
        s = self._grid_size
        # Source on left edge, sink on right edge (random rows)
        src_row = int(self.rng.integers(0, s))
        sink_row = int(self.rng.integers(0, s))
        self._source_pos = (0, src_row)
        self._sink_pos = (s - 1, sink_row)

        # BFS with random neighbor ordering to find a path
        visited: set[tuple[int, int]] = {self._source_pos}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {self._source_pos: None}
        queue: deque[tuple[int, int]] = deque([self._source_pos])
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left

        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == self._sink_pos:
                break
            neighbors = [(cx + dx, cy + dy) for dx, dy in deltas]
            # Shuffle for variety
            order = list(range(4))
            self.rng.shuffle(order)  # type: ignore[arg-type]
            for i in order:
                nx, ny = neighbors[i]
                if 0 <= nx < s and 0 <= ny < s and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))

        # Trace back the path
        path: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = self._sink_pos
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        return path

    def _direction_between(self, fx: int, fy: int, tx: int, ty: int) -> int:
        """Get the direction from (fx,fy) to (tx,ty)."""
        dx, dy = tx - fx, ty - fy
        if dx == 1:
            return RIGHT
        if dx == -1:
            return LEFT
        if dy == 1:
            return DOWN
        return UP

    def _generate_puzzle(self) -> None:
        """Create a valid pipe puzzle and scramble it."""
        s = self._grid_size
        self._pipes = [[None for _ in range(s)] for _ in range(s)]

        path = self._generate_path()

        # Place pipes along the path with correct connectivity
        for i, (px, py) in enumerate(path):
            dirs: set[int] = set()
            if i > 0:
                prev = path[i - 1]
                dirs.add(self._direction_between(px, py, prev[0], prev[1]))
            if i < len(path) - 1:
                nxt = path[i + 1]
                dirs.add(self._direction_between(px, py, nxt[0], nxt[1]))

            # Source and sink glyphs cover their pipe character on the grid, so
            # each endpoint gets an outward opening to keep its HUD shape
            # renderable while the real in-grid opening still controls flow.
            if (px, py) == self._source_pos and i == 0:
                dirs.add(LEFT)  # connects outward (conceptually)
            elif (px, py) == self._sink_pos and i == len(path) - 1:
                dirs.add(RIGHT)  # connects outward (conceptually)

            self._pipes[py][px] = frozenset(dirs)

        # Add some extra random pipes on empty cells for complexity
        empty_cells = [
            (x, y) for y in range(s) for x in range(s)
            if self._pipes[y][x] is None
        ]
        n_extra = min(len(empty_cells), s)
        if empty_cells:
            self.rng.shuffle(empty_cells)  # type: ignore[arg-type]
            for i in range(n_extra):
                ex, ey = empty_cells[i]
                ptype = _PIPE_TYPES[int(self.rng.integers(0, len(_PIPE_TYPES)))]
                rot = int(self.rng.integers(0, 4))
                self._pipes[ey][ex] = _rotate_dirs(ptype, rot)

        # Now randomly rotate all pipe pieces to scramble
        for y in range(s):
            for x in range(s):
                if self._pipes[y][x] is not None:
                    rot = int(self.rng.integers(0, 4))
                    self._pipes[y][x] = _rotate_dirs(self._pipes[y][x], rot)

        # Avoid a degenerate episode that is solved before the player acts.
        if self._trace_flow():
            sx, sy = self._source_pos
            src_dirs = self._pipes[sy][sx]
            if src_dirs is not None:
                self._pipes[sy][sx] = _rotate_dirs(src_dirs, 1)

    # ------------------------------------------------------------------
    # Flow tracing
    # ------------------------------------------------------------------

    def _trace_flow(self) -> bool:
        """Check if source is connected to sink by tracing pipe connections."""
        s = self._grid_size
        sx, sy = self._source_pos
        visited: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque()

        # Start from source
        src_dirs = self._pipes[sy][sx]
        if src_dirs is None:
            return False
        visited.add((sx, sy))
        queue.append((sx, sy))

        deltas = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}

        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == self._sink_pos:
                return True
            cur_dirs = self._pipes[cy][cx]
            if cur_dirs is None:
                continue
            for d in cur_dirs:
                dx, dy = deltas[d]
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < s and 0 <= ny < s and (nx, ny) not in visited:
                    neighbor_dirs = self._pipes[ny][nx]
                    if neighbor_dirs is not None and OPPOSITE[d] in neighbor_dirs:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._generate_puzzle()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        s = self._grid_size
        r, c = divmod(action, s)

        # Rotate the pipe at row r, col c if it exists.
        if 0 <= r < s and 0 <= c < s and self._pipes[r][c] is not None:
            self._pipes[r][c] = _rotate_dirs(self._pipes[r][c], 1)

        # Check if connected
        connected = self._trace_flow()
        reward = 0.0
        terminated = False

        if connected:
            reward = 1.0
            terminated = True
            info["connected"] = True

        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        s = self._grid_size
        grid = make_empty_grid(s, s, SYM_EMPTY)

        for y in range(s):
            for x in range(s):
                if (x, y) == self._source_pos:
                    grid[y][x] = SYM_SOURCE
                elif (x, y) == self._sink_pos:
                    grid[y][x] = SYM_SINK
                elif self._pipes[y][x] is not None:
                    grid[y][x] = _char_for_dirs(self._pipes[y][x])

        legend_map: dict[str, str] = {
            SYM_SOURCE: "source (flow starts here)",
            SYM_SINK: "sink (connect flow here to win)",
            SYM_EMPTY: "empty cell",
            "\u2550": "straight pipe (horizontal: left-right)",
            "\u2551": "straight pipe (vertical: up-down)",
            "\u2554": "elbow pipe (right-down)",
            "\u2557": "elbow pipe (left-down)",
            "\u255a": "elbow pipe (right-up)",
            "\u255d": "elbow pipe (left-up)",
            "\u2566": "T-junction (left-right-down)",
            "\u2569": "T-junction (left-right-up)",
            "\u2560": "T-junction (up-right-down)",
            "\u2563": "T-junction (up-left-down)",
            "\u256c": "cross pipe (all four directions)",
        }

        # Only include symbols that are actually on the grid
        present_syms: set[str] = set()
        for row in grid:
            for ch in row:
                present_syms.add(ch)
        filtered_legend = {k: v for k, v in legend_map.items() if k in present_syms}
        legend = build_legend(filtered_legend)

        # Source and sink glyphs hide their underlying pipe openings.
        src_dirs = self._pipes[self._source_pos[1]][self._source_pos[0]]
        sink_dirs = self._pipes[self._sink_pos[1]][self._sink_pos[0]]
        src_str = _char_for_dirs(src_dirs) if src_dirs else "none"
        sink_str = _char_for_dirs(sink_dirs) if sink_dirs else "none"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Source pipe: {src_str}    "
            f"Sink pipe: {sink_str}"
        )

        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message="")

    def system_prompt(self) -> str:
        s = self._grid_size
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Rotate pipe pieces to create a connected path from the source to the sink.\n\n"
            "RULES\n"
            f"- The grid is {s}x{s}. Source and sink positions are visible on the [Grid].\n"
            "- Each action rotates one pipe piece 90 degrees clockwise.\n"
            "- Source and sink cells also contain rotatable pipes; their glyphs hide the\n"
            "  pipe shape, so use the HUD's Source pipe and Sink pipe fields.\n"
            "- Pipes connect when adjacent pipes have openings facing each other.\n"
            "- Win by creating a continuous connection from source to sink; this gives\n"
            "  +1 reward and ends the episode. Other turns give 0 reward.\n"
            "- Pipe types: straight (connects 2 opposite sides), elbow (connects 2 adjacent sides),\n"
            "  T-junction (connects 3 sides), cross (connects all 4 sides).\n"
            "- Extra pipes may be decoys; not every pipe needs to be on the solution path.\n"
            f"- Actions are ROTATE_0 through ROTATE_{s*s - 1} where index = row * {s} + col.\n"
            "- Rotating an empty cell does nothing.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class PipeConnectEasyEnv(_PipeConnectBase):
    _grid_size = 5
    _difficulty = "easy"

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)


class PipeConnectMediumEnv(_PipeConnectBase):
    _grid_size = 7
    _difficulty = "medium"

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)


class PipeConnectHardEnv(_PipeConnectBase):
    _grid_size = 9
    _difficulty = "hard"

    def __init__(self, max_turns: int = 300) -> None:
        super().__init__(max_turns=max_turns)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
