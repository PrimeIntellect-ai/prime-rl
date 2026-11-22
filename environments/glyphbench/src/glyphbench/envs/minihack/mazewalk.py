"""MiniHack MazeWalk environments.

Procedurally generated mazes using recursive backtracking.
Player starts at one corner, stairs at the opposite corner.

Variants:
  - 9x9, 15x15, 45x19 (dark by default)
  - Mapped variants: same but fully visible (not dark)

Gym IDs:
  glyphbench/minihack-mazewalk-9x9-v0
  glyphbench/minihack-mazewalk-15x15-v0
  glyphbench/minihack-mazewalk-45x19-v0
  glyphbench/minihack-mazewalk-mapped-9x9-v0
  glyphbench/minihack-mazewalk-mapped-15x15-v0
  glyphbench/minihack-mazewalk-mapped-45x19-v0
"""

from __future__ import annotations

from glyphbench.envs.minihack.base import MiniHackBase


class _MazeWalkBase(MiniHackBase):
    """Base class for MazeWalk environments.

    Uses recursive backtracking to carve a perfect maze.
    Maze dimensions must be odd so that walls and passages alternate.
    """

    _maze_w: int = 9
    _maze_h: int = 9
    _is_dark: bool = True  # MazeWalk is dark by default

    def _generate_maze(self, w: int, h: int) -> list[list[str]]:
        """Generate a perfect maze using recursive backtracking.

        The maze grid uses '█' for walls and '·' for passages.
        w and h must be odd numbers >= 3.
        """
        # Ensure odd dimensions
        w = w if w % 2 == 1 else w - 1
        h = h if h % 2 == 1 else h - 1

        maze: list[list[str]] = [["█"] * w for _ in range(h)]

        # Start carving from (1, 1)
        maze[1][1] = "·"
        stack = [(1, 1)]

        while stack:
            cx, cy = stack[-1]
            neighbors: list[tuple[int, int, int, int]] = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 1 <= nx < w - 1 and 1 <= ny < h - 1 and maze[ny][nx] == "█":
                    # Wall between current and neighbor
                    wx, wy = cx + dx // 2, cy + dy // 2
                    neighbors.append((nx, ny, wx, wy))
            if neighbors:
                idx = int(self.rng.integers(0, len(neighbors)))
                nx, ny, wx, wy = neighbors[idx]
                maze[wy][wx] = "·"
                maze[ny][nx] = "·"
                stack.append((nx, ny))
            else:
                stack.pop()

        return maze

    def _generate_level(self, seed: int) -> None:
        w = self._maze_w
        h = self._maze_h

        # Generate the maze
        maze = self._generate_maze(w, h)

        # Init grid and copy maze into it
        self._init_grid(w, h)
        for y in range(h):
            for x in range(w):
                self._grid[y][x] = maze[y][x]

        # Fix borders: use - for top/bottom, | for sides (NetHack style)
        for x in range(w):
            self._grid[0][x] = "-"
            self._grid[h - 1][x] = "-"
        for y in range(1, h - 1):
            self._grid[y][0] = "|"
            self._grid[y][w - 1] = "|"

        self._dark = self._is_dark

        # Player at top-left passage (1, 1)
        self._place_player(1, 1)

        # Stairs at bottom-right: find the nearest walkable cell to (w-2, h-2)
        # In a perfect maze with odd dimensions, (w-2, h-2) is always a passage
        sx, sy = w - 2, h - 2
        if self._grid[sy][sx] != "·":
            # Search for nearest floor cell from bottom-right corner
            for dy in range(h - 2, 0, -1):
                for dx in range(w - 2, 0, -1):
                    if self._grid[dy][dx] == "·":
                        sx, sy = dx, dy
                        break
                else:
                    continue
                break
        self._place_stairs(sx, sy)


# --- Dark variants (default) ---


class MiniHackMazeWalk9x9Env(_MazeWalkBase):
    """9x9 maze, dark."""

    _maze_w = 9
    _maze_h = 9

    def env_id(self) -> str:
        return "glyphbench/minihack-mazewalk-9x9-v0"


class MiniHackMazeWalk15x15Env(_MazeWalkBase):
    """15x15 maze, dark."""

    _maze_w = 15
    _maze_h = 15

    def env_id(self) -> str:
        return "glyphbench/minihack-mazewalk-15x15-v0"


class MiniHackMazeWalk45x19Env(_MazeWalkBase):
    """45x19 maze, dark."""

    _maze_w = 45
    _maze_h = 19

    def __init__(self, max_turns: int = 240) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/minihack-mazewalk-45x19-v0"


# --- Mapped variants (fully visible) ---


class MiniHackMazeWalkMapped9x9Env(_MazeWalkBase):
    """9x9 maze, fully visible."""

    _maze_w = 9
    _maze_h = 9
    _is_dark = False

    def env_id(self) -> str:
        return "glyphbench/minihack-mazewalk-mapped-9x9-v0"


class MiniHackMazeWalkMapped15x15Env(_MazeWalkBase):
    """15x15 maze, fully visible."""

    _maze_w = 15
    _maze_h = 15
    _is_dark = False

    def env_id(self) -> str:
        return "glyphbench/minihack-mazewalk-mapped-15x15-v0"


class MiniHackMazeWalkMapped45x19Env(_MazeWalkBase):
    """45x19 maze, fully visible."""

    _maze_w = 45
    _maze_h = 19
    _is_dark = False

    def __init__(self, max_turns: int = 240) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/minihack-mazewalk-mapped-45x19-v0"
