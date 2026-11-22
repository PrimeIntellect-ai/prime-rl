"""Gravity maze: navigate a ball through platforms with gravity.

The ball falls unless standing on a platform. Move left/right or jump.

Gym IDs:
  glyphbench/classics-gravitymaze-v0  (12x8)
"""

from __future__ import annotations

from collections import deque
from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAVITY_ACTION_SPEC = ActionSpec(
    names=("LEFT", "RIGHT", "JUMP"),
    descriptions=(
        "move the ball one cell to the left",
        "move the ball one cell to the right",
        "jump up 2 cells (only works if grounded)",
    ),
)

WIDTH = 12
HEIGHT = 8

SYM_BALL = "\u25cf"      # ●
SYM_PLATFORM = "\u25ac"  # ▬
SYM_GOAL = "\u2605"      # ★
SYM_AIR = "\u00b7"       # ·
SYM_WALL = "\u2588"      # █

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class GravityMazeEnv(BaseGlyphEnv):
    """Navigate a ball through a gravity maze to reach the goal."""

    action_spec = GRAVITY_ACTION_SPEC
    noop_action_name: str = "LEFT"

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._ball: tuple[int, int] = (0, 0)  # (x, y)
        self._goal: tuple[int, int] = (0, 0)
        self._platforms: set[tuple[int, int]] = set()

    def env_id(self) -> str:
        return "glyphbench/classics-gravitymaze-v0"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_solid(self, x: int, y: int) -> bool:
        """Check if a cell is solid (wall or platform)."""
        if x <= 0 or x >= WIDTH - 1 or y <= 0 or y >= HEIGHT - 1:
            return True  # walls
        return (x, y) in self._platforms

    def _is_grounded(self) -> bool:
        """Check if ball is standing on a solid surface."""
        return self._is_grounded_at(self._ball)

    def _is_grounded_at(self, pos: tuple[int, int]) -> bool:
        bx, by = pos
        return self._is_solid(bx, by + 1)

    def _apply_gravity_to(self, pos: tuple[int, int]) -> tuple[int, int]:
        """Apply gravity: ball falls if not grounded."""
        bx, by = pos
        while not self._is_solid(bx, by + 1):
            by += 1
            if by >= HEIGHT - 1:
                break
        return bx, by

    def _apply_gravity(self) -> None:
        """Apply gravity: ball falls if not grounded."""
        self._ball = self._apply_gravity_to(self._ball)

    def _fell_from(self, pos: tuple[int, int]) -> bool:
        return pos[1] >= HEIGHT - 2 and (pos[0], HEIGHT - 2) not in self._platforms

    def _next_position(
        self, pos: tuple[int, int], action_name: str
    ) -> tuple[int, int] | None:
        """Return next ball position, or None if the action falls into a gap."""
        bx, by = pos
        skip_gravity = False
        if action_name == "LEFT":
            nx = bx - 1
            if not self._is_solid(nx, by):
                bx = nx
        elif action_name == "RIGHT":
            nx = bx + 1
            if not self._is_solid(nx, by):
                bx = nx
        elif action_name == "JUMP" and self._is_grounded_at((bx, by)):
            # A successful jump occupies the air for this turn. On the next
            # action gravity resumes, so moving sideways can land on a nearby
            # platform top.
            ny = by - 1
            if not self._is_solid(bx, ny):
                ny2 = by - 2
                by = ny2 if not self._is_solid(bx, ny2) else ny
                skip_gravity = True

        new_pos = (bx, by)
        if not skip_gravity:
            new_pos = self._apply_gravity_to(new_pos)
        if self._fell_from(new_pos):
            return None
        return new_pos

    def _has_path_to_goal(self) -> bool:
        """Check the generated puzzle is solvable under the actual dynamics."""
        queue: deque[tuple[int, int]] = deque([self._ball])
        seen = {self._ball}
        while queue:
            pos = queue.popleft()
            if pos == self._goal:
                return True
            for action_name in self.action_spec.names:
                nxt = self._next_position(pos, action_name)
                if nxt is None or nxt in seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return False

    def _build_fallback_maze(self) -> None:
        """Small deterministic maze with a real jump-to-platform route."""
        self._platforms = {(x, HEIGHT - 2) for x in range(1, WIDTH - 1)}
        for x in range(4, 7):
            self._platforms.add((x, HEIGHT - 4))
        self._ball = (2, HEIGHT - 3)
        self._goal = (5, HEIGHT - 5)

    def _generate_maze(self) -> None:
        """Procedurally generate platforms and place goal."""
        for _attempt in range(100):
            self._platforms = set()

            # Create a ground floor
            for x in range(1, WIDTH - 1):
                self._platforms.add((x, HEIGHT - 2))

            # Create platforms at various heights
            num_platforms = int(self.rng.integers(4, 8))
            for _ in range(num_platforms):
                py = int(self.rng.integers(2, HEIGHT - 3))
                px_start = int(self.rng.integers(1, WIDTH - 4))
                length = int(self.rng.integers(2, 5))
                for x in range(px_start, min(px_start + length, WIDTH - 1)):
                    self._platforms.add((x, py))

            # Remove some ground floor sections to create gaps
            n_gaps = int(self.rng.integers(1, 4))
            for _ in range(n_gaps):
                gx = int(self.rng.integers(3, WIDTH - 3))
                gap_len = int(self.rng.integers(1, 3))
                for x in range(gx, min(gx + gap_len, WIDTH - 1)):
                    self._platforms.discard((x, HEIGHT - 2))

            # Place ball on a platform
            ground_cells = [
                (x, HEIGHT - 3) for x in range(1, WIDTH - 1)
                if (x, HEIGHT - 2) in self._platforms
            ]
            if not ground_cells:
                continue
            self._ball = ground_cells[int(self.rng.integers(0, len(ground_cells)))]

            # Place goal on a platform, preferably elevated.
            platform_tops: list[tuple[int, int]] = []
            for px, py in self._platforms:
                if py > 1 and (px, py - 1) not in self._platforms:
                    platform_tops.append((px, py - 1))
            platform_tops = [p for p in platform_tops if p != self._ball]
            if not platform_tops:
                continue
            self._goal = platform_tops[int(self.rng.integers(0, len(platform_tops)))]
            if self._has_path_to_goal():
                return

        self._build_fallback_maze()

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._generate_maze()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]
        next_pos = self._next_position(self._ball, name)
        if next_pos is None:
            reward = -1.0
            terminated = True
            info["fell"] = True
            return self._render_current_observation(), reward, terminated, False, info
        self._ball = next_pos

        reward = 0.0
        terminated = False

        # Check if at bottom wall row (fell into a gap)
        if self._fell_from(self._ball):
            # Ball is at wall level = fell off
            reward = -1.0
            terminated = True
            info["fell"] = True
        elif self._ball == self._goal:
            reward = 1.0
            terminated = True
            info["goal_reached"] = True

        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(WIDTH, HEIGHT, SYM_AIR)

        # Walls (border)
        for x in range(WIDTH):
            grid[0][x] = SYM_WALL
            grid[HEIGHT - 1][x] = SYM_WALL
        for y in range(HEIGHT):
            grid[y][0] = SYM_WALL
            grid[y][WIDTH - 1] = SYM_WALL

        # Platforms
        for px, py in self._platforms:
            if 0 < px < WIDTH - 1 and 0 < py < HEIGHT - 1:
                grid[py][px] = SYM_PLATFORM

        # Goal
        gx, gy = self._goal
        grid[gy][gx] = SYM_GOAL

        # Ball
        bx, by = self._ball
        if 0 < bx < WIDTH - 1 and 0 < by < HEIGHT - 1:
            grid[by][bx] = SYM_BALL

        legend = build_legend({
            SYM_BALL: "ball (you)",
            SYM_PLATFORM: "platform",
            SYM_GOAL: "goal",
            SYM_AIR: "air",
            SYM_WALL: "wall",
        })

        grounded = self._is_grounded()
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Grounded: {'yes' if grounded else 'no'}"
        )

        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message="")

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Navigate a ball through a gravity maze to reach the goal.\n\n"
            "RULES\n"
            f"- The grid is {WIDTH}x{HEIGHT} (width x height).\n"
            "- Gravity pulls the ball down each step. The ball falls until it lands on a platform.\n"
            "- LEFT/RIGHT move the ball one cell horizontally. Gravity applies after.\n"
            "- JUMP moves the ball up 2 cells when grounded; gravity resumes on the next action.\n"
            "- Use JUMP then LEFT/RIGHT to land on nearby elevated platform tops.\n"
            "- Reaching the goal gives +1 reward.\n"
            "- Falling off the bottom gives -1 reward.\n"
            "- Plan jumps carefully to reach elevated platforms.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
