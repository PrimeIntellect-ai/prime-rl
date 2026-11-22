"""Pokemon-style ice sliding puzzle.

Player slides in a direction until hitting a wall or rock. Reach the goal.

Gym IDs:
  glyphbench/classics-icesliding-easy-v0    (8x8)
  glyphbench/classics-icesliding-medium-v0  (10x10)
  glyphbench/classics-icesliding-hard-v0    (12x12, fewer rocks + trapdoors)

Phase 3 audit fix (2026-05-03): mechanics D2 (slippery ice deepening) +
B6 (trapdoors) on the hard variant. The hard variant inverted difficulty
because more rocks gave random sliding more stopping points; reducing
rocks and adding 2-3 hidden trapdoor tiles forces deliberate planning.
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

ICE_ACTION_SPEC = ActionSpec(
    names=("UP", "DOWN", "LEFT", "RIGHT"),
    descriptions=(
        "slide upward until hitting a wall or rock",
        "slide downward until hitting a wall or rock",
        "slide left until hitting a wall or rock",
        "slide right until hitting a wall or rock",
    ),
)

SYM_PLAYER = "@"
SYM_ICE = "░"     # ░
SYM_ROCK = "◆"    # ◆
SYM_GOAL = "★"    # ★
SYM_WALL = "█"    # █
# B6 (trapdoor): hidden under ice. Renders as the trapdoor glyph ONLY once
# the player has landed on it (final-step reveal); otherwise renders as ice.
SYM_TRAPDOOR = "☒"  # ☒ (BALLOT BOX WITH X)

# Trapdoor terminal penalty.
_TRAPDOOR_PENALTY = -0.5

_DIR_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class _IceSlidingBase(BaseGlyphEnv):
    """Pokemon-style ice puzzle: slide until hitting wall/rock."""

    action_spec = ICE_ACTION_SPEC
    noop_action_name: str = "UP"

    _grid_size: int = 8
    _num_rocks: int = 6
    _difficulty: str = "easy"
    # Number of trapdoor tiles disguised as ice. 0 for non-hard variants.
    _num_trapdoors: int = 0
    # Reject generated puzzles whose BFS-optimal solution is shorter than this
    # many slides. 0 disables the floor. Set > 1 on variants that would
    # otherwise be dominated by trivial 1-2 slide layouts that a blind random
    # policy can brute-force (audit 2026-05: classics-icesliding-medium-v0).
    _min_optimal_len: int = 0

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)
        self._player: tuple[int, int] = (0, 0)
        self._goal: tuple[int, int] = (0, 0)
        self._rocks: set[tuple[int, int]] = set()
        self._trapdoors: set[tuple[int, int]] = set()
        # Set to True the moment the player lands on a trapdoor — used by the
        # renderer to reveal the trapdoor glyph at the final observation.
        self._fell_into_trapdoor: bool = False
        # Track cumulative reward emitted so the trapdoor penalty can be
        # clamped to keep cumulative >= -1.0.
        self._cumulative_reward: float = 0.0

    def env_id(self) -> str:
        return f"glyphbench/classics-icesliding-{self._difficulty}-v0"

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _is_wall(self, x: int, y: int) -> bool:
        s = self._grid_size
        return x <= 0 or x >= s - 1 or y <= 0 or y >= s - 1

    def _blocked(self, x: int, y: int) -> bool:
        return self._is_wall(x, y) or (x, y) in self._rocks

    def _slide_result(self, sx: int, sy: int, dx: int, dy: int) -> tuple[int, int]:
        """Return where a slide from (sx, sy) in direction (dx, dy) ends up.

        Trapdoors do NOT stop the slide; they only act when the player would
        terminate ON them (i.e., final landing position equals a trapdoor)."""
        cx, cy = sx, sy
        while True:
            nx, ny = cx + dx, cy + dy
            if self._blocked(nx, ny):
                return (cx, cy)
            cx, cy = nx, ny

    def _optimal_len_bfs(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> int | None:
        """Length (in slides) of the shortest safe slide path from start to
        goal, or None if the goal is unreachable. Trapdoors are TREATED AS
        UNSAFE — the BFS may not pass through them, so the generator only
        ships puzzles whose solution avoids every trapdoor."""
        visited: dict[tuple[int, int], int] = {start: 0}
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == goal:
                return visited[(cx, cy)]
            for dx, dy in _DIR_DELTAS.values():
                nx, ny = self._slide_result(cx, cy, dx, dy)
                if (nx, ny) == (cx, cy):
                    continue
                if (nx, ny) in self._trapdoors:
                    continue
                if (nx, ny) not in visited:
                    visited[(nx, ny)] = visited[(cx, cy)] + 1
                    queue.append((nx, ny))
        return None

    def _generate_puzzle(self) -> None:
        """Generate a solvable ice-sliding puzzle.

        The hard variant additionally seeds 2-3 trapdoor tiles disguised as
        ice; the BFS only accepts puzzles whose goal is reachable WITHOUT
        stepping on any trapdoor.
        """
        s = self._grid_size
        interior = [(x, y) for x in range(1, s - 1) for y in range(1, s - 1)]

        for _ in range(500):
            self.rng.shuffle(interior)  # type: ignore[arg-type]
            self._player = interior[0]
            self._goal = interior[1]
            self._rocks = set()
            for i in range(2, 2 + self._num_rocks):
                if i < len(interior):
                    self._rocks.add(interior[i])

            self._trapdoors = set()
            if self._num_trapdoors > 0:
                # Pick trapdoor positions from cells not used as player/goal/
                # rock.
                start_idx = 2 + self._num_rocks
                end_idx = start_idx + self._num_trapdoors
                for i in range(start_idx, end_idx):
                    if i < len(interior):
                        self._trapdoors.add(interior[i])

            opt = self._optimal_len_bfs(self._player, self._goal)
            if opt is not None and opt >= self._min_optimal_len:
                self._fell_into_trapdoor = False
                return
        # Fallback: no rocks/trapdoors, player and goal on same row
        self._rocks = set()
        self._trapdoors = set()
        self._player = (1, 1)
        self._goal = (s - 2, 1)
        self._fell_into_trapdoor = False

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._generate_puzzle()
        self._cumulative_reward = 0.0
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]
        dx, dy = _DIR_DELTAS[name]

        px, py = self._player
        nx, ny = self._slide_result(px, py, dx, dy)
        self._player = (nx, ny)

        # Pattern A with bounded shaping: -1/max_turns per step (total floor
        # = -1.0 if agent never reaches the goal), +1.0 on goal (terminal).
        # Clamp the penalty so float summation of max_turns copies of
        # -1/max_turns can never drive the cumulative episode return strictly
        # below -1.0 (mirrors the trapdoor clamp below).
        reward = max(-1.0 / self.max_turns, -1.0 - self._cumulative_reward)
        terminated = False

        # B6: trapdoor terminates the episode with -0.5, clamped so that
        # the cumulative episode return stays >= -1.0.
        if self._player in self._trapdoors:
            self._fell_into_trapdoor = True
            # Clamp the trapdoor reward so cumulative >= -1.0.
            min_floor = -1.0 - self._cumulative_reward
            reward = max(_TRAPDOOR_PENALTY, min_floor)
            terminated = True
            info["trapdoor"] = True
            self._cumulative_reward += reward
            return (
                self._render_current_observation(),
                reward,
                terminated,
                False,
                info,
            )

        if self._player == self._goal:
            reward = 1.0
            terminated = True
            info["goal_reached"] = True

        self._cumulative_reward += reward
        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        s = self._grid_size
        grid = make_empty_grid(s, s, SYM_ICE)

        # Walls (border)
        for x in range(s):
            grid[0][x] = SYM_WALL
            grid[s - 1][x] = SYM_WALL
        for y in range(s):
            grid[y][0] = SYM_WALL
            grid[y][s - 1] = SYM_WALL

        # Rocks
        for rx, ry in self._rocks:
            grid[ry][rx] = SYM_ROCK

        # Goal
        gx, gy = self._goal
        grid[gy][gx] = SYM_GOAL

        # Trapdoors stay disguised as ice unless we've already triggered one.
        if self._fell_into_trapdoor:
            for tx, ty in self._trapdoors:
                grid[ty][tx] = SYM_TRAPDOOR

        # Player
        px, py = self._player
        grid[py][px] = SYM_PLAYER

        legend_map = {
            SYM_PLAYER: "you",
            SYM_ICE: "ice (you slide over this)",
            SYM_ROCK: "rock (stops sliding)",
            SYM_GOAL: "goal",
            SYM_WALL: "wall",
        }
        if self._num_trapdoors > 0:
            legend_map[SYM_TRAPDOOR] = "trapdoor (revealed only after you fall in)"
        legend = build_legend(legend_map)

        hud_parts = [f"Step: {self._turn} / {self.max_turns}"]
        if self._num_trapdoors > 0:
            hud_parts.append(f"Trapdoors hidden: {len(self._trapdoors)}")
        hud = "    ".join(hud_parts)

        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message="",
        )

    def system_prompt(self) -> str:
        trap_clause = ""
        if self._num_trapdoors > 0:
            trap_clause = (
                "\n- HIDDEN TRAPDOORS: this variant secretly contains "
                f"{self._num_trapdoors} trapdoor tiles disguised as ice. "
                "If your slide ENDS on a trapdoor, you fall in: -0.5 reward "
                "and the episode ends. The trapdoor reveals itself only at "
                "the moment you land on it, so plan carefully — every slide "
                "stop is a risk."
            )
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Navigate an ice-sliding puzzle. When you move in a direction, "
            "you slide continuously until you hit a wall or a rock. Reach "
            "the goal.\n\n"
            "RULES\n"
            f"- The grid is {self._grid_size}x{self._grid_size} with walls "
            "around the border.\n"
            "- Moving on ice causes you to slide until you hit a wall or "
            "rock.\n"
            "- Rocks stop your sliding but you cannot pass through them.\n"
            "- Reach the goal to win (+1 reward). The goal does not stop your "
            "slide; you must come to rest exactly on it.\n"
            "- Each move costs a small step penalty scaled so that, if you "
            "never reach the goal, cumulative reward bottoms out at -1 "
            "over the full episode budget. Reach the goal as fast as you "
            "can.\n"
            f"- Plan your path carefully: you cannot stop on ice mid-slide."
            f"{trap_clause}\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class IceSlidingEasyEnv(_IceSlidingBase):
    _grid_size = 8
    _num_rocks = 6
    _difficulty = "easy"

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)


class IceSlidingMediumEnv(_IceSlidingBase):
    _grid_size = 10
    _num_rocks = 10
    _difficulty = "medium"
    # Audit 2026-05 retune: reject puzzles solvable in < 3 slides (the old
    # generator was 44.7% solvable in <=2 slides, dominated by trivia) so a
    # blind random slider cannot stumble onto the goal.
    _min_optimal_len = 3

    def __init__(self, max_turns: int = 30) -> None:
        # Audit 2026-05 retune: was 150 (mean optimal length 3.4) — far too
        # generous, let uniform-random win 70% with positive mean. A budget of
        # 30 leaves ample room for a planner (max observed optimal len = 11)
        # while denying random the brute-force window.
        super().__init__(max_turns=max_turns)


class IceSlidingHardEnv(_IceSlidingBase):
    _grid_size = 12
    _num_rocks = 8  # Phase 3 fix: was 15 (too many stopping points).
    _difficulty = "hard"
    # Phase 3 fix: 2-3 trapdoor tiles disguised as ice. Audit specified
    # 2-3; we ship with the upper bound (3) so random sliding is more
    # likely to step on at least one trapdoor.
    _num_trapdoors = 3

    def __init__(self, max_turns: int = 80) -> None:
        # Trim the budget so random sliding cannot brute-force the goal.
        super().__init__(max_turns=max_turns)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
