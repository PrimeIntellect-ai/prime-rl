"""Classic Snake game.

Gym IDs:
  glyphbench/classics-snake-easy-v0      (10x10)
  glyphbench/classics-snake-medium-v0    (12x12)
  glyphbench/classics-snake-hard-v0      (15x15)
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

SNAKE_ACTION_SPEC = ActionSpec(
    names=("UP", "DOWN", "LEFT", "RIGHT"),
    descriptions=(
        "move the snake up",
        "move the snake down",
        "move the snake left",
        "move the snake right",
    ),
)

SYM_HEAD = "\u25c6"   # ◆
SYM_BODY = "\u25c7"   # ◇
SYM_FOOD = "\u2726"   # ✦
SYM_WALL = "\u2588"   # █
SYM_FLOOR = "\u00b7"  # ·

_DIR_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

_OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

# ---------------------------------------------------------------------------
# Base Snake Env
# ---------------------------------------------------------------------------


class _SnakeBase(BaseGlyphEnv):
    """Classic Snake: eat food to grow, avoid walls and yourself."""

    action_spec = SNAKE_ACTION_SPEC
    noop_action_name: str = "UP"

    _grid_size: int = 10
    _difficulty: str = "easy"
    _target_food_goal: int = 5

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._snake: list[tuple[int, int]] = []
        self._direction: str = "RIGHT"
        self._food: tuple[int, int] = (0, 0)
        self._score: int = 0
        self._alive: bool = True
        # Pattern A: cumulative reward = 1.0 once the difficulty target is
        # reached. The target is intentionally much smaller than the theoretical
        # board-fill maximum so each food event is a meaningful signal.
        self._target_food: int = self._target_food_goal

    def env_id(self) -> str:
        return f"glyphbench/classics-snake-{self._difficulty}-v0"

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        s = self._grid_size
        mid = s // 2
        self._snake = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]
        self._direction = "RIGHT"
        self._score = 0
        self._alive = True
        self._spawn_food()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]

        # Prevent reversing into self
        if _OPPOSITE.get(name) == self._direction:
            name = self._direction
        self._direction = name

        dx, dy = _DIR_DELTAS[name]
        hx, hy = self._snake[0]
        nx, ny = hx + dx, hy + dy

        # Wall collision (border is wall, playable area 1..size-2)
        if nx <= 0 or nx >= self._grid_size - 1 or ny <= 0 or ny >= self._grid_size - 1:
            self._alive = False
            return self._render_current_observation(), 0.0, True, False, info

        will_eat = (nx, ny) == self._food

        # Self collision. Moving into the current tail is allowed on non-food
        # steps because that tail cell is vacated at the end of the move.
        body_cells = self._snake if will_eat else self._snake[:-1]
        if (nx, ny) in body_cells:
            self._alive = False
            return self._render_current_observation(), 0.0, True, False, info

        self._snake.insert(0, (nx, ny))
        reward = 0.0

        if will_eat:
            self._score += 1
            reward = 1.0 / self._target_food
            if self._score >= self._target_food:
                self._alive = False
                info["score"] = self._score
                return self._render_current_observation(), reward, True, False, info
            self._spawn_food()
        else:
            self._snake.pop()

        info["score"] = self._score
        return self._render_current_observation(), reward, False, False, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _spawn_food(self) -> None:
        occupied = set(self._snake)
        while True:
            x = int(self.rng.integers(1, self._grid_size - 1))
            y = int(self.rng.integers(1, self._grid_size - 1))
            if (x, y) not in occupied:
                self._food = (x, y)
                return

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        s = self._grid_size
        grid = make_empty_grid(s, s, SYM_FLOOR)

        for x in range(s):
            grid[0][x] = SYM_WALL
            grid[s - 1][x] = SYM_WALL
        for y in range(s):
            grid[y][0] = SYM_WALL
            grid[y][s - 1] = SYM_WALL

        fx, fy = self._food
        grid[fy][fx] = SYM_FOOD

        for bx, by in self._snake[1:]:
            grid[by][bx] = SYM_BODY
        hx, hy = self._snake[0]
        grid[hy][hx] = SYM_HEAD

        legend = build_legend({
            SYM_HEAD: "snake head",
            SYM_BODY: "snake body",
            SYM_FOOD: "food",
            SYM_WALL: "wall",
            SYM_FLOOR: "floor",
        })

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Score: {self._score}/{self._target_food}    "
            f"Length: {len(self._snake)}    "
            f"Direction: {self._direction}"
        )

        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message="")

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Control a snake on a grid. Eat food to grow longer and earn points. "
            "Avoid hitting the walls or your own body.\n\n"
            "RULES\n"
            f"- The grid is {self._grid_size}x{self._grid_size} with walls around the border.\n"
            "- The snake starts at length 3 moving right.\n"
            "- Each food eaten grows the snake by 1 and yields a fractional\n"
            f"  reward of 1/{self._target_food}; cumulative reward = 1.0 after "
            f"{self._target_food} foods.\n"
            "- The game ends if you hit a wall or your own body. Moving into\n"
            "  the current tail cell is safe when you are not eating food.\n"
            "- You cannot reverse direction (e.g., moving RIGHT cannot switch to LEFT).\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class SnakeEasyEnv(_SnakeBase):
    _grid_size = 10
    _difficulty = "easy"
    _target_food_goal = 5

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)


class SnakeMediumEnv(_SnakeBase):
    _grid_size = 12
    _difficulty = "medium"
    _target_food_goal = 10

    def __init__(self, max_turns: int = 300) -> None:
        super().__init__(max_turns=max_turns)


class SnakeHardEnv(_SnakeBase):
    _grid_size = 15
    _difficulty = "hard"
    _target_food_goal = 20

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
