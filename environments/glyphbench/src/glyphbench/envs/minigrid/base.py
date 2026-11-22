"""Shared base class for all MiniGrid envs.

MiniGrid envs share a common 7-action space and wall-rendering conventions.
Individual envs override mechanics (grid contents, goal conditions, etc.).

``MiniGridBase`` owns the full grid state (object grid, agent position/direction,
carrying slot) and implements the 7-action dispatch loop. Concrete subclasses
only need to implement ``_generate_grid(seed)`` and ``env_id()``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation
from glyphbench.envs.minigrid.objects import (
    Box,
    Door,
    Goal,
    Lava,
    Wall,
    Water,
    WorldObject,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

MINIGRID_ACTION_SPEC = ActionSpec(
    names=(
        "TURN_LEFT",
        "TURN_RIGHT",
        "MOVE_FORWARD",
        "PICKUP",
        "DROP",
        "TOGGLE",
        "DONE",
    ),
    descriptions=(
        "rotate 90 degrees counter-clockwise",
        "rotate 90 degrees clockwise",
        "move one cell in the direction you are facing",
        "pick up an object in the cell ahead",
        "drop the carried object in the cell ahead",
        "toggle/activate the object in the cell ahead",
        "declare the task is done (no-op)",
    ),
)

# Direction constants: 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP
DIR_RIGHT = 0
DIR_DOWN = 1
DIR_LEFT = 2
DIR_UP = 3

# Direction vectors: (dx, dy) for each direction
DIR_TO_VEC: dict[int, tuple[int, int]] = {
    DIR_RIGHT: (1, 0),
    DIR_DOWN: (0, 1),
    DIR_LEFT: (-1, 0),
    DIR_UP: (0, -1),
}

# Agent glyphs per direction (Unicode arrows to avoid collision with other symbols)
DIR_TO_CHAR: dict[int, str] = {
    DIR_RIGHT: "\u2192",  # →
    DIR_DOWN: "\u2193",   # ↓
    DIR_LEFT: "\u2190",   # ←
    DIR_UP: "\u2191",     # ↑
}

# ---------------------------------------------------------------------------
# MiniGridBase
# ---------------------------------------------------------------------------


class MiniGridBase(BaseGlyphEnv):
    """Abstract base for all MiniGrid environments.

    Subclasses MUST implement:
      - ``env_id()`` -> str
      - ``_generate_grid(seed)`` -> None  (must call ``_init_grid``,
        ``_place_agent``, and place any task-specific objects)

    Subclasses MAY override:
      - ``_task_description()`` for a custom task paragraph in the system prompt
      - ``_reward_on_goal()`` for custom goal rewards
    """

    action_spec = MINIGRID_ACTION_SPEC
    noop_action_name: str = "DONE"

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        # Grid state -- populated by _generate_grid via _init_grid
        self._grid: list[list[WorldObject | None]] = []
        self._grid_w: int = 0
        self._grid_h: int = 0
        # Agent state
        self._agent_pos: tuple[int, int] = (0, 0)
        self._agent_dir: int = DIR_RIGHT
        self._carrying: WorldObject | None = None
        # Goal tracking
        self._goal_pos: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Grid setup helpers (called from _generate_grid)
    # ------------------------------------------------------------------

    def _init_grid(self, width: int, height: int) -> None:
        """Create a *width x height* grid with Wall borders and None interior."""
        self._grid_w = width
        self._grid_h = height
        self._grid = [[None for _ in range(width)] for _ in range(height)]
        # Place walls around the border
        for x in range(width):
            self._grid[0][x] = Wall()
            self._grid[height - 1][x] = Wall()
        for y in range(1, height - 1):
            self._grid[y][0] = Wall()
            self._grid[y][width - 1] = Wall()

    def _place_agent(self, x: int, y: int, direction: int = DIR_RIGHT) -> None:
        """Set the agent's starting position and direction."""
        self._agent_pos = (x, y)
        self._agent_dir = direction

    def _place_obj(self, x: int, y: int, obj: WorldObject) -> None:
        """Place an object on the grid. Tracks Goal position automatically."""
        self._grid[y][x] = obj
        if isinstance(obj, Goal):
            self._goal_pos = (x, y)

    def _get_obj(self, x: int, y: int) -> WorldObject | None:
        """Return the object at (x, y), or None."""
        return self._grid[y][x]

    def _front_pos(self) -> tuple[int, int]:
        """Cell directly ahead of the agent."""
        dx, dy = DIR_TO_VEC[self._agent_dir]
        return (self._agent_pos[0] + dx, self._agent_pos[1] + dy)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _reward_on_goal(self) -> float:
        """Reward granted when the agent reaches the goal.

        Default: 1 - 0.9 * (step_count / max_turns), so faster = better.
        """
        step_count = self._turn
        return 1.0 - 0.9 * (step_count / self.max_turns)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._carrying = None
        self._goal_pos = None
        self._generate_grid(seed)
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        if name == "TURN_LEFT":
            self._agent_dir = (self._agent_dir - 1) % 4
        elif name == "TURN_RIGHT":
            self._agent_dir = (self._agent_dir + 1) % 4
        elif name == "MOVE_FORWARD":
            fx, fy = self._front_pos()
            if 0 <= fx < self._grid_w and 0 <= fy < self._grid_h:
                obj = self._get_obj(fx, fy)
                if obj is None or obj.can_overlap:
                    self._agent_pos = (fx, fy)
        elif name == "PICKUP":
            fx, fy = self._front_pos()
            if 0 <= fx < self._grid_w and 0 <= fy < self._grid_h:
                obj = self._get_obj(fx, fy)
                if obj is not None and obj.can_pickup and self._carrying is None:
                    self._carrying = obj
                    self._grid[fy][fx] = None
        elif name == "DROP":
            fx, fy = self._front_pos()
            if (
                0 <= fx < self._grid_w
                and 0 <= fy < self._grid_h
                and self._carrying is not None
                and self._get_obj(fx, fy) is None
            ):
                self._grid[fy][fx] = self._carrying
                self._carrying = None
        elif name == "TOGGLE":
            fx, fy = self._front_pos()
            if 0 <= fx < self._grid_w and 0 <= fy < self._grid_h:
                obj = self._get_obj(fx, fy)
                if isinstance(obj, Door):
                    obj.toggle(self._carrying)
                elif isinstance(obj, Box):
                    contents = obj.toggle()
                    # Replace the closed box with its contents (or clear
                    # the cell if the box was empty).
                    self._grid[fy][fx] = contents
        # DONE: no-op

        # --- Post-action checks ---
        ax, ay = self._agent_pos

        # Lava: terminates with zero reward
        cell = self._get_obj(ax, ay)
        if isinstance(cell, Lava):
            terminated = True
            info["lava"] = True
            info["agent_pos"] = self._agent_pos
            return self._render_current_observation(), 0.0, terminated, False, info

        # Water: flag only
        if isinstance(cell, Water):
            info["water"] = True

        # Goal check
        if self._goal_pos is not None and self._agent_pos == self._goal_pos:
            reward = self._reward_on_goal()
            terminated = True
            info["goal_reached"] = True

        info["agent_pos"] = self._agent_pos
        return self._render_current_observation(), reward, terminated, False, info

    @abstractmethod
    def _generate_grid(self, seed: int) -> None:
        """Populate the grid. Must call ``_init_grid``, ``_place_agent``, and
        place any task-specific objects."""
        ...

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        render_grid = make_empty_grid(self._grid_w, self._grid_h)
        symbol_meanings: dict[str, str] = {
            "\u00b7": "floor",
        }

        for y in range(self._grid_h):
            for x in range(self._grid_w):
                obj = self._grid[y][x]
                if obj is not None:
                    ch = obj.render_char()
                    render_grid[y][x] = ch
                    if ch not in symbol_meanings:
                        symbol_meanings[ch] = obj.legend_name()

        # Stamp the agent on top
        agent_char = DIR_TO_CHAR[self._agent_dir]
        ax, ay = self._agent_pos
        render_grid[ay][ax] = agent_char

        # Only the actually-rendered facing glyph goes in the legend; the
        # other three would just be noise for the LLM.
        _facing_name = {
            "\u2192": "right", "\u2193": "down",
            "\u2190": "left", "\u2191": "up",
        }[agent_char]
        symbol_meanings[agent_char] = f"you, facing {_facing_name}"

        legend = build_legend(symbol_meanings)

        carrying_str = (
            f"Carrying: {self._carrying.legend_name()}"
            if self._carrying is not None
            else "Carrying: nothing"
        )
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"{carrying_str}"
        )

        return GridObservation(
            grid=grid_to_string(render_grid),
            legend=legend,
            hud=hud,
            message="",
        )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _task_description(self) -> str:
        """Override in subclasses for a custom task paragraph."""
        goal = Goal().render_char()
        return (
            f"Navigate the grid to reach the goal ({goal}). You earn a reward "
            "based on how quickly you reach the goal: reward = 1 - 0.9 * "
            "(steps_taken / max_steps). The faster you reach the goal, the "
            "higher your reward."
        )

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            f"TASK\n{self._task_description()}\n\n"
            "GRID\n"
            "The room is rendered as a Unicode glyph grid. Each cell is one "
            "glyph; per-turn observations include a legend block listing "
            "every glyph visible right now.\n\n"
            "MOVEMENT\n"
            "You move relative to your facing direction. MOVE_FORWARD advances "
            "one cell in the direction you face. TURN_LEFT and TURN_RIGHT rotate "
            "you 90 degrees. Bumping into a wall or ordinary solid object does "
            "nothing; task-specific hazards can override this when touched.\n\n"
            "INTERACTIONS\n"
            "PICKUP picks up the object in the cell you are facing (if it can be "
            "picked up and you are not already carrying something). DROP drops "
            "what you are carrying into the cell ahead (if that cell is empty). "
            "TOGGLE opens/closes doors ahead of you. Locked doors open only "
            "when you are carrying a matching-colour key; the key stays in "
            "your hand. Some envs allow TOGGLE on a closed box to reveal its "
            "contents.\n\n"
            + self.action_spec.render_for_prompt()
        )
