"""Procgen CoinRun environment.

Side-scrolling platformer. Agent runs right, jumps over obstacles, and
collects a coin at the end of the level. Level is procedurally generated
from the seed.

Gym ID: glyphbench/procgen-coinrun-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import JUMP_ARC_DY, VIEW_HEIGHT, VIEW_WIDTH

# Cell types in the level
CELL_EMPTY = "\u00b7"
CELL_GROUND = "\u25ac"
CELL_PLATFORM = "\u2588"
CELL_PIT = "P"
CELL_SAW = "S"
CELL_ENEMY_SMALL = "m"
CELL_ENEMY_LARGE = "M"
CELL_COIN = "C"
CELL_AGENT = "@"


class CoinRunEnv(BaseGlyphEnv):
    """Procgen CoinRun: side-scrolling platformer with procedural levels.

    The agent sees a 20x12 partial-observation window centered on itself.
    The level scrolls as the agent moves right. Collect the coin at the
    end of the level for +1 reward. Death (pit/saw/enemy) gives 0.

    Actions: NOOP, LEFT, RIGHT, JUMP, JUMP_RIGHT
    """

    # Reward shaping (Pattern A): single terminal goal worth +1.0.
    _GOAL_REWARD = 1.0

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "JUMP", "JUMP_RIGHT"),
        descriptions=(
            "do nothing this step",
            "move one cell left",
            "move one cell right",
            "jump straight up (if on ground)",
            "jump and move right simultaneously",
        ),
    )
    noop_action_name = "NOOP"

    # Level geometry constants
    _LEVEL_HEIGHT = 12  # matches view height
    _BASE_LEVEL_WIDTH = 40
    _GROUND_ROW = 9  # ground level (rows 9, 10, 11 are ground by default)
    _GROUND_DEPTH = 3  # rows of ground

    _DIR_CHARS: dict[tuple[int, int], str] = {
        (1, 0): "\u2192", (-1, 0): "\u2190",
        (0, -1): "\u2191", (0, 1): "\u2193", (0, 0): "@",
    }
    _DIR_NAMES: dict[tuple[int, int], str] = {
        (1, 0): "right", (-1, 0): "left",
        (0, -1): "up", (0, 1): "down", (0, 0): "none",
    }

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._level_data: list[list[str]] = []
        self._level_width: int = 0
        self._agent_x: int = 0
        self._agent_y: int = 0
        self._agent_dir: tuple[int, int] = (0, 0)
        self._on_ground: bool = True
        self._jump_step: int = -1  # -1 = not jumping, 0..len(arc)-1 = in arc
        self._coin_x: int = 0
        self._coin_y: int = 0
        self._camera_x: int = 0
        self._alive: bool = True
        self._message: str = ""
        self._ground_y: int = self._GROUND_ROW  # top of ground
        self._vel_x: int = 0
        self._vel_y: int = 0
        self._level_seed: int = 0

    def env_id(self) -> str:
        return "glyphbench/procgen-coinrun-v0"

    def system_prompt(self) -> str:
        return (
            "You are playing Procgen CoinRun.\n\n"
            "TASK\n"
            "Move through a side-scrolling level, jump over obstacles, "
            "and collect the coin (C) for +1 reward. Falling into "
            "a pit (P), touching a saw blade (S), or hitting an enemy (m/M) "
            "kills you (0 reward).\n\n"
            "VIEW\n"
            "You see a 20x12 window centered on your position. The level "
            "extends beyond the visible area. As you move right, the view "
            "scrolls.\n\n"
            "PHYSICS\n"
            "Integer-cell movement. LEFT/RIGHT move one cell horizontally. "
            "JUMP rises for 3 steps, plateaus for 1, then falls for 3 steps "
            "(total arc: 7 steps, peak height: 3 cells above launch). "
            "JUMP_RIGHT combines jumping with rightward movement. You cannot "
            "double-jump. Gravity pulls you down 1 cell per step when in the "
            "air and not jumping.\n\n"
            "GRID\n"
            "The level is rendered as a Unicode glyph grid. Each cell is one "
            "glyph; per-turn observations include a legend block listing "
            "every glyph visible right now.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _generate_level(self) -> None:
        """Procedurally generate a level from self.rng."""
        self._level_width = self._BASE_LEVEL_WIDTH + int(self.rng.integers(0, 20))
        self._level_data = []

        for y in range(self._LEVEL_HEIGHT):
            row: list[str] = []
            for _x in range(self._level_width):
                if y >= self._ground_y:
                    row.append(CELL_GROUND)
                else:
                    row.append(CELL_EMPTY)
            self._level_data.append(row)

        # Place pits (gaps in ground)
        num_pits = int(self.rng.integers(2, 5))
        pit_positions: list[int] = []
        for _ in range(num_pits):
            pit_x = int(self.rng.integers(8, self._level_width - 8))
            pit_width = int(self.rng.integers(2, 4))
            # Don't place pits too close to start or end
            if pit_x < 5 or pit_x + pit_width > self._level_width - 5:
                continue
            # Don't overlap with other pits
            overlap = False
            for px in pit_positions:
                if abs(pit_x - px) < 5:
                    overlap = True
                    break
            if overlap:
                continue
            pit_positions.append(pit_x)
            for dx in range(pit_width):
                for dy in range(self._GROUND_DEPTH):
                    if pit_x + dx < self._level_width:
                        self._level_data[self._ground_y + dy][pit_x + dx] = CELL_PIT

        # Place platforms
        num_platforms = int(self.rng.integers(3, 7))
        for _ in range(num_platforms):
            plat_x = int(self.rng.integers(5, self._level_width - 5))
            plat_y = int(self.rng.integers(self._ground_y - 5, self._ground_y - 1))
            plat_width = int(self.rng.integers(3, 6))
            if plat_y < 1:
                plat_y = 1
            for dx in range(plat_width):
                if plat_x + dx < self._level_width:
                    self._level_data[plat_y][plat_x + dx] = CELL_PLATFORM

        # Place saw blades (on ground level, between pits)
        num_saws = int(self.rng.integers(1, 4))
        for _ in range(num_saws):
            saw_x = int(self.rng.integers(10, self._level_width - 10))
            if self._level_data[self._ground_y][saw_x] == CELL_GROUND:
                self._level_data[self._ground_y - 1][saw_x] = CELL_SAW

        # Place enemies (on ground)
        num_enemies = int(self.rng.integers(1, 3))
        for _ in range(num_enemies):
            enemy_x = int(self.rng.integers(10, self._level_width - 10))
            if self._level_data[self._ground_y][enemy_x] == CELL_GROUND:
                enemy_type = CELL_ENEMY_SMALL if self.rng.random() < 0.6 else CELL_ENEMY_LARGE
                self._level_data[self._ground_y - 1][enemy_x] = enemy_type

        # Place coin at end of level
        self._coin_x = self._level_width - 3
        self._coin_y = self._ground_y - 1
        self._level_data[self._coin_y][self._coin_x] = CELL_COIN

    def _get_cell(self, x: int, y: int) -> str:
        """Get cell value, bounds-checked."""
        if 0 <= x < self._level_width and 0 <= y < self._LEVEL_HEIGHT:
            return self._level_data[y][x]
        return CELL_EMPTY

    def _set_cell(self, x: int, y: int, value: str) -> None:
        """Set cell value, bounds-checked."""
        if 0 <= x < self._level_width and 0 <= y < self._LEVEL_HEIGHT:
            self._level_data[y][x] = value

    def _is_solid(self, x: int, y: int) -> bool:
        """Check if a cell is solid (can be stood on)."""
        cell = self._get_cell(x, y)
        return cell in (CELL_GROUND, CELL_PLATFORM)

    def _is_deadly(self, x: int, y: int) -> bool:
        """Check if a cell kills the agent."""
        cell = self._get_cell(x, y)
        return cell in (CELL_PIT, CELL_SAW, CELL_ENEMY_SMALL, CELL_ENEMY_LARGE)

    def _reset(self, seed: int) -> GridObservation:
        self._level_seed = seed
        self._generate_level()
        self._agent_x = 2
        self._agent_y = self._ground_y - 1  # on top of ground
        self._agent_dir = (0, 0)
        self._on_ground = True
        self._jump_step = -1
        self._camera_x = 0
        self._alive = True
        self._message = ""
        self._vel_x = 0
        self._vel_y = 0
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        self._message = ""
        reward = 0.0
        terminated = False

        if not self._alive:
            # Already dead, just return
            return self._render_current_observation(), 0.0, True, False, self._build_info(True)

        # --- 1. Horizontal movement ---
        dx = 0
        if name in ("LEFT",):
            dx = -1
            self._agent_dir = (-1, 0)
        elif name in ("RIGHT", "JUMP_RIGHT"):
            dx = 1
            self._agent_dir = (1, 0)

        # --- 2. Initiate jump ---
        if name in ("JUMP",) and self._on_ground:
            self._agent_dir = (0, -1)
        if name in ("JUMP", "JUMP_RIGHT") and self._on_ground:
            self._jump_step = 0
            self._on_ground = False

        # --- 3. Apply jump arc or gravity ---
        dy = 0
        if self._jump_step >= 0:
            if self._jump_step < len(JUMP_ARC_DY):
                dy = JUMP_ARC_DY[self._jump_step]
                self._jump_step += 1
            else:
                # Jump arc finished, start falling
                self._jump_step = -1
                dy = 1  # gravity
        elif not self._on_ground:
            dy = 1  # gravity

        # --- 4. Apply movement ---
        new_x = self._agent_x + dx
        new_y = self._agent_y + dy

        # Horizontal bounds
        if new_x < 0:
            new_x = 0
        elif new_x >= self._level_width:
            new_x = self._level_width - 1

        # Vertical: check solid collision
        if dy > 0:
            # Moving down (falling or gravity)
            if self._is_solid(new_x, new_y):
                # Land on surface
                new_y = self._agent_y  # stay at current y
                self._on_ground = True
                self._jump_step = -1
            elif new_y >= self._LEVEL_HEIGHT:
                # Fell off the bottom
                new_y = self._LEVEL_HEIGHT - 1
                # Check if this is a pit
                if self._is_deadly(new_x, new_y):
                    self._alive = False
                    terminated = True
                    self._message = "You fell into a pit and died."
                else:
                    self._on_ground = True
                    self._jump_step = -1
        elif dy < 0:
            # Moving up (jumping)
            if self._is_solid(new_x, new_y):
                # Hit ceiling/platform from below
                new_y = self._agent_y  # stay at current y
                self._jump_step = -1  # cancel jump
        else:
            # No vertical movement, check if still on ground
            if not self._is_solid(new_x, self._agent_y + 1) and self._on_ground:
                # No ground beneath -- start falling
                    self._on_ground = False

        # Horizontal collision with solid
        if self._is_solid(new_x, new_y) and dx != 0:
            new_x = self._agent_x  # can't move horizontally into wall

        self._agent_x = new_x
        self._agent_y = new_y

        # --- 5. Check hazards at new position ---
        if self._alive and self._is_deadly(self._agent_x, self._agent_y):
            cell = self._get_cell(self._agent_x, self._agent_y)
            self._alive = False
            terminated = True
            if cell == CELL_PIT:
                self._message = "You fell into a pit and died."
            elif cell == CELL_SAW:
                self._message = "You hit a saw blade and died."
            elif cell in (CELL_ENEMY_SMALL, CELL_ENEMY_LARGE):
                self._message = "You were killed by an enemy."

        # --- 6. Check coin collection ---
        if self._alive and self._agent_x == self._coin_x and self._agent_y == self._coin_y:
            reward = self._GOAL_REWARD
            terminated = True
            self._message = "You got the coin!"

        # --- 7. Check if on ground after all movement ---
        if self._alive and not terminated:
            if self._is_solid(self._agent_x, self._agent_y + 1):
                if self._jump_step < 0:
                    self._on_ground = True
            else:
                if self._jump_step < 0:
                    self._on_ground = False

        # --- 8. Update camera ---
        # Camera follows agent, keeping agent roughly centered
        target_camera = self._agent_x - VIEW_WIDTH // 2
        if target_camera < 0:
            target_camera = 0
        if target_camera > self._level_width - VIEW_WIDTH:
            target_camera = max(0, self._level_width - VIEW_WIDTH)
        self._camera_x = target_camera

        self._vel_x = dx
        self._vel_y = dy

        return (
            self._render_current_observation(),
            reward,
            terminated,
            False,
            self._build_info(terminated),
        )

    def _build_info(self, terminated: bool) -> dict[str, Any]:
        killed_by: str | None = None
        if not self._alive:
            cell = self._get_cell(self._agent_x, self._agent_y)
            if cell == CELL_PIT:
                killed_by = "pit"
            elif cell == CELL_SAW:
                killed_by = "saw"
            elif cell in (CELL_ENEMY_SMALL, CELL_ENEMY_LARGE):
                killed_by = "enemy"
            else:
                killed_by = "unknown"
        return {
            "level_seed": self._level_seed,
            "agent_pos": (self._agent_x, self._agent_y),
            "coins_remaining": (
                0
                if (not self._alive or self._message == "You got the coin!")
                else 1
            ),
            "distance_to_finish": max(0, self._coin_x - self._agent_x),
            "killed_by": killed_by,
        }

    def _render_current_observation(self) -> GridObservation:
        # Build 20x12 window
        pch = self._DIR_CHARS.get(self._agent_dir, "@")
        dir_name = self._DIR_NAMES.get(
            self._agent_dir, "none"
        )
        grid: list[list[str]] = []
        for wy in range(VIEW_HEIGHT):
            row: list[str] = []
            for wx in range(VIEW_WIDTH):
                world_x = self._camera_x + wx
                world_y = wy
                if (
                    world_x == self._agent_x
                    and world_y == self._agent_y
                    and self._alive
                ):
                    row.append(pch)
                else:
                    row.append(self._get_cell(world_x, world_y))
            grid.append(row)

        alive_str = "yes" if self._alive else "no"
        state = "grounded" if self._on_ground else "airborne"
        hud = (
            f"Level seed: {self._level_seed}    "
            f"Step: {self._turn} / {self.max_turns}    "
            f"Vel: ({self._vel_x:+d}, {self._vel_y:+d})    "
            f"State: {state}    "
            f"Alive: {alive_str}"
        )

        legend = build_legend({
            pch: f"you (facing {dir_name})",
            "\u25ac": "ground",
            "\u2588": "platform",
            "P": "pit (deadly)",
            "S": "saw blade (deadly)",
            "m": "small enemy (deadly)",
            "M": "large enemy (deadly)",
            "C": "coin (goal, +1 reward)",
            "\u00b7": "air (empty)",
        })

        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message=self._message,
        )
