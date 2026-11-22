"""Lunar Lander -- text-based spacecraft landing.

Gym IDs:
  glyphbench/classics-lunarlander-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIDTH = 20
HEIGHT = 15
PAD_WIDTH = 3
PAD_Y = HEIGHT - 1  # bottom row
PAD_CENTER = WIDTH // 2  # 10

GRAVITY = 0.5
MAX_FUEL = 100
THRUST_COST = 1
THRUST_POWER = 0.8  # thrust counteracts/exceeds gravity

LUNAR_ACTION_SPEC = ActionSpec(
    names=("THRUST_UP", "THRUST_LEFT", "THRUST_RIGHT", "NOOP"),
    descriptions=(
        "fire main thruster — pushes craft upward",
        "fire side thruster — pushes craft to the LEFT",
        "fire side thruster — pushes craft to the RIGHT",
        "do nothing, drift with gravity",
    ),
)

SYM_CRAFT = "\u25bc"    # ▼
SYM_PAD = "\u25ac"       # ▬
SYM_AIR = "\u00b7"       # ·
SYM_GROUND = "\u2588"    # █
SYM_FLAME = "*"

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class LunarLanderEnv(BaseGlyphEnv):
    """Land a spacecraft on the landing pad with low velocity."""

    action_spec = LUNAR_ACTION_SPEC
    noop_action_name: str = "NOOP"

    def __init__(self, max_turns: int = 300) -> None:
        super().__init__(max_turns=max_turns)
        self._x: float = 0.0
        self._y: float = 0.0
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._fuel: int = MAX_FUEL
        self._terrain: list[int] = []
        self._last_action: int = 3  # NOOP
        self._done: bool = False
        self._done_msg: str = ""

    def env_id(self) -> str:
        return "glyphbench/classics-lunarlander-v0"

    def _generate_terrain(self) -> list[int]:
        """Generate terrain heights for each column. Bottom row is y=HEIGHT-1."""
        terrain: list[int] = []
        base = HEIGHT - 1  # ground row
        for x in range(WIDTH):
            # Flat landing area in center
            pad_left = PAD_CENTER - PAD_WIDTH // 2
            pad_right = pad_left + PAD_WIDTH - 1
            if pad_left <= x <= pad_right:
                terrain.append(base)
            else:
                # Hills: sine-based terrain with some randomness
                hill = int(2 * np.sin(x * 0.7) + self.rng.integers(0, 2))
                h = max(0, hill)
                terrain.append(base - h)
        return terrain

    def _reset(self, seed: int) -> GridObservation:
        self._terrain = self._generate_terrain()
        # Start near top center with slight random offset and velocity
        self._x = float(PAD_CENTER + self.rng.uniform(-3, 3))
        self._y = float(1 + self.rng.uniform(0, 2))
        self._vx = float(self.rng.uniform(-0.5, 0.5))
        self._vy = 0.0
        self._fuel = MAX_FUEL
        self._last_action = 3  # NOOP
        self._done = False
        self._done_msg = ""
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        self._last_action = action
        name = self.action_spec.names[action]

        # Apply thrust if fuel available
        if name == "THRUST_UP" and self._fuel > 0:
            self._vy -= THRUST_POWER
            self._fuel -= THRUST_COST
        elif name == "THRUST_LEFT" and self._fuel > 0:
            self._vx -= THRUST_POWER * 0.6
            self._fuel -= THRUST_COST
        elif name == "THRUST_RIGHT" and self._fuel > 0:
            self._vx += THRUST_POWER * 0.6
            self._fuel -= THRUST_COST

        # Apply gravity
        self._vy += GRAVITY

        # Update position
        self._x += self._vx
        self._y += self._vy

        # Clamp fuel
        self._fuel = max(0, self._fuel)

        # Check off-screen
        gx = int(round(self._x))
        gy = int(round(self._y))

        if gx < 0 or gx >= WIDTH or self._y < -1:
            self._done = True
            self._done_msg = "Drifted off screen!"
            return self._render_current_observation(), -1.0, True, False, info

        if self._y >= HEIGHT + 2:
            self._done = True
            self._done_msg = "Fell off the bottom!"
            return self._render_current_observation(), -1.0, True, False, info

        # Check ground contact
        if gy >= 0 and gx >= 0 and gx < WIDTH:
            terrain_y = self._terrain[min(gx, WIDTH - 1)]
            if gy >= terrain_y:
                # Landed or crashed
                pad_left = PAD_CENTER - PAD_WIDTH // 2
                pad_right = pad_left + PAD_WIDTH - 1
                on_pad_columns = pad_left <= gx <= pad_right

                low_vel = abs(self._vx) <= 1.0 and abs(self._vy) <= 1.5

                if on_pad_columns and low_vel:
                    self._done = True
                    self._done_msg = "Successful landing!"
                    self._y = float(terrain_y)
                    return self._render_current_observation(), 1.0, True, False, info
                else:
                    self._done = True
                    if not on_pad_columns:
                        self._done_msg = "Crashed! Missed the landing pad."
                    else:
                        self._done_msg = f"Crashed! Too fast (vx={self._vx:.1f}, vy={self._vy:.1f})."
                        self._y = float(terrain_y)
                    return self._render_current_observation(), -1.0, True, False, info

        return self._render_current_observation(), 0.0, False, False, info

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(WIDTH, HEIGHT, SYM_AIR)

        # Draw terrain/ground
        for x in range(WIDTH):
            ty = self._terrain[x]
            for y in range(ty, HEIGHT):
                grid[y][x] = SYM_GROUND

        # Draw landing pad
        pad_left = PAD_CENTER - PAD_WIDTH // 2
        for i in range(PAD_WIDTH):
            px = pad_left + i
            if 0 <= px < WIDTH:
                grid[PAD_Y][px] = SYM_PAD

        # Draw craft
        gx = int(round(self._x))
        gy = int(round(self._y))
        if 0 <= gx < WIDTH and gy < HEIGHT:
            draw_y = max(0, gy)
            grid[draw_y][gx] = SYM_CRAFT

            # Draw thrust flame
            name = self.action_spec.names[self._last_action]
            if (
                name == "THRUST_UP"
                and draw_y + 1 < HEIGHT
                and grid[draw_y + 1][gx] == SYM_AIR
            ):
                grid[draw_y + 1][gx] = SYM_FLAME
            elif (
                name == "THRUST_LEFT"
                and gx + 1 < WIDTH
                and grid[draw_y][gx + 1] == SYM_AIR
            ):
                grid[draw_y][gx + 1] = SYM_FLAME
            elif (
                name == "THRUST_RIGHT"
                and gx - 1 >= 0
                and grid[draw_y][gx - 1] == SYM_AIR
            ):
                grid[draw_y][gx - 1] = SYM_FLAME

        legend = build_legend({
            SYM_CRAFT: "your spacecraft",
            SYM_PAD: "landing pad",
            SYM_AIR: "empty air",
            SYM_GROUND: "ground / terrain",
            SYM_FLAME: "thrust flame",
        })

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Vel: ({self._vx:.1f}, {self._vy:.1f})    "
            f"Fuel: {self._fuel}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message=self._done_msg,
        )

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Land your spacecraft on the landing pad at the bottom center of the screen.\n\n"
            "RULES\n"
            f"- The grid is {WIDTH} wide x {HEIGHT} tall.\n"
            f"- Landing pad (\u25ac) is {PAD_WIDTH} cells wide; its location is shown on the [Grid].\n"
            f"- Gravity pulls the craft down by {GRAVITY} per step.\n"
            "- Velocity is shown as (vx, vy); negative vy moves upward and positive vy moves downward.\n"
            "- Thrusters add velocity in the chosen direction.\n"
            f"- You start with {MAX_FUEL} fuel. Each thrust costs {THRUST_COST} fuel.\n"
            "- When fuel is 0, thrust actions have no effect.\n"
            "- Terrain outside the pad is uneven; touching any ground/terrain cell crashes you.\n"
            "- The thrust flame appears opposite the thrust direction.\n"
            "- Successful landing (on pad, |vx| <= 1, |vy| <= 1.5): +1 reward.\n"
            "- Crash (high velocity or off-pad): -1 reward.\n"
            "- Drift off screen: -1 reward.\n"
            "- Position updates: new_pos = old_pos + velocity each step.\n"
            "- THRUST_UP adds upward velocity (counteracts gravity).\n"
            "- THRUST_LEFT / THRUST_RIGHT add horizontal velocity.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
