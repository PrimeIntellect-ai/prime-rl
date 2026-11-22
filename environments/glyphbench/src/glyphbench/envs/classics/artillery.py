"""Artillery: Worms/Angry Birds inspired angle+power shooting game.

Gym IDs:
  glyphbench/classics-artillery-v0
"""

from __future__ import annotations

import math
from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTILLERY_ACTION_SPEC = ActionSpec(
    names=("AIM_UP", "AIM_DOWN", "POWER_UP", "POWER_DOWN", "FIRE"),
    descriptions=(
        "increase launch angle by 5 degrees",
        "decrease launch angle by 5 degrees",
        "increase power by 1",
        "decrease power by 1",
        "fire the projectile at current angle and power",
    ),
)

GRID_W = 30
GRID_H = 15
ANGLE_MIN = 10
ANGLE_MAX = 80
ANGLE_STEP = 5
POWER_MIN = 1
POWER_MAX = 10
MAX_SHOTS = 30
NUM_TARGETS = 3
CRATER_RADIUS = 1

SYM_CANNON = "\u25b2"   # ▲
SYM_TARGET = "\u2605"   # ★
SYM_PROJECTILE = "\u25cf"  # ●
SYM_TERRAIN = "\u2593"  # ▓
SYM_AIR = "\u00b7"      # ·
SYM_CRATER = "\u25cb"   # ○

# Gravity constant for arc simulation
GRAVITY = 0.3

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class ArtilleryEnv(BaseGlyphEnv):
    """Aim angle + power, fire projectile to hit targets on terrain."""

    action_spec = ARTILLERY_ACTION_SPEC
    noop_action_name: str = "AIM_UP"  # no true noop; aim_up is least harmful

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._angle: int = 45  # degrees
        self._power: int = 5
        self._shots_taken: int = 0
        self._terrain: list[int] = []  # terrain height at each x (from bottom)
        self._targets: list[tuple[int, int]] = []  # (x, y) world coords
        self._craters: set[tuple[int, int]] = set()
        self._cannon_x: int = 2
        self._cannon_y: int = 0  # will be set based on terrain
        self._arc_trail: list[tuple[int, int]] = []  # projectile path (visual, 1 step)
        self._message: str = ""

    def env_id(self) -> str:
        return "glyphbench/classics-artillery-v0"

    # ------------------------------------------------------------------
    # Terrain generation
    # ------------------------------------------------------------------

    def _generate_terrain(self) -> None:
        """Generate a heightmap for the terrain."""
        self._terrain = []
        height = int(self.rng.integers(3, 5))

        for _x in range(GRID_W):
            self._terrain.append(height)
            # Random walk with smoothing
            delta = int(self.rng.integers(-1, 2))  # -1, 0, or 1
            height = max(2, min(GRID_H - 3, height + delta))

        # Ensure cannon area is flat
        cannon_h = self._terrain[self._cannon_x]
        for x in range(0, self._cannon_x + 2):
            self._terrain[x] = cannon_h

        # Set cannon position (on top of terrain)
        self._cannon_y = GRID_H - 1 - cannon_h

    def _place_targets(self) -> None:
        """Place targets on terrain surface, spread across the level."""
        self._targets = []
        # Divide the right portion into zones for spread
        zone_start = GRID_W // 3
        zone_width = (GRID_W - zone_start - 1) // NUM_TARGETS

        for i in range(NUM_TARGETS):
            zs = zone_start + i * zone_width
            ze = min(zs + zone_width, GRID_W - 1)
            tx = int(self.rng.integers(zs, ze))
            ty = GRID_H - 1 - self._terrain[tx]  # on terrain surface
            self._targets.append((tx, ty))

    # ------------------------------------------------------------------
    # Arc simulation
    # ------------------------------------------------------------------

    def _simulate_arc(self) -> tuple[list[tuple[int, int]], tuple[int, int] | None, str]:
        """Simulate projectile arc. Returns (trail, hit_pos, result).

        result is one of: 'target_hit', 'terrain_hit', 'miss'
        """
        angle_rad = math.radians(self._angle)
        vx = self._power * math.cos(angle_rad)
        vy = -self._power * math.sin(angle_rad)  # negative = upward (y increases down)

        # Launch from just above cannon
        fx = float(self._cannon_x) + 0.5
        fy = float(self._cannon_y) - 0.5

        trail: list[tuple[int, int]] = []
        visited: set[tuple[int, int]] = set()
        dt = 0.15  # time step for simulation

        for _ in range(500):  # safety limit
            fx += vx * dt
            fy += vy * dt
            vy += GRAVITY * dt  # gravity

            gx = int(round(fx))
            gy = int(round(fy))

            # Out of bounds
            if gx < 0 or gx >= GRID_W or gy >= GRID_H:
                return trail, None, "miss"
            if gy < 0:
                continue  # still going up off screen

            cell = (gx, gy)
            if cell not in visited:
                visited.add(cell)
                trail.append(cell)

            # Check target hit
            for target in self._targets:
                if gx == target[0] and gy == target[1]:
                    return trail, target, "target_hit"

            # Check terrain hit
            terrain_top_y = GRID_H - 1 - self._terrain[gx]
            if gy >= terrain_top_y:
                return trail, (gx, gy), "terrain_hit"

        return trail, None, "miss"

    def _create_crater(self, cx: int, cy: int) -> None:
        """Destroy terrain in a radius around impact point."""
        for dx in range(-CRATER_RADIUS, CRATER_RADIUS + 1):
            for dy in range(-CRATER_RADIUS, CRATER_RADIUS + 1):
                if dx * dx + dy * dy <= CRATER_RADIUS * CRATER_RADIUS + 1:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                        # Lower terrain at this x if the crater is at or below surface
                        terrain_top = GRID_H - 1 - self._terrain[nx]
                        if ny >= terrain_top:
                            self._craters.add((nx, ny))
                            # Reduce terrain height
                            new_height = GRID_H - 1 - ny
                            if new_height < self._terrain[nx]:
                                self._terrain[nx] = max(0, new_height - 1)

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._angle = 45
        self._power = 5
        self._shots_taken = 0
        self._craters = set()
        self._arc_trail = []
        self._message = "Aim and fire to destroy all targets!"
        self._generate_terrain()
        self._place_targets()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        reward = 0.0
        terminated = False
        self._arc_trail = []
        self._message = ""

        name = self.action_spec.names[action]

        if name == "AIM_UP":
            self._angle = min(ANGLE_MAX, self._angle + ANGLE_STEP)
            self._message = f"Angle: {self._angle}\u00b0"
        elif name == "AIM_DOWN":
            self._angle = max(ANGLE_MIN, self._angle - ANGLE_STEP)
            self._message = f"Angle: {self._angle}\u00b0"
        elif name == "POWER_UP":
            self._power = min(POWER_MAX, self._power + 1)
            self._message = f"Power: {self._power}"
        elif name == "POWER_DOWN":
            self._power = max(POWER_MIN, self._power - 1)
            self._message = f"Power: {self._power}"
        elif name == "FIRE":
            self._shots_taken += 1
            trail, hit_pos, result = self._simulate_arc()
            self._arc_trail = trail

            if result == "target_hit" and hit_pos is not None:
                # Remove the hit target
                self._targets = [t for t in self._targets if t != hit_pos]
                # Pattern A: each target hit yields 1/N_TARGETS so total = 1.0
                reward = 1.0 / NUM_TARGETS
                self._message = f"Target hit! {len(self._targets)} remaining."
                self._create_crater(hit_pos[0], hit_pos[1])

                if len(self._targets) == 0:
                    terminated = True
                    self._message = "All targets destroyed! Victory!"
                    info["outcome"] = "victory"
            elif result == "terrain_hit" and hit_pos is not None:
                reward = 0.0
                self._create_crater(hit_pos[0], hit_pos[1])
                self._message = "Hit terrain. Try adjusting angle or power."
            else:
                reward = 0.0
                self._message = "Shot went off the map. Try adjusting."

            # Check max shots
            if self._shots_taken >= MAX_SHOTS and not terminated:
                terminated = True
                self._message = "Out of shots! Game over."
                info["outcome"] = "out_of_shots"

        info["shots_taken"] = self._shots_taken
        info["targets_remaining"] = len(self._targets)
        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(GRID_W, GRID_H, SYM_AIR)

        # Draw terrain
        for x in range(GRID_W):
            h = self._terrain[x]
            for dy in range(h):
                y = GRID_H - 1 - dy
                if 0 <= y < GRID_H:
                    grid[y][x] = SYM_TERRAIN

        # Draw craters
        for cx, cy in self._craters:
            if 0 <= cx < GRID_W and 0 <= cy < GRID_H:
                grid[cy][cx] = SYM_CRATER

        # Draw arc trail
        for tx, ty in self._arc_trail:
            if 0 <= tx < GRID_W and 0 <= ty < GRID_H and grid[ty][tx] == SYM_AIR:
                grid[ty][tx] = SYM_PROJECTILE

        # Draw targets
        for tx, ty in self._targets:
            if 0 <= tx < GRID_W and 0 <= ty < GRID_H:
                grid[ty][tx] = SYM_TARGET

        # Draw cannon
        if 0 <= self._cannon_y < GRID_H:
            grid[self._cannon_y][self._cannon_x] = SYM_CANNON

        legend = build_legend({
            SYM_CANNON: "your cannon",
            SYM_TARGET: "target (destroy these)",
            SYM_PROJECTILE: "projectile trail",
            SYM_TERRAIN: "terrain",
            SYM_AIR: "air",
            SYM_CRATER: "crater",
        })

        shots_remaining = MAX_SHOTS - self._shots_taken
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Angle: {self._angle}\u00b0    "
            f"Power: {self._power}    "
            f"Shots remaining: {shots_remaining}    "
            f"Targets remaining: {len(self._targets)}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message=self._message,
        )

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "You control a cannon on the left side of a terrain map. Aim and adjust "
            f"power to hit {NUM_TARGETS} targets scattered across the landscape.\n\n"
            "RULES\n"
            f"- Grid is {GRID_W} wide x {GRID_H} tall with hilly terrain.\n"
            f"- Angle range: {ANGLE_MIN}\u00b0 to {ANGLE_MAX}\u00b0 "
            f"(in {ANGLE_STEP}\u00b0 increments). Higher angle = more arc.\n"
            f"- Power range: {POWER_MIN} to {POWER_MAX}. Higher power = farther shot.\n"
            "- FIRE launches a projectile that follows a parabolic arc.\n"
            f"- Hitting a target destroys it (+{1.0 / NUM_TARGETS:.4f} reward).\n"
            "- Hitting terrain creates a crater (destroys 1-cell radius).\n"
            "- Missing or hitting terrain yields 0 reward.\n"
            f"- All {NUM_TARGETS} targets destroyed = victory (cumulative reward = 1.0).\n"
            f"- Maximum {MAX_SHOTS} shots before game over.\n"
            "- AIM_UP/AIM_DOWN and POWER_UP/POWER_DOWN do not consume a shot.\n\n"
            "STRATEGY\n"
            "- Observe target positions relative to your cannon.\n"
            "- Higher angle with moderate power for nearby targets.\n"
            "- Lower angle with high power for distant targets.\n"
            "- Use crater impacts to gauge your aim.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
