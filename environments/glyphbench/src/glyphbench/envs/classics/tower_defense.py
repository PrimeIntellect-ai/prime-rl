"""Tower defense game.

Place towers along a path to stop enemy waves from reaching your base.

Gym IDs:
  glyphbench/classics-towerdefense-v0
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

GRID_SIZE = 11
TOWER_RANGE = 1
TOWER_COST = 3
STARTING_GOLD = 7  # Phase 2 retune: supports multi-tower play without trivial upgrades.
GOLD_PER_KILL = 1
TOTAL_WAVES = 5

# E4 (item slots): cap simultaneous towers; reuse beyond the cap requires
# upgrading what you already placed.
MAX_TOWERS = 3
# Linear-in-level upgrade pricing: upgrading a tower from level L to L+1 costs
# (L+1) * TOWER_COST gold. Each level adds +1 damage per attack.
UPGRADE_COST_PER_LEVEL = TOWER_COST
MAX_TOWER_LEVEL = 4

SYM_TOWER = "\u25b2"   # ▲
SYM_GROUND = "\u00b7"  # ·
SYM_PATH = "\u2550"    # ═
SYM_ENEMY = "E"
SYM_BASE = "\u2691"    # ⚑
SYM_WALL = "\u2588"    # █
SYM_ENTRY = "\u00bb"   # »

# Predefined winding path coordinates (left to right)
# Path enters from left edge (row 5), winds through the grid, exits to base on right
_PATH_COORDS: list[tuple[int, int]] = [
    # Enter from left
    (0, 5),
    (1, 5), (2, 5), (3, 5), (4, 5),
    # Turn up
    (4, 4), (4, 3), (4, 2), (4, 1),
    # Turn right
    (5, 1), (6, 1),
    # Turn down
    (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9),
    # Turn right
    (7, 9), (8, 9),
    # Turn up
    (8, 8), (8, 7), (8, 6), (8, 5),
    # Turn right to base
    (9, 5), (10, 5),
]

_PATH_SET = set(_PATH_COORDS)


def _build_action_spec() -> ActionSpec:
    """Build an action spec with PLACE_N and UPGRADE_N for every non-path,
    non-wall interior cell, plus NEXT_WAVE and WAIT."""
    placeable: list[tuple[int, int]] = []
    for y in range(1, GRID_SIZE - 1):
        for x in range(1, GRID_SIZE - 1):
            if (x, y) not in _PATH_SET:
                placeable.append((x, y))

    names: list[str] = []
    descs: list[str] = []
    for i, (px, py) in enumerate(placeable):
        names.append(f"PLACE_{i}")
        descs.append(f"place a tower at ({px}, {py}) for {TOWER_COST} gold")
    for i, (px, py) in enumerate(placeable):
        names.append(f"UPGRADE_{i}")
        descs.append(
            f"upgrade your tower at ({px}, {py}) (cost = next-level x "
            f"{UPGRADE_COST_PER_LEVEL} gold; +1 damage per level; max level "
            f"{MAX_TOWER_LEVEL})"
        )

    names.append("NEXT_WAVE")
    descs.append("start the next wave of enemies")
    names.append("WAIT")
    descs.append("do nothing for one step")

    return ActionSpec(names=tuple(names), descriptions=tuple(descs)), placeable


_ACTION_SPEC_AND_CELLS = _build_action_spec()
TD_ACTION_SPEC: ActionSpec = _ACTION_SPEC_AND_CELLS[0]
_PLACEABLE_CELLS: list[tuple[int, int]] = _ACTION_SPEC_AND_CELLS[1]
_NEXT_WAVE_IDX = TD_ACTION_SPEC.index_of("NEXT_WAVE")
_WAIT_IDX = TD_ACTION_SPEC.index_of("WAIT")


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class TowerDefenseEnv(BaseGlyphEnv):
    """Tower defense: place towers to stop enemy waves on a winding path."""

    action_spec = TD_ACTION_SPEC
    noop_action_name: str = "WAIT"

    def __init__(self, max_turns: int = 300) -> None:
        super().__init__(max_turns=max_turns)
        self._gold: int = 0
        # Tower cell -> level (1..MAX_TOWER_LEVEL). Level n tower deals n damage.
        self._towers: dict[tuple[int, int], int] = {}
        # Enemy state: list of dicts {"path_idx": int, "hp": int}
        self._enemies: list[dict[str, int]] = []
        self._wave: int = 0
        self._wave_active: bool = False
        self._waves_cleared: int = 0
        self._base_reached: bool = False
        self._all_waves_survived: bool = False
        self._message: str = ""

    def env_id(self) -> str:
        return "glyphbench/classics-towerdefense-v0"

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._gold = STARTING_GOLD
        self._towers = {}
        self._enemies = []
        self._wave = 0
        self._wave_active = False
        self._waves_cleared = 0
        self._base_reached = False
        self._all_waves_survived = False
        self._message = f"Place towers and press NEXT_WAVE to start. Gold: {self._gold}"
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        reward = 0.0
        terminated = False
        self._message = ""

        name = self.action_spec.names[action]

        if name.startswith("PLACE_"):
            idx = int(name.split("_")[1])
            if idx < len(_PLACEABLE_CELLS):
                cell = _PLACEABLE_CELLS[idx]
                if cell in self._towers:
                    self._message = "A tower is already there."
                elif len(self._towers) >= MAX_TOWERS:
                    self._message = (
                        f"Tower limit reached ({MAX_TOWERS}). Upgrade an existing "
                        "tower instead."
                    )
                elif self._gold < TOWER_COST:
                    self._message = (
                        f"Not enough gold (need {TOWER_COST}, have {self._gold})."
                    )
                else:
                    self._towers[cell] = 1
                    self._gold -= TOWER_COST
                    self._message = (
                        f"Tower placed at {cell} (lvl 1). Gold: {self._gold}"
                    )
        elif name.startswith("UPGRADE_"):
            idx = int(name.split("_")[1])
            if idx < len(_PLACEABLE_CELLS):
                cell = _PLACEABLE_CELLS[idx]
                if cell not in self._towers:
                    self._message = "No tower to upgrade at that cell."
                else:
                    current_level = self._towers[cell]
                    if current_level >= MAX_TOWER_LEVEL:
                        self._message = (
                            f"Tower already at max level ({MAX_TOWER_LEVEL})."
                        )
                    else:
                        next_level = current_level + 1
                        cost = next_level * UPGRADE_COST_PER_LEVEL
                        if self._gold < cost:
                            self._message = (
                                f"Not enough gold to upgrade (need {cost}, have "
                                f"{self._gold})."
                            )
                        else:
                            self._towers[cell] = next_level
                            self._gold -= cost
                            self._message = (
                                f"Tower upgraded to lvl {next_level} at {cell}. "
                                f"Gold: {self._gold}"
                            )
        elif name == "NEXT_WAVE" and not self._wave_active:
            if self._wave < TOTAL_WAVES:
                self._wave += 1
                self._wave_active = True
                self._spawn_wave()
                self._message = f"Wave {self._wave}/{TOTAL_WAVES} started!"
            else:
                self._message = "All waves already sent."
        elif name == "NEXT_WAVE":
            self._message = "Wave already in progress; survive it first."
        # WAIT: do nothing

        # Simulate combat if wave is active
        if self._wave_active:
            # Towers shoot. Each tower deals damage equal to its level.
            for (tx, ty), level in self._towers.items():
                if not self._enemies:
                    break
                # Find nearest enemy in range (Manhattan distance).
                best_enemy = None
                best_dist = float("inf")
                for e in self._enemies:
                    pi = e["path_idx"]
                    if pi < 0 or pi >= len(_PATH_COORDS):
                        continue
                    ex, ey = _PATH_COORDS[pi]
                    dist = abs(ex - tx) + abs(ey - ty)
                    if dist <= TOWER_RANGE and dist < best_dist:
                        best_dist = dist
                        best_enemy = e
                if best_enemy is not None:
                    best_enemy["hp"] -= level

            # Remove dead enemies
            alive_before = len(self._enemies)
            self._enemies = [e for e in self._enemies if e["hp"] > 0]
            kills = alive_before - len(self._enemies)
            self._gold += kills * GOLD_PER_KILL

            # Move enemies along path
            for e in self._enemies:
                e["path_idx"] += 1

            # Enemy reached the base (terminal failure). Additive penalty
            # so wave-clear / kill rewards earned this step survive into
            # the cumulative MC return used by GRPO.
            for e in self._enemies:
                if e["path_idx"] >= len(_PATH_COORDS):
                    self._base_reached = True
                    terminated = True
                    reward += -1.0
                    self._message = "An enemy reached the base! You lose."
                    break

            # Remove enemies that passed the end (shouldn't happen if base_reached triggers)
            self._enemies = [e for e in self._enemies if e["path_idx"] < len(_PATH_COORDS)]

            # Check if wave cleared. Pattern A progress: each wave cleared
            # yields +1/TOTAL_WAVES so cumulative reward = 1.0 on full
            # survival.
            if not terminated and not self._enemies:
                self._wave_active = False
                self._waves_cleared += 1
                reward = 1.0 / TOTAL_WAVES
                if self._waves_cleared >= TOTAL_WAVES:
                    self._all_waves_survived = True
                    terminated = True
                    self._message = "All waves survived! You win!"
                else:
                    self._message = f"Wave {self._wave} cleared! Gold: {self._gold}"

        info["gold"] = self._gold
        info["wave"] = self._wave
        info["waves_cleared"] = self._waves_cleared
        info["towers"] = len(self._towers)
        info["enemies_alive"] = len(self._enemies)

        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Wave spawning
    # ------------------------------------------------------------------

    def _spawn_wave(self) -> None:
        """Spawn enemies for the current wave. Later waves have more/tougher enemies."""
        num_enemies = 2 + self._wave
        hp = 2 if self._wave <= 3 else 3
        # Enemies spawn staggered along the start of the path
        for i in range(num_enemies):
            # Negative path_idx means they haven't entered yet; they'll enter on successive steps
            self._enemies.append({"path_idx": -i * 2, "hp": hp})

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(GRID_SIZE, GRID_SIZE, SYM_GROUND)

        # Walls (border)
        for x in range(GRID_SIZE):
            grid[0][x] = SYM_WALL
            grid[GRID_SIZE - 1][x] = SYM_WALL
        for y in range(GRID_SIZE):
            grid[y][0] = SYM_WALL
            grid[y][GRID_SIZE - 1] = SYM_WALL

        # Path
        for px, py in _PATH_COORDS:
            if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
                grid[py][px] = SYM_PATH

        # Entry point
        grid[5][0] = SYM_ENTRY

        # Base
        grid[5][GRID_SIZE - 1] = SYM_BASE

        # Towers — render with level digit.
        legend_extras: dict[str, str] = {}
        for (tx, ty), lvl in self._towers.items():
            sym = str(lvl) if 1 <= lvl <= 9 else SYM_TOWER
            grid[ty][tx] = sym
            legend_extras[sym] = f"tower at level {lvl} (deals {lvl} damage)"

        # Enemies
        for e in self._enemies:
            idx = e["path_idx"]
            if 0 <= idx < len(_PATH_COORDS):
                ex, ey = _PATH_COORDS[idx]
                grid[ey][ex] = SYM_ENEMY

        legend_dict = {
            SYM_GROUND: "ground (can place tower)",
            SYM_PATH: "enemy path",
            SYM_ENEMY: "enemy",
            SYM_BASE: "your base (defend this!)",
            SYM_WALL: "wall",
            SYM_ENTRY: "enemy entry point",
        }
        legend_dict.update(legend_extras)
        legend = build_legend(legend_dict)

        tower_summary = (
            ", ".join(
                f"({tx},{ty}) lvl {lvl}" for (tx, ty), lvl in self._towers.items()
            )
            or "(none)"
        )
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Gold: {self._gold}    "
            f"Wave: {self._wave}/{TOTAL_WAVES}    "
            f"Towers: {len(self._towers)}/{MAX_TOWERS}    "
            f"Enemies: {len(self._enemies)}\n"
            f"Tower roster: {tower_summary}"
        )

        return GridObservation(
            grid=grid_to_string(grid), legend=legend, hud=hud, message=self._message
        )

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Place and upgrade towers along the winding path to stop waves of "
            "enemies from reaching your base.\n\n"
            "RULES\n"
            f"- The grid is {GRID_SIZE}x{GRID_SIZE}. The path's start, end, and shape "
            "are visible on the [Grid].\n"
            f"- You start with {STARTING_GOLD} gold. A new tower costs {TOWER_COST} "
            "gold.\n"
            f"- You may have at most {MAX_TOWERS} towers on the board at once. To "
            "boost a tower instead of building a new one, UPGRADE it.\n"
            "- Tower placements are permanent; choose cells carefully.\n"
            f"- Upgrading a tower from level L to L+1 costs (L+1) x "
            f"{UPGRADE_COST_PER_LEVEL} gold and adds +1 damage per attack. Max "
            f"level: {MAX_TOWER_LEVEL}.\n"
            f"- Towers shoot the nearest enemy within range ({TOWER_RANGE} cell, "
            "Manhattan distance) each step. Damage equals the tower's level.\n"
            "- A level-1 tower deals 1 damage; level 2 = 2; up to level "
            f"{MAX_TOWER_LEVEL}.\n"
            "- Waves 1-3 spawn 2-HP enemies; waves 4-5 spawn 3-HP enemies.\n"
            f"- Earn {GOLD_PER_KILL} gold per kill.\n"
            f"- There are {TOTAL_WAVES} waves total.\n"
            "- You may build or upgrade any time; combat resolves after each step.\n"
            "- Use NEXT_WAVE when no wave is active to start the next wave.\n"
            "- If any enemy reaches the base: -1 reward, game over.\n"
            f"- Each wave cleared: +{1.0 / TOTAL_WAVES:.4f} reward. Cumulative reward "
            "= 1.0 if you survive all waves.\n"
            "- Towers in the [Grid] are rendered as a digit equal to their level.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
