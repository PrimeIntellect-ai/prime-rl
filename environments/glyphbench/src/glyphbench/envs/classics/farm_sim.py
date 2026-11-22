"""Farm simulator game.

Plant, water, harvest, and sell crops to earn gold.

Gym IDs:
  glyphbench/classics-farm-v0
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

GRID_SIZE = 8

# Target gold for cumulative reward = 1.0. A skilled player can approach this
# cap by cycling multiple plots near the market; post-cap sales add no reward.
_TARGET_GOLD = 30

FARM_ACTION_SPEC = ActionSpec(
    names=("UP", "DOWN", "LEFT", "RIGHT", "PLANT", "WATER", "HARVEST", "WAIT"),
    descriptions=(
        "move farmer up one cell",
        "move farmer down one cell",
        "move farmer left one cell",
        "move farmer right one cell",
        "plant a seed on the current dirt tile",
        "water the plot you are standing on",
        "harvest the ready crop you are standing on",
        "do nothing for one step",
    ),
)

SYM_FARMER = "@"
SYM_DIRT = "\u00b7"    # ·
SYM_SEEDED = "\u2660"  # ♠
SYM_GROWING = "\u2663" # ♣
SYM_READY = "\u273f"   # ✿
SYM_MARKET = "\u25a3"  # ▣
SYM_FENCE = "\u2588"   # █

# Growth: seeded -> growing -> ready
# Each stage needs 3 waterings to advance
STAGE_DIRT = 0
STAGE_SEEDED = 1
STAGE_GROWING = 2
STAGE_READY = 3
WATERS_PER_STAGE = 3

_DIR_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class FarmSimEnv(BaseGlyphEnv):
    """Farm simulator: plant, water, harvest, and sell crops."""

    action_spec = FARM_ACTION_SPEC
    noop_action_name: str = "WAIT"

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._farmer_pos: tuple[int, int] = (0, 0)
        # Plot state: dict of (x, y) -> {"stage": int, "waters": int}
        self._plots: dict[tuple[int, int], dict[str, int]] = {}
        self._market_pos: tuple[int, int] = (0, 0)
        self._gold: int = 0
        self._crops_held: int = 0
        self._fence: set[tuple[int, int]] = set()

    def env_id(self) -> str:
        return "glyphbench/classics-farm-v0"

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._gold = 0
        self._crops_held = 0

        # Fence around the border
        self._fence = set()
        for x in range(GRID_SIZE):
            self._fence.add((x, 0))
            self._fence.add((x, GRID_SIZE - 1))
        for y in range(GRID_SIZE):
            self._fence.add((0, y))
            self._fence.add((GRID_SIZE - 1, y))

        # Market in top-right interior corner
        self._market_pos = (GRID_SIZE - 2, 1)

        # Farmer starts bottom-left interior
        self._farmer_pos = (1, GRID_SIZE - 2)

        # All interior cells (except market) start as dirt
        self._plots = {}
        for y in range(1, GRID_SIZE - 1):
            for x in range(1, GRID_SIZE - 1):
                if (x, y) != self._market_pos:
                    self._plots[(x, y)] = {"stage": STAGE_DIRT, "waters": 0}

        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]
        reward = 0.0
        msg = ""

        if name in _DIR_DELTAS:
            dx, dy = _DIR_DELTAS[name]
            nx, ny = self._farmer_pos[0] + dx, self._farmer_pos[1] + dy
            if (nx, ny) not in self._fence:
                self._farmer_pos = (nx, ny)
            else:
                msg = "Blocked by fence."
            # Auto-sell at market. Pattern A: each gold earned = 1/_TARGET_GOLD,
            # cumulative reward capped at 1.0.
            if self._farmer_pos == self._market_pos and self._crops_held > 0:
                earned = self._crops_held
                # Cap so cumulative reward never exceeds 1.0.
                already = (self._gold / _TARGET_GOLD)
                room = max(0.0, 1.0 - already)
                reward = min(float(earned) / _TARGET_GOLD, room)
                self._gold += earned
                msg = f"Sold {self._crops_held} crop(s) for {earned} gold!"
                self._crops_held = 0

        elif name == "PLANT":
            pos = self._farmer_pos
            if pos in self._plots and self._plots[pos]["stage"] == STAGE_DIRT:
                self._plots[pos]["stage"] = STAGE_SEEDED
                self._plots[pos]["waters"] = 0
                msg = "Planted a seed."
            else:
                msg = "Cannot plant here."

        elif name == "WATER":
            pos = self._farmer_pos
            if pos in self._plots:
                plot = self._plots[pos]
                if plot["stage"] in (STAGE_SEEDED, STAGE_GROWING):
                    plot["waters"] += 1
                    if plot["waters"] >= WATERS_PER_STAGE:
                        plot["stage"] += 1
                        plot["waters"] = 0
                        if plot["stage"] == STAGE_GROWING:
                            msg = "The plant is growing!"
                        elif plot["stage"] == STAGE_READY:
                            msg = "The crop is ready to harvest!"
                    else:
                        msg = f"Watered ({plot['waters']}/{WATERS_PER_STAGE})."
                else:
                    msg = "Cannot water this plot now."
            else:
                msg = "Cannot water here."

        elif name == "HARVEST":
            pos = self._farmer_pos
            if pos in self._plots and self._plots[pos]["stage"] == STAGE_READY:
                self._plots[pos]["stage"] = STAGE_DIRT
                self._plots[pos]["waters"] = 0
                self._crops_held += 1
                msg = f"Harvested! Carrying {self._crops_held} crop(s)."
            else:
                msg = "Nothing ready to harvest here."

        # WAIT: do nothing

        info["gold"] = self._gold
        info["crops_held"] = self._crops_held

        obs = self._render_current_observation()
        if msg:
            obs = GridObservation(grid=obs.grid, legend=obs.legend, hud=obs.hud, message=msg)

        return obs, reward, False, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(GRID_SIZE, GRID_SIZE, SYM_DIRT)

        # Fence
        for fx, fy in self._fence:
            grid[fy][fx] = SYM_FENCE

        # Market
        mx, my = self._market_pos
        grid[my][mx] = SYM_MARKET

        # Plots
        stage_to_sym = {
            STAGE_DIRT: SYM_DIRT,
            STAGE_SEEDED: SYM_SEEDED,
            STAGE_GROWING: SYM_GROWING,
            STAGE_READY: SYM_READY,
        }
        for (px, py), plot in self._plots.items():
            grid[py][px] = stage_to_sym[plot["stage"]]

        # Farmer (on top)
        fx, fy = self._farmer_pos
        grid[fy][fx] = SYM_FARMER

        legend = build_legend({
            SYM_FARMER: "farmer (you)",
            SYM_DIRT: "empty dirt",
            SYM_SEEDED: "seeded plot",
            SYM_GROWING: "growing plot",
            SYM_READY: "ready to harvest",
            SYM_MARKET: "market (sell crops here)",
            SYM_FENCE: "fence",
        })

        standing_on = "market"
        plot = self._plots.get(self._farmer_pos)
        if plot is not None:
            stage_names = {
                STAGE_DIRT: "dirt",
                STAGE_SEEDED: "seeded plot",
                STAGE_GROWING: "growing plot",
                STAGE_READY: "ready crop",
            }
            standing_on = stage_names[plot["stage"]]

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Gold: {self._gold}    "
            f"Crops held: {self._crops_held}    "
            f"Standing on: {standing_on}"
        )

        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message="")

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Manage a small farm. Plant seeds, water them, harvest crops, and sell at the market "
            "to earn gold. Maximize gold by the end of the episode.\n\n"
            "RULES\n"
            f"- The grid is {GRID_SIZE}x{GRID_SIZE} surrounded by a fence.\n"
            "- PLANT places a seed on the dirt tile you stand on (free).\n"
            "- WATER waters the plot you stand on. Each growth stage needs 3 waterings.\n"
            "- Growth stages: dirt \u2192 seeded (\u2660) \u2192 growing (\u2663) \u2192 ready (\u273f).\n"
            "- The farmer glyph (@) covers the tile underneath; use the HUD's Standing on field to inspect it.\n"
            "- HARVEST picks up a ready crop. You can carry multiple crops.\n"
            "- Walk to the market tile (\u25a3) to auto-sell all held crops for 1 gold each.\n"
            "- The episode lasts 200 steps. Reward is normalized so that cumulative\n"
            f"  reward = 1.0 once you have earned {_TARGET_GOLD} gold (any further gold\n"
            "  yields no additional reward).\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
