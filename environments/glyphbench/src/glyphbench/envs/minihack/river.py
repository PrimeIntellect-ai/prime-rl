"""MiniHack River environments. Cross a river of water or lava.

Phase 3: river-narrow gets D2 slippery ice — stepping stones are ice
tiles; stepping onto one slides the player forward until they hit a
non-ice cell. Plan slide direction.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MOVE_VECTORS, MiniHackBase
from glyphbench.envs.minihack.creatures import KOBOLD, ORC

ICE_GLYPH = "❄"  # ice tile (slippery)


class _RiverBase(MiniHackBase):
    _grid_size: tuple[int, int] = (11, 7)
    _river_type: str = "water"  # "water" or "lava"
    _num_stones: int = 3
    _has_monsters: bool = False
    _slippery_stones: bool = False  # if True, stones are ice

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._ice_positions: set[tuple[int, int]] = set()

    def _generate_level(self, seed: int) -> None:
        w, h = self._grid_size
        self._init_grid(w, h)
        self._ice_positions = set()

        river_y = h // 2

        place_fn = self._place_water if self._river_type == "water" else self._place_lava
        for x in range(1, w - 1):
            place_fn(x, river_y)

        stone_positions: set[int] = set()
        while len(stone_positions) < self._num_stones:
            sx = int(self.rng.integers(1, w - 1))
            stone_positions.add(sx)
        for sx in stone_positions:
            if self._slippery_stones:
                self._grid[river_y][sx] = ICE_GLYPH
                self._ice_positions.add((sx, river_y))
            else:
                self._grid[river_y][sx] = "·"

        px = int(self.rng.integers(1, w - 1))
        self._place_player(px, 1)

        stx = int(self.rng.integers(1, w - 1))
        self._place_stairs(stx, h - 2)

        if self._has_monsters:
            for i in range(2):
                while True:
                    mx = int(self.rng.integers(1, w - 1))
                    my = int(self.rng.integers(river_y + 1, h - 1))
                    if (
                        self._grid[my][mx] == "·"
                        and (mx, my) != (stx, h - 2)
                        and self._creature_at(mx, my) is None
                    ):
                        break
                ctype = [KOBOLD, ORC][i % 2]
                self._spawn_creature(ctype, mx, my)

    def _is_walkable(self, x: int, y: int) -> bool:
        # Allow ice tiles to be walked onto
        if (x, y) in self._ice_positions:
            return True
        return super()._is_walkable(x, y)

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        # Slippery-ice handling: when stepping onto ice, slide in same direction.
        name = self.action_spec.names[action]
        if (
            self._slippery_stones
            and name in MOVE_VECTORS
            and self._ice_positions
        ):
            dx, dy = MOVE_VECTORS[name]
            cx, cy = self._player_pos
            nx, ny = cx + dx, cy + dy
            # Step into stone? Slide.
            if (nx, ny) in self._ice_positions:
                # Move onto ice, then continue sliding while:
                #  next cell is ice OR walkable
                self._player_pos = (nx, ny)
                slid = True
                while slid:
                    fx, fy = self._player_pos[0] + dx, self._player_pos[1] + dy
                    next_terrain = self._terrain_at(fx, fy)
                    if next_terrain in ("█", "-", "|"):
                        # Hit a wall — stop on current cell
                        break
                    if (fx, fy) in self._ice_positions:
                        self._player_pos = (fx, fy)
                        continue
                    if next_terrain == "·" or next_terrain == "⇣":
                        # Land on solid floor or stairs — stop here.
                        self._player_pos = (fx, fy)
                        if (fx, fy) == self._goal_pos:
                            self._message = "You skid to a stop on the stairs!"
                            return self._render_current_observation(), 1.0, True, False, {
                                "goal_reached": True
                            }
                        break
                    if next_terrain == "≈":
                        # Slide into water = drown
                        self._player_pos = (fx, fy)
                        self._player_hp = 0
                        self._message = "You slide into water and drown!"
                        return self._render_current_observation(), -1.0, True, False, {
                            "cause_of_death": "hazard"
                        }
                    if next_terrain == "♨":
                        self._player_pos = (fx, fy)
                        self._player_hp = 0
                        self._message = "You slide into lava!"
                        return self._render_current_observation(), -1.0, True, False, {
                            "cause_of_death": "hazard"
                        }
                    break
                # After slide: do monster turns + check goal again.
                self._move_monsters()
                if self._player_hp <= 0:
                    return self._render_current_observation(), -1.0, True, False, {
                        "cause_of_death": "monster"
                    }
                self._message = f"You slide on ice and stop at {self._player_pos}."
                return self._render_current_observation(), 0.0, False, False, {
                    "player_pos": self._player_pos,
                    "hp": self._player_hp,
                }
        return super()._step(action)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        if self._slippery_stones and ICE_GLYPH not in obs.legend:
            # Add ice symbol to legend if missing
            new_legend = obs.legend
            if ICE_GLYPH in obs.grid:
                new_legend = obs.legend + f"\n{ICE_GLYPH} — slippery ice (you slide forward)"
            return GridObservation(
                grid=obs.grid,
                legend=new_legend,
                hud=obs.hud,
                message=obs.message,
            )
        return obs

    def _task_description(self) -> str:
        danger = "lava (instant death)" if self._river_type == "lava" else "water"
        parts = [
            f"Cross a river of {danger} using stepping stones to reach the "
            "stairs (⇣).",
        ]
        if self._slippery_stones:
            parts.append(
                "The stones are ICE (❄): stepping onto a stone makes you "
                "SLIDE in your movement direction until you hit solid ground "
                "or a hazard. Plan your slide trajectory."
            )
        if self._has_monsters:
            parts.append("Hostile monsters wait on the far side.")
        parts.append("Reward: +1 on reaching stairs, -1 on death.")
        return " ".join(parts)


class MiniHackRiverEnv(_RiverBase):
    _river_type = "water"
    _num_stones = 3

    def env_id(self) -> str:
        return "glyphbench/minihack-river-v0"


class MiniHackRiverNarrowEnv(_RiverBase):
    """D2 slippery-ice variant — narrow river with ice stepping stones."""

    _river_type = "water"
    _num_stones = 2
    _slippery_stones = True

    def env_id(self) -> str:
        return "glyphbench/minihack-river-narrow-v0"


class MiniHackRiverMonsterEnv(_RiverBase):
    _river_type = "water"
    _num_stones = 3
    _has_monsters = True

    def env_id(self) -> str:
        return "glyphbench/minihack-river-monster-v0"


class MiniHackRiverLavaEnv(_RiverBase):
    _river_type = "lava"
    _num_stones = 3

    def env_id(self) -> str:
        return "glyphbench/minihack-river-lava-v0"


class MiniHackRiverMonsterLavaEnv(_RiverBase):
    _river_type = "lava"
    _num_stones = 3
    _has_monsters = True

    def env_id(self) -> str:
        return "glyphbench/minihack-river-monsterlava-v0"
