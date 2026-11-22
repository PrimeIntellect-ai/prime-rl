"""MiniHack LavaCross skill tasks."""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MiniHackBase
from glyphbench.envs.minihack.items import POTION_LEVITATION, RING_LEVITATION, Item


class _LavaCrossBase(MiniHackBase):
    _potion_on_floor: bool = True
    _ring_on_floor: bool = True
    _potion_in_inv: bool = False
    _ring_in_inv: bool = False

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._levitating_turns: int = 0

    def _generate_level(self, seed: int) -> None:
        self._init_grid(11, 7)
        self._levitating_turns = 0
        layouts: tuple[tuple[int, int, int], ...] = (
            (1, 3, 5),
            (2, 4, 1),
            (3, 5, 2),
            (4, 1, 3),
        )
        path_y, potion_y, ring_y = layouts[seed % len(layouts)]
        self._place_player(1, path_y)
        self._place_stairs(9, path_y)

        # Lava pit in the middle (x=4 to x=6)
        for x in range(4, 7):
            for y in range(1, 6):
                self._place_lava(x, y)

        # Place levitation items
        if self._potion_on_floor:
            self._place_item(2, potion_y, POTION_LEVITATION)
        if self._ring_on_floor:
            self._place_item(2, ring_y, RING_LEVITATION)
        if self._potion_in_inv:
            self._inventory.append(POTION_LEVITATION)
        if self._ring_in_inv:
            self._inventory.append(RING_LEVITATION)

    def _on_quaff_potion(self, potion: Item) -> None:
        if potion.name == "potion of levitation":
            self._levitating_turns = 20
            self._message += " You start to float!"

    def _is_walkable(self, x: int, y: int) -> bool:
        # During levitation, lava is walkable without modifying the grid
        if self._levitating_turns > 0 and self._terrain_at(x, y) == "♨":
            return True
        return super()._is_walkable(x, y)

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)

        # Decrement levitation
        if self._levitating_turns > 0:
            self._levitating_turns -= 1
            if self._levitating_turns == 0:
                self._message += " You float gently to the ground."
                # Check if standing on lava when levitation expires
                px, py = self._player_pos
                if self._terrain_at(px, py) == "♨":
                    self._player_hp = 0
                    self._message += " You fall into the lava! You die."
                    terminated = True
                    # Additive death penalty so any progress reward emitted
                    # earlier this step / episode persists in the
                    # cumulative MC return used by GRPO.
                    reward += -1.0
                    info["cause_of_death"] = "hazard"
                # Re-render to include updated message
                obs = self._render_current_observation()

        return obs, reward, terminated, truncated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        if self._levitating_turns > 0:
            lev = f"Levitating: {self._levitating_turns} turns"
        else:
            lev = "Levitating: no"
        extras = [lev]
        if self._ring_in_inv:
            extras.append(
                f"Ring equipped: {'yes' if self._ring_equipped else 'no'}"
            )
        new_hud = obs.hud + "    " + "    ".join(extras)
        return GridObservation(
            grid=obs.grid,
            legend=obs.legend,
            hud=new_hud,
            message=obs.message,
        )

    def _task_description(self) -> str:
        desc = (
            "A lava pit (♨) blocks your path to the stairs (⇣). "
            "Use a potion of levitation (QUAFF) or ring of levitation to float over it. "
            "Stepping on lava without levitation is fatal. "
            "Reward: +1 stairs, -1 death."
        )
        if self._ring_in_inv:
            desc += " The ring starts in your inventory; WIELD it to put it on."
        return desc


class MiniHackLavaCrossFullEnv(_LavaCrossBase):
    _potion_on_floor = True
    _ring_on_floor = True

    def env_id(self) -> str:
        return "glyphbench/minihack-lavacross-full-v0"


class MiniHackLavaCrossLevitateEnv(_LavaCrossBase):
    _potion_on_floor = True
    _ring_on_floor = False

    def env_id(self) -> str:
        return "glyphbench/minihack-lavacross-levitate-v0"


class MiniHackLavaCrossPotionInvEnv(_LavaCrossBase):
    _potion_on_floor = False
    _ring_on_floor = False
    _potion_in_inv = True

    def env_id(self) -> str:
        return "glyphbench/minihack-lavacross-levitate-potion-inv-v0"


class MiniHackLavaCrossRingInvEnv(_LavaCrossBase):
    _potion_on_floor = False
    _ring_on_floor = False
    _ring_in_inv = True

    def env_id(self) -> str:
        return "glyphbench/minihack-lavacross-levitate-ring-inv-v0"
