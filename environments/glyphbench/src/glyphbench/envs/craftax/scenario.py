"""Focused Craftax scenario base.

Registered ``glyphbench/craftax-*`` tasks are focused scenarios cut from the
single GlyphBench Craftax implementation. This module provides a compatibility
base for older surface-style tasks while keeping the public action space,
achievement namespace, player state, and step dispatch on ``CraftaxFullEnv``.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.craftax.base import TILE_GRASS
from glyphbench.envs.craftaxfull.full import (
    _DAY_LENGTH,
    _MAX_ENERGY,
    _MAX_FOOD,
    _MAX_WATER,
    _MOB_STATS,
    _MOB_TILES,
    _SURFACE_SIZE,
    DUNGEON_WALKABLE,
    SURFACE_WALKABLE,
    UPSTREAM_ACHIEVEMENT_NAMES,
    CraftaxFullEnv,
    Mob,
)


class CraftaxScenarioEnv(CraftaxFullEnv):
    """Full Craftax mechanics base for focused scenarios.

    Older focused tasks still use ``_world`` and compact curated maps. The
    property below aliases that surface to ``_floors[_current_floor]`` so the
    inherited full Craftax action handlers, mob helpers, projectile logic, and
    renderer can operate on the same state.
    """

    craftax_focused_subtask = True
    _WORLD_SIZE = _SURFACE_SIZE

    @property
    def _world(self) -> list[list[str]]:
        if not self._floors:
            self._floors = {0: []}
        return self._floors.setdefault(getattr(self, "_current_floor", 0), [])

    @_world.setter
    def _world(self, value: list[list[str]]) -> None:
        if not hasattr(self, "_floors"):
            self._floors = {}
        floor = getattr(self, "_current_floor", 0)
        self._floors[floor] = value

    def _floor_size(self) -> int:
        grid = self._current_grid()
        return len(grid)

    def _walkable_set(self) -> frozenset[str]:
        return SURFACE_WALKABLE | DUNGEON_WALKABLE

    def _reset_scenario_state(
        self,
        seed: int,
        *,
        world_size: int,
        fill: str = TILE_GRASS,
        floor: int = 0,
    ) -> None:
        """Reset full Craftax state, then install a focused scenario map."""
        from glyphbench.envs.craftax.mechanics.potions import make_potion_mapping

        self._floors = {
            floor: [[fill for _ in range(world_size)] for _ in range(world_size)]
        }
        self._current_floor = floor
        self._agent_x = world_size // 2
        self._agent_y = world_size // 2
        self._facing = (1, 0)
        self._inventory = {
            "bow": 0,
            "arrows": 0,
            "torches": 0,
            "sapphire": 0,
            "ruby": 0,
            "book": 0,
            "potions": {
                "red": 0,
                "green": 0,
                "blue": 0,
                "pink": 0,
                "cyan": 0,
                "yellow": 0,
            },
        }
        self._achievements_unlocked = set()
        self._achievements_phase_beta = {n: False for n in UPSTREAM_ACHIEVEMENT_NAMES}
        self._message = ""
        self._learned_spells = {"fireball": False, "iceball": False}
        self._day_counter = 0
        self._day_night = "day"
        self._night_count = 0
        self._mobs = []
        self._player_projectiles = []
        self._mob_projectiles = []
        self._is_sleeping = False
        self._is_resting = False
        self._pending_step_reward = 0.0
        self._plants = {}
        self._potions = []
        self._potion_mapping = make_potion_mapping(seed)
        self._speed_turns = 0
        self._sword_enchantment = 0
        self._bow_enchantment = 0
        self._armor_slots = {"helmet": 0, "chest": 0, "legs": 0, "boots": 0}
        self._armor_enchants = {"helmet": 0, "chest": 0, "legs": 0, "boots": 0}
        self._total_kills = 0
        self._total_crafts = 0
        self._total_blocks_placed = 0
        self._total_plants_eaten = 0
        self._total_water_drunk = 0
        self._stairs_down_pos = {}
        self._stairs_up_pos = {}
        self._bosses_alive = {}
        self._chests_opened = {}
        self._first_chest_opened = {}
        # Lighting / torch state (2026-06 port-vs-fork alignment).
        self._torches_by_floor = {}
        self._floor_ambient_light = {}
        self._lava_light_cache = {}
        self._xp = 0
        self._xp_floors_visited = set()
        self._dex = 1
        self._str = 1
        self._int_attr = 1
        self._boss_progress = 0
        self._boss_summon_timer = 0
        self._recompute_max_stats()
        self._hp = self._max_hp
        self._food = self._max_food
        self._water = self._max_drink
        self._energy = self._max_energy
        self._mana = self._max_mana

    def _normalize_scenario_state(self) -> None:
        if not hasattr(self, "_player_projectiles"):
            self._player_projectiles = []
        if not hasattr(self, "_mob_projectiles"):
            self._mob_projectiles = []
        # Lighting / torch state (defensive defaults for older scenario states).
        if not hasattr(self, "_torches_by_floor"):
            self._torches_by_floor = {}
        if not hasattr(self, "_floor_ambient_light"):
            self._floor_ambient_light = {}
        if not hasattr(self, "_lava_light_cache"):
            self._lava_light_cache = {}
        self._inventory.setdefault("torches", 0)
        if not hasattr(self, "_achievements_phase_beta"):
            self._achievements_phase_beta = {
                n: False for n in UPSTREAM_ACHIEVEMENT_NAMES
            }
        if not hasattr(self, "_armor_slots"):
            self._armor_slots = {"helmet": 0, "chest": 0, "legs": 0, "boots": 0}
        if not hasattr(self, "_armor_enchants"):
            self._armor_enchants = {"helmet": 0, "chest": 0, "legs": 0, "boots": 0}
        self._inventory.setdefault("potions", {})
        for mob in self._mobs:
            mob.setdefault("floor", self._current_floor)
            mob.setdefault("is_boss", False)
            mob.setdefault("attack_cooldown", 0)

    def _mob_at(
        self,
        x: int,
        y: int,
        floor: int | None = None,
    ) -> Mob | None:
        fl = floor if floor is not None else self._current_floor
        for mob in self._mobs:
            mob.setdefault("floor", fl)
            mob.setdefault("is_boss", False)
            if mob["x"] == x and mob["y"] == y and mob["floor"] == fl:
                return mob
        return None

    def _try_unlock_achievement(self, name: str) -> float:
        return super()._try_unlock(name)

    def _try_unlock(self, name: str) -> float:
        return self._try_unlock_achievement(name)

    def _advance_day_counter(self, steps: int = 1) -> None:
        if getattr(self, "_disable_day_night", False):
            return
        super()._advance_day_counter(steps)

    def _apply_survival_drain(self) -> None:
        if (
            getattr(self, "_disable_survival", False)
            or getattr(self, "_disable_survival_drain", False)
        ):
            return
        super()._apply_survival_drain()

    def _step(
        self,
        action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        self._normalize_scenario_state()
        return super()._step(action)

    def _render_current_observation(self) -> GridObservation:
        self._normalize_scenario_state()
        return super()._render_current_observation()


__all__ = [
    "CraftaxScenarioEnv",
    "Mob",
    "SURFACE_WALKABLE",
    "_MAX_ENERGY",
    "_MAX_FOOD",
    "_MAX_WATER",
    "_DAY_LENGTH",
    "_MOB_STATS",
    "_MOB_TILES",
]
