"""Craftax focused sub-task environments.

Each sub-task starts the player in a curated initial state and has a clear
success condition, testing a specific skill (navigation, crafting, combat,
dungeon exploration). Registered tasks use the universal Craftax action space
and full GlyphBench Craftax mechanics through ``CraftaxScenarioEnv``.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.craftax.base import (
    TILE_ARROW,
    TILE_COAL,
    TILE_DUNGEON_FLOOR,
    TILE_DUNGEON_WALL,
    TILE_FURNACE,
    TILE_GRASS,
    TILE_IRON,
    TILE_SKELETON,
    TILE_STAIRS_DOWN,
    TILE_STONE,
    TILE_TABLE,
    TILE_TREE,
    TILE_ZOMBIE,
    VIEW_HEIGHT,
    VIEW_WIDTH,
)
from glyphbench.envs.craftax.scenario import (
    _DAY_LENGTH,
    _MOB_STATS,
    _MOB_TILES,
    CraftaxScenarioEnv,
    Mob,
)

# ---------------------------------------------------------------------------
# Helpers shared across sub-tasks
# ---------------------------------------------------------------------------

_SUBTASK_WORLD_SIZE = 16  # small curated arenas


def _blank_world(size: int = _SUBTASK_WORLD_SIZE, fill: str = TILE_GRASS) -> list[list[str]]:
    """Return a square grid filled with *fill*."""
    return [[fill for _ in range(size)] for _ in range(size)]


def _clear_area(
    world: list[list[str]],
    cx: int,
    cy: int,
    radius: int,
    tile: str = TILE_GRASS,
) -> None:
    size = len(world)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < size and 0 <= y < size:
                world[y][x] = tile


def _spawn_mob(mobs: list[Mob], mob_type: str, x: int, y: int) -> None:
    """Append a fresh mob of *mob_type* at (x, y)."""
    stats = _MOB_STATS[mob_type]
    mobs.append(Mob(
        type=mob_type,
        x=x,
        y=y,
        hp=stats["hp"],
        max_hp=stats["hp"],
        attack_cooldown=0,
    ))


def _scatter_tiles_far_from_spawn(
    env: CraftaxScenarioEnv,
    *,
    tile: str,
    count: int,
    min_distance: int,
    radius: int,
) -> None:
    """Place resource tiles outside the spawn pocket, with seed variation."""
    cx, cy = env._agent_x, env._agent_y
    placed = 0
    for _ in range(count * 80):
        if placed >= count:
            break
        dx = int(env.rng.integers(-radius, radius + 1))
        dy = int(env.rng.integers(-radius, radius + 1))
        if abs(dx) + abs(dy) < min_distance:
            continue
        x, y = cx + dx, cy + dy
        if not (1 <= x < env._WORLD_SIZE - 1 and 1 <= y < env._WORLD_SIZE - 1):
            continue
        if env._world[y][x] != TILE_GRASS:
            continue
        env._world[y][x] = tile
        placed += 1


# ===================================================================
# Base mixin that patches CraftaxScenarioEnv for small-arena sub-tasks
# ===================================================================

class _SubtaskMixin:
    """Shared overrides for all Craftax sub-task envs.

    Must be listed *before* CraftaxScenarioEnv in the MRO so that its
    ``_reset`` / ``_step`` run first.

    Subclasses implement:
    - ``_setup_world(seed)`` -- populate ``self._world``, ``self._inventory``,
      ``self._mobs``, agent position, etc.
    - ``_subtask_check(reward, info)`` -- return ``(extra_reward, terminated)``
      after the parent ``_step`` has run.
    """

    # Sub-tasks use a small arena
    _WORLD_SIZE: int = _SUBTASK_WORLD_SIZE
    craftax_focused_subtask: bool = True

    # Disable survival drains (food/water/energy never decrease) so the
    # sub-task tests only the intended skill.
    _disable_survival: bool = True
    # Disable day/night mob spawns
    _disable_day_night: bool = True

    # Sub-tasks manage their own death penalty in ``_subtask_check``; do
    # NOT let the parent emit its own -1.0 on death (that would double the
    # penalty out to -2.0, breaking the cumulative bound).
    _emit_death_penalty: bool = False

    # Subclasses set these
    _subtask_max_turns: int = 50

    def __init__(self, max_turns: int | None = None, **kwargs: Any) -> None:
        # External override wins; else use the sub-task's tight budget
        effective = max_turns if max_turns is not None else self._subtask_max_turns
        super().__init__(max_turns=effective, **kwargs)  # type: ignore[call-arg]

    # -- reset -----------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:  # type: ignore[override]
        # Reset full Craftax player/mechanics state, then install the curated map.
        self._reset_scenario_state(  # type: ignore[attr-defined]
            seed,
            world_size=self._WORLD_SIZE,
            fill=TILE_GRASS,
        )
        self._world = _blank_world(self._WORLD_SIZE)  # type: ignore[attr-defined]
        self._agent_x = self._WORLD_SIZE // 2  # type: ignore[attr-defined]
        self._agent_y = self._WORLD_SIZE // 2  # type: ignore[attr-defined]
        self._facing = (1, 0)  # type: ignore[attr-defined]
        self._inventory = {"potions": {}}  # type: ignore[attr-defined]
        self._message = ""  # type: ignore[attr-defined]
        self._hp = 9  # type: ignore[attr-defined]
        self._max_hp = 9  # type: ignore[attr-defined]
        self._food = 9  # type: ignore[attr-defined]
        self._water = 9  # type: ignore[attr-defined]
        self._energy = 9  # type: ignore[attr-defined]
        self._day_counter = 0  # type: ignore[attr-defined]
        self._day_night = "day"  # type: ignore[attr-defined]
        self._mobs = []  # type: ignore[attr-defined]
        self._player_projectiles = []  # type: ignore[attr-defined]
        self._mob_projectiles = []  # type: ignore[attr-defined]
        self._plants = {}  # type: ignore[attr-defined]

        # Subclass populates the curated world
        self._setup_world(seed)  # type: ignore[attr-defined]
        return self._render_current_observation()  # type: ignore[attr-defined]

    # -- step (wraps parent, then checks sub-task goal) ------------------

    def _step(  # type: ignore[override]
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        # Run the full Craftax step machinery
        obs, reward, terminated, truncated, info = super()._step(action)  # type: ignore[misc]

        # Always invoke the subtask check, even when the parent already
        # terminated the episode (e.g. the agent died). Otherwise per-
        # subtask death-penalty branches are unreachable. The check is
        # responsible for keeping `terminated` True when the parent
        # already set it.
        extra_reward, done = self._subtask_check(reward, info)  # type: ignore[attr-defined]
        reward += extra_reward
        terminated = bool(terminated) or bool(done)

        return obs, reward, terminated, truncated, info

    # -- disable survival / day-night if flags set -----------------------

    def _apply_survival_drain(self) -> None:  # type: ignore[override]
        if getattr(self, "_disable_survival", False):
            return
        super()._apply_survival_drain()  # type: ignore[misc]

    def _advance_day_counter(self, steps: int = 1) -> None:  # type: ignore[override]
        if getattr(self, "_disable_day_night", False):
            # Still tick the counter (some rendering uses it) but skip
            # mob spawn / despawn logic.
            self._day_counter += steps  # type: ignore[attr-defined]
            return
        super()._advance_day_counter(steps)  # type: ignore[misc]

    # -- subclass hooks (abstract) ---------------------------------------

    def _setup_world(self, seed: int) -> None:
        raise NotImplementedError

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        """Return (extra_reward, terminated)."""
        raise NotImplementedError


# ======================================================================
# 1. CraftaxChopTreesEnv -- collect 5 wood
# ======================================================================

class CraftaxChopTreesEnv(_SubtaskMixin, CraftaxScenarioEnv):
    """Start in a meadow with trees.  Goal: collect 5 wood."""

    _subtask_max_turns = 120
    # Pattern A: +1/N per wood collected (sums to +1.0 at 5 wood).
    _WOOD_TARGET = 10

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "legend:projectiles",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee", "combat:projectiles",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-choptrees-v0"

    def _task_description(self) -> str:
        return (
            "Collect 10 wood by navigating a seed-generated grove and "
            "chopping trees. Face a tree and use DO. Reward: +1/10 per wood "
            "collected (sums to +1.0). Episode ends when you have 10 wood "
            "or after 120 steps."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        # Grass arena with trees around the edges and scattered inside
        for x in range(size):
            for y in range(size):
                # Border trees
                if x <= 1 or x >= size - 2 or y <= 1 or y >= size - 2 or self.rng.random() < 0.20:
                    self._world[y][x] = TILE_TREE
        # Clear agent spawn area
        cx, cy = size // 2, size // 2
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                self._world[cy + dy][cx + dx] = TILE_GRASS
        self._agent_x = cx
        self._agent_y = cy
        self._wood_progress = 0

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        wood = getattr(self, "_wood_progress", self._inventory.get("wood", 0))
        if wood >= self._WOOD_TARGET:
            self._message = "Goal complete! Collected 10 wood."
            return 0.0, True
        return 0.0, False

    # Pattern A: +1/N per wood collected (clipped at the target).
    def _handle_do(self) -> float:
        old_wood = self._inventory.get("wood", 0)
        reward = super()._handle_do()
        new_wood = self._inventory.get("wood", 0)
        progress = getattr(self, "_wood_progress", 0)
        gained = min(self._WOOD_TARGET - progress, max(0, new_wood - old_wood))
        if gained > 0:
            self._wood_progress = progress + gained
            reward += gained / self._WOOD_TARGET
        return reward


# ======================================================================
# 2. CraftaxMineStoneEnv -- mine 5 stone with wood pickaxe
# ======================================================================

class CraftaxMineStoneEnv(_SubtaskMixin, CraftaxScenarioEnv):
    """Start with a wood pickaxe near stone deposits.  Goal: mine 5 stone."""

    _subtask_max_turns = 120
    # Pattern A: +1/N per stone mined (sums to +1.0 at 5 stone).
    _STONE_TARGET = 10

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-minestone-v0"

    def _task_description(self) -> str:
        return (
            "Mine 10 stone from a seed-generated quarry. You start with a "
            "wood pickaxe. Face stone (S) and use DO to mine. Reward: +1/10 "
            "per stone mined (sums to +1.0). Episode ends when you have 10 "
            "stone or after 120 steps."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        cx, cy = size // 2, size // 2
        self._agent_x = cx
        self._agent_y = cy
        _scatter_tiles_far_from_spawn(
            self,
            tile=TILE_STONE,
            count=24,
            min_distance=5,
            radius=7,
        )
        _scatter_tiles_far_from_spawn(
            self,
            tile=TILE_TREE,
            count=6,
            min_distance=4,
            radius=6,
        )
        self._inventory["wood_pickaxe"] = 1
        self._stone_progress = 0

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        stone = getattr(self, "_stone_progress", self._inventory.get("stone", 0))
        if stone >= self._STONE_TARGET:
            self._message = "Goal complete! Mined 10 stone."
            return 0.0, True
        return 0.0, False

    def _handle_do(self) -> float:
        old_stone = self._inventory.get("stone", 0)
        reward = super()._handle_do()
        new_stone = self._inventory.get("stone", 0)
        progress = getattr(self, "_stone_progress", 0)
        gained = min(self._STONE_TARGET - progress, max(0, new_stone - old_stone))
        if gained > 0:
            self._stone_progress = progress + gained
            reward += gained / self._STONE_TARGET
        return reward


# ======================================================================
# 3. CraftaxGatherResourcesEnv -- get 3 wood + 3 stone (multi-step)
# ======================================================================

class CraftaxGatherResourcesEnv(_SubtaskMixin, CraftaxScenarioEnv):
    """Start empty-handed.  Collect wood, craft pickaxe, mine stone.
    Goal: have 5 wood + 5 stone in inventory.  Multi-step planning."""

    _subtask_max_turns = 160
    # Pattern A: 10 total progress units (5 wood + 5 stone) -> +1/10 each.
    _WOOD_TARGET = 5
    _STONE_TARGET = 5

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-gatherresources-v0"

    def _task_description(self) -> str:
        return (
            "Gather 5 wood and 5 stone. Reward: +1/10 per resource collected "
            "(sums to +1.0 when both targets are met)."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        cx, cy = size // 2, size // 2
        # Trees in top half, stone in bottom half
        for x in range(2, size - 2):
            for y in range(2, cy - 1):
                if self.rng.random() < 0.30:
                    self._world[y][x] = TILE_TREE
        for x in range(2, size - 2):
            for y in range(cy + 2, size - 2):
                if self.rng.random() < 0.30:
                    self._world[y][x] = TILE_STONE
        required_wood = self._WOOD_TARGET + 3  # table + wood pickaxe costs
        tree_slots = [
            (x, y)
            for y in range(2, cy - 1)
            for x in range(2, size - 2)
        ]
        stone_slots = [
            (x, y)
            for y in range(cy + 2, size - 2)
            for x in range(2, size - 2)
        ]
        existing_trees = sum(1 for x, y in tree_slots if self._world[y][x] == TILE_TREE)
        existing_stone = sum(1 for x, y in stone_slots if self._world[y][x] == TILE_STONE)
        for x, y in tree_slots:
            if existing_trees >= required_wood:
                break
            if self._world[y][x] == TILE_GRASS:
                self._world[y][x] = TILE_TREE
                existing_trees += 1
        for x, y in stone_slots:
            if existing_stone >= self._STONE_TARGET:
                break
            if self._world[y][x] == TILE_GRASS:
                self._world[y][x] = TILE_STONE
                existing_stone += 1
        # Clear spawn
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < size and 0 <= ny < size:
                    self._world[ny][nx] = TILE_GRASS
        existing_trees = sum(1 for x, y in tree_slots if self._world[y][x] == TILE_TREE)
        existing_stone = sum(1 for x, y in stone_slots if self._world[y][x] == TILE_STONE)
        for x, y in tree_slots:
            if existing_trees >= required_wood:
                break
            if self._world[y][x] == TILE_GRASS:
                self._world[y][x] = TILE_TREE
                existing_trees += 1
        for x, y in stone_slots:
            if existing_stone >= self._STONE_TARGET:
                break
            if self._world[y][x] == TILE_GRASS:
                self._world[y][x] = TILE_STONE
                existing_stone += 1
        self._agent_x = cx
        self._agent_y = cy
        self._wood_progress = 0
        self._stone_progress = 0

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        wood = self._inventory.get("wood", 0)
        stone = self._inventory.get("stone", 0)
        if wood >= self._WOOD_TARGET and stone >= self._STONE_TARGET:
            self._message = "Goal complete! Gathered 5 wood and 5 stone."
            return 0.0, True
        return 0.0, False

    def _handle_do(self) -> float:
        old_wood = self._inventory.get("wood", 0)
        old_stone = self._inventory.get("stone", 0)
        reward = super()._handle_do()
        new_wood = self._inventory.get("wood", 0)
        new_stone = self._inventory.get("stone", 0)
        target_total = self._WOOD_TARGET + self._STONE_TARGET
        wood_progress = getattr(self, "_wood_progress", 0)
        stone_progress = getattr(self, "_stone_progress", 0)
        wood_gained = min(
            self._WOOD_TARGET - wood_progress,
            max(0, new_wood - old_wood),
        )
        stone_gained = min(
            self._STONE_TARGET - stone_progress,
            max(0, new_stone - old_stone),
        )
        self._wood_progress = wood_progress + wood_gained
        self._stone_progress = stone_progress + stone_gained
        gained = wood_gained + stone_gained
        if gained > 0:
            reward += gained / target_total
        return reward


# ======================================================================
# 4. CraftaxCraftPickaxeEnv -- gather wood, place table, craft pickaxe
# ======================================================================

class CraftaxCraftPickaxeEnv(_SubtaskMixin, CraftaxScenarioEnv):
    """Start empty-handed in a small meadow.
    Goal: craft a wood pickaxe."""

    _subtask_max_turns = 140

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-craftpickaxe-v0"

    def _task_description(self) -> str:
        return (
            "Craft a wood pickaxe from scratch. Chop trees with DO to collect "
            "at least 3 wood, PLACE_TABLE using 2 wood, stand next to the "
            "table, then use MAKE_WOOD_PICKAXE with 1 wood. Reward: +1 on "
            "craft, episode ends. Time limit: 140 steps."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        cx, cy = size // 2, size // 2
        self._agent_x = cx
        self._agent_y = cy
        # Procedural meadow: enough nearby wood to build a table and a
        # pickaxe, but no pre-placed crafting station.
        for _ in range(14):
            for _attempt in range(30):
                dx = int(self.rng.integers(-5, 6))
                dy = int(self.rng.integers(-4, 5))
                if abs(dx) + abs(dy) < 2:
                    continue
                x, y = cx + dx, cy + dy
                if 1 <= x < size - 1 and 1 <= y < size - 1:
                    self._world[y][x] = TILE_TREE
                    break
        self._inventory = {}

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        # Pattern A: +1.0 on the terminal craft event.
        if self._inventory.get("wood_pickaxe", 0) >= 1:
            self._message = "Goal complete! Crafted wood pickaxe."
            return 1.0, True
        return 0.0, False


# ======================================================================
# 5. CraftaxCraftSwordEnv -- bootstrap to a stone sword
# ======================================================================

class CraftaxCraftSwordEnv(_SubtaskMixin, CraftaxScenarioEnv):
    """Start empty-handed near trees and stone.
    Goal: craft a stone sword."""

    _subtask_max_turns = 200

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-craftsword-v0"

    def _task_description(self) -> str:
        return (
            "Craft a stone sword from scratch. Chop wood, PLACE_TABLE, craft "
            "a wood pickaxe, mine stone with DO, then use MAKE_STONE_SWORD "
            "while adjacent to the table. Reward: +1 on craft, episode ends. "
            "Time limit: 200 steps."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        cx, cy = size // 2, size // 2
        self._agent_x = cx
        self._agent_y = cy
        # Seed-varied resource field. The task has all ingredients nearby,
        # but the agent must execute the whole wood -> table -> pickaxe ->
        # stone -> sword chain.
        for _ in range(16):
            for _attempt in range(30):
                dx = int(self.rng.integers(-5, 1))
                dy = int(self.rng.integers(-4, 5))
                if abs(dx) + abs(dy) < 2:
                    continue
                x, y = cx + dx, cy + dy
                if 1 <= x < size - 1 and 1 <= y < size - 1:
                    self._world[y][x] = TILE_TREE
                    break
        for _ in range(10):
            for _attempt in range(30):
                dx = int(self.rng.integers(1, 6))
                dy = int(self.rng.integers(-4, 5))
                if abs(dx) + abs(dy) < 2:
                    continue
                x, y = cx + dx, cy + dy
                if 1 <= x < size - 1 and 1 <= y < size - 1:
                    self._world[y][x] = TILE_STONE
                    break
        self._inventory = {}

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        # Pattern A: +1.0 on the terminal craft event.
        if self._inventory.get("stone_sword", 0) >= 1:
            self._message = "Goal complete! Crafted stone sword."
            return 1.0, True
        return 0.0, False


# ======================================================================
# 6. CraftaxCraftChainEnv -- full surface bootstrap chain
# ======================================================================

class CraftaxCraftChainEnv(_SubtaskMixin, CraftaxScenarioEnv):
    """Start empty near raw resources.
    Goal: bootstrap to an iron sword without preplaced stations."""

    _subtask_max_turns = 260
    # Pattern A: 6 coarse milestones → +1/6 each, capped at +1.0.
    _MILESTONE_COUNT = 6

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-craftchain-v0"

    def _task_description(self) -> str:
        return (
            "Bootstrap from an empty inventory to an iron sword. Chop enough "
            "wood, PLACE_TABLE, craft a wood pickaxe, mine stone, "
            "PLACE_FURNACE, craft a stone pickaxe, mine coal and iron, then "
            "craft an iron sword near the table and furnace. Reward: +1/6 "
            "per first-time milestone (table, wood pickaxe, furnace, stone "
            "pickaxe, coal+iron, iron sword), capped at +1.0. Episode ends "
            "when the iron sword is crafted or after 260 steps."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        cx, cy = size // 2, size // 2
        # Raw-resource pockets. Stations are not preplaced; the agent must
        # craft and place them from mined resources.
        layout = {
            (-5, -5): TILE_TREE, (-3, -5): TILE_TREE, (-1, -5): TILE_TREE,
            (-6, -3): TILE_TREE, (-4, -2): TILE_TREE, (-2, -3): TILE_TREE,
            (3, -5): TILE_STONE, (5, -5): TILE_STONE, (4, -3): TILE_STONE,
            (6, -2): TILE_STONE, (3, -1): TILE_STONE, (5, 0): TILE_STONE,
            (-6, 3): TILE_COAL, (-4, 4): TILE_COAL, (-2, 5): TILE_COAL,
            (2, 5): TILE_IRON, (4, 4): TILE_IRON, (6, 3): TILE_IRON,
        }
        for (dx, dy), tile in layout.items():
            x, y = cx + dx, cy + dy
            if 1 <= x < size - 1 and 1 <= y < size - 1:
                self._world[y][x] = tile
        # Seed-varied clutter keeps the visible start from being a fixed map.
        for _ in range(18):
            for _attempt in range(30):
                dx = int(self.rng.integers(-6, 7))
                dy = int(self.rng.integers(-6, 7))
                if abs(dx) + abs(dy) < 4:
                    continue
                x, y = cx + dx, cy + dy
                if (
                    1 <= x < size - 1
                    and 1 <= y < size - 1
                    and self._world[y][x] == TILE_GRASS
                ):
                    self._world[y][x] = str(self.rng.choice([
                        TILE_TREE,
                        TILE_STONE,
                        TILE_COAL,
                        TILE_IRON,
                    ]))
                    break
        _clear_area(self._world, cx, cy, 2)
        self._agent_x = cx
        self._agent_y = cy
        # Track milestone rewards
        self._chain_milestones: set[str] = set()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        per_milestone = 1.0 / self._MILESTONE_COUNT
        extra = 0.0
        has_table = any(TILE_TABLE in row for row in self._world)
        has_furnace = any(TILE_FURNACE in row for row in self._world)
        if "table_placed" not in self._chain_milestones and has_table:
            self._chain_milestones.add("table_placed")
            extra += per_milestone
        if (
            "pickaxe_crafted" not in self._chain_milestones
            and self._inventory.get("wood_pickaxe", 0) >= 1
        ):
            self._chain_milestones.add("pickaxe_crafted")
            extra += per_milestone
        if "furnace_placed" not in self._chain_milestones and has_furnace:
            self._chain_milestones.add("furnace_placed")
            extra += per_milestone
        if (
            "stone_pickaxe_crafted" not in self._chain_milestones
            and self._inventory.get("stone_pickaxe", 0) >= 1
        ):
            self._chain_milestones.add("stone_pickaxe_crafted")
            extra += per_milestone
        if (
            "iron_coal_mined" not in self._chain_milestones
            and self._inventory.get("iron", 0) >= 1
            and self._inventory.get("coal", 0) >= 1
        ):
            self._chain_milestones.add("iron_coal_mined")
            extra += per_milestone
        if self._inventory.get("iron_sword", 0) >= 1:
            if "iron_sword_crafted" not in self._chain_milestones:
                self._chain_milestones.add("iron_sword_crafted")
                extra += per_milestone
            self._message = "Goal complete! Crafted iron sword via chain."
            return extra, True
        return extra, False


# ======================================================================
# 7. CraftaxFightZombieEnv -- kill one zombie
# ======================================================================

class CraftaxFightZombieEnv(_SubtaskMixin, CraftaxScenarioEnv):
    """Start with a wood sword, two zombies nearby.
    Goal: kill both without dying."""

    _subtask_max_turns = 50
    _disable_survival = False
    # Pattern D: Kill 2 zombies → +1.0; death → -1.0.
    _KILL_TARGET = 2
    _DEATH_PENALTY = -1.0

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-fightzombie-v0"

    def _task_description(self) -> str:
        return (
            "Fight two zombies on the overworld with only a wood sword. Each "
            "zombie has 5 HP, so you must position carefully and land several "
            "DO attacks while facing the target. Reward: +1 after both kills, "
            "-1 on death. Time limit: 50 steps."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        cx, cy = size // 2, size // 2
        self._agent_x = cx
        self._agent_y = cy
        self._inventory["wood_sword"] = 1
        # Spawn two zombies at seed-varied offsets, far enough that the
        # agent must maneuver before attacking.
        offsets = [(4, 0), (-4, 0), (0, 3), (0, -3), (3, 2), (-3, -2)]
        start = seed % len(offsets)
        for idx in range(self._KILL_TARGET):
            dx, dy = offsets[(start + idx * 2) % len(offsets)]
            _spawn_mob(self._mobs, "zombie", cx + dx, cy + dy)
        # Add harmless visible terrain variation so the initial observation
        # is not the same arena every seed.
        terrain_offsets = [(-4, -3), (4, -3), (-4, 3), (4, 3)]
        tx, ty = terrain_offsets[(seed // len(offsets)) % len(terrain_offsets)]
        self._world[cy + ty][cx + tx] = TILE_TREE

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        # Check if zombie is dead (no hostile mobs left)
        hostiles = [m for m in self._mobs if m["type"] != "cow"]
        if not hostiles:
            self._message = "Goal complete! Zombie defeated."
            return 1.0, True
        # Death handled by parent (terminated = True when hp <= 0)
        if self._hp <= 0:
            return self._DEATH_PENALTY, True
        return 0.0, False


# ======================================================================
# 8. CraftaxSurviveHordeEnv -- survive and kill 5 mobs
# ======================================================================

class CraftaxSurviveHordeEnv(_SubtaskMixin, CraftaxScenarioEnv):
    """Start with stone sword. Survive a staggered night horde."""

    _subtask_max_turns = 160
    _disable_survival = False
    _disable_day_night = False
    # Pattern D: 8 mobs to kill. +1/N per kill (sums to +1.0). -1.0 on death.
    _KILL_TARGET = 8
    _DEATH_PENALTY = -1.0
    _EXTRA_SPAWN_STEPS = (25, 50, 75)

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-survivehorde-v0"

    def _task_description(self) -> str:
        return (
            "Survive a staggered night horde. Skeletons fire arrows; use DO "
            "to fight, PLACE_STONE for emergency cover, and sleep/rest only "
            "when safe. You have a stone sword, not an iron sword, and more "
            "mobs arrive during the episode. Reward: +1/8 per mob killed; "
            "-1 on death."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        cx, cy = size // 2, size // 2
        self._agent_x = cx
        self._agent_y = cy
        self._inventory["stone_sword"] = 1
        self._inventory["stone"] = 6
        self._hp = 9
        self._day_counter = _DAY_LENGTH
        self._day_night = "night"
        # Spawn 3 zombies and 2 skeletons in a ring around the player
        positions = [
            (cx + 3, cy),
            (cx - 3, cy),
            (cx, cy + 3),
            (cx + 2, cy + 2),
            (cx - 2, cy - 2),
        ]
        if seed % 2:
            positions = [(cx - (x - cx), y) for x, y in positions]
        if (seed // 2) % 2:
            positions = [(x, cy - (y - cy)) for x, y in positions]
        for i, (mx, my) in enumerate(positions):
            mob_type = "zombie" if i < 3 else "skeleton"
            _spawn_mob(self._mobs, mob_type, mx, my)
        self._horde_kills: int = 0
        self._horde_extra_spawned = 0

    def _spawn_extra_horde_mob(self) -> None:
        size = self._WORLD_SIZE
        cx, cy = self._agent_x, self._agent_y
        mob_type = "skeleton" if self._horde_extra_spawned % 2 else "zombie"
        for _attempt in range(80):
            dx = int(self.rng.integers(-7, 8))
            dy = int(self.rng.integers(-7, 8))
            if abs(dx) + abs(dy) < 5:
                continue
            x, y = cx + dx, cy + dy
            if (
                0 <= x < size
                and 0 <= y < size
                and self._world[y][x] == TILE_GRASS
                and not self._mob_at(x, y)
            ):
                _spawn_mob(self._mobs, mob_type, x, y)
                self._horde_extra_spawned += 1
                return

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _handle_do(self) -> float:
        old_count = len([m for m in self._mobs if m["type"] != "cow"])
        reward = super()._handle_do()
        new_count = len([m for m in self._mobs if m["type"] != "cow"])
        killed = max(0, old_count - new_count)
        if killed > 0:
            previous_kills = self._horde_kills
            self._horde_kills = min(self._KILL_TARGET, self._horde_kills + killed)
            credited = self._horde_kills - previous_kills
            reward += credited * (1.0 / self._KILL_TARGET)  # +1/N per mob killed
        return reward

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        if self._horde_kills >= self._KILL_TARGET:
            self._message = "Goal complete! Horde defeated."
            info["subtask_success"] = True
            return 0.0, True
        if self._hp <= 0:
            info["subtask_success"] = False
            return self._DEATH_PENALTY, True
        return 0.0, False

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        if (
            self._horde_extra_spawned < len(self._EXTRA_SPAWN_STEPS)
            and self._turn >= self._EXTRA_SPAWN_STEPS[self._horde_extra_spawned]
        ):
            self._spawn_extra_horde_mob()
        return super()._step(action)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        return GridObservation(
            grid=obs.grid,
            legend=obs.legend,
            hud=obs.hud + f"    Horde kills: {self._horde_kills}/{self._KILL_TARGET}",
            message=obs.message,
        )


# ======================================================================
# 9. CraftaxDungeonExploreEnv -- find stairs down in a dungeon
# ======================================================================

class CraftaxDungeonExploreEnv(_SubtaskMixin, CraftaxScenarioEnv):
    """Start at dungeon entrance. Defeat guards, then find stairs down."""

    _subtask_max_turns = 180
    _WORLD_SIZE = 24  # slightly larger for dungeon layout
    _KILL_TARGET = 3
    # Death is a real risk on the dungeon floor (skeletons + zombies). Make
    # it discriminative vs timeout: -1 on death, 0 on timeout, +1 on win.
    _DEATH_PENALTY = -1.0

    tutorial_sections = (
        "legend:player", "legend:terrain",
        "legend:mobs:overworld",
        "legend:items",
        "survival:hp_food_drink", "survival:energy_sleep",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-dungeonexplore-v0"

    def _task_description(self) -> str:
        return (
            "Explore the dungeon (floor 1). Find the stairs down (⇣) by "
            "navigating rooms and corridors, but the stairs are sealed until "
            "you defeat 3 floor mobs. In this focused scenario, standing on "
            "the stairs after those kills completes the episode even though "
            "DESCEND/ASCEND remain part of the universal action menu. You "
            "begin with a stone sword. Reward: +1 "
            "for reaching the stairs after those kills, -1 on death, 0 on timeout."
        )

    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        # Fill with dungeon wall
        self._world = [[TILE_DUNGEON_WALL for _ in range(size)] for _ in range(size)]

        # Generate rooms
        rooms: list[tuple[int, int, int, int]] = []
        for _ in range(int(self.rng.integers(4, 7))):
            rw = int(self.rng.integers(3, 6))
            rh = int(self.rng.integers(3, 6))
            rx = int(self.rng.integers(1, size - rw - 1))
            ry = int(self.rng.integers(1, size - rh - 1))
            overlap = False
            for ex_rx, ex_ry, ex_rw, ex_rh in rooms:
                if (
                    rx < ex_rx + ex_rw + 1
                    and rx + rw + 1 > ex_rx
                    and ry < ex_ry + ex_rh + 1
                    and ry + rh + 1 > ex_ry
                ):
                    overlap = True
                    break
            if overlap:
                continue
            rooms.append((rx, ry, rw, rh))
            for dy in range(rh):
                for dx in range(rw):
                    self._world[ry + dy][rx + dx] = TILE_DUNGEON_FLOOR

        # Ensure at least 2 rooms
        if len(rooms) < 2:
            rooms = [(2, 2, 5, 5), (size - 8, size - 8, 5, 5)]
            for rx, ry, rw, rh in rooms:
                for dy in range(rh):
                    for dx in range(rw):
                        if 0 <= ry + dy < size and 0 <= rx + dx < size:
                            self._world[ry + dy][rx + dx] = TILE_DUNGEON_FLOOR

        # Connect rooms with corridors
        for i in range(len(rooms) - 1):
            r1 = rooms[i]
            r2 = rooms[i + 1]
            cx1 = r1[0] + r1[2] // 2
            cy1 = r1[1] + r1[3] // 2
            cx2 = r2[0] + r2[2] // 2
            cy2 = r2[1] + r2[3] // 2
            x = cx1
            while x != cx2:
                if 0 <= x < size and 0 <= cy1 < size:
                    self._world[cy1][x] = TILE_DUNGEON_FLOOR
                x += 1 if cx2 > cx1 else -1
            if 0 <= cx2 < size and 0 <= cy1 < size:
                self._world[cy1][cx2] = TILE_DUNGEON_FLOOR
            y = cy1
            while y != cy2:
                if 0 <= cx2 < size and 0 <= y < size:
                    self._world[y][cx2] = TILE_DUNGEON_FLOOR
                y += 1 if cy2 > cy1 else -1
            if 0 <= cx2 < size and 0 <= cy2 < size:
                self._world[cy2][cx2] = TILE_DUNGEON_FLOOR

        # Agent starts in first room
        r0 = rooms[0]
        self._agent_x = r0[0] + r0[2] // 2
        self._agent_y = r0[1] + r0[3] // 2

        # Stairs down in last room
        rl = rooms[-1]
        self._stairs_goal = (
            rl[0] + rl[2] // 2,
            rl[1] + rl[3] // 2,
        )
        sx, sy = self._stairs_goal
        self._world[sy][sx] = TILE_STAIRS_DOWN
        self._inventory = {"stone_sword": 1}
        self._explore_kills = 0

        # Place three seed-varied guards on reachable dungeon floor cells.
        guard_types = ("zombie", "skeleton", "zombie")
        reserved = {(self._agent_x, self._agent_y), self._stairs_goal}
        for guard_type in guard_types:
            for _attempt in range(80):
                x = int(self.rng.integers(1, size - 1))
                y = int(self.rng.integers(1, size - 1))
                if (
                    self._world[y][x] == TILE_DUNGEON_FLOOR
                    and (x, y) not in reserved
                    and not self._mob_at(x, y)
                ):
                    _spawn_mob(self._mobs, guard_type, x, y)
                    reserved.add((x, y))
                    break

    def _handle_move(self, name: str) -> float:
        return super()._handle_move(name)

    def _mob_ai(self) -> None:
        super()._mob_ai()

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        if self._hp <= 0:
            info["subtask_success"] = False
            return self._DEATH_PENALTY, True
        sx, sy = self._stairs_goal
        if self._agent_x == sx and self._agent_y == sy:
            if self._explore_kills >= self._KILL_TARGET:
                self._message = "Goal complete! Cleared guards and found the stairs down."
                info["subtask_success"] = True
                return 1.0, True
            self._message = (
                f"Stairs sealed: defeat {self._KILL_TARGET - self._explore_kills} "
                f"more floor mobs ({self._explore_kills}/{self._KILL_TARGET})."
            )
        return 0.0, False

    def _handle_do(self) -> float:
        old_count = len([m for m in self._mobs if m["type"] != "cow"])
        reward = super()._handle_do()
        new_count = len([m for m in self._mobs if m["type"] != "cow"])
        if new_count < old_count:
            self._explore_kills += old_count - new_count
        return reward

    # Custom rendering for dungeon tiles
    def _render_current_observation(self) -> GridObservation:
        from glyphbench.envs.craftaxfull.full import _DIR_CHARS, _DIR_NAMES

        half_w = VIEW_WIDTH // 2
        half_h = VIEW_HEIGHT // 2
        grid: list[list[str]] = []
        symbols_seen: set[str] = set()
        facing_ch = _DIR_CHARS.get(self._facing, "@")
        facing_name = _DIR_NAMES.get(self._facing, "right")

        mob_chars: dict[tuple[int, int], str] = {}
        visible_mobs: list[Mob] = []
        for mob in self._mobs:
            mob_chars[(mob["x"], mob["y"])] = _MOB_TILES[mob["type"]]
        projectile_chars = {
            (p.x, p.y): TILE_ARROW
            for p in getattr(self, "_mob_projectiles", [])
        }

        for wy in range(VIEW_HEIGHT):
            row: list[str] = []
            for wx in range(VIEW_WIDTH):
                world_x = self._agent_x - half_w + wx
                world_y = self._agent_y - half_h + wy
                if world_x == self._agent_x and world_y == self._agent_y:
                    row.append(facing_ch)
                    symbols_seen.add(facing_ch)
                elif (world_x, world_y) in mob_chars:
                    char = mob_chars[(world_x, world_y)]
                    row.append(char)
                    symbols_seen.add(char)
                elif (world_x, world_y) in projectile_chars:
                    char = projectile_chars[(world_x, world_y)]
                    row.append(char)
                    symbols_seen.add(char)
                elif (
                    0 <= world_x < self._WORLD_SIZE
                    and 0 <= world_y < self._WORLD_SIZE
                ):
                    tile = self._world[world_y][world_x]
                    row.append(tile)
                    symbols_seen.add(tile)
                else:
                    row.append(TILE_DUNGEON_WALL)
                    symbols_seen.add(TILE_DUNGEON_WALL)
            grid.append(row)

        for mob in self._mobs:
            dx = mob["x"] - self._agent_x
            dy = mob["y"] - self._agent_y
            if abs(dx) <= half_w and abs(dy) <= half_h:
                visible_mobs.append(mob)

        mob_parts = [
            f"{m['type']} ({m['hp']}/{m['max_hp']} HP)"
            for m in visible_mobs
        ]
        mob_str = ", ".join(mob_parts) if mob_parts else "none"
        hostiles_left = len([m for m in self._mobs if m["type"] != "cow"])

        hud = (
            f"Step: {self._turn} / {self.max_turns}  HP: {self._hp}/{self._max_hp}\n"
            "Next drain: disabled  Time: disabled\n"
            f"Nearby mobs: {mob_str}\n"
            f"Enemies remaining: {hostiles_left}  "
            f"Kills: {self._explore_kills}/{self._KILL_TARGET}\n"
            "Inventory: stone_sword x1"
        )

        tile_meanings: dict[str, str] = {
            TILE_DUNGEON_WALL: "dungeon wall (impassable)",
            TILE_DUNGEON_FLOOR: "dungeon floor",
            TILE_STAIRS_DOWN: "stairs down (GOAL)",
            TILE_ZOMBIE: "zombie (hostile)",
            TILE_SKELETON: "skeleton (ranged, hostile)",
            TILE_ARROW: "arrow projectile",
        }
        legend_entries: dict[str, str] = {}
        agent_legend = f"you (facing {facing_name})"
        for sym in symbols_seen:
            if sym == facing_ch:
                legend_entries[sym] = agent_legend
            elif sym in tile_meanings:
                legend_entries[sym] = tile_meanings[sym]
        legend = build_legend(legend_entries)

        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message=self._message,
        )


# ======================================================================
# 10. CraftaxDungeonClearEnv -- kill all enemies on a dungeon floor
# ======================================================================

class CraftaxDungeonClearEnv(_SubtaskMixin, CraftaxScenarioEnv):
    """Start in dungeon with a sword.  Kill all enemies on the floor."""

    _subtask_max_turns = 220
    _WORLD_SIZE = 24
    # Pattern D: kills sum to +1.0; death → -1.0.
    _DEATH_PENALTY = -1.0

    tutorial_sections = (
        "legend:player", "legend:terrain",
        "legend:mobs:overworld",
        "legend:items",
        "survival:hp_food_drink", "survival:energy_sleep",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-dungeonclear-v0"

    def _task_description(self) -> str:
        return (
            "Clear all hostile mobs on dungeon floor 1. Reward: +1/N per mob "
            "killed (sums to +1.0 when the floor is cleared); -1 on death. "
            "Episode ends when all mobs are dead or you die."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        # Fill with dungeon wall
        self._world = [[TILE_DUNGEON_WALL for _ in range(size)] for _ in range(size)]

        # Generate rooms
        rooms: list[tuple[int, int, int, int]] = []
        for _ in range(int(self.rng.integers(4, 7))):
            rw = int(self.rng.integers(4, 7))
            rh = int(self.rng.integers(4, 7))
            rx = int(self.rng.integers(1, size - rw - 1))
            ry = int(self.rng.integers(1, size - rh - 1))
            overlap = False
            for ex_rx, ex_ry, ex_rw, ex_rh in rooms:
                if (
                    rx < ex_rx + ex_rw + 1
                    and rx + rw + 1 > ex_rx
                    and ry < ex_ry + ex_rh + 1
                    and ry + rh + 1 > ex_ry
                ):
                    overlap = True
                    break
            if overlap:
                continue
            rooms.append((rx, ry, rw, rh))
            for dy in range(rh):
                for dx in range(rw):
                    self._world[ry + dy][rx + dx] = TILE_DUNGEON_FLOOR

        # Ensure at least 2 rooms
        if len(rooms) < 2:
            rooms = [(2, 2, 6, 6), (size - 9, size - 9, 6, 6)]
            for rx, ry, rw, rh in rooms:
                for dy in range(rh):
                    for dx in range(rw):
                        if 0 <= ry + dy < size and 0 <= rx + dx < size:
                            self._world[ry + dy][rx + dx] = TILE_DUNGEON_FLOOR

        # Connect rooms with corridors
        for i in range(len(rooms) - 1):
            r1 = rooms[i]
            r2 = rooms[i + 1]
            cx1 = r1[0] + r1[2] // 2
            cy1 = r1[1] + r1[3] // 2
            cx2 = r2[0] + r2[2] // 2
            cy2 = r2[1] + r2[3] // 2
            x = cx1
            while x != cx2:
                if 0 <= x < size and 0 <= cy1 < size:
                    self._world[cy1][x] = TILE_DUNGEON_FLOOR
                x += 1 if cx2 > cx1 else -1
            if 0 <= cx2 < size and 0 <= cy1 < size:
                self._world[cy1][cx2] = TILE_DUNGEON_FLOOR
            y = cy1
            while y != cy2:
                if 0 <= cx2 < size and 0 <= y < size:
                    self._world[y][cx2] = TILE_DUNGEON_FLOOR
                y += 1 if cy2 > cy1 else -1
            if 0 <= cx2 < size and 0 <= cy2 < size:
                self._world[cy2][cx2] = TILE_DUNGEON_FLOOR

        # Agent starts in first room
        r0 = rooms[0]
        self._agent_x = r0[0] + r0[2] // 2
        self._agent_y = r0[1] + r0[3] // 2
        self._inventory["iron_sword"] = 1
        self._hp = 9

        # Spawn enemies in other rooms
        self._initial_mob_count = 0
        for room_idx in range(1, len(rooms)):
            r = rooms[room_idx]
            # 1-2 enemies per room
            num = int(self.rng.integers(1, 3))
            for _ in range(num):
                mob_type = "zombie" if self.rng.random() < 0.5 else "skeleton"
                for _att in range(20):
                    mx = int(self.rng.integers(r[0], r[0] + r[2]))
                    my = int(self.rng.integers(r[1], r[1] + r[3]))
                    if (
                        self._world[my][mx] == TILE_DUNGEON_FLOOR
                        and not self._mob_at(mx, my)
                        and (mx, my) != (self._agent_x, self._agent_y)
                    ):
                        _spawn_mob(self._mobs, mob_type, mx, my)
                        self._initial_mob_count += 1
                        break

    def _handle_move(self, name: str) -> float:
        return super()._handle_move(name)

    def _mob_ai(self) -> None:
        super()._mob_ai()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _handle_do(self) -> float:
        old_count = len(self._mobs)
        reward = super()._handle_do()
        new_count = len(self._mobs)
        killed = old_count - new_count
        if killed > 0 and self._initial_mob_count > 0:
            reward += killed * (1.0 / self._initial_mob_count)
        return reward

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        if self._hp <= 0:
            return self._DEATH_PENALTY, True
        hostiles = [m for m in self._mobs if m["type"] != "cow"]
        if not hostiles:
            self._message = "Goal complete! Dungeon floor cleared."
            info["subtask_success"] = True
            return 0.0, True
        return 0.0, False

    # Custom rendering for dungeon tiles
    def _render_current_observation(self) -> GridObservation:
        from glyphbench.envs.craftaxfull.full import _DIR_CHARS, _DIR_NAMES

        half_w = VIEW_WIDTH // 2
        half_h = VIEW_HEIGHT // 2
        grid: list[list[str]] = []
        symbols_seen: set[str] = set()
        facing_ch = _DIR_CHARS.get(self._facing, "@")
        facing_name = _DIR_NAMES.get(self._facing, "right")

        mob_chars: dict[tuple[int, int], str] = {}
        visible_mobs: list[Mob] = []
        for mob in self._mobs:
            mob_chars[(mob["x"], mob["y"])] = _MOB_TILES[mob["type"]]
        projectile_chars = {
            (p.x, p.y): TILE_ARROW
            for p in getattr(self, "_mob_projectiles", [])
        }

        for wy in range(VIEW_HEIGHT):
            row: list[str] = []
            for wx in range(VIEW_WIDTH):
                world_x = self._agent_x - half_w + wx
                world_y = self._agent_y - half_h + wy
                if world_x == self._agent_x and world_y == self._agent_y:
                    row.append(facing_ch)
                    symbols_seen.add(facing_ch)
                elif (world_x, world_y) in mob_chars:
                    char = mob_chars[(world_x, world_y)]
                    row.append(char)
                    symbols_seen.add(char)
                elif (world_x, world_y) in projectile_chars:
                    char = projectile_chars[(world_x, world_y)]
                    row.append(char)
                    symbols_seen.add(char)
                elif (
                    0 <= world_x < self._WORLD_SIZE
                    and 0 <= world_y < self._WORLD_SIZE
                ):
                    tile = self._world[world_y][world_x]
                    row.append(tile)
                    symbols_seen.add(tile)
                else:
                    row.append(TILE_DUNGEON_WALL)
                    symbols_seen.add(TILE_DUNGEON_WALL)
            grid.append(row)

        for mob in self._mobs:
            dx = mob["x"] - self._agent_x
            dy = mob["y"] - self._agent_y
            if abs(dx) <= half_w and abs(dy) <= half_h:
                visible_mobs.append(mob)

        mob_parts = []
        for m in visible_mobs:
            mob_parts.append(f"{m['type']} ({m['hp']}/{m['max_hp']} HP)")
        mob_str = ", ".join(mob_parts) if mob_parts else "none"

        inv_parts = []
        for item, count in sorted(self._inventory.items()):
            if isinstance(count, dict):
                for subitem, subcount in sorted(count.items()):
                    if subcount > 0:
                        inv_parts.append(f"{item}.{subitem} x{subcount}")
            elif count > 0:
                inv_parts.append(f"{item} x{count}")
        inv_str = ", ".join(inv_parts) if inv_parts else "(empty)"

        hostiles_left = len([m for m in self._mobs if m["type"] != "cow"])

        hud = (
            f"HP: {self._hp}/{self._max_hp}  "
            f"Step: {self._turn} / {self.max_turns}\n"
            "Next drain: disabled  Time: disabled\n"
            f"Nearby mobs: {mob_str}\n"
            f"Inventory: {inv_str}\n"
            f"Enemies remaining: {hostiles_left}"
        )

        tile_meanings: dict[str, str] = {
            TILE_DUNGEON_WALL: "dungeon wall (impassable)",
            TILE_DUNGEON_FLOOR: "dungeon floor",
            TILE_ZOMBIE: "zombie (hostile)",
            TILE_SKELETON: "skeleton (ranged, hostile)",
            TILE_ARROW: "arrow projectile",
        }
        legend_entries: dict[str, str] = {}
        agent_legend = f"you (facing {facing_name})"
        for sym in symbols_seen:
            if sym == facing_ch:
                legend_entries[sym] = agent_legend
            elif sym in tile_meanings:
                legend_entries[sym] = tile_meanings[sym]
        legend = build_legend(legend_entries)

        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message=self._message,
        )
