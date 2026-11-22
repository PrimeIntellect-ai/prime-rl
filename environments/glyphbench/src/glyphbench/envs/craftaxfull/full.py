"""In-repo Craftax mechanics environment (93 achievements).

Implements the single Craftax mechanics universe used by focused GlyphBench
Craftax scenarios: multi-floor dungeons, magic, bosses, armor, enchantments,
potions, and the full mob roster.

Gym ID: glyphbench/craftaxfull-v0
"""

from __future__ import annotations

import random
from typing import Any, TypedDict

from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.craftax.base import (
    ALL_FULL_ACHIEVEMENTS,
    CRAFTAX_FULL_ACTION_SPEC,
    FULL_VIEW_HEIGHT,
    FULL_VIEW_WIDTH,
    TILE_ARROW,
    TILE_ARROW2,
    TILE_BAT,
    TILE_BOSS,
    TILE_BOSS_DOOR,
    TILE_CHEST,
    TILE_COAL,
    TILE_COW,
    TILE_DAGGER,
    TILE_DEEP_THING,
    TILE_DIAMOND,
    TILE_DUNGEON_FLOOR,
    TILE_DUNGEON_WALL,
    TILE_ENCHANT_FIRE,
    TILE_ENCHANT_ICE,
    TILE_FIRE_ELEMENTAL,
    TILE_FIRE_TREE,
    TILE_FIREBALL,
    TILE_FIREBALL2,
    TILE_FOUNTAIN,
    TILE_FROST_TROLL,
    TILE_FURNACE,
    TILE_GNOME_ARCHER,
    TILE_GNOME_WARRIOR,
    TILE_GRASS,
    TILE_GRAVE,
    TILE_ICE_ELEMENTAL,
    TILE_ICE_SHRUB,
    TILE_ICEBALL,
    TILE_ICEBALL2,
    TILE_IRON,
    TILE_KNIGHT,
    TILE_KNIGHT_ARCHER,
    TILE_KOBOLD,
    TILE_LAVA,
    TILE_LIZARD,
    TILE_NECROMANCER,
    TILE_NECROMANCER_VULNERABLE,
    TILE_ORC_MAGE,
    TILE_ORC_SOLDIER,
    TILE_PIGMAN,
    TILE_PLACED_STONE,
    TILE_PLANT,
    TILE_RIPE_PLANT,
    TILE_RUBY,
    TILE_SAND,
    TILE_SAPLING,
    TILE_SAPPHIRE,
    TILE_SKELETON_ARCHER,
    TILE_SLIMEBALL,
    TILE_SNAIL,
    TILE_STAIRS_DOWN,
    TILE_STAIRS_UP,
    TILE_STONE,
    TILE_TABLE,
    TILE_TREE,
    TILE_TROLL,
    TILE_WATER,
    TILE_ZOMBIE,
    _CraftaxTutorialMixin,
)

# ----------------------------------------------------------------
# Upstream achievement names (from Craftax-main constants.py Achievement enum)
# Used by phase-β tasks to fire upstream-named achievements.
# ----------------------------------------------------------------
UPSTREAM_ACHIEVEMENT_NAMES: tuple[str, ...] = (
    "COLLECT_WOOD",
    "PLACE_TABLE",
    "EAT_COW",
    "COLLECT_SAPLING",
    "COLLECT_DRINK",
    "MAKE_WOOD_PICKAXE",
    "MAKE_WOOD_SWORD",
    "PLACE_PLANT",
    "DEFEAT_ZOMBIE",
    "COLLECT_STONE",
    "PLACE_STONE",
    "EAT_PLANT",
    "DEFEAT_SKELETON",
    "MAKE_STONE_PICKAXE",
    "MAKE_STONE_SWORD",
    "WAKE_UP",
    "PLACE_FURNACE",
    "COLLECT_COAL",
    "COLLECT_IRON",
    "COLLECT_DIAMOND",
    "MAKE_IRON_PICKAXE",
    "MAKE_IRON_SWORD",
    "MAKE_ARROW",
    "COLLECT_SAPPHIRE",
    "COLLECT_RUBY",
    "MAKE_DIAMOND_PICKAXE",
    "MAKE_DIAMOND_SWORD",
    "MAKE_IRON_ARMOUR",
    "MAKE_DIAMOND_ARMOUR",
    "ENTER_GNOMISH_MINES",
    "ENTER_DUNGEON",
    "ENTER_SEWERS",
    "ENTER_VAULT",
    "ENTER_TROLL_MINES",
    "ENTER_FIRE_REALM",
    "ENTER_ICE_REALM",
    "ENTER_GRAVEYARD",
    "DEFEAT_GNOME_WARRIOR",
    "DEFEAT_GNOME_ARCHER",
    "DEFEAT_ORC_SOLIDER",
    "DEFEAT_ORC_MAGE",
    "DEFEAT_LIZARD",
    "DEFEAT_KOBOLD",
    "DEFEAT_KNIGHT",
    "DEFEAT_ARCHER",
    "DEFEAT_TROLL",
    "DEFEAT_DEEP_THING",
    "DEFEAT_PIGMAN",
    "DEFEAT_FIRE_ELEMENTAL",
    "DEFEAT_FROST_TROLL",
    "DEFEAT_ICE_ELEMENTAL",
    "DAMAGE_NECROMANCER",
    "DEFEAT_NECROMANCER",
    "EAT_BAT",
    "EAT_SNAIL",
    "FIND_BOW",
    "FIRE_BOW",
    "LEARN_FIREBALL",
    "CAST_FIREBALL",
    "LEARN_ICEBALL",
    "CAST_ICEBALL",
    "OPEN_CHEST",
    "DRINK_POTION",
    "ENCHANT_SWORD",
    "ENCHANT_ARMOUR",
    # Torch/lighting system (2026-06 port-vs-fork alignment) — fork
    # Achievement enum includes MAKE_TORCH/PLACE_TORCH.
    "MAKE_TORCH",
    "PLACE_TORCH",
)

# ----------------------------------------------------------------
# Phase γ T21γ: mapping from glyphbench lowercase achievement names to their
# upstream UPPERCASE bitmap keys (for the cases that differ in spelling).
# ----------------------------------------------------------------
_BITMAP_ALIAS: dict[str, str] = {
    "ENCHANT_ARMOR": "ENCHANT_ARMOUR",
    "MAKE_IRON_ARMOR": "MAKE_IRON_ARMOUR",
    "MAKE_DIAMOND_ARMOR": "MAKE_DIAMOND_ARMOUR",
}

# ----------------------------------------------------------------
# Walkable / interactable tile sets
# ----------------------------------------------------------------
SURFACE_WALKABLE = frozenset({
    TILE_GRASS, TILE_SAND, TILE_TABLE, TILE_FURNACE,
    TILE_PLACED_STONE, TILE_SAPLING, TILE_RIPE_PLANT,
    TILE_STAIRS_DOWN,
    # Phase β T07β: lava is now walkable but deals 2 dmg/tick.
    TILE_LAVA,
})

DUNGEON_WALKABLE = frozenset({
    TILE_DUNGEON_FLOOR, TILE_TABLE, TILE_FURNACE,
    TILE_PLACED_STONE, TILE_STAIRS_DOWN, TILE_STAIRS_UP,
    TILE_BOSS_DOOR,
    # Phase β T07β: lava is walkable in dungeons too.
    TILE_LAVA,
    # Phase γ T13-T15γ: smoothgen biome base tile (floor path) is DUNGEON_FLOOR;
    # water appears on floor 7 — also walkable.
    TILE_WATER,
})

_CLOSE_BLOCKS = (
    (0, -1), (0, 1), (-1, 0), (1, 0),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)

INTERACTABLE_TILES: dict[str, str] = {
    TILE_TREE: "wood",
    TILE_FIRE_TREE: "wood",
    TILE_ICE_SHRUB: "wood",
    TILE_STONE: "stone",
    TILE_COAL: "coal",
    TILE_IRON: "iron",
    TILE_DIAMOND: "diamond",
    # Phase β ore gems (T05β/T06β)
    TILE_SAPPHIRE: "sapphire",
    TILE_RUBY: "ruby",
}

PICKAXE_REQUIRED: dict[str, int] = {
    TILE_STONE: 0,
    TILE_COAL: 0,
    TILE_IRON: 1,
    # Diamond needs an iron pickaxe (fork: pickaxe >= 3 = iron).
    TILE_DIAMOND: 2,
    # Gem ores need a DIAMOND pickaxe (fork: pickaxe >= 4 = diamond).
    TILE_SAPPHIRE: 3,
    TILE_RUBY: 3,
}

_PICKAXE_TIERS = (
    "wood_pickaxe", "stone_pickaxe",
    "iron_pickaxe", "diamond_pickaxe",
)

# ----------------------------------------------------------------
# Timing constants
# ----------------------------------------------------------------
_DAY_LENGTH = 200
_NIGHT_LENGTH = 100
_CYCLE_LENGTH = _DAY_LENGTH + _NIGHT_LENGTH

# Night mob spawn base chance. Visibility is always on; night mobs no longer
# key off light levels.
_NIGHT_SPAWN_BASE_CHANCE = 1.0

_FOOD_DRAIN_INTERVAL = 50
_WATER_DRAIN_INTERVAL = 40
_ENERGY_DRAIN_INTERVAL = 100

_MAX_FOOD = 9
_MAX_WATER = 9
_MAX_ENERGY = 9
_MAX_MANA = 10
_MANA_REGEN_INTERVAL = 20
_FAST_FORWARD_MAX_TICKS = 10000

# Phase γ T09γ: base max HP before attribute scaling.
_BASE_MAX_HP = 9

_PLANT_RIPEN_STEPS = 20
_SAPLING_DROP_CHANCE = 0.3

# ----------------------------------------------------------------
# Lighting / torch system (2026-06 port-vs-fork alignment)
# ----------------------------------------------------------------
# The fork masks the local view by a per-floor light_map: tiles with
# light <= 0.05 render as blank darkness (ascii_renderer.py light_mask).
# Underground floors start dark and are lit only around the arrival ladder,
# near lava, and by placed torches. Torches are crafted (MAKE_TORCH: 1 wood +
# 1 coal at a table -> 4) and placed (PLACE_TORCH) to extend a radius-~4 light
# pocket (fork TORCH_LIGHT_MAP = clip(1 - dist/5, 0, 1) over a 9x9).
_LIGHT_VISIBLE_THRESHOLD = 0.05
_TORCH_MAKE_YIELD = 4
_TORCH_FALLOFF_DIVISOR = 5.0   # light = 1 - euclidean_dist / divisor  (radius ~4)
_LAVA_FALLOFF_DIVISOR = 3.0    # lava glows a smaller radius (~2)
# Default ambient light for dungeon floors. 1.0 = fully lit (legacy/back-compat
# behaviour: every focused scenario stays fully visible unless it opts into
# darkness by setting `_dungeon_ambient_light = 0.0`). The faithful full-game
# dungeons are dark (0.0).
_DUNGEON_DARK_AMBIENT = 0.0

# Directional player characters
_DIR_CHARS: dict[tuple[int, int], str] = {
    (1, 0): "\u2192", (-1, 0): "\u2190", (0, -1): "\u2191", (0, 1): "\u2193",  # →←↑↓
}
_DIR_NAMES: dict[tuple[int, int], str] = {
    (1, 0): "right", (-1, 0): "left",
    (0, -1): "up", (0, 1): "down",
}

# Spell names ordered by learn index
_SPELL_NAMES = ("fireball", "iceball", "heal")

# Solid tiles that block projectiles (they cannot pass through walls / trees /
# stone / placed stone / closed boss doors).
_SOLID_TILES: frozenset[str] = frozenset({
    TILE_DUNGEON_WALL, TILE_TREE, TILE_FIRE_TREE, TILE_ICE_SHRUB,
    TILE_STONE, TILE_PLACED_STONE,
    TILE_BOSS_DOOR,
    # Crafting structures and ores also block projectiles.
    TILE_TABLE, TILE_FURNACE, TILE_COAL, TILE_IRON, TILE_DIAMOND,
    # Phase β gem ores block projectiles too.
    TILE_SAPPHIRE, TILE_RUBY,
    # Plants block projectiles too (upstream constants.py:370-371).
    TILE_PLANT, TILE_RIPE_PLANT,
})

# ----------------------------------------------------------------
# World sizes
# ----------------------------------------------------------------
_SURFACE_SIZE = 64
_DUNGEON_SIZE = 32
# Phase γ T13-T15γ: bumped from 5 to 7 for floors 5 (Troll Mines),
# 6 (Fire Realm), 7 (Ice Realm).  Floor 8 (Graveyard) added in T16γ.
# Floors 0-8 = 9 total (0=surface, 1-8=dungeons).
_NUM_DUNGEON_FLOORS = 8

# ----------------------------------------------------------------
# Mob definitions
# ----------------------------------------------------------------
_MOB_STATS: dict[str, dict[str, int]] = {
    # Upstream Craftax constants.py MOB_TYPE_HEALTH_MAPPING and
    # MOB_TYPE_DAMAGE_MAPPING, collapsed to scalar damage for this Python env.
    "zombie": {"hp": 5, "damage": 2},
    "cow": {"hp": 3, "damage": 0},
    "skeleton": {"hp": 3, "damage": 2},
    "kobold": {"hp": 8, "damage": 4},
    "bat": {"hp": 4, "damage": 0},
    "gnome_warrior": {"hp": 9, "damage": 4},
    "gnome_archer": {"hp": 6, "damage": 4},
    "orc_soldier": {"hp": 7, "damage": 3},
    "orc_mage": {"hp": 5, "damage": 3},
    "lizard": {"hp": 11, "damage": 5},
    "knight": {"hp": 12, "damage": 6},
    "knight_archer": {"hp": 12, "damage": 5},
    "troll": {"hp": 20, "damage": 8},
    "deep_thing": {"hp": 4, "damage": 10},
    "snail": {"hp": 6, "damage": 0},
    "pigman": {"hp": 20, "damage": 8},
    "fire_elemental": {"hp": 14, "damage": 8},
    "frost_troll": {"hp": 24, "damage": 9},
    "ice_elemental": {"hp": 16, "damage": 9},
}

_MOB_TILES: dict[str, str] = {
    "zombie": TILE_ZOMBIE,
    "cow": TILE_COW,
    "skeleton": TILE_SKELETON_ARCHER,  # glyph "a" (upstream archer convention)
    "kobold": TILE_KOBOLD,             # glyph "q"
    "bat": TILE_BAT,
    "gnome_warrior": TILE_GNOME_WARRIOR,
    "gnome_archer": TILE_GNOME_ARCHER,
    "orc_soldier": TILE_ORC_SOLDIER,
    "orc_mage": TILE_ORC_MAGE,
    "lizard": TILE_LIZARD,
    "knight": TILE_KNIGHT,
    "knight_archer": TILE_KNIGHT_ARCHER,
    # Phase γ T13-T15γ: new floor mobs.
    "troll": TILE_TROLL,
    "deep_thing": TILE_DEEP_THING,
    "snail": TILE_SNAIL,
    "pigman": TILE_PIGMAN,
    "fire_elemental": TILE_FIRE_ELEMENTAL,
    "frost_troll": TILE_FROST_TROLL,
    "ice_elemental": TILE_ICE_ELEMENTAL,
}

# Boss definitions per floor (1-indexed)
_BOSS_DEFS: dict[int, dict[str, Any]] = {
    1: {"name": "knight", "hp": 10, "damage": 3,
        "ach": "defeat_knight", "ranged": False},
    2: {"name": "archer_boss", "hp": 12, "damage": 4,
        "ach": "defeat_archer_boss", "ranged": True},
    3: {"name": "mage", "hp": 8, "damage": 5,
        "ach": "defeat_mage", "ranged": True},
    4: {"name": "dragon", "hp": 20, "damage": 6,
        "ach": "defeat_dragon", "ranged": False},
    5: {"name": "lich", "hp": 15, "damage": 7,
        "ach": "defeat_lich", "ranged": True},
}

# Weapon damage bonuses. Total melee damage = 1 (bare hand) + bonus, matching
# the fork's per-sword-tier melee table [1,2,3,5,8]:
# none=1, wood=2, stone=3, iron=5, diamond=8 (fork get_player_damage_vector).
_WEAPON_BONUS: dict[str, int] = {
    "wood_sword": 1,
    "stone_sword": 2,
    "iron_sword": 4,
    "diamond_sword": 7,
}

# Armor defense values
_ARMOR_DEFENSE: dict[str, int] = {
    "wood_armor": 1,
    "stone_armor": 2,
    "iron_armor": 3,
    "diamond_armor": 4,
}


class Mob(TypedDict):
    """Mob state dictionary."""

    type: str
    x: int
    y: int
    hp: int
    max_hp: int
    is_boss: bool
    floor: int
    attack_cooldown: int  # ticks remaining before this mob can attack again


class CraftaxFullEnv(_CraftaxTutorialMixin, BaseGlyphEnv):
    """Craftax survival crafting with dungeons and magic.

    93 achievements spanning resource gathering, crafting, combat,
    dungeon exploration, magic, lighting (torches), bosses, and survival
    milestones.

    Surface: 64x64, Dungeons: 32x32 per floor, 9 floors total.
    Visible window: 11x9 centered on agent.
    Reward: each first-time achievement unlock yields +1/N (where N is the
    total achievement count). Cumulative reward sums to 1.0 if every
    achievement is unlocked (defeat_necromancer included). Death overrides
    any in-step progress with a terminal -1.0, so cumulative reward is
    bounded in [-1, 1].
    """

    action_spec = CRAFTAX_FULL_ACTION_SPEC
    noop_action_name = "NOOP"

    _ALL_ACHIEVEMENTS = ALL_FULL_ACHIEVEMENTS
    # Pattern B: when True, _step emits a terminal -1.0 on death. Inherited
    # subclasses (e.g. craftax subtasks) can set False to manage their own
    # death penalty.
    _emit_death_penalty: bool = True

    # Lighting: ambient light for dungeon floors (floor > 0). 1.0 = fully lit
    # (back-compat default — focused scenarios stay fully visible). Set to
    # _DUNGEON_DARK_AMBIENT (0.0) to make dungeons dark so the agent must use
    # the up-ladder glow, lava glow, and placed torches to see (fork-faithful).
    _dungeon_ambient_light: float = 1.0

    # Full game uses every anchor in the canonical tutorial.
    from glyphbench.envs.craftax.docs import ALL_SECTIONS as _FULL_SECTIONS
    tutorial_sections: tuple[str, ...] = _FULL_SECTIONS
    del _FULL_SECTIONS

    def _task_description(self) -> str:
        ach = ", ".join(self._ALL_ACHIEVEMENTS)
        return (
            f"Gather resources, craft tools, fight mobs, explore dungeons, "
            f"learn magic, and survive. Each new achievement gives "
            f"+1/{len(self._ALL_ACHIEVEMENTS)} reward (cumulative sums to 1.0 if all are unlocked).\n"
            f"Death yields a terminal -1.0 reward. "
            f"Achievements ({len(self._ALL_ACHIEVEMENTS)}): {ach}."
        )

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        # World state
        self._floors: dict[int, list[list[str]]] = {}
        self._current_floor: int = 0  # 0 = surface
        self._agent_x: int = 0
        self._agent_y: int = 0
        self._facing: tuple[int, int] = (1, 0)
        # Inventory and equipment
        self._inventory: dict[str, int] = {}
        # Phase gamma T10-T12: element-aware enchantment state.
        # 0=none, 1=fire, 2=ice.
        self._sword_enchantment: int = 0
        self._bow_enchantment: int = 0
        # Phase gamma T03: 4-slot armour state.
        # Per-slot tier: 0=none, 1=iron, 2=diamond.
        # Per-slot enchant element: 0=none, 1=fire, 2=ice.
        self._armor_slots: dict[str, int] = {
            "helmet": 0, "chest": 0, "legs": 0, "boots": 0
        }
        self._armor_enchants: dict[str, int] = {
            "helmet": 0, "chest": 0, "legs": 0, "boots": 0
        }
        # Achievements
        self._achievements_unlocked: set[str] = set()
        # Phase-β parallel bitmap keyed by upstream Achievement enum names.
        self._achievements_phase_beta: dict[str, bool] = {
            n: False for n in UPSTREAM_ACHIEVEMENT_NAMES
        }
        self._message: str = ""
        # Vitals — Phase γ T09γ: per-instance max stats (attribute-scaled).
        # At init, str=dex=int_attr=1, so max values equal the base constants.
        self._max_hp: int = _BASE_MAX_HP
        self._max_food: int = _MAX_FOOD
        self._max_drink: int = _MAX_WATER
        self._max_energy: int = _MAX_ENERGY
        self._max_mana: int = _MAX_MANA
        self._hp: int = self._max_hp
        self._food: int = self._max_food
        self._water: int = self._max_drink
        self._energy: int = self._max_energy
        self._mana: int = self._max_mana
        # Spells learned (per-spell dict; T10β)
        self._learned_spells: dict[str, bool] = {"fireball": False, "iceball": False}
        # Day/night
        self._day_counter: int = 0
        self._day_night: str = "day"
        self._night_count: int = 0
        # Mobs (all floors)
        self._mobs: list[Mob] = []
        # Projectile entities (phase α).
        from glyphbench.envs.craftax.mechanics.projectiles import ProjectileEntity
        self._player_projectiles: list[ProjectileEntity] = []
        self._mob_projectiles: list[ProjectileEntity] = []
        self._is_sleeping: bool = False
        self._is_resting: bool = False
        self._pending_step_reward: float = 0.0
        # Plants
        self._plants: dict[tuple[int, int], int] = {}
        # Potions — phase β: per-color dict + hidden per-episode mapping
        self._potions: list[str] = []  # compatibility field; potions dict is authoritative
        self._potion_mapping: tuple[int, ...] = (0, 1, 2, 3, 4, 5)  # will be set in _reset
        # Active effects
        self._speed_turns: int = 0
        # Milestone counters
        self._total_kills: int = 0
        self._total_crafts: int = 0
        self._total_blocks_placed: int = 0
        self._total_plants_eaten: int = 0
        self._total_water_drunk: int = 0
        # Stairs locations  floor -> (x, y)
        self._stairs_down_pos: dict[int, tuple[int, int]] = {}
        self._stairs_up_pos: dict[int, tuple[int, int]] = {}
        # Boss alive tracking
        self._bosses_alive: dict[int, bool] = {}
        # Chest tracking (T12β): per-floor set of opened (x, y) coords.
        self._chests_opened: dict[int, set[tuple[int, int]]] = {}
        # First-chest gating (T14β): per-floor bool for first-chest bonus.
        self._first_chest_opened: dict[int, bool] = {}
        # Phase γ T06γ: experience points gained by first-floor-entry grants.
        self._xp: int = 0
        # Tracks which floors have already granted XP (first-visit gate).
        self._xp_floors_visited: set[int] = set()
        # Phase γ T07γ: 3 RPG attributes (cap 5 each; initial value 1).
        # Use _int_attr to avoid shadowing Python builtin `int`.
        self._dex: int = 1
        self._str: int = 1
        self._int_attr: int = 1
        # Phase γ T17γ: Necromancer state machine (floor 8).
        # _boss_progress: number of vulnerable hits landed on the necromancer [0..8].
        # _boss_summon_timer: turns remaining before necromancer becomes vulnerable again.
        self._boss_progress: int = 0
        self._boss_summon_timer: int = 0
        # Lighting / torch state (2026-06 port-vs-fork alignment).
        # Placed torch positions per floor; light sources for dark floors.
        self._torches_by_floor: dict[int, list[tuple[int, int]]] = {}
        # Per-floor ambient-light override (else _floor_ambient() default).
        self._floor_ambient_light: dict[int, float] = {}
        # Lazily-cached lava positions per floor (static glow sources).
        self._lava_light_cache: dict[int, list[tuple[int, int]]] = {}

    # ---------------------------------------------------------------
    # Identity
    # ---------------------------------------------------------------

    def env_id(self) -> str:
        return "glyphbench/craftaxfull-v0"

    # system_prompt() inherited from _CraftaxTutorialMixin (uses
    # tutorial_sections + _task_description above).

    # ---------------------------------------------------------------
    # Floor helpers
    # ---------------------------------------------------------------

    def _floor_size(self) -> int:
        if self._current_floor == 0:
            return _SURFACE_SIZE
        return _DUNGEON_SIZE

    def _tile_at(self, x: int, y: int) -> str:
        return self._floors[self._current_floor][y][x]

    def _is_in_bounds(self, x: int, y: int) -> bool:
        size = self._floor_size()
        return 0 <= x < size and 0 <= y < size

    def _current_grid(self) -> list[list[str]]:
        return self._floors[self._current_floor]

    def _walkable_set(self) -> frozenset[str]:
        if self._current_floor == 0:
            return SURFACE_WALKABLE
        return DUNGEON_WALKABLE

    # ---------------------------------------------------------------
    # World generation
    # ---------------------------------------------------------------

    def _generate_surface(self) -> None:
        size = _SURFACE_SIZE
        grid = [
            [TILE_GRASS for _ in range(size)]
            for _ in range(size)
        ]

        # Forests
        for _ in range(int(self.rng.integers(8, 15))):
            cx = int(self.rng.integers(5, size - 5))
            cy = int(self.rng.integers(5, size - 5))
            r = int(self.rng.integers(3, 7))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = cx + dx, cy + dy
                        if (
                            0 <= x < size
                            and 0 <= y < size
                            and self.rng.random() < 0.7
                        ):
                            grid[y][x] = TILE_TREE

        # Stone deposits
        for _ in range(int(self.rng.integers(5, 10))):
            cx = int(self.rng.integers(5, size - 5))
            cy = int(self.rng.integers(5, size - 5))
            r = int(self.rng.integers(2, 4))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = cx + dx, cy + dy
                        if (
                            0 <= x < size
                            and 0 <= y < size
                            and self.rng.random() < 0.6
                        ):
                            grid[y][x] = TILE_STONE

        # Coal
        for _ in range(int(self.rng.integers(3, 6))):
            cx = int(self.rng.integers(10, size - 10))
            cy = int(self.rng.integers(10, size - 10))
            r = int(self.rng.integers(1, 3))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = cx + dx, cy + dy
                        if (
                            0 <= x < size
                            and 0 <= y < size
                            and self.rng.random() < 0.5
                        ):
                            grid[y][x] = TILE_COAL

        # Iron
        for _ in range(int(self.rng.integers(3, 5))):
            cx = int(self.rng.integers(10, size - 10))
            cy = int(self.rng.integers(10, size - 10))
            r = int(self.rng.integers(1, 3))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = cx + dx, cy + dy
                        if (
                            0 <= x < size
                            and 0 <= y < size
                            and self.rng.random() < 0.4
                        ):
                            grid[y][x] = TILE_IRON

        # Water
        for _ in range(int(self.rng.integers(2, 5))):
            cx = int(self.rng.integers(10, size - 10))
            cy = int(self.rng.integers(10, size - 10))
            r = int(self.rng.integers(2, 5))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = cx + dx, cy + dy
                        if (
                            0 <= x < size
                            and 0 <= y < size
                            and self.rng.random() < 0.8
                        ):
                            grid[y][x] = TILE_WATER

        # Sand
        for _ in range(int(self.rng.integers(2, 5))):
            cx = int(self.rng.integers(8, size - 8))
            cy = int(self.rng.integers(8, size - 8))
            r = int(self.rng.integers(2, 4))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = cx + dx, cy + dy
                        if (
                            0 <= x < size
                            and 0 <= y < size
                            and grid[y][x] == TILE_GRASS
                            and self.rng.random() < 0.6
                        ):
                            grid[y][x] = TILE_SAND

        # Lava
        for _ in range(int(self.rng.integers(1, 3))):
            cx = int(self.rng.integers(15, size - 15))
            cy = int(self.rng.integers(15, size - 15))
            r = int(self.rng.integers(1, 3))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = cx + dx, cy + dy
                        if (
                            0 <= x < size
                            and 0 <= y < size
                            and grid[y][x] == TILE_GRASS
                            and self.rng.random() < 0.7
                        ):
                            grid[y][x] = TILE_LAVA

        # Diamond (rare single tiles)
        for _ in range(int(self.rng.integers(1, 3))):
            x = int(self.rng.integers(15, size - 15))
            y = int(self.rng.integers(15, size - 15))
            grid[y][x] = TILE_DIAMOND

        # Clear starting area
        center = size // 2
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                x, y = center + dx, center + dy
                if 0 <= x < size and 0 <= y < size:
                    grid[y][x] = TILE_GRASS

        # Dungeon entrance on surface
        entrance_x = center + 5
        entrance_y = center + 5
        if 0 <= entrance_x < size and 0 <= entrance_y < size:
            grid[entrance_y][entrance_x] = TILE_STAIRS_DOWN
            self._stairs_down_pos[0] = (entrance_x, entrance_y)

        self._floors[0] = grid

    def _generate_dungeon_floor(self, floor: int) -> None:
        """Generate a dungeon floor with rooms and corridors.

        Floors 1 and 3 use the phase-β dungeon-room biome generator
        (``mechanics/world_gen.generate_dungeon_floor``) which places 8
        non-overlapping rooms with chests and fountains.

        Floor 2 uses the original room generator with sapphire/ruby ore; floor
        5 keeps the original room-generator layout.

        Phase γ T13-T15γ: floors 5, 6, 7 use the new smoothgen generator.
        """
        # ---- Phase-β biome generator for floors 1, 3, and 4 (T18β/T19β/T20β) ----
        if floor in (1, 3, 4):
            self._generate_dungeon_floor_biome(floor)
            return

        # ---- Phase-γ smoothgen generator for floors 5-7 (T13-T15γ) ----
        if floor in (5, 6, 7):
            self._generate_dungeon_floor_smoothgen(floor)
            return

        # ---- Phase-γ floor 8: Graveyard with necromancer (T16γ) ----
        if floor == 8:
            self._generate_dungeon_floor_graveyard()
            return

        # ---- Legacy generator for floors 2, 5 ----
        size = _DUNGEON_SIZE
        grid = [
            [TILE_DUNGEON_WALL for _ in range(size)]
            for _ in range(size)
        ]

        # Generate rooms
        rooms: list[tuple[int, int, int, int]] = []
        for _ in range(int(self.rng.integers(5, 9))):
            rw = int(self.rng.integers(4, 8))
            rh = int(self.rng.integers(4, 8))
            rx = int(self.rng.integers(1, size - rw - 1))
            ry = int(self.rng.integers(1, size - rh - 1))
            # Check overlap
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
                    grid[ry + dy][rx + dx] = TILE_DUNGEON_FLOOR

        # Ensure at least 2 rooms
        if len(rooms) < 2:
            rooms = [(2, 2, 6, 6), (20, 20, 6, 6)]
            for rx, ry, rw, rh in rooms:
                for dy in range(rh):
                    for dx in range(rw):
                        if (
                            0 <= ry + dy < size
                            and 0 <= rx + dx < size
                        ):
                            grid[ry + dy][rx + dx] = (
                                TILE_DUNGEON_FLOOR
                            )

        # Connect rooms with corridors
        for i in range(len(rooms) - 1):
            r1 = rooms[i]
            r2 = rooms[i + 1]
            cx1 = r1[0] + r1[2] // 2
            cy1 = r1[1] + r1[3] // 2
            cx2 = r2[0] + r2[2] // 2
            cy2 = r2[1] + r2[3] // 2
            # Horizontal then vertical
            x = cx1
            while x != cx2:
                if 0 <= x < size and 0 <= cy1 < size:
                    grid[cy1][x] = TILE_DUNGEON_FLOOR
                x += 1 if cx2 > cx1 else -1
            if 0 <= cx2 < size and 0 <= cy1 < size:
                grid[cy1][cx2] = TILE_DUNGEON_FLOOR
            y = cy1
            while y != cy2:
                if 0 <= cx2 < size and 0 <= y < size:
                    grid[y][cx2] = TILE_DUNGEON_FLOOR
                y += 1 if cy2 > cy1 else -1
            if 0 <= cx2 < size and 0 <= cy2 < size:
                grid[cy2][cx2] = TILE_DUNGEON_FLOOR

        # Place stairs up in first room
        r0 = rooms[0]
        up_x = r0[0] + 1
        up_y = r0[1] + 1
        grid[up_y][up_x] = TILE_STAIRS_UP
        self._stairs_up_pos[floor] = (up_x, up_y)

        # Place stairs down in last room (except floor 5)
        if floor < _NUM_DUNGEON_FLOORS:
            rl = rooms[-1]
            down_x = rl[0] + rl[2] - 2
            down_y = rl[1] + rl[3] - 2
            if 0 <= down_x < size and 0 <= down_y < size:
                grid[down_y][down_x] = TILE_STAIRS_DOWN
                self._stairs_down_pos[floor] = (
                    down_x, down_y,
                )

        # Place boss in a dedicated room (middle-ish room)
        boss_room_idx = len(rooms) // 2
        br = rooms[boss_room_idx]
        boss_x = br[0] + br[2] // 2
        boss_y = br[1] + br[3] // 2
        # Place boss door at room entrance (just mark it)
        door_x = br[0]
        door_y = br[1] + br[3] // 2
        if (
            0 <= door_x < size
            and 0 <= door_y < size
            and grid[door_y][door_x] != TILE_STAIRS_UP
            and grid[door_y][door_x] != TILE_STAIRS_DOWN
        ):
            grid[door_y][door_x] = TILE_BOSS_DOOR

        # Spawn boss mob
        bdef = _BOSS_DEFS[floor]
        boss_mob: Mob = {
            "type": bdef["name"],
            "x": boss_x,
            "y": boss_y,
            "hp": bdef["hp"],
            "max_hp": bdef["hp"],
            "is_boss": True,
            "floor": floor,
            "attack_cooldown": 0,
        }
        self._mobs.append(boss_mob)
        self._bosses_alive[floor] = True

        # Scatter dungeon-specific resources
        num_res = int(self.rng.integers(3, 6))
        for _ in range(num_res):
            res = self.rng.choice(
                [TILE_STONE, TILE_COAL, TILE_IRON]
            )
            for _att in range(20):
                rx = int(self.rng.integers(1, size - 1))
                ry = int(self.rng.integers(1, size - 1))
                if grid[ry][rx] == TILE_DUNGEON_FLOOR:
                    grid[ry][rx] = res
                    break

        # Phase β (T06β): sapphire and ruby ore placement.
        # Floor 2 (Gnomish Mines): 2.5% each on dungeon-floor tiles.
        # Floors 4-7 deferred to phase γ.
        if floor == 2:
            for ry in range(size):
                for rx in range(size):
                    if grid[ry][rx] == TILE_DUNGEON_FLOOR:
                        roll = self.rng.random()
                        if roll < 0.025:
                            grid[ry][rx] = TILE_SAPPHIRE
                        elif roll < 0.05:
                            grid[ry][rx] = TILE_RUBY

        # Spawn dungeon mobs from the upstream per-floor class mapping.
        from glyphbench.envs.craftax.mechanics.mobs import FLOOR_MOB_MAPPING

        mapping = FLOOR_MOB_MAPPING[floor]
        mob_types = list(dict.fromkeys(
            (mapping["passive"], mapping["melee"], mapping["ranged"])
        ))
        num_mobs = floor + 2
        for _ in range(num_mobs):
            mtype = str(self.rng.choice(mob_types))
            if mtype not in _MOB_STATS:
                continue
            for _att in range(30):
                mx = int(self.rng.integers(1, size - 1))
                my = int(self.rng.integers(1, size - 1))
                if (
                    grid[my][mx] == TILE_DUNGEON_FLOOR
                    and not self._mob_at(mx, my, floor)
                ):
                    stats = _MOB_STATS[mtype]
                    mob: Mob = {
                        "type": mtype,
                        "x": mx,
                        "y": my,
                        "hp": stats["hp"],
                        "max_hp": stats["hp"],
                        "is_boss": False,
                        "floor": floor,
                        "attack_cooldown": 0,
                    }
                    self._mobs.append(mob)
                    break

        self._floors[floor] = grid

    def _generate_dungeon_floor_biome(self, floor: int) -> None:
        """Generate floors 1, 3, and 4 using the phase-β dungeon-room biome generator.

        Uses ``mechanics.world_gen.generate_dungeon_floor`` to create a grid
        with 8 rooms, L-shaped corridors, 1 chest per room, and ~50% fountain
        probability per room.  Boss, mob spawning, and resource scattering are
        then layered on top of the generated grid.

        Floor 3 (Sewers) receives 1 ``TILE_ENCHANT_ICE`` tile and floor 4
        (Vaults) receives 1 ``TILE_ENCHANT_FIRE`` tile, each placed on the
        first available dungeon-floor cell in row-major order (deterministic
        for a given seed).  Phase γ will wire the enchantment-table
        interaction semantics; this task only places the tiles.
        """
        from glyphbench.envs.craftax.mechanics.world_gen import generate_dungeon_floor

        size = _DUNGEON_SIZE

        (
            grid,
            _chest_positions,
            _fountain_positions,
            stairs_down_pos,
            stairs_up_pos,
            _agent_spawn,
        ) = generate_dungeon_floor(
            self.rng,
            size,
            num_rooms=8,
            with_chests=True,
            with_fountains=True,
        )

        # Wire stair positions into env state.
        self._stairs_up_pos[floor] = stairs_up_pos
        if floor < _NUM_DUNGEON_FLOORS:
            self._stairs_down_pos[floor] = stairs_down_pos

        # Place boss in a random interior floor cell (avoid stair positions).
        skip_positions = {stairs_up_pos, stairs_down_pos}
        boss_x, boss_y = size // 2, size // 2  # fallback
        bdef = _BOSS_DEFS[floor]
        for _att in range(50):
            bx = int(self.rng.integers(size // 4, 3 * size // 4))
            by = int(self.rng.integers(size // 4, 3 * size // 4))
            if (
                grid[by][bx] == TILE_DUNGEON_FLOOR
                and (bx, by) not in skip_positions
            ):
                boss_x, boss_y = bx, by
                break

        # Place boss door adjacent to boss (one cell to the left).
        door_x = boss_x - 1
        door_y = boss_y
        if (
            0 <= door_x < size
            and grid[door_y][door_x] == TILE_DUNGEON_FLOOR
            and (door_x, door_y) not in skip_positions
        ):
            grid[door_y][door_x] = TILE_BOSS_DOOR

        boss_mob: Mob = {
            "type": bdef["name"],
            "x": boss_x,
            "y": boss_y,
            "hp": bdef["hp"],
            "max_hp": bdef["hp"],
            "is_boss": True,
            "floor": floor,
            "attack_cooldown": 0,
        }
        self._mobs.append(boss_mob)
        self._bosses_alive[floor] = True

        # Scatter dungeon-specific resources on floor cells.
        num_res = int(self.rng.integers(3, 6))
        for _ in range(num_res):
            res = str(self.rng.choice([TILE_STONE, TILE_COAL, TILE_IRON]))
            for _att in range(20):
                rx = int(self.rng.integers(1, size - 1))
                ry = int(self.rng.integers(1, size - 1))
                if grid[ry][rx] == TILE_DUNGEON_FLOOR:
                    grid[ry][rx] = res
                    break

        # T20β fixup: Floor 3 (Sewers) — place 1 TILE_ENCHANT_ICE tile.
        # T20β:        Floor 4 (Vaults) — place 1 TILE_ENCHANT_FIRE tile.
        # We scan the grid in row-major order and place it on the first
        # TILE_DUNGEON_FLOOR cell that is not a stair or reserved position.
        # This gives a deterministic location for a given seed.
        if floor == 3:
            _reserved = {stairs_up_pos, stairs_down_pos}
            _placed_enchant = False
            for _ey in range(size):
                if _placed_enchant:
                    break
                for _ex in range(size):
                    if (
                        grid[_ey][_ex] == TILE_DUNGEON_FLOOR
                        and (_ex, _ey) not in _reserved
                    ):
                        grid[_ey][_ex] = TILE_ENCHANT_ICE
                        _placed_enchant = True
                        break

        if floor == 4:
            _reserved = {stairs_up_pos, stairs_down_pos}
            _placed_enchant = False
            for _ey in range(size):
                if _placed_enchant:
                    break
                for _ex in range(size):
                    if (
                        grid[_ey][_ex] == TILE_DUNGEON_FLOOR
                        and (_ex, _ey) not in _reserved
                    ):
                        grid[_ey][_ex] = TILE_ENCHANT_FIRE
                        _placed_enchant = True
                        break

        # Spawn dungeon mobs from the upstream per-floor class mapping.
        from glyphbench.envs.craftax.mechanics.mobs import FLOOR_MOB_MAPPING

        mapping = FLOOR_MOB_MAPPING[floor]
        mob_types = list(dict.fromkeys(
            (mapping["passive"], mapping["melee"], mapping["ranged"])
        ))
        num_mobs = floor + 2
        for _ in range(num_mobs):
            mtype = str(self.rng.choice(mob_types))
            if mtype not in _MOB_STATS:
                continue
            for _att in range(30):
                mx = int(self.rng.integers(1, size - 1))
                my = int(self.rng.integers(1, size - 1))
                if (
                    grid[my][mx] == TILE_DUNGEON_FLOOR
                    and not self._mob_at(mx, my, floor)
                ):
                    stats = _MOB_STATS[mtype]
                    mob: Mob = {
                        "type": mtype,
                        "x": mx,
                        "y": my,
                        "hp": stats["hp"],
                        "max_hp": stats["hp"],
                        "is_boss": False,
                        "floor": floor,
                        "attack_cooldown": 0,
                    }
                    self._mobs.append(mob)
                    break

        self._floors[floor] = grid

    def _generate_dungeon_floor_smoothgen(self, floor: int) -> None:
        """Generate floors 5-7 using a smoothgen open-area biome.

        Unlike the room-based biome generator used for floors 1/3/4, these
        floors are open-area maps filled with the biome's primary tile, with
        scattered decorative/resource/ore tiles placed using a circular
        scattering pass.

        Floor 5 — Troll Mines: DUNGEON_FLOOR base, sapphire + ruby ore (1%),
          decorative TILE_COAL/IRON patches, troll + deep_thing mobs.
        Floor 6 — Fire Realm: DUNGEON_FLOOR base, LAVA (7% scatter),
          FIRE_TREE (~3%), RUBY ore (2.5%), pigman + fire_elemental mobs.
        Floor 7 — Ice Realm: DUNGEON_FLOOR base, WATER (7% scatter),
          ICE_SHRUB (~3%), SAPPHIRE ore (2%), frost_troll + ice_elemental mobs.
          Light baseline 0.0.

        All three floors:
        - Stair-up placed in top-left quadrant.
        - Stair-down placed in bottom-right quadrant (floor 7 now has stair-down
          to floor 8; added T16γ).
        - Boss from _BOSS_DEFS[floor] spawned only on floor 5 (lich).
          The necromancer boss lives on floor 8 and is NOT a mob entity — it is
          a tile (TILE_NECROMANCER) managed by the boss.py state machine (T17γ).
        - Per-floor mob roster from FLOOR_MOB_MAPPING.
        """
        from glyphbench.envs.craftax.mechanics.mobs import FLOOR_MOB_MAPPING

        size = _DUNGEON_SIZE

        # --- Base grid: all DUNGEON_FLOOR (open area) ---
        grid = [
            [TILE_DUNGEON_FLOOR for _ in range(size)]
            for _ in range(size)
        ]

        # --- Thin wall border ---
        for x in range(size):
            grid[0][x] = TILE_DUNGEON_WALL
            grid[size - 1][x] = TILE_DUNGEON_WALL
        for y in range(size):
            grid[y][0] = TILE_DUNGEON_WALL
            grid[y][size - 1] = TILE_DUNGEON_WALL

        # --- Scatter biome tiles ---
        if floor == 5:
            # Troll Mines: some coal/iron patches for atmosphere.
            for _ in range(int(self.rng.integers(3, 6))):
                cx = int(self.rng.integers(2, size - 2))
                cy = int(self.rng.integers(2, size - 2))
                patch = str(self.rng.choice([TILE_COAL, TILE_IRON]))
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        x, y = cx + dx, cy + dy
                        if 1 <= x < size - 1 and 1 <= y < size - 1 and self.rng.random() < 0.5:
                            grid[y][x] = patch

        elif floor == 6:
            # Fire Realm: large lava patches (~7% of cells).
            for _ in range(int(self.rng.integers(6, 10))):
                cx = int(self.rng.integers(3, size - 3))
                cy = int(self.rng.integers(3, size - 3))
                r = int(self.rng.integers(2, 4))
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if dx * dx + dy * dy <= r * r:
                            x, y = cx + dx, cy + dy
                            if 1 <= x < size - 1 and 1 <= y < size - 1 and self.rng.random() < 0.75:
                                grid[y][x] = TILE_LAVA

            # Fire trees (~3%) are mineable for wood.
            for y in range(1, size - 1):
                for x in range(1, size - 1):
                    if grid[y][x] == TILE_DUNGEON_FLOOR and self.rng.random() < 0.03:
                        grid[y][x] = TILE_FIRE_TREE

        elif floor == 7:
            # Ice Realm: water patches (frozen lakes, ~7%).
            for _ in range(int(self.rng.integers(6, 10))):
                cx = int(self.rng.integers(3, size - 3))
                cy = int(self.rng.integers(3, size - 3))
                r = int(self.rng.integers(2, 4))
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if dx * dx + dy * dy <= r * r:
                            x, y = cx + dx, cy + dy
                            if 1 <= x < size - 1 and 1 <= y < size - 1 and self.rng.random() < 0.75:
                                grid[y][x] = TILE_WATER

            # Ice shrubs (~3%) are mineable for wood.
            for y in range(1, size - 1):
                for x in range(1, size - 1):
                    if grid[y][x] == TILE_DUNGEON_FLOOR and self.rng.random() < 0.03:
                        grid[y][x] = TILE_ICE_SHRUB

        # --- Ore placement ---
        if floor == 5:
            # Troll Mines: sapphire 1%, ruby 1%.
            for y in range(1, size - 1):
                for x in range(1, size - 1):
                    if grid[y][x] == TILE_DUNGEON_FLOOR:
                        roll = self.rng.random()
                        if roll < 0.01:
                            grid[y][x] = TILE_SAPPHIRE
                        elif roll < 0.02:
                            grid[y][x] = TILE_RUBY

        elif floor == 6:
            # Fire Realm: ruby 2.5%.
            for y in range(1, size - 1):
                for x in range(1, size - 1):
                    if grid[y][x] == TILE_DUNGEON_FLOOR and self.rng.random() < 0.025:
                        grid[y][x] = TILE_RUBY

        elif floor == 7:
            # Ice Realm: sapphire 2%.
            for y in range(1, size - 1):
                for x in range(1, size - 1):
                    if grid[y][x] == TILE_DUNGEON_FLOOR and self.rng.random() < 0.02:
                        grid[y][x] = TILE_SAPPHIRE

        # --- Stairs up (top-left quadrant) ---
        up_x = int(self.rng.integers(2, size // 3))
        up_y = int(self.rng.integers(2, size // 3))
        grid[up_y][up_x] = TILE_STAIRS_UP
        self._stairs_up_pos[floor] = (up_x, up_y)

        # --- Stairs down (bottom-right quadrant) ---
        # Floor 7 now has stair-down (to floor 8 Graveyard, added T16γ).
        # Floor 8 itself has no stair-down (it is the terminal floor).
        if floor < _NUM_DUNGEON_FLOORS:
            down_x = int(self.rng.integers(2 * size // 3, size - 2))
            down_y = int(self.rng.integers(2 * size // 3, size - 2))
            grid[down_y][down_x] = TILE_STAIRS_DOWN
            self._stairs_down_pos[floor] = (down_x, down_y)

        # --- Boss placement (floor 5 only; T16γ moves boss to floor 8) ---
        if floor == 5 and floor in _BOSS_DEFS:
            bdef = _BOSS_DEFS[floor]
            # Place boss in centre area, avoiding stairs.
            skip = {self._stairs_up_pos.get(floor), self._stairs_down_pos.get(floor)}
            boss_x, boss_y = size // 2, size // 2
            for _att in range(50):
                bx = int(self.rng.integers(size // 4, 3 * size // 4))
                by = int(self.rng.integers(size // 4, 3 * size // 4))
                if (bx, by) not in skip:
                    boss_x, boss_y = bx, by
                    break
            # Boss door one cell to the left.
            door_x = boss_x - 1
            door_y = boss_y
            if 1 <= door_x < size - 1:
                grid[door_y][door_x] = TILE_BOSS_DOOR
            boss_mob: Mob = {
                "type": bdef["name"],
                "x": boss_x,
                "y": boss_y,
                "hp": bdef["hp"],
                "max_hp": bdef["hp"],
                "is_boss": True,
                "floor": floor,
                "attack_cooldown": 0,
            }
            self._mobs.append(boss_mob)
            self._bosses_alive[floor] = True

        # --- Spawn non-boss mobs from per-floor roster ---
        mapping = FLOOR_MOB_MAPPING[floor]
        floor_mob_types = list(dict.fromkeys(
            (mapping["passive"], mapping["melee"], mapping["ranged"])
        ))
        num_mobs = floor + 2
        for _ in range(num_mobs):
            mtype = str(self.rng.choice(floor_mob_types))
            if mtype not in _MOB_STATS:
                continue  # guard against roster mismatches
            for _att in range(30):
                mx = int(self.rng.integers(1, size - 1))
                my = int(self.rng.integers(1, size - 1))
                if (
                    grid[my][mx] == TILE_DUNGEON_FLOOR
                    and not self._mob_at(mx, my, floor)
                ):
                    stats = _MOB_STATS[mtype]
                    spawn_mob: Mob = {
                        "type": mtype,
                        "x": mx,
                        "y": my,
                        "hp": stats["hp"],
                        "max_hp": stats["hp"],
                        "is_boss": False,
                        "floor": floor,
                        "attack_cooldown": 0,
                    }
                    self._mobs.append(spawn_mob)
                    break

        self._floors[floor] = grid

    def _generate_dungeon_floor_graveyard(self) -> None:
        """Generate floor 8 — the Graveyard — where the necromancer boss lives.

        Layout:
        - DUNGEON_FLOOR base tile (open area).
        - Thin wall border around the perimeter.
        - 10-20 GRAVE (⚰) tombstone decorations scattered randomly.
        - NECROMANCER tile (N) placed at the centre of the map.
        - NO STAIRS_UP or STAIRS_DOWN — floor 8 is the terminal dungeon floor.
        - No pre-spawned regular mobs; the necromancer's wave-summoning
          system (T17γ) handles mob creation during the boss fight.

        The necromancer is represented as a static tile (TILE_NECROMANCER or
        TILE_NECROMANCER_VULNERABLE), not a mob entity. The boss.py state
        machine manages the fight logic.
        """
        floor = 8
        size = _DUNGEON_SIZE

        # --- Base grid: all DUNGEON_FLOOR ---
        grid = [
            [TILE_DUNGEON_FLOOR for _ in range(size)]
            for _ in range(size)
        ]

        # --- Thin wall border ---
        for x in range(size):
            grid[0][x] = TILE_DUNGEON_WALL
            grid[size - 1][x] = TILE_DUNGEON_WALL
        for y in range(size):
            grid[y][0] = TILE_DUNGEON_WALL
            grid[y][size - 1] = TILE_DUNGEON_WALL

        # --- Scatter 10-20 grave decorations ---
        num_graves = int(self.rng.integers(10, 21))
        for _ in range(num_graves):
            for _att in range(20):
                gx = int(self.rng.integers(2, size - 2))
                gy = int(self.rng.integers(2, size - 2))
                if grid[gy][gx] == TILE_DUNGEON_FLOOR:
                    grid[gy][gx] = TILE_GRAVE
                    break

        # --- Place necromancer at centre ---
        cx = size // 2
        cy = size // 2
        # Ensure centre is not a wall (it shouldn't be, but guard against
        # edge cases in very small maps).
        if grid[cy][cx] in (TILE_DUNGEON_WALL,):
            cx = size // 2 + 1
        grid[cy][cx] = TILE_NECROMANCER
        # Record necromancer position for DO targeting.
        self._necromancer_pos: tuple[int, int] = (cx, cy)

        # --- NO STAIRS (floor 8 is the end) ---
        # self._stairs_up_pos[8] and self._stairs_down_pos[8] are
        # intentionally NOT set.

        self._floors[floor] = grid

    # ---------------------------------------------------------------
    # Cow spawning
    # ---------------------------------------------------------------

    def _spawn_initial_cows(self) -> None:
        num_cows = int(self.rng.integers(3, 6))
        size = _SURFACE_SIZE
        for _ in range(num_cows):
            for _att in range(50):
                x = int(self.rng.integers(5, size - 5))
                y = int(self.rng.integers(5, size - 5))
                if (
                    self._floors[0][y][x] == TILE_GRASS
                    and not self._mob_at(x, y, 0)
                ):
                    mob: Mob = {
                        "type": "cow",
                        "x": x, "y": y,
                        "hp": _MOB_STATS["cow"]["hp"],
                        "max_hp": _MOB_STATS["cow"]["hp"],
                        "is_boss": False,
                        "floor": 0,
                        "attack_cooldown": 0,
                    }
                    self._mobs.append(mob)
                    break

    # ---------------------------------------------------------------
    # Necromancer boss helpers (T17γ)
    # ---------------------------------------------------------------

    def _spawn_necromancer_wave(self) -> None:
        """Spawn 1 zombie (melee) + 1 skeleton (ranged) near the player on floor 8.

        Respects caps: at most 3 alive melee and 2 alive ranged per floor-8.
        Tries to place each mob on a walkable DUNGEON_FLOOR tile within 6
        tiles (Chebyshev) of the player. If no valid tile is found in 30
        attempts, that mob is skipped.

        Per upstream wave semantics: mob types are taken from
        FLOOR_MOB_MAPPING[8] (zombie + skeleton), which intensify with each
        necromancer hit at phase-γ fidelity (a simplification of the upstream
        progress-indexed roster, which is wired in T22γ).
        """
        floor = 8
        size = _DUNGEON_SIZE
        grid = self._floors.get(floor)
        if grid is None:
            return
        px, py = self._agent_x, self._agent_y
        radius = 6
        # Count current floor-8 melee and ranged mobs.
        floor8_melee = [
            m for m in self._mobs
            if m["floor"] == floor and m["type"] == "zombie" and m["hp"] > 0
        ]
        floor8_ranged = [
            m for m in self._mobs
            if m["floor"] == floor and m["type"] == "skeleton" and m["hp"] > 0
        ]
        spawn_specs = []
        if len(floor8_melee) < 3:
            spawn_specs.append("zombie")
        if len(floor8_ranged) < 2:
            spawn_specs.append("skeleton")
        for mtype in spawn_specs:
            stats = _MOB_STATS[mtype]
            for _att in range(30):
                dx = int(self.rng.integers(-radius, radius + 1))
                dy = int(self.rng.integers(-radius, radius + 1))
                mx, my = px + dx, py + dy
                if not (1 <= mx < size - 1 and 1 <= my < size - 1):
                    continue
                tile = grid[my][mx]
                if tile not in (TILE_DUNGEON_FLOOR,):
                    continue
                if self._mob_at(mx, my, floor):
                    continue
                wave_mob: Mob = {
                    "type": mtype,
                    "x": mx,
                    "y": my,
                    "hp": stats["hp"],
                    "max_hp": stats["hp"],
                    "is_boss": False,
                    "floor": floor,
                    "attack_cooldown": 0,
                }
                self._mobs.append(wave_mob)
                break

    def _update_necromancer_tile_glyph(self) -> None:
        """Flip the necromancer tile between NECROMANCER and NECROMANCER_VULNERABLE.

        The rendered observation shows NECROMANCER_VULNERABLE when the boss is
        hittable; NECROMANCER when invulnerable (summoning / timer active).
        This gives the agent explicit visual feedback about when to attack.
        """
        from glyphbench.envs.craftax.mechanics.boss import is_necromancer_vulnerable
        floor = 8
        grid = self._floors.get(floor)
        if grid is None:
            return
        # Find the necromancer tile on the grid.
        size = _DUNGEON_SIZE
        target_glyph = (
            TILE_NECROMANCER_VULNERABLE
            if is_necromancer_vulnerable(self)
            else TILE_NECROMANCER
        )
        for y in range(size):
            for x in range(size):
                if grid[y][x] in (TILE_NECROMANCER, TILE_NECROMANCER_VULNERABLE):
                    grid[y][x] = target_glyph
                    return

    # ---------------------------------------------------------------
    # Adjacency helpers
    # ---------------------------------------------------------------

    def _near_table(self) -> bool:
        grid = self._current_grid()
        fsize = self._floor_size()
        for dx, dy in _CLOSE_BLOCKS:
            nx, ny = self._agent_x + dx, self._agent_y + dy
            if (
                0 <= nx < fsize
                and 0 <= ny < fsize
                and grid[ny][nx] == TILE_TABLE
            ):
                return True
        return False

    def _near_furnace(self) -> bool:
        grid = self._current_grid()
        fsize = self._floor_size()
        for dx, dy in _CLOSE_BLOCKS:
            nx, ny = self._agent_x + dx, self._agent_y + dy
            if (
                0 <= nx < fsize
                and 0 <= ny < fsize
                and grid[ny][nx] == TILE_FURNACE
            ):
                return True
        return False

    def _near_enchant_fire(self) -> bool:
        """Return True iff the agent faces a TILE_ENCHANT_FIRE tile."""
        grid = self._current_grid()
        fsize = self._floor_size()
        nx = self._agent_x + self._facing[0]
        ny = self._agent_y + self._facing[1]
        return (
            0 <= nx < fsize
            and 0 <= ny < fsize
            and grid[ny][nx] == TILE_ENCHANT_FIRE
        )

    def _near_enchant_ice(self) -> bool:
        """Return True iff the agent faces a TILE_ENCHANT_ICE tile."""
        grid = self._current_grid()
        fsize = self._floor_size()
        nx = self._agent_x + self._facing[0]
        ny = self._agent_y + self._facing[1]
        return (
            0 <= nx < fsize
            and 0 <= ny < fsize
            and grid[ny][nx] == TILE_ENCHANT_ICE
        )

    def _pick_enchant_element(self) -> int:
        """Return 1 (fire) or 2 (ice) based on faced table and gem availability.

        Returns 0 if the faced enchant table lacks the matching gem.
        """
        near_fire = self._near_enchant_fire()
        near_ice = self._near_enchant_ice()
        has_ruby = self._inventory.get("ruby", 0) >= 1
        has_sapphire = self._inventory.get("sapphire", 0) >= 1
        if near_fire and has_ruby:
            return 1
        if near_ice and has_sapphire:
            return 2
        return 0

    # ---------------------------------------------------------------
    # Achievement system
    # ---------------------------------------------------------------

    def _try_unlock(self, name: str) -> float:
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
            pretty = name.replace("_", " ").title()
            self._message = f"ACHIEVEMENT: {pretty}!"
            # Phase γ T21γ: also flip the upstream-faithful bitmap if this
            # achievement has an UPSTREAM_ACHIEVEMENT_NAMES entry.
            # Special-case: upstream spells "ARMOUR" not "ARMOR".
            upstream_key = name.upper()
            upstream_key = _BITMAP_ALIAS.get(upstream_key, upstream_key)
            if upstream_key in self._achievements_phase_beta:
                self._achievements_phase_beta[upstream_key] = True
            # Pattern B: each achievement = 1/N_ACHIEVEMENTS so cumulative
            # progress reward sums to 1.0 if all achievements are unlocked.
            return 1.0 / len(self._ALL_ACHIEVEMENTS)
        return 0.0

    # ---------------------------------------------------------------
    # Mob helpers
    # ---------------------------------------------------------------

    def _mob_at(
        self, x: int, y: int, floor: int | None = None,
    ) -> Mob | None:
        fl = floor if floor is not None else self._current_floor
        for mob in self._mobs:
            if (
                mob["x"] == x
                and mob["y"] == y
                and mob["floor"] == fl
            ):
                return mob
        return None

    def _best_weapon_bonus(self) -> int:
        best = 0
        for weapon, bonus in _WEAPON_BONUS.items():
            if self._inventory.get(weapon, 0) > 0 and bonus > best:
                best = bonus
        return best

    def _best_armor_defense(self) -> int:
        """Return whether any armor slot is equipped as a scalar count.

        Phase γ T03γ: derived from per-slot tier dict.  Returns 0 iff all slots
        are tier 0 (no armour equipped).  T04γ replaces the actual damage
        reduction with the multiplicative per-element formula; this method is
        kept only for the no-armor achievement gate.
        """
        return sum(1 for t in self._armor_slots.values() if t > 0)

    def _take_damage(self, raw: int, damage_vec=None) -> None:
        """Take damage. Accepts a scalar physical value or a 3-vector.

        Phase γ T04γ: 3-vector damage with per-slot armour + per-slot enchant
        reduction. Scalar callers (lava, potions, etc.) are treated as pure
        physical damage.

        Sleep multiplier: phase α 3.5× still applies after armour reduction.
        """
        from glyphbench.envs.craftax.mechanics.boss import NECROMANCER_FLOOR
        from glyphbench.envs.craftax.mechanics.damage import damage_dealt_to_player
        if damage_vec is None:
            # Scalar path: treat as physical damage.
            damage_vec = (float(raw), 0.0, 0.0)
        actual = damage_dealt_to_player(self._armor_slots, self._armor_enchants, damage_vec)
        # Phase α: 3.5× damage multiplier while sleeping (upstream
        # game_logic.py:1100-1291). Applied after armour reduction.
        if self._is_sleeping:
            actual = int(round(actual * 3.5))
        # Upstream boss extra damage is tied to the floor-8 necromancer fight,
        # not every per-floor boss mob.
        if self._current_floor == NECROMANCER_FLOOR and self._is_in_boss_fight():
            actual = int(round(actual * 1.5))
        actual = max(0, actual)
        self._hp = max(0, self._hp - actual)
        # Phase β: damage cancels REST and SLEEP states.
        self._is_resting = False
        self._is_sleeping = False

    def _attack_mob(self, mob: Mob) -> float:
        from glyphbench.envs.craftax.mechanics.mobs import damage_dealt_to_mob
        from glyphbench.envs.craftax.mechanics.progression import damage_scale_phys
        base_damage = 1 + self._best_weapon_bonus()
        phys_damage = base_damage * damage_scale_phys(self._str)
        # Phase γ T10γ: sword enchantment adds 0.5× phys_damage as elemental.
        if self._sword_enchantment == 1:
            dvec = (phys_damage, 0.5 * phys_damage, 0.0)
        elif self._sword_enchantment == 2:
            dvec = (phys_damage, 0.0, 0.5 * phys_damage)
        else:
            dvec = (phys_damage, 0.0, 0.0)
        damage = damage_dealt_to_mob(mob["type"], dvec)
        mob["hp"] -= damage
        reward = 0.0
        if mob["hp"] <= 0:
            self._mobs.remove(mob)
            self._total_kills += 1
            reward += self._check_kill_milestones()
            mtype = mob["type"]
            if mob["is_boss"]:
                fl = mob["floor"]
                self._bosses_alive[fl] = False
                bdef = _BOSS_DEFS.get(fl)
                if bdef:
                    self._message = (
                        f"Defeated {bdef['name']}!"
                    )
                    reward += self._try_unlock(bdef["ach"])
                    # Drop boss loot
                    loot_key = f"boss_loot_{fl}"
                    self._inventory[loot_key] = (
                        self._inventory.get(loot_key, 0) + 1
                    )
                    ach = f"collect_boss_loot_{fl}"
                    reward += self._try_unlock(ach)
                    # No-armor boss kill
                    if self._best_armor_defense() == 0:
                        reward += self._try_unlock(
                            "defeat_boss_no_armor"
                        )
            elif mtype == "zombie":
                self._message = "Defeated a zombie!"
                reward += self._try_unlock("defeat_zombie")
            elif mtype == "skeleton":
                self._message = "Defeated a skeleton!"
                reward += self._try_unlock("defeat_skeleton")
            elif mtype == "cow":
                self._food = min(self._max_food, self._food + 6)
                self._message = (
                    "Defeated a cow! Ate beef. (+6 food)"
                )
                reward += self._try_unlock("eat_cow")
            elif mtype == "kobold":
                self._message = "Defeated a kobold!"
                reward += self._try_unlock("defeat_kobold")
            elif mtype == "bat":
                self._food = min(self._max_food, self._food + 6)
                self._message = "Defeated a bat! Ate meat. (+6 food)"
                reward += self._try_unlock("eat_bat")
        else:
            self._message = (
                f"Hit {mob['type']}! "
                f"({mob['hp']}/{mob['max_hp']} HP)"
            )
        return reward

    def _check_kill_milestones(self) -> float:
        reward = 0.0
        if self._total_kills >= 5:
            reward += self._try_unlock("kill_5_mobs")
        if self._total_kills >= 10:
            reward += self._try_unlock("kill_10_mobs")
        if self._total_kills >= 25:
            reward += self._try_unlock("kill_25_mobs")
        if self._total_kills >= 50:
            reward += self._try_unlock("kill_50_mobs")
        return reward

    def _check_exploration_milestones(self) -> float:
        r = 0.0
        # All floors visited
        all_floor_achs = {
            "enter_dungeon", "reach_floor_2",
            "reach_floor_3", "reach_floor_4",
            "reach_floor_5",
        }
        if all_floor_achs.issubset(
            self._achievements_unlocked
        ):
            r += self._try_unlock("explore_all_floors")
        # All boss loot collected
        all_loot = all(
            f"collect_boss_loot_{i}"
            in self._achievements_unlocked
            for i in range(1, 6)
        )
        if all_loot:
            r += self._try_unlock("collect_all_boss_loot")
        # Spells learned — all spells known
        if all(self._learned_spells.values()):
            r += self._try_unlock("learn_all_spells")
        # Clear dungeon floor (no mobs on current floor)
        if self._current_floor > 0:
            floor_mobs = [
                m for m in self._mobs
                if m["floor"] == self._current_floor
            ]
            if len(floor_mobs) == 0:
                r += self._try_unlock(
                    "clear_dungeon_floor"
                )
        return r

    def _is_in_boss_fight(self) -> bool:
        """True iff the agent is on a floor with an active boss encounter.

        For floors 1-5: a boss mob (is_boss=True) with HP > 0 on this floor.
        For floor 8 (Graveyard): the necromancer tile is still alive
          (boss_progress < win threshold); the necromancer is a tile, not a mob.
        """
        from glyphbench.envs.craftax.mechanics.boss import (
            BOSS_PROGRESS_WIN_THRESHOLD,
            NECROMANCER_FLOOR,
        )
        if self._current_floor == NECROMANCER_FLOOR:
            return self._boss_progress < BOSS_PROGRESS_WIN_THRESHOLD
        return any(
            m["is_boss"] and m["floor"] == self._current_floor and m["hp"] > 0
            for m in self._mobs
        )

    def _step_player_projectiles(self) -> None:
        """Advance live player projectiles one tile and resolve hits.

        - Map bounds: drop projectiles that leave the floor.
        - Solid-block collision: projectiles dying on solid tiles do not damage.
        - Mob hit: damage the first mob the projectile lands on; projectile dies.
        Mirrors upstream _move_player_projectile:1719-1823 (post-advance check
        only; pre-advance dual-position check is deferred to phase γ).

        blocked_fn and hit_fn both receive the ProjectileEntity directly.
        This avoids the stale next()-lookup bug where two projectiles advancing
        to the same tile would both resolve to the first one in the list.
        pending_kills is a local variable (not an instance attribute) so a raise
        inside _attack_mob_kill cannot leave a dangling attribute.
        """
        if not self._player_projectiles:
            return
        from glyphbench.envs.craftax.mechanics.projectiles import (
            step_player_projectiles,
        )

        size = self._floor_size()
        pending_kills: list = []  # local, no instance attr

        def _blocked(p) -> bool:
            return self._tile_at(p.x, p.y) in _SOLID_TILES

        def _hit(p) -> bool:
            from glyphbench.envs.craftax.mechanics.mobs import damage_dealt_to_mob
            for mob in self._mobs:
                if (
                    mob["floor"] == self._current_floor
                    and mob["x"] == p.x
                    and mob["y"] == p.y
                    and mob["hp"] > 0
                ):
                    if p.damage_vec is not None:
                        # Phase γ T12γ: use 3-vec damage with mob elemental resistance.
                        effective = damage_dealt_to_mob(mob["type"], p.damage_vec)
                    else:
                        # Legacy scalar: treat as pure physical.
                        effective = damage_dealt_to_mob(
                            mob["type"], (float(p.damage), 0.0, 0.0)
                        )
                    mob["hp"] -= effective
                    if mob["hp"] <= 0:
                        pending_kills.append(mob)
                    return True
            return False

        self._player_projectiles = step_player_projectiles(
            self._player_projectiles,
            map_w=size, map_h=size,
            blocked_fn=_blocked, hit_fn=_hit,
        )

        # Resolve any pending kills (fire achievement, accumulate reward).
        # _attack_mob_kill removes the mob from self._mobs itself.
        for dead_mob in pending_kills:
            self._pending_step_reward += self._attack_mob_kill(dead_mob)

    def _step_mob_projectiles(self) -> None:
        """Advance live mob projectiles; damage player on hit; destroy
        furnace/crafting_table on impact; cancel sleep when hit.

        Symmetric to _step_player_projectiles (T12). The order of the
        per-step driver matters: T12 (player projectiles) runs BEFORE
        _mob_ai; T25 (mob projectiles) runs AFTER _mob_ai so projectiles
        spawned by ranged mobs in this tick travel immediately.
        """
        if not self._mob_projectiles:
            return
        from glyphbench.envs.craftax.mechanics.projectiles import (
            step_mob_projectiles,
        )
        size = self._floor_size()

        def _blocked(p) -> bool:
            return self._tile_at(p.x, p.y) in _SOLID_TILES

        def _block_destroy(p) -> bool:
            tile = self._tile_at(p.x, p.y)
            if tile in (TILE_TABLE, TILE_FURNACE):
                # Replace with floor tile (per upstream behaviour).
                empty = (
                    TILE_GRASS if self._current_floor == 0
                    else TILE_DUNGEON_FLOOR
                )
                self._current_grid()[p.y][p.x] = empty
                return True
            return False

        def _hit_player(p) -> bool:
            if p.x == self._agent_x and p.y == self._agent_y:
                from glyphbench.envs.craftax.mechanics.damage import (
                    damage_vec_from_projectile,
                )

                self._take_damage(
                    p.damage,
                    damage_vec=(
                        p.damage_vec
                        if p.damage_vec is not None
                        else damage_vec_from_projectile(p.kind, p.damage)
                    ),
                )
                self._is_sleeping = False  # upstream: hit cancels sleep
                return True
            return False

        self._mob_projectiles = step_mob_projectiles(
            self._mob_projectiles,
            map_w=size, map_h=size,
            blocked_fn=_blocked,
            block_destruction_fn=_block_destroy,
            hit_player_fn=_hit_player,
        )

    def _mob_ai(self) -> None:
        """Move mobs on current floor and handle attacks.

        Melee, ranged, and passive mobs use the upstream-shaped helpers in
        mechanics/mobs.py: cooldown-aware melee, ranged kiting/projectiles,
        and passive random walk.

        Turn order: attack-then-move per upstream. step_melee_mob and
        step_ranged_mob encapsulate that; the inline branches mirror it for
        non-melee/non-ranged.
        """
        from glyphbench.envs.craftax.mechanics.damage import damage_vec_from_mob
        from glyphbench.envs.craftax.mechanics.mobs import (
            MELEE_MOB_NAMES,
            PASSIVE_MOB_NAMES,
            RANGED_MOB_NAMES,
            step_melee_mob,
            step_passive_mob,
        )
        from glyphbench.envs.craftax.mechanics.mobs import (
            RANGED_MOB_TO_PROJECTILE as _RMTP,
        )

        fsize = self._floor_size()
        walkable = self._walkable_set()

        def _is_blocked_for_mob_type(mtype: str, x: int, y: int) -> bool:
            if not (
                0 <= x < fsize
                and 0 <= y < fsize
                and (x, y) != (self._agent_x, self._agent_y)
                and not self._mob_at(x, y)
            ):
                return True
            tile = self._current_grid()[y][x]
            if tile in _SOLID_TILES:
                return True
            if mtype in {"bat", "fire_elemental", "ice_elemental"}:
                return False
            if mtype == "deep_thing":
                return tile != TILE_WATER
            if mtype == "lizard":
                return tile == TILE_LAVA or tile not in (walkable | {TILE_WATER})
            return tile in {TILE_WATER, TILE_LAVA} or tile not in walkable

        def _damage_for(mob: dict) -> int:
            mtype = mob["type"]
            if mob.get("is_boss"):
                bdef = _BOSS_DEFS.get(mob["floor"], {})
                return bdef.get("damage", 3)
            return _MOB_STATS.get(mtype, {"damage": 1})["damage"]

        for mob in list(self._mobs):
            if mob["floor"] != self._current_floor:
                continue
            mx, my = mob["x"], mob["y"]
            mtype = mob["type"]

            # Melee mobs use the upstream roster and elemental damage mix.
            if mtype in MELEE_MOB_NAMES:
                def _apply_melee_damage(dmg: int, _mob=mob) -> None:
                    self._take_damage(
                        dmg,
                        damage_vec=damage_vec_from_mob(_mob["type"], dmg),
                    )

                step_melee_mob(
                    mob,
                    player_x=self._agent_x,
                    player_y=self._agent_y,
                    is_blocked_for_mob=lambda x, y, _mtype=mtype: (
                        _is_blocked_for_mob_type(_mtype, x, y)
                    ),
                    apply_damage_to_player=_apply_melee_damage,
                    rng=random.Random(int(self.rng.integers(0, 2**31))),
                    is_fighting_boss=self._is_in_boss_fight(),
                    damage_for_mob=_damage_for,
                )
                continue

            # Ranged mobs use the upstream roster/projectile mapping.
            elif mtype in RANGED_MOB_NAMES:
                from glyphbench.envs.craftax.mechanics.mobs import step_ranged_mob
                from glyphbench.envs.craftax.mechanics.projectiles import (
                    ProjectileEntity,
                    ProjectileType,
                )

                def _spawn_proj(kind, x, y, dx, dy, dmg):
                    self._mob_projectiles.append(
                        ProjectileEntity(kind=kind, x=x, y=y, dx=dx, dy=dy, damage=dmg)
                    )

                def _proj_kind(m: dict, _rmtp=_RMTP) -> ProjectileType:
                    return _rmtp.get(m["type"], ProjectileType.ARROW2)

                step_ranged_mob(
                    mob,
                    player_x=self._agent_x,
                    player_y=self._agent_y,
                    is_blocked_for_mob=lambda x, y, _mtype=mtype: (
                        _is_blocked_for_mob_type(_mtype, x, y)
                    ),
                    spawn_mob_projectile=_spawn_proj,
                    rng=random.Random(int(self.rng.integers(0, 2**31))),
                    max_mob_projectiles_room=3 - sum(
                        1 for _ in self._mob_projectiles
                    ),
                    projectile_kind_for_mob=_proj_kind,
                    damage_for_mob=_damage_for,
                )
                continue

            elif mtype in PASSIVE_MOB_NAMES:
                step_passive_mob(
                    mob,
                    is_blocked_for_mob=lambda x, y, _mtype=mtype: (
                        _is_blocked_for_mob_type(_mtype, x, y)
                    ),
                    rng=random.Random(int(self.rng.integers(0, 2**31))),
                )
                continue

            # === existing inline logic for per-floor boss mobs === #

            # 1. ATTACK (hostile mobs only). Adjacency is measured at
            # mob's pre-move position so the player escaping melee on
            # the agent's turn breaks the damage chain this turn.
            if mtype != "cow":
                start_adj = (
                    abs(mx - self._agent_x) + abs(my - self._agent_y)
                )
                if mob["is_boss"]:
                    bdef = _BOSS_DEFS.get(mob["floor"], {})
                    dmg = bdef.get("damage", 3)
                    is_ranged = bdef.get("ranged", False)
                else:
                    dmg = _MOB_STATS.get(
                        mtype, {"damage": 1}
                    )["damage"]
                    is_ranged = False

                if start_adj <= 1:
                    self._take_damage(dmg)
                    if not self._message:
                        self._message = (
                            f"A {mtype} hits you for {dmg}!"
                        )
                elif is_ranged and start_adj <= 4:
                    self._take_damage(max(1, dmg // 2))
                    if not self._message:
                        self._message = (
                            f"A {mtype} shoots you!"
                        )

            # 2. MOVE.
            ddx, ddy = 0, 0
            dist_x = self._agent_x - mx
            dist_y = self._agent_y - my
            if abs(dist_x) + abs(dist_y) <= 8:
                if abs(dist_x) >= abs(dist_y):
                    ddx = 1 if dist_x > 0 else (-1 if dist_x < 0 else 0)
                else:
                    ddy = 1 if dist_y > 0 else (-1 if dist_y < 0 else 0)

            nx, ny = mx + ddx, my + ddy
            if (
                0 <= nx < fsize
                and 0 <= ny < fsize
                and (nx, ny) != (self._agent_x, self._agent_y)
                and not self._mob_at(nx, ny)
                and self._current_grid()[ny][nx] in walkable
            ):
                mob["x"] = nx
                mob["y"] = ny

    def _spawn_night_mobs(self) -> None:
        if self._current_floor != 0:
            return
        num = int(self.rng.integers(2, 5))
        for _ in range(num):
            mt = "zombie"  # upstream floor-0 melee mob only (T_FOLLOWUP_A)
            for _att in range(20):
                dx = int(self.rng.integers(-6, 7))
                dy = int(self.rng.integers(-6, 7))
                x = self._agent_x + dx
                y = self._agent_y + dy
                dist = abs(dx) + abs(dy)
                if (
                    3 <= dist <= 6
                    and 0 <= x < _SURFACE_SIZE
                    and 0 <= y < _SURFACE_SIZE
                    and self._floors[0][y][x]
                    in SURFACE_WALKABLE
                    and not self._mob_at(x, y, 0)
                    and (x, y)
                    != (self._agent_x, self._agent_y)
                ):
                    if self.rng.random() < _NIGHT_SPAWN_BASE_CHANCE:
                        mob: Mob = {
                            "type": mt,
                            "x": x, "y": y,
                            "hp": _MOB_STATS[mt]["hp"],
                            "max_hp": _MOB_STATS[mt]["hp"],
                            "is_boss": False,
                            "floor": 0,
                            "attack_cooldown": 0,
                        }
                        self._mobs.append(mob)
                    break

    def _despawn_night_mobs(self) -> None:
        self._mobs = [
            m for m in self._mobs
            if m["floor"] != 0
            or m["type"] != "zombie"  # only zombie spawns at night on floor 0 (T_FOLLOWUP_A)
        ]

    # ---------------------------------------------------------------
    # Day/night cycle
    # ---------------------------------------------------------------

    def _advance_day_counter(self, steps: int = 1) -> None:
        for _ in range(steps):
            old = self._day_night
            self._day_counter += 1
            pos = self._day_counter % _CYCLE_LENGTH
            new = "day" if pos < _DAY_LENGTH else "night"
            self._day_night = new

            if old == "day" and new == "night":
                self._night_count += 1
                self._message = (
                    "The sun sets. Monsters emerge."
                )
                self._spawn_night_mobs()
                # Night survival milestones
                if self._night_count >= 5:
                    self._try_unlock("survive_5_nights")
                if self._night_count >= 10:
                    self._try_unlock("survive_10_nights")
                if self._night_count >= 20:
                    self._try_unlock("survive_20_nights")
            elif old == "night" and new == "day":
                self._message = "The sun rises."
                self._despawn_night_mobs()

    # ---------------------------------------------------------------
    # Survival mechanics
    # ---------------------------------------------------------------

    def _recompute_max_stats(self) -> None:
        """Phase γ T09γ: recompute all 5 stat ceilings from current attributes.

        Call after attributes change (reset + each LEVEL_UP_* handler).
        """
        from glyphbench.envs.craftax.mechanics.progression import (
            max_drink_from_dex,
            max_energy_from_dex,
            max_food_from_dex,
            max_hp_from_str,
            max_mana_from_int,
        )
        self._max_hp = max_hp_from_str(_BASE_MAX_HP, self._str)
        self._max_food = max_food_from_dex(_MAX_FOOD, self._dex)
        self._max_drink = max_drink_from_dex(_MAX_WATER, self._dex)
        self._max_energy = max_energy_from_dex(_MAX_ENERGY, self._dex)
        self._max_mana = max_mana_from_int(_MAX_MANA, self._int_attr)

    def _apply_survival_drain(self) -> None:
        from glyphbench.envs.craftax.mechanics.boss import NECROMANCER_FLOOR
        from glyphbench.envs.craftax.mechanics.progression import (
            decay_scale,
            mana_regen_scale,
        )
        # Floor 8 (Necromancer) freezes vitals per docs: the wave-summon timer
        # is the only pressure during the boss encounter.
        if self._current_floor == NECROMANCER_FLOOR:
            return
        if self._food == 0:
            self._hp -= 1
        if self._water == 0:
            self._hp -= 1
        self._hp = max(0, self._hp)

        # Phase γ T09γ: DEX slows need-decay; decay_scale(dex=1) == 1.0 (no change).
        # We implement stochastic fractional decay: on each drain tick, apply the
        # fractional multiplier by converting the interval: higher DEX means the
        # effective drain fires less often. We use a probabilistic approach:
        # on the normal tick, drain with probability decay_scale(dex).
        ds = decay_scale(self._dex)

        step = self._day_counter
        if step > 0 and step % _FOOD_DRAIN_INTERVAL == 0 and self.rng.random() < ds:
            self._food = max(0, self._food - 1)
        if step > 0 and step % _WATER_DRAIN_INTERVAL == 0 and self.rng.random() < ds:
            self._water = max(0, self._water - 1)
        if step > 0 and step % _ENERGY_DRAIN_INTERVAL == 0 and self.rng.random() < ds:
            self._energy = max(0, self._energy - 1)
        # Mana regen — Phase γ T09γ: INT scales regen amount.
        if step > 0 and step % _MANA_REGEN_INTERVAL == 0:
            regen = max(1, int(round(mana_regen_scale(self._int_attr))))
            self._mana = min(self._max_mana, self._mana + regen)

        # Tick effects
        if self._speed_turns > 0:
            self._speed_turns -= 1

    def _tick_plants(self) -> None:
        to_ripen: list[tuple[int, int]] = []
        for pos, remaining in self._plants.items():
            remaining -= 1
            self._plants[pos] = remaining
            if remaining <= 0:
                to_ripen.append(pos)
        for pos in to_ripen:
            del self._plants[pos]
            x, y = pos
            grid = self._floors[0]
            if (
                0 <= x < _SURFACE_SIZE
                and 0 <= y < _SURFACE_SIZE
                and grid[y][x] == TILE_SAPLING
            ):
                grid[y][x] = TILE_RIPE_PLANT

    # ---------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._floors = {}
        self._mobs = []
        self._player_projectiles = []
        self._mob_projectiles = []
        self._is_sleeping = False
        self._is_resting = False
        self._pending_step_reward: float = 0.0
        self._stairs_down_pos = {}
        self._stairs_up_pos = {}
        self._bosses_alive = {}
        self._chests_opened = {}
        self._first_chest_opened = {}
        # Lighting / torch state.
        self._torches_by_floor = {}
        self._floor_ambient_light = {}
        self._lava_light_cache = {}

        self._generate_surface()
        for fl in range(1, _NUM_DUNGEON_FLOORS + 1):
            self._generate_dungeon_floor(fl)

        self._current_floor = 0
        self._agent_x = _SURFACE_SIZE // 2
        self._agent_y = _SURFACE_SIZE // 2
        self._facing = (1, 0)
        self._inventory = {
            "bow": 0,
            "arrows": 0,
            "torches": 0,
            "sapphire": 0,
            "ruby": 0,
            "book": 0,
            "potions": {"red": 0, "green": 0, "blue": 0, "pink": 0, "cyan": 0, "yellow": 0},
        }
        # Phase β T08β: hidden per-episode color->effect mapping.
        from glyphbench.envs.craftax.mechanics.potions import make_potion_mapping
        self._potion_mapping = make_potion_mapping(seed)
        self._achievements_unlocked = set()
        self._achievements_phase_beta = {n: False for n in UPSTREAM_ACHIEVEMENT_NAMES}
        self._message = ""
        self._learned_spells = {"fireball": False, "iceball": False}
        self._day_counter = 0
        self._day_night = "day"
        self._night_count = 0
        self._plants = {}
        self._potions = []  # compatibility list; unused in phase beta
        self._speed_turns = 0
        # Phase γ T10-T12γ: reset element-aware enchantment state.
        self._sword_enchantment = 0
        self._bow_enchantment = 0
        # Phase γ T03γ: reset 4-slot armour state.
        self._armor_slots = {
            "helmet": 0, "chest": 0, "legs": 0, "boots": 0
        }
        self._armor_enchants = {
            "helmet": 0, "chest": 0, "legs": 0, "boots": 0
        }
        self._total_kills = 0
        self._total_crafts = 0
        self._total_blocks_placed = 0
        self._total_plants_eaten = 0
        self._total_water_drunk = 0
        # Phase γ T06γ: XP resets each episode.
        self._xp = 0
        self._xp_floors_visited = set()
        # Phase γ T07γ: attributes reset to baseline.
        self._dex = 1
        self._str = 1
        self._int_attr = 1
        # Phase γ T17γ: necromancer boss state reset.
        self._boss_progress = 0
        self._boss_summon_timer = 0
        # Phase γ T09γ: recompute max stats from fresh attributes, then fill.
        self._recompute_max_stats()
        self._hp = self._max_hp
        self._food = self._max_food
        self._water = self._max_drink
        self._energy = self._max_energy
        self._mana = self._max_mana

        self._spawn_initial_cows()
        return self._render_current_observation()

    # ---------------------------------------------------------------
    # Step
    # ---------------------------------------------------------------

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        self._message = ""
        reward = 0.0

        reward += self._apply_sleep_rest_tick()
        self._apply_lava_tick()

        if not (self._is_sleeping or self._is_resting):
            handler = self._ACTION_DISPATCH.get(name)
            if handler is not None:
                # Resolve through the instance so subclass overrides of a
                # dispatched handler (e.g. CraftaxChopTreesEnv._handle_do) are
                # actually called. The dict stores unbound CraftaxFullEnv
                # functions; binding by name via getattr respects the MRO.
                reward += getattr(self, handler.__name__)()
        # (NOOP falls through with 0 reward)

        reward += self._apply_world_tick()
        fast_forward_ticks = 0
        fast_forward_capped = False
        terminated = self._is_terminal()
        while (
            (self._is_sleeping or self._is_resting)
            and not terminated
            and fast_forward_ticks < _FAST_FORWARD_MAX_TICKS
        ):
            reward += self._apply_sleep_rest_tick()
            self._apply_lava_tick()
            reward += self._apply_world_tick()
            fast_forward_ticks += 1
            terminated = self._is_terminal()
        if self._is_sleeping or self._is_resting:
            fast_forward_capped = not terminated

        reward, terminated = self._finalize_step_reward(reward)

        info: dict[str, Any] = {
            "agent_pos": (self._agent_x, self._agent_y),
            "floor": self._current_floor,
            "inventory": dict(self._inventory),
            "achievements": list(self._achievements_unlocked),
            "achievements_this_step": (
                [self._message.split(": ")[1].rstrip("!")]
                if self._message.startswith("ACHIEVEMENT")
                else []
            ),
            "day_night": self._day_night,
            "food": self._food,
            "water": self._water,
            "energy": self._energy,
            "mana": self._mana,
            "fast_forward_ticks": fast_forward_ticks,
            "fast_forward_capped": fast_forward_capped,
        }

        obs = self._render_current_observation()
        return obs, reward, terminated, False, info

    def _apply_sleep_rest_tick(self) -> float:
        """Apply one inert sleep/rest tick before any player action.

        When called from the fast-forward loop, this preserves local Craftax
        ordering: sleep/rest regen first, lava/dynamics afterward, and no
        player action while either state is active.
        """
        reward = 0.0
        if self._is_resting:
            self._hp = min(self._max_hp, self._hp + 1)
            if (
                self._hp >= self._max_hp
                or self._food <= 0
                or self._water <= 0
            ):
                self._is_resting = False

        if self._is_sleeping:
            self._hp = min(self._max_hp, self._hp + 2)
            self._energy = min(self._max_energy, self._energy + 2)
            if self._energy >= self._max_energy:
                self._is_sleeping = False
                reward += self._try_unlock("wake_up")
        return reward

    def _apply_lava_tick(self) -> None:
        # Phase β T07β: lava damage tick. Damage cancels REST/SLEEP via
        # _take_damage before any movement or inert fast-forward continuation.
        if (
            self._is_in_bounds(self._agent_x, self._agent_y)
            and self._tile_at(self._agent_x, self._agent_y) == TILE_LAVA
        ):
            self._take_damage(2)

    def _apply_world_tick(self) -> float:
        reward = 0.0
        self._advance_day_counter()
        self._apply_survival_drain()
        self._tick_plants()
        # Phase α — advance player projectiles after the action handler runs.
        self._step_player_projectiles()
        reward += self._pending_step_reward
        self._pending_step_reward = 0.0
        self._mob_ai()
        # Phase α — advance mob projectiles after _mob_ai so projectiles
        # spawned by ranged mobs in this tick travel immediately (T25).
        self._step_mob_projectiles()

        # Phase α: despawn mobs that drifted beyond MOB_DESPAWN_DISTANCE.
        from glyphbench.envs.craftax.mechanics.mobs import should_despawn
        is_boss_fight = self._is_in_boss_fight()
        gated_task_floors = set(getattr(self, "_GATED_FLOORS", ()))
        self._mobs = [
            m for m in self._mobs
            if not (
                m["floor"] == self._current_floor
                and m["floor"] not in gated_task_floors
                and should_despawn(
                    m,
                    player_x=self._agent_x,
                    player_y=self._agent_y,
                    is_fighting_boss=is_boss_fight,
                )
            )
        ]

        # Phase γ T17γ: Necromancer boss timer + wave summoning.
        from glyphbench.envs.craftax.mechanics.boss import (
            BOSS_PROGRESS_WIN_THRESHOLD,
            NECROMANCER_FLOOR,
        )
        if (
            self._current_floor == NECROMANCER_FLOOR
            and self._boss_progress < BOSS_PROGRESS_WIN_THRESHOLD
        ):
            if self._boss_summon_timer > 0:
                self._boss_summon_timer -= 1
                self._spawn_necromancer_wave()
            self._update_necromancer_tile_glyph()

        if self._hp == self._max_hp:
            reward += self._try_unlock("full_health")
        if self._mana == self._max_mana:
            reward += self._try_unlock("full_mana")

        reward += self._check_exploration_milestones()
        return reward

    def _necromancer_won(self) -> bool:
        from glyphbench.envs.craftax.mechanics.boss import (
            NECROMANCER_FLOOR,
            boss_progress_win,
        )
        return self._current_floor == NECROMANCER_FLOOR and boss_progress_win(self)

    def _is_terminal(self) -> bool:
        return self._hp <= 0 or self._necromancer_won()

    def _finalize_step_reward(self, reward: float) -> tuple[float, bool]:
        necromancer_won = self._necromancer_won()
        terminated = self._hp <= 0 or necromancer_won
        if self._hp <= 0:
            self._message = "You died."
            # Pattern B: -1.0 death penalty applied ADDITIVELY so any
            # upstream-achievement reward earned this step (and the
            # cumulative MC return across the episode) persists for GRPO.
            # Subtasks disable this and manage their own death penalty.
            if self._emit_death_penalty:
                reward += -1.0
        elif necromancer_won:
            self._message = "The necromancer is defeated! You win!"
        return reward, terminated

    # ---------------------------------------------------------------
    # Action handlers
    # ---------------------------------------------------------------

    def _handle_move(self, name: str) -> float:
        direction_map: dict[str, tuple[int, int]] = {
            "MOVE_LEFT": (-1, 0),
            "MOVE_RIGHT": (1, 0),
            "MOVE_UP": (0, -1),
            "MOVE_DOWN": (0, 1),
        }
        dx, dy = direction_map[name]
        self._facing = (dx, dy)

        if self._energy == 0 and self.rng.random() < 0.5:
            self._message = "Too exhausted to move!"
            return 0.0

        steps = 2 if self._speed_turns > 0 else 1
        for _ in range(steps):
            nx = self._agent_x + dx
            ny = self._agent_y + dy
            fsize = self._floor_size()
            grid = self._current_grid()
            walkable = self._walkable_set()
            if (
                0 <= nx < fsize
                and 0 <= ny < fsize
                and grid[ny][nx] in walkable
                and not self._mob_at(nx, ny)
            ):
                self._agent_x = nx
                self._agent_y = ny
        return 0.0

    def _handle_move_left(self) -> float:
        return self._handle_move("MOVE_LEFT")

    def _handle_move_right(self) -> float:
        return self._handle_move("MOVE_RIGHT")

    def _handle_move_up(self) -> float:
        return self._handle_move("MOVE_UP")

    def _handle_move_down(self) -> float:
        return self._handle_move("MOVE_DOWN")

    def _handle_do(self) -> float:
        fsize = self._floor_size()
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if not (0 <= fx < fsize and 0 <= fy < fsize):
            return 0.0

        reward = 0.0
        mob = self._mob_at(fx, fy)
        if mob is not None:
            reward += self._attack_mob(mob)
            return reward

        grid = self._current_grid()
        tile = grid[fy][fx]

        # ---- Necromancer interaction (T17γ) ----
        # Both TILE_NECROMANCER and TILE_NECROMANCER_VULNERABLE are valid
        # targets — the vulnerability check is done inside boss.py.
        if tile in (TILE_NECROMANCER, TILE_NECROMANCER_VULNERABLE):
            from glyphbench.envs.craftax.mechanics.boss import (
                boss_progress_win,
                damage_necromancer_if_vulnerable,
            )
            if damage_necromancer_if_vulnerable(self):
                self._message = "You struck the necromancer!"
                # Flip necromancer tile back to invulnerable glyph immediately.
                grid[fy][fx] = TILE_NECROMANCER
                if boss_progress_win(self):
                    # Necromancer defeated! The unlock alone yields the
                    # normalized achievement reward; no extra bonus is
                    # added so cumulative reward stays bounded by [-1, 1].
                    self._message = "The necromancer is defeated! You win!"
                    reward += self._try_unlock("defeat_necromancer")
            else:
                self._message = "The necromancer is invulnerable!"
            return reward

        # ---- Chest interaction (T12β / T13β / T14β) ----
        if tile == TILE_CHEST:
            floor = self._current_floor
            opened_on_floor = self._chests_opened.get(floor, set())
            if (fx, fy) in opened_on_floor:
                self._message = "This chest is already open."
                return 0.0
            reward += self._open_chest(fx, fy, floor)
            return reward

        # ---- Fountain interaction (T18β) ----
        # Drinking from a fountain refills water to max. The tile is NOT
        # consumed — fountains are reusable.
        if tile == TILE_FOUNTAIN:
            if self._water >= self._max_drink:
                self._message = "The fountain is full — you are not thirsty."
                return 0.0
            self._water = self._max_drink
            self._total_water_drunk += 1
            self._message = "You drink deeply from the fountain. Water restored!"
            reward += self._try_unlock("collect_drink")
            if self._total_water_drunk >= 10:
                reward += self._try_unlock("drink_10_water")
            return reward

        if tile in INTERACTABLE_TILES:
            resource = INTERACTABLE_TILES[tile]
            if tile in PICKAXE_REQUIRED:
                min_tier = PICKAXE_REQUIRED[tile]
                has_pick = any(
                    self._inventory.get(p, 0) > 0
                    for p in _PICKAXE_TIERS[min_tier:]
                )
                if not has_pick:
                    needed = _PICKAXE_TIERS[min_tier]
                    self._message = (
                        f"Need a {needed.replace('_', ' ')}."
                    )
                    return 0.0
            self._inventory[resource] = (
                self._inventory.get(resource, 0) + 1
            )
            # Replace with floor tile
            if self._current_floor == 0:
                grid[fy][fx] = TILE_GRASS
            else:
                grid[fy][fx] = TILE_DUNGEON_FLOOR
            self._message = f"Collected {resource}."
            reward += self._try_unlock(
                f"collect_{resource}"
            )
            if (
                tile == TILE_TREE
                and self.rng.random() < _SAPLING_DROP_CHANCE
            ):
                self._inventory["sapling"] = (
                    self._inventory.get("sapling", 0) + 1
                )
                self._message += " Found a sapling!"
                reward += self._try_unlock(
                    "collect_sapling"
                )
            # Resource milestones
            reward += self._check_resource_milestones()
        return reward

    # Pickaxe and sword tiers ordered by level (1-indexed as in upstream).
    _PICKAXE_BY_LEVEL: tuple[str, ...] = (
        "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe",
    )
    _SWORD_BY_LEVEL: tuple[str, ...] = (
        "wood_sword", "stone_sword", "iron_sword", "diamond_sword",
    )

    def _open_chest(self, fx: int, fy: int, floor: int) -> float:
        """Roll loot from a chest and apply first-chest gating (T13β + T14β).

        Parameters
        ----------
        fx, fy : int
            Position of the chest tile on the current floor grid.
        floor : int
            Current dungeon floor number.
        """
        reward = 0.0
        loot_parts: list[str] = []

        # ---- Generic loot table (T13β) ----
        # Wood (60 %)
        if self.rng.random() < 0.6:
            amount = int(self.rng.integers(1, 6))  # 1-5 inclusive
            self._inventory["wood"] = self._inventory.get("wood", 0) + amount
            loot_parts.append(f"{amount} wood")

        # Ore (60 %, mutually exclusive)
        if self.rng.random() < 0.6:
            ore_roll = self.rng.random()
            # Probabilities: coal 30 %, iron 30 %, diamond 15 %,
            # sapphire 12.5 %, ruby 12.5 %
            if ore_roll < 0.30:
                amount = int(self.rng.integers(1, 4))  # 1-3
                self._inventory["coal"] = self._inventory.get("coal", 0) + amount
                loot_parts.append(f"{amount} coal")
            elif ore_roll < 0.60:
                amount = int(self.rng.integers(1, 3))  # 1-2
                self._inventory["iron"] = self._inventory.get("iron", 0) + amount
                loot_parts.append(f"{amount} iron")
            elif ore_roll < 0.75:
                self._inventory["diamond"] = self._inventory.get("diamond", 0) + 1
                loot_parts.append("1 diamond")
            elif ore_roll < 0.875:
                self._inventory["sapphire"] = self._inventory.get("sapphire", 0) + 1
                loot_parts.append("1 sapphire")
            else:
                self._inventory["ruby"] = self._inventory.get("ruby", 0) + 1
                loot_parts.append("1 ruby")

        # Potions (50 %)
        if self.rng.random() < 0.5:
            _potion_colors = ("red", "green", "blue", "pink", "cyan", "yellow")
            color_idx = int(self.rng.integers(0, 6))
            amount = int(self.rng.integers(1, 3))  # 1-2
            color = _potion_colors[color_idx]
            potions = self._inventory.get("potions", {})
            potions[color] = potions.get(color, 0) + amount
            self._inventory["potions"] = potions
            loot_parts.append(f"{amount} {color} potion(s)")

        # Arrows (25 %)
        if self.rng.random() < 0.25:
            amount = int(self.rng.integers(1, 5))  # 1-4
            self._inventory["arrows"] = self._inventory.get("arrows", 0) + amount
            loot_parts.append(f"{amount} arrows")

        # Pickaxe upgrade (20 %)
        if self.rng.random() < 0.2:
            level_roll = self.rng.random()
            if level_roll < 0.40:
                level = 1
            elif level_roll < 0.70:
                level = 2
            elif level_roll < 0.90:
                level = 3
            else:
                level = 4
            new_key = self._PICKAXE_BY_LEVEL[level - 1]
            # Keep max: only grant if this is better than what we have.
            current_best = 0
            for li, pk in enumerate(self._PICKAXE_BY_LEVEL):
                if self._inventory.get(pk, 0) > 0:
                    current_best = li + 1
            if level > current_best:
                self._inventory[new_key] = self._inventory.get(new_key, 0) + 1
                loot_parts.append(f"{new_key.replace('_', ' ')}")

        # Sword upgrade (20 %)
        if self.rng.random() < 0.2:
            level_roll = self.rng.random()
            if level_roll < 0.40:
                level = 1
            elif level_roll < 0.70:
                level = 2
            elif level_roll < 0.90:
                level = 3
            else:
                level = 4
            new_key = self._SWORD_BY_LEVEL[level - 1]
            current_best = 0
            for li, sw in enumerate(self._SWORD_BY_LEVEL):
                if self._inventory.get(sw, 0) > 0:
                    current_best = li + 1
            if level > current_best:
                self._inventory[new_key] = self._inventory.get(new_key, 0) + 1
                loot_parts.append(f"{new_key.replace('_', ' ')}")

        # Torches (60 %, 4-7) — fork chest loot. Added at the end of the loot
        # rolls so the earlier draws keep their deterministic stream positions.
        if self.rng.random() < 0.6:
            amount = int(self.rng.integers(4, 8))  # 4-7 inclusive
            self._inventory["torches"] = self._inventory.get("torches", 0) + amount
            loot_parts.append(f"{amount} torches")

        # Mark chest as opened.
        self._chests_opened.setdefault(floor, set()).add((fx, fy))
        reward += self._try_unlock("open_chest")

        # ---- First-chest gating (T14β) ----
        if floor == 1 and not self._first_chest_opened.get(1):
            self._inventory["bow"] = max(self._inventory.get("bow", 0), 1)
            self._first_chest_opened[1] = True
            reward += self._try_unlock("find_bow")
            loot_parts.append("bow")

        if floor in (3, 4):
            already_got_book = (
                self._first_chest_opened.get(3)
                or self._first_chest_opened.get(4)
            )
            if not already_got_book:
                self._inventory["book"] = self._inventory.get("book", 0) + 1
                self._first_chest_opened[floor] = True
                reward += self._try_unlock("find_book")
                loot_parts.append("book")

        loot_str = ", ".join(loot_parts) if loot_parts else "nothing"
        self._message = f"Opened chest: {loot_str}."
        return reward

    def _check_resource_milestones(self) -> float:
        r = 0.0
        if self._inventory.get("wood", 0) >= 10:
            r += self._try_unlock("collect_10_wood")
        if self._inventory.get("wood", 0) >= 20:
            r += self._try_unlock("max_inventory_wood")
        if self._inventory.get("stone", 0) >= 5:
            r += self._try_unlock("collect_5_stone")
        if self._inventory.get("iron", 0) >= 3:
            r += self._try_unlock("collect_3_iron")
        return r

    def _handle_sleep(self) -> float:
        """Phase β: enters continuous sleep state. Regen and energy restore
        happen each tick in _step; this action just sets the flag."""
        if self._energy >= self._max_energy:
            self._message = "You are not tired enough to sleep."
            return 0.0
        self._is_sleeping = True
        self._message = "You fall asleep."
        return 0.0

    # -- Placement --

    def _handle_place_stone(self) -> float:
        fsize = self._floor_size()
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        empty = (
            TILE_GRASS if self._current_floor == 0
            else TILE_DUNGEON_FLOOR
        )
        if (
            0 <= fx < fsize
            and 0 <= fy < fsize
            and self._current_grid()[fy][fx] in {empty, TILE_WATER, TILE_LAVA}
            and not self._mob_at(fx, fy)
            and self._inventory.get("stone", 0) >= 1
        ):
            self._inventory["stone"] -= 1
            self._current_grid()[fy][fx] = TILE_PLACED_STONE
            self._total_blocks_placed += 1
            self._message = "Placed stone."
            r = self._try_unlock("place_stone")
            if self._total_blocks_placed >= 10:
                r += self._try_unlock("place_10_blocks")
            return r
        return 0.0

    def _handle_place_table(self) -> float:
        fsize = self._floor_size()
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        empty = (
            TILE_GRASS if self._current_floor == 0
            else TILE_DUNGEON_FLOOR
        )
        if (
            0 <= fx < fsize
            and 0 <= fy < fsize
            and self._current_grid()[fy][fx] == empty
            and not self._mob_at(fx, fy)
            and self._inventory.get("wood", 0) >= 2
        ):
            self._inventory["wood"] -= 2
            self._current_grid()[fy][fx] = TILE_TABLE
            self._total_blocks_placed += 1
            self._message = "Placed crafting table."
            r = self._try_unlock("place_table")
            if self._total_blocks_placed >= 10:
                r += self._try_unlock("place_10_blocks")
            return r
        return 0.0

    def _handle_place_furnace(self) -> float:
        fsize = self._floor_size()
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        empty = (
            TILE_GRASS if self._current_floor == 0
            else TILE_DUNGEON_FLOOR
        )
        if (
            0 <= fx < fsize
            and 0 <= fy < fsize
            and self._current_grid()[fy][fx] == empty
            and not self._mob_at(fx, fy)
            and self._inventory.get("stone", 0) >= 1
        ):
            self._inventory["stone"] -= 1
            self._current_grid()[fy][fx] = TILE_FURNACE
            self._total_blocks_placed += 1
            self._message = "Placed furnace."
            r = self._try_unlock("place_furnace")
            if self._total_blocks_placed >= 10:
                r += self._try_unlock("place_10_blocks")
            return r
        return 0.0

    def _handle_place_plant(self) -> float:
        if self._current_floor != 0:
            return 0.0
        fsize = self._floor_size()
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if (
            0 <= fx < fsize
            and 0 <= fy < fsize
            and self._floors[0][fy][fx] == TILE_GRASS
            and not self._mob_at(fx, fy)
            and self._inventory.get("sapling", 0) >= 1
        ):
            self._inventory["sapling"] -= 1
            self._floors[0][fy][fx] = TILE_SAPLING
            self._plants[(fx, fy)] = _PLANT_RIPEN_STEPS
            self._message = "Planted a sapling."
            return self._try_unlock("place_plant")
        return 0.0

    # -- Crafting: pickaxes --

    def _craft_increment(self) -> float:
        self._total_crafts += 1
        r = 0.0
        if self._total_crafts >= 10:
            r += self._try_unlock("craft_10_items")
        if self._total_crafts >= 25:
            r += self._try_unlock("craft_25_items")
        return r

    def _handle_make_wood_pickaxe(self) -> float:
        if (
            self._near_table()
            and self._inventory.get("wood", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["wood_pickaxe"] = (
                self._inventory.get("wood_pickaxe", 0) + 1
            )
            self._message = "Crafted wood pickaxe."
            r = self._try_unlock("make_wood_pickaxe")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_stone_pickaxe(self) -> float:
        # Stone tier needs a table only (upstream Craftax recipe);
        # furnace is tier-3+ (iron and beyond).
        if (
            self._near_table()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("stone", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["stone"] -= 1
            self._inventory["stone_pickaxe"] = (
                self._inventory.get("stone_pickaxe", 0) + 1
            )
            self._message = "Crafted stone pickaxe."
            r = self._try_unlock("make_stone_pickaxe")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_iron_pickaxe(self) -> float:
        if (
            self._near_table()
            and self._near_furnace()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("stone", 0) >= 1
            and self._inventory.get("iron", 0) >= 1
            and self._inventory.get("coal", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["stone"] -= 1
            self._inventory["iron"] -= 1
            self._inventory["coal"] -= 1
            self._inventory["iron_pickaxe"] = (
                self._inventory.get("iron_pickaxe", 0) + 1
            )
            self._message = "Crafted iron pickaxe."
            r = self._try_unlock("make_iron_pickaxe")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_diamond_pickaxe(self) -> float:
        if (
            self._near_table()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("diamond", 0) >= 3
        ):
            self._inventory["wood"] -= 1
            self._inventory["diamond"] -= 3
            self._inventory["diamond_pickaxe"] = (
                self._inventory.get("diamond_pickaxe", 0) + 1
            )
            self._message = "Crafted diamond pickaxe."
            r = self._try_unlock("make_diamond_pickaxe")
            return r + self._craft_increment()
        return 0.0

    # -- Crafting: swords --

    def _handle_make_wood_sword(self) -> float:
        if (
            self._near_table()
            and self._inventory.get("wood", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["wood_sword"] = (
                self._inventory.get("wood_sword", 0) + 1
            )
            self._message = "Crafted wood sword."
            r = self._try_unlock("make_wood_sword")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_stone_sword(self) -> float:
        # Stone tier needs a table only (upstream Craftax recipe).
        if (
            self._near_table()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("stone", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["stone"] -= 1
            self._inventory["stone_sword"] = (
                self._inventory.get("stone_sword", 0) + 1
            )
            self._message = "Crafted stone sword."
            r = self._try_unlock("make_stone_sword")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_iron_sword(self) -> float:
        if (
            self._near_table()
            and self._near_furnace()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("stone", 0) >= 1
            and self._inventory.get("iron", 0) >= 1
            and self._inventory.get("coal", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["stone"] -= 1
            self._inventory["iron"] -= 1
            self._inventory["coal"] -= 1
            self._inventory["iron_sword"] = (
                self._inventory.get("iron_sword", 0) + 1
            )
            self._message = "Crafted iron sword."
            r = self._try_unlock("make_iron_sword")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_diamond_sword(self) -> float:
        if (
            self._near_table()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("diamond", 0) >= 2
        ):
            self._inventory["wood"] -= 1
            self._inventory["diamond"] -= 2
            self._inventory["diamond_sword"] = (
                self._inventory.get("diamond_sword", 0) + 1
            )
            self._message = "Crafted diamond sword."
            r = self._try_unlock("make_diamond_sword")
            return r + self._craft_increment()
        return 0.0

    # -- Crafting: armor --
    # Phase γ T03γ: MAKE_WOOD_ARMOR and MAKE_STONE_ARMOR removed (upstream
    # only has iron + diamond tiers). Their action indices are retired.

    def _handle_make_arrow(self) -> float:
        if (
            self._near_table()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("stone", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["stone"] -= 1
            self._inventory["arrows"] = (
                self._inventory.get("arrows", 0) + 2
            )
            self._message = "Crafted 2 arrows."
            return self._try_unlock("make_arrow")
        return 0.0

    # -- Torches (2026-06 port-vs-fork alignment) --

    def _handle_make_torch(self) -> float:
        """MAKE_TORCH: 1 wood + 1 coal at a crafting table -> 4 torches."""
        if (
            self._near_table()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("coal", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["coal"] -= 1
            self._inventory["torches"] = (
                self._inventory.get("torches", 0) + _TORCH_MAKE_YIELD
            )
            self._message = f"Crafted {_TORCH_MAKE_YIELD} torches."
            r = self._try_unlock("make_torch")
            return r + self._craft_increment()
        self._message = "Need a nearby table, 1 wood and 1 coal to craft torches."
        return 0.0

    def _handle_place_torch(self) -> float:
        """PLACE_TORCH: drop a torch on the faced floor cell, lighting a radius."""
        if self._inventory.get("torches", 0) < 1:
            self._message = "You have no torches."
            return 0.0
        fsize = self._floor_size()
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if not (0 <= fx < fsize and 0 <= fy < fsize):
            self._message = "Cannot place a torch there."
            return 0.0
        tile = self._current_grid()[fy][fx]
        # Torches go on open, non-hazard, non-solid floor cells.
        if tile in _SOLID_TILES or tile in (TILE_WATER, TILE_LAVA):
            self._message = "Cannot place a torch there (blocked or hazardous)."
            return 0.0
        self._inventory["torches"] -= 1
        torches = self._torches_by_floor.setdefault(self._current_floor, [])
        if (fx, fy) not in torches:
            torches.append((fx, fy))
        self._message = "Placed a torch. The area lights up."
        return self._try_unlock("place_torch")

    def _handle_make_iron_armor(self) -> float:
        # Phase γ T03γ: place tier-1 armour into the lowest-tier empty slot.
        _SLOTS = ("helmet", "chest", "legs", "boots")
        target_slot = None
        for s in _SLOTS:
            if self._armor_slots.get(s, 0) < 1:
                target_slot = s
                break
        if target_slot is None:
            self._message = "All armour slots already iron+."
            return 0.0
        if (
            self._near_table()
            and self._near_furnace()
            and self._inventory.get("iron", 0) >= 3
            and self._inventory.get("coal", 0) >= 3
        ):
            self._inventory["iron"] -= 3
            self._inventory["coal"] -= 3
            self._armor_slots[target_slot] = 1
            self._message = f"Crafted iron armor ({target_slot})."
            r = self._try_unlock("make_iron_armor")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_diamond_armor(self) -> float:
        # Phase γ T03γ: upgrade the lowest-tier slot (strict < 2) to tier 2.
        _SLOTS = ("helmet", "chest", "legs", "boots")
        target_slot = None
        for s in _SLOTS:
            if self._armor_slots.get(s, 0) < 2:
                target_slot = s
                break
        if target_slot is None:
            self._message = "All armour slots already diamond."
            return 0.0
        if (
            self._near_table()
            and self._inventory.get("diamond", 0) >= 3
        ):
            self._inventory["diamond"] -= 3
            self._armor_slots[target_slot] = 2
            self._message = f"Crafted diamond armor ({target_slot})."
            r = self._try_unlock("make_diamond_armor")
            return r + self._craft_increment()
        return 0.0

    # -- Magic --

    def _handle_cast_fireball(self) -> float:
        """Spawn a fireball projectile in front of the agent.

        Mirrors upstream cast_spell:2547-2599 — point projectile, NOT AOE.
        Cost: 2 mana (matches upstream). Phase α uses scalar damage; phase γ
        promotes to a (physical, fire, ice) 3-vector.
        """
        from glyphbench.envs.craftax.mechanics.damage import (
            damage_vec_from_projectile,
        )
        from glyphbench.envs.craftax.mechanics.projectiles import (
            ProjectileEntity,
            ProjectileType,
        )

        if not self._learned_spells["fireball"]:
            self._message = "No spells learned yet."
            return 0.0
        if self._mana < 2:
            self._message = "Not enough mana! (need 2)"
            return 0.0
        # Spawn at the player's tile (upstream spawn_projectile places at
        # state.player_position; the same-step advance by _move_player_projectile
        # then carries it to player_pos + 1*dir). Player is always in-bounds,
        # so the old bounds check on spawn_x/spawn_y was vacuous.
        from glyphbench.envs.craftax.mechanics.progression import damage_scale_spell
        dx, dy = self._facing
        self._mana -= 2
        # Fork base fireball damage = 3 (MOB_TYPE_DAMAGE_MAPPING projectile col),
        # scaled by intelligence via damage_scale_spell.
        fireball_damage = int(round(3 * damage_scale_spell(self._int_attr)))
        self._player_projectiles.append(
            ProjectileEntity(
                kind=ProjectileType.FIREBALL,
                x=self._agent_x, y=self._agent_y, dx=dx, dy=dy,
                damage=fireball_damage,
                damage_vec=damage_vec_from_projectile(
                    ProjectileType.FIREBALL, fireball_damage
                ),
            )
        )
        self._message = "You cast a fireball."
        return self._try_unlock("cast_fireball")

    def _handle_cast_iceball(self) -> float:
        """Spawn an iceball projectile in front of the agent.

        Mirrors upstream cast_spell:2547-2599 — point projectile, NOT a freeze
        and NOT an AOE. Cost: 2 mana. Phase α uses scalar damage; phase γ
        promotes to a (physical, fire, ice) 3-vector that bypasses ice
        defense.
        """
        from glyphbench.envs.craftax.mechanics.damage import (
            damage_vec_from_projectile,
        )
        from glyphbench.envs.craftax.mechanics.projectiles import (
            ProjectileEntity,
            ProjectileType,
        )

        if not self._learned_spells["iceball"]:
            self._message = "No spells learned yet."
            return 0.0
        if self._mana < 2:
            self._message = "Not enough mana! (need 2)"
            return 0.0
        # Spawn at the player's tile (upstream-faithful; same-step advance
        # carries it to player_pos + 1*dir). Player is always in-bounds.
        from glyphbench.envs.craftax.mechanics.progression import damage_scale_spell
        dx, dy = self._facing
        self._mana -= 2
        iceball_damage = int(round(3 * damage_scale_spell(self._int_attr)))
        self._player_projectiles.append(
            ProjectileEntity(
                kind=ProjectileType.ICEBALL,
                x=self._agent_x, y=self._agent_y, dx=dx, dy=dy,
                damage=iceball_damage,
                damage_vec=damage_vec_from_projectile(
                    ProjectileType.ICEBALL, iceball_damage
                ),
            )
        )
        self._message = "You cast an iceball."
        return self._try_unlock("cast_iceball")

    def _handle_read_book(self) -> float:
        """Read a book to learn an unlearned spell (T11β).

        Consumes 1 book from inventory. Picks a random unlearned spell from
        _learned_spells and marks it True. Fires the learn_<spell> achievement.
        No-ops if the player has no book or already knows all spells.
        """
        if self._inventory.get("book", 0) < 1:
            self._message = "No book to read."
            return 0.0
        unlearned = [
            spell for spell, known in self._learned_spells.items() if not known
        ]
        if not unlearned:
            self._message = "You already know all spells."
            return 0.0
        chosen = unlearned[int(self.rng.integers(0, len(unlearned)))]
        self._learned_spells[chosen] = True
        self._inventory["book"] -= 1
        self._message = f"You read the book. Learned {chosen}!"
        return self._try_unlock(f"learn_{chosen}")

    def _handle_shoot_arrow(self) -> float:
        """Spawn an arrow projectile at the player's tile.

        Mirrors upstream SHOOT_ARROW (action 24). Requires a bow and at least
        1 arrow in inventory. Spawn position is the player's own tile; the
        per-step driver advances the projectile +1 tile/turn in the agent's
        facing direction.
        """
        from glyphbench.envs.craftax.mechanics.projectiles import (
            ProjectileEntity,
            ProjectileType,
        )

        if self._inventory.get("bow", 0) < 1:
            self._message = "You need a bow to shoot arrows."
            return 0.0
        if self._inventory.get("arrows", 0) < 1:
            self._message = "No arrows."
            return 0.0
        from glyphbench.envs.craftax.mechanics.progression import damage_scale_arrow
        dx, dy = self._facing
        self._inventory["arrows"] -= 1
        arrow_damage = int(round(5 * damage_scale_arrow(self._dex)))
        # Phase γ T12γ: bow enchantment adds 0.5× elemental damage component.
        if self._bow_enchantment == 1:
            dvec: tuple[float, float, float] | None = (
                float(arrow_damage), 0.5 * arrow_damage, 0.0
            )
        elif self._bow_enchantment == 2:
            dvec = (float(arrow_damage), 0.0, 0.5 * arrow_damage)
        else:
            dvec = None  # scalar physical-only
        self._player_projectiles.append(
            ProjectileEntity(
                kind=ProjectileType.ARROW2,
                x=self._agent_x, y=self._agent_y,
                dx=dx, dy=dy,
                damage=arrow_damage,
                damage_vec=dvec,
            )
        )
        self._message = "You fired an arrow."
        return self._try_unlock("fire_bow")

    def _attack_mob_kill(self, mob: Mob) -> float:
        """Handle mob death from spells (mob already at <= 0 HP)."""
        if mob in self._mobs:
            self._mobs.remove(mob)
        self._total_kills += 1
        reward = self._check_kill_milestones()
        mtype = mob["type"]
        if mob["is_boss"]:
            fl = mob["floor"]
            self._bosses_alive[fl] = False
            bdef = _BOSS_DEFS.get(fl)
            if bdef:
                reward += self._try_unlock(bdef["ach"])
                loot = f"boss_loot_{fl}"
                self._inventory[loot] = (
                    self._inventory.get(loot, 0) + 1
                )
                reward += self._try_unlock(
                    f"collect_boss_loot_{fl}"
                )
        elif mtype == "zombie":
            reward += self._try_unlock("defeat_zombie")
        elif mtype == "skeleton":
            reward += self._try_unlock("defeat_skeleton")
        elif mtype == "cow":
            self._food = min(self._max_food, self._food + 6)
            reward += self._try_unlock("eat_cow")
        elif mtype == "kobold":
            reward += self._try_unlock("defeat_kobold")
        elif mtype == "bat":
            self._food = min(self._max_food, self._food + 6)
            reward += self._try_unlock("eat_bat")
        return reward

    # -- Potions (phase β: 6 color-keyed handlers) --

    def _handle_drink_potion_color(self, color: str, color_idx: int) -> float:
        """Shared implementation for all DRINK_POTION_* handlers.

        *color* is lowercase (e.g. "red"), *color_idx* is the index into
        POTION_COLORS (0=RED, 1=GREEN, …).  The actual effect is determined
        by looking up self._potion_mapping[color_idx] into POTION_EFFECTS —
        this mapping is hidden and never exposed in the observation.
        """
        from glyphbench.envs.craftax.mechanics.potions import (
            POTION_EFFECTS,
            apply_potion_effect,
        )
        potions_dict = self._inventory.get("potions", {})
        if potions_dict.get(color, 0) < 1:
            self._message = f"No {color} potion."
            return 0.0
        potions_dict[color] -= 1
        effect_idx = self._potion_mapping[color_idx]
        effect = POTION_EFFECTS[effect_idx]
        apply_potion_effect(self, effect)
        self._message = f"You drank a {color} potion."
        return self._try_unlock("drink_potion")

    def _handle_drink_potion_red(self) -> float:
        return self._handle_drink_potion_color("red", 0)

    def _handle_drink_potion_green(self) -> float:
        return self._handle_drink_potion_color("green", 1)

    def _handle_drink_potion_blue(self) -> float:
        return self._handle_drink_potion_color("blue", 2)

    def _handle_drink_potion_pink(self) -> float:
        return self._handle_drink_potion_color("pink", 3)

    def _handle_drink_potion_cyan(self) -> float:
        return self._handle_drink_potion_color("cyan", 4)

    def _handle_drink_potion_yellow(self) -> float:
        return self._handle_drink_potion_color("yellow", 5)

    # -- Eat / drink --

    def _handle_eat_plant(self) -> float:
        fsize = self._floor_size()
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if (
            0 <= fx < fsize
            and 0 <= fy < fsize
            and self._current_grid()[fy][fx] == TILE_RIPE_PLANT
        ):
            if self._current_floor == 0:
                self._floors[0][fy][fx] = TILE_PLANT
            else:
                self._current_grid()[fy][fx] = TILE_PLANT
            self._food = min(self._max_food, self._food + 4)
            self._total_plants_eaten += 1
            self._message = "Ate a plant. (+4 food)"
            r = self._try_unlock("eat_plant")
            if self._total_plants_eaten >= 5:
                r += self._try_unlock("eat_5_plants")
            return r
        return 0.0

    def _handle_drink_water(self) -> float:
        fsize = self._floor_size()
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if (
            0 <= fx < fsize
            and 0 <= fy < fsize
            and self._current_grid()[fy][fx] == TILE_WATER
        ):
            self._water = min(self._max_drink, self._water + 1)
            self._total_water_drunk += 1
            self._message = "Drank water. (+1 drink)"
            r = self._try_unlock("collect_drink")
            if self._total_water_drunk >= 10:
                r += self._try_unlock("drink_10_water")
            return r
        return 0.0

    # -- Stairs --

    def _handle_descend(self) -> float:
        grid = self._current_grid()
        ax, ay = self._agent_x, self._agent_y
        if grid[ay][ax] != TILE_STAIRS_DOWN:
            self._message = "No stairs here."
            return 0.0

        new_floor = self._current_floor + 1
        if new_floor > _NUM_DUNGEON_FLOORS:
            self._message = "Cannot go deeper."
            return 0.0

        self._current_floor = new_floor
        # Place agent at stairs up of new floor
        up_pos = self._stairs_up_pos.get(new_floor)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        else:
            self._agent_x = _DUNGEON_SIZE // 2
            self._agent_y = _DUNGEON_SIZE // 2

        reward = 0.0
        if new_floor == 1:
            self._message = "Entered the dungeon!"
            reward += self._try_unlock("enter_dungeon")
        elif new_floor == 2:
            self._message = "Reached floor 2."
            reward += self._try_unlock("reach_floor_2")
        elif new_floor == 3:
            self._message = "Reached floor 3."
            reward += self._try_unlock("reach_floor_3")
        elif new_floor == 4:
            self._message = "Reached floor 4."
            reward += self._try_unlock("reach_floor_4")
        elif new_floor == 5:
            self._message = "Reached floor 5 — Troll Mines."
            reward += self._try_unlock("reach_floor_5")
        elif new_floor == 6:
            self._message = "Entered the Fire Realm."
            reward += self._try_unlock("enter_fire_realm")
        elif new_floor == 7:
            self._message = "Entered the Ice Realm."
            reward += self._try_unlock("enter_ice_realm")
        elif new_floor == 8:
            self._message = "Entered the Graveyard. The necromancer awaits..."
            reward += self._try_unlock("enter_graveyard")
        self._player_projectiles = []
        self._mob_projectiles = []
        # Phase γ T06γ: first entry to any new floor grants +1 XP.
        if new_floor not in self._xp_floors_visited:
            self._xp_floors_visited.add(new_floor)
            self._xp += 1
        return reward

    def _handle_ascend(self) -> float:
        grid = self._current_grid()
        ax, ay = self._agent_x, self._agent_y
        if grid[ay][ax] != TILE_STAIRS_UP:
            self._message = "No stairs here."
            return 0.0
        if self._current_floor <= 0:
            self._message = "Already on surface."
            return 0.0

        new_floor = self._current_floor - 1
        self._current_floor = new_floor

        reward = 0.0
        if new_floor == 0:
            # Return to surface
            down_pos = self._stairs_down_pos.get(0)
            if down_pos:
                self._agent_x = down_pos[0]
                self._agent_y = down_pos[1]
            else:
                self._agent_x = _SURFACE_SIZE // 2
                self._agent_y = _SURFACE_SIZE // 2
            self._message = "Returned to the surface!"
            reward += self._try_unlock("return_to_surface")
        else:
            down_pos = self._stairs_down_pos.get(new_floor)
            if down_pos:
                self._agent_x = down_pos[0]
                self._agent_y = down_pos[1]
            self._message = f"Ascended to floor {new_floor}."
        self._player_projectiles = []
        self._mob_projectiles = []
        return reward

    # -- Enchantments (T10γ/T11γ/T12γ) --

    def _handle_enchant_weapon(self) -> float:
        """T10γ: ENCHANT_SWORD — fire/ice table + ruby/sapphire + 9 mana.

        Requires facing TILE_ENCHANT_FIRE or TILE_ENCHANT_ICE.
        Costs 1 ruby (fire) or 1 sapphire (ice) + 9 mana.
        Sets _sword_enchantment to 1 (fire) or 2 (ice).
        """
        # Check sword in inventory
        has_sword = any(
            self._inventory.get(w, 0) > 0
            for w in ("wood_sword", "stone_sword", "iron_sword", "diamond_sword")
        )
        if not has_sword:
            self._message = "No sword to enchant."
            return 0.0
        if self._sword_enchantment != 0:
            self._message = "Sword already enchanted."
            return 0.0
        if self._mana < 9:
            self._message = "Not enough mana to enchant (need 9)."
            return 0.0
        element = self._pick_enchant_element()
        if element == 0:
            self._message = "No faced enchant table (or missing ruby/sapphire)."
            return 0.0
        # Consume gem + mana
        if element == 1:
            self._inventory["ruby"] = self._inventory.get("ruby", 0) - 1
            self._sword_enchantment = 1
            self._message = "Sword enchanted with FIRE!"
        else:
            self._inventory["sapphire"] = self._inventory.get("sapphire", 0) - 1
            self._sword_enchantment = 2
            self._message = "Sword enchanted with ICE!"
        self._mana -= 9
        return self._try_unlock("enchant_sword")

    def _handle_enchant_armor(self) -> float:
        """T11γ: ENCHANT_ARMOUR — fire/ice table + ruby/sapphire + 9 mana.

        Requires facing TILE_ENCHANT_FIRE or TILE_ENCHANT_ICE.
        Targets the lowest-tier slot with armour AND no current enchant.
        Costs 1 ruby (fire) or 1 sapphire (ice) + 9 mana.
        Sets _armor_enchants[slot] to 1 (fire) or 2 (ice).
        """
        _SLOTS = ("helmet", "chest", "legs", "boots")
        target_slot = None
        for s in _SLOTS:
            if self._armor_enchants.get(s, 0) == 0 and self._armor_slots.get(s, 0) > 0:
                target_slot = s
                break
        if target_slot is None:
            self._message = "No unenchanted armour slot to enchant."
            return 0.0
        if self._mana < 9:
            self._message = "Not enough mana to enchant (need 9)."
            return 0.0
        element = self._pick_enchant_element()
        if element == 0:
            self._message = "No faced enchant table (or missing ruby/sapphire)."
            return 0.0
        if element == 1:
            self._inventory["ruby"] = self._inventory.get("ruby", 0) - 1
            self._armor_enchants[target_slot] = 1
            self._message = f"Armor ({target_slot}) enchanted with FIRE!"
        else:
            self._inventory["sapphire"] = self._inventory.get("sapphire", 0) - 1
            self._armor_enchants[target_slot] = 2
            self._message = f"Armor ({target_slot}) enchanted with ICE!"
        self._mana -= 9
        return self._try_unlock("enchant_armor")

    def _handle_enchant_bow(self) -> float:
        """T12γ: ENCHANT_BOW — fire/ice table + ruby/sapphire + 9 mana.

        Requires bow in inventory, a faced enchant table, and mana.
        Sets _bow_enchantment to 1 (fire) or 2 (ice).
        """
        if self._inventory.get("bow", 0) < 1:
            self._message = "No bow to enchant."
            return 0.0
        if self._bow_enchantment != 0:
            self._message = "Bow already enchanted."
            return 0.0
        if self._mana < 9:
            self._message = "Not enough mana to enchant (need 9)."
            return 0.0
        element = self._pick_enchant_element()
        if element == 0:
            self._message = "No faced enchant table (or missing ruby/sapphire)."
            return 0.0
        if element == 1:
            self._inventory["ruby"] = self._inventory.get("ruby", 0) - 1
            self._bow_enchantment = 1
            self._message = "Bow enchanted with FIRE!"
        else:
            self._inventory["sapphire"] = self._inventory.get("sapphire", 0) - 1
            self._bow_enchantment = 2
            self._message = "Bow enchanted with ICE!"
        self._mana -= 9
        return self._try_unlock("enchant_bow")

    # -- REST --

    def _handle_rest(self) -> float:
        """Enter REST state: regen +1 HP/tick until full, starving, or hit."""
        if self._hp >= self._max_hp:
            self._message = "You are already at full health."
            return 0.0
        self._is_resting = True
        self._message = "You rest."
        return 0.0

    # -- Phase γ T08γ: LEVEL_UP attribute actions --

    def _handle_level_up_dexterity(self) -> float:
        """Spend 1 XP to raise dexterity (cap 5)."""
        if self._xp < 1:
            self._message = "Not enough XP to level up."
            return 0.0
        if self._dex >= 5:
            self._message = "Dexterity is already at maximum (5)."
            return 0.0
        self._xp -= 1
        self._dex += 1
        self._recompute_max_stats()
        self._message = f"Dexterity raised to {self._dex}!"
        return self._try_unlock("level_up_dexterity")

    def _handle_level_up_strength(self) -> float:
        """Spend 1 XP to raise strength (cap 5)."""
        if self._xp < 1:
            self._message = "Not enough XP to level up."
            return 0.0
        if self._str >= 5:
            self._message = "Strength is already at maximum (5)."
            return 0.0
        self._xp -= 1
        self._str += 1
        self._recompute_max_stats()
        self._message = f"Strength raised to {self._str}!"
        return self._try_unlock("level_up_strength")

    def _handle_level_up_intelligence(self) -> float:
        """Spend 1 XP to raise intelligence (cap 5)."""
        if self._xp < 1:
            self._message = "Not enough XP to level up."
            return 0.0
        if self._int_attr >= 5:
            self._message = "Intelligence is already at maximum (5)."
            return 0.0
        self._xp -= 1
        self._int_attr += 1
        self._recompute_max_stats()
        self._message = f"Intelligence raised to {self._int_attr}!"
        return self._try_unlock("level_up_intelligence")

    # -- Action dispatch table --

    _ACTION_DISPATCH: dict[str, Any] = {  # method refs
        "MOVE_LEFT": _handle_move_left,
        "MOVE_RIGHT": _handle_move_right,
        "MOVE_UP": _handle_move_up,
        "MOVE_DOWN": _handle_move_down,
        "DO": _handle_do,
        "SLEEP": _handle_sleep,
        "PLACE_STONE": _handle_place_stone,
        "PLACE_TABLE": _handle_place_table,
        "PLACE_FURNACE": _handle_place_furnace,
        "PLACE_PLANT": _handle_place_plant,
        "MAKE_WOOD_PICKAXE": _handle_make_wood_pickaxe,
        "MAKE_STONE_PICKAXE": _handle_make_stone_pickaxe,
        "MAKE_IRON_PICKAXE": _handle_make_iron_pickaxe,
        "MAKE_DIAMOND_PICKAXE": _handle_make_diamond_pickaxe,
        "MAKE_WOOD_SWORD": _handle_make_wood_sword,
        "MAKE_STONE_SWORD": _handle_make_stone_sword,
        "MAKE_IRON_SWORD": _handle_make_iron_sword,
        "MAKE_DIAMOND_SWORD": _handle_make_diamond_sword,
        "MAKE_IRON_ARMOR": _handle_make_iron_armor,
        "MAKE_DIAMOND_ARMOR": _handle_make_diamond_armor,
        "CAST_FIREBALL": _handle_cast_fireball,
        "CAST_ICEBALL": _handle_cast_iceball,
        "DRINK_POTION_RED": _handle_drink_potion_red,
        "DRINK_POTION_GREEN": _handle_drink_potion_green,
        "DRINK_POTION_BLUE": _handle_drink_potion_blue,
        "DRINK_POTION_PINK": _handle_drink_potion_pink,
        "DRINK_POTION_CYAN": _handle_drink_potion_cyan,
        "DRINK_POTION_YELLOW": _handle_drink_potion_yellow,
        "EAT_PLANT": _handle_eat_plant,
        "DRINK_WATER": _handle_drink_water,
        "DESCEND": _handle_descend,
        "ASCEND": _handle_ascend,
        "ENCHANT_WEAPON": _handle_enchant_weapon,
        "ENCHANT_ARMOR": _handle_enchant_armor,
        "ENCHANT_BOW": _handle_enchant_bow,
        "MAKE_ARROW": _handle_make_arrow,
        "SHOOT_ARROW": _handle_shoot_arrow,
        "REST": _handle_rest,
        "READ_BOOK": _handle_read_book,
        "LEVEL_UP_DEXTERITY": _handle_level_up_dexterity,
        "LEVEL_UP_STRENGTH": _handle_level_up_strength,
        "LEVEL_UP_INTELLIGENCE": _handle_level_up_intelligence,
        "MAKE_TORCH": _handle_make_torch,
        "PLACE_TORCH": _handle_place_torch,
    }

    # ---------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------

    # -- Lighting (fork-faithful local-view masking) --

    def _floor_ambient(self, floor: int) -> float:
        """Ambient light level for a floor (surface lit; dungeons per config)."""
        if floor in self._floor_ambient_light:
            return self._floor_ambient_light[floor]
        if floor == 0:
            return 1.0
        return self._dungeon_ambient_light

    def _light_phase_name(self) -> str:
        """Day/Dusk/Night label for the HUD (matches the fork's Light field)."""
        cycle = self._day_counter % _CYCLE_LENGTH
        if self._day_night == "night":
            return "Night"
        if cycle >= _DAY_LENGTH - 20:
            return "Dusk"
        return "Day"

    def _lava_sources(self, floor: int) -> list[tuple[int, int]]:
        """Lazily-cached lava tile positions on a floor (glow sources)."""
        cache = self._lava_light_cache.get(floor)
        if cache is None:
            grid = self._floors.get(floor, [])
            cache = [
                (x, y)
                for y, row in enumerate(grid)
                for x, c in enumerate(row)
                if c == TILE_LAVA
            ]
            self._lava_light_cache[floor] = cache
        return cache

    @staticmethod
    def _radial_light(dx: int, dy: int, divisor: float) -> float:
        """Torch-style radial falloff: 1 - euclidean(dx,dy)/divisor, clipped."""
        dist = (dx * dx + dy * dy) ** 0.5
        return max(0.0, 1.0 - dist / divisor)

    def _light_at(self, floor: int, x: int, y: int) -> float:
        """Light level [0,1] at a cell, combining ambient + torch/ladder/lava."""
        amb = self._floor_ambient(floor)
        if amb >= 1.0:
            return 1.0
        light = amb
        for (tx, ty) in self._torches_by_floor.get(floor, []):
            light = max(light, self._radial_light(x - tx, y - ty, _TORCH_FALLOFF_DIVISOR))
            if light >= 1.0:
                return 1.0
        up = self._stairs_up_pos.get(floor)
        if up is not None:
            light = max(light, self._radial_light(x - up[0], y - up[1], _TORCH_FALLOFF_DIVISOR))
        for (lx, ly) in self._lava_sources(floor):
            light = max(light, self._radial_light(x - lx, y - ly, _LAVA_FALLOFF_DIVISOR))
            if light >= 1.0:
                return 1.0
        return min(1.0, light)

    def _is_visible(self, wx: int, wy: int) -> bool:
        """A cell is visible if in-bounds and lit above the darkness threshold.

        On fully-lit floors (surface, or dungeons with ambient 1.0) every
        in-bounds cell is visible. On dark floors only cells lit by the arrival
        ladder, lava, or a placed torch are visible (fork light_mask).
        """
        if not (0 <= wx < self._floor_size() and 0 <= wy < self._floor_size()):
            return False
        floor = self._current_floor
        if self._floor_ambient(floor) >= 1.0:
            return True
        return self._light_at(floor, wx, wy) > _LIGHT_VISIBLE_THRESHOLD

    def _render_current_observation(self) -> GridObservation:
        half_w = FULL_VIEW_WIDTH // 2
        half_h = FULL_VIEW_HEIGHT // 2
        grid: list[list[str]] = []
        symbols_seen: set[str] = set()
        fsize = self._floor_size()
        cur_grid = self._current_grid()

        # Directional player char
        facing_ch = _DIR_CHARS.get(self._facing, "@")
        facing_name = _DIR_NAMES.get(self._facing, "right")

        mob_chars: dict[tuple[int, int], str] = {}
        visible_mobs: list[Mob] = []
        for mob in self._mobs:
            if mob["floor"] == self._current_floor:
                if mob["is_boss"]:
                    mob_chars[(mob["x"], mob["y"])] = (
                        TILE_BOSS
                    )
                elif mob["type"] in _MOB_TILES:
                    mob_chars[(mob["x"], mob["y"])] = (
                        _MOB_TILES[mob["type"]]
                    )

        from glyphbench.envs.craftax.mechanics.projectiles import ProjectileType
        _PROJECTILE_GLYPH = {
            ProjectileType.ARROW: TILE_ARROW,
            ProjectileType.ARROW2: TILE_ARROW2,
            ProjectileType.DAGGER: TILE_DAGGER,
            ProjectileType.FIREBALL: TILE_FIREBALL,
            ProjectileType.FIREBALL2: TILE_FIREBALL2,
            ProjectileType.ICEBALL: TILE_ICEBALL,
            ProjectileType.ICEBALL2: TILE_ICEBALL2,
            ProjectileType.SLIMEBALL: TILE_SLIMEBALL,
        }
        proj_chars: dict[tuple[int, int], str] = {}
        for p in self._player_projectiles + self._mob_projectiles:
            proj_chars[(p.x, p.y)] = _PROJECTILE_GLYPH[p.kind]

        # Placed torches (light sources). Fork renders a torch as "*"; reuse the
        # ripe-plant glyph for format parity. Torches are always visible.
        torch_chars: dict[tuple[int, int], str] = {
            pos: TILE_RIPE_PLANT
            for pos in self._torches_by_floor.get(self._current_floor, [])
        }

        for wy in range(FULL_VIEW_HEIGHT):
            row: list[str] = []
            for wx in range(FULL_VIEW_WIDTH):
                world_x = self._agent_x - half_w + wx
                world_y = self._agent_y - half_h + wy
                if (
                    world_x == self._agent_x
                    and world_y == self._agent_y
                ):
                    row.append(facing_ch)
                    symbols_seen.add(facing_ch)
                elif (world_x, world_y) in mob_chars:
                    if self._is_visible(world_x, world_y):
                        c = mob_chars[(world_x, world_y)]
                        row.append(c)
                        symbols_seen.add(c)
                    else:
                        row.append(" ")
                elif (world_x, world_y) in proj_chars:
                    if self._is_visible(world_x, world_y):
                        c = proj_chars[(world_x, world_y)]
                        row.append(c)
                        symbols_seen.add(c)
                    else:
                        row.append(" ")
                elif (world_x, world_y) in torch_chars:
                    # Torches are light sources — always rendered.
                    c = torch_chars[(world_x, world_y)]
                    row.append(c)
                    symbols_seen.add(c)
                elif (
                    0 <= world_x < fsize
                    and 0 <= world_y < fsize
                ):
                    if self._is_visible(world_x, world_y):
                        tile = cur_grid[world_y][world_x]
                        row.append(tile)
                        symbols_seen.add(tile)
                    else:
                        row.append(" ")
                else:
                    row.append(" ")
            grid.append(row)

        # Collect visible mobs for HUD. A mob must be inside the view window AND
        # in a lit cell (on dark floors, unlit mobs are hidden — fork-faithful).
        for mob in self._mobs:
            if mob["floor"] != self._current_floor:
                continue
            dx = mob["x"] - self._agent_x
            dy = mob["y"] - self._agent_y
            if (
                abs(dx) <= half_w
                and abs(dy) <= half_h
                and self._is_visible(mob["x"], mob["y"])
            ):
                visible_mobs.append(mob)

        # Inventory string (skip nested dicts like "potions"; those are shown separately)
        inv_parts: list[str] = []
        for item, count in sorted(self._inventory.items()):
            if isinstance(count, dict):
                continue  # "potions" sub-dict rendered separately
            if count > 0:
                inv_parts.append(f"{item} x{count}")
        inv_str = (
            ", ".join(inv_parts) if inv_parts else "(empty)"
        )

        # Best weapon/armor with numeric values
        best_wpn = "none"
        best_wpn_bonus = 0
        for w in reversed(list(_WEAPON_BONUS.keys())):
            if self._inventory.get(w, 0) > 0:
                best_wpn = w.replace("_", " ")
                best_wpn_bonus = _WEAPON_BONUS[w]
                break
        _ENCHANT_ELEM = {0: "", 1: "[fire]", 2: "[ice]"}
        if best_wpn != "none":
            wpn_str = f"{best_wpn} (+{best_wpn_bonus} dmg)"
            if self._sword_enchantment != 0:
                wpn_str += f" {_ENCHANT_ELEM[self._sword_enchantment]}"
        else:
            wpn_str = "none"
        # Bow enchant status
        bow_ench_str = _ENCHANT_ELEM.get(self._bow_enchantment, "")

        # Phase γ T03γ: per-slot armour HUD.
        _TIER_NAMES = {0: "none", 1: "iron", 2: "diamond"}
        _ENCHANT_NAMES = {0: "", 1: "[fire]", 2: "[ice]"}
        slot_parts = []
        for _slot in ("helmet", "chest", "legs", "boots"):
            _tier = self._armor_slots.get(_slot, 0)
            _ench = self._armor_enchants.get(_slot, 0)
            _s = f"{_slot}={_TIER_NAMES[_tier]}"
            if _ench:
                _s += _ENCHANT_NAMES[_ench]
            slot_parts.append(_s)
        arm_str = ", ".join(slot_parts)

        floor_str = (
            "Surface" if self._current_floor == 0
            else f"Dungeon F{self._current_floor}"
        )

        ach_count = len(self._achievements_unlocked)
        total_ach = len(self._ALL_ACHIEVEMENTS)
        # Phase γ T21γ: upstream-faithful bitmap count (67 upstream achievements).
        bitmap_count = sum(1 for v in self._achievements_phase_beta.values() if v)
        _UPSTREAM_TOTAL = len(UPSTREAM_ACHIEVEMENT_NAMES)  # 67
        # Phase β: per-color potion inventory (never reveal effect mapping)
        potions_dict = self._inventory.get("potions", {})
        potion_parts = [
            f"{color} x{cnt}"
            for color, cnt in potions_dict.items()
            if cnt > 0
        ]
        potions_str = ", ".join(potion_parts) if potion_parts else "none"

        # Day/night cycle position
        cycle_pos = self._day_counter % _CYCLE_LENGTH
        if self._day_night == "day":
            phase_step = cycle_pos
            phase_len = _DAY_LENGTH
        else:
            phase_step = cycle_pos - _DAY_LENGTH
            phase_len = _NIGHT_LENGTH
        time_str = (
            f"{self._day_night} "
            f"(step {phase_step}/{phase_len})"
        )

        # Survival drain countdowns
        if self._current_floor == 8:
            drain_line = "Next drain: disabled on floor 8"
        else:
            food_drain = (
                _FOOD_DRAIN_INTERVAL
                - (self._day_counter % _FOOD_DRAIN_INTERVAL)
            )
            water_drain = (
                _WATER_DRAIN_INTERVAL
                - (self._day_counter % _WATER_DRAIN_INTERVAL)
            )
            energy_drain = (
                _ENERGY_DRAIN_INTERVAL
                - (self._day_counter % _ENERGY_DRAIN_INTERVAL)
            )
            drain_line = (
                f"Next drain: food in {food_drain}, "
                f"water in {water_drain}, "
                f"energy in {energy_drain}"
            )

        # Nearby mobs info
        mob_parts: list[str] = []
        for m in visible_mobs:
            mob_parts.append(
                f"{m['type']} ({m['hp']}/{m['max_hp']} HP)"
            )
        mob_str = (
            ", ".join(mob_parts) if mob_parts else "none"
        )

        # Boss info for visible bosses
        boss_parts: list[str] = []
        for m in visible_mobs:
            if m["is_boss"]:
                bdef = _BOSS_DEFS.get(m["floor"])
                if bdef:
                    boss_parts.append(
                        f"{bdef['name'].title()} "
                        f"(HP: {m['hp']}/{m['max_hp']}"
                        f", Dmg: {bdef['damage']})"
                    )

        # Spells learned (per-spell dict; T10β)
        spell_parts = [
            f"{spell}: {'known' if known else 'unknown'}"
            for spell, known in self._learned_spells.items()
        ]
        spells_str = ", ".join(spell_parts) if spell_parts else "none"

        # Active potion effects
        effects: list[str] = []
        if self._speed_turns > 0:
            effects.append(
                f"speed ({self._speed_turns} turns)"
            )
        effects_str = (
            ", ".join(effects) if effects else "none"
        )

        # Achievement names
        ach_names = sorted(self._achievements_unlocked)
        ach_name_str = (
            ", ".join(ach_names) if ach_names else "(none)"
        )

        # Growing saplings in viewport
        grow_parts: list[str] = []
        for (px, py), remaining in self._plants.items():
            pdx = px - self._agent_x
            pdy = py - self._agent_y
            if abs(pdx) <= half_w and abs(pdy) <= half_h:
                grow_parts.append(
                    f"sapling {remaining} steps left"
                )

        light_label = self._light_phase_name()
        if self._floor_ambient(self._current_floor) < 1.0:
            light_label += " (dark — place torches to see)"

        hud = (
            f"HP: {self._hp}/{self._max_hp}  "
            f"Food: {self._food}/{self._max_food}  "
            f"Water: {self._water}/{self._max_drink}  "
            f"Energy: {self._energy}/{self._max_energy}  "
            f"Mana: {self._mana}/{self._max_mana}\n"
            f"Floor: {floor_str}  "
            f"Light: {light_label}  "
            f"Time: {time_str}  "
            f"Step: {self._turn} / {self.max_turns}\n"
            f"{drain_line}\n"
            f"Weapon: {wpn_str}  "
            f"Bow: {'bow' if self._inventory.get('bow', 0) > 0 else 'none'}"
            f"{(' ' + bow_ench_str) if bow_ench_str else ''}  "
            f"Torches: {self._inventory.get('torches', 0)}  "
            f"Armor: {arm_str}\n"
            f"Spells: {spells_str}\n"
            f"Effects: {effects_str}\n"
            f"Potions: {potions_str}\n"
            f"Nearby mobs: {mob_str}\n"
            f"Inventory: {inv_str}\n"
            f"Achievements: {ach_name_str} "
            f"({ach_count}/{total_ach})\n"
            f"Upstream achievements: {bitmap_count} / {_UPSTREAM_TOTAL}"
        )
        if boss_parts:
            hud += "\nBoss: " + "; ".join(boss_parts)
        if grow_parts:
            hud += "\nGrowing: " + "; ".join(grow_parts)

        # Legend -- tile meanings (without player)
        agent_legend = f"you (facing {facing_name})"
        tile_meanings: dict[str, str] = {
            TILE_GRASS: "grass",
            TILE_TREE: "tree (chop for wood)",
            TILE_FIRE_TREE: "fire tree (DO for wood)",
            TILE_ICE_SHRUB: "ice shrub (DO for wood)",
            TILE_STONE: "stone (mine with pickaxe)",
            TILE_COAL: "coal ore",
            TILE_IRON: "iron ore",
            TILE_DIAMOND: "diamond",
            TILE_SAPPHIRE: "sapphire ore (needs diamond pickaxe)",
            TILE_RUBY: "ruby ore (needs diamond pickaxe)",
            TILE_WATER: "water (DRINK_WATER)",
            TILE_LAVA: "lava (2 dmg/tick)",
            TILE_SAND: "sand",
            TILE_TABLE: "crafting table",
            TILE_FURNACE: "furnace",
            TILE_PLACED_STONE: "placed stone",
            TILE_SAPLING: "sapling (growing)",
            TILE_RIPE_PLANT: "ripe plant (EAT_PLANT) / placed torch (light source)",
            TILE_ENCHANT_FIRE: "fire enchantment table",
            TILE_ENCHANT_ICE: "ice enchantment table",
            TILE_ZOMBIE: "zombie",
            TILE_COW: "cow (passive)",
            TILE_SKELETON_ARCHER: "skeleton or knight archer (ranged)",
            TILE_KOBOLD: "kobold (ranged)",
            TILE_BAT: "bat",
            TILE_GNOME_WARRIOR: "gnome warrior (melee)",
            TILE_GNOME_ARCHER: "gnome archer (ranged)",
            TILE_ORC_SOLDIER: "orc soldier (melee)",
            TILE_ORC_MAGE: "orc mage (ranged) or floor boss",
            TILE_LIZARD: "lizard (melee)",
            TILE_KNIGHT: "knight (melee)",
            TILE_TROLL: "troll (melee)",
            TILE_DEEP_THING: "deep thing (ranged)",
            TILE_SNAIL: "snail (passive)",
            TILE_PIGMAN: "pigman (melee)",
            TILE_FIRE_ELEMENTAL: "fire elemental (ranged)",
            TILE_FROST_TROLL: "frost troll (melee)",
            TILE_ICE_ELEMENTAL: "ice elemental (ranged)",
            TILE_STAIRS_DOWN: "stairs down (DESCEND)",
            TILE_STAIRS_UP: "stairs up (ASCEND)",
            TILE_DUNGEON_WALL: "dungeon wall",
            TILE_DUNGEON_FLOOR: "dungeon floor",
            TILE_BOSS_DOOR: "boss door (walkable, blocks projectiles)",
            # Phase β dungeon features (T18β)
            TILE_CHEST: "chest (DO to open — 1 chest per room)",
            TILE_FOUNTAIN: "fountain (DO to refill water)",
            # Projectiles (phase α — T26)
            TILE_ARROW: "arrow projectile",
            TILE_ARROW2: "arrow2 projectile",
            TILE_DAGGER: "dagger projectile",
            TILE_FIREBALL: "fireball projectile",
            TILE_FIREBALL2: "fireball2 projectile",
            TILE_ICEBALL: "iceball projectile",
            TILE_ICEBALL2: "iceball2 projectile",
            TILE_SLIMEBALL: "slimeball projectile",
            # Phase γ floor-8 Graveyard tiles (T16-T17γ)
            TILE_GRAVE: "grave marker (decoration)",
            TILE_NECROMANCER: "necromancer (invulnerable — clear arena first)",
            TILE_NECROMANCER_VULNERABLE: "necromancer (VULNERABLE — DO to hit!)",
        }
        legend_entries: dict[str, str] = {}
        for sym in symbols_seen:
            if sym == facing_ch:
                # Player char; combine with tile meaning if
                # the same symbol is also a tile
                base = tile_meanings.get(sym)
                if base:
                    legend_entries[sym] = (
                        f"{agent_legend} / {base}"
                    )
                else:
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
