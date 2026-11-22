"""Shared base for Craftax envs.

Provides constants for biome types, resource types, and the shared action spec.
"""

from __future__ import annotations

from glyphbench.core.action import ActionSpec

# Full Craftax action spec (43 actions)
# Phase β: DRINK_POTION (generic) dropped; 6 DRINK_POTION_* color actions added.
# T11β: READ_BOOK appended at index 42.
# Phase γ T03γ: MAKE_WOOD_ARMOR + MAKE_STONE_ARMOR removed (43 → 41).
# Phase γ T08γ: LEVEL_UP_DEXTERITY/STRENGTH/INTELLIGENCE appended (41 → 44).
# Phase γ T12γ: ENCHANT_BOW appended (idx 42).
# 2026-06 port-vs-fork alignment: PLACE_TORCH + MAKE_TORCH appended (43 actions
# 0-42, then 43=PLACE_TORCH, 44=MAKE_TORCH -> 45 total) to match the fork's
# torch/lighting system. Dispatch is by NAME (not index), so order is not
# load-bearing; torches are appended to minimise churn.
# Index breakdown: 0-6 noop/movement/do/sleep, 7-10 placement, 11-20 crafting,
# 21-22 spells, 23-24 eat/drink, 25-26 stairs, 27-28 enchant,
# 29 make_arrow, 30 shoot_arrow, 31 rest,
# 32-37 drink_potion_{red,green,blue,pink,cyan,yellow}, 38 read_book,
# 39-41 level_up_{dexterity,strength,intelligence}, 42 enchant_bow,
# 43 place_torch, 44 make_torch.
CRAFTAX_FULL_ACTION_SPEC = ActionSpec(
    names=(
        "NOOP", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN",
        "DO", "SLEEP",
        "PLACE_STONE", "PLACE_TABLE", "PLACE_FURNACE",
        "PLACE_PLANT",
        "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE",
        "MAKE_IRON_PICKAXE", "MAKE_DIAMOND_PICKAXE",
        "MAKE_WOOD_SWORD", "MAKE_STONE_SWORD",
        "MAKE_IRON_SWORD", "MAKE_DIAMOND_SWORD",
        "MAKE_IRON_ARMOR", "MAKE_DIAMOND_ARMOR",
        "CAST_FIREBALL", "CAST_ICEBALL",
        "EAT_PLANT", "DRINK_WATER",
        "DESCEND", "ASCEND",
        "ENCHANT_WEAPON", "ENCHANT_ARMOR",
        "MAKE_ARROW",
        "SHOOT_ARROW",
        "REST",
        "DRINK_POTION_RED",
        "DRINK_POTION_GREEN",
        "DRINK_POTION_BLUE",
        "DRINK_POTION_PINK",
        "DRINK_POTION_CYAN",
        "DRINK_POTION_YELLOW",
        "READ_BOOK",
        "LEVEL_UP_DEXTERITY",
        "LEVEL_UP_STRENGTH",
        "LEVEL_UP_INTELLIGENCE",
        "ENCHANT_BOW",
        "PLACE_TORCH",
        "MAKE_TORCH",
    ),
    descriptions=(
        "do nothing this turn",
        "move one cell left",
        "move one cell right",
        "move one cell up",
        "move one cell down",
        "interact with the cell you face (chop/mine/attack)",
        "sleep only when tired; time passes and mobs still act",
        "place a stone block (costs 1 stone)",
        "place a crafting table (costs 2 wood)",
        "place a furnace (costs 1 stone)",
        "place a sapling (costs 1 sapling)",
        "craft wood pickaxe (1 wood, table)",
        "craft stone pickaxe (1 wood+1 stone, table)",
        "craft iron pickaxe (1 wood+1 stone+1 iron+1 coal, table+furnace)",
        "craft diamond pickaxe (1 wood+3 diamond, table)",
        "craft wood sword (1 wood, table)",
        "craft stone sword (1 wood+1 stone, table)",
        "craft iron sword (1 wood+1 stone+1 iron+1 coal, table+furnace)",
        "craft diamond sword (1 wood+2 diamond, table)",
        "craft iron armor piece (3 iron+3 coal, table+furnace; fills lowest-tier slot)",
        "craft diamond armor piece (3 diamond, table; upgrades lowest slot)",
        "cast fireball (2 mana, single-target projectile)",
        "cast iceball (2 mana, single-target projectile)",
        "eat a ripe plant you face to restore food",
        "drink water you face to restore thirst",
        "descend stairs (\u21e3) to next dungeon floor",
        "ascend stairs (\u21e1) to previous floor",
        "enchant weapon (adds fire/ice element, 1 ruby/sapphire + 9 mana, facing enchant table)",
        "enchant armor (+1 def, 1 ruby/sapphire + 9 mana, facing enchant table)",
        "craft 2 arrows (1 wood + 1 stone, requires nearby table)",
        "shoot an arrow in your facing direction (requires bow + 1 arrow)",
        "rest in place to recover HP (wakes on full HP, 0 food, or 0 drink)",
        "drink a red potion (effect determined by per-game shuffle)",
        "drink a green potion (effect determined by per-game shuffle)",
        "drink a blue potion (effect determined by per-game shuffle)",
        "drink a pink potion (effect determined by per-game shuffle)",
        "drink a cyan potion (effect determined by per-game shuffle)",
        "drink a yellow potion (effect determined by per-game shuffle)",
        "read a book to learn fireball or iceball (random)",
        "spend 1 XP to raise dexterity (cap 5)",
        "spend 1 XP to raise strength (cap 5)",
        "spend 1 XP to raise intelligence (cap 5)",
        "enchant bow with fire or ice (1 ruby/sapphire + 9 mana, requires bow + facing enchant table)",
        "place a torch in the cell you face to light a radius (~4); needs 1 torch",
        "craft 4 torches (1 wood + 1 coal, requires nearby table)",
    ),
)

# Terrain/biome tile characters (Unicode)
TILE_GRASS = "\u00b7"          # · middle dot
TILE_TREE = "\u2663"           # ♣ club suit
TILE_STONE = "S"
TILE_COAL = "C"
TILE_IRON = "I"
TILE_DIAMOND = "D"
TILE_WATER = "\u2248"          # ≈ almost equal
TILE_LAVA = "\u2668"           # ♨ hot springs
TILE_SAND = "\u2591"           # ░ light shade
TILE_TABLE = "t"
TILE_FURNACE = "f"
TILE_PLACED_STONE = "="
TILE_PLANT = "+"

# Dungeon-specific tile characters (Unicode)
TILE_STAIRS_DOWN = "\u21e3"    # ⇣ downwards dashed arrow
TILE_STAIRS_UP = "\u21e1"      # ⇡ upwards dashed arrow
TILE_DUNGEON_WALL = "\u2588"   # █ full block
TILE_DUNGEON_FLOOR = "\u25aa"  # ▪ black small square
TILE_BOSS_DOOR = "B"

# Mob tile characters
TILE_ZOMBIE = "z"
TILE_SKELETON = "k"
TILE_COW = "c"

# Full-version mob tiles
TILE_SKELETON_ARCHER = "a"  # upstream ranged skeleton glyph (used by full env)
TILE_KOBOLD = "q"           # upstream kobold
TILE_BAT = "b"
TILE_BOSS = "W"

# Plant tile characters
TILE_SAPLING = ";"
TILE_RIPE_PLANT = "*"

# Projectile tile characters (phase α — single-codepoint, disjoint from
# existing tile/mob/direction palettes). T26 of the phase-α plan.
# Dir chars use →←↑↓ (U+2192/90/91/93); these are all distinct.
TILE_ARROW = "↗"     # ↗ diagonal NE arrow
TILE_ARROW2 = "↘"    # ↘ diagonal SE arrow
TILE_DAGGER = "†"    # † dagger
TILE_FIREBALL = "●"  # ● black circle
TILE_FIREBALL2 = "◉" # ◉ fish-eye
TILE_ICEBALL = "○"   # ○ white circle
TILE_ICEBALL2 = "◎"  # ◎ bullseye
TILE_SLIMEBALL = "◐" # ◐ half black circle (left)

# Phase β ore tile characters (T05β). Both are single-codepoint and disjoint
# from the existing palette above.
TILE_SAPPHIRE = "♦"  # ♦ U+2666 black diamond suit — sapphire ore
TILE_RUBY = "▲"      # ▲ U+25B2 black up-pointing triangle — ruby ore

# Phase β chest tile (T12β). "$" is single-codepoint and disjoint from the
# existing palette.
TILE_CHEST = "$"  # $ dollar sign — chest (interactable loot container)

# Phase β fountain tile (T18β). "⊙" U+2299 is single-codepoint and disjoint
# from the existing palette.
TILE_FOUNTAIN = "⊙"  # ⊙ U+2299 circled dot — dungeon fountain (refills water)

# Phase β enchantment-table tiles (T20β). "Ⓔ" U+24BA (circled E) and
# "Ⓘ" U+24BE (circled I) are both single-codepoint and disjoint from the
# existing palette. Phase γ will wire the enchant-table semantics.
# Floor 3 (Sewers) hosts the ice table; floor 4 (Vaults) hosts the fire table.
TILE_ENCHANT_FIRE = "Ⓔ"  # Ⓔ U+24BA circled latin capital letter E — fire enchantment table (floor 4)
TILE_ENCHANT_ICE  = "Ⓘ"  # Ⓘ U+24BE circled latin capital letter I — ice enchantment table (floor 3)

# Phase γ biome tree tiles (T13-T15γ). Both are single-codepoint and disjoint
# from the existing palette above.
TILE_FIRE_TREE = "♠"  # ♠ U+2660 black spade suit — fire tree (floor 6)
TILE_ICE_SHRUB = "❄"  # ❄ U+2744 snowflake — ice shrub (floor 7)

# Phase γ mob tiles for floors 5-7 (T13-T15γ). All single-codepoint and disjoint.
TILE_GNOME_WARRIOR = "G"    # G — gnome warrior (floor 2 melee)
TILE_GNOME_ARCHER = "g"     # g — gnome archer (floor 2 ranged)
TILE_ORC_SOLDIER = "O"      # O — orc soldier (floor 1 melee)
TILE_ORC_MAGE = "W"         # W — orc mage (floor 1 ranged; shared with floor-boss mobs)
TILE_LIZARD = "L"           # L — lizard (floor 3 melee)
TILE_KNIGHT = "K"           # K — knight (floor 4 melee)
TILE_KNIGHT_ARCHER = "a"    # a — knight archer (floor 4 ranged; shared archer glyph)
TILE_TROLL = "T"            # T — troll (floor 5 melee)
TILE_DEEP_THING = "d"       # d — deep thing (floor 5 ranged)
TILE_SNAIL = "s"            # s — snail (passive)
TILE_PIGMAN = "p"           # p — pigman (floor 6 melee)
TILE_FIRE_ELEMENTAL = "F"   # F — fire elemental (floor 6 ranged)
TILE_FROST_TROLL = "r"      # r — frost troll (floor 7 melee)
TILE_ICE_ELEMENTAL = "i"    # i — ice elemental (floor 7 ranged)

# Phase γ floor-8 Graveyard tiles (T16γ). All single-codepoint and disjoint.
# TILE_GRAVE: tombstone decoration tile for the Graveyard biome.
# TILE_NECROMANCER / TILE_NECROMANCER_VULNERABLE: the boss tile (changes glyph
# to signal vulnerability — used by the renderer so the agent can read state).
TILE_GRAVE = "⚰"              # ⚰ U+26B0 coffin — grave marker decoration
TILE_NECROMANCER = "N"         # N — necromancer (invulnerable / summoning)
TILE_NECROMANCER_VULNERABLE = "n"  # n — necromancer (vulnerable — can be hit)

# Craftax achievement namespace (91 entries; phase beta: 3 potion entries -> 1 unified)
ALL_FULL_ACHIEVEMENTS = (
    # -- Surface progression achievements --
    "collect_wood",
    "place_table",
    "make_wood_pickaxe",
    "collect_stone",
    "place_furnace",
    "make_stone_pickaxe",
    "collect_iron",
    "collect_coal",
    "place_stone",
    "collect_drink",
    "collect_sapling",
    "place_plant",
    "eat_plant",
    "defeat_zombie",
    "defeat_skeleton",
    "wake_up",
    "collect_diamond",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_wood_sword",
    "make_stone_sword",
    "eat_cow",
    # -- Diamond tier (3) --
    "make_diamond_pickaxe",
    "make_diamond_sword",
    # -- Armor (4) --
    "make_wood_armor",
    "make_stone_armor",
    "make_iron_armor",
    "make_diamond_armor",
    # -- Magic (2) --
    "cast_fireball",
    "cast_iceball",
    # -- Potions (1; phase beta: unified drink_potion replaces the old color split) --
    "drink_potion",
    # -- Enchantments (3; T10-T12γ: enchant_weapon split to sword/armor/bow) --
    "enchant_sword",
    "enchant_armor",
    "enchant_bow",
    # -- Dungeon progression (9; phase γ +3 for floors 6/7/8) --
    "enter_dungeon",
    "reach_floor_2",
    "reach_floor_3",
    "reach_floor_4",
    "reach_floor_5",
    "enter_fire_realm",
    "enter_ice_realm",
    "enter_graveyard",
    "return_to_surface",
    # -- Bosses (5) --
    "defeat_knight",
    "defeat_archer_boss",
    "defeat_mage",
    "defeat_dragon",
    "defeat_lich",
    # -- New mobs (2) --
    "defeat_kobold",
    "eat_bat",
    # -- Boss loot (5) --
    "collect_boss_loot_1",
    "collect_boss_loot_2",
    "collect_boss_loot_3",
    "collect_boss_loot_4",
    "collect_boss_loot_5",
    # -- Survival milestones (2) --
    "survive_10_nights",
    "survive_20_nights",
    # -- Stat milestones (2) --
    "full_health",
    "full_mana",
    # -- Kill milestones (3) --
    "kill_10_mobs",
    "kill_25_mobs",
    "kill_50_mobs",
    # -- Craft milestones (2) --
    "craft_10_items",
    "craft_25_items",
    # -- Misc milestones (3) --
    "place_10_blocks",
    "eat_5_plants",
    "drink_10_water",
    # -- Exploration milestones (4) --
    "explore_all_floors",
    "collect_all_boss_loot",
    "learn_all_spells",
    "max_inventory_wood",
    # -- Combat milestones (4) --
    "kill_5_mobs",
    "survive_5_nights",
    "defeat_boss_no_armor",
    "clear_dungeon_floor",
    # -- Resource milestones (3) --
    "collect_10_wood",
    "collect_5_stone",
    "collect_3_iron",
    # -- Phase α additions (3) --
    "make_arrow",
    "fire_bow",
    # -- Phase β gem ores (2) --
    "collect_sapphire",
    "collect_ruby",
    # -- Phase β spells + items (3) --
    "learn_fireball",
    "learn_iceball",
    "find_book",
    # -- Phase β chest system (2; T12-T14β) --
    "open_chest",
    "find_bow",
    # -- Phase γ attribute levelling (3; T08γ) --
    "level_up_dexterity",
    "level_up_strength",
    "level_up_intelligence",
    # -- Phase γ necromancer boss kill (1; T17γ) --
    "defeat_necromancer",
    # -- Torch/lighting system (2; 2026-06 port-vs-fork alignment) --
    # Restored to match the fork: torches are how the agent sees underground.
    "make_torch",
    "place_torch",
    # -- Total: 93 (91 + make_torch/place_torch).
)

# Visible window dimensions
VIEW_WIDTH = 9
VIEW_HEIGHT = 7

# Larger viewport used by multi-floor Craftax scenarios.
FULL_VIEW_WIDTH = 11
FULL_VIEW_HEIGHT = 9


CRAFTAX_MECHANICS_PROMPT = (
    "Craftax mechanics means the upstream Craftax rules from the bundled "
    "Craftax submodule: movement and facing, DO interactions, walkability, "
    "blocking and collision, HP, food, drink, energy, mana, day/night, mobs, "
    "combat, projectiles, drops, resources, crafting, placement, plants, "
    "water, lava, inventory/equipment state, achievements, rewards, floor "
    "transitions and gates, bows/arrows, books, spells, potions, enchantments, "
    "bosses, and terminal conditions. Registered glyphbench/craftax-* tasks "
    "use the universal Craftax action menu; actions that are irrelevant or "
    "blocked by a focused TASK/SCENARIO setup remain part of the action space "
    "and are unavailable only because of that scenario state, not because this "
    "is a separate Craftax variant."
)

CRAFTAX_FOCUSED_SUBTASK_PROMPT = (
    "This is a focused GlyphBench subtask, not a claim that the whole upstream "
    "Craftax game is present. The TASK block is authoritative for the goal, "
    "reward, max-step budget, and any documented scaffold. Smaller maps, "
    "curated starts, fixed inventory, disabled drains, scripted waves, and "
    "shorter horizons are valid only when stated here; otherwise movement, "
    "facing, DO interactions, crafting, combat, mobs, drops, survival ticks, "
    "floor transitions, terminal conditions, and reward accounting should "
    "follow the Craftax mechanics above."
)


def craftax_prompt_contract(
    *,
    focused_subtask: bool = False,
) -> str:
    """Return the single source of truth for Craftax prompt mechanics wording."""
    parts = ["CRAFTAX MECHANICS", CRAFTAX_MECHANICS_PROMPT]
    if focused_subtask:
        parts.extend(("", "FOCUSED SUBTASK CONTRACT", CRAFTAX_FOCUSED_SUBTASK_PROMPT))
    return "\n".join(parts)


class _CraftaxTutorialMixin:
    """Mixin providing the canonical craftax system_prompt template.

    Subclasses must define:
      - `tutorial_sections: tuple[str, ...]` — anchor names from
        `glyphbench.envs.craftax.docs.ALL_SECTIONS`.
      - `_task_description(self) -> str` — one short paragraph stating the
        env-specific goal and reward shape.
      - `env_id(self) -> str` and `action_spec` — provided by BaseGlyphEnv.

    Output format (matches the glyphbench standard):
        "You are playing X.\n\nTASK\n<goal>\n\n<composed tutorial>\n\n<actions>"
    """

    tutorial_sections: tuple[str, ...] = ()
    craftax_focused_subtask: bool = False

    def _task_description(self) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must override _task_description()"
        )

    def system_prompt(self) -> str:  # type: ignore[override]
        from glyphbench.envs.craftax.docs import compose

        if not self.tutorial_sections:
            raise RuntimeError(
                f"{type(self).__name__}: tutorial_sections is empty; "
                "every craftax env must declare its tutorial slice."
            )
        env_id = self.env_id()
        focused_subtask = bool(self.craftax_focused_subtask) or env_id.startswith(
            "glyphbench/craftax-"
        )
        mechanics_block = (
            craftax_prompt_contract(focused_subtask=focused_subtask) + "\n\n"
        )
        scenario_notes: list[str] = []
        if getattr(self, "_disable_survival", False):
            scenario_notes.append(
                "Survival drains are DISABLED in this scenario: food, drink, "
                "and energy do not decrease over time. The HUD shows "
                "`Next drain: disabled` to confirm this. Ignore the "
                "drain-rate tables in the survival section below."
            )
        if getattr(self, "_disable_day_night", False):
            scenario_notes.append(
                "Day/night progression is DISABLED in this scenario: there "
                "is no nightfall, no night-spawned mobs, and time-of-day "
                "never advances. The HUD shows `Time: disabled` to confirm "
                "this. Ignore the day/night section below."
            )
        gated_floors = tuple(getattr(self, "_GATED_FLOORS", ()))
        if gated_floors:
            floors = ", ".join(str(floor) for floor in gated_floors)
            scenario_notes.append(
                "Focused target mobs on floor(s) "
                f"{floors} do not despawn when they move far from the "
                "player. The global despawn rule still applies elsewhere."
            )
        scenario_block = ""
        if scenario_notes:
            scenario_block = (
                "SCENARIO CONSTRAINTS\n"
                + "\n".join(f"- {note}" for note in scenario_notes)
                + "\n\n"
            )
        return (
            f"You are playing {env_id}.\n\n"
            f"TASK\n{self._task_description()}\n\n"
            f"{mechanics_block}"
            f"{scenario_block}"
            f"{compose(self.tutorial_sections)}\n\n"
            f"{self.action_spec.render_for_prompt()}"
        )
