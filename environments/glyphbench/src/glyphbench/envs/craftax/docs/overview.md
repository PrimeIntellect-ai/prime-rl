# Craftax — Overview

> Canonical anchors (locked API): `overview`.

<!-- :section overview -->
Craftax is a 9-floor survival crafting game. The agent gathers resources, crafts tools and weapons, explores progressively harder dungeon floors, learns magic, and ultimately defeats the Necromancer on floor 8. Each floor is harder than the last; floor 8 is the final boss floor with no exit.

The `CRAFTAX MECHANICS` block near the top of each system prompt is the mechanics contract. Registered `glyphbench/craftax-*` tasks use the universal Craftax action menu. Focused GlyphBench subtasks may use curated starts, smaller maps, scripted waves, disabled drains, fixed inventory, or shorter horizons only when the TASK or SCENARIO blocks say so; otherwise mechanics should follow upstream Craftax.

## Floor stack

| Floor | Name | Biome |
|---|---|---|
| 0 | Overworld | Smoothgen 64x64 |
| 1 | Dungeon | Dungeon-rooms 32x32 |
| 2 | Gnomish Mines | Smoothgen 32x32 |
| 3 | Sewers | Dungeon-rooms 32x32 |
| 4 | Vaults | Dungeon-rooms 32x32 |
| 5 | Troll Mines | Smoothgen 32x32 |
| 6 | Fire Realm | Smoothgen 32x32 |
| 7 | Ice Realm | Smoothgen 32x32 |
| 8 | Graveyard | Open boss arena 32x32 |

See `floors.md` for per-floor mob rosters, special tiles, and transition rules.

## Action structure

The full action spec has 43 named actions. Key categories:

- **Movement**: MOVE_LEFT / MOVE_RIGHT / MOVE_UP / MOVE_DOWN
- **Interaction**: DO (face target and interact: mine, attack, open chest, drink, eat)
- **Placement**: PLACE_STONE / PLACE_TABLE / PLACE_FURNACE / PLACE_PLANT
- **Crafting**: MAKE_WOOD/STONE/IRON/DIAMOND_PICKAXE; MAKE_WOOD/STONE/IRON/DIAMOND_SWORD; MAKE_IRON_ARMOR; MAKE_DIAMOND_ARMOR; MAKE_ARROW
- **Combat**: SHOOT_ARROW; CAST_FIREBALL; CAST_ICEBALL; ENCHANT_WEAPON; ENCHANT_ARMOR; ENCHANT_BOW
- **Survival**: SLEEP; REST; EAT_PLANT; DRINK_WATER; DRINK_POTION_RED/GREEN/BLUE/PINK/CYAN/YELLOW
- **Exploration**: DESCEND; ASCEND; READ_BOOK
- **Progression**: LEVEL_UP_DEXTERITY; LEVEL_UP_STRENGTH; LEVEL_UP_INTELLIGENCE; NOOP

## Reward

- For focused GlyphBench subtasks, the TASK block at the top of the prompt is the authoritative reward contract.
- In the base Craftax environment, first-time achievements give normalized reward and death yields terminal `-1.0`.
- If this overview conflicts with the TASK block, follow the TASK block.

## Win and death

- **Death**: HP reaches 0. Focused subtasks may override the exact terminal reward in the TASK block.
- **Base CraftaxFull win**: `boss_progress >= 8` (8 hits on the vulnerable Necromancer). Focused subtasks define their own win condition in the TASK block.

## Observation conventions

Each turn the agent receives:

- **Grid**: Unicode text viewport centered on the agent. Focused surface tasks commonly use 9x7; full multi-floor tasks commonly use 11x9. Each cell is one Unicode codepoint.
- **Legend**: per-turn glyph key listing every glyph visible in the current grid (see `legend.md` for the full palette).
- **HUD**: Step plus current vitals, location/progress counters, inventory,
  armor, spells, potions, and task-specific progress fields where relevant.
  Focused subtasks may omit fields that cannot change in that task.
- **Message**: last game message (achievement unlocks, damage events, crafting results, etc.).

The agent selects one action name per turn. Invalid or blocked actions usually no-op; the Message field may explain why an action failed.
<!-- :end -->
