# Craftax — Progression

> Canonical anchors (locked API): `progression:xp`, `progression:attributes`, `progression:achievements`.

<!-- :section progression:xp -->
## Experience points (XP)

XP is earned by descending to new floors:

- **+1 XP** on first DESCEND to each new floor (floors 1 through 8).
- Maximum XP from floor entries: **8 XP** (one per dungeon floor).
- XP is tracked by `_xp_floors_visited`; revisiting a floor does not grant XP.

Unspent XP accumulates and is shown in the HUD. Each LEVEL_UP_* action spends 1 XP.
<!-- :end -->

<!-- :section progression:attributes -->
## Attributes (DEX, STR, INT)

Three RPG attributes: DEX (Dexterity), STR (Strength), INT (Intelligence). Each starts at **1** and caps at **5**.

Spend 1 XP per level-up via LEVEL_UP_DEXTERITY, LEVEL_UP_STRENGTH, or LEVEL_UP_INTELLIGENCE. Assignments are permanent within the episode.

**Per-attribute scaling formulas:**

| Attribute | Effect | Formula |
|---|---|---|
| DEX | Max Food / Drink / Energy | base 9 + (DEX - 1) × 2 |
| DEX | Need-decay rate | × (1 - 0.125 × (DEX - 1)) |
| DEX | Arrow damage | × (1 + 0.2 × (DEX - 1)) |
| STR | Max HP | base 9 + (STR - 1) |
| STR | Melee physical damage | × (1 + 0.25 × (STR - 1)) |
| INT | Max Mana | base 10 + (INT - 1) × 3 |
| INT | Mana regen rate | × (1 + 0.25 × (INT - 1)) |
| INT | Spell damage | × (1 + 0.05 × (INT - 1)) |

**Cap values (level 5):**

- DEX 5: max food/drink/energy = 17 each; decay = 0.5×; arrow dmg = 1.8×.
- STR 5: max HP = 13; melee dmg = 2.0×.
- INT 5: max mana = 22; regen = 2.0× (1 per 10 steps); spell dmg = 1.2×.

Achievements: `level_up_dexterity`, `level_up_strength`, `level_up_intelligence` fire on first level-up of each attribute.

**Recommended XP splits** (8 XP available total, all 3 attrs start at 1):
- DEX 3, STR 2, INT 3 = 8 XP: balanced survival + damage.
- DEX 2, STR 2, INT 4 = 8 XP: magic-focused (max INT 5 needs 4 XP).
- STR 5, INT 1, DEX 2 = 8 XP: melee focus (only viable with enchanted weapons for floors 6-7).

Reaching floor 8 requires surviving floor 7 (ice realm), which requires fire damage. Pure STR builds will struggle without enchanted weapons.
<!-- :end -->

<!-- :section progression:achievements -->
## Achievements

Achievements are the primary reward signal. Each first-time unlock grants **+1/N reward**, where `N` is the achievement count for the env.

- Craftax env: achievements span surface, dungeon, magic, combat, crafting, lighting (torches), and boss progression. The HUD shows the live total.
- Defeating the Necromancer unlocks `defeat_necromancer`; there is no extra win bonus.

Achievements span: resource collection (wood, stone, iron, diamond, sapphire, ruby), tool crafting (each tier of pickaxe/sword/armor), placement (table, furnace, plant, stone), survival (sleep, eat, drink), combat kills (zombie, skeleton, kobold, bat, knight, archer_boss, mage, dragon, lich, necromancer), magic (cast_fireball, cast_iceball, learn_*), enchantments (sword/armor/bow), dungeon entry (each floor), survival milestones (X nights, full HP/mana), and exploration milestones (explore_all_floors, collect_all_boss_loot).

The HUD shows the achievement count `X / N` (N is the env's live total). Each new unlock is also surfaced via the [Message] field on the turn it fires.
<!-- :end -->
