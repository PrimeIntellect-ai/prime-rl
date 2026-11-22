# Craftax — Legend

> Canonical anchors (locked API): `legend:player`, `legend:terrain`, `legend:mobs:overworld`, `legend:mobs:dungeon`, `legend:mobs:boss`, `legend:items`, `legend:projectiles`, `legend:hud`.

The text renderer uses a single Unicode codepoint per grid cell. Color encoding is not used; the glyph alone identifies the tile.

<!-- :section legend:player -->
## Player

The agent is rendered as a directional arrow indicating its current facing direction:

| Glyph | Facing |
|---|---|
| `→` | facing right (east) |
| `←` | facing left (west) |
| `↑` | facing up (north) |
| `↓` | facing down (south) |
| `@` | fallback if facing is undefined (rare) |

Reading the arrow tells you which cell DO, and where available SHOOT_ARROW, will target. MOVE_LEFT/RIGHT/UP/DOWN both move and update facing.
<!-- :end -->

<!-- :section legend:terrain -->
## Terrain tiles

Surface terrain (floor 0):

| Glyph | Name |
|---|---|
| `·` | Grass |
| `♣` | Tree (choppable) |
| `S` | Stone (mineable) |
| `C` | Coal ore |
| `I` | Iron ore |
| `D` | Diamond ore |
| `≈` | Water |
| `♨` | Lava |
| `░` | Sand |
| `=` | Placed stone |
| `;` | Sapling (growing) |
| `*` | Ripe plant (edible) |

Dungeon terrain (floors 1+):

| Glyph | Name |
|---|---|
| `█` | Dungeon wall |
| `▪` | Dungeon floor |

Biome decorations (per floor):

| Glyph | Name | Floor |
|---|---|---|
| `♠` | Fire tree (DO for wood) | 6 |
| `❄` | Ice shrub (DO for wood) | 7 |
| `⚰` | Grave marker (decoration) | 8 |
<!-- :end -->

<!-- :section legend:mobs:overworld -->
## Overworld mobs (floor 0)

| Glyph | Name | Behavior |
|---|---|---|
| `z` | Zombie | melee, hostile |
| `c` | Cow | passive, edible |
| `a` | Skeleton archer | ranged (arrows), hostile |
<!-- :end -->

<!-- :section legend:mobs:dungeon -->
## Dungeon mobs (floors 1–7)

| Glyph | Name | Behavior | Floor |
|---|---|---|---|
| `q` | Kobold | ranged (daggers), hostile | 3 |
| `b` | Bat | passive, edible | 2/5/6/7 |
| `s` | Snail | passive | 1/3/4 |
| `T` | Troll | melee, hostile | 5 |
| `d` | Deep thing | ranged (slimeballs), hostile | 5 |
| `p` | Pigman | melee, hostile, fire-immune | 6 |
| `F` | Fire elemental | ranged (fireballs), hostile, fire-immune | 6 |
| `r` | Frost troll | melee, hostile, ice-immune | 7 |
| `i` | Ice elemental | ranged (iceballs), hostile, ice-immune | 7 |
| `W` | Orc mage / floor boss | ranged or boss-specific, hostile | 1 / boss floors |

Note: gnome_warrior, gnome_archer (floor 2), orc_soldier, orc_mage (floor 1), lizard, knight, knight_archer (floor 4) use upstream tile mappings from `mechanics/mobs.py`.
<!-- :end -->

<!-- :section legend:mobs:boss -->
## Boss mob (floor 8)

| Glyph | Name | Notes |
|---|---|---|
| `N` | Necromancer (invulnerable) | DO has no effect |
| `n` | Necromancer (vulnerable) | DO while adjacent + facing lands a hit |

The Necromancer's glyph alternates between `N` and `n` as the boss state machine advances. See `boss.md`.
<!-- :end -->

<!-- :section legend:items -->
## Special / interactive tiles and items

| Glyph | Name |
|---|---|
| `⇣` | Stairs down |
| `⇡` | Stairs up |
| `t` | Crafting table |
| `f` | Furnace |
| `B` | Boss door |
| `$` | Chest (lootable) |
| `⊙` | Fountain (refills water) |
| `Ⓔ` | Fire enchantment table (floor 4) |
| `Ⓘ` | Ice enchantment table (floor 3) |
| `♦` | Sapphire ore |
| `▲` | Ruby ore |
<!-- :end -->

<!-- :section legend:projectiles -->
## Projectile tiles

| Glyph | Name | Source |
|---|---|---|
| `↗` | Arrow (variant 1) | player or skeleton/gnome_archer |
| `↘` | Arrow (variant 2) | knight_archer |
| `†` | Dagger | kobold |
| `●` | Fireball (variant 1) | player CAST_FIREBALL or orc_mage |
| `◉` | Fireball (variant 2) | fire_elemental |
| `○` | Iceball (variant 1) | player CAST_ICEBALL |
| `◎` | Iceball (variant 2) | ice_elemental |
| `◐` | Slimeball | deep_thing (mixed phys/fire/ice damage) |

Projectiles travel 1 tile/turn; collision is checked at pre-advance and post-advance positions. See `combat.md` (`combat:projectiles`).
<!-- :end -->

<!-- :section legend:hud -->
## HUD fields

The HUD shows the following fields every turn:

| Field | Description |
|---|---|
| HP | Current / max hit points |
| Food | Food level (0 = HP drains) |
| Drink | Water level (0 = HP drains) |
| Energy | Energy (0 = 50% move failure) |
| Mana | Mana for spells and enchanting |
| XP | Unspent experience points |
| DEX | Dexterity attribute (1-5) |
| STR | Strength attribute (1-5) |
| INT | Intelligence attribute (1-5) |
| Floor | Current floor number (0-8) |
| Light | Day/Dusk/Night; underground floors are flagged `dark` here (blank cells = unlit, use torches) |
| Torches | Torches carried (MAKE_TORCH / PLACE_TORCH) |
| Step | Turn counter formatted `T / N` |
| Inventory | Per-item counts |
| Armor | Per-slot tier and enchant element |
| Achievements | Achievement count (X / N total) |

For per-floor tile details see `floors.md`. For attribute scaling see `progression.md`.
<!-- :end -->
