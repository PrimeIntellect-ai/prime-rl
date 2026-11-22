# Craftax — Crafting

> Canonical anchors (locked API): `crafting:wood`, `crafting:stone`, `crafting:iron`, `crafting:diamond`, `crafting:placement`, `crafting:arrows`.

Most crafting actions require a nearby placed crafting table (`t`) in the 8-neighborhood, including diagonals. Iron-tier also requires a nearby furnace (`f`) in that same 8-neighborhood. Enchanting is different: face the enchantment table (`Ⓔ` fire or `Ⓘ` ice) directly.

<!-- :section crafting:placement -->
## Placement actions

| Action | Cost | Prerequisite | Result |
|---|---|---|---|
| PLACE_STONE | 1 stone | none | Places `=` (placed stone) in faced cell |
| PLACE_TABLE | 2 wood | none | Places `t` (crafting table) |
| PLACE_FURNACE | 1 stone | none | Places `f` (furnace) |
| PLACE_PLANT | 1 sapling | none | Places `;` sapling; ripens to `*` after 20 steps |
<!-- :end -->

<!-- :section crafting:wood -->
## Wood-tier crafting

Wood-tier crafting requires a crafting table in the 8-neighborhood.

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_WOOD_PICKAXE | 1 wood | nearby table | Wood pickaxe (mines stone; tier 0) |
| MAKE_WOOD_SWORD | 1 wood | nearby table | Wood sword (+1 melee bonus) |

Tier 0 pickaxes can mine stone and coal. To mine iron, craft a stone pickaxe.
<!-- :end -->

<!-- :section crafting:stone -->
## Stone-tier crafting

Stone-tier crafting requires a crafting table in the 8-neighborhood.

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_STONE_PICKAXE | 1 wood + 1 stone | nearby table | Stone pickaxe (mines iron; tier 1) |
| MAKE_STONE_SWORD | 1 wood + 1 stone | nearby table | Stone sword (+2 melee bonus) |

Stone tier requires **only** a crafting table (no furnace needed). Iron and above need both.
<!-- :end -->

<!-- :section crafting:iron -->
## Iron-tier crafting

Iron-tier crafting requires both a crafting table and a furnace in the 8-neighborhood.

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_IRON_PICKAXE | 1 wood + 1 stone + 1 iron + 1 coal | nearby table + furnace | Iron pickaxe (mines diamond; tier 2) |
| MAKE_IRON_SWORD | 1 wood + 1 stone + 1 iron + 1 coal | nearby table + furnace | Iron sword (melee damage 5) |

Iron pickaxe gates **diamond** ore; sapphire and ruby need a **diamond pickaxe**. Stock coal before iron-tier crafting; iron tools consume coal and stone as recipe inputs.
<!-- :end -->

<!-- :section crafting:diamond -->
## Diamond-tier crafting

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_DIAMOND_PICKAXE | 1 wood + 3 diamond | nearby table | Diamond pickaxe (tier 3; mines sapphire + ruby) |
| MAKE_DIAMOND_SWORD | 1 wood + 2 diamond | nearby table | Diamond sword (melee damage 8) |
| MAKE_DIAMOND_ARMOR | 3 diamond | nearby table | Diamond armor (tier 2; fills or upgrades lowest slot) |

Diamond armor pieces upgrade existing iron-tier slots when all 4 slots are already filled.

**Resource mining prerequisites:**

| Ore | Minimum pickaxe tier |
|---|---|
| Stone | 0 (any pickaxe) |
| Coal | 1 (stone+) |
| Iron | 1 (stone+) |
| Diamond | 2 (iron+) |
| Sapphire (`♦`) | 2 (iron+) |
| Ruby (`▲`) | 2 (iron+) |
<!-- :end -->

<!-- :section crafting:arrows -->
## Arrow crafting

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_ARROW | 1 wood + 1 stone | nearby table | 2 arrows |

Arrows are consumed one per SHOOT_ARROW. The bow itself is **not crafted** — it drops from the first chest opened on floor 1. See `items:bow`.
<!-- :end -->

<!-- :section crafting:torches -->
## Torch crafting (lighting)

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_TORCH | 1 wood + 1 coal | nearby table | 4 torches |
| PLACE_TORCH | 1 torch | face an open floor cell | Lights a ~4 radius around that cell |

Underground floors can be **dark**: cells outside your current light render as
blank spaces, and mobs/projectiles in unlit cells are hidden. You arrive on a
dark floor inside a small lit pocket around the up-ladder, and lava glows.
Craft torches at a table (carry wood + coal down) and `PLACE_TORCH` as you move
to keep vision. The HUD shows `Torches: N` and a `Light:` field. Chests can also
contain torches. The surface (floor 0) is always fully lit in the local view.
<!-- :end -->
