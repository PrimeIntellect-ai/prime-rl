# Craftax — Floors

> Canonical anchors (locked API): `floors:0`, `floors:1`, `floors:2`, `floors:3`, `floors:4`, `floors:5`, `floors:6`, `floors:7`, `floors:8`, `floors:navigation`.

<!-- :section floors:navigation -->
## Floor navigation

- **DESCEND** (action: stand on `⇣` stair-down tile): travel to the next floor.
- **ASCEND** (action: stand on `⇡` stair-up tile): return to the previous floor.
- First DESCEND to each floor grants **+1 XP** (max 8 XP across floors 1-8).
- Achievements fire on first entry: `enter_dungeon` (floor 1), `reach_floor_2`, `reach_floor_3`, `reach_floor_4`, `reach_floor_5`, `enter_fire_realm` (floor 6), `enter_ice_realm` (floor 7), `enter_graveyard` (floor 8). `return_to_surface` fires on first ASCEND back to floor 0.
- Each dungeon floor has a stair-up tile to return upward, **except floor 8** (no exit; final floor).

**Chest first-grant rules:**
- Floor 1 first chest opened → 1 bow (achievement `find_bow`).
- Floor 3 first chest opened → 1 book (achievement `find_book`).
- Floor 4 first chest opened → 1 book (only if floor 3 chest unopened).
- All other chests roll standard loot (wood, coal, iron, diamond, potions, arrows, iron pickaxe, iron sword, torches).

**Lighting & torches (underground):**
- The surface is always fully lit in the local view. Underground floors can be **dark**: any cell whose light is below the visibility threshold renders as a blank space, and mobs/projectiles standing in dark cells are invisible until lit.
- You always arrive on a dark floor inside a small lit pocket around the up-ladder. **Lava glows**, lighting nearby cells.
- Craft torches with `MAKE_TORCH` (1 wood + 1 coal at a crafting table → 4 torches) and drop them with `PLACE_TORCH` on the cell you face. Each torch lights a radius of ~4. Chests can also contain torches.
- The HUD shows `Torches: N` and a `Light:` field. Bring wood + coal underground so you can keep making torches.
<!-- :end -->

<!-- :section floors:0 -->
## Floor 0: Overworld

- **Biome**: smoothgen, 64×64.
- **Visibility**: local viewport. The surface is fully lit day and night in the local view (no darkness on floor 0). Floor 0 still has the day/night cycle (200-step day, 100-step night), which controls night pressure (more mobs spawn at night).
- **Mob roster**: cow (`c`, passive), zombie (`z`, melee), skeleton archer (`a`, ranged).
- **Special tiles**: STAIRS_DOWN (`⇣`) to floor 1 in full-floor scenarios; focused surface scenarios may omit stairs when the TASK block says the scenario stays on floor 0.
- **Resources**: grass, trees (`♣`), stone (`S`), coal (`C`), iron (`I`), diamond (`D`), water (`≈`), sand (`░`).
- **Strategy**: chop trees → table → wood pickaxe → stone pickaxe → iron pickaxe. Iron tools need coal. Eat cows (DO while facing) for food. Drink at water tiles. SLEEP only works when tired and still lets mobs act during the skipped sleep ticks.
- **Notes**: Overworld is the only floor with diamonds on the surface and the only floor where saplings can be grown. No sapphire or ruby here.
<!-- :end -->

<!-- :section floors:1 -->
## Floor 1: Dungeon

- **Biome**: dungeon-rooms (32×32), 8 rooms connected by L-shaped corridors.
- **Visibility**: local viewport. This floor may be **dark** — cells outside your current light render as blank spaces; you start in a small lit area around the up-ladder, lava glows, and you can `MAKE_TORCH` (1 wood + 1 coal at a table → 4 torches) then `PLACE_TORCH` to light a ~4 radius. Mobs/projectiles in unlit cells are hidden until lit. The HUD `Light:` field flags dark floors; the TASK block is authoritative for this scenario.
- **Mob roster**: snail (`s`, passive, edible), orc_soldier (melee), orc_mage (ranged, fires fireballs).
- **Special tiles**: STAIRS_UP (`⇡`) to floor 0; STAIRS_DOWN (`⇣`) to floor 2; chests (`$`); fountains (`⊙`).
- **First-chest grant**: 1 bow + `find_bow` achievement.
- **Fountains**: refill water when interacted with (DO while facing `⊙`).
- **Notes**: Orc mages fire fireball projectiles; they kite. Collect wood from the surface before descending — wood is scarce underground.
<!-- :end -->

<!-- :section floors:2 -->
## Floor 2: Gnomish Mines

- **Biome**: smoothgen (cave-like), 32×32.
- **Visibility**: local viewport. This floor may be **dark** — cells outside your current light render as blank spaces; you start in a small lit area around the up-ladder, lava glows, and you can `MAKE_TORCH` (1 wood + 1 coal at a table → 4 torches) then `PLACE_TORCH` to light a ~4 radius. Mobs/projectiles in unlit cells are hidden until lit. The HUD `Light:` field flags dark floors; the TASK block is authoritative for this scenario.
- **Mob roster**: bat (`b`, passive, edible), gnome_warrior (melee), gnome_archer (ranged, fires arrows).
- **Special tiles**: STAIRS_UP (`⇡`) to floor 1; STAIRS_DOWN (`⇣`) to floor 3.
- **Resources**: coal, iron, diamond, sapphire (`♦` at 2.5%), ruby (`▲` at 2.5%). Sapphire/ruby placement is mutually exclusive with diamond/iron/coal on eligible tiles.
- **Ore gating**: diamond needs an iron pickaxe; sapphire and ruby need a **diamond pickaxe**.
- **Strategy**: bring a diamond pickaxe to mine sapphire/ruby (iron pickaxe only mines up to diamond ore). Mine cave edges for richer ore deposits.
- **Notes**: Gnomes are stronger than floor 1 mobs. Avoid being surrounded in open smoothgen spaces. Bat meat restores food.
<!-- :end -->

<!-- :section floors:3 -->
## Floor 3: Sewers

- **Biome**: dungeon-rooms, 32×32.
- **Visibility**: local viewport. This floor may be **dark** — cells outside your current light render as blank spaces; you start in a small lit area around the up-ladder, lava glows, and you can `MAKE_TORCH` (1 wood + 1 coal at a table → 4 torches) then `PLACE_TORCH` to light a ~4 radius. Mobs/projectiles in unlit cells are hidden until lit. The HUD `Light:` field flags dark floors; the TASK block is authoritative for this scenario.
- **Mob roster**: snail (`s`, passive), lizard (melee), kobold (`q`, ranged, fires daggers).
- **Special tiles**: STAIRS_UP (`⇡`) to floor 2; STAIRS_DOWN (`⇣`) to floor 4; chests; fountains; **ICE enchantment table** (`Ⓘ`).
- **First-chest grant**: 1 book + `find_book` achievement.
- **Enchanting**: face `Ⓘ` with 1 sapphire + 9 mana for ice enchantments (ENCHANT_WEAPON, ENCHANT_ARMOR, ENCHANT_BOW).
- **Notes**: Kobolds throw daggers (physical projectiles). Lizards are strong melee. The ice enchant table is critical for floor 7 prep.
<!-- :end -->

<!-- :section floors:4 -->
## Floor 4: Vaults

- **Biome**: dungeon-rooms, 32×32.
- **Visibility**: local viewport. This floor may be **dark** — cells outside your current light render as blank spaces; you start in a small lit area around the up-ladder, lava glows, and you can `MAKE_TORCH` (1 wood + 1 coal at a table → 4 torches) then `PLACE_TORCH` to light a ~4 radius. Mobs/projectiles in unlit cells are hidden until lit. The HUD `Light:` field flags dark floors; the TASK block is authoritative for this scenario.
- **Mob roster**: snail (`s`, passive), knight (melee, 0.5 phys defense), knight_archer (ranged, 0.5 phys defense, fires arrow2 projectiles).
- **Special tiles**: STAIRS_UP (`⇡`) to floor 3; STAIRS_DOWN (`⇣`) to floor 5; chests; fountains; **FIRE enchantment table** (`Ⓔ`).
- **First-chest grant**: 1 book (only if floor 3 first chest is unopened).
- **Enchanting**: face `Ⓔ` with 1 ruby + 9 mana for fire enchantments.
- **Knight defense**: knight and knight_archer have 0.5 physical defense. Pure physical melee deals half damage. Use enchanted weapons, spells, or enchanted bows.
<!-- :end -->

<!-- :section floors:5 -->
## Floor 5: Troll Mines

- **Biome**: smoothgen cave, 32×32.
- **Visibility**: local viewport. This floor may be **dark** — cells outside your current light render as blank spaces; you start in a small lit area around the up-ladder, lava glows, and you can `MAKE_TORCH` (1 wood + 1 coal at a table → 4 torches) then `PLACE_TORCH` to light a ~4 radius. Mobs/projectiles in unlit cells are hidden until lit. The HUD `Light:` field flags dark floors; the TASK block is authoritative for this scenario.
- **Mob roster**: bat (`b`, passive), troll (`T`, melee, 0.2 phys defense), deep_thing (`d`, ranged, no defense, fires slimeballs).
- **Special tiles**: STAIRS_UP (`⇡`) to floor 4; STAIRS_DOWN (`⇣`) to floor 6.
- **Resources**: sapphire (`♦`), ruby (`▲`) at low density. Best floor to accumulate ore for diamond armor and enchanting.
- **Strategy**: craft full diamond armor here. Stock both sapphire (ice enchant) and ruby (fire enchant). Deep things fire slimeball projectiles (mixed phys/fire/ice; armor only partially mitigates).
- **Notes**: Trolls are slow but hit hard. Deep things kite from range 4-5. Bring arrows and food.
<!-- :end -->

<!-- :section floors:6 -->
## Floor 6: Fire Realm

- **Biome**: smoothgen with lava patches, 32×32.
- **Visibility**: local viewport. This floor may be **dark** — cells outside your current light render as blank spaces; you start in a small lit area around the up-ladder, lava glows, and you can `MAKE_TORCH` (1 wood + 1 coal at a table → 4 torches) then `PLACE_TORCH` to light a ~4 radius. Mobs/projectiles in unlit cells are hidden until lit. The HUD `Light:` field flags dark floors; the TASK block is authoritative for this scenario.
- **Mob roster**: bat (`b`, passive), pigman (`p`, melee, 0.9 phys + 1.0 fire immune), fire_elemental (`F`, ranged, 0.9 phys + 1.0 fire immune, fires fireball2).
- **Special tiles**: STAIRS_UP (`⇡`) to floor 5; STAIRS_DOWN (`⇣`) to floor 7; fire trees (`♠`, DO for wood); lava (`♨`).
- **Resources**: ruby (`▲` at 2.5%), coal; **no water** on this floor.
- **Lava**: 2 HP per tick on lava tiles. Build stone bridges (PLACE_STONE) to cross. Return to floor 5 for water.
- **Immunity**: fire-realm mobs are immune to fire (1.0) and 90% resistant to physical. **Iceball + ice-enchanted weapons mandatory.** Sapphire enchants must be prepared in advance (floor 3 ice table or floor 5 sapphire stockpile).
- **Notes**: no food source except bat meat. Stockpile food before entering.
<!-- :end -->

<!-- :section floors:7 -->
## Floor 7: Ice Realm

- **Biome**: smoothgen with water pools, 32×32.
- **Visibility**: local viewport. This floor may be **dark** — cells outside your current light render as blank spaces; you start in a small lit area around the up-ladder, lava glows, and you can `MAKE_TORCH` (1 wood + 1 coal at a table → 4 torches) then `PLACE_TORCH` to light a ~4 radius. Mobs/projectiles in unlit cells are hidden until lit. The HUD `Light:` field flags dark floors; the TASK block is authoritative for this scenario.
- **Mob roster**: bat (`b`, passive), frost_troll (`r`, melee, 0.9 phys + 1.0 ice immune), ice_elemental (`i`, ranged, 0.9 phys + 1.0 ice immune, fires iceball2).
- **Special tiles**: STAIRS_UP (`⇡`) to floor 6; STAIRS_DOWN (`⇣`) to floor 8; ice shrubs (`❄`, DO for wood); water (`≈`).
- **Resources**: sapphire (`♦`); some ruby; **no food source**.
- **Immunity**: ice-realm mobs are immune to ice (1.0) and 90% resistant to physical. **Fireball + fire-enchanted weapons mandatory.** Ruby enchants must be prepared in advance (floor 4 fire table or floor 5 ruby stockpile).
- **Water**: water tiles are walkable and drinkable (DRINK_WATER while facing `≈`). This is the last floor before the boss.
<!-- :end -->

<!-- :section floors:8 -->
## Floor 8: Graveyard (Boss Floor)

- **Biome**: smoothgen, 32×32.
- **Visibility**: local viewport. This floor may be **dark** — cells outside your current light render as blank spaces; you start in a small lit area around the up-ladder, lava glows, and you can `MAKE_TORCH` (1 wood + 1 coal at a table → 4 torches) then `PLACE_TORCH` to light a ~4 radius. Mobs/projectiles in unlit cells are hidden until lit. The HUD `Light:` field flags dark floors; the TASK block is authoritative for this scenario.
- **Special mob**: Necromancer (`N` invulnerable / `n` vulnerable). This is the win-condition target.
- **Wave-summoned mobs**: zombies (`z`) and skeletons (`a`) spawn near the player after each Necromancer hit.
- **Special tiles**: grave markers (`⚰`, decoration). No STAIRS_DOWN; no STAIRS_UP — floor 8 has no exit.
- **Survival**: food, drink, and energy do **not** decay on floor 8. Vitals are frozen for the boss encounter.
- **Boss-floor damage multiplier**: all incoming damage × **1.5** (after armor reduction).
- **Despawn**: melee/ranged mobs are exempt from the normal 14-tile despawn rule during the boss fight.
- **Strategy**: arrive with full diamond armor, enchanted weapons, and learned spells. See `boss.md` for the Necromancer state machine.
<!-- :end -->
