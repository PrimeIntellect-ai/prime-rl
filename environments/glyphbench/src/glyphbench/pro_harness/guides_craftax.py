"""Craftax domain guide for the Pro harness.

The guide is based on the original fork harness
(``third_party/Craftax/examples/run_azure_agent.py`` + ``agent_memory.py``),
which the author validated works well. Inline glyph hints are remapped from
the ASCII glyph set to glyphbench's Unicode observations. A small number of
mechanics-grounded survival reminders are included for recurrent long-horizon
failure modes; they remain prompt guidance rather than action-selection code.

``glyphbench/craftaxfull-v0`` is the JAX fork (real upstream Craftax). The Pro
harness renders its observation as a strict 1:1 Unicode beautification of the
fork's original ascii local view (``UNICODE_CHAR_TABLE`` is indexed by the same
char-ids as the ascii ``CHAR_TABLE``). So the Unicode glyphs used in the obs and
referenced here line up exactly:
  ``♣`` tree, ``S`` stone, ``≈`` water, ``▼`` ladder-down, ``△`` ladder-up,
  ``⊙`` fountain, ``$`` chest, ``♨`` lava, ``N`` necromancer, ``Ⓔ`` enchant
  table, ``⚰`` grave, ``c`` cow, ``s`` snail, ``b`` bat, ``←→↑↓`` you (facing).
Action names (``LEFT``/``RIGHT``/``UP``/``DOWN``/``DO``/…) are verbatim; the
fork aliases these to its ``MOVE_*`` canonical names.

Only the memory *format* differs from the original (this harness uses the
two-call action/memory split, described in the PRO HARNESS PROTOCOL), so the
original single-call "--- / Action:" output block is intentionally omitted;
the landmark selectivity + grounding rules below are kept verbatim.
"""

from __future__ import annotations

# --- from run_azure_agent.py:GENERAL_GUIDE (glyph hints remapped to Unicode) -
GENERAL_GUIDE = """\
# Craftax Game Guide

## Goal
Descend through all 9 floors (0-8) and defeat the Necromancer on floor 8.

Craftax is a roguelike survival game. You control a character (shown as ←→↑↓ indicating your facing direction) on a grid map.

## Core Mechanics
- You have 5 intrinsics: HP, Food, Drink, Energy, Mana.
- Food, Drink, Energy decay over time. Replenish by eating (kill cows/eat ripe plants), drinking (face water + DO), sleeping (SLEEP restores HP when food/drink/energy > 0).
- HP recovers when food, drink AND energy are all > 0.  If HP reaches 0 you die.
- Mana recovers naturally; used for spells and enchanting.
- To interact with the block you are FACING, use DO.  You must be adjacent and facing toward it.
- To face a direction, move in that direction first (LEFT/RIGHT/UP/DOWN).  If blocked, you turn without moving.
- Your absolute position Pos:(row,col) is shown in the status bar. Use it to navigate and record locations.
- The map is 48x48 (rows 0-47, cols 0-47). Blank spaces at the edges are out of bounds — you cannot go there. Turn around!
- When eating or drinking, repeat DO until the stat is FULL. Leaving with partial food/drink is risky.

## Progression
- Mine trees (face tree + DO) to get wood. A crafting table costs 2 wood;
  collect at least 3 wood before placing it so 1 wood remains for a pickaxe.
  PLACE_TABLE puts the table on the empty tile you face. Stand adjacent to the
  visible table, then craft tools.
- Craft pickaxe to mine stone/ores.  Place furnace (PLACE_FURNACE) to smelt iron.
- Kill 8 monsters per floor to unlock the ladder down (▼). Stand on ▼ and DESCEND.
- Each new floor awards XP to spend on Dexterity, Strength, or Intelligence.

## Attributes
- Dexterity: increases max food/drink/energy, slows decay, increases bow damage.
- Strength: increases melee damage and max HP.
- Intelligence: increases max mana, spell/enchantment damage.

## Potions
- Found in chests. 6 colours, effects randomized each game (+-health, +-mana, +-energy).
- Experiment carefully!

## Day/Night Cycle
- The overworld has a day/night cycle shown in [Status] as Day, Dusk, or Night.
- At night, more hostile mobs spawn on the overworld. Be cautious or sleep through it.
- Dark cavern floors are unlit regardless of time — you need torches (PLACE_TORCH).
- Torches are crafted from 1 wood + 1 coal (MAKE_TORCH).

## Deep-floor survival discipline
- Armour protection is cumulative. Each MAKE_IRON_ARMOUR action crafts exactly
  one next missing piece for 3 iron + 3 coal; one helm is only partial
  protection. Before sustained combat in the Gnomish Mines, deliberately craft
  several iron pieces (ideally all four when resources permit), rather than
  spending every ore on tools or descending with a single piece.
- Never SLEEP exposed in an open room or while a hostile or projectile lane is
  visible: sleeping multiplies incoming damage by 2.5. First seek a naturally
  narrow cavity, corner, or one-tile corridor. Use PLACE_STONE to close the
  remaining one or two entrances, verify that the shelter is enclosed and
  quiet, then SLEEP. If no useful natural shelter is nearby, build a compact
  four-sided stone enclosure around yourself instead. Mine an exit after
  waking if necessary.
- The Gnomish Mines are the third level (Floor 2) and are truly dark. Carry a
  reserve of torches, place one before advancing beyond the lit area, and add
  well-spaced lights at new corridors/intersections. Confirm that the torch was
  consumed and terrain became visible; explore the revealed area cautiously
  instead of walking or fighting into darkness.

## Combat Tips
- Armour significantly reduces damage taken. Iron armour helps a lot; diamond
  armour (3 diamonds per piece) is even better. Worth crafting when you can.
- Diamond sword (1 wood + 2 diamonds) deals the most melee damage in the game.
- Important: you take 2.5x damage while asleep. Consider blocking entry points with
  PLACE_STONE or using natural walls/corners/tunnels before sleeping.
- Craft arrows (MAKE_ARROW) for ranged combat with your bow (SHOOT_ARROW).

## Crafting Reference
Crafting is automatic — use the crafting ACTION while standing adjacent to the required
station(s) with the right materials in your inventory. There is no smelting step.
Raw ores (iron, coal, diamond, etc.) are used directly in recipes. Do not use DO on a crafting
table or furnace — that will destroy it. Only use the specific MAKE_* action keys.
- PLACE_TABLE costs 2 wood and needs an empty faced tile. PLACE_FURNACE costs 1 stone.
- Adjacent to crafting table: MAKE_WOOD_PICKAXE (1 wood), MAKE_WOOD_SWORD (1 wood)
- MAKE_STONE_PICKAXE (1 wood + 1 stone), MAKE_STONE_SWORD (1 wood + 1 stone)
- Adjacent to crafting table AND furnace: MAKE_IRON_PICKAXE (1 wood + 1 stone + 1 iron + 1 coal), MAKE_IRON_SWORD (1 wood + 1 stone + 1 iron + 1 coal)
- MAKE_DIAMOND_PICKAXE (1 wood + 3 diamonds), MAKE_DIAMOND_SWORD (1 wood + 2 diamonds)
- MAKE_IRON_ARMOUR (3 iron + 3 coal per piece), MAKE_DIAMOND_ARMOUR (3 diamonds per piece)
- MAKE_ARROW (1 wood + 1 stone), MAKE_TORCH (1 wood + 1 coal)
- Enchanting: adjacent to enchantment table, 9 mana + sapphire (ice) or ruby (fire)
- All crafting/enchanting actions work instantly if you have materials + correct adjacency.

## Action-result grounding (critical)
- An action is successful ONLY when the next observation confirms it: inventory
  is consumed/gained, [Equip] changes, a station appears in the grid, an
  achievement unlocks, or the relevant state changes.
- Zero reward does not itself imply failure, but an unchanged HUD/grid does.
  Never mark a table/tool/furnace as placed or crafted merely because you sent
  its action. If state is unchanged, diagnose the missing material, adjacency,
  facing, or empty target and adapt immediately.
- Add a placed-station landmark only after the station is visibly confirmed.
"""

# --- from run_azure_agent.py:FLOOR_GUIDES (glyph hints remapped to Unicode) --
FLOOR_GUIDES: dict[int, str] = {
    0: """\
## Floor 0: Overworld (Current Floor)
Grasslands, lakes and mountains.  Trees (♣) give wood, stone (S) needs a pickaxe.
- PRIORITY: collect at least 3 wood -> place crafting table (2 wood) -> craft wood pickaxe (1 wood) -> mine stone -> craft stone tools -> mine coal/iron -> craft iron tools + iron armour.
- Progress promptly: once you have a stone pickaxe, stone sword, >=4 spare
  wood, and safe food/drink/energy, locating the open ladder and descending is
  the main objective. Do not overfarm common stone, but keep collecting useful
  coal and iron: tools consume some of each and multiple armour pieces are a
  key safety investment before sustained combat on floor 2. Floor 2 is richer
  in ore, so it is reasonable to finish the set from a lit base there.
- Eat cows (c) for food, drink from water (≈), sleep when energy is low.
- Collect saplings from grass (DO on grass) and plant them (PLACE_PLANT) for future food.
- Collect wood before descending - it's scarce underground.
- Find the ladder down (▼) and DESCEND when ready.
- The overworld ladder is always open (no kill requirement).""",
    1: """\
## Floor 1: Dungeon (Current Floor)
Connected rooms with paths.  Contains fountains (⊙) and chests ($).
- First chest gives a bow.  Craft arrows (MAKE_ARROW) at a crafting table.
- Kill 8 orcs/mages to unlock the ladder down (▼).
- Snails (s) can be eaten for food.
- Drink from fountains (⊙) by facing them + DO.
- If low HP or energy: move into a narrow room/corridor, seal its one or two
  approaches with stone, verify no hostile has access, then SLEEP or REST.
- Before descending, check armour and torch reserves. A single iron piece is
  weak preparation for sustained combat on the darker, more dangerous floor 2.""",
    2: """\
## Floor 2: Gnomish Mines (Current Floor)
Dark cavern - you MUST place torches (PLACE_TORCH) to see. Craft torches from wood + coal.
- Establish a lit, defensible foothold near the ladder before ranging outward.
  Place lights ahead at useful spacing, inspect each newly revealed area, and
  avoid pushing into darkness while injured or pursued.
- Rich in ores: coal, iron, diamonds, sapphires, rubies along cavern edges.
- Bats (b) can be eaten.  Pools of water (≈) for drinking.
- Gnomes are stronger than orcs - avoid being surrounded in open spaces.
- PRIORITY: craft diamond sword (1 wood + 2 diamonds) for massive damage, then diamond pickaxe (1 wood + 3 diamonds) to mine sapphires/rubies.
- Craft iron armour repeatedly (3 iron + 3 coal per piece) or diamond armour
  (3 diamonds per piece). One action makes only one armour piece; aim for
  several pieces before sustained gnome combat. Mine from the lit foothold and
  return to a table + furnace as resources accumulate.""",
    3: """\
## Floor 3: Sewers (Current Floor)
Dungeon layout with water patches.  Fill water gaps by placing stone then mining it.
- Lizards swim through water - very dangerous!  Kobolds throw high-damage daggers.
- First chest contains a BOOK - read it (READ_BOOK) to learn fireball or iceball spell.
- Spells cost 2 mana (CAST_FIREBALL / CAST_ICEBALL).  They do fire/ice damage.
- Later enemies resist physical damage - spells become essential!
- ICE enchantment table (Ⓔ) is on this floor.  Enchant items with sapphires (9 mana + sapphire).
- Enchanting sword/bow: +50% ice damage.  Enchanting armour: -20% ice damage per piece.""",
    4: """\
## Floor 4: Vaults (Current Floor)
Another dungeon level.  Contains a second book and the FIRE enchantment table (Ⓔ).
- Knights and archers are armoured - physical damage is HALVED against them.
- Use enchanted weapons or spells (fire/ice) to deal effective damage.
- Enchant with rubies at the fire enchantment table (9 mana + ruby).""",
    5: """\
## Floor 5: Troll Mines (Current Floor)
Dark cavern, richest in ores.  Place torches to see.
- PRIORITY: Mine everything - craft full diamond armour if possible.
- Trolls are very strong.  Deep things in water are weak but have deadly ranged attacks.
- Rich in coal, iron, diamonds, sapphires, rubies.""",
    6: """\
## Floor 6: Fire Realm (Current Floor)
Islands separated by lava (♨).  No water on this level!
- Build bridges: PLACE_STONE on lava, then mine the stone to create a path.
- Pigmen and fire elementals are immune to fire damage and resist physical.
- ICE damage is required (ice spells or ice-enchanted weapons).
- Periodically ASCEND to Troll Mines to drink water.
- Lots of coal and rubies available.""",
    7: """\
## Floor 7: Ice Realm (Current Floor)
Dark level - place torches.  Frost trolls and ice elementals are the strongest creatures.
- They are immune to ice damage - FIRE damage required.
- No food on this level - ASCEND to Fire Realm to eat.
- Lots of sapphires and rubies in the mountains.
- This is the last floor before the boss!""",
    8: """\
## Floor 8: Graveyard - BOSS LEVEL (Current Floor)
The necromancer (N) is the final boss.  No ladder out.  No food/water, but intrinsics don't decay.
- The necromancer summons waves of enemies from graves (⚰).
- Kill all enemies in a wave -> necromancer becomes VULNERABLE (N turns yellow).
- Attack the vulnerable necromancer (face + DO) to trigger the next wave.
- Waves correspond to creatures from each floor (floor 0 through floor 7).
- After defeating the final wave (ice realm), attack the necromancer to WIN!""",
}

# --- from run_azure_agent.py:ACTION_SPACE_TEXT (ladder glyphs remapped) ------
ACTION_SPACE_TEXT = """\
## Available Actions
Movement: LEFT, RIGHT, UP, DOWN (also turns you to face that direction)
Interact: DO (mine block / attack creature / drink water / eat plant / open chest - in your facing direction)
Rest: SLEEP (restores energy; HP recovers while asleep if food/drink/energy>0; wakes at max energy or if attacked; 2.5x damage while asleep)
      REST (idles until HP is full, food=0, drink=0, or attacked; normal damage)
Place: PLACE_STONE, PLACE_TABLE, PLACE_FURNACE, PLACE_PLANT, PLACE_TORCH
Craft (adjacent to table): MAKE_WOOD_PICKAXE, MAKE_STONE_PICKAXE, MAKE_IRON_PICKAXE, MAKE_DIAMOND_PICKAXE
  MAKE_WOOD_SWORD, MAKE_STONE_SWORD, MAKE_IRON_SWORD, MAKE_DIAMOND_SWORD
  MAKE_IRON_ARMOUR, MAKE_DIAMOND_ARMOUR, MAKE_ARROW, MAKE_TORCH
Combat: SHOOT_ARROW, CAST_FIREBALL, CAST_ICEBALL
Spells: READ_BOOK, ENCHANT_SWORD, ENCHANT_ARMOUR, ENCHANT_BOW
Potions: DRINK_POTION_RED, DRINK_POTION_GREEN, DRINK_POTION_BLUE, DRINK_POTION_PINK, DRINK_POTION_CYAN, DRINK_POTION_YELLOW
Navigate: DESCEND (go down ladder ▼), ASCEND (go up ladder △)
Level up: LEVEL_UP_DEXTERITY, LEVEL_UP_STRENGTH, LEVEL_UP_INTELLIGENCE
Do nothing: NOOP"""

# Landmark selectivity + grounding — verbatim from agent_memory.MEMORY_INSTRUCTIONS
# (the "what to record" + grounding parts; the single-call output-format block is
# omitted because this harness uses the two-call action/memory split). MAP
# commands are keyed by `floor=` (the parser also accepts area=/level=).
LANDMARK_INSTRUCTIONS = """\
The landmark map is a persistent (floor, type, row, col) database for KEY
locations. Use MAP_ADD / MAP_GET / MAP_REMOVE on the memory turn, keyed by
floor:
  MAP_ADD: floor=0, type=water, row=20, col=30, note=large lake
  MAP_ADD: floor=1, type=crafting_table, row=12, col=8, note=my table
  MAP_GET: floor=0                 (recall all saved locations on floor 0)
  MAP_GET: floor=0, type=water     (recall only water on floor 0)
  MAP_REMOVE: floor=0, type=chest, row=12, col=8

What to record (only high-value persistent landmarks):
  YES: water sources, crafting tables YOU placed, furnaces YOU placed, ladders,
       enchantment tables, chests, ore deposits (iron/diamond/sapphire/ruby)
  NO:  trees (too common), cows/mobs (they move), grass, stone (everywhere),
       coal (too common), individual tiles you've walked on
Memory is limited per floor — be selective. Use MAP_REMOVE when a resource is
consumed (ore mined, chest opened). Do not re-add a landmark you already saved.

Grounding rules:
- The map view is ALWAYS the ground truth. Read it FIRST before consulting memory.
- Your spatial memory or scratchpad may be outdated — verify against the map.
- Use your recent-action history to detect loops. If repeating actions, change strategy."""
