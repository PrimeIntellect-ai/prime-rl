# Craftax — Survival

> Canonical anchors (locked API): `survival:hp_food_drink`, `survival:energy_sleep`, `survival:rest`, `survival:mana`, `survival:day_night`.

The agent has five vitals: HP, Food, Drink (water), Energy, and Mana. Base environments start at maximum vitals; focused subtasks may override the starting values in the TASK block and HUD. Full Craftax caps are re-evaluated at runtime as attributes change.

<!-- :section survival:hp_food_drink -->
## HP, Food, Drink

| Vital | Base max | Scales with |
|---|---|---|
| HP | 9 | STR (+1 per point above 1) |
| Food | 9 | DEX (+2 per point above 1) |
| Drink | 9 | DEX (+2 per point above 1) |

**Need decay:**

| Vital | Drain interval (base) |
|---|---|
| Food | 1 per 50 steps |
| Drink | 1 per 40 steps |

DEX scaling: decay multiplier `1.0 - 0.125 × (DEX - 1)`. At DEX 5 the multiplier is 0.5.

**Consequences of depletion:**
- Food = 0 → -1 HP per step.
- Drink = 0 → -1 HP per step.
- HP can deplete from multiple sources simultaneously (food + drink both 0 = -2 HP/step).

**Recovery:**
- Eat ripe plants (`*`) via EAT_PLANT for +4 food; the tile becomes a growing plant again.
- Eat passive mob meat (kill cow `c` or bat `b` with DO) for +6 food.
- Drink water (`≈`) via DRINK_WATER for +1 drink.
- Drink at fountains (`⊙`, dungeon floors) via DO.

**Lava:** `♨` is walkable and standing on it deals 2 HP per tick regardless of armor. Focused scenarios may add a terminal death/reward wrapper around the same lava damage when the TASK block says so.
<!-- :end -->

<!-- :section survival:energy_sleep -->
## Energy and SLEEP

Energy drains 1 per 100 steps (base). DEX scaling matches food/drink (× `1.0 - 0.125 × (DEX - 1)`). Energy = 0 → 50% chance any move action fails.

**Action SLEEP** (action index 6):
1. If Energy is below max, the agent enters sleep and GlyphBench fast-forwards the inert sleep ticks inside this single action until an exit condition fires. The next observation is after wake/interruption; do not spend follow-up actions waiting.
2. Per-tick while sleeping:
   - +2 HP (capped at max HP).
   - +2 Energy (capped at max Energy).
3. Exit conditions:
   - Energy reaches max → normal wake (fires `wake_up` achievement).
   - Incoming damage → sleep is cancelled immediately. No `wake_up`.
4. **Vulnerability:** while sleeping, all incoming damage is multiplied by **3.5×** (after armor reduction, before boss-floor multiplier). Do not sleep in open areas with mobs nearby; block yourself in with placed stone first.
5. Combined multiplier on floor 8: sleeping × boss = **3.5 × 1.5 = 5.25×**.

Food and Drink continue to drain while sleeping.

Focused scenarios use the same SLEEP action semantics unless the SCENARIO block explicitly disables survival or day/night ticks.
<!-- :end -->

<!-- :section survival:rest -->
## REST

Action REST (action index 33):
1. If HP is below max, the agent enters rest and GlyphBench fast-forwards the inert no-op ticks inside this single action until an exit condition fires. The next observation is after wake/interruption; do not spend follow-up actions waiting.
2. Per-tick while resting: +1 HP (capped at max HP).
3. Exit conditions (any):
   - HP reaches max.
   - Food reaches 0.
   - Drink reaches 0.
   - Incoming damage (mob hit, lava, etc.).
4. REST does **not** restore energy. Use SLEEP for energy; use REST for HP top-up when energy is already full.
5. While resting the 3.5× sleep damage multiplier does **not** apply.
<!-- :end -->

<!-- :section survival:mana -->
## Mana

- Base max: **10**. Scales with INT: `+3 per INT point above 1` (INT 5 → max 22).
- Regen: 1 per 20 steps × `(1.0 + 0.25 × (INT - 1))`. INT 5 → 1 per 10 steps.
- Costs:
  - Spells (CAST_FIREBALL, CAST_ICEBALL): 2 mana each.
  - Enchanting (ENCHANT_WEAPON, ENCHANT_ARMOR, ENCHANT_BOW): 9 mana each.

See `magic.md` and `crafting.md` for spell + enchant details.
<!-- :end -->

<!-- :section survival:day_night -->
## Day/night and visibility (floor 0)

- Day length: 200 steps. Night length: 100 steps. Cycle: 300 steps total.
- The surface (floor 0) is fully lit in the local view, day and night — there is no surface darkness.
- Nighttime changes mob spawning and survival pressure, not the surface viewport rules.
- Night spawns use the floor-0 hostile roster for the active Craftax implementation. Sleeping in the open at night is dangerous.

Dungeon floors do **not** have day/night cycles, but they **can be dark**: cells
outside your current light render as blank spaces and unlit mobs are hidden. You
start lit near the up-ladder; lava glows; craft (`MAKE_TORCH`) and place
(`PLACE_TORCH`) torches to see. See the Floors section for details.

Note: floor 8 vitals (food, drink, energy) do **not** decay during the boss encounter.
<!-- :end -->
