# Craftax — Boss: The Necromancer

> Canonical anchors (locked API): `boss`.

<!-- :section boss -->
## Location and context

The Necromancer is the final boss, located exclusively on **floor 8 (Graveyard)**. Floor 8 has no STAIRS_DOWN; the player cannot leave once the fight begins. Food, drink, and energy do **not** decay on floor 8 — vitals are frozen for the encounter.

The Necromancer is rendered as a tile in the floor grid:
- `N` (TILE_NECROMANCER): invulnerable — hitting it has no effect.
- `n` (TILE_NECROMANCER_VULNERABLE): vulnerable — DO while adjacent and facing it registers a hit.

This explicit visual signal is the primary way the agent should determine when to attack.

## State machine

The Necromancer has two state variables:

- `_boss_progress` (int, 0–8): number of successful vulnerable hits landed.
- `_boss_summon_timer` (int): turns remaining before the Necromancer can become vulnerable again.

### Vulnerability condition

The Necromancer is vulnerable **if and only if both**:
1. `_boss_summon_timer <= 0` (no active summon window), AND
2. No hostile (non-passive) mobs are alive on floor 8.

Passive mobs (cow, bat, snail) do not block vulnerability. The check excludes the Necromancer tile itself.

### On a successful hit

When the player performs DO while facing the vulnerable Necromancer (`n`):
1. `_boss_progress += 1`
2. `_boss_summon_timer = 7` (7-turn summon window begins)
3. The Necromancer becomes invulnerable (`N`) until conditions are met again.

### During the 7-turn summon window

Each turn while `_boss_summon_timer > 0`, wave mobs spawn near the player:
- Up to **3 melee mobs** (zombies) and **2 ranged mobs** (skeletons) spawn.
- The cap prevents the arena from becoming infinitely crowded.
- `_boss_summon_timer` decrements by 1 per turn.

The player must kill all summoned mobs before the Necromancer becomes vulnerable again.

### Attacking while invulnerable

If DO is used on the invulnerable Necromancer (`N`), the hit is **ignored** — no damage, no effect, no boss_progress increment. Verify the glyph is `n` before attacking.

## Win condition

When `_boss_progress >= 8` (8 successful hits landed):
- The `defeat_necromancer` achievement is unlocked.
- This is the game win state.
Focused subtasks may replace the native achievement reward with their own
bounded reward contract.

## Mob despawn exemption

On floor 8, melee and ranged mobs are **exempt from the normal despawn rule** (distance ≥ 14). Summoned waves persist until killed. This forces the player to engage the waves rather than kiting mobs away.

## Boss-floor damage multiplier

All incoming damage on floor 8 is multiplied by **1.5×** (applied after armor reduction). If the player is also sleeping, the combined multiplier is 3.5 × 1.5 = **5.25×**. Do not sleep on floor 8.

## Recommended approach

1. Arrive fully prepared: diamond armor (all 4 slots), enchanted sword and/or bow, both spells learned, full potions.
2. Descend to floor 8. Locate the Necromancer tile.
3. Kill all mobs on the floor. Once the floor is clear and `_boss_summon_timer = 0`, the glyph will change to `n`.
4. Move adjacent to `n`, face it, and perform DO to land a hit.
5. A 7-turn wave of zombies and skeletons spawns. Kill them all.
6. Wait for `_boss_summon_timer` to expire and no mobs remain. The glyph returns to `n`.
7. Repeat steps 4–6 **8 times total** to win.

Each wave is manageable individually. The danger is being overwhelmed if vitals are low or armor is insufficient. Potions and REST can be used between waves. Since vitals do not decay on floor 8, there is no time pressure between waves.
<!-- :end -->
