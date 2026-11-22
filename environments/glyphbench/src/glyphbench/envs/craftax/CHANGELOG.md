# Craftax Current Notes

This file records the current behavioral contract for the maintained
GlyphBench Craftax port.

## Current Task Set

- The old numbered floor task IDs were removed.
- Named floor task IDs are:
  - `glyphbench/craftax-dungeon-v0`
  - `glyphbench/craftax-gnomish-mines-v0`
  - `glyphbench/craftax-sewers-v0`
  - `glyphbench/craftax-vaults-v0`
  - `glyphbench/craftax-troll-mines-v0`
  - `glyphbench/craftax-fire-realm-v0`
  - `glyphbench/craftax-ice-realm-v0`
- Focused crafting tasks no longer start adjacent to a prebuilt solution.
  They require the relevant gather/place/craft chain from an empty inventory.
- Named floor descent tasks require at least eight non-boss kills on the
  current floor before `DESCEND` opens the stairs.
- Every Craftax subtask reset is deterministic for a given seed and varies
  the initial layout across different seeds.
- Focused tasks now cover book-to-spell learning, weapon enchanting, bow
  kiting, and hidden potion triage:
  - `glyphbench/craftax-learn-spell-v0`
  - `glyphbench/craftax-enchant-weapon-v0`
  - `glyphbench/craftax-bow-kite-v0`
  - `glyphbench/craftax-potion-triage-v0`

## Visibility

- Visibility is always on for every in-bounds cell.
- Torch actions, torch inventory, placed torch tiles, and the lightmap
  subsystem were removed.
- Night still affects survival pressure and night mob spawning.

## Rewards

- Registered `glyphbench/craftax-*` tasks share the 91-entry Craftax
  achievement namespace.
- Each upstream achievement reward is scaled by `+1/91`; all upstream
  achievement rewards sum to `+1.0` where achievement rewards are enabled.
- Death yields terminal `-1.0`.
- Defeating the necromancer unlocks `defeat_necromancer`; there is no extra
  win bonus.

## Action Spec

- Registered `glyphbench/craftax-*` tasks expose the 43-action universal
  Craftax action menu.
- Removed historical actions include `MAKE_TORCH`, `PLACE_TORCH`,
  `MAKE_WOOD_ARMOR`, and `MAKE_STONE_ARMOR`.
