# Craftax — Magic

> Canonical anchors (locked API): `magic:books`, `magic:spells`, `magic:enchants`.

The agent can learn two spells: fireball and iceball. Spells must be learned (via READ_BOOK) before they can be cast. Each spell costs 2 mana and fires a single-target traveling projectile.

<!-- :section magic:books -->
## Books and READ_BOOK

Books are quest items that teach spells. They are not craftable; they appear from chest grants:

- Floor 3 first chest: grants 1 book (achievement `find_book`).
- Floor 4 first chest: grants 1 book if floor 3's first chest was not yet opened.

Books are stored in `inventory["book"]`.

**READ_BOOK** action:
- Consumes 1 book from inventory.
- Randomly teaches one **unlearned** spell (fireball or iceball).
- If both spells are already known, the book is consumed with no benefit. Check `_learned_spells` before reading.

Achievements: `learn_fireball`, `learn_iceball` fire on first learning. There is no way to obtain additional books once first-chest grants are exhausted.
<!-- :end -->

<!-- :section magic:spells -->
## CAST_FIREBALL and CAST_ICEBALL

| Spell | Action | Mana cost | Element | Player projectile glyph |
|---|---|---|---|---|
| Fireball | CAST_FIREBALL | 2 | Fire | `●` |
| Iceball | CAST_ICEBALL | 2 | Ice | `○` |

Both spells:
- Spawn a projectile at the player's tile facing in the current direction.
- Travel 1 tile per turn.
- Collision is checked at pre-advance and post-advance positions each tick.
- Are consumed on hitting a mob, a solid tile, or leaving bounds.

CAST_FIREBALL is gated on `_learned_spells["fireball"]`. CAST_ICEBALL is gated on `_learned_spells["iceball"]`.

**Damage scaling:** spell damage × `(1.0 + 0.5 × (INT - 1))` (fork `magic_damage_coeff`). Base fireball and iceball damage are both **3**.

| INT | Multiplier |
|---|---|
| 1 | 1.0× |
| 2 | 1.5× |
| 3 | 2.0× |
| 4 | 2.5× |
| 5 | 3.0× |

The scaled damage is applied as `(0, dmg, 0)` (fire) or `(0, 0, dmg)` (ice). Per-mob fire/ice immunity (defense = 1.0) blocks this damage entirely. Use fireball against ice-realm mobs (frost_troll, ice_elemental); use iceball against fire-realm mobs (pigman, fire_elemental). Physical attacks are nearly ineffective on floors 6 and 7.

**Mana management:**
- Base max mana: 10. INT 5 → 22.
- Regen: 1 per 20 steps × `(1.0 + 0.25 × (INT - 1))`. INT 5 → 1 per 10 steps.
- A spell costs 2 mana; an enchant costs 9 mana.
<!-- :end -->

<!-- :section magic:enchants -->
## Enchanting

Enchanting requires facing an enchantment table, a gemstone, and 9 mana.

| Action | Inputs | Prerequisite | Effect |
|---|---|---|---|
| ENCHANT_WEAPON | 1 ruby (fire) OR 1 sapphire (ice) + 9 mana | facing `Ⓔ` (floor 4) OR `Ⓘ` (floor 3) | Sets sword enchantment to fire or ice |
| ENCHANT_ARMOR | 1 ruby OR 1 sapphire + 9 mana | facing `Ⓔ` OR `Ⓘ` | Adds fire/ice enchant to lowest unenchanted armor slot |
| ENCHANT_BOW | 1 ruby OR 1 sapphire + 9 mana | facing `Ⓔ` OR `Ⓘ` (bow required in inventory) | Sets bow enchantment to fire or ice |

- Ruby (`▲`) → fire enchantment.
- Sapphire (`♦`) → ice enchantment.
- Enchanted sword adds `0.5 × physical_damage` in the enchanted element.
- Each enchanted armor slot grants 0.2 resistance to the matching element.
- Enchanted bow adds the elemental component to fired arrows.

Achievements: `enchant_sword`, `enchant_armor`, `enchant_bow`. Floor 6 (Fire Realm) requires ice enchants/spells; floor 7 (Ice Realm) requires fire enchants/spells.
<!-- :end -->
