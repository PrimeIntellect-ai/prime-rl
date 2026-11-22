"""Craftax mob rosters and per-turn movement and combat helpers."""
from __future__ import annotations

import random
from collections.abc import Callable
from typing import TypedDict

from glyphbench.envs.craftax.mechanics.projectiles import ProjectileType

PASSIVE_MOB_NAMES: tuple[str, ...] = ("cow", "bat", "snail")

MELEE_MOB_NAMES: tuple[str, ...] = (
    "zombie", "gnome_warrior", "orc_soldier", "lizard",
    "knight", "troll", "pigman", "frost_troll",
)

RANGED_MOB_NAMES: tuple[str, ...] = (
    "skeleton", "gnome_archer", "orc_mage", "kobold",
    "knight_archer", "deep_thing", "fire_elemental", "ice_elemental",
)


# Per-floor roster: which mob name spawns in each class on each floor.
# Indices are resolved from upstream FLOOR_MOB_MAPPING (constants.py:138-152)
# using the texture-load order as the name index:
#   passive[0]=cow, [1]=bat, [2]=snail
#   melee[0]=zombie, [1]=gnome_warrior, [2]=orc_soldier, [3]=lizard,
#         [4]=knight, [5]=troll, [6]=pigman, [7]=frost_troll
#   ranged[0]=skeleton, [1]=gnome_archer, [2]=orc_mage, [3]=kobold,
#          [4]=knight_archer, [5]=deep_thing, [6]=fire_elemental, [7]=ice_elemental
FLOOR_MOB_MAPPING: tuple[dict[str, str], ...] = (
    # 0 — Overworld: [0, 0, 0]
    {"passive": "cow", "melee": "zombie", "ranged": "skeleton"},
    # 1 — Dungeon: [2, 2, 2]
    {"passive": "snail", "melee": "orc_soldier", "ranged": "orc_mage"},
    # 2 — Gnomish Mines: [1, 1, 1]
    {"passive": "bat", "melee": "gnome_warrior", "ranged": "gnome_archer"},
    # 3 — Sewers: [2, 3, 3]
    {"passive": "snail", "melee": "lizard", "ranged": "kobold"},
    # 4 — Vaults: [2, 4, 4]
    {"passive": "snail", "melee": "knight", "ranged": "knight_archer"},
    # 5 — Troll Mines: [1, 5, 5]
    {"passive": "bat", "melee": "troll", "ranged": "deep_thing"},
    # 6 — Fire Realm: [1, 6, 6]
    {"passive": "bat", "melee": "pigman", "ranged": "fire_elemental"},
    # 7 — Ice Realm: [1, 7, 7]
    {"passive": "bat", "melee": "frost_troll", "ranged": "ice_elemental"},
    # 8 — Boss: [0, 0, 0] — wave-summon defaults
    {"passive": "cow", "melee": "zombie", "ranged": "skeleton"},
)


# Maps each ranged-mob name to the projectile type it fires.
# Upstream: constants.py:320-331 — RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING.
RANGED_MOB_TO_PROJECTILE: dict[str, ProjectileType] = {
    "skeleton": ProjectileType.ARROW,
    "gnome_archer": ProjectileType.ARROW,
    "orc_mage": ProjectileType.FIREBALL,
    "kobold": ProjectileType.DAGGER,
    "knight_archer": ProjectileType.ARROW2,
    "deep_thing": ProjectileType.SLIMEBALL,
    "fire_elemental": ProjectileType.FIREBALL2,
    "ice_elemental": ProjectileType.ICEBALL2,
}


class MobLike(TypedDict):
    """Mob state used by movement and combat helpers."""
    type: str
    x: int
    y: int
    hp: int
    max_hp: int
    is_boss: bool
    floor: int
    attack_cooldown: int


MELEE_ATTACK_COOLDOWN: int = 5
RANGED_ATTACK_COOLDOWN: int = 4
MOB_DESPAWN_DISTANCE: int = 14
_CARDINAL_DIRS: tuple[tuple[int, int], ...] = (
    (0, -1), (0, 1), (-1, 0), (1, 0),
)
_PASSIVE_DIRS: tuple[tuple[int, int], ...] = _CARDINAL_DIRS + (
    (0, 0), (0, 0), (0, 0), (0, 0),
)


def _axis_step_toward(
    *,
    from_x: int,
    from_y: int,
    to_x: int,
    to_y: int,
    rng: random.Random,
) -> tuple[int, int]:
    """Single-cardinal step toward target, randomizing axis ties upstream-style."""
    dist_x = to_x - from_x
    dist_y = to_y - from_y
    abs_x = abs(dist_x)
    abs_y = abs(dist_y)
    if abs_x == 0 and abs_y == 0:
        return (0, 0)
    if abs_x > abs_y:
        return (1 if dist_x > 0 else -1, 0)
    if abs_y > abs_x:
        return (0, 1 if dist_y > 0 else -1)
    if rng.choice((0, 1)) == 0:
        return (1 if dist_x > 0 else -1, 0)
    return (0, 1 if dist_y > 0 else -1)


def step_melee_mob(
    mob: MobLike,
    *,
    player_x: int,
    player_y: int,
    is_blocked_for_mob: Callable[[int, int], bool],
    apply_damage_to_player: Callable[[int], None],
    rng: random.Random,
    is_fighting_boss: bool,
    damage_for_mob: Callable[[MobLike], int],
) -> None:
    """Advance one melee mob for one tick. Mirrors upstream
    _move_melee_mob (game_logic.py:1100-1291) at phase-α fidelity:

    - Movement candidate is chosen first: random cardinal step, or a 75%
      chase step if dist_sum < 10 OR is_fighting_boss.
    - If adjacent (Manhattan == 1) and cooldown <= 0: attack, reset cooldown
      to MELEE_ATTACK_COOLDOWN, and do not move.
    - Otherwise tick cooldown down by 1, matching upstream's negative cooldown
      drift.

    Damage value is supplied by `damage_for_mob` so the caller controls
    the per-mob damage table. Movement is bounded by `is_blocked_for_mob`.
    """
    manhattan = abs(mob["x"] - player_x) + abs(mob["y"] - player_y)
    chase = is_fighting_boss or (manhattan < 10 and rng.random() < 0.75)
    if chase:
        dx, dy = _axis_step_toward(
            from_x=mob["x"], from_y=mob["y"],
            to_x=player_x, to_y=player_y, rng=rng,
        )
    else:
        dx, dy = rng.choice(_CARDINAL_DIRS)

    if manhattan == 1 and mob["attack_cooldown"] <= 0:
        apply_damage_to_player(damage_for_mob(mob))
        mob["attack_cooldown"] = MELEE_ATTACK_COOLDOWN
        return

    mob["attack_cooldown"] -= 1
    nx, ny = mob["x"] + dx, mob["y"] + dy
    if not is_blocked_for_mob(nx, ny):
        mob["x"], mob["y"] = nx, ny


def step_ranged_mob(
    mob: MobLike,
    *,
    player_x: int,
    player_y: int,
    is_blocked_for_mob: Callable[[int, int], bool],
    spawn_mob_projectile: Callable[..., None],
    rng: random.Random,
    max_mob_projectiles_room: int,
    projectile_kind_for_mob: Callable[[MobLike], ProjectileType],
    damage_for_mob: Callable[[MobLike], int],
) -> None:
    """Advance one ranged mob for one tick. Mirrors upstream
    _move_ranged_mob (game_logic.py:1391-1608) at phase-γ fidelity:

    - Movement candidate is usually random: upstream uses the strategic
      toward/away candidate only 15% of non-attacking turns.
    - Shoot window: dist_sum in [4, 5] AND cooldown <= 0 AND projectile slot
      available → spawn projectile aimed at player; cooldown = 4; no move.
    - Cornered fallback: dist <= 3 AND the selected retreat/random move is
      blocked AND cooldown <= 0 AND projectile slot available → shoot.
    - dist >= 6: advance toward player.
    - dist <= 3: retreat away.
    - 3 < dist < 6 (and not in shoot window): random walk.

    `spawn_mob_projectile(kind, x, y, dx, dy, damage)` is the callback that
    appends a ProjectileEntity to the env's mob-projectile list.
    `projectile_kind_for_mob(mob)` returns the ProjectileType for this mob's
    type; the caller maps via RANGED_MOB_TO_PROJECTILE or a scalar fallback.
    `damage_for_mob(mob)` returns scalar damage for phase-α.
    """
    dist = abs(mob["x"] - player_x) + abs(mob["y"] - player_y)
    aim_dx, aim_dy = _axis_step_toward(
        from_x=mob["x"], from_y=mob["y"],
        to_x=player_x, to_y=player_y, rng=rng,
    )
    strategic_dx, strategic_dy = aim_dx, aim_dy
    if dist <= 3:
        strategic_dx, strategic_dy = -aim_dx, -aim_dy
    elif dist < 6:
        strategic_dx, strategic_dy = rng.choice(_CARDINAL_DIRS)

    random_dx, random_dy = rng.choice(_CARDINAL_DIRS)
    if rng.random() > 0.85:
        dx, dy = strategic_dx, strategic_dy
    else:
        dx, dy = random_dx, random_dy

    wants_to_shoot = 4 <= dist <= 5
    if dist <= 3 and is_blocked_for_mob(mob["x"] + dx, mob["y"] + dy):
        wants_to_shoot = True

    if (
        wants_to_shoot
        and mob["attack_cooldown"] <= 0
        and max_mob_projectiles_room > 0
    ):
        kind = projectile_kind_for_mob(mob)
        spawn_mob_projectile(
            kind, mob["x"], mob["y"], aim_dx, aim_dy, damage_for_mob(mob),
        )
        mob["attack_cooldown"] = RANGED_ATTACK_COOLDOWN
        return

    mob["attack_cooldown"] -= 1
    nx, ny = mob["x"] + dx, mob["y"] + dy
    if not is_blocked_for_mob(nx, ny):
        mob["x"], mob["y"] = nx, ny


def step_passive_mob(
    mob: MobLike,
    *,
    is_blocked_for_mob: Callable[[int, int], bool],
    rng: random.Random,
) -> None:
    """Move one passive mob with upstream 4-cardinal + 50% stay random walk."""
    dx, dy = rng.choice(_PASSIVE_DIRS)
    nx, ny = mob["x"] + dx, mob["y"] + dy
    if not is_blocked_for_mob(nx, ny):
        mob["x"], mob["y"] = nx, ny


def should_despawn(
    mob: MobLike,
    *,
    player_x: int,
    player_y: int,
    is_fighting_boss: bool,
) -> bool:
    """Return True if this mob should be removed this tick.

    Upstream rule (game_logic.py:1100-1291 + 1391-1608):
    - Manhattan distance from player >= MOB_DESPAWN_DISTANCE (=14).
    - Boss-fight melee + ranged mobs are exempt so the necromancer
      summon wave persists.

    Passive mobs (cow, snail, bat) follow the same despawn rule even
    in boss fights upstream — but the boss floor has no passives, so
    the distinction doesn't bite in practice.
    """
    if is_fighting_boss and mob["type"] in MELEE_MOB_NAMES + RANGED_MOB_NAMES + (
        "zombie",  # floor-0 melee
    ):
        return False
    dist_sum = abs(mob["x"] - player_x) + abs(mob["y"] - player_y)
    return dist_sum >= MOB_DESPAWN_DISTANCE


from glyphbench.envs.craftax.mechanics.damage import DamageVec  # noqa: E402

# Per-mob defense vector: fraction of incoming damage REDUCED for each
# element (phys, fire, ice). Mirrors upstream constants.py:289-318.
# Names match T18β's MELEE_MOB_NAMES + RANGED_MOB_NAMES + PASSIVE_MOB_NAMES.
MOB_DEFENSE_VEC: dict[str, DamageVec] = {
    # Floors 0-3 mobs: no defense.
    "zombie": (0.0, 0.0, 0.0),
    "skeleton": (0.0, 0.0, 0.0),
    "kobold": (0.0, 0.0, 0.0),
    "gnome_warrior": (0.0, 0.0, 0.0),
    "gnome_archer": (0.0, 0.0, 0.0),
    "orc_soldier": (0.0, 0.0, 0.0),
    "orc_mage": (0.0, 0.0, 0.0),
    "lizard": (0.0, 0.0, 0.0),
    # Floor 4: vault knight has 0.5 phys defense.
    "knight": (0.5, 0.0, 0.0),
    "knight_archer": (0.5, 0.0, 0.0),
    # Floor 5: troll has 0.2 phys; deep_thing has no defense upstream.
    "troll": (0.2, 0.0, 0.0),
    "deep_thing": (0.0, 0.0, 0.0),
    # Floor 6 fire realm: 0.9 phys + 1.0 fire (immune to fire).
    "pigman": (0.9, 1.0, 0.0),
    "fire_elemental": (0.9, 1.0, 0.0),
    # Floor 7 ice realm: 0.9 phys + 1.0 ice (immune to ice).
    "frost_troll": (0.9, 0.0, 1.0),
    "ice_elemental": (0.9, 0.0, 1.0),
    # Passive mobs: no defense (rarely targeted but listed for completeness).
    "cow": (0.0, 0.0, 0.0),
    "bat": (0.0, 0.0, 0.0),
    "snail": (0.0, 0.0, 0.0),
}


def damage_dealt_to_mob(mob_type: str, damage_vec: DamageVec) -> int:
    """Apply per-mob defense vector to incoming damage. Returns rounded int."""
    def_v = MOB_DEFENSE_VEC.get(mob_type, (0.0, 0.0, 0.0))
    net = sum(d * (1.0 - dv) for d, dv in zip(damage_vec, def_v, strict=False))
    return max(0, int(round(net)))
