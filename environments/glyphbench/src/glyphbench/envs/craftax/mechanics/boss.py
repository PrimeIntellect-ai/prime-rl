"""Phase-γ necromancer state machine (T17γ).

Mirrors upstream game_logic.py:450-472 + 2050-2340.

State (fields on env):
    _boss_progress: int in [0, 8] — number of vulnerable hits landed.
    _boss_summon_timer: int — turns remaining in current summon window.

Transitions:
    Vulnerable iff:
        (a) no melee/ranged mobs alive on floor 8 AND
        (b) _boss_summon_timer <= 0.
    Player damage to necromancer during vulnerable:
        _boss_progress += 1, _boss_summon_timer = BOSS_FIGHT_SPAWN_TURNS.
    Player damage during invulnerable: ignored (no effect).

Win:
    _boss_progress >= BOSS_PROGRESS_WIN_THRESHOLD →
        terminate episode with success + "defeat_necromancer" achievement reward.
"""
from __future__ import annotations

# Turns of wave-summoning that follow each hit on the necromancer.
# Mirrors upstream constant (game_logic.py: summon_timer = 7).
BOSS_FIGHT_SPAWN_TURNS: int = 7

# Number of vulnerable hits needed to defeat the necromancer.
BOSS_PROGRESS_WIN_THRESHOLD: int = 8

# Passive mob types that do NOT count as arena threats (they should not
# block the vulnerable transition).
_PASSIVE_MOB_TYPES: frozenset[str] = frozenset({"snail", "cow", "bat"})

# Floor number that hosts the necromancer boss.
NECROMANCER_FLOOR: int = 8


def is_necromancer_vulnerable(env) -> bool:
    """Return True iff the necromancer is currently hittable.

    Conditions (both must hold):
    1.  The summon timer has expired (<=0).
    2.  No hostile (non-passive) mobs remain alive on floor 8.
    """
    if env._boss_summon_timer > 0:
        return False
    for m in env._mobs:
        if (
            m["floor"] == NECROMANCER_FLOOR
            and m.get("type", "") not in _PASSIVE_MOB_TYPES
            and not m.get("is_boss", False)  # the necromancer tile itself is not a mob
            and m["hp"] > 0
        ):
            return False
    return True


def damage_necromancer_if_vulnerable(env) -> bool:
    """Attempt to hit the necromancer; returns True if the hit was registered.

    Side effects on success:
    - env._boss_progress += 1
    - env._boss_summon_timer = BOSS_FIGHT_SPAWN_TURNS

    Caller is responsible for verifying that the player is facing the
    necromancer tile before calling this function.
    """
    if not is_necromancer_vulnerable(env):
        return False
    env._boss_progress += 1
    env._boss_summon_timer = BOSS_FIGHT_SPAWN_TURNS
    return True


def boss_progress_win(env) -> bool:
    """Return True iff the necromancer has been defeated (8 hits landed)."""
    return env._boss_progress >= BOSS_PROGRESS_WIN_THRESHOLD
