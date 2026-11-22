"""Phase-γ per-attribute scaling helpers.

Mirrors upstream util/game_logic_utils.py:362-378.
"""
from __future__ import annotations


def max_hp_from_str(base: int, str_attr: int) -> int:
    """Phase γ: max HP scales with strength."""
    return base + max(0, str_attr - 1)


def max_food_from_dex(base: int, dex: int) -> int:
    return base + max(0, dex - 1) * 2


def max_drink_from_dex(base: int, dex: int) -> int:
    return base + max(0, dex - 1) * 2


def max_energy_from_dex(base: int, dex: int) -> int:
    return base + max(0, dex - 1) * 2


def max_mana_from_int(base: int, int_attr: int) -> int:
    return base + max(0, int_attr - 1) * 3


def damage_scale_phys(str_attr: int) -> float:
    """Physical-damage multiplier from strength. Caps at 2× at str=5."""
    return 1.0 + 0.25 * max(0, str_attr - 1)


def damage_scale_arrow(dex: int) -> float:
    """Arrow-damage multiplier from dexterity. Caps at 1.8× at dex=5."""
    return 1.0 + 0.2 * max(0, dex - 1)


def damage_scale_spell(int_attr: int) -> float:
    """Spell-damage multiplier from intelligence. Caps at 3× at int=5.

    Matches the fork's magic_damage_coeff = 1 + 0.5*(intelligence-1)
    (game_logic.py:1752).
    """
    return 1.0 + 0.5 * max(0, int_attr - 1)


def mana_regen_scale(int_attr: int) -> float:
    """Mana-regen multiplier from intelligence."""
    return 1.0 + 0.25 * max(0, int_attr - 1)


def decay_scale(dex: int) -> float:
    """Need-decay multiplier from dexterity. <1 means decay slower."""
    return 1.0 - 0.125 * max(0, dex - 1)
