"""Craftax phase-β potion infrastructure.

6-color hidden-shuffle system:
  POTION_COLORS[i] (the color label) maps to POTION_EFFECTS[perm[i]] (its
  actual effect) where perm is generated once per episode from the episode
  seed.  The mapping is never exposed in the observation, forcing agents to
  learn color-effect associations by experimentation.
"""

from __future__ import annotations

import random as _random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # CraftaxFullEnv imported lazily to avoid circular imports

# 6 potion colors (index i in POTION_COLORS maps to index perm[i] in POTION_EFFECTS)
POTION_COLORS: tuple[str, ...] = (
    "RED", "GREEN", "BLUE", "PINK", "CYAN", "YELLOW"
)

# 6 effects (one per color slot, permuted per episode)
POTION_EFFECTS: tuple[str, ...] = (
    "heal_8",
    "poison_3",
    "mana_8",
    "mana_drain_3",
    "energy_8",
    "energy_drain_3",
)


def make_potion_mapping(seed: int) -> tuple[int, ...]:
    """Return a permutation of (0,1,2,3,4,5) deterministic from *seed*.

    The i-th element is the index into POTION_EFFECTS that the i-th color
    maps to.  Using Python's stdlib ``random.Random`` so the result is
    fully reproducible regardless of numpy version.
    """
    rng = _random.Random(seed)
    perm = list(range(6))
    rng.shuffle(perm)
    return tuple(perm)


def apply_potion_effect(env: object, effect: str) -> None:  # type: ignore[type-arg]
    """Mutate *env* state in-place according to *effect*.

    Reads ``env._max_hp``, ``env._max_mana``, ``env._max_energy`` (or falls
    back to the module-level constants from full.py) so the caps are always
    respected.

    ``poison_3`` uses ``env._take_damage(3)`` so the standard armor + sleep
    multiplier path applies (consistent with mob damage).
    """
    from glyphbench.envs.craftaxfull.full import (  # local import avoids circular
        _MAX_ENERGY,
        _MAX_MANA,
    )

    max_hp = getattr(env, "_max_hp", 9)
    max_mana = getattr(env, "_max_mana", _MAX_MANA)
    max_energy = getattr(env, "_max_energy", _MAX_ENERGY)

    if effect == "heal_8":
        env._hp = min(max_hp, env._hp + 8)  # type: ignore[union-attr]
    elif effect == "poison_3":
        env._take_damage(3)  # type: ignore[union-attr]  # standard damage flow
    elif effect == "mana_8":
        env._mana = min(max_mana, env._mana + 8)  # type: ignore[union-attr]
    elif effect == "mana_drain_3":
        env._mana = max(0, env._mana - 3)  # type: ignore[union-attr]
    elif effect == "energy_8":
        env._energy = min(max_energy, env._energy + 8)  # type: ignore[union-attr]
    elif effect == "energy_drain_3":
        env._energy = max(0, env._energy - 3)  # type: ignore[union-attr]
    # Unknown effects are silently ignored (forward compat).
