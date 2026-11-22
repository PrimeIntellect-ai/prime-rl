"""MiniHack creature types. Subset of NetHack monsters used by MiniHack tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CreatureType:
    """Template for a monster type."""

    name: str
    char: str  # NetHack glyph
    max_hp: int
    damage: int  # average damage per hit
    speed: int  # 1 = normal (moves every turn), 2 = fast
    ai: str  # "hostile", "passive", "peaceful"
    xp_value: int

    def legend_name(self) -> str:
        return self.name


@dataclass
class Creature:
    """A live instance of a creature on the map."""

    ctype: CreatureType
    x: int
    y: int
    hp: int

    @classmethod
    def spawn(cls, ctype: CreatureType, x: int, y: int) -> Creature:
        return cls(ctype=ctype, x=x, y=y, hp=ctype.max_hp)


# Common MiniHack monsters
NEWT = CreatureType("newt", ":", 1, 1, 1, "hostile", 1)
RAT = CreatureType("rat", "r", 2, 1, 1, "hostile", 2)
KOBOLD = CreatureType("kobold", "k", 4, 2, 1, "hostile", 5)
GNOME = CreatureType("gnome", "G", 6, 3, 1, "hostile", 8)
ORC = CreatureType("orc", "o", 8, 4, 1, "hostile", 10)
ZOMBIE = CreatureType("zombie", "Z", 10, 3, 1, "hostile", 12)
OGRE = CreatureType("ogre", "O", 14, 6, 1, "hostile", 20)
TROLL = CreatureType("troll", "T", 20, 8, 1, "hostile", 30)
