"""MiniHack item types for skill tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Item:
    """Base item on the dungeon floor or in inventory."""

    name: str
    char: str  # NetHack glyph: % food, ? scroll, ! potion, / wand, ) weapon, = ring
    item_type: str  # "food", "scroll", "potion", "wand", "weapon", "ring"
    display_name: str | None = None

    def legend_name(self) -> str:
        return self.display_name or self.name


# Common items used by MiniHack skill tasks
FOOD_RATION = Item("food ration", "%", "food")
APPLE = Item("apple", "%", "food")
CORPSE = Item("corpse", "%", "food")

SCROLL_IDENTIFY = Item("scroll of identify", "?", "scroll")
SCROLL_TELEPORT = Item("scroll of teleportation", "?", "scroll")
SCROLL_LIGHT = Item("scroll of light", "?", "scroll")

POTION_HEALING = Item("potion of healing", "!", "potion")
POTION_SPEED = Item("potion of speed", "!", "potion")
POTION_LEVITATION = Item("potion of levitation", "!", "potion")

WAND_DEATH = Item("wand of death", "/", "wand")
WAND_FIRE = Item("wand of fire", "/", "wand")
WAND_COLD = Item("wand of cold", "/", "wand")

SWORD = Item("long sword", ")", "weapon")
DAGGER = Item("dagger", ")", "weapon")

RING_LEVITATION = Item("ring of levitation", "=", "ring")

BRASS_KEY = Item("brass key", "(", "key")
