"""MiniGrid world objects: walls, doors, keys, balls, boxes, lava, goals, water.

Each object type defines its canonical grid glyph, color, and interaction rules.
Objects are mutable (doors can open/close) but position is tracked by the grid,
not by the object itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorldObject:
    """Base for all grid objects."""

    obj_type: str = field(init=False)
    char: str = field(init=False)
    color: str = ""
    can_overlap: bool = field(init=False, default=False)
    can_pickup: bool = field(init=False, default=False)

    def render_char(self) -> str:
        return self.char

    def legend_name(self) -> str:
        if self.color:
            return f"{self.obj_type} ({self.color})"
        return self.obj_type


@dataclass
class Wall(WorldObject):
    def __post_init__(self) -> None:
        self.obj_type = "wall"
        self.char = "\u2588"  # █
        self.can_overlap = False
        self.can_pickup = False


@dataclass
class Goal(WorldObject):
    def __post_init__(self) -> None:
        self.obj_type = "goal"
        self.char = "\u2605"  # ★
        self.can_overlap = True
        self.can_pickup = False


@dataclass
class Lava(WorldObject):
    def __post_init__(self) -> None:
        self.obj_type = "lava"
        # Was \u2667 (collided with purple Key). Use \u2668 (hot springs).
        self.char = "\u2668"  # ♨ (hot springs / lava)
        self.can_overlap = True
        self.can_pickup = False


@dataclass
class Water(WorldObject):
    def __post_init__(self) -> None:
        self.obj_type = "water"
        self.char = "\u2248"  # ≈
        self.can_overlap = False
        self.can_pickup = False


@dataclass
class Trap(WorldObject):
    """Decoy-goal trap — terminates with -0.5 reward when stepped on.

    Used by distshift to penalise picking the wrong gap. Walkable (so
    stepping on it triggers the penalty), not pickup-able.
    """

    def __post_init__(self) -> None:
        self.obj_type = "trap"
        self.char = "☠"  # ☠
        self.can_overlap = True
        self.can_pickup = False


# Distinct single-character render for each (type, color) combination.
# This lets the LLM distinguish same-type objects of different colors.
# Unicode glyphs per color — each (type, color) pair is unique across all objects.
# Pads: dotted-circle family
_PAD_CHARS: dict[str, str] = {
    "red": "◉",    # ◉ (fisheye)
    "green": "◎",  # ◎ (bullseye)
    "blue": "⊙",   # ⊙ (circled dot operator)
    "yellow": "◈", # ◈ (white diamond containing black small diamond)
    "purple": "◐", # ◐ (circle with left half black)
    "grey": "◯",   # ◯ (large circle)
}
# Keys: ♠♤♡♢♧♦ (suit-family, one per color)
_KEY_CHARS: dict[str, str] = {
    "red": "\u2660",    # ♠
    "green": "\u2664",  # ♤
    "blue": "\u2661",   # ♡
    "yellow": "\u2662", # ♢
    "purple": "\u2667", # ♧
    "grey": "\u2666",   # ♦
}
# Balls: geometric shapes
_BALL_CHARS: dict[str, str] = {
    "red": "\u25cf",    # ●
    "green": "\u25cb",  # ○
    "blue": "\u25c6",   # ◆
    "yellow": "\u2b21",  # ⬡
    "purple": "\u25a0",  # ■
    "grey": "\u25a1",   # □
}
# Boxes: varied geometric
_BOX_CHARS: dict[str, str] = {
    "red": "\u25b2",    # ▲
    "green": "\u25b3",  # △
    "blue": "\u25bc",   # ▼
    "yellow": "\u25bd", # ▽
    "purple": "\u25c0", # ◀
    "grey": "\u25b7",   # ▷
}
# Doors (closed): line-drawing
_DOOR_CLOSED_CHARS: dict[str, str] = {
    "red": "\u2563",    # ╣
    "green": "\u2560",  # ╠
    "blue": "\u2566",   # ╦
    "yellow": "\u2569", # ╩
    "purple": "\u256c", # ╬
    "grey": "\u2551",   # ║
}
# Doors (open): thin line-drawing
_DOOR_OPEN_CHARS: dict[str, str] = {
    "red": "\u2524",    # ┤
    "green": "\u251c",  # ├
    "blue": "\u252c",   # ┬
    "yellow": "\u2534", # ┴
    "purple": "\u253c", # ┼
    "grey": "\u2502",   # │
}


@dataclass
class Pad(WorldObject):
    """Coloured drop-pad — used by twin-goal envs.

    Walkable, not pickup-able. The dropping logic is env-specific (the
    parent DROP-into-empty rule does not handle pad-targeted drops). Envs
    that use Pad implement custom DROP semantics. Each colour gets a
    distinct dotted-circle glyph so the agent can map glyph to colour.
    """

    color: str = "yellow"

    def __post_init__(self) -> None:
        self.obj_type = "pad"
        self.char = _PAD_CHARS.get(self.color, "◉")
        self.can_overlap = True
        self.can_pickup = False

    def render_char(self) -> str:
        return _PAD_CHARS.get(self.color, "◉")

    def legend_name(self) -> str:
        return f"pad ({self.color})" if self.color else "pad"


@dataclass
class Key(WorldObject):
    color: str = "red"

    def __post_init__(self) -> None:
        self.obj_type = "key"
        self.char = _KEY_CHARS.get(self.color, "K")
        self.can_overlap = False
        self.can_pickup = True

    def render_char(self) -> str:
        return _KEY_CHARS.get(self.color, "K")


@dataclass
class Ball(WorldObject):
    color: str = "red"

    def __post_init__(self) -> None:
        self.obj_type = "ball"
        self.char = _BALL_CHARS.get(self.color, "O")
        self.can_overlap = False
        self.can_pickup = True

    def render_char(self) -> str:
        return _BALL_CHARS.get(self.color, "O")


@dataclass
class Box(WorldObject):
    color: str = "red"
    contains: WorldObject | None = None
    is_open: bool = False

    def __post_init__(self) -> None:
        self.obj_type = "box"
        self.char = _BOX_CHARS.get(self.color, "B")
        # Closed boxes block movement; opened boxes are removed from the
        # grid when revealing their contents (see toggle).
        self.can_overlap = False
        # Closed boxes can be picked up; once toggled-open they are gone.
        self.can_pickup = True

    def render_char(self) -> str:
        return _BOX_CHARS.get(self.color, "B")

    def toggle(self) -> WorldObject | None:
        """Open the box. Returns the contents (the agent's environment is
        responsible for swapping the box cell for the contained object).

        Returns None if the box was already opened or empty.
        """
        if self.is_open:
            return None
        self.is_open = True
        return self.contains


@dataclass
class Door(WorldObject):
    color: str = "red"
    is_open: bool = False
    is_locked: bool = False

    def __post_init__(self) -> None:
        self.obj_type = "door"
        self.char = (
            _DOOR_OPEN_CHARS.get(self.color, "d") if self.is_open
            else _DOOR_CLOSED_CHARS.get(self.color, "D")
        )
        self.can_overlap = self.is_open
        self.can_pickup = False

    def legend_name(self) -> str:
        state = (
            "locked "
            if self.is_locked
            else ("open " if self.is_open else "closed ")
        )
        return f"{state}door ({self.color})"

    def toggle(self, carrying: WorldObject | None) -> bool:
        """Toggle the door. Returns True if state changed."""
        if self.is_locked:
            if isinstance(carrying, Key) and carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                self.char = _DOOR_OPEN_CHARS.get(self.color, "d")
                self.can_overlap = True
                return True
            return False
        self.is_open = not self.is_open
        self.char = (
            _DOOR_OPEN_CHARS.get(self.color, "d") if self.is_open
            else _DOOR_CLOSED_CHARS.get(self.color, "D")
        )
        self.can_overlap = self.is_open
        return True

    def render_char(self) -> str:
        return self.char
