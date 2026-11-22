"""Curses color mapping for all GlyphBench env tile symbols.

Provides a single function `char_attr(ch)` that returns the curses attribute
(color pair + bold/dim) for any grid glyph.  The mapping is env-agnostic:
known glyphs get stable colors for cross-family visual consistency.
"""

from __future__ import annotations

import curses

# Pair IDs (1-based; 0 is reserved for default white-on-black)
_PAIR_RED = 1
_PAIR_GREEN = 2
_PAIR_YELLOW = 3
_PAIR_BLUE = 4
_PAIR_MAGENTA = 5
_PAIR_CYAN = 6
_PAIR_WHITE = 7
_PAIR_RED_BG = 8      # red background (lava)
_PAIR_BLUE_BG = 9     # blue background (water)
_PAIR_GREEN_BG = 10   # green background (goal)
_PAIR_YELLOW_BG = 11  # yellow on black bold (gold/coins)

_initialized = False


def init_colors() -> None:
    """Initialize curses color pairs.  Call once after curses.initscr()."""
    global _initialized
    if _initialized:
        return
    curses.start_color()
    curses.use_default_colors()

    curses.init_pair(_PAIR_RED, curses.COLOR_RED, -1)
    curses.init_pair(_PAIR_GREEN, curses.COLOR_GREEN, -1)
    curses.init_pair(_PAIR_YELLOW, curses.COLOR_YELLOW, -1)
    curses.init_pair(_PAIR_BLUE, curses.COLOR_BLUE, -1)
    curses.init_pair(_PAIR_MAGENTA, curses.COLOR_MAGENTA, -1)
    curses.init_pair(_PAIR_CYAN, curses.COLOR_CYAN, -1)
    curses.init_pair(_PAIR_WHITE, curses.COLOR_WHITE, -1)
    curses.init_pair(_PAIR_RED_BG, curses.COLOR_WHITE, curses.COLOR_RED)
    curses.init_pair(_PAIR_BLUE_BG, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(_PAIR_GREEN_BG, curses.COLOR_BLACK, curses.COLOR_GREEN)
    curses.init_pair(_PAIR_YELLOW_BG, curses.COLOR_YELLOW, -1)

    _initialized = True


# ---------------------------------------------------------------------------
# Character -> attribute mapping
# ---------------------------------------------------------------------------
# Keys: single grid glyph.  Values: curses attr (color_pair | A_BOLD etc).
# Built lazily on first call to char_attr() so curses is already initialized.

_CHAR_ATTRS: dict[str, int] | None = None


def _build_char_attrs() -> dict[str, int]:
    """Build the char -> curses attribute dict."""
    R = curses.color_pair(_PAIR_RED)
    G = curses.color_pair(_PAIR_GREEN)
    Y = curses.color_pair(_PAIR_YELLOW)
    B = curses.color_pair(_PAIR_BLUE)
    M = curses.color_pair(_PAIR_MAGENTA)
    C = curses.color_pair(_PAIR_CYAN)
    W = curses.color_pair(_PAIR_WHITE)
    RB = curses.color_pair(_PAIR_RED_BG)
    curses.color_pair(_PAIR_BLUE_BG)
    GB = curses.color_pair(_PAIR_GREEN_BG)
    BOLD = curses.A_BOLD
    DIM = curses.A_DIM

    return {
        # --- Agent / player ---
        "@": Y | BOLD,
        ">": G | BOLD,  # minigrid agent facing right / minihack stairs
        "<": G | BOLD,
        "^": G | BOLD,
        "v": G | BOLD,

        # --- Terrain (shared across families) ---
        "#": W | DIM,        # wall
        ".": W | DIM,        # floor / empty
        " ": 0,              # empty space
        "~": C | BOLD,       # water
        "L": RB,             # lava (minigrid/craftax)
        "}": RB,             # lava (minihack)

        # --- MiniGrid objects (color-variant chars) ---
        "G": GB,             # goal (bright green bg)
        # Keys
        "K": R | BOLD,       # key red
        "k": G | BOLD,       # key green
        "j": B | BOLD,       # key blue
        "Y": Y | BOLD,       # key yellow
        "y": M | BOLD,       # key purple
        "J": W,              # key grey
        # Balls
        "O": R | BOLD,       # ball red
        "o": G | BOLD,       # ball green
        "Q": B | BOLD,       # ball blue
        "q": Y | BOLD,       # ball yellow
        "0": M | BOLD,       # ball purple
        "9": W,              # ball grey
        # Boxes
        "B": R | BOLD,       # box red
        "b": G | BOLD,       # box green
        "P": B | BOLD,       # box blue
        "p": Y | BOLD,       # box yellow
        "8": M | BOLD,       # box purple
        "7": W,              # box grey
        # Doors closed
        "D": R,              # door closed red
        "A": G,              # door closed green
        "E": B,              # door closed blue
        "F": Y,              # door closed yellow
        "H": M,              # door closed purple
        "I": W,              # door closed grey
        # Doors open
        "d": R | DIM,        # door open red
        "a": G | DIM,        # door open green
        "e": B | DIM,        # door open blue
        "f": Y | DIM,        # door open yellow
        "h": M | DIM,        # door open purple
        "i": W | DIM,        # door open grey

        # --- Craftax ---
        "T": G | BOLD,       # tree
        "S": W | BOLD,       # stone
        "C": W | DIM,        # coal
        "s": Y,              # sand
        "t": Y | BOLD,       # crafting table
        ";": G,              # sapling
        "*": R | BOLD,       # ripe plant / power pellet
        "=": W,              # placed stone / ground / ring
        "z": M | BOLD,       # zombie
        "c": W | BOLD,       # cow

        # --- Procgen ---
        "$": Y | BOLD,       # gold
        "%": R,              # fruit / food
        "+": Y,              # door (minihack) / wall corner

        # --- Atari ---
        "-": W | DIM,        # wall horizontal
        "|": W | DIM,        # wall vertical

        # --- MiniHack items ---
        "!": C | BOLD,       # potion
        "?": W | BOLD,       # scroll
        "/": M | BOLD,       # wand
        ")": W | BOLD,       # weapon
        "{": C,              # sink

        # --- MiniHack creatures ---
        ":": Y,              # newt
        "r": M,              # rat
        "Z": M | BOLD,       # zombie (minihack)
    }


def char_attr(ch: str) -> int:
    """Return curses attribute for a single grid character.

    Falls back to default (white) for unknown chars.
    """
    global _CHAR_ATTRS
    if _CHAR_ATTRS is None:
        _CHAR_ATTRS = _build_char_attrs()
    return _CHAR_ATTRS.get(ch, 0)
