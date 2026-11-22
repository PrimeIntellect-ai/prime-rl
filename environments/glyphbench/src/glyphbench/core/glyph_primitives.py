"""Shared glyph-grid building helpers for env rendering.

Envs are encouraged to use these instead of rolling their own string joining.
Kept small and pure — no state, no side effects beyond the in-place mutations
documented per function.
"""

from __future__ import annotations


def make_empty_grid(width: int, height: int, fill: str = "\u00b7") -> list[list[str]]:
    """Create a height x width grid filled with `fill`. Fill must be exactly 1 char."""
    assert len(fill) == 1, f"fill must be a single char, got {fill!r}"
    return [[fill for _ in range(width)] for _ in range(height)]


def stamp_sprite(grid: list[list[str]], x: int, y: int, sprite: str) -> None:
    """Place a single character at (x, y). Raises IndexError if out of bounds."""
    assert len(sprite) == 1, f"sprite must be a single char, got {sprite!r}"
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    if not (0 <= x < width and 0 <= y < height):
        raise IndexError(f"stamp_sprite ({x},{y}) out of bounds for {width}x{height} grid")
    grid[y][x] = sprite


def draw_box(
    grid: list[list[str]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    horizontal: str = "-",
    vertical: str = "|",
    corner: str = "+",
) -> None:
    """Draw an axis-aligned box outline in-place. (x0,y0) is top-left; (x1,y1) is bottom-right."""
    if not (x0 < x1 and y0 < y1):
        raise ValueError(
            f"draw_box requires x0 < x1 and y0 < y1, got ({x0},{y0})-({x1},{y1})"
        )
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    if not (x0 >= 0 and y0 >= 0 and x1 < width and y1 < height):
        raise IndexError(
            f"draw_box ({x0},{y0})-({x1},{y1}) out of bounds for {width}x{height} grid"
        )
    for x in range(x0 + 1, x1):
        grid[y0][x] = horizontal
        grid[y1][x] = horizontal
    for y in range(y0 + 1, y1):
        grid[y][x0] = vertical
        grid[y][x1] = vertical
    grid[y0][x0] = corner
    grid[y0][x1] = corner
    grid[y1][x0] = corner
    grid[y1][x1] = corner


def grid_to_string(grid: list[list[str]]) -> str:
    """Join a 2D char grid into the canonical newline-separated string."""
    return "\n".join("".join(row) for row in grid)


def build_legend(symbol_meanings: dict[str, str]) -> str:
    """Deterministic, sorted legend string.

    Sort order: by symbol character (stable, Unicode codepoint order). Every
    entry appears as '<sym> — <meaning>' on its own line.
    """
    lines: list[str] = []
    for sym in sorted(symbol_meanings.keys()):
        lines.append(f"{sym} — {symbol_meanings[sym]}")
    return "\n".join(lines)
