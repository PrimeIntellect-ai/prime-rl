"""Dungeon-room biome generator.

Generates a dungeon floor with multiple non-overlapping rectangular rooms
connected by L-shaped corridors.  Returns the grid plus metadata positions
for chests, fountains, stairs, and the agent spawn.

The generator is pure Python (no env state) so it can be unit-tested
independently.
"""
from __future__ import annotations

import math
import random

import numpy as np

from glyphbench.envs.craftax.base import (
    TILE_CHEST,
    TILE_DUNGEON_FLOOR,
    TILE_DUNGEON_WALL,
    TILE_FOUNTAIN,
    TILE_STAIRS_DOWN,
    TILE_STAIRS_UP,
)

# A room is (rx, ry, rw, rh) in grid coordinates.
Room = tuple[int, int, int, int]

_RNG_T = np.random.Generator | random.Random


def _rng_int(rng: _RNG_T, lo: int, hi_exclusive: int) -> int:
    """Return a random integer in [lo, hi_exclusive) for either RNG type."""
    if isinstance(rng, np.random.Generator):
        return int(rng.integers(lo, hi_exclusive))
    # random.Random
    return rng.randint(lo, hi_exclusive - 1)


def _rng_float(rng: _RNG_T) -> float:
    if isinstance(rng, np.random.Generator):
        return float(rng.random())
    return rng.random()


def _room_center(room: Room) -> tuple[int, int]:
    rx, ry, rw, rh = room
    return rx + rw // 2, ry + rh // 2


def _carve_room(grid: list[list[str]], room: Room) -> None:
    rx, ry, rw, rh = room
    for dy in range(rh):
        for dx in range(rw):
            grid[ry + dy][rx + dx] = TILE_DUNGEON_FLOOR


def _carve_corridor(
    grid: list[list[str]],
    c1: tuple[int, int],
    c2: tuple[int, int],
    size: int,
) -> None:
    """Carve an L-shaped corridor from *c1* to *c2* (horizontal then vertical)."""
    x1, y1 = c1
    x2, y2 = c2
    # Horizontal leg
    step = 1 if x2 >= x1 else -1
    for x in range(x1, x2 + step, step):
        if 0 <= x < size and 0 <= y1 < size:
            grid[y1][x] = TILE_DUNGEON_FLOOR
    # Vertical leg
    step = 1 if y2 >= y1 else -1
    for y in range(y1, y2 + step, step):
        if 0 <= x2 < size and 0 <= y < size:
            grid[y][x2] = TILE_DUNGEON_FLOOR


def _interior_positions(room: Room) -> list[tuple[int, int]]:
    """Return a list of interior (x, y) positions (1-tile inset from edges)."""
    rx, ry, rw, rh = room
    positions: list[tuple[int, int]] = []
    for dy in range(1, rh - 1):
        for dx in range(1, rw - 1):
            positions.append((rx + dx, ry + dy))
    return positions


def generate_dungeon_floor(
    rng: _RNG_T,
    size: int,
    num_rooms: int = 8,
    with_chests: bool = True,
    with_fountains: bool = True,
) -> tuple[
    list[list[str]],          # grid[y][x]
    list[tuple[int, int]],    # chest positions
    list[tuple[int, int]],    # fountain positions
    tuple[int, int],           # stairs_down_pos  (0,0) sentinel if floor has no down
    tuple[int, int],           # stairs_up_pos
    tuple[int, int],           # agent spawn pos
]:
    """Generate a dungeon floor with *num_rooms* rooms.

    Algorithm
    ---------
    1. Place *num_rooms* non-overlapping rooms (width/height 5-10 each) by
       rejection sampling up to 200 attempts per room.  If fewer than 2 rooms
       are placed, fall back to two guaranteed rooms so the floor is always
       playable.
    2. Connect adjacent rooms with L-shaped corridors (horizontal then vertical).
    3. Place 1 ``TILE_CHEST`` per room at a random interior position
       (if *with_chests*).
    4. With 50% probability per room place 1 ``TILE_FOUNTAIN`` at a different
       interior position (if *with_fountains*).
    5. Place ``TILE_STAIRS_UP`` in the first room, ``TILE_STAIRS_DOWN`` in the
       last room.
    6. Agent spawn is a safe interior tile in the first room (near stairs-up).

    Returns
    -------
    grid, chest_positions, fountain_positions, stairs_down_pos, stairs_up_pos,
    agent_spawn_pos
    """
    # 1. Initialise grid as all walls.
    grid: list[list[str]] = [
        [TILE_DUNGEON_WALL] * size for _ in range(size)
    ]

    # Place rooms using a chunk-based approach to guarantee exactly num_rooms.
    # Divide the grid into a 4x4 arrangement of equal chunks (requires num_rooms=8;
    # we use a 2-column x 4-row layout for 8 rooms). Each chunk gets exactly one
    # room, placed with a small random jitter within the chunk interior.
    #
    # Chunk size is determined by the grid size and the number of chunks.
    # For num_rooms rooms we arrange them in a ceil(sqrt(num_rooms)) x floor grid.
    cols = math.ceil(math.sqrt(num_rooms))   # e.g. 3 for 8 rooms
    rows = math.ceil(num_rooms / cols)       # e.g. 3 for 8 rooms

    # Minimum room dimension is 5; maximum is 7 (reduced from 10 to fit chunks).
    _MIN_ROOM = 5
    _MAX_ROOM = 7  # capped so rooms always fit within chunks

    chunk_w = size // cols
    chunk_h = size // rows

    rooms: list[Room] = []
    for chunk_row in range(rows):
        for chunk_col in range(cols):
            if len(rooms) >= num_rooms:
                break
            # Chunk origin (with a 1-cell border so rooms never touch the edge).
            ox = chunk_col * chunk_w + 1
            oy = chunk_row * chunk_h + 1
            # Available interior for the room inside the chunk.
            avail_w = chunk_w - 2
            avail_h = chunk_h - 2
            if avail_w < _MIN_ROOM or avail_h < _MIN_ROOM:
                # Chunk too small; skip (shouldn't happen for size=32, num_rooms=8).
                continue
            rw = _rng_int(rng, _MIN_ROOM, min(_MAX_ROOM, avail_w) + 1)
            rh = _rng_int(rng, _MIN_ROOM, min(_MAX_ROOM, avail_h) + 1)
            # Jitter within the chunk so rooms don't all align to the top-left.
            max_jx = avail_w - rw
            max_jy = avail_h - rh
            jx = _rng_int(rng, 0, max(1, max_jx + 1))
            jy = _rng_int(rng, 0, max(1, max_jy + 1))
            rx = ox + jx
            ry = oy + jy
            rooms.append((rx, ry, rw, rh))
            _carve_room(grid, (rx, ry, rw, rh))

    # Guarantee at least 2 rooms so stairs always fit.
    if len(rooms) < 2:
        rooms = [(2, 2, 7, 7), (size - 11, size - 11, 7, 7)]
        for room in rooms:
            _carve_room(grid, room)

    # 2. Connect rooms with L-shaped corridors.
    for i in range(len(rooms) - 1):
        _carve_corridor(grid, _room_center(rooms[i]), _room_center(rooms[i + 1]), size)

    # 5 (pre-compute). Stair positions so we can exclude them from chest/fountain
    # placement — stairs are placed last and would otherwise overwrite items.
    r0 = rooms[0]
    up_x = r0[0] + 1
    up_y = r0[1] + 1
    stairs_up_pos: tuple[int, int] = (up_x, up_y)

    rl = rooms[-1]
    down_x = rl[0] + rl[2] - 2
    down_y = rl[1] + rl[3] - 2
    if not (0 <= down_x < size and 0 <= down_y < size):
        down_x = rl[0] + rl[2] // 2
        down_y = rl[1] + rl[3] // 2
    stairs_down_pos: tuple[int, int] = (down_x, down_y)
    _stair_cells: frozenset[tuple[int, int]] = frozenset({stairs_up_pos, stairs_down_pos})

    # 3 & 4. Place chests and fountains in each room (excluding stair cells).
    chest_positions: list[tuple[int, int]] = []
    fountain_positions: list[tuple[int, int]] = []

    for room in rooms:
        interior = [
            pos for pos in _interior_positions(room)
            if pos not in _stair_cells
        ]
        if not interior:
            # Tiny room or all positions reserved — skip items for this room.
            continue

        # Shuffle interior positions for random placement.
        # numpy generator doesn't have .shuffle for Python lists, so convert.
        shuffled_interior = list(interior)
        if isinstance(rng, np.random.Generator):
            perm = list(rng.permutation(len(shuffled_interior)))
            shuffled_interior = [shuffled_interior[i] for i in perm]
        else:
            rng.shuffle(shuffled_interior)

        used_idx = 0

        if with_chests and shuffled_interior:
            cx, cy = shuffled_interior[used_idx]
            grid[cy][cx] = TILE_CHEST
            chest_positions.append((cx, cy))
            used_idx += 1

        if with_fountains and _rng_float(rng) < 0.5 and used_idx < len(shuffled_interior):
            fx, fy = shuffled_interior[used_idx]
            grid[fy][fx] = TILE_FOUNTAIN
            fountain_positions.append((fx, fy))
            used_idx += 1

    # 5. Place stair tiles (now guaranteed not to overwrite chests/fountains).
    grid[up_y][up_x] = TILE_STAIRS_UP
    grid[down_y][down_x] = TILE_STAIRS_DOWN

    # 6. Agent spawn: interior of first room, offset from stairs-up.
    spawn_x = r0[0] + 2
    spawn_y = r0[1] + 2
    if grid[spawn_y][spawn_x] not in {TILE_DUNGEON_FLOOR, TILE_CHEST, TILE_FOUNTAIN}:
        # Fall back to room centre.
        spawn_x, spawn_y = _room_center(r0)
    agent_spawn_pos: tuple[int, int] = (spawn_x, spawn_y)

    return (
        grid,
        chest_positions,
        fountain_positions,
        stairs_down_pos,
        stairs_up_pos,
        agent_spawn_pos,
    )
