"""MiniHack Memento environments.

Each variant exercises a DISTINCT memory mechanic:

  * ``short`` (A2 sequence-of-cues): three coloured cue glyphs flash on turns
    1/2/3 in the start room. The final room presents three coloured plates;
    the agent must STEP on them in the same order.
  * ``hard`` (A6 counted memory): torches are placed in the corridor row.
    The final room contains an NPC scribe; the agent must step on the
    numbered plate matching the torch count.
  * ``f2`` (A7 spatial memory): a marker glyph is shown at a random tile
    in the start room. After descending one floor, the agent must return
    to the same (relative) tile in the second floor's start room.
  * ``f4`` (A4 inverse memory): a colour cue is visible in every room
    EXCEPT the final fog room. The final fog room has three doors; the
    agent must enter the door whose colour matches the cue.

All four mechanics share: the agent moves N/S/E/W, rooms are 5x5, and the
final reward is +1 only if the memory test is solved correctly.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MOVE_VECTORS, MiniHackBase

# Cue-colour glyphs. Single-codepoint distinct symbols.
CUE_GLYPHS: tuple[str, str, str] = ("♠", "♣", "♦")
CUE_NAMES: dict[str, str] = {
    "♠": "spade",
    "♣": "club",
    "♦": "diamond",
}

PLATE_GLYPHS: tuple[str, str, str] = ("①", "②", "③")
PLATE_NAMES: dict[str, str] = {
    "①": "plate 1",
    "②": "plate 2",
    "③": "plate 3",
}

# Number glyphs for counted-memory final answer
NUM_GLYPHS: tuple[str, str, str, str, str, str] = ("➀", "➁", "➂", "➃", "➄", "➅")
NUM_NAMES: dict[str, str] = {
    "➀": "answer 1",
    "➁": "answer 2",
    "➂": "answer 3",
    "➃": "answer 4",
    "➄": "answer 5",
    "➅": "answer 6",
}

TORCH_GLYPH = "ψ"  # torch
MARKER_GLYPH = "✦"  # spatial marker (used by F2 spatial-memory variant)
COLOR_DOOR_GLYPHS: tuple[str, str, str] = ("Ⓡ", "Ⓖ", "Ⓑ")
COLOR_DOOR_NAMES: dict[str, str] = {
    "Ⓡ": "red door",
    "Ⓖ": "green door",
    "Ⓑ": "blue door",
}
CUE_MARKER_GLYPH = "★"  # marks the actual cue (F4 inverse-memory)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _carve_two_room_layout(env: MiniHackBase, room_w: int = 5) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    int,
]:
    """Carve a horizontal two-room layout with a 1-cell corridor between them.

    Returns (start_room_rect, final_room_rect, corridor_y).
    Each rect is (x0, y0, x1, y1) inclusive interior bounds.
    """
    total_w = (room_w + 1) * 2 + 1  # walls + corridor
    total_h = room_w + 2
    env._init_grid(total_w, total_h)

    # Fill interior with walls, then carve rooms
    for y in range(1, total_h - 1):
        for x in range(1, total_w - 1):
            env._place_wall(x, y)

    # Start room (left)
    start_rect = (1, 1, room_w, room_w)
    for y in range(1, room_w + 1):
        for x in range(1, room_w + 1):
            env._grid[y][x] = "·"

    # Final room (right)
    final_x0 = room_w + 2
    final_rect = (final_x0, 1, final_x0 + room_w - 1, room_w)
    for y in range(1, room_w + 1):
        for x in range(final_x0, final_x0 + room_w):
            env._grid[y][x] = "·"

    # Corridor between rooms (1-cell wide at room midline)
    cy = total_h // 2
    env._grid[cy][room_w + 1] = "·"

    return start_rect, final_rect, cy


# ---------------------------------------------------------------------------
# Short: A2 sequence-of-cues
# ---------------------------------------------------------------------------


class _MementoShortBase(MiniHackBase):
    """Short Memento: three cue glyphs flash on turns 1/2/3 in the start room.

    The final room contains three plates ①②③ at fixed positions; the agent
    must step on them in the same colour order as the cues. Wrong order → -1.
    Correct full order → +1.
    """

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._cue_sequence: list[str] = []
        self._plate_to_color: dict[str, str] = {}
        self._cue_step: int = 0
        self._stepped_sequence: list[str] = []
        self._cue_pos: tuple[int, int] = (0, 0)
        self._plate_color_markers: dict[tuple[int, int], str] = {}

    def _generate_level(self, seed: int) -> None:
        start_rect, final_rect, cy = _carve_two_room_layout(self)
        # Player on left middle of start room
        self._place_player(1, cy)
        # Cue display position: visible top-left corner of start room
        self._cue_pos = (3, 1)
        # Stairs do NOT exist as terrain; goal is plate sequence.
        # We use _goal_pos = None so base step's goal-check never fires.
        self._goal_pos = None

        # Sequence of three distinct colours
        colours = list(CUE_GLYPHS)
        order = list(self.rng.permutation(len(colours)))
        self._cue_sequence = [colours[i] for i in order]

        # Plate-to-colour mapping (random, plates are at fixed positions)
        plate_positions = [
            (final_rect[0] + 0, final_rect[1] + 1),
            (final_rect[0] + 2, final_rect[1] + 1),
            (final_rect[0] + 4, final_rect[1] + 1),
        ]
        plate_glyphs = list(PLATE_GLYPHS)
        plate_order = list(self.rng.permutation(len(plate_positions)))
        self._plate_to_color = {}
        self._plate_positions: dict[tuple[int, int], str] = {}
        self._plate_color_markers = {}
        for i, plate_idx in enumerate(plate_order):
            px, py = plate_positions[i]
            plate_glyph = plate_glyphs[i]
            colour_glyph = colours[plate_idx]
            self._grid[py][px] = plate_glyph
            self._plate_to_color[plate_glyph] = colour_glyph
            self._plate_positions[(px, py)] = plate_glyph
            marker_pos = (px, py - 1)
            self._plate_color_markers[marker_pos] = colour_glyph

        self._cue_step = 0
        self._stepped_sequence = []

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        # Treat plate stepping specially — we override after movement.
        name = self.action_spec.names[action]
        info: dict[str, Any] = {}
        terminated = False
        reward = 0.0

        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy
            terrain = self._terrain_at(nx, ny)
            # Plate cells render their plate glyph; treat them as walkable.
            if terrain in PLATE_GLYPHS:
                self._player_pos = (nx, ny)
                plate_glyph = terrain
                expected_colour = self._cue_sequence[len(self._stepped_sequence)]
                actual_colour = self._plate_to_color[plate_glyph]
                self._stepped_sequence.append(actual_colour)
                if actual_colour != expected_colour:
                    self._message = (
                        f"You step on {PLATE_NAMES[plate_glyph]} ({CUE_NAMES[actual_colour]})."
                        f" The wrong order — the seal punishes you."
                    )
                    terminated = True
                    reward = -1.0
                    info["wrong_sequence"] = True
                    return self._render_current_observation(), reward, terminated, False, info
                if len(self._stepped_sequence) == len(self._cue_sequence):
                    self._message = (
                        f"You step on {PLATE_NAMES[plate_glyph]}. "
                        "The full sequence is correct! You descend."
                    )
                    terminated = True
                    reward = 1.0
                    info["goal_reached"] = True
                    return self._render_current_observation(), reward, terminated, False, info
                self._message = (
                    f"You step on {PLATE_NAMES[plate_glyph]} ({CUE_NAMES[actual_colour]})."
                )
                return self._render_current_observation(), 0.0, False, False, info
            elif self._is_walkable(nx, ny):
                self._player_pos = (nx, ny)
                self._message = ""
            elif terrain in (
                "█", "-", "|",
            ):
                # blocked
                self._message = ""

        # Other actions: WAIT / no-op
        return self._render_current_observation(), 0.0, False, False, info

    def _render_current_observation(self) -> GridObservation:
        from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid

        render = make_empty_grid(self._grid_w, self._grid_h, fill=" ")
        symbols: dict[str, str] = {}

        for y in range(self._grid_h):
            for x in range(self._grid_w):
                ch = self._grid[y][x]
                render[y][x] = ch
                if ch == "·":
                    symbols["·"] = "floor"
                elif ch in ("-", "|", "█"):
                    symbols[ch] = "wall"
                elif ch in PLATE_GLYPHS:
                    symbols[ch] = PLATE_NAMES[ch]

        for (mx, my), colour in self._plate_color_markers.items():
            render[my][mx] = colour
            symbols[colour] = f"{CUE_NAMES[colour]} plate colour marker"

        # Cue display: visible only on turns 1, 2, 3
        cue_index = self._turn
        if 0 <= cue_index < len(self._cue_sequence):
            cue = self._cue_sequence[cue_index]
            cx, cy = self._cue_pos
            # Don't overdraw walls / plates
            if 0 <= cx < self._grid_w and 0 <= cy < self._grid_h and self._grid[cy][cx] == "·":
                render[cy][cx] = cue
                symbols[cue] = f"{CUE_NAMES[cue]} flashing cue"

        # Player on top
        px, py = self._player_pos
        render[py][px] = "@"
        symbols["@"] = "you"

        legend = build_legend(symbols)

        cue_status = "(no cue)"
        if 0 <= cue_index < len(self._cue_sequence):
            cue = self._cue_sequence[cue_index]
            cue_status = f"Cue {cue_index + 1}/3: {CUE_NAMES[cue]} ({cue})"
        elif cue_index >= len(self._cue_sequence):
            cue_status = "Cues finished. Step plates in colour order."

        progress = (
            f"Plate progress: {len(self._stepped_sequence)}/{len(self._cue_sequence)}"
        )
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"{cue_status}    {progress}"
        )
        return GridObservation(
            grid=grid_to_string(render),
            legend=legend,
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            f"You are in the start room. Three coloured cues ({', '.join(CUE_GLYPHS)}) "
            "flash one at a time in the initial observation and the next two turns — "
            "REMEMBER the order. The final room (east) contains three numbered "
            "plates (①②③), each marked by an adjacent colour glyph showing that "
            "plate's colour. After viewing all three cues, walk east and step on "
            "the numbered plates in the SAME COLOUR ORDER as the cues. Reward: +1 on correct "
            "full sequence, -1 on first wrong plate."
        )


class MiniHackMementoShortEnv(_MementoShortBase):
    """A2 sequence-of-cues memory."""

    _num_rooms = 2
    _num_floors = 1

    def env_id(self) -> str:
        return "glyphbench/minihack-memento-short-v0"


# ---------------------------------------------------------------------------
# Hard: A6 counted memory
# ---------------------------------------------------------------------------


class _MementoHardBase(MiniHackBase):
    """Counted memory: count torches passed in the corridor; final room asks."""

    def __init__(self, max_turns: int = 300) -> None:
        super().__init__(max_turns=max_turns)
        self._torch_count: int = 0
        self._answer_positions: dict[tuple[int, int], int] = {}

    def _generate_level(self, seed: int) -> None:
        # Three rooms in a line: start, mid (corridor with torches), final.
        room_w = 5
        # 3 rooms separated by 1-cell walls + the mid room is wider as corridor
        # We use a layout: start (5x5) | corridor with torches (9 cells wide at midline) | final (5x5)
        corridor_len = 7
        total_w = (room_w + 2) + corridor_len + (room_w + 1)
        total_h = room_w + 2
        self._init_grid(total_w, total_h)
        self._dark = True
        self._vision_radius = 2

        # Fill interior with walls
        for y in range(1, total_h - 1):
            for x in range(1, total_w - 1):
                self._place_wall(x, y)

        # Carve start room
        for y in range(1, room_w + 1):
            for x in range(1, room_w + 1):
                self._grid[y][x] = "·"

        # Carve corridor (1-tile high at midline) with torches
        cy = total_h // 2
        corridor_x0 = room_w + 1
        for x in range(corridor_x0, corridor_x0 + corridor_len + 1):
            self._grid[cy][x] = "·"

        # Carve final room
        final_x0 = corridor_x0 + corridor_len + 1
        for y in range(1, room_w + 1):
            for x in range(final_x0, final_x0 + room_w):
                if x < total_w - 1:
                    self._grid[y][x] = "·"

        # Place torches in corridor: random count between 2 and 5
        torch_count = int(self.rng.integers(2, 6))
        self._torch_count = torch_count
        # Randomise which corridor cells get torches (excluding endpoints)
        corridor_cells = list(range(corridor_x0 + 1, corridor_x0 + corridor_len))
        rng_indices = list(self.rng.permutation(len(corridor_cells)))[:torch_count]
        for ti in rng_indices:
            cx = corridor_cells[ti]
            self._grid[cy][cx] = TORCH_GLYPH

        # Player at left middle of start room
        self._place_player(1, cy)
        # Goal: NO terrain stairs. Goal is the answer plate.
        self._goal_pos = None

        # Place 6 numbered answer plates in final room (positions 1..6)
        # Layout in final room: top row plates ➀ ➁ ➂, bottom row plates ➃ ➄ ➅
        nums = list(NUM_GLYPHS)
        positions = []
        for ny_off in (1, 3):
            for nx_off in (0, 2, 4):
                positions.append((final_x0 + nx_off, ny_off))
        self._answer_positions = {}
        for i, pos in enumerate(positions):
            x, y = pos
            if 1 <= x < total_w - 1 and 1 <= y < total_h - 1:
                self._grid[y][x] = nums[i]
                self._answer_positions[(x, y)] = i + 1

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        info: dict[str, Any] = {}
        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy
            terrain = self._terrain_at(nx, ny)
            if terrain in NUM_GLYPHS:
                self._player_pos = (nx, ny)
                value = self._answer_positions.get((nx, ny), 0)
                if value == self._torch_count:
                    self._message = (
                        f"You step on {NUM_NAMES[terrain]}. "
                        f"Correct — there were {self._torch_count} torches."
                    )
                    return self._render_current_observation(), 1.0, True, False, {
                        "goal_reached": True
                    }
                self._message = (
                    f"You step on {NUM_NAMES[terrain]}. The scribe shakes her head."
                )
                return self._render_current_observation(), -1.0, True, False, {
                    "wrong_answer": True,
                    "expected": self._torch_count,
                    "given": value,
                }
            elif terrain == TORCH_GLYPH:
                # Walking past a torch consumes it (so the agent can't recount).
                self._player_pos = (nx, ny)
                self._grid[ny][nx] = "·"
                self._message = "You walk past a torch (ψ)."
                return self._render_current_observation(), 0.0, False, False, info
            elif self._is_walkable(nx, ny):
                self._player_pos = (nx, ny)
                self._message = ""
                return self._render_current_observation(), 0.0, False, False, info
            else:
                self._message = ""
                return self._render_current_observation(), 0.0, False, False, info
        # Other actions: no-op
        return self._render_current_observation(), 0.0, False, False, info

    def _render_current_observation(self) -> GridObservation:
        from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid

        render = make_empty_grid(self._grid_w, self._grid_h, fill="?")
        symbols: dict[str, str] = {"?": "unseen cell outside current vision"}
        px, py = self._player_pos

        for y in range(self._grid_h):
            for x in range(self._grid_w):
                if (
                    abs(x - px) > self._vision_radius
                    or abs(y - py) > self._vision_radius
                ):
                    continue
                ch = self._grid[y][x]
                render[y][x] = ch
                if ch == "·":
                    symbols["·"] = "floor"
                elif ch in ("-", "|", "█"):
                    symbols[ch] = "wall"
                elif ch == TORCH_GLYPH:
                    symbols[ch] = "torch"
                elif ch in NUM_GLYPHS:
                    symbols[ch] = NUM_NAMES[ch]

        render[py][px] = "@"
        symbols["@"] = "you"

        legend = build_legend(symbols)

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            "Vision: limited    Walked past torches: count them as you go."
        )
        return GridObservation(
            grid=grid_to_string(render),
            legend=legend,
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Walk east through the corridor. Torches (ψ) are scattered along the "
            "corridor row — COUNT them as you walk past (each torch is consumed "
            "after you walk over it). Your vision is limited, so distant torches "
            "are hidden until you approach them. In the final room you will see numbered "
            "answer plates (➀➁➂➃➄➅). Step on the plate matching the exact torch "
            "count. Reward: +1 correct count, -1 wrong count."
        )


class MiniHackMementoHardEnv(_MementoHardBase):
    """A6 counted-memory."""

    _num_rooms = 3
    _num_floors = 1

    def env_id(self) -> str:
        return "glyphbench/minihack-memento-hard-v0"


# ---------------------------------------------------------------------------
# F2: A7 spatial memory
# ---------------------------------------------------------------------------


class _MementoF2Base(MiniHackBase):
    """Spatial memory: remember a marker's tile, return there after descending.

    Floor 1: one start room with a marker (✦) at a random tile. Walk east
    to the stairs (⇣) to descend. Floor 2 looks identical EXCEPT the marker
    is gone — the agent must walk to the same (x, y) tile.
    """

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._floor: int = 1
        self._marker_pos: tuple[int, int] = (0, 0)
        self._target_room_origin: tuple[int, int] = (0, 0)

    def _generate_level(self, seed: int) -> None:
        self._floor = 1
        self._generate_floor()

    def _generate_floor(self) -> None:
        start_rect, final_rect, cy = _carve_two_room_layout(self)
        # Player at left of start room
        self._place_player(1, cy)
        if self._floor == 1:
            # Pick a random non-player tile in start room for marker
            candidates = [
                (x, y)
                for y in range(start_rect[1], start_rect[3] + 1)
                for x in range(start_rect[0], start_rect[2] + 1)
                if (x, y) != self._player_pos
            ]
            idx = int(self.rng.integers(0, len(candidates)))
            self._marker_pos = candidates[idx]
            mx, my = self._marker_pos
            self._grid[my][mx] = MARKER_GLYPH
            # Stairs in final room middle
            self._place_stairs(final_rect[0] + 2, cy)
        else:
            # Floor 2: no visible marker, but the same room dimensions.
            # The "answer tile" is the same coordinates as the floor-1 marker.
            # Floor 2 uses no stairs; the answer is submitted with WAIT.
            self._goal_pos = None

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        info: dict[str, Any] = {"floor": self._floor}

        # Floor 1: standard movement; reaching stairs descends.
        if self._floor == 1:
            obs, reward, terminated, truncated, info1 = super()._step(action)
            info1.update(info)
            if terminated and info1.get("goal_reached"):
                # Descend to floor 2 — DON'T give reward yet.
                self._floor = 2
                self._creatures = []
                self._floor_items = {}
                self._inventory = []
                self._generate_floor()
                self._message = (
                    "You descend to floor 2. Move to the remembered marker tile and WAIT."
                )
                info1["goal_reached"] = False
                info1["floor"] = self._floor
                return self._render_current_observation(), 0.0, False, False, info1
            return obs, reward, terminated, truncated, info1

        # Floor 2: move to the remembered tile, then WAIT to submit exactly
        # one answer. This prevents sweep-to-win while keeping all marker
        # positions reachable.
        if name == "WAIT":
            if self._player_pos == self._marker_pos:
                self._message = (
                    "You submit the floor-1 marker tile. The seal opens."
                )
                return self._render_current_observation(), 1.0, True, False, {
                    "goal_reached": True,
                    "floor": self._floor,
                }
            self._message = "Wrong tile. The remembered seal stays closed."
            return self._render_current_observation(), 0.0, True, False, {
                "goal_reached": False,
                "floor": self._floor,
            }

        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy
            if self._is_walkable(nx, ny):
                self._player_pos = (nx, ny)
                self._message = "Choose this tile with WAIT when you are sure."
        return self._render_current_observation(), 0.0, False, False, info

    def _render_current_observation(self) -> GridObservation:
        from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid

        render = make_empty_grid(self._grid_w, self._grid_h, fill=" ")
        symbols: dict[str, str] = {}

        for y in range(self._grid_h):
            for x in range(self._grid_w):
                ch = self._grid[y][x]
                render[y][x] = ch
                if ch == "·":
                    symbols["·"] = "floor"
                elif ch in ("-", "|", "█"):
                    symbols[ch] = "wall"
                elif ch == MARKER_GLYPH:
                    symbols[ch] = "marker"
                elif ch == "⇣":
                    symbols[ch] = "stairs down (descend)"

        px, py = self._player_pos
        render[py][px] = "@"
        symbols["@"] = "you"

        legend = build_legend(symbols)

        if self._floor == 1:
            instruction = "Floor 1: remember the marker tile, then descend via ⇣."
        else:
            instruction = "Floor 2: move to the remembered marker tile, then WAIT once."
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Floor: {self._floor}/2    {instruction}"
        )
        return GridObservation(
            grid=grid_to_string(render),
            legend=legend,
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Floor 1: a marker glyph (✦) is shown at a random tile in the start "
            "room. MEMORISE its (column, row) position, then walk east through "
            "the corridor to the stairs (⇣) to descend. Floor 2 has the SAME "
            "room layout but the marker is gone — move to the SAME tile, then "
            "WAIT to submit your answer. On floor 2, the first wrong WAIT "
            "submission ends the episode with 0. Reward: +1 for a correct "
            "submission, 0 otherwise."
        )


class MiniHackMementoF2Env(_MementoF2Base):
    """A7 spatial memory."""

    _num_rooms = 2
    _num_floors = 2

    def env_id(self) -> str:
        return "glyphbench/minihack-memento-f2-v0"


# ---------------------------------------------------------------------------
# F4: A4 inverse memory
# ---------------------------------------------------------------------------


class _MementoF4Base(MiniHackBase):
    """Inverse memory: a colour cue (Ⓡ/Ⓖ/Ⓑ) is visible in EVERY room except
    the final fog room. The final fog room has three coloured doors; the
    agent must enter the door whose colour matches the cue.
    """

    def __init__(self, max_turns: int = 300) -> None:
        super().__init__(max_turns=max_turns)
        self._target_color: str = ""
        self._door_positions: dict[tuple[int, int], str] = {}
        self._in_fog_room: bool = False

    def _generate_level(self, seed: int) -> None:
        # Three rooms in a line: start, mid, final-fog.
        room_w = 5
        total_w = (room_w + 1) * 3 + 1
        total_h = room_w + 2
        self._init_grid(total_w, total_h)

        for y in range(1, total_h - 1):
            for x in range(1, total_w - 1):
                self._place_wall(x, y)

        cy = total_h // 2
        # Carve 3 rooms separated by 1-cell corridors.
        for i in range(3):
            rx = (room_w + 1) * i + 1
            for y in range(1, room_w + 1):
                for x in range(rx, rx + room_w):
                    if x < total_w - 1:
                        self._grid[y][x] = "·"
        # Corridors between rooms
        for i in range(2):
            cx = (room_w + 1) * (i + 1)
            self._grid[cy][cx] = "·"

        # Choose target colour
        colours = list(COLOR_DOOR_GLYPHS)
        idx = int(self.rng.integers(0, len(colours)))
        self._target_color = colours[idx]

        # Place ALL THREE colour glyphs in the start/mid rooms (one each)
        # so a naive frequency-count cannot identify the target.
        # The CUE_MARKER glyph (★) sits adjacent to the TARGET colour glyph
        # only — the agent must locate the marker and read its neighbour.
        start_row = 1  # top row of rooms
        # Three cue cells: one in start room, two in mid room (so frequency
        # of each colour is 1 in the cue area). Random colour-to-cell mapping.
        cue_cells = [
            (2, start_row),                       # start room
            (room_w + 1 + 1, start_row),          # mid room left
            (room_w + 1 + 3, start_row),          # mid room right
        ]
        cue_order = list(self.rng.permutation(len(colours)))
        self._cue_cells: dict[tuple[int, int], str] = {}
        for cell, c_idx in zip(cue_cells, cue_order, strict=True):
            cx, cy_cell = cell
            colour_glyph = colours[c_idx]
            self._grid[cy_cell][cx] = colour_glyph
            self._cue_cells[(cx, cy_cell)] = colour_glyph

        # Place ★ marker BELOW the cue cell whose colour is the target.
        # The marker is a single occurrence (no colour bias).
        target_cue_pos = next(
            pos for pos, c in self._cue_cells.items() if c == self._target_color
        )
        marker_pos = (target_cue_pos[0], target_cue_pos[1] + 1)
        # Ensure marker is on floor (not on wall/corridor)
        if 1 <= marker_pos[0] < total_w - 1 and 1 <= marker_pos[1] < total_h - 1:
            if self._grid[marker_pos[1]][marker_pos[0]] == "·":
                self._grid[marker_pos[1]][marker_pos[0]] = CUE_MARKER_GLYPH
                self._marker_pos = marker_pos
            else:
                # Fallback: place marker to the right of cue
                alt = (target_cue_pos[0] + 1, target_cue_pos[1])
                if (
                    1 <= alt[0] < total_w - 1
                    and self._grid[alt[1]][alt[0]] == "·"
                ):
                    self._grid[alt[1]][alt[0]] = CUE_MARKER_GLYPH
                    self._marker_pos = alt
                else:
                    self._marker_pos = target_cue_pos  # rare fallback
        else:
            self._marker_pos = target_cue_pos

        # Place player on left of start room
        self._place_player(1, cy)
        self._goal_pos = None

        # Final room: place three doors in a vertical line
        final_x0 = (room_w + 1) * 2 + 1
        # Doors in middle-left column of final room
        # Random shuffle of door order
        door_order = list(self.rng.permutation(len(colours)))
        door_xs = [final_x0 + 2]  # middle column of final room
        door_ys = [1, cy, room_w]  # top, middle, bottom
        self._door_positions = {}
        for i, didx in enumerate(door_order):
            dx = door_xs[0]
            dy = door_ys[i]
            colour_glyph = colours[didx]
            self._grid[dy][dx] = colour_glyph
            self._door_positions[(dx, dy)] = colour_glyph

        self._fog_room_x0 = final_x0
        self._in_fog_room = False

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        info: dict[str, Any] = {}
        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy
            terrain = self._terrain_at(nx, ny)
            # The colour-door glyphs (Ⓡ/Ⓖ/Ⓑ) are reused as the recall CUE
            # markers in the start/middle rooms. Only treat them as the decision
            # door inside the fog room, so stepping onto a cue cell does not
            # spuriously end the episode.
            if (
                terrain in COLOR_DOOR_GLYPHS
                and self._fog_room_x0 is not None
                and nx >= self._fog_room_x0
            ):
                self._player_pos = (nx, ny)
                if terrain == self._target_color:
                    self._message = (
                        f"You step through the {COLOR_DOOR_NAMES[terrain]}. "
                        "It matches the cue. You descend."
                    )
                    return self._render_current_observation(), 1.0, True, False, {
                        "goal_reached": True
                    }
                self._message = (
                    f"You step through the {COLOR_DOOR_NAMES[terrain]}. "
                    "Wrong colour — the seal traps you."
                )
                return self._render_current_observation(), -1.0, True, False, {
                    "wrong_door": True
                }
            elif self._is_walkable(nx, ny):
                self._player_pos = (nx, ny)
                self._message = ""
                # Update fog state
                self._in_fog_room = nx >= self._fog_room_x0
                return self._render_current_observation(), 0.0, False, False, info
            else:
                self._message = ""
                return self._render_current_observation(), 0.0, False, False, info
        return self._render_current_observation(), 0.0, False, False, info

    def _render_current_observation(self) -> GridObservation:
        from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid

        render = make_empty_grid(self._grid_w, self._grid_h, fill=" ")
        symbols: dict[str, str] = {}

        for y in range(self._grid_h):
            for x in range(self._grid_w):
                ch = self._grid[y][x]
                # When the player is inside the fog room, the cue/marker
                # tiles in earlier rooms are HIDDEN (rendered as floor) —
                # the agent cannot peek back at the cue.
                if self._in_fog_room and x < self._fog_room_x0 and (ch in COLOR_DOOR_GLYPHS or ch == CUE_MARKER_GLYPH):
                    ch = "·"
                render[y][x] = ch
                if ch == "·":
                    symbols["·"] = "floor"
                elif ch in ("-", "|", "█"):
                    symbols[ch] = "wall"
                elif ch in COLOR_DOOR_GLYPHS:
                    if x >= self._fog_room_x0:
                        symbols[ch] = COLOR_DOOR_NAMES[ch]
                    else:
                        symbols[ch] = (
                            f"colour glyph ({COLOR_DOOR_NAMES[ch].split()[0]})"
                        )
                elif ch == CUE_MARKER_GLYPH:
                    symbols[ch] = "cue marker (the cue is the colour glyph it sits next to)"

        px, py = self._player_pos
        render[py][px] = "@"
        symbols["@"] = "you"

        legend = build_legend(symbols)

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"In fog room: {self._in_fog_room}    "
            "(★ marks the cue; cue+marker hidden once you enter fog room.)"
        )
        return GridObservation(
            grid=grid_to_string(render),
            legend=legend,
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Three colour glyphs (Ⓡ/Ⓖ/Ⓑ) — one of each — are scattered in the "
            "start and middle rooms. A ★ marker sits next to ONE of them: that "
            "glyph is the CUE colour. As soon as you enter the final fog room, "
            "the cue glyphs and ★ marker disappear from view — memorise the cue "
            "colour BEFORE entering. The fog room has three coloured doors "
            "(Ⓡ/Ⓖ/Ⓑ) — step into the door whose colour matches the cue. "
            "Reward: +1 correct door, -1 wrong door. Note: simple frequency "
            "counts are useless — every colour appears the same number of times "
            "in the initial layout."
        )


class MiniHackMementoF4Env(_MementoF4Base):
    """A4 inverse memory."""

    _num_rooms = 3
    _num_floors = 4

    def env_id(self) -> str:
        return "glyphbench/minihack-memento-f4-v0"
