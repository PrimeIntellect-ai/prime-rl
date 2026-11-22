"""miniatari Centipede.

Identity: Snake-like centipede slithers down toward the player; shoot segments.
Win condition: destroy all 8 segments of a single centipede.
Reward: Pattern A, +1/8 per segment killed.
Loss: centipede head reaches player row (no -1 per Pattern A; just terminates).

Tuning notes (Phase 3 hardening): the head now moves every tick and
each segment has 2 HP — the first FIRE damages it (no reward, glyph
flips to 'c'), the second FIRE destroys it (+1/8). This lifts random
baseline well below +0.4 while keeping a smart agent capable of clean
+1.0 wins.

Gym ID: glyphbench/miniatari-centipede-v0
"""
from __future__ import annotations

from collections import deque
from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniCentipedeEnv(MiniatariBase):
    """Mini Centipede: 16x10 grid, single 8-segment centipede.

    Player at the bottom row (y=9). The centipede starts as a horizontal
    line of 8 segments at the top, snake-style: each segment follows the
    previous segment's prior position. The head moves right, drops one
    row when it hits a wall, then reverses. Shooting a non-head segment
    splits the centipede into two but for simplicity we treat it as a
    list and remove segments by index. FIRE destroys the lowest segment
    in the player's column (instantly). Clear all 8 segments to win.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move blaster left",
            "move blaster right",
            "fire a bullet up your column",
        ),
    )

    default_max_turns = 300

    _WIDTH = 16
    _HEIGHT = 10
    _WIN_TARGET = 8
    _PLAYER_Y = 9
    _START_Y = 0
    _N_SEGMENTS = 8

    _FIRE_COOLDOWN = 3
    _SEGMENT_HP = 2  # each segment takes 2 hits to destroy

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._segments: deque[tuple[int, int]] = deque()  # head=index 0
        # _segment_hp is keyed by index into _segments; rebuilt whenever
        # segments are popped or remapped.
        self._segment_hp: dict[int, int] = {}
        self._head_dx: int = 1
        self._head_dy: int = 0
        self._dropping: bool = False
        self._progress: int = 0
        self._move_every: int = 1  # tick rate: head moves every N ticks
        self._tick_count: int = 0
        self._fire_cd: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-centipede-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        self._fire_cd = 0
        self._head_dx = 1
        self._head_dy = 0
        self._dropping = False
        # Segments: head at column N_SEGMENTS-1, tail at column 0
        self._segments = deque(
            (self._N_SEGMENTS - 1 - i, self._START_Y) for i in range(self._N_SEGMENTS)
        )
        self._segment_hp = {i: self._SEGMENT_HP for i in range(self._N_SEGMENTS)}
        self._player_x = self._WIDTH // 2
        self._player_y = self._PLAYER_Y

    def _move_centipede(self) -> None:
        if not self._segments:
            return
        head_x, head_y = self._segments[0]
        # Compute new head
        if self._dropping:
            # Drop one row, reverse direction, clear dropping
            new_head = (head_x, head_y + 1)
            self._head_dx = -self._head_dx
            self._dropping = False
        else:
            new_head = (head_x + self._head_dx, head_y)
            # Hit wall? Schedule drop next move
            if new_head[0] < 0 or new_head[0] >= self._WIDTH:
                # Don't go through wall: drop instead this tick
                new_head = (head_x, head_y + 1)
                self._head_dx = -self._head_dx
        # Snake-style: each segment moves to position of segment ahead
        new_segments = deque([new_head])
        for prev in list(self._segments)[:-1]:
            new_segments.append(prev)
        self._segments = new_segments

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1
        if self._fire_cd > 0:
            self._fire_cd -= 1

        # 1. Player move
        if action_name == "LEFT" and self._player_x > 0:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 1:
            self._player_x += 1
            self._player_dir = (1, 0)

        # 2. Fire (instantaneous column shot — damage lowest segment in column)
        if action_name == "FIRE" and self._fire_cd == 0:
            self._fire_cd = self._FIRE_COOLDOWN
            best: int | None = None
            best_y: int = -1
            for i, (sx, sy) in enumerate(self._segments):
                if sx == self._player_x and sy < self._PLAYER_Y and sy > best_y:
                    best = i
                    best_y = sy
            if best is not None:
                # Apply 1 damage. If HP drops to 0 the segment is removed
                # and reward fires. Otherwise just damage (silent).
                self._segment_hp[best] = self._segment_hp.get(best, self._SEGMENT_HP) - 1
                if self._segment_hp[best] <= 0:
                    # Remove segment at index `best`; remap _segment_hp keys.
                    seg_list = list(self._segments)
                    seg_list.pop(best)
                    self._segments = deque(seg_list)
                    new_hp: dict[int, int] = {}
                    for old_i, hp in self._segment_hp.items():
                        if old_i == best:
                            continue
                        new_idx = old_i if old_i < best else old_i - 1
                        new_hp[new_idx] = hp
                    self._segment_hp = new_hp
                    reward += self._progress_reward(self._WIN_TARGET)
                    self._progress += 1
                    self._message = f"Segment! ({self._progress}/{self._WIN_TARGET})"
                    if self._progress >= self._WIN_TARGET:
                        self._on_won()
                        return reward, self._game_over, info
                else:
                    self._message = "Hit! (segment damaged)"

        # 3. Move centipede every _move_every ticks
        if self._tick_count % self._move_every == 0 and self._segments:
            self._move_centipede()

        # 4. Check head reached player row
        if self._segments:
            head_x, head_y = self._segments[0]
            if head_y >= self._PLAYER_Y:
                self._message = "Centipede reached you!"
                self._game_over = True
                self._won = False
                return reward, True, info

        info["progress"] = self._progress
        info["segments_left"] = len(self._segments)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Player row marker
        for x in range(self._WIDTH):
            grid[self._PLAYER_Y][x] = "_"
        # Segments — head/body, with damaged variants 'h'/'c' when below
        # full HP.
        for i, (sx, sy) in enumerate(self._segments):
            if 0 <= sx < self._WIDTH and 0 <= sy < self._HEIGHT:
                hp = self._segment_hp.get(i, self._SEGMENT_HP)
                if i == 0:
                    grid[sy][sx] = "H" if hp >= self._SEGMENT_HP else "h"
                else:
                    grid[sy][sx] = "C" if hp >= self._SEGMENT_HP else "c"
        # Player
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "empty",
            "_": "your row",
            "H": "centipede head (full HP)",
            "h": "centipede head (damaged, 1 HP)",
            "C": "centipede body (full HP)",
            "c": "centipede body (damaged, 1 HP)",
            "Y": "your blaster",
        }
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Segments killed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Head dx={self._head_dx:+d}    "
            f"Segments: {len(self._segments)}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Centipede on a 16x10 grid. An 8-segment centipede "
            "starts at the top row, sliding right; when its head hits a "
            "wall it drops one row and reverses, then continues. Every "
            "tick all segments follow the head snake-style. The head is "
            "H/h (full/damaged), body segments are C/c. Your blaster (Y) "
            "sits at row 9; LEFT/RIGHT moves it. FIRE damages the lowest "
            "centipede segment in your current column by 1 HP. Each "
            "segment has 2 HP — you must hit the same segment TWICE for "
            "it to die. The first hit flips the glyph to lowercase "
            "(h/c). FIRE has a short cooldown (3 ticks) and can be "
            "queued whenever cd reaches 0. Destroy all 8 segments to "
            "win. If the centipede head reaches your row, the run ends. "
            "Reward: +1/8 per destroyed segment (no reward for damage)."
        )
