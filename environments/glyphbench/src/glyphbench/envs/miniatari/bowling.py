"""miniatari Bowling.

Identity: Side-view bowling lane; aim and roll a ball at 10 pins.
Win condition: knock down all 20 pins across 2 frames.
Reward: Pattern N3 (stake doubling) — terminal-only. The +1.0 reward
fires ONLY on a full 20/20 clear. Partial knockdowns grant nothing,
so a sloppy game scores 0 instead of accumulating per-pin progress.
Loss: time runs out or all 4 rolls in both frames are spent without
clearing all pins.

Gym ID: glyphbench/miniatari-bowling-v0
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniBowlingEnv(MiniatariBase):
    """Mini Bowling: 12-wide x 10-tall side-view lane, 2 frames.

    The bowler sits at row 8. Pins occupy a tight triangle in rows 1..4
    spanning columns 3..6, with adjacent columns so the single-hop
    horizontal cascade can chain across the rack. The bowler moves
    LEFT/RIGHT along the foul line. FIRE rolls a ball straight up the
    bowler's column. Pin interactions: the ball hits any pin in its
    column and "dominoes" each knocked pin against its horizontally
    adjacent pins (one cell each side). Knocked pins are removed;
    reward = +1/_TOTAL_PINS per pin. After all 10 pins of a frame fall
    (or 4 rolls in the frame), reset to a fresh pinset for frame 2.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "shift bowler one cell left",
            "shift bowler one cell right",
            "roll the ball straight up your current column",
        ),
    )

    default_max_turns = 200

    _WIDTH = 12
    _HEIGHT = 10
    _BOWLER_Y = 8
    _N_FRAMES = 2
    _PINS_PER_FRAME = 10
    _TOTAL_PINS = 20
    _MAX_ROLLS_PER_FRAME = 4

    # Pin layout: tight triangle in cols 3..6 within rows 1..4. Adjacent
    # columns let the single-hop horizontal cascade chain across the rack
    # so reaching the 20-pin (2-frame) target with 4 rolls/frame is
    # mechanically possible.
    _PIN_LAYOUT: tuple[tuple[int, int], ...] = (
        (5, 1),                                  # apex (1)
        (4, 2), (5, 2),                          # row 2 (2)
        (4, 3), (5, 3), (6, 3),                  # row 3 (3)
        (3, 4), (4, 4), (5, 4), (6, 4),          # row 4 (4)
    )

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._pins: set[tuple[int, int]] = set()
        self._frame: int = 1
        self._rolls_in_frame: int = 0
        self._progress: int = 0  # total pins knocked across all frames

    def env_id(self) -> str:
        return "glyphbench/miniatari-bowling-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._frame = 1
        self._rolls_in_frame = 0
        self._pins = set(self._PIN_LAYOUT)
        self._player_x = self._WIDTH // 2
        self._player_y = self._BOWLER_Y

    def _start_new_frame(self) -> None:
        self._frame += 1
        self._rolls_in_frame = 0
        if self._frame <= self._N_FRAMES:
            self._pins = set(self._PIN_LAYOUT)

    def _knock_pins_in_column(self, col: int) -> int:
        """Knock down pins in `col`, plus one horizontally adjacent pin per
        knocked pin (cascade effect, single hop only). Returns count."""
        knocked: set[tuple[int, int]] = set()
        # Direct hits
        for px, py in list(self._pins):
            if px == col:
                knocked.add((px, py))
        # Cascade: each knocked pin can spread to its horizontal neighbors
        spread: set[tuple[int, int]] = set()
        for kx, ky in knocked:
            for dx in (-1, 1):
                neighbor = (kx + dx, ky)
                if neighbor in self._pins:
                    spread.add(neighbor)
        knocked.update(spread)
        for k in knocked:
            self._pins.discard(k)
        return len(knocked)

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # 1. Bowler movement
        if action_name == "LEFT" and self._player_x > 0:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 1:
            self._player_x += 1
            self._player_dir = (1, 0)

        # 2. Roll. Reward fires only on a full game clear (N3 stake
        # doubling): partial knockdowns give nothing.
        if action_name == "FIRE":
            if self._frame > self._N_FRAMES:
                # No more frames; ignore
                return reward, self._game_over, info
            self._rolls_in_frame += 1
            count = self._knock_pins_in_column(self._player_x)
            self._progress += count
            self._message = f"Roll: {count} pin{'s' if count != 1 else ''} knocked!"
            # End frame conditions: no pins left or max rolls reached
            if not self._pins or self._rolls_in_frame >= self._MAX_ROLLS_PER_FRAME:
                self._start_new_frame()
                if self._frame > self._N_FRAMES:
                    # Out of frames — end the run; reward only on clean clear.
                    self._game_over = True
                    self._won = self._progress >= self._TOTAL_PINS
                    if self._won:
                        reward = 1.0
                        self._message = "PERFECT GAME — 20/20!"
                    else:
                        self._message = (
                            f"Game over — only {self._progress}/{self._TOTAL_PINS} pins."
                        )
                    return reward, True, info

        info["progress"] = self._progress
        info["frame"] = self._frame
        info["rolls_in_frame"] = self._rolls_in_frame
        info["pins_left"] = len(self._pins)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Lane gutters (columns 0 and _WIDTH-1)
        for y in range(self._HEIGHT):
            grid[y][0] = "│"
            grid[y][self._WIDTH - 1] = "│"
        # Foul line at row 7
        for x in range(1, self._WIDTH - 1):
            grid[7][x] = "─"
        # Pins
        for px, py in self._pins:
            if 0 <= px < self._WIDTH and 0 <= py < self._HEIGHT:
                grid[py][px] = "P"
        # Bowler
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "lane",
            "│": "gutter wall",
            "─": "foul line",
            "P": "pin",
            "Y": "bowler",
        }

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Pins down: {self._progress}/{self._TOTAL_PINS}    "
            f"Frame: {self._frame}/{self._N_FRAMES}    "
            f"Rolls: {self._rolls_in_frame}/{self._MAX_ROLLS_PER_FRAME}    "
            f"Pins remaining: {len(self._pins)}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Bowling: 12x10 lane, 2 frames of 10 pins each. Pins "
            "(P) are racked in a tight 4-row triangle (1-2-3-4) in rows "
            "1..4 across columns 3..6. Your bowler (Y) stands at row 8 "
            "(foul line). LEFT/RIGHT slides you along the foul line. "
            "FIRE rolls a ball straight up your current column: any pin "
            "in that column is knocked, plus each horizontally-adjacent "
            "pin per knocked pin (single-hop cascade). A frame ends "
            "after all pins fall OR after 4 rolls; frame 2 spawns a "
            "fresh set. The reward is PERFECT-GAME-ONLY: knocking down "
            "all 20 pins across both frames pays out +1.0 at the end of "
            "frame 2. ANY partial finish (19 or fewer pins) pays out 0. "
            "There is no per-pin partial credit — sloppy rolls earn "
            "nothing."
        )
