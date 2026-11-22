"""Atari Bowling environment.

10-pin bowling. Aim and throw a ball down the lane.

Gym ID: glyphbench/atari-bowling-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase


class BowlingEnv(AtariBase):
    """Bowling: 10-pin bowling with standard scoring.

    20x12 grid. Player aims left/right then throws ball.
    10 frames, 2 rolls each (unless strike).

    Actions: NOOP, AIM_LEFT, AIM_RIGHT, THROW
    Pattern A: +1/_WIN_TARGET per pin knocked down (full-scope = 10
    frames x 10 pins = 100). No failure penalty.
    """

    action_spec = ActionSpec(
        names=("NOOP", "AIM_LEFT", "AIM_RIGHT", "THROW"),
        descriptions=(
            "do nothing",
            "aim one column left",
            "aim one column right",
            "throw the ball",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 12
    _LANE_LEFT = 3
    _LANE_RIGHT = 16
    _PLAYER_ROW = 10
    _PIN_ROW = 1

    # Pattern A full-scope target: 100 pins (10 frames x 10 pins).
    _WIN_TARGET: int = 100

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._aim_x: int = 0
        self._frame: int = 0
        self._roll_in_frame: int = 0
        self._frame_scores: list[list[int]] = []
        self._pins: list[bool] = []
        self._ball_active: bool = False
        self._ball_x: int = 0
        self._ball_y: int = 0
        self._total_frames: int = 10
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-bowling-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._frame = 0
        self._roll_in_frame = 0
        self._frame_scores = [[] for _ in range(self._total_frames)]
        self._ball_active = False
        self._aim_x = self._WIDTH // 2
        self._player_x = self._aim_x
        self._player_y = self._PLAYER_ROW
        self._reset_pins()
        self._redraw()

    def _reset_pins(self) -> None:
        # 10 pins in triangle: row0=1, row1=2, row2=3, row3=4
        self._pins = [True] * 10

    def _pin_positions(self) -> list[tuple[int, int]]:
        cx = self._WIDTH // 2
        positions = []
        # Row 0: 1 pin
        positions.append((cx, self._PIN_ROW))
        # Row 1: 2 pins
        positions.append((cx - 1, self._PIN_ROW + 1))
        positions.append((cx + 1, self._PIN_ROW + 1))
        # Row 2: 3 pins
        positions.append((cx - 2, self._PIN_ROW + 2))
        positions.append((cx, self._PIN_ROW + 2))
        positions.append((cx + 2, self._PIN_ROW + 2))
        # Row 3: 4 pins
        positions.append((cx - 3, self._PIN_ROW + 3))
        positions.append((cx - 1, self._PIN_ROW + 3))
        positions.append((cx + 1, self._PIN_ROW + 3))
        positions.append((cx + 3, self._PIN_ROW + 3))
        return positions

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        if self._frame >= self._total_frames:
            return 0.0, True, info

        if self._ball_active:
            # Ball is rolling; advance it
            self._ball_y -= 1
            if self._ball_y <= self._PIN_ROW + 3:
                knocked = self._resolve_roll()
                # Pattern A: cap progress at _WIN_TARGET.
                granted = min(
                    knocked, self._WIN_TARGET - self._progress_count
                )
                if granted > 0:
                    reward = granted / self._WIN_TARGET
                    self._progress_count += granted
                self._ball_active = False
                self._advance_frame(knocked)
            self._redraw()
            return reward, self._frame >= self._total_frames, info

        # Aiming phase
        if action_name == "AIM_LEFT":
            self._aim_x = max(
                self._LANE_LEFT + 1, self._aim_x - 1
            )
            self._player_x = self._aim_x
        elif action_name == "AIM_RIGHT":
            self._aim_x = min(
                self._LANE_RIGHT - 1, self._aim_x + 1
            )
            self._player_x = self._aim_x
        elif action_name == "THROW":
            self._ball_active = True
            self._ball_x = self._aim_x
            self._ball_y = self._PLAYER_ROW - 1

        self._redraw()
        return reward, False, info

    def _resolve_roll(self) -> int:
        """Knock down pins near the ball's x position."""
        knocked = 0
        positions = self._pin_positions()
        for i, (px, _py) in enumerate(positions):
            if self._pins[i] and abs(px - self._ball_x) <= 1:
                self._pins[i] = False
                knocked += 1
        return knocked

    def _advance_frame(self, knocked: int) -> None:
        if self._frame >= self._total_frames:
            return
        self._frame_scores[self._frame].append(knocked)
        self._on_point_scored(knocked)

        standing = sum(self._pins)
        is_strike = (
            self._roll_in_frame == 0 and standing == 0
        )
        is_spare = (
            self._roll_in_frame == 1 and standing == 0
        )

        if is_strike:
            self._message = "STRIKE!"
            self._frame += 1
            self._roll_in_frame = 0
            self._reset_pins()
        elif self._roll_in_frame == 0:
            self._roll_in_frame = 1
        else:
            if is_spare:
                self._message = "Spare!"
            self._frame += 1
            self._roll_in_frame = 0
            self._reset_pins()

        if self._frame >= self._total_frames:
            self._message = (
                f"Game over! Final score: {self._score}"
            )
            self._game_over = True

    def _redraw(self) -> None:
        # Clear
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")

        # Lane borders
        for y in range(self._HEIGHT):
            self._set_cell(self._LANE_LEFT, y, "│")
            self._set_cell(self._LANE_RIGHT, y, "│")

        # Gutters
        for y in range(self._HEIGHT):
            for x in range(self._LANE_LEFT):
                self._set_cell(x, y, "~")
            for x in range(self._LANE_RIGHT + 1, self._WIDTH):
                self._set_cell(x, y, "~")

        # Draw pins
        positions = self._pin_positions()
        for i, (px, py) in enumerate(positions):
            if self._pins[i]:
                self._set_cell(px, py, "I")

        # Draw ball
        if (
            self._ball_active
            and 0 <= self._ball_x < self._WIDTH
            and 0 <= self._ball_y < self._HEIGHT
        ):
            self._set_cell(
                self._ball_x, self._ball_y, "o"
            )

    def _advance_entities(self) -> None:
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "│": "lane border",
            "~": "gutter",
            "I": "pin",
            "o": "ball",
            " ": "lane",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        f = min(self._frame + 1, self._total_frames)
        r = self._roll_in_frame + 1
        standing = sum(self._pins)
        extra = (
            f"Frame: {f}/10  Roll: {r}"
            f"  Pins: {standing}/10"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "10-pin bowling. Use AIM_LEFT/AIM_RIGHT to "
            "position your aim, then THROW. "
            "Knock down all pins for a strike. "
            "10 frames total."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Bowling.\n\n"
            "TASK\n"
            "Play a 10-frame game of 10-pin bowling. Aim and throw the "
            "ball to knock down pins. Score as many points as possible "
            "over the 10 frames.\n\n"
            "BOARD\n"
            "20 columns by 12 rows. Lane is columns 3-16 (bounded by "
            "'|'); columns outside are gutters '~'. Pins 'I' stand in a "
            "standard bowling triangle near the head of the lane. Ball "
            "is 'o' once released; you (the bowler) appear at the foot "
            "of the lane at your current aim column.\n\n"
            "MECHANICS\n"
            "While aiming: AIM_LEFT / AIM_RIGHT shift your aim one column "
            "within the lane. THROW releases the ball from the aim "
            "column; the ball rolls straight up one row per step until "
            "it reaches the pin rows. A pin is knocked down if the ball "
            "passes within 1 column of its position (|ball.x - pin.x| "
            "<= 1). Each frame has up to 2 rolls; rolling a strike (all "
            "10 on first throw) skips the second roll. After any roll "
            "the pins that were knocked down stay down for the next "
            "roll; both rolls' pins are reset between frames.\n\n"
            "SCORING\n"
            "Reward each roll = number of pins knocked down on that "
            "roll (0-10). No bonus for strikes or spares beyond the "
            "basic pin counts. Total score accumulates across 10 frames "
            "(max 100 in this simplified model).\n\n"
            "TERMINATION\n"
            "Episode ends after the 10th frame completes (game over).\n\n"
            "HUD\n"
            "Shows score, current frame (1-10), roll number in frame, "
            "and standing pins.\n\n"
            + self.action_spec.render_for_prompt()
        )
