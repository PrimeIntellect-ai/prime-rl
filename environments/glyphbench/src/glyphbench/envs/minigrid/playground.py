"""MiniGrid Playground environment — L1 repeat-the-tune.

A 16x16 room with three coloured pads arranged at fixed corners. At episode
start, three coloured cue balls reveal a sequence (e.g. red → green → blue).
After the agent's first action, the cue balls vanish; the agent must then
stand on the pads in the order shown to win.

Stepping on a pad whose colour is NOT the next-in-sequence resets the
sequence progress to 0 (no penalty); the cue is gone, so this is a memory
test.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minigrid.base import MiniGridBase
from glyphbench.envs.minigrid.objects import Ball, Pad, Wall

_PAD_COLORS = ["red", "green", "blue"]


class MiniGridPlaygroundEnv(MiniGridBase):
    """16x16 room — L1 repeat-the-tune."""

    def env_id(self) -> str:
        return "glyphbench/minigrid-playground-v0"

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._target_sequence: list[str] = []
        self._sequence_progress: int = 0
        self._cue_visible: bool = False
        self._cue_positions: list[tuple[int, int]] = []
        self._pad_positions: dict[str, tuple[int, int]] = {}

    def _generate_grid(self, seed: int) -> None:
        self._init_grid(18, 18)

        # Sample a 3-colour permutation as the target sequence.
        colors = list(_PAD_COLORS)
        self.rng.shuffle(colors)
        self._target_sequence = list(colors)
        self._sequence_progress = 0

        # Pads at fixed corners (independent of cue order — agent must
        # remember the cue order, not the pad positions).
        pad_layout = [
            (3, 3, "red"),
            (14, 3, "green"),
            (8, 14, "blue"),
        ]
        self._pad_positions = {}
        for px, py, pcolor in pad_layout:
            self._place_obj(px, py, Pad(color=pcolor))
            self._pad_positions[pcolor] = (px, py)

        # Cue balls at fixed positions in the centre, ordered left-to-right
        # to convey the sequence visually.
        cue_y = 8
        self._cue_positions = []
        for i, color in enumerate(self._target_sequence):
            cx = 7 + i  # 7, 8, 9
            self._place_obj(cx, cue_y, Ball(color=color))
            self._cue_positions.append((cx, cue_y))
        self._cue_visible = True

        # Sprinkle a few decorative wall segments to give the room texture
        # but not too much — agent must traverse to all three pads. Keep the
        # cue row clean so the first observation reads left-to-right.
        for _ in range(2):
            wx = int(self.rng.integers(3, 15))
            wy = int(self.rng.integers(3, 15))
            length = int(self.rng.integers(2, 4))
            horizontal = bool(self.rng.integers(0, 2))
            for j in range(length):
                if horizontal:
                    px, py = wx + j, wy
                else:
                    px, py = wx, wy + j
                if (
                    1 <= px < 17
                    and 1 <= py < 17
                    and py != cue_y
                    and self._get_obj(px, py) is None
                ):
                    self._place_obj(px, py, Wall())

        # Agent at a free interior cell, not on a pad.
        while True:
            ax = int(self.rng.integers(1, 17))
            ay = int(self.rng.integers(1, 17))
            if self._get_obj(ax, ay) is None:
                break
        self._place_agent(ax, ay, int(self.rng.integers(0, 4)))

    # -- step: erase cues + advance sequence ---------------------------

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        pre_pos = self._agent_pos

        # Erase the cue balls after the first action.
        if self._cue_visible:
            for cx, cy in self._cue_positions:
                if isinstance(self._grid[cy][cx], Ball):
                    self._grid[cy][cx] = None
            self._cue_visible = False

        obs, reward, terminated, truncated, info = super()._step(action)
        if terminated:
            return obs, reward, terminated, truncated, info

        # Did the agent newly step onto a pad? Turning, waiting, or bumping
        # while already standing on a pad must not replay that pad hit.
        ax, ay = self._agent_pos
        cell = self._get_obj(ax, ay)
        if self._agent_pos != pre_pos and isinstance(cell, Pad):
            expected = self._target_sequence[self._sequence_progress]
            if cell.color == expected:
                self._sequence_progress += 1
                info["pad_hit"] = cell.color
                if self._sequence_progress >= len(self._target_sequence):
                    info["tune_complete"] = True
                    info["goal_reached"] = True
                    return obs, self._reward_on_goal(), True, False, info
            else:
                # Wrong colour — reset progress (no penalty, but the cue is
                # gone so this is a real memory test).
                self._sequence_progress = 0
                info["pad_hit"] = cell.color
                info["wrong_pad"] = True

        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        return (
            "A 16x16 room with three coloured pads (red, green, blue) at "
            "fixed positions. At episode start, three coloured cue balls "
            "reveal a sequence (e.g. red ball, then green ball, then blue "
            "ball, arranged left-to-right at the centre). After your first "
            "action the cue balls vanish — you must remember the order. "
            "Step onto the pads in that exact order to win. Stepping onto a "
            "wrong-colour pad resets your progress to 0; turning or taking "
            "another non-movement action while already standing on a pad does "
            "not count as another pad step. "
            "Reward = 1 - 0.9 * (steps / max_steps) on completing the tune."
        )
