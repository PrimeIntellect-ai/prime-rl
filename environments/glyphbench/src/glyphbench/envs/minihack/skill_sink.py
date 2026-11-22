"""MiniHack Sink skill tasks.

Phase 3:
  * sink: randomise sink + passage + player + stairs positions.
  * sink-distract: I1 periodic gate — APPLY only operates on turns where
    turn % 5 == 0 (windows of length 1 every 5 steps); HUD shows the
    countdown.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MiniHackBase


class _SinkBase(MiniHackBase):
    _distract: bool = False
    _sink_used: bool = False

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._sink_pos: tuple[int, int] = (3, 3)
        self._passage_pos: tuple[int, int] = (4, 3)
        self._sink_used = False

    def _generate_level(self, seed: int) -> None:
        self._init_grid(7, 7)
        self._sink_used = False

        # Build a vertical wall at random x in [2, 4] separating left/right
        wall_x = int(self.rng.integers(2, 5))
        for y in range(1, 6):
            self._place_wall(wall_x, y)

        # Player at random position in left half
        left_candidates = [
            (x, y) for y in range(1, 6) for x in range(1, wall_x)
        ]
        idx = int(self.rng.integers(0, len(left_candidates)))
        px, py = left_candidates[idx]
        self._place_player(px, py)

        # Stairs at random position in right half
        right_candidates = [
            (x, y) for y in range(1, 6) for x in range(wall_x + 1, 6)
        ]
        idx = int(self.rng.integers(0, len(right_candidates)))
        self._place_stairs(*right_candidates[idx])

        # Sink at random position in left half (not on player)
        sink_candidates = [c for c in left_candidates if c != (px, py)]
        idx = int(self.rng.integers(0, len(sink_candidates)))
        sx, sy = sink_candidates[idx]
        self._sink_pos = (sx, sy)
        self._grid[sy][sx] = "{"

        # Passage cell at the same row as the sink, on the wall
        # (so APPLYing opens a passage at (wall_x, sy))
        self._passage_pos = (wall_x, sy)

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        if self.action_spec.names[action] == "APPLY":
            px, py = self._player_pos
            sx, sy = self._sink_pos
            if abs(px - sx) <= 1 and abs(py - sy) <= 1:
                # Distract: only allowed when turn % 5 == 0 (i.e. on next-step turn=5,10,...)
                # self._turn is incremented in base.step BEFORE _step is called
                # so self._turn corresponds to the turn that just happened.
                if self._distract and self._turn % 5 != 0:
                    self._message = (
                        f"The sink rattles but does nothing. (Window opens "
                        f"at turn % 5 == 0; current turn={self._turn})"
                    )
                    return self._render_current_observation(), 0.0, False, False, {}
                wx, wy = self._passage_pos
                self._grid[wy][wx] = "·"
                self._sink_used = True
                self._message = "You apply the sink. A hidden passage opens."
                return self._render_current_observation(), 0.0, False, False, {
                    "sink_used": True,
                    "player_pos": self._player_pos,
                    "hp": self._player_hp,
                }
        return super()._step(action)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        if self._distract:
            window = 5 - (self._turn % 5)
            if window == 5:
                window = 0
            extra = (
                f"    APPLY window: opens at turn % 5 == 0 (next window in "
                f"{window} turns)"
            )
            new_hud = obs.hud + extra
            return GridObservation(
                grid=obs.grid,
                legend=obs.legend,
                hud=new_hud,
                message=obs.message,
            )
        return obs

    def _task_description(self) -> str:
        if self._distract:
            return (
                "A wall blocks the stairs (⇣). Stand next to the sink ({) and "
                "use APPLY to open a hidden passage — but the sink only works "
                "when turn % 5 == 0 (every 5th step). Wait for the window. "
                "Reward: +1 on reaching stairs."
            )
        return (
            "A wall blocks the stairs (⇣). Stand on or next to the sink ({) "
            "and use APPLY to open a hidden passage, then navigate through it. "
            "Reward: +1 on reaching stairs."
        )


class MiniHackSinkEnv(_SinkBase):
    """MiniHack Sink: room with a sink terrain."""

    def env_id(self) -> str:
        return "glyphbench/minihack-sink-v0"


class MiniHackSinkDistractEnv(_SinkBase):
    """MiniHack Sink (Distract): I1 periodic gate."""

    _distract = True

    def env_id(self) -> str:
        return "glyphbench/minihack-sink-distract-v0"
