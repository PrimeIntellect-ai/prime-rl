"""MiniGrid DistShift environments.

Original framing (single lava-strip nav with one gap) made the name's
distribution-shift promise unobservable in the eval framework: distshift1
and distshift2 differed only by where the lava sat. They are now repurposed
as B1 (decoy-goal trap):

  * Lava strip with TWO gaps. One gap leads to the real goal; the other
    side hides one or more trap glyphs (☠) that terminate the episode
    with -0.5 reward when stepped on.
  * The agent must observe the goal star to pick the correct gap; choosing
    the wrong gap commits to a costly trap.
  * distshift1 = 1 trap on the decoy side; distshift2 = 2 traps (denser
    decoy zone) — a difficulty curve, not a meaningless x-shift.

Reward bound: max approaches +1 for fast goal reaches; min -0.5
(trap-then-end). Hitting a trap then continuing is impossible because the
trap terminates.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from glyphbench.envs.minigrid.objects import Goal, Lava, Trap


class _DistShiftBase(MiniGridBase):
    _lava_x: int = 4  # x position of lava strip
    _trap_count: int = 1  # number of traps on the decoy side

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._trap_positions: list[tuple[int, int]] = []
        self._goal_gap_y: int = 0
        self._decoy_gap_y: int = 0

    def _generate_grid(self, seed: int) -> None:
        self._init_grid(9, 7)
        self._trap_positions = []

        # Lava strip at _lava_x.
        for y in range(1, 6):
            self._place_obj(self._lava_x, y, Lava())

        # Two gaps: one real (gap to goal) + one decoy (gap to trap).
        gap_options = list(range(1, 6))
        self.rng.shuffle(gap_options)
        goal_gap = gap_options[0]
        decoy_gap = gap_options[1]
        self._goal_gap_y = goal_gap
        self._decoy_gap_y = decoy_gap
        self._grid[goal_gap][self._lava_x] = None
        self._grid[decoy_gap][self._lava_x] = None

        # Goal on the right side, on the goal-gap row but past the lava.
        gx = int(self.rng.integers(self._lava_x + 1, 8))
        self._place_obj(gx, goal_gap, Goal())

        # Place traps on the right side at the decoy-gap row (and adjacent
        # rows for the denser variant). Traps must not block the real gap row:
        # the right side is only two columns wide in DistShift2, so a trap
        # directly past the real gap can make the goal unreachable.
        candidate_trap_xs = list(range(self._lava_x + 1, 8))
        self.rng.shuffle(candidate_trap_xs)
        # Pin the first trap directly past the decoy gap so the agent who
        # picks the wrong gap immediately walks into it.
        primary_trap_x = self._lava_x + 1
        primary_trap_pos = (primary_trap_x, decoy_gap)
        if primary_trap_pos != (gx, goal_gap):
            self._place_obj(*primary_trap_pos, Trap())
            self._trap_positions.append(primary_trap_pos)

        # Add additional traps for the denser variant.
        extras_needed = self._trap_count - len(self._trap_positions)
        for tx in candidate_trap_xs:
            if extras_needed <= 0:
                break
            for ty in (decoy_gap, decoy_gap - 1, decoy_gap + 1):
                if not (1 <= ty < 6):
                    continue
                if ty == goal_gap:
                    continue
                if (tx, ty) == (gx, goal_gap):
                    continue
                if (tx, ty) in self._trap_positions:
                    continue
                if self._get_obj(tx, ty) is not None:
                    continue
                self._place_obj(tx, ty, Trap())
                self._trap_positions.append((tx, ty))
                extras_needed -= 1
                if extras_needed <= 0:
                    break

        # Agent on the left side.
        ax = int(self.rng.integers(1, self._lava_x))
        ay = int(self.rng.integers(1, 6))
        self._place_agent(ax, ay, DIR_RIGHT)

    # -- step: trap detection ---------------------------------------------

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # If the parent terminated (lava or goal), respect it.
        if terminated:
            return obs, reward, terminated, truncated, info
        ax, ay = self._agent_pos
        cell = self._get_obj(ax, ay)
        if isinstance(cell, Trap):
            return obs, -0.5, True, False, {"trap": True}
        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        lava = Lava().render_char()
        goal = Goal().render_char()
        trap = Trap().render_char()
        return (
            f"A vertical strip of lava ({lava}) cuts the grid in two. There "
            f"are TWO gaps in the lava — one leads to the goal ({goal}); the "
            f"other leads to a trap zone marked with skull glyphs ({trap}). "
            "The goal-gap row is free of traps. "
            f"Stepping on a trap ends the episode with -0.5 reward; stepping "
            f"on lava ends with 0 reward. Reach the goal for "
            f"reward = 1 - 0.9 * (steps / max_steps)."
        )


class MiniGridDistShift1Env(_DistShiftBase):
    """Lava with two gaps, single trap on the decoy side."""

    _lava_x = 4
    _trap_count = 1

    def env_id(self) -> str:
        return "glyphbench/minigrid-distshift1-v0"


class MiniGridDistShift2Env(_DistShiftBase):
    """Lava with two gaps, denser trap zone (2 traps) — harder variant."""

    _lava_x = 5
    _trap_count = 2

    def env_id(self) -> str:
        return "glyphbench/minigrid-distshift2-v0"
