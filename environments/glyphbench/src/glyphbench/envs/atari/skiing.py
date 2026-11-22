"""Atari Skiing environment.

Downhill slalom. Navigate gates, avoid flags.

Gym ID: glyphbench/atari-skiing-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec

from .base import AtariBase


class SkiingEnv(AtariBase):
    """Skiing: downhill slalom racing.

    20x24 viewport scrolling vertically. Navigate between
    gate flags.

    Actions: NOOP, LEFT, RIGHT
    Pattern A: +1/_WIN_TARGET per gate passed; no penalty
    for missed gates. Cumulative reward bound: [0, +1].
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT"),
        descriptions=(
            "go straight",
            "steer left",
            "steer right",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 24
    _COURSE_LEN = 60
    _GATE_SPACING = 6
    _PENALTY_PER_MISS = 5

    # Pattern A full-scope: 10 gates on the course
    # (_COURSE_LEN / _GATE_SPACING). Each passed gate yields
    # +1/10. Missed gates do not penalise reward (only the time
    # display).
    _WIN_TARGET: int = 10

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._gates: list[tuple[int, int]] = []
        self._scroll_y: int = 0
        self._time: int = 0
        self._gates_passed: int = 0
        self._gates_missed: int = 0
        self._total_gates: int = 0
        self._finished: bool = False
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-skiing-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._scroll_y = 0
        self._time = 0
        self._gates_passed = 0
        self._gates_missed = 0
        self._finished = False
        self._lives = 1

        rng = self.rng
        self._gates = []
        self._total_gates = self._COURSE_LEN // self._GATE_SPACING

        for i in range(self._total_gates):
            gate_y = (i + 1) * self._GATE_SPACING
            gate_cx = int(
                rng.integers(4, self._WIDTH - 4)
            )
            self._gates.append((gate_cx, gate_y))

        # Player at top center
        self._player_x = self._WIDTH // 2
        self._player_y = 3
        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        if self._finished:
            return 0.0, True, info

        self._time += 1

        # Player movement (horizontal only)
        if action_name == "LEFT":
            nx = self._player_x - 1
            if nx >= 1:
                self._player_x = nx
                self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx = self._player_x + 1
            if nx < self._WIDTH - 1:
                self._player_x = nx
                self._player_dir = (1, 0)

        # Scroll down (skier moves downhill)
        self._scroll_y += 1

        # Check if passing through a gate row.
        # Audit BROKEN fix: gate width tightened from 5 cells
        # (gcx ± 2) to 2 cells (gcx ± 0..1). On a 20-wide course
        # the random-walk hit rate was 25% before; now it's ~10%
        # so smarter steering matters.
        for _i, (gcx, gy) in enumerate(self._gates):
            # Gate row just scrolled past player
            if gy == self._scroll_y + self._player_y:
                gate_l = gcx
                gate_r = gcx + 1
                if gate_l <= self._player_x <= gate_r:
                    self._gates_passed += 1
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Gate passed!"
                else:
                    # Missed gates only penalise the displayed time;
                    # reward stays bounded.
                    self._gates_missed += 1
                    self._message = "Gate missed!"

        # Check if course complete
        total_distance = (
            self._total_gates * self._GATE_SPACING + 10
        )
        if self._scroll_y >= total_distance:
            self._finished = True
            final = self._time + (
                self._gates_missed * self._PENALTY_PER_MISS
            )
            self._message = (
                f"Finished! Time: {final} "
                f"({self._gates_missed} missed)"
            )

        info["gates_passed"] = self._gates_passed
        info["gates_missed"] = self._gates_missed
        info["time"] = self._time
        self._redraw()
        return reward, self._finished, info

    def _redraw(self) -> None:
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                self._set_cell(x, y, "·")

        # Draw borders (trees)
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "T")
            self._set_cell(self._WIDTH - 1, y, "T")

        # Draw gates visible in viewport.
        # Phase 3: 2-cell gate (gcx, gcx+1) bracketed by flags
        # at gcx-1 (left) and gcx+2 (right).
        for gcx, gy in self._gates:
            screen_y = gy - self._scroll_y + self._player_y
            if 0 <= screen_y < self._HEIGHT:
                gate_l = gcx - 1
                gate_r = gcx + 2
                if 0 <= gate_l < self._WIDTH:
                    self._set_cell(gate_l, screen_y, "[")
                if 0 <= gate_r < self._WIDTH:
                    self._set_cell(gate_r, screen_y, "]")
                # Gate opening between flags (the 2 valid cells).
                for gx in range(gate_l + 1, gate_r):
                    if 0 < gx < self._WIDTH - 1:
                        self._set_cell(gx, screen_y, " ")

        # Draw random trees for atmosphere
        for y in range(self._HEIGHT):
            for x in range(2, self._WIDTH - 2):
                world_y = y + self._scroll_y - self._player_y
                if (
                    self._grid_at(x, y) == "·"
                    and (world_y * 7 + x * 13) % 19 == 0
                ):
                    self._set_cell(x, y, "T")

    def _advance_entities(self) -> None:
        pass

    def _render_current_observation(self, **kw: Any):  # type: ignore[override]
        """Append skiing-specific HUD on top of base HUD.

        Base HUD provides ``Step: T / N`` (X4 cross-cutting).
        """
        from glyphbench.core.observation import GridObservation

        obs = super()._render_current_observation()
        extra = (
            f"Gates: {self._gates_passed}/{self._total_gates}  "
            f"Time: {self._time}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid,
            legend=obs.legend,
            hud=new_hud,
            message=obs.message,
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "·": "snow",
            "T": "tree",
            "[": "left gate flag",
            "]": "right gate flag",
            " ": "gate opening",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Ski downhill through slalom gates. "
            "Steer LEFT/RIGHT to pass between the "
            "[ ] flags. Missing a gate adds a time "
            "penalty. Lower time is better."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Skiing.\n\n"
            "TASK\n"
            "Ski downhill through a slalom course of 10 gates, "
            "passing between the left '[' and right ']' flags. "
            "Time yourself: lower total time wins.\n\n"
            "BOARD\n"
            "20 columns by 24 rows. Snow '.' background, tree 'T' "
            "at side walls and scattered through the course. Each "
            "gate '[..]' is 2 cells wide between the flags, placed "
            "every 6 rows down. You are an arrow glyph fixed near "
            "the top of the viewport. The viewport scrolls down as "
            "you descend.\n\n"
            "MECHANICS\n"
            "You only have lateral controls. LEFT / RIGHT steers "
            "you 1 column; NOOP continues straight. Every step the "
            "world scrolls down 1 row (scroll_y += 1). When a "
            "gate's row reaches your row, you are judged: if your "
            "column is within the 2-cell opening [gcx, gcx+1] it "
            "is 'passed', else 'missed' (5-second penalty).\n\n"
            "SCORING\n"
            "Pattern A: +1/10 reward per gate passed (10 gates on "
            "the course). Missing a gate adds a 5-second time "
            "penalty but does not affect reward. Cumulative "
            "reward bound: [0, +1].\n\n"
            "TERMINATION\n"
            "Episode ends when you reach the bottom of the course "
            "(total_gates * 6 + 10 rows scrolled). Final message "
            "reports the time + penalty total.\n\n"
            "HUD\n"
            "Shows score, gates passed / total, and elapsed time.\n\n"
            + self.action_spec.render_for_prompt()
        )
