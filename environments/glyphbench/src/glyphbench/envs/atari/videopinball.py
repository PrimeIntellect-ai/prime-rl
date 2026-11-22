"""Atari Video Pinball environment.

Pinball machine with flippers, bumpers, and targets.

Gym ID: glyphbench/atari-videopinball-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class VideoPinballEnv(AtariBase):
    """Video Pinball: pinball with flippers and bumpers.

    12x24 grid. Ball bounces hitting bumpers and targets.
    Use flippers to keep ball in play.

    Actions: NOOP, LEFT_FLIPPER, RIGHT_FLIPPER, NUDGE
    Pattern A: +1/_WIN_TARGET per scoring event (bumper,
    target, or spinner hit). No reward penalty for ball drains.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT_FLIPPER", "RIGHT_FLIPPER", "NUDGE"),
        descriptions=(
            "do nothing",
            "activate left flipper",
            "activate right flipper",
            "nudge the table (shifts ball left)",
        ),
    )

    _WIDTH = 12
    _HEIGHT = 24
    _FLIPPER_Y = 21
    _DRAIN_Y = 22

    # Pattern D full-scope: 100 scoring events (bumpers, targets,
    # spinners). Each event yields +1/100. Cumulative reward
    # bound: [-1, +1]. Audit BROKEN fix: drain is now single-life
    # terminal with reward -1 (was: silent reset, which let
    # random play rack 100 events on a busy board).
    _WIN_TARGET: int = 100
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._ball_x = 0
        self._ball_y = 0
        self._ball_dx = 0
        self._ball_dy = 0
        self._bumpers: list[tuple[int, int]] = []
        self._targets: list[AtariEntity] = []
        self._spinners: list[AtariEntity] = []
        self._step_counter = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-videopinball-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _reset_ball(self) -> None:
        self._ball_x = self._WIDTH // 2
        self._ball_y = self._FLIPPER_Y - 1
        self._ball_dx = int(self.rng.choice([-1, 1]))
        self._ball_dy = -1

    def _spawn_targets(self) -> None:
        rng = self.rng
        self._targets = []
        for _ in range(3 + self._level):
            for _ in range(20):
                tx = int(rng.integers(2, self._WIDTH - 2))
                ty = int(rng.integers(3, 15))
                if (tx, ty) not in self._bumpers:
                    self._targets.append(
                        self._add_entity("target", "X", tx, ty)
                    )
                    break

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._targets = []
        self._spinners = []
        self._bumpers = []
        self._step_counter = 0
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")
        # Flippers and drain
        self._set_cell(3, self._FLIPPER_Y, "\\")
        self._set_cell(4, self._FLIPPER_Y, "_")
        self._set_cell(self._WIDTH - 5, self._FLIPPER_Y, "_")
        self._set_cell(self._WIDTH - 4, self._FLIPPER_Y, "/")
        for x in range(1, self._WIDTH - 1):
            self._set_cell(x, self._DRAIN_Y, "·")
        # Bumpers
        for bx, by in [(4, 6), (7, 6), (3, 10),
                        (8, 10), (5, 8), (6, 12)]:
            if bx < self._WIDTH - 1 and by < self._HEIGHT - 1:
                self._bumpers.append((bx, by))
        # Targets and spinners
        self._spawn_targets()
        for sx in (3, self._WIDTH - 4):
            if 0 < sx < self._WIDTH - 1:
                self._spinners.append(
                    self._add_entity("spinner", "S", sx, 4)
                )
        self._reset_ball()
        # Hide player marker (ball is the focus)
        self._player_x = -1
        self._player_y = -1
        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1
        left_flip = action_name == "LEFT_FLIPPER"
        right_flip = action_name == "RIGHT_FLIPPER"
        nudge = action_name == "NUDGE"
        # Move ball
        nx = self._ball_x + self._ball_dx
        ny = self._ball_y + self._ball_dy
        if nx <= 0 or nx >= self._WIDTH - 1:
            self._ball_dx = -self._ball_dx
            nx = self._ball_x + self._ball_dx
        if ny <= 0:
            self._ball_dy = -self._ball_dy
            ny = self._ball_y + self._ball_dy
        # Flipper collision
        if ny >= self._FLIPPER_Y:
            if left_flip and 2 <= nx <= 5:
                self._ball_dy = -1
                self._ball_dx = 1 if nx >= 4 else -1
                ny = self._FLIPPER_Y - 1
            elif right_flip and (
                self._WIDTH - 6 <= nx <= self._WIDTH - 3
            ):
                self._ball_dy = -1
                self._ball_dx = (
                    -1 if nx <= self._WIDTH - 5 else 1
                )
                ny = self._FLIPPER_Y - 1
            elif ny >= self._DRAIN_Y:
                # Audit BROKEN fix: drain is single-life terminal.
                # Previously the env silently reset the ball, which
                # let random play rack up 100 events.
                self._message = "Ball drained!"
                self._on_life_lost()
                reward = self._DEATH_PENALTY
                self._redraw()
                return reward, True, info
        if nudge:
            nx = max(1, nx - 1)
        self._ball_x = max(1, min(nx, self._WIDTH - 2))
        self._ball_y = max(1, min(ny, self._HEIGHT - 2))
        # Bumper collision
        for bx, by in self._bumpers:
            if self._ball_x == bx and self._ball_y == by:
                self._ball_dx = -self._ball_dx
                self._ball_dy = -self._ball_dy
                self._on_point_scored(1)
                if self._progress_count < self._WIN_TARGET:
                    reward += 1.0 / self._WIN_TARGET
                    self._progress_count += 1
                self._message = "Bumper!"
                self._ball_x += self._ball_dx
                self._ball_y += self._ball_dy
                break
        # Target collision
        for t in self._targets:
            if (t.alive and t.x == self._ball_x
                    and t.y == self._ball_y):
                t.alive = False
                self._ball_dy = -self._ball_dy
                self._on_point_scored(1)
                if self._progress_count < self._WIN_TARGET:
                    reward += 1.0 / self._WIN_TARGET
                    self._progress_count += 1
                self._message = "Target!"
        # Spinner collision
        for sp in self._spinners:
            if (sp.alive and sp.x == self._ball_x
                    and sp.y == self._ball_y):
                self._ball_dy = -self._ball_dy
                self._on_point_scored(1)
                if self._progress_count < self._WIN_TARGET:
                    reward += 1.0 / self._WIN_TARGET
                    self._progress_count += 1
                self._message = "Spinner!"
        # All targets cleared (just respawn — progress already counted)
        alive = [t for t in self._targets if t.alive]
        if not alive and self._targets:
            self._level += 1
            self._message = "All targets cleared!"
            self._spawn_targets()

        # Win check
        if (
            self._progress_count >= self._WIN_TARGET
            and not self._game_over
        ):
            self._game_over = True
            info["won"] = True
            self._message = "Highscore reached!"

        self._targets = [t for t in self._targets if t.alive]
        self._redraw()
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        self._set_cell(3, self._FLIPPER_Y, "\\")
        self._set_cell(4, self._FLIPPER_Y, "_")
        self._set_cell(self._WIDTH - 5, self._FLIPPER_Y, "_")
        self._set_cell(self._WIDTH - 4, self._FLIPPER_Y, "/")
        for x in range(1, self._WIDTH - 1):
            self._set_cell(x, self._DRAIN_Y, "·")
        for bx, by in self._bumpers:
            self._set_cell(bx, by, "O")
        for t in self._targets:
            if t.alive:
                self._set_cell(t.x, t.y, "X")
        for sp in self._spinners:
            if sp.alive:
                self._set_cell(sp.x, sp.y, "S")
        if (1 <= self._ball_x < self._WIDTH - 1
                and 1 <= self._ball_y < self._HEIGHT - 1):
            self._set_cell(self._ball_x, self._ball_y, "o")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall", "│": "wall",
            "\\": "left flipper", "/": "right flipper",
            "_": "flipper", "·": "drain (ball lost here)",
            "O": "bumper (10pts)", "X": "target (50pts)",
            "S": "spinner (100pts)", "o": "ball", " ": "empty",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        ball = f"Ball vel=({self._ball_dx},{self._ball_dy})"
        new_hud = obs.hud + "\n" + ball
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Keep the ball (o) in play using flippers. "
            "Hit bumpers (O) for 10pts, targets (X) for 50pts, "
            "spinners (S) for 100pts. "
            "Ball draining past flippers ends the episode."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Video Pinball.\n\n"
            "TASK\n"
            "Play pinball: keep the ball in play using two flippers "
            "and rack up points by hitting bumpers, targets, and "
            "spinners. Ball draining past the flippers ends the "
            "episode (single-life).\n\n"
            "BOARD\n"
            "12x24 table. Walls '-' and '|' on the borders. Two "
            "flippers sit at the bottom with a drain '·' below "
            "them. Bumpers 'O', targets, and spinners 'S' are "
            "scattered across the playfield (read the [Grid]). "
            "Ball 'o'.\n\n"
            "MECHANICS\n"
            "You do not move a character; the playfield is fixed. "
            "LEFT_FLIPPER / RIGHT_FLIPPER raise the respective "
            "flipper for this step: if the ball is in the flipper "
            "region (columns 2-5 for left, W-6..W-3 for right) at "
            "row >= 21 it is redirected upward with vy=-1 and a "
            "new vx based on hit position. NUDGE shifts the ball "
            "1 column left (up to once per step). Ball bounces off "
            "walls and ceiling; hitting a bumper reverses both "
            "velocities.\n\n"
            "SCORING\n"
            "Pattern D: +1/100 reward per scoring event "
            "(bumper, target, or spinner hit). Targets respawn "
            "after they are all cleared. -1 reward when the ball "
            "drains. Cumulative reward bound: [-1, +1].\n\n"
            "TERMINATION\n"
            "Single-life: ball crossing row 22 (drain) ends the "
            "episode with reward -1. The episode also ends after "
            "100 scoring events (cumulative +1) or after "
            "max_turns.\n\n"
            "HUD\n"
            "Shows score, level, and ball position and "
            "velocity.\n\n"
            + self.action_spec.render_for_prompt()
        )
