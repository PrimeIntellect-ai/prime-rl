"""Flappy bird game.

Gym IDs:
  glyphbench/classics-flappy-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation

WIDTH = 20
HEIGHT = 10
GAP_SIZE = 3

# Target pipes successfully passed for cumulative reward = 1.0. The cap is
# tuned so each clean pass pays a visible +0.1 progress signal.
_TARGET_PIPES = 10

FLAPPY_ACTION_SPEC = ActionSpec(
    names=("FLAP", "NOOP"),
    descriptions=(
        "flap wings to go up",
        "do nothing (fall with gravity)",
    ),
)

SYM_BIRD = "@"
SYM_PIPE = "\u2588"
SYM_EMPTY = "\u00b7"
SYM_GROUND = "\u2550"


class FlappyEnv(BaseGlyphEnv):
    """Flappy bird: flap to navigate through pipe gaps."""

    action_spec = FLAPPY_ACTION_SPEC
    noop_action_name: str = "NOOP"

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._bird_y: float = 0.0
        self._bird_vy: float = 0.0
        self._pipes: list[dict[str, Any]] = []
        self._score: int = 0
        self._alive: bool = True
        self._pipe_timer: int = 0

    def env_id(self) -> str:
        return "glyphbench/classics-flappy-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._bird_y = float(HEIGHT // 2)
        self._bird_vy = 0.0
        self._pipes = []
        self._score = 0
        self._alive = True
        self._pipe_timer = 0
        self._spawn_pipe()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]

        if name == "FLAP":
            self._bird_vy = -2.0
        else:
            self._bird_vy += 1.0

        self._bird_y += self._bird_vy
        self._bird_y = max(0.0, min(float(HEIGHT - 1), self._bird_y))

        for p in self._pipes:
            p["x"] -= 1

        # Reward: +1/_TARGET_PIPES per pipe successfully passed. 0 on idle,
        # 0 on death. Cumulative capped at 1.0 once _TARGET_PIPES are passed.
        reward = 0.0
        for p in self._pipes:
            if p["x"] == 1 and not p.get("scored"):
                p["scored"] = True
                already_scored = self._score
                self._score += 1
                # Cap so cumulative cannot exceed 1.0.
                room = max(0.0, 1.0 - already_scored / _TARGET_PIPES)
                reward += min(1.0 / _TARGET_PIPES, room)

        self._pipes = [p for p in self._pipes if p["x"] >= 0]

        self._pipe_timer += 1
        if self._pipe_timer >= 6:
            self._spawn_pipe()
            self._pipe_timer = 0

        bird_row = int(round(self._bird_y))
        for p in self._pipes:
            if p["x"] == 2:
                gap_top = p["gap_y"]
                gap_bot = gap_top + GAP_SIZE
                if bird_row < gap_top or bird_row >= gap_bot:
                    self._alive = False
                    return self._render_current_observation(), 0.0, True, False, info

        if bird_row >= HEIGHT - 1:
            self._alive = False
            return self._render_current_observation(), 0.0, True, False, info

        info["score"] = self._score
        return self._render_current_observation(), reward, False, False, info

    def _spawn_pipe(self) -> None:
        gap_y = int(self.rng.integers(1, HEIGHT - GAP_SIZE - 1))
        self._pipes.append({"x": WIDTH - 1, "gap_y": gap_y, "scored": False})

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(WIDTH, HEIGHT, SYM_EMPTY)
        for x in range(WIDTH):
            grid[HEIGHT - 1][x] = SYM_GROUND
        for p in self._pipes:
            x = p["x"]
            if 0 <= x < WIDTH:
                gap_top = p["gap_y"]
                gap_bot = gap_top + GAP_SIZE
                for y in range(HEIGHT - 1):
                    if y < gap_top or y >= gap_bot:
                        grid[y][x] = SYM_PIPE
        bird_row = max(0, min(HEIGHT - 1, int(round(self._bird_y))))
        grid[bird_row][2] = SYM_BIRD
        legend = build_legend({
            SYM_BIRD: "bird (you)", SYM_PIPE: "pipe",
            SYM_EMPTY: "empty sky", SYM_GROUND: "ground",
        })
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Score: {self._score}    Vy: {self._bird_vy:.1f}"
        )
        msg = "You crashed!" if not self._alive else ""
        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message=msg)

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\nGuide a bird through gaps in pipes.\n\n"
            "RULES\n"
            f"- Grid is {WIDTH}x{HEIGHT}. Bird at column 2.\n"
            "- Velocity is stateful. NOOP increases downward velocity by 1; "
            "FLAP sets upward velocity to -2.\n"
            f"- Pipe gaps are {GAP_SIZE} rows tall. Pipes spawn every 6 steps "
            "and move left 1 column per step.\n"
            "- Hitting a pipe or ground ends the game.\n\n"
            "SCORING\n"
            f"- Passing a pipe gives +1/{_TARGET_PIPES}; cumulative reward caps "
            "at +1. Death gives 0 reward.\n\n"
            + self.action_spec.render_for_prompt()
        )
