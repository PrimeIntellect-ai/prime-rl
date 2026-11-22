"""Procgen Climber environment.

Vertical platformer. Agent climbs upward collecting stars on platforms,
avoiding patrolling enemies, and reaching the top for a big reward.

Gym ID: glyphbench/procgen-climber-v0
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import JUMP_ARC_DY, ProcgenBase


class ClimberEnv(ProcgenBase):
    """Vertical platformer: climb platforms, collect stars, reach the top.

    World: 14 wide x 40 tall.  View: 14 x 20.
    Gravity enabled; agent falls when not on solid ground.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "JUMP", "JUMP_LEFT", "JUMP_RIGHT"),
        descriptions=(
            "do nothing this step",
            "move one cell left",
            "move one cell right",
            "jump straight up (if on ground)",
            "jump and move left simultaneously",
            "jump and move right simultaneously",
        ),
    )
    noop_action_name = "NOOP"

    # Reward shaping (Pattern B): +0.8 distributed across stars, +0.2 on goal.
    # Cumulative best-case: collect every star + reach goal = 1.0.
    _STAR_BUDGET = 0.8
    _STAR_TARGET = 8
    _GOAL_REWARD = 0.2

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._has_gravity = True
        self._view_w = 14
        self._view_h = 20
        self._stars_collected: int = 0
        self._stars_rewarded: int = 0
        self._total_stars: int = 0

    def env_id(self) -> str:
        return "glyphbench/procgen-climber-v0"

    # ------------------------------------------------------------------
    # Goal-reachability is a GENERATION INVARIANT. The base random layout can
    # place the top platform out of jump reach of the goal, or leave a
    # mid-level horizontal dead-end, making `G` geometrically unreachable on
    # ~1/6 of seeds (see audit). We therefore re-roll the layout (with a
    # deterministic per-attempt RNG) until a faithful physics BFS confirms the
    # goal is reachable. This keeps the env's identity (random rising
    # platforms, stars, patrolling enemies) while guaranteeing a skilled agent
    # always has a route to +1.0. Reset stays deterministic + seed-sensitive
    # because attempt RNGs derive deterministically from the episode seed.
    _MAX_LAYOUT_ATTEMPTS = 64

    def _generate_level(self, seed: int) -> None:
        for attempt in range(self._MAX_LAYOUT_ATTEMPTS):
            # First attempt uses the canonical episode RNG so unchanged
            # (already-solvable) seeds keep their original layout; later
            # attempts derive a fresh, deterministic stream from (seed, attempt).
            rng = self.rng if attempt == 0 else np.random.default_rng([seed, attempt])
            self._build_layout(rng)
            if self._goal_is_reachable():
                return
        # Extremely unlikely: every attempt failed. Keep the last layout; it is
        # still a valid (if hard) level and the invariants below still hold.

    def _build_layout(self, rng: np.random.Generator) -> None:
        W, H = 14, 40
        self._entities = []
        self._init_world(W, H, fill="\u00b7")
        self._total_stars = 0
        self._stars_collected = 0
        self._stars_rewarded = 0

        # Bottom ground
        for x in range(W):
            self._set_cell(x, H - 1, "\u25ac")

        # Place platforms rising upward, every 2-3 rows
        y = H - 4
        plat_idx = 0
        while y > 2:
            gap = int(rng.integers(2, 4))
            px = int(rng.integers(1, W - 5))
            pw = int(rng.integers(4, min(8, W - px)))
            for dx in range(pw):
                self._set_cell(px + dx, y, "\u25ac")

            # Star on platform
            star_x = px + int(rng.integers(0, pw))
            self._set_cell(star_x, y - 1, "*")
            self._total_stars += 1

            # Enemy on every other platform
            if plat_idx % 2 == 1 and pw >= 4:
                ex = px + 1
                self._add_entity("enemy", "E", ex, y - 1, dx=1)

            y -= gap
            plat_idx += 1

        # Top goal row
        for x in range(W):
            self._set_cell(x, 1, "\u25ac")
        self._set_cell(W // 2, 0, "G")
        for y in range(1, min(5, H)):
            self._set_cell(W // 2, y, "\u00b7")

        # Walls on sides
        for row in range(H):
            self._set_cell(0, row, "\u2588")
            self._set_cell(W - 1, row, "\u2588")

        # Agent starts on bottom ground
        self._agent_x = W // 2
        self._agent_y = H - 2

    # ------------------------------------------------------------------
    def _goal_is_reachable(self) -> bool:
        """Faithful pure-grid BFS over the real climber physics.

        A search state is (x, y, jump_step, on_ground); one transition is one
        environment step (action -> jump/gravity tick). Enemies are ignored:
        this checks pure geometric reachability of `G` for a skilled agent.
        Validated to match a deep-copy env-stepping BFS on every probed seed.
        """
        W, H = self._world_w, self._world_h

        def is_solid(x: int, y: int) -> bool:
            if 0 <= x < W and 0 <= y < H:
                return self._world[y][x] in ("\u2588", "\u25ac", "+", "|", "-")
            return True

        def transition(
            x: int, y: int, jstep: int, og: bool, action: int
        ) -> tuple[int, int, int, bool]:
            # --- _game_step: horizontal move / jump start ---
            if action == 1 and not is_solid(x - 1, y):  # LEFT
                x -= 1
            elif action == 2 and not is_solid(x + 1, y):  # RIGHT
                x += 1
            elif action == 3:  # JUMP
                if og:
                    jstep, og = 0, False
            elif action == 4:  # JUMP_LEFT
                if og:
                    jstep, og = 0, False
                if not is_solid(x - 1, y):
                    x -= 1
            elif action == 5:  # JUMP_RIGHT
                if og:
                    jstep, og = 0, False
                if not is_solid(x + 1, y):
                    x += 1
            # --- _process_jump (one arc tick) ---
            if jstep >= 0:
                if jstep < len(JUMP_ARC_DY):
                    dy = JUMP_ARC_DY[jstep]
                    ny = y + dy
                    if dy < 0:
                        if not is_solid(x, ny):
                            y = ny
                    elif dy > 0:
                        if not is_solid(x, ny):
                            y = ny
                        else:
                            og, jstep = True, -1
                    if jstep != -1:
                        jstep += 1
                else:
                    jstep = -1
            # --- gravity (only when not mid-jump) ---
            if jstep < 0:
                below = self._world[y + 1][x] if y + 1 < H else "\u2588"
                if below not in ("\u2588", "\u25ac", "+") and y + 1 < H:
                    y, og = y + 1, False
                else:
                    og = True
            return x, y, jstep, og

        sx, sy = self._agent_x, self._agent_y
        below0 = self._world[sy + 1][sx] if sy + 1 < H else "\u2588"
        og0 = below0 in ("\u2588", "\u25ac", "+") or sy + 1 >= H
        start = (sx, sy, -1, og0)
        seen = {start}
        q: deque[tuple[int, int, int, bool]] = deque([start])
        while q:
            x, y, jstep, og = q.popleft()
            for a in range(6):
                nx, ny, nj, nog = transition(x, y, jstep, og, a)
                if self._world[ny][nx] == "G":
                    return True
                s = (nx, ny, nj, nog)
                if s not in seen:
                    seen.add(s)
                    q.append(s)
        return False

    # ------------------------------------------------------------------
    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0

        # Movement
        if action_name == "LEFT":
            self._try_move(-1, 0)
        elif action_name == "RIGHT":
            self._try_move(1, 0)
        elif action_name == "JUMP":
            self._start_jump()
            self._agent_dir = (0, -1)
        elif action_name == "JUMP_LEFT":
            self._start_jump()
            self._try_move(-1, 0)
            self._agent_dir = (-1, 0)
        elif action_name == "JUMP_RIGHT":
            self._start_jump()
            self._try_move(1, 0)
            self._agent_dir = (1, 0)

        self._process_jump()

        # Collect star
        ch = self._world_at(self._agent_x, self._agent_y)
        if ch == "*":
            self._set_cell(self._agent_x, self._agent_y, "\u00b7")
            if self._stars_rewarded < self._STAR_TARGET:
                reward += self._STAR_BUDGET / self._STAR_TARGET
                self._stars_rewarded += 1
            self._stars_collected += 1
            self._message = "Collected a star!"

        # Goal
        if ch == "G":
            reward += self._GOAL_REWARD
            self._message = "Reached the top!"
            return reward, True, {}

        # Enemy collision
        for e in self._entities:
            if e.alive and e.x == self._agent_x and e.y == self._agent_y:
                self._message = "Hit by an enemy!"
                return reward, True, {"killed_by": "enemy"}

        # Fell off bottom
        if self._agent_y >= self._world_h - 1:
            pass  # on ground, fine

        return reward, False, {}

    # ------------------------------------------------------------------
    def _advance_entities(self) -> float:
        """Enemies patrol: bounce off walls / platform edges."""
        for e in self._entities:
            if not e.alive or e.etype != "enemy":
                continue
            nx = e.x + e.dx
            # Bounce off walls / solid / edge of platform
            below = self._world_at(nx, e.y + 1)
            if (
                self._is_solid(nx, e.y)
                or below not in ("\u25ac", "\u2588")
                or nx <= 0
                or nx >= self._world_w - 1
            ):
                e.dx = -e.dx
            else:
                e.x = nx

        # Check collision after move
        for e in self._entities:
            if e.alive and e.x == self._agent_x and e.y == self._agent_y:
                self._message = "Hit by an enemy!"
                self._entity_terminated = True
        return 0.0

    # ------------------------------------------------------------------
    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        extra = (
            f"Stars: {self._stars_collected}/{self._total_stars}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    # ------------------------------------------------------------------
    def _symbol_meaning(self, ch: str) -> str:
        m: dict[str, str] = {
            "\u00b7": "empty",
            "\u25ac": "platform/ground",
            "\u2588": "wall",
            "*": "star (collect for partial reward)",
            "G": "goal (top, finishes the level)",
            "E": "enemy",
            "@": "you",
        }
        return m.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Jump between platforms to climb. Collect stars (*) and "
            "reach the goal (G) at the top — the first 8 stars yield "
            "+0.8 total and the goal yields +0.2 (total +1.0). "
            "Avoid enemies (E) that patrol platforms."
        )
