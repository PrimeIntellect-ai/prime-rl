"""Procgen FruitBot environment.

Vertical auto-scroller. Agent falls through a tunnel collecting fruit
and dodging obstacles. Level scrolls downward automatically.

Gym ID: glyphbench/procgen-fruitbot-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase


class FruitBotEnv(ProcgenBase):
    """FruitBot: vertical auto-scroller collecting fruit.

    World: 14 wide x 40 tall.  View: 14 x 20.
    Agent auto-falls 1 cell per step. Collect fruit (%) for +1,
    avoid obstacles (x) for -1 penalty.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "DOWN"),
        descriptions=(
            "do nothing (still falls 1 cell)",
            "move one cell left while falling",
            "move one cell right while falling",
            "fall 2 cells instead of 1",
        ),
    )
    noop_action_name = "NOOP"

    # Reward shaping (Pattern D): collecting all fruit yields +0.7, hitting
    # every obstacle yields -0.6, reaching the bottom yields +0.3. Best case
    # +1.0; worst case including capped wall bounces is exactly -1.0.
    _FRUIT_BUDGET = 0.7
    _OBSTACLE_BUDGET = 0.6
    _BOTTOM_REWARD = 0.3
    _FRUIT_TARGET = 7
    _OBSTACLE_TARGET = 6
    # Per-bounce wall rebound penalty. Walls are no longer terminal; the
    # agent rebounds (no fall progress this tick) and pays a small fee
    # so dawdling on a blocked column carries cost (B2 bait-and-detour).
    # Bounded so obstacle penalties (-0.6) plus the bounce cap (-0.4) never
    # push cumulative return below -1.0.
    _WALL_BOUNCE_PENALTY = -0.05
    _WALL_BOUNCE_BUDGET = -0.4

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._has_gravity = False  # we manage falling ourselves
        self._view_w = 14
        self._view_h = 20
        self._fruits_collected: int = 0
        self._obstacles_hit: int = 0
        self._total_fruit: int = 0
        self._total_obstacle: int = 0
        self._wall_bounces: int = 0
        self._bounce_penalty_paid: float = 0.0
        self._fruit_rewarded: int = 0
        self._obstacle_penalized: int = 0

    def env_id(self) -> str:
        return "glyphbench/procgen-fruitbot-v0"

    # ------------------------------------------------------------------
    def _generate_level(self, seed: int) -> None:
        W, H = 14, 40
        self._init_world(W, H, fill="\u00b7")
        self._fruits_collected = 0
        self._obstacles_hit = 0
        self._total_fruit = 0
        self._total_obstacle = 0
        self._wall_bounces = 0
        self._bounce_penalty_paid = 0.0
        self._fruit_rewarded = 0
        self._obstacle_penalized = 0

        # Walls on sides
        for y in range(H):
            self._set_cell(0, y, "\u2588")
            self._set_cell(W - 1, y, "\u2588")

        # Scatter fruit and obstacles throughout the tunnel
        for y in range(3, H - 1):
            n_items = int(self.rng.integers(0, 3))
            for _ in range(n_items):
                ix = int(self.rng.integers(1, W - 1))
                if self._world_at(ix, y) == "\u00b7":
                    if self.rng.random() < 0.6:
                        self._set_cell(ix, y, "%")
                        self._total_fruit += 1
                    else:
                        self._set_cell(ix, y, "x")
                        self._total_obstacle += 1

            # Occasional internal walls to make navigation interesting.
            # Density tuned down from 15% to 8% so the tunnel stays
            # mostly traversable; combined with the non-fatal wall
            # rebound this gives the agent room to detour.
            if self.rng.random() < 0.08:
                wall_x = int(self.rng.integers(2, W - 2))
                wall_len = int(self.rng.integers(2, 5))
                for dx in range(wall_len):
                    wx = wall_x + dx
                    if 1 <= wx < W - 1:
                        self._set_cell(wx, y, "\u2588")

        self._ensure_min_item_count("%", self._FRUIT_TARGET)
        self._ensure_min_item_count("x", self._OBSTACLE_TARGET)

        # Agent starts at top center
        self._agent_x = W // 2
        self._agent_y = 1

    def _ensure_min_item_count(self, ch: str, target: int) -> None:
        total_attr = "_total_fruit" if ch == "%" else "_total_obstacle"
        current = getattr(self, total_attr)
        if current >= target:
            return
        candidates = [
            (x, y)
            for y in range(3, self._world_h - 1)
            for x in range(1, self._world_w - 1)
            if self._world_at(x, y) == "\u00b7"
        ]
        self.rng.shuffle(candidates)
        for x, y in candidates[: target - current]:
            self._set_cell(x, y, ch)
            current += 1
        setattr(self, total_attr, current)

    # ------------------------------------------------------------------
    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0

        # Horizontal movement
        if action_name == "LEFT":
            self._try_move(-1, 0)
        elif action_name == "RIGHT":
            self._try_move(1, 0)
        elif action_name == "DOWN":
            self._agent_dir = (0, 1)

        # Fall (non-fatal wall rebound: B2). When the cell below is a
        # wall, the agent does not advance; instead it pays a small
        # rebound penalty (capped over the episode) and the step ends
        # without further progress so the agent can use LEFT/RIGHT to
        # detour next tick.
        fall_dist = 2 if action_name == "DOWN" else 1
        bounced = False
        for _ in range(fall_dist):
            ny = self._agent_y + 1
            if ny >= self._world_h:
                # Reached bottom
                reward += self._BOTTOM_REWARD
                self._message = "Reached the bottom!"
                return reward, True, {}
            if self._is_solid(self._agent_x, ny):
                # Wall below: rebound (no movement progress this tick)
                # and apply a small penalty bounded by _WALL_BOUNCE_BUDGET.
                self._wall_bounces += 1
                remaining = (
                    self._WALL_BOUNCE_BUDGET - self._bounce_penalty_paid
                )
                if remaining < 0.0:
                    pen = max(self._WALL_BOUNCE_PENALTY, remaining)
                    reward += pen
                    self._bounce_penalty_paid += pen
                self._message = "Bounced off wall!"
                bounced = True
                break
            self._agent_y = ny

            # Check what we landed on
            ch = self._world_at(self._agent_x, self._agent_y)
            if ch == "%":
                self._set_cell(self._agent_x, self._agent_y, "\u00b7")
                if self._fruit_rewarded < self._FRUIT_TARGET:
                    reward += self._FRUIT_BUDGET / self._FRUIT_TARGET
                    self._fruit_rewarded += 1
                self._fruits_collected += 1
            elif ch == "x":
                self._set_cell(self._agent_x, self._agent_y, "\u00b7")
                if self._obstacle_penalized < self._OBSTACLE_TARGET:
                    reward -= self._OBSTACLE_BUDGET / self._OBSTACLE_TARGET
                    self._obstacle_penalized += 1
                self._obstacles_hit += 1

        # Reached bottom (only credit if the agent actually advanced
        # this tick; bouncing off a wall is not a bottom reach).
        if not bounced and self._agent_y >= self._world_h - 1:
            reward += self._BOTTOM_REWARD
            self._message = "Reached the bottom!"
            return reward, True, {}

        return reward, False, {}

    # ------------------------------------------------------------------
    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        extra = (
            f"Fruits: {self._fruits_collected}"
            f"  Obstacles hit: {self._obstacles_hit}"
            f"  Wall bounces: {self._wall_bounces}"
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
            "\u2588": "wall (rebound, small penalty)",
            "%": "fruit (small reward)",
            "x": "obstacle (small penalty)",
            "@": "you",
        }
        return m.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "You are a FruitBot falling through a tunnel. Collecting the "
            f"first {self._FRUIT_TARGET} fruit (%) yields +0.7 total; "
            f"hitting each of the first {self._OBSTACLE_TARGET} obstacles (x) "
            "costs -0.6 total. Reaching the bottom yields an additional "
            "+0.3 (best case +1.0). You fall 1 cell per step automatically. "
            "Use LEFT/RIGHT to detour around walls (\u2588) and DOWN to fall "
            "2 cells; DOWN collects or hits both cells crossed. Walls are NOT "
            "terminal: hitting one rebounds the "
            "agent in place and applies a small penalty (capped over the "
            "episode), so plan a path that uses lateral moves when the "
            "column ahead is blocked."
        )
