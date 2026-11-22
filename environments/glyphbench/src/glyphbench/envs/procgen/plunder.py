"""Procgen Plunder environment.

Naval top-down shooter. Agent ship at bottom, pirate ships from top.
Shoot pirates, avoid hitting civilian ships.

Gym ID: glyphbench/procgen-plunder-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase


class PlunderEnv(ProcgenBase):
    """Procgen Plunder: naval shooter, sink pirates, spare civilians."""

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "fire cannon upward",
        ),
    )

    GRID_W = 20
    GRID_H = 14
    # Reward shaping (Pattern D): a [0, 1] "treasure protected" progress bar.
    #   +1/_WIN_TARGET  per pirate sunk          (you stopped a raider)
    #   -1/_WIN_TARGET  per civilian ship shot   (friendly fire)
    #   -1/_WIN_TARGET  per pirate that slips past you and reaches the docks
    # Progress is clamped to [0, 1] each step, so the per-step delta telescopes
    # exactly to a final value in [0, 1] (no farming, no escape from budget).
    # Being rammed by a pirate ends the episode with an additive -1.0.
    #
    # Pirates spawn spread across ALL columns and fall STRAIGHT down (no random
    # drift), so a static "spam FIRE" agent only ever covers its own column and
    # lets every other raider through -> net progress ~0. A skilled captain must
    # traverse to each raider's column and fire in time, while dodging the ones
    # about to ram and sparing the civilian ships. Civilians never ram; they
    # sail harmlessly off the docks, so holding fire is a safe, rewarded choice.
    _WIN_TARGET = 10
    _DEATH_PENALTY = -1.0
    # A single cannon defends one column at a time and the captain moves one
    # cell per turn, so raiders are spaced out enough that a skilled defender
    # can travel-and-intercept while a static spammer is swamped. Tuned (sweep
    # in reviews/_sweep/retune/procgen-plunder.md): period 12 keeps the trivial
    # FIRE-only policy at ~0.0 while a skilled policy wins ~70% with 0 deaths.
    _PIRATE_PERIOD = 12
    _CIVILIAN_PERIOD = 17

    def env_id(self) -> str:
        return "glyphbench/procgen-plunder-v0"

    def _generate_level(self, seed: int) -> None:
        w, h = self.GRID_W, self.GRID_H
        self._init_world(w, h, fill="\u2248")

        # Agent ship at bottom center
        self._agent_x = w // 2
        self._agent_y = h - 2

        self._pirates_sunk = 0
        self._civilians_hit = 0
        self._pirates_escaped = 0
        self._spawn_timer = 0
        # Net cumulative progress paid out so far in [0, 1]. Used to clamp
        # the per-step delta so cumulative reward never escapes the budget.
        self._progress_paid = 0.0

    def _maybe_spawn(self) -> None:
        """Spawn pirate and civilian ships from the top.

        Ships appear in any column across the full width and fall straight
        down (dx=0). The agent must travel to a raider's column to sink it.
        """
        w = self.GRID_W
        self._spawn_timer += 1

        if self._spawn_timer % self._PIRATE_PERIOD == 0:
            px = int(self.rng.integers(0, w))
            self._add_entity("pirate", "P", px, 0, dx=0, dy=1)

        if self._spawn_timer % self._CIVILIAN_PERIOD == 0:
            cx = int(self.rng.integers(0, w))
            self._add_entity("civilian", "c", cx, 0, dx=0, dy=1)

    def _adjust_progress(self, delta: float) -> float:
        """Move the [0, 1] progress bar by ``delta`` and return the clamped
        per-step reward. Deltas telescope exactly to the final progress, so
        cumulative reward stays in [0, 1] (then -1 at most once for a ram)."""
        new_progress = min(1.0, max(0.0, self._progress_paid + delta))
        reward = new_progress - self._progress_paid
        self._progress_paid = new_progress
        return reward

    def _hit_ship(self, ship: Any) -> float:
        unit = 1.0 / self._WIN_TARGET
        if ship.etype == "pirate":
            self._pirates_sunk += 1
            return self._adjust_progress(+unit)

        self._civilians_hit += 1
        self._message = "You hit a civilian ship!"
        return self._adjust_progress(-unit)

    def _pirate_escaped(self) -> float:
        """A pirate slipped past and reached the docks: lose protection."""
        unit = 1.0 / self._WIN_TARGET
        self._pirates_escaped += 1
        return self._adjust_progress(-unit)

    def _resolve_cannonball_hits(
        self,
        old_positions: dict[int, tuple[int, int]] | None = None,
    ) -> float:
        reward = 0.0
        balls = [e for e in self._entities if e.etype == "cannonball" and e.alive]
        ships = [
            e
            for e in self._entities
            if e.etype in ("pirate", "civilian") and e.alive
        ]
        for ball in balls:
            if not ball.alive:
                continue
            for ship in ships:
                if not ship.alive:
                    continue
                same_cell = ball.x == ship.x and ball.y == ship.y
                crossed = False
                if old_positions is not None and ball.x == ship.x:
                    old_ball = old_positions.get(id(ball))
                    old_ship = old_positions.get(id(ship))
                    crossed = (
                        old_ball is not None
                        and old_ship is not None
                        and old_ball[0] == old_ship[0] == ball.x
                        and old_ball[1] == ship.y
                        and old_ship[1] == ball.y
                    )
                if same_cell or crossed:
                    ball.alive = False
                    ship.alive = False
                    reward += self._hit_ship(ship)
                    break
        return reward

    def _advance_entities(self) -> float:
        """Move entities: cannonballs up, ships straight down (no drift).

        A pirate that sails off the bottom slipped past the agent and costs
        progress; a civilian off the bottom is harmless. Only pirates ram.
        """
        old_positions = {id(e): (e.x, e.y) for e in self._entities if e.alive}
        reward = 0.0
        collision_reward: float | None = None
        for e in self._entities:
            if not e.alive:
                continue
            e.x += e.dx
            e.y += e.dy
            if e.x < 0 or e.x >= self.GRID_W:
                e.alive = False
                continue
            if e.y < 0:
                e.alive = False  # cannonball off the top
            elif e.y >= self.GRID_H:
                e.alive = False  # ship reached the docks
                if e.etype == "pirate":
                    reward += self._pirate_escaped()

        reward += self._resolve_cannonball_hits(old_positions)

        self._entities = [e for e in self._entities if e.alive]
        for e in self._entities:
            if (
                e.etype == "pirate"
                and e.x == self._agent_x
                and e.y == self._agent_y
            ):
                self._message = "Rammed by a pirate ship!"
                self._entity_terminated = True
                # Additive death penalty: preserve any progress reward earned
                # this step. Cumulative episode return = progress_paid - 1,
                # bounded in [-1, 0] on death.
                collision_reward = reward + self._DEATH_PENALTY
                break
        if collision_reward is not None:
            return collision_reward
        return reward

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        # Movement (horizontal + forward)
        if action_name == "LEFT":
            self._agent_dir = (-1, 0)
            nx = self._agent_x - 1
            if nx >= 0:
                self._agent_x = nx
        elif action_name == "RIGHT":
            self._agent_dir = (1, 0)
            nx = self._agent_x + 1
            if nx < self.GRID_W:
                self._agent_x = nx
        elif action_name == "FIRE":
            by = self._agent_y - 1
            if by >= 0:
                self._add_entity(
                    "cannonball", "^", self._agent_x, by, dx=0, dy=-1,
                )

        # Spawn new ships
        self._maybe_spawn()

        # Check cannonball-ship collisions. Pirate kill adds positive
        # progress (clamped at +1.0); civilian hit subtracts (floored at 0).
        reward += self._resolve_cannonball_hits()

        # Agent collision with a pirate (terminal failure). Civilians are
        # harmless and never ram. Additive death penalty so any progress
        # earned this step survives into the cumulative episode return.
        for e in self._entities:
            if not e.alive or e.etype != "pirate":
                continue
            if e.x == self._agent_x and e.y == self._agent_y:
                reward += self._DEATH_PENALTY
                terminated = True
                self._message = "Rammed by a pirate ship!"
                return reward, terminated, info

        # Clean dead entities
        self._entities = [e for e in self._entities if e.alive]

        info["pirates_sunk"] = self._pirates_sunk
        info["civilians_hit"] = self._civilians_hit
        info["pirates_escaped"] = self._pirates_escaped
        return reward, terminated, info

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        info["pirates_sunk"] = self._pirates_sunk
        info["civilians_hit"] = self._civilians_hit
        info["pirates_escaped"] = self._pirates_escaped
        return obs, reward, terminated, truncated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        pirates = sum(
            1 for e in self._entities
            if e.alive and e.etype == "pirate"
        )
        civilians = sum(
            1 for e in self._entities
            if e.alive and e.etype == "civilian"
        )
        extra = (
            f"Pirates sunk: {self._pirates_sunk}"
            f"  Slipped past: {self._pirates_escaped}"
            f"  Civilians hit: {self._civilians_hit}"
            f"  Nearby: {pirates}P {civilians}c"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "You captain a ship (@) guarding the docks at the bottom of the "
            "sea (\u2248). Pirate ships (P) and civilian ships (c) sail in "
            "from the top in any column and descend straight down. FIRE "
            "launches a cannonball (^) straight up your current column, so "
            "you must steer (LEFT / RIGHT) into a target's column to hit it. "
            f"Sinking one of the first {self._WIN_TARGET} pirates adds "
            f"+1/{self._WIN_TARGET} progress. A pirate that slips past you to "
            f"the docks costs 1/{self._WIN_TARGET}, and shooting a civilian "
            f"costs 1/{self._WIN_TARGET}; progress is clamped to [0, 1]. "
            "Civilian ships are harmless and sail past on their own, so hold "
            "your fire on them. Letting a pirate ram you ends the episode at a "
            "net cumulative -1."
        )

    def _symbol_meaning(self, ch: str) -> str:
        meanings = {
            "\u2248": "sea",
            "P": "pirate ship (sink it; rams you if it reaches the docks)",
            "c": "civilian ship (harmless; reverses progress if shot)",
            "^": "cannonball",
        }
        return meanings.get(ch, super()._symbol_meaning(ch))
