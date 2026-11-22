"""Procgen Dodgeball environment.

Arena where the agent throws balls at enemies while dodging.

Gym ID: glyphbench/procgen-dodgeball-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase


class DodgeballEnv(ProcgenBase):
    """Procgen Dodgeball: throw balls to hit wandering enemies."""

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "THROW"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
            "throw ball in facing direction",
        ),
    )

    GRID_W = 14
    GRID_H = 12
    # Reward shaping (Pattern D): first _WIN_TARGET kills each yield
    # +1/_WIN_TARGET; subsequent kills add 0 (cumulative caps at +1.0).
    # Wave-clear bonuses removed; terminal -1.0 on getting hit.
    _WIN_TARGET = 10
    _DEATH_PENALTY = -1.0

    def env_id(self) -> str:
        return "glyphbench/procgen-dodgeball-v0"

    def _generate_level(self, seed: int) -> None:
        w, h = self.GRID_W, self.GRID_H
        self._init_world(w, h, fill=" ")

        # Walls around the arena
        for x in range(w):
            self._set_cell(x, 0, "\u2588")
            self._set_cell(x, h - 1, "\u2588")
        for y in range(h):
            self._set_cell(0, y, "\u2588")
            self._set_cell(w - 1, y, "\u2588")

        # Agent starts center-bottom
        self._agent_x = w // 2
        self._agent_y = h - 3
        self._facing_dx = 0
        self._facing_dy = -1  # facing up by default
        self._agent_dir = (0, -1)

        self._enemies_killed = 0
        self._level = 1

        # Spawn enemies
        self._spawn_enemies(count=4)

    def _spawn_enemies(self, count: int) -> None:
        """Spawn enemies at random positions away from agent."""
        w, h = self.GRID_W, self.GRID_H
        occupied = {(self._agent_x, self._agent_y)}
        occupied.update((e.x, e.y) for e in self._entities if e.etype == "enemy")
        for _ in range(count):
            for _attempt in range(50):
                x = int(self.rng.integers(2, w - 2))
                y = int(self.rng.integers(2, h - 2))
                dist = abs(x - self._agent_x) + abs(y - self._agent_y)
                if dist > 3 and (x, y) not in occupied:
                    ddx = 1 if int(self.rng.integers(0, 2)) == 0 else -1
                    ddy = 0
                    self._add_entity("enemy", "E", x, y, dx=ddx, dy=ddy)
                    occupied.add((x, y))
                    break

    def _advance_entities(self) -> float:
        """Move entities and detect post-advance ball/enemy collisions.

        The pre-advance check in _game_step only catches contemporaneous
        overlap; if a ball with dx=+1 and an enemy with dx=-1 swap cells
        in the same tick, neither was at the other's tile pre-advance
        and both ended in different tiles post-advance — they would
        tunnel through each other. This method tracks each entity's
        prior position and resolves swap-collisions (audit MINOR fix).
        Returns reward for any kills credited at this stage so the
        cumulative bound and budget stay correct.
        """
        # Snapshot pre-advance positions for ball-enemy swap detection.
        ball_prev: dict[int, tuple[int, int]] = {}
        enemy_prev: dict[int, tuple[int, int]] = {}
        for e in self._entities:
            if not e.alive:
                continue
            if e.etype == "ball":
                ball_prev[id(e)] = (e.x, e.y)
            elif e.etype == "enemy":
                enemy_prev[id(e)] = (e.x, e.y)

        # Advance.
        for e in self._entities:
            if not e.alive:
                continue
            if e.etype == "ball":
                e.x += e.dx
                e.y += e.dy
                if self._is_solid(e.x, e.y):
                    e.alive = False
            elif e.etype == "enemy":
                if float(self.rng.random()) < 0.3:
                    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    idx = int(self.rng.integers(0, len(dirs)))
                    e.dx, e.dy = dirs[idx]
                nx, ny = e.x + e.dx, e.y + e.dy
                if not self._is_solid(nx, ny):
                    e.x = nx
                    e.y = ny
                else:
                    e.dx = -e.dx
                    e.dy = -e.dy

        # Post-advance ball-enemy collisions: cohabitation OR swap.
        # Credit each new kill the same way _game_step does, capped at
        # _WIN_TARGET so cumulative reward never exceeds +1.0.
        reward = 0.0
        balls = [e for e in self._entities if e.etype == "ball" and e.alive]
        enemies = [e for e in self._entities if e.etype == "enemy" and e.alive]
        for ball in balls:
            if not ball.alive:
                continue
            bx0, by0 = ball_prev.get(id(ball), (ball.x, ball.y))
            for enemy in enemies:
                if not enemy.alive:
                    continue
                ex0, ey0 = enemy_prev.get(id(enemy), (enemy.x, enemy.y))
                cohabit = ball.x == enemy.x and ball.y == enemy.y
                swap = (
                    ball.x == ex0 and ball.y == ey0
                    and enemy.x == bx0 and enemy.y == by0
                )
                if cohabit or swap:
                    ball.alive = False
                    enemy.alive = False
                    if self._enemies_killed < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                    self._enemies_killed += 1
                    break

        self._entities = [e for e in self._entities if e.alive]
        for e in self._entities:
            if e.etype == "enemy" and e.x == self._agent_x and e.y == self._agent_y:
                self._message = "Hit by an enemy!"
                self._entity_terminated = True
                # Additive: preserve any kill credit earned this step
                # before the agent was hit by a remaining enemy.
                return reward + self._DEATH_PENALTY
        return reward

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        # Movement
        if action_name == "LEFT":
            self._try_move(-1, 0)
            self._facing_dx, self._facing_dy = -1, 0
            self._agent_dir = (-1, 0)
        elif action_name == "RIGHT":
            self._try_move(1, 0)
            self._facing_dx, self._facing_dy = 1, 0
            self._agent_dir = (1, 0)
        elif action_name == "UP":
            self._try_move(0, -1)
            self._facing_dx, self._facing_dy = 0, -1
            self._agent_dir = (0, -1)
        elif action_name == "DOWN":
            self._try_move(0, 1)
            self._facing_dx, self._facing_dy = 0, 1
            self._agent_dir = (0, 1)
        elif action_name == "THROW":
            bx = self._agent_x + self._facing_dx
            by = self._agent_y + self._facing_dy
            if not self._is_solid(bx, by):
                self._add_entity(
                    "ball", "*", bx, by,
                    dx=self._facing_dx, dy=self._facing_dy,
                )

        # Check ball-enemy collisions
        for ball in [e for e in self._entities if e.etype == "ball" and e.alive]:
            for enemy in [
                e for e in self._entities if e.etype == "enemy" and e.alive
            ]:
                if ball.x == enemy.x and ball.y == enemy.y:
                    ball.alive = False
                    enemy.alive = False
                    if self._enemies_killed < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                    self._enemies_killed += 1

        # Agent-enemy collision (terminal failure). Additive so any
        # kill credit earned this step survives into cumulative.
        for e in self._entities:
            if not e.alive or e.etype != "enemy":
                continue
            if e.x == self._agent_x and e.y == self._agent_y:
                reward += self._DEATH_PENALTY
                terminated = True
                self._message = "Hit by an enemy!"
                return reward, terminated, info

        # Clean dead entities
        self._entities = [e for e in self._entities if e.alive]

        # When a wave is cleared, spawn the next one (no extra reward —
        # progress is already paid per kill).
        enemies_alive = sum(1 for e in self._entities if e.etype == "enemy")
        if enemies_alive == 0:
            self._level += 1
            self._message = f"Level {self._level - 1} cleared!"
            self._spawn_enemies(count=3 + self._level)

        info["enemies_killed"] = self._enemies_killed
        info["level"] = self._level
        return reward, terminated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        enemies = sum(
            1 for e in self._entities
            if e.alive and e.etype == "enemy"
        )
        extra = (
            f"Enemies: {enemies}"
            f"  Wave: {self._level}"
            f"  Kills: {self._enemies_killed}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "You are in an arena (@), initially facing up. Throw balls (*) at "
            "enemies (E) by moving in a direction then using THROW; each ball "
            "spawns one tile ahead, travels until it hits a wall/enemy, and "
            "is lost if thrown into a wall. Enemy waves continue until the "
            "step limit; clearing a visible wave spawns a larger one. The first "
            f"{self._WIN_TARGET} enemy kills each yield +1/{self._WIN_TARGET}; "
            "extra kills add nothing. Touching an enemy ends the episode at -1."
        )

    def _symbol_meaning(self, ch: str) -> str:
        meanings = {
            " ": "arena floor",
            "\u2588": "wall",
            "E": "enemy",
            "*": "ball",
        }
        return meanings.get(ch, super()._symbol_meaning(ch))
