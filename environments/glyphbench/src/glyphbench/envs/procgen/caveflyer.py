"""Procgen CaveFlyer environment.

Procedural cave corridors with enemies. Agent flies a ship and fires bullets.

Gym ID: glyphbench/procgen-caveflyer-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase


class CaveFlyerEnv(ProcgenBase):
    """Procgen CaveFlyer: fly through caves, shoot enemies, reach the end."""

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
            "fire bullet right",
        ),
    )

    GRID_W = 40
    GRID_H = 12
    # Reward shaping (Pattern D): +0.4 across enemies, +0.6 on reaching the
    # exit, terminal -1.0 on wall/enemy collision.
    _ENEMY_BUDGET = 0.4
    _EXIT_REWARD = 0.6
    _DEATH_PENALTY = -1.0

    def env_id(self) -> str:
        return "glyphbench/procgen-caveflyer-v0"

    def _generate_level(self, seed: int) -> None:
        w, h = self.GRID_W, self.GRID_H
        self._init_world(w, h, fill="\u2588")

        # Generate cave corridor using random walk of ceiling/floor
        top = 1
        bot = h - 2
        for x in range(w):
            # Carve open space between top and bot
            for y in range(top, bot + 1):
                self._set_cell(x, y, " ")

            # Random walk ceiling and floor
            if x < w - 1:
                top += int(self.rng.integers(-1, 2))  # -1, 0, or 1
                bot += int(self.rng.integers(-1, 2))
                top = max(1, min(top, h // 2 - 1))
                bot = max(h // 2 + 1, min(bot, h - 2))

        # Place exit marker at right side
        exit_y = (top + bot) // 2
        # Find a non-wall cell near the right edge
        for x in range(w - 2, w - 5, -1):
            for y in range(h):
                if self._world_at(x, y) == " ":
                    exit_y = y
                    self._set_cell(x, y, ">")
                    self._exit_x = x
                    self._exit_y = exit_y
                    break
            else:
                continue
            break

        # Agent starts at left
        self._agent_x = 1
        # Find open cell for agent
        for y in range(h):
            if self._world_at(1, y) == " ":
                self._agent_y = y
                break

        self._enemies_killed = 0
        self._total_enemies = 0

        # Spawn four enemies so each kill pays a meaningful +0.1 reward.
        num_enemies = 4
        for _ in range(num_enemies):
            for _attempt in range(50):
                ex = int(self.rng.integers(5, w - 3))
                ey = int(self.rng.integers(1, h - 1))
                if self._world_at(ex, ey) == " ":
                    dy = 1 if int(self.rng.integers(0, 2)) == 0 else -1
                    self._add_entity("enemy", "E", ex, ey, dx=0, dy=dy)
                    self._total_enemies += 1
                    break

    def _advance_entities(self) -> float:
        """Move entities: bullets fly right, enemies patrol vertically."""
        for e in self._entities:
            if not e.alive:
                continue
            e.x += e.dx
            e.y += e.dy
            # Bullets die on walls or out of bounds
            if e.etype == "bullet":
                if (
                    e.x < 0
                    or e.x >= self.GRID_W
                    or e.y < 0
                    or e.y >= self.GRID_H
                    or self._is_solid(e.x, e.y)
                ):
                    e.alive = False
            # Enemies bounce off walls
            elif e.etype == "enemy" and self._is_solid(e.x, e.y):
                e.x -= e.dx
                e.y -= e.dy
                e.dx = -e.dx
                e.dy = -e.dy
        self._entities = [e for e in self._entities if e.alive]
        for e in self._entities:
            if e.etype == "enemy" and e.x == self._agent_x and e.y == self._agent_y:
                self._message = "Hit by an enemy!"
                self._entity_terminated = True
                return self._DEATH_PENALTY
        return 0.0

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        # Movement
        if action_name == "LEFT":
            self._agent_dir = (-1, 0)
            nx = self._agent_x - 1
            if not self._is_solid(nx, self._agent_y):
                self._agent_x = nx
        elif action_name == "RIGHT":
            self._agent_dir = (1, 0)
            nx = self._agent_x + 1
            if not self._is_solid(nx, self._agent_y):
                self._agent_x = nx
        elif action_name == "UP":
            self._agent_dir = (0, -1)
            ny = self._agent_y - 1
            if not self._is_solid(self._agent_x, ny):
                self._agent_y = ny
        elif action_name == "DOWN":
            self._agent_dir = (0, 1)
            ny = self._agent_y + 1
            if not self._is_solid(self._agent_x, ny):
                self._agent_y = ny
        elif action_name == "FIRE":
            bx = self._agent_x + 1
            if not self._is_solid(bx, self._agent_y):
                self._add_entity(
                    "bullet", "*", bx, self._agent_y, dx=1, dy=0,
                )

        # Check bullet-enemy collisions
        for bullet in [e for e in self._entities if e.etype == "bullet" and e.alive]:
            for enemy in [
                e for e in self._entities if e.etype == "enemy" and e.alive
            ]:
                if bullet.x == enemy.x and bullet.y == enemy.y:
                    bullet.alive = False
                    enemy.alive = False
                    if self._total_enemies > 0:
                        reward += self._ENEMY_BUDGET / self._total_enemies
                    self._enemies_killed += 1

        # Wall collision damage (terminal failure). Additive so any
        # enemy-kill reward earned this step survives into cumulative.
        if self._is_solid(self._agent_x, self._agent_y):
            reward += self._DEATH_PENALTY
            terminated = True
            self._message = "Crashed into a wall!"
            return reward, terminated, info

        # Agent-enemy collision (terminal failure). Additive so any kill
        # credit earned this step survives.
        for e in self._entities:
            if not e.alive or e.etype != "enemy":
                continue
            if e.x == self._agent_x and e.y == self._agent_y:
                reward += self._DEATH_PENALTY
                terminated = True
                self._message = "Hit by an enemy!"
                return reward, terminated, info

        # Check if reached exit
        if (
            self._agent_x == self._exit_x
            and self._agent_y == self._exit_y
        ):
            reward += self._EXIT_REWARD
            terminated = True
            self._message = "Reached the exit!"

        # Clean dead entities
        self._entities = [e for e in self._entities if e.alive]

        info["enemies_killed"] = self._enemies_killed
        return reward, terminated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        alive = sum(
            1 for e in self._entities
            if e.alive and e.etype == "enemy"
        )
        # Exit coordinates removed — agent must locate the exit visually.
        extra = f"Enemies remaining: {alive}"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Fly your ship (@) through the cave. Reach the exit (>). "
            "FIRE shoots bullets (*) to the right to destroy enemies (E). "
            "Killing each of the 4 enemies yields +0.1; reaching the exit "
            "yields +0.6 (best case +1.0). Hitting a wall or enemy ends "
            "the episode at -1.0."
        )

    def _symbol_meaning(self, ch: str) -> str:
        meanings = {
            " ": "open cave",
            "\u2588": "cave wall",
            ">": "exit",
            "E": "enemy",
            "*": "bullet",
        }
        return meanings.get(ch, super()._symbol_meaning(ch))
