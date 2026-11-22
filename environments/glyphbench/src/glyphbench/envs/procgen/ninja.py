"""Procgen Ninja environment.

Platformer with throwing stars. Agent traverses platforms, defeats
enemies by throwing projectiles, breaks through walls, and reaches goal.

Gym ID: glyphbench/procgen-ninja-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase


class NinjaEnv(ProcgenBase):
    """Ninja platformer: throw stars at enemies, break walls, reach goal.

    World: 40 wide x 12 tall.  View: 20 x 12.
    Gravity enabled.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "JUMP", "JUMP_RIGHT", "THROW"),
        descriptions=(
            "do nothing this step",
            "move one cell left",
            "move one cell right",
            "jump straight up (if on ground)",
            "jump and move right simultaneously",
            "throw a shuriken in the direction you face",
        ),
    )
    noop_action_name = "NOOP"

    # Reward shaping (Pattern B): +0.4 split across enemy kills, +0.6 on
    # reaching the goal. Cumulative best-case: kill all enemies + reach
    # goal = 1.0. Death gives 0 (Pattern A: no failure penalty).
    _ENEMY_BUDGET = 0.4
    _GOAL_REWARD = 0.6

    # Pits carved into the traverse (ground) row. A blind RIGHT/THROW cycle
    # walks into a pit, falls to the trench floor and is trapped there (the
    # flanking ground tiles are solid), so it can never reach the goal. A
    # skilled agent must read the grid and clear each pit with a JUMP_RIGHT
    # followed by mid-air RIGHT presses. Pit widths stay <= the jump-arc reach
    # so the level remains solvable. This is what makes the task discriminate.
    _MIN_PIT_WIDTH = 2
    _MAX_PIT_WIDTH = 3

    # Small shaping costs that sharpen the failure signal without breaking the
    # [-1, 1] bound (worst case = death after a full clear; see _task_description
    # / the bound argument in the retune note). A per-step time cost makes a
    # policy that flails, stalls in a trench, or times out score strictly worse
    # than direct progress, and a death penalty makes touching an enemy worse
    # than partial progress. A skilled agent reaches the goal in well under
    # ~120 steps, so the accrued time cost stays tiny for it.
    _STEP_COST = 0.001
    _DEATH_PENALTY = 0.25

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._has_gravity = True
        self._view_w = 20
        self._view_h = 12
        self._facing: int = 1  # +1 right, -1 left
        self._enemies_killed: int = 0
        self._total_enemies: int = 0

    def env_id(self) -> str:
        return "glyphbench/procgen-ninja-v0"

    # ------------------------------------------------------------------
    def _generate_level(self, seed: int) -> None:
        W, H = 40, 12
        self._init_world(W, H, fill="\u00b7")
        self._enemies_killed = 0
        self._total_enemies = 0
        ground_y = H - 2

        # Ground
        for x in range(W):
            self._set_cell(x, ground_y, "\u25ac")
            self._set_cell(x, H - 1, "\u25ac")

        # Pits in the traverse row. Carved out of the ground tier (ground_y)
        # only; the lower floor row (H-1) stays solid so a faller lands in a
        # trench it cannot climb out of. Pits live strictly inside the
        # corridor (clear of the spawn cells x<=4 and the goal column W-2),
        # never touch, and never abut so the agent has solid take-off/landing
        # cells on both sides of every gap.
        pit_cols: set[int] = set()
        num_pits = int(self.rng.integers(1, 4))
        cursor = 6
        for _ in range(num_pits):
            if cursor >= W - 6:
                break
            pw = int(self.rng.integers(self._MIN_PIT_WIDTH, self._MAX_PIT_WIDTH + 1))
            px = int(self.rng.integers(cursor, max(cursor + 1, W - 5 - pw)))
            if px + pw > W - 4:
                break
            for dx in range(pw):
                self._set_cell(px + dx, ground_y, "\u00b7")
                pit_cols.add(px + dx)
            # Leave at least two solid ground cells before the next pit so the
            # agent always has a stable landing + take-off run.
            cursor = px + pw + 3

        def _corridor_free(x: int) -> bool:
            return x not in pit_cols and (x - 1) not in pit_cols and (x + 1) not in pit_cols

        # Platforms
        num_plats = int(self.rng.integers(3, 7))
        for _ in range(num_plats):
            px = int(self.rng.integers(4, W - 6))
            py = int(self.rng.integers(ground_y - 5, ground_y - 2))
            pw = int(self.rng.integers(3, 6))
            if py < 1:
                py = 1
            for dx in range(pw):
                if px + dx < W:
                    self._set_cell(px + dx, py, "\u2588")

        # Breakable walls (never on a pit edge: the agent must keep clear
        # take-off ground around every gap).
        num_walls = int(self.rng.integers(1, 4))
        for _ in range(num_walls):
            wx = int(self.rng.integers(8, W - 8))
            if self._world_at(wx, ground_y) == "\u25ac" and _corridor_free(wx):
                for wy in range(ground_y - 2, ground_y):
                    self._set_cell(wx, wy, "B")

        # Enemies on ground (never spawned over a pit). At least one enemy is
        # guaranteed so the 0.4 kill budget is always reachable and the reward
        # ceiling is consistent across seeds.
        num_enemies = int(self.rng.integers(2, 5))
        for _ in range(num_enemies):
            ex = int(self.rng.integers(6, W - 6))
            if self._world_at(ex, ground_y - 1) == "\u00b7" and ex not in pit_cols:
                self._add_entity("enemy", "E", ex, ground_y - 1, dx=1)
                self._total_enemies += 1
        if self._total_enemies == 0:
            for ex in range(6, W - 6):
                if self._world_at(ex, ground_y - 1) == "\u00b7" and ex not in pit_cols:
                    self._add_entity("enemy", "E", ex, ground_y - 1, dx=1)
                    self._total_enemies += 1
                    break

        # Goal
        self._set_cell(W - 2, ground_y - 1, "G")

        # Agent
        self._agent_x = 2
        self._agent_y = ground_y - 1
        self._facing = 1

    # ------------------------------------------------------------------
    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        # Per-step time cost: standing still, flailing, or sitting in a trench
        # bleeds score, so reaching the goal quickly strictly dominates timing
        # out. Tiny enough that a skilled run (well under ~120 steps) keeps the
        # vast majority of its +1.0 ceiling.
        reward = -self._STEP_COST

        if action_name == "LEFT":
            self._try_move(-1, 0)
            self._facing = -1
            self._agent_dir = (-1, 0)
        elif action_name == "RIGHT":
            self._try_move(1, 0)
            self._facing = 1
            self._agent_dir = (1, 0)
        elif action_name == "JUMP":
            self._start_jump()
            self._agent_dir = (0, -1)
        elif action_name == "JUMP_RIGHT":
            self._start_jump()
            self._try_move(1, 0)
            self._facing = 1
            self._agent_dir = (1, 0)
        elif action_name == "THROW":
            self._add_entity(
                "shuriken", "-", self._agent_x + self._facing, self._agent_y,
                dx=self._facing,
            )
            reward += self._resolve_shuriken_contacts()

        self._process_jump()

        # Check goal
        ch = self._world_at(self._agent_x, self._agent_y)
        if ch == "G":
            self._message = "Reached the goal!"
            return self._GOAL_REWARD + reward, True, {}

        # Enemy collision with agent
        for e in self._entities:
            if e.alive and e.etype == "enemy" and e.x == self._agent_x and e.y == self._agent_y:
                self._message = "Hit by an enemy!"
                return reward - self._DEATH_PENALTY, True, {"killed_by": "enemy"}

        return reward, False, {}

    # ------------------------------------------------------------------
    def _resolve_shuriken_contacts(self) -> float:
        reward = 0.0
        shurikens = [e for e in self._entities if e.alive and e.etype == "shuriken"]
        enemies = [e for e in self._entities if e.alive and e.etype == "enemy"]
        for shuriken in shurikens:
            ch = self._world_at(shuriken.x, shuriken.y)
            if ch == "B":
                self._set_cell(shuriken.x, shuriken.y, "\u00b7")
                shuriken.alive = False
                continue
            if self._is_solid(shuriken.x, shuriken.y):
                shuriken.alive = False
                continue
            for enemy in enemies:
                if enemy.alive and enemy.x == shuriken.x and enemy.y == shuriken.y:
                    enemy.alive = False
                    shuriken.alive = False
                    self._enemies_killed += 1
                    reward += self._enemy_reward()
                    self._message = "Defeated an enemy!"
                    break
        return reward

    def _enemy_reward(self) -> float:
        if self._total_enemies <= 0:
            return 0.0
        return self._ENEMY_BUDGET / self._total_enemies

    # ------------------------------------------------------------------
    def _advance_entities(self) -> float:
        """Move shurikens and enemies; handle collisions."""
        reward = 0.0
        for e in self._entities:
            if not e.alive:
                continue
            if e.etype == "shuriken":
                e.x += e.dx
                # Out of bounds
                if e.x < 0 or e.x >= self._world_w:
                    e.alive = False
                    continue
                # Hit breakable wall
                ch = self._world_at(e.x, e.y)
                if ch == "B":
                    self._set_cell(e.x, e.y, "\u00b7")
                    e.alive = False
                    continue
                # Hit solid wall
                if self._is_solid(e.x, e.y):
                    e.alive = False
                    continue
                # Hit enemy
                for other in self._entities:
                    if other.alive and other.etype == "enemy" and other.x == e.x and other.y == e.y:
                        other.alive = False
                        e.alive = False
                        self._enemies_killed += 1
                        reward += self._enemy_reward()
                        self._message = "Defeated an enemy!"
            elif e.etype == "enemy":
                nx = e.x + e.dx
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

        reward += self._resolve_shuriken_contacts()
        self._entities = [e for e in self._entities if e.alive]
        for e in self._entities:
            if e.etype == "enemy" and e.x == self._agent_x and e.y == self._agent_y:
                self._message = "Hit by an enemy!"
                self._entity_terminated = True
                return reward - self._DEATH_PENALTY
        return reward

    # ------------------------------------------------------------------
    def _is_solid(self, x: int, y: int) -> bool:
        ch = self._world_at(x, y)
        return ch in ("\u2588", "\u25ac", "+", "|", "-", "B")

    def _symbol_meaning(self, ch: str) -> str:
        m: dict[str, str] = {
            "\u00b7": "empty",
            "\u25ac": "ground",
            "\u2588": "platform",
            "B": "breakable wall",
            "G": "goal (finishes the level)",
            "E": "enemy",
            "-": "shuriken",
            "@": "you",
        }
        return m.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        enemies = sum(
            1 for e in self._entities
            if e.alive and e.etype == "enemy"
        )
        extra = (
            f"Kills: {self._enemies_killed}"
            f"  Enemies: {enemies}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Traverse the level as a ninja and reach the goal (G). Throw "
            "shurikens (THROW) to defeat enemies (E) and break walls (B). "
            "The ground has pits (gaps where the ground tile is empty); step "
            "into one and you fall into a trench you cannot climb out of, so "
            "clear each pit with JUMP_RIGHT and keep pressing RIGHT while "
            "airborne to glide across. Killing every enemy yields +0.4 total; "
            "reaching the goal yields +0.6 (best case +1.0). Each step costs a "
            "small amount of score, and touching an enemy ends the run with a "
            "penalty, so move deliberately toward the goal."
        )
