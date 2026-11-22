"""Procgen BossFight environment.

Space arena with a multi-phase boss. Agent at bottom dodges boss projectiles
and fires back.

Gym ID: glyphbench/procgen-bossfight-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase

_BOSS_MAX_HP = 5


class BossFightEnv(ProcgenBase):
    """Procgen BossFight: multi-phase boss with projectile patterns."""

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
            "fire bullet upward",
        ),
    )

    GRID_W = 20
    GRID_H = 14
    # Reward shaping (Pattern D): +0.5 distributed across the _BOSS_MAX_HP
    # hits + +0.5 on defeating the boss = +1.0 best case; terminal -1.0
    # on getting hit by a boss projectile.
    _HIT_BUDGET = 0.5
    _DEFEAT_REWARD = 0.5
    _DEATH_PENALTY = -1.0

    def env_id(self) -> str:
        return "glyphbench/procgen-bossfight-v0"

    def _generate_level(self, seed: int) -> None:
        w, h = self.GRID_W, self.GRID_H
        self._init_world(w, h, fill=" ")

        # Place a few static obstacles (cover) along the central rows so
        # arena layout differs per seed. Cover obstacles use the wall
        # glyph; the agent must navigate around them.
        n_cover = int(self.rng.integers(2, 5))
        cover_y_choices = list(range(4, h - 3))
        for _ in range(n_cover):
            cy = int(self.rng.choice(cover_y_choices))
            cx = int(self.rng.integers(2, w - 2))
            self._set_cell(cx, cy, "█")

        # Agent at bottom; column varies with seed so the optimal angle
        # changes per fight.
        self._agent_x = int(self.rng.integers(2, w - 2))
        self._agent_y = h - 2

        # Boss starts at random column on top row; horizontal direction
        # picked from rng so the patrol arc differs per seed.
        self._boss_x = int(self.rng.integers(2, w - 2))
        self._boss_y = 1
        self._boss_hp = _BOSS_MAX_HP
        self._boss_dir = int(self.rng.choice([-1, 1]))
        # Random initial timer offset so attack cadence is desynchronised
        # across seeds (a strategy memorised on seed A no longer maps to
        # seed B without re-planning).
        self._boss_timer = int(self.rng.integers(0, 4))
        self._hits_on_boss = 0
        # Per-seed attack variant (0 / 1 / 2). Variants change which
        # diagonal the spread shot leads with and which side the aimed
        # shot leans toward. Boss patterns are still deterministic given
        # the variant + timer, but vary across seeds.
        self._attack_variant = int(self.rng.integers(0, 3))

    def _boss_phase(self) -> int:
        """Determine boss phase based on HP."""
        if self._boss_hp > 3:
            return 1
        if self._boss_hp > 1:
            return 2
        return 3

    def _boss_attack(self) -> None:
        """Boss fires projectiles based on current phase + attack variant.

        Phase determines cadence (faster as HP drops). Attack variant
        chooses which spread arm leads and which side the aimed shot
        favours. Phase 3 also summons a short-lived "minion" projectile
        every 5 ticks (HP <= 25%): see audit triage G5 phases-with-summon.
        """
        phase = self._boss_phase()
        self._boss_timer += 1
        v = self._attack_variant

        if phase == 1:
            # Phase 1: single shot downward every 4 steps.
            if self._boss_timer % 4 == 0:
                self._add_entity(
                    "boss_bullet", "v", self._boss_x, self._boss_y + 1,
                    dx=0, dy=1,
                )
        elif phase == 2:
            # Phase 2: spread shot every 3 steps. Variant flips which
            # diagonal arm spawns first so per-seed bullet patterns
            # differ even when timing aligns.
            if self._boss_timer % 3 == 0:
                self._add_entity(
                    "boss_bullet", "v", self._boss_x, self._boss_y + 1,
                    dx=0, dy=1,
                )
                lead_dx, trail_dx = (1, -1) if v == 0 else (-1, 1)
                self._add_entity(
                    "boss_bullet",
                    "\\" if lead_dx == 1 else "/",
                    self._boss_x + lead_dx, self._boss_y + 1,
                    dx=lead_dx, dy=1,
                )
                self._add_entity(
                    "boss_bullet",
                    "/" if trail_dx == -1 else "\\",
                    self._boss_x + trail_dx, self._boss_y + 1,
                    dx=trail_dx, dy=1,
                )
        else:
            # Phase 3: rapid fire every 2 steps + aimed shots + summon
            # every 5 ticks (HP <= ~25%).
            if self._boss_timer % 2 == 0:
                self._add_entity(
                    "boss_bullet", "v", self._boss_x, self._boss_y + 1,
                    dx=0, dy=1,
                )
                # Aimed shot toward agent; variant biases the lean by
                # one cell so the line changes per seed.
                aim_dx = 0
                if self._agent_x < self._boss_x:
                    aim_dx = -1
                elif self._agent_x > self._boss_x:
                    aim_dx = 1
                if aim_dx == 0:
                    aim_dx = 1 if v == 1 else -1
                self._add_entity(
                    "boss_bullet", "o", self._boss_x + aim_dx,
                    self._boss_y + 1, dx=aim_dx, dy=1,
                )
            if self._boss_timer % 5 == 0:
                # Summon a slower-moving wide projectile from the
                # off-side; variant 2 spawns from the right, 0/1 from
                # the left.
                spawn_dx = 1 if v == 2 else -1
                spawn_x = self._boss_x + spawn_dx * 2
                spawn_x = max(0, min(self.GRID_W - 1, spawn_x))
                self._add_entity(
                    "boss_bullet", "x", spawn_x, self._boss_y + 1,
                    dx=spawn_dx, dy=1,
                )

    def _advance_entities(self) -> float:
        """Move all entities and resolve post-advance collisions with
        swap detection (audit MINOR fix: previously the pre-advance
        check missed bullets that ended on the agent / boss tile after
        they moved that tick).
        """
        # If the boss already died this tick (terminal in _game_step),
        # nothing else needs to move and any further collision check
        # would risk double-counting.
        if self._boss_hp <= 0:
            return 0.0

        # Snapshot pre-advance positions for swap detection.
        prev: dict[int, tuple[int, int]] = {}
        for e in self._entities:
            if e.alive:
                prev[id(e)] = (e.x, e.y)
        boss_prev = (self._boss_x, self._boss_y)
        agent_pos = (self._agent_x, self._agent_y)

        # Move boss along its patrol arc (top row, away from cover).
        self._boss_x += self._boss_dir
        if self._boss_x <= 1 or self._boss_x >= self.GRID_W - 2:
            self._boss_dir = -self._boss_dir

        # Move entities; bullets that hit cover terrain die in place
        # so cover is a meaningful obstruction.
        for e in self._entities:
            if not e.alive:
                continue
            e.x += e.dx
            e.y += e.dy
            if e.x < 0 or e.x >= self.GRID_W or e.y < 0 or e.y >= self.GRID_H:
                e.alive = False
                continue
            if self._is_solid(e.x, e.y):
                e.alive = False

        # Post-advance: player bullets vs boss (cohabit OR swap).
        reward = 0.0
        for bullet in [
            e for e in self._entities if e.etype == "bullet" and e.alive
        ]:
            bx0, by0 = prev.get(id(bullet), (bullet.x, bullet.y))
            cohabit = bullet.x == self._boss_x and bullet.y == self._boss_y
            swap = (
                bullet.x == boss_prev[0] and bullet.y == boss_prev[1]
                and self._boss_x == bx0 and self._boss_y == by0
            )
            if (cohabit or swap) and self._boss_hp > 0:
                bullet.alive = False
                self._boss_hp -= 1
                self._hits_on_boss += 1
                reward += self._HIT_BUDGET / _BOSS_MAX_HP
                self._message = f"Hit! Boss HP: {self._boss_hp}/{_BOSS_MAX_HP}"
                if self._boss_hp <= 0:
                    reward += self._DEFEAT_REWARD
                    self._entity_terminated = True
                    self._message = "Boss defeated!"
                    break

        # Post-advance: boss bullets vs agent (cohabit OR swap).
        if not self._entity_terminated:
            for e in self._entities:
                if not e.alive or e.etype != "boss_bullet":
                    continue
                bx0, by0 = prev.get(id(e), (e.x, e.y))
                cohabit = e.x == self._agent_x and e.y == self._agent_y
                swap = (
                    e.x == agent_pos[0] and e.y == agent_pos[1]
                    and self._agent_x == bx0 and self._agent_y == by0
                )
                if cohabit or swap:
                    # Additive: preserve any boss-hit reward earned this
                    # step before the agent was hit by a boss projectile.
                    reward += self._DEATH_PENALTY
                    self._entity_terminated = True
                    self._message = "Hit by boss projectile!"
                    break

        self._entities = [e for e in self._entities if e.alive]
        return reward

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        # Movement (constrain to arena, blocked by cover obstacles).
        if action_name == "LEFT":
            self._agent_dir = (-1, 0)
            nx = self._agent_x - 1
            if nx >= 0 and not self._is_solid(nx, self._agent_y):
                self._agent_x = nx
        elif action_name == "RIGHT":
            self._agent_dir = (1, 0)
            nx = self._agent_x + 1
            if nx < self.GRID_W and not self._is_solid(nx, self._agent_y):
                self._agent_x = nx
        elif action_name == "UP":
            self._agent_dir = (0, -1)
            ny = self._agent_y - 1
            if ny >= 0 and not self._is_solid(self._agent_x, ny):
                self._agent_y = ny
        elif action_name == "DOWN":
            self._agent_dir = (0, 1)
            ny = self._agent_y + 1
            if ny < self.GRID_H and not self._is_solid(self._agent_x, ny):
                self._agent_y = ny
        elif action_name == "FIRE":
            by = self._agent_y - 1
            if by >= 0 and not self._is_solid(self._agent_x, by):
                self._add_entity(
                    "bullet", "|", self._agent_x, by, dx=0, dy=-1,
                )

        # Boss attacks (spawns boss bullets that will be advanced and
        # collision-checked by _advance_entities).
        self._boss_attack()

        # Pre-advance contemporaneous overlap (e.g. agent walked onto
        # an existing bullet, or player just FIREd into an adjacent
        # boss). _advance_entities does the post-advance / swap-aware
        # check on the next tick.
        for bullet in [
            e for e in self._entities if e.etype == "bullet" and e.alive
        ]:
            if bullet.x == self._boss_x and bullet.y == self._boss_y:
                bullet.alive = False
                self._boss_hp -= 1
                self._hits_on_boss += 1
                reward += self._HIT_BUDGET / _BOSS_MAX_HP
                self._message = f"Hit! Boss HP: {self._boss_hp}/{_BOSS_MAX_HP}"

        if self._boss_hp <= 0:
            reward += self._DEFEAT_REWARD
            terminated = True
            self._message = "Boss defeated!"
            return reward, terminated, info

        for e in self._entities:
            if not e.alive or e.etype != "boss_bullet":
                continue
            if e.x == self._agent_x and e.y == self._agent_y:
                # Additive: preserve any boss-hit reward earned this step.
                reward += self._DEATH_PENALTY
                terminated = True
                self._message = "Hit by boss projectile!"
                return reward, terminated, info

        self._entities = [e for e in self._entities if e.alive]

        info["boss_hp"] = self._boss_hp
        info["boss_phase"] = self._boss_phase()
        info["hits_on_boss"] = self._hits_on_boss
        return reward, terminated, info

    def _render_current_observation(self) -> GridObservation:
        """Override to render boss and add boss state to HUD."""
        if self._boss_hp > 0:
            boss_entity = self._add_entity(
                "boss_render", "B", self._boss_x, self._boss_y,
            )
            obs = super()._render_current_observation()
            boss_entity.alive = False
            self._entities = [
                e for e in self._entities if e.alive
            ]
        else:
            obs = super()._render_current_observation()
        phase = self._boss_phase()
        # Boss x/y removed — boss glyph is visible on the grid.
        extra = (
            f"Boss: HP={self._boss_hp}/{_BOSS_MAX_HP}"
            f"  phase={phase}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "You face a boss (B) in a space arena with cover obstacles "
            "(█) scattered between you. FIRE shoots bullets (|) "
            "upward. Cover blocks both your bullets and boss projectiles. "
            "The boss has three phases as HP decreases: single shots "
            "(HP > 3), spread shots (HP 2-3), and rapid aimed fire with "
            "summoned wide projectiles (HP <= 1). Dodge boss projectiles "
            "(v, \\, /, o, x). Per-seed boss start, patrol direction, and "
            "attack variant change the dance. "
            f"Each hit yields +0.5/{_BOSS_MAX_HP}; defeating the boss yields "
            f"+0.5 (best case +1.0). Getting hit ends the episode at -1.0. "
            f"Boss HP: {_BOSS_MAX_HP}."
        )

    def _symbol_meaning(self, ch: str) -> str:
        meanings = {
            " ": "space",
            "█": "cover (blocks bullets)",
            "B": "boss",
            "|": "your bullet",
            "v": "boss bullet (down)",
            "\\": "boss bullet (diagonal)",
            "/": "boss bullet (diagonal)",
            "o": "boss aimed shot",
            "x": "boss summoned projectile",
        }
        return meanings.get(ch, super()._symbol_meaning(ch))
