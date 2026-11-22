"""Procgen StarPilot environment.

Horizontal shoot-em-up. Agent on left, enemies approach from right.

Gym ID: glyphbench/procgen-starpilot-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase


class StarPilotEnv(ProcgenBase):
    """Procgen StarPilot: horizontal shooter with power-ups."""

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
    # Reward shaping (Pattern D): cumulative best case +1.0 (kill the first
    # _ENEMY_TARGET enemies AND grab _POWERUP_TARGET power-ups), worst case
    # -1.0 (terminal collision with an enemy ship).
    #
    # Calibration (2026-05 retune): a stationary FIRE-only spammer used to win
    # (median +1.0, ~59% perfect) because (a) bullets travelled the full row as
    # an impenetrable wall that pre-cleared every approaching enemy, (b) the
    # 8-kill cap saturated in the first tens of turns, and (c) over 500 steps it
    # rarely got hit. Discrimination fixes, all in this file:
    #   - bullets travel ONE cell/tick with SWEPT collision (kills no longer
    #     tunnel on unlucky x-parity; was dx=2 + same-cell-only);
    #   - a fire COOLDOWN so no continuous bullet wall;
    #   - a finite bullet RANGE so a parked ship only controls its own short
    #     firing zone and must MOVE to other rows to clear them;
    #   - swept agent-enemy collision so a fast (dx=-2) enemy can no longer leap
    #     over the agent and a slow enemy landing on it mid-advance is lethal;
    #   - a higher kill quota plus a shorter horizon (see __init__) so own-lane
    #     fire cannot reach the quota in time, while a mobile dodge-and-snipe
    #     policy clears many lanes early and then survives to bank the win.
    # Validated (40 seeds): lookahead planner +0.992 mean / 37 perfect / 40
    # survive vs FIRE-only -0.10 mean / 2 wins, NOOP -0.98, random -0.78.
    _ENEMY_TARGET = 12
    _ENEMY_BUDGET = 0.8
    _POWERUP_TARGET = 2
    _POWERUP_BUDGET = 0.2
    _DEATH_PENALTY = -1.0
    # Ticks the agent must wait between shots. >1 breaks the full-row wall a
    # fire-every-tick policy used to maintain.
    _FIRE_COOLDOWN = 2
    # Bullets expire after travelling this many cells. A short range means the
    # agent only controls a zone directly in front of it, so distant enemies in
    # OTHER lanes are NOT pre-cleared -- the agent must move to a lane to shoot
    # the enemies in it. Aiming/movement (not a fixed bullet wall) is what
    # earns kills toward the quota.
    _BULLET_RANGE = 6

    def __init__(self, max_turns: int = 220) -> None:
        # Shorter horizon than the 500-step default. A win requires earning the
        # full kill/power-up budget AND then surviving the rest of the episode;
        # over 500 steps survival was effectively impossible for any policy
        # (terminal death negates the banked +1.0), so the task was unwinnable.
        # A ~220-step horizon is long enough that a parked FIRE-only agent
        # cannot reach the kill quota from its single lane, yet short enough
        # that a mobile agent which clears lanes early can survive to the end.
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/procgen-starpilot-v0"

    def _generate_level(self, seed: int) -> None:
        w, h = self.GRID_W, self.GRID_H
        self._init_world(w, h, fill=" ")

        # Agent on the left side
        self._agent_x = 2
        self._agent_y = h // 2

        self._enemies_killed = 0
        self._powerups_collected = 0
        self._spawn_timer = int(seed % 12)
        self._scroll_offset = int(seed % 5)
        self._fire_ready = 0  # ticks remaining before the agent may fire again

        initial_y = 1 + int(seed % (h - 2))
        if seed % 2 == 0:
            self._add_entity("enemy", "E", w - 4, initial_y, dx=-1, dy=0)
        else:
            self._add_entity("powerup", "$", w - 4, initial_y, dx=-1, dy=0)

    def _maybe_spawn(self) -> None:
        """Spawn enemies and power-ups from the right edge.

        Enemies are distributed across all lanes. Because the agent only fires
        down its own row within a finite range, racking up the kill quota means
        moving between lanes; a parked agent only ever clears the spawns that
        happen to land in its single lane and falls short of the quota.
        """
        w, h = self.GRID_W, self.GRID_H
        self._spawn_timer += 1

        # Spawn an enemy every fourth step from a random lane. Most are slow
        # (dx=-1); a fast variant (V, dx=-2) appears occasionally to punish a
        # parked agent that lets its lane fill, but is rare enough that a
        # mobile agent can keep ahead of the stream.
        if self._spawn_timer % 4 == 0:
            ey = int(self.rng.integers(1, h - 1))
            speed = -1
            char = "E"
            if float(self.rng.random()) < 0.15:
                char = "V"
                speed = -2
            self._add_entity("enemy", char, w - 1, ey, dx=speed, dy=0)

        # Spawn power-up occasionally.
        if self._spawn_timer % 12 == 0:
            py = int(self.rng.integers(1, h - 1))
            self._add_entity("powerup", "$", w - 1, py, dx=-1, dy=0)

    def _advance_entities(self) -> float:
        """Move all entities with swept collision. Remove out-of-bounds ones.

        Enemies hold their spawn lane (with an occasional vertical wobble), so
        the agent must MOVE to a lane to shoot the enemies in it -- a parked
        agent only ever controls its own short firing zone. Agent-enemy contact
        is tested across the *swept* horizontal path so a fast enemy can no
        longer leap over the agent in a single tick.
        """
        reward = 0.0
        for e in self._entities:
            if not e.alive:
                continue

            # Remember the horizontal span swept this tick so a fast enemy
            # cannot leap over the agent without contact (closes the fast-enemy
            # and turn-order escape windows).
            if e.etype == "enemy":
                e.data["sweep_from"] = e.x

            e.x += e.dx
            e.y += e.dy

            # Remove if out of bounds
            if e.x < 0 or e.x >= self.GRID_W or e.y < 0 or e.y >= self.GRID_H:
                e.alive = False
                continue

            # Enemies: slight vertical wobble so lanes are not perfectly static
            # (a parked agent cannot assume its row stays clear forever).
            if e.etype == "enemy" and float(self.rng.random()) < 0.15:
                wobble = 1 if int(self.rng.integers(0, 2)) == 0 else -1
                ny = e.y + wobble
                if 0 <= ny < self.GRID_H:
                    e.y = ny

        # Swept bullet-enemy resolution: a bullet that crossed an enemy's path
        # during this tick destroys it even if their cells never coincide on a
        # single frame (fixes the dx-parity tunnelling bug). Resolve hits first
        # so a bullet that intercepts an enemy this tick prevents the collision
        # that swept death would otherwise register.
        reward += self._resolve_bullet_hits()

        # Bullets expire once they have travelled their range (after resolving
        # hits, so a bullet still scores on the tick it reaches max range).
        for e in self._entities:
            if e.alive and e.etype == "bullet":
                origin = e.data.get("origin_x", e.x)
                if e.x - origin >= self._BULLET_RANGE:
                    e.alive = False

        self._entities = [e for e in self._entities if e.alive]

        # An enemy that swept across the agent's row this tick is terminal --
        # evaluated over SURVIVING enemies (a bullet kill this tick spares the
        # agent). Apply the additive death penalty and flag the base loop to end
        # the episode (see BaseGlyphEnv via ProcgenBase._step).
        for e in self._entities:
            if e.etype != "enemy" or e.y != self._agent_y:
                continue
            old_x = e.data.get("sweep_from", e.x)
            lo, hi = (e.x, old_x) if e.x <= old_x else (old_x, e.x)
            if lo <= self._agent_x <= hi:
                reward += self._DEATH_PENALTY
                self._entity_terminated = True
                self._message = "Destroyed by an enemy!"
                break
        return reward

    def _resolve_bullet_hits(self) -> float:
        """Destroy enemies whose path a bullet swept across this tick."""
        reward = 0.0
        bullets = [e for e in self._entities if e.etype == "bullet" and e.alive]
        for bullet in bullets:
            b_old = bullet.x - bullet.dx  # position before this tick's move
            b_lo, b_hi = (b_old, bullet.x) if b_old <= bullet.x else (bullet.x, b_old)
            for enemy in self._entities:
                if not enemy.alive or enemy.etype != "enemy":
                    continue
                if enemy.y != bullet.y:
                    continue
                if b_lo <= enemy.x <= b_hi:
                    bullet.alive = False
                    enemy.alive = False
                    if self._enemies_killed < self._ENEMY_TARGET:
                        reward += self._ENEMY_BUDGET / self._ENEMY_TARGET
                    self._enemies_killed += 1
                    break
        return reward

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        if self._fire_ready > 0:
            self._fire_ready -= 1

        # Movement -- constrain to left portion
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
        elif action_name == "UP":
            self._agent_dir = (0, -1)
            ny = self._agent_y - 1
            if ny >= 0:
                self._agent_y = ny
        elif action_name == "DOWN":
            self._agent_dir = (0, 1)
            ny = self._agent_y + 1
            if ny < self.GRID_H:
                self._agent_y = ny
        elif action_name == "FIRE":
            # Cooldown-gated: an agent cannot maintain a continuous bullet
            # wall by firing every tick. Bullets travel one cell per tick;
            # swept collision (in _advance_entities) still makes every shot
            # reliable regardless of x-parity.
            if self._fire_ready == 0:
                bx = self._agent_x + 1
                if bx < self.GRID_W:
                    self._add_entity(
                        "bullet", "-", bx, self._agent_y, dx=1, dy=0,
                        data={"origin_x": bx},
                    )
                    self._fire_ready = self._FIRE_COOLDOWN

        # Spawn new entities
        self._maybe_spawn()

        # Same-cell agent-enemy collision (terminal). The swept variant during
        # _advance_entities catches contacts that only occur mid-move; this
        # catches an enemy already standing on the agent at the frame start.
        for e in self._entities:
            if not e.alive or e.etype != "enemy":
                continue
            if e.x == self._agent_x and e.y == self._agent_y:
                reward += self._DEATH_PENALTY
                terminated = True
                self._message = "Destroyed by an enemy!"
                return reward, terminated, info

        # Same-cell bullet-enemy collision before movement (e.g. point-blank).
        # Swept resolution during _advance_entities handles in-flight hits.
        for bullet in [e for e in self._entities if e.etype == "bullet" and e.alive]:
            for enemy in [
                e for e in self._entities if e.etype == "enemy" and e.alive
            ]:
                if bullet.x == enemy.x and bullet.y == enemy.y:
                    bullet.alive = False
                    enemy.alive = False
                    if self._enemies_killed < self._ENEMY_TARGET:
                        reward += self._ENEMY_BUDGET / self._ENEMY_TARGET
                    self._enemies_killed += 1

        # Check agent picks up power-up
        for e in self._entities:
            if not e.alive or e.etype != "powerup":
                continue
            if e.x == self._agent_x and e.y == self._agent_y:
                e.alive = False
                if self._powerups_collected < self._POWERUP_TARGET:
                    reward += self._POWERUP_BUDGET / self._POWERUP_TARGET
                self._powerups_collected += 1
                self._message = "Power-up collected!"

        # Clean dead entities
        self._entities = [e for e in self._entities if e.alive]

        info["enemies_killed"] = self._enemies_killed
        info["powerups_collected"] = self._powerups_collected
        return reward, terminated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        alive = sum(
            1 for e in self._entities
            if e.alive and e.etype == "enemy"
        )
        gun = "ready" if self._fire_ready == 0 else f"cooldown {self._fire_ready}"
        extra = (
            f"Enemies: {alive}"
            f"  Kills: {self._enemies_killed}/{self._ENEMY_TARGET}"
            f"  Gun: {gun}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "You pilot a ship (@) in a horizontal shoot-em-up. Enemies (E, and "
            "faster V) fly in from the right, each holding its own row (with "
            "an occasional vertical wobble). FIRE shoots a bullet (-) straight "
            "down your CURRENT row only, and reaches a limited distance ahead, "
            "so you must move UP/DOWN to line up the enemies in other rows -- a "
            "stationary ship only ever clears its own lane. Your gun recharges "
            "for a couple of ticks after each shot (see Gun in the HUD), so you "
            "cannot hold a continuous wall of fire. Any enemy that reaches your "
            "position destroys you and ends the episode at -1, so dodge as well "
            "as shoot. "
            f"Each of the first {self._ENEMY_TARGET} kills yields "
            f"+{self._ENEMY_BUDGET}/{self._ENEMY_TARGET}; each of the first "
            f"{self._POWERUP_TARGET} power-ups ($) yields "
            f"+{self._POWERUP_BUDGET}/{self._POWERUP_TARGET} (best case +1.0). "
            "Reward is banked as you earn it; surviving to the end keeps it."
        )

    def _symbol_meaning(self, ch: str) -> str:
        meanings = {
            " ": "space",
            "E": "enemy ship",
            "V": "fast enemy",
            "-": "bullet",
            "$": "power-up (small reward)",
        }
        return meanings.get(ch, super()._symbol_meaning(ch))
