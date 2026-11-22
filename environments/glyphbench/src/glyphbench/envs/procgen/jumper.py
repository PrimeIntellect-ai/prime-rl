"""Procgen Jumper environment.

2D platformer with gravity. Agent runs and jumps across platforms,
avoiding spikes and enemies, to reach the goal at the right end.

Gym ID: glyphbench/procgen-jumper-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase


class JumperEnv(ProcgenBase):
    """Side-scrolling platformer: reach the goal flag at the right end.

    World: 40 wide x 12 tall.  View: 20 x 12.
    Gravity enabled.
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

    # Reward shaping (Pattern A): single terminal goal worth +1.0.
    _GOAL_REWARD = 1.0

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._has_gravity = True
        self._view_w = 20
        self._view_h = 12

    def env_id(self) -> str:
        return "glyphbench/procgen-jumper-v0"

    # ------------------------------------------------------------------
    def _generate_level(self, seed: int) -> None:
        W, H = 40, 12
        self._init_world(W, H, fill="\u00b7")
        ground_y = H - 2

        # Ground row
        for x in range(W):
            self._set_cell(x, ground_y, "\u25ac")
            self._set_cell(x, H - 1, "\u25ac")

        # Gaps in ground
        num_gaps = int(self.rng.integers(2, 5))
        gap_positions: list[int] = []
        for _ in range(num_gaps):
            gx = int(self.rng.integers(6, W - 6))
            gw = int(self.rng.integers(2, 4))
            # avoid overlapping
            overlap = any(abs(gx - p) < 5 for p in gap_positions)
            if overlap or gx < 4 or gx + gw > W - 4:
                continue
            gap_positions.append(gx)
            for dx in range(gw):
                self._set_cell(gx + dx, ground_y, "\u00b7")
                self._set_cell(gx + dx, H - 1, "\u00b7")

        # Floating platforms
        num_plats = int(self.rng.integers(3, 6))
        for _ in range(num_plats):
            px = int(self.rng.integers(4, W - 6))
            py = int(self.rng.integers(ground_y - 5, ground_y - 2))
            pw = int(self.rng.integers(3, 6))
            if py < 1:
                py = 1
            for dx in range(pw):
                if px + dx < W:
                    self._set_cell(px + dx, py, "\u2588")

        # Spikes on ground
        num_spikes = int(self.rng.integers(2, 5))
        for _ in range(num_spikes):
            sx = int(self.rng.integers(6, W - 6))
            if self._world_at(sx, ground_y) == "\u25ac":
                self._set_cell(sx, ground_y - 1, "^")

        # Enemies
        num_enemies = int(self.rng.integers(1, 3))
        for _ in range(num_enemies):
            ex = int(self.rng.integers(8, W - 8))
            if self._world_at(ex, ground_y) == "\u25ac":
                self._add_entity("enemy", "E", ex, ground_y - 1, dx=1)

        # Goal at right end
        self._set_cell(W - 2, ground_y - 1, "G")

        # Agent starts left
        self._agent_x = 2
        self._agent_y = ground_y - 1

    # ------------------------------------------------------------------
    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
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

        # Check spike
        ch = self._world_at(self._agent_x, self._agent_y)
        if ch == "^":
            self._message = "Stepped on a spike!"
            return 0.0, True, {"killed_by": "spike"}

        # Check goal
        if ch == "G":
            self._message = "Reached the goal!"
            return self._GOAL_REWARD, True, {}

        # Enemy collision
        for e in self._entities:
            if e.alive and e.x == self._agent_x and e.y == self._agent_y:
                self._message = "Hit by an enemy!"
                return 0.0, True, {"killed_by": "enemy"}

        # Fell into void
        if self._in_void():
            self._message = "Fell into the void!"
            return 0.0, True, {"killed_by": "fall"}

        return 0.0, False, {}

    def _in_void(self) -> bool:
        return self._agent_y >= self._world_h - 1 and not self._is_solid(
            self._agent_x, self._agent_y
        )

    def _step(self, action: int):
        obs, reward, terminated, truncated, info = super()._step(action)
        if not terminated and self._in_void():
            self._message = "Fell into the void!"
            info["killed_by"] = "fall"
            info["agent_pos"] = (self._agent_x, self._agent_y)
            return self._render_current_observation(), reward, True, truncated, info
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _advance_entities(self) -> float:
        """Enemies patrol on ground, bouncing at edges."""
        for e in self._entities:
            if not e.alive or e.etype != "enemy":
                continue
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
        self._entities = [e for e in self._entities if e.alive]
        for e in self._entities:
            if e.etype == "enemy" and e.x == self._agent_x and e.y == self._agent_y:
                self._message = "Hit by an enemy!"
                self._entity_terminated = True
                return 0.0
        return 0.0

    # ------------------------------------------------------------------
    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        extra = "Goal: reach right end"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    # ------------------------------------------------------------------
    def _symbol_meaning(self, ch: str) -> str:
        m: dict[str, str] = {
            "\u00b7": "empty",
            "\u25ac": "ground",
            "\u2588": "platform",
            "^": "spike (deadly)",
            "G": "goal (+1)",
            "E": "enemy",
            "@": "you",
        }
        return m.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Run and jump across platforms to reach the goal (G) for "
            "+1 reward. Avoid spikes (^), enemies (E), and ground gaps; "
            "falling to the bottom of a gap ends the episode. JUMP_LEFT and "
            "JUMP_RIGHT also move laterally while airborne."
        )
