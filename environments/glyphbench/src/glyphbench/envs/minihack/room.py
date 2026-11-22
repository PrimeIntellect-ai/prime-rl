"""MiniHack Room environments.

Room variants of increasing difficulty, all built on MiniHackBase.

Gym IDs: glyphbench/minihack-room-{variant}-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MiniHackBase
from glyphbench.envs.minihack.creatures import KOBOLD, ORC, RAT
from glyphbench.envs.minihack.items import BRASS_KEY


class _RoomBase(MiniHackBase):
    """Shared logic for all Room variants."""

    _room_size: int = 5  # interior size
    _has_monsters: bool = False
    _has_traps: bool = False
    _is_dark: bool = False
    _random_start: bool = True
    _random_goal: bool = True
    _high_trap_density: bool = False  # 12-18% on 15x15, ramps up trap count
    _wind_drift_prob: float = 0.0  # 0.0 = no drift; 0.25 = 25% / step (D1)
    # Brass-key gate is OPT-IN: only the trap-15x15 variant uses it. The
    # other 11 Room variants are pure stairs-reach tasks (audit-OK,
    # untouched by Phase 3.5 — see I1 in the audit brief).
    _has_brass_key_gate: bool = False

    def _generate_level(self, seed: int) -> None:
        size = self._room_size + 2
        self._init_grid(size, size)
        self._dark = self._is_dark

        interior = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
        ]
        occupied: set[tuple[int, int]] = set()

        # Player position
        if self._random_start:
            idx = int(self.rng.integers(0, len(interior)))
            px, py = interior[idx]
        else:
            px, py = 1, 1
        self._place_player(px, py)
        occupied.add((px, py))

        # Goal (stairs) at random position, not on player, unless this is
        # one of the fixed baseline rooms. The explicit Random variants keep
        # the seed-varied start/goal contract.
        if self._random_goal:
            while True:
                idx = int(self.rng.integers(0, len(interior)))
                gx, gy = interior[idx]
                if (gx, gy) not in occupied:
                    break
        else:
            preferred_goal = (self._room_size, self._room_size)
            if preferred_goal not in occupied:
                gx, gy = preferred_goal
            else:
                gx, gy = max(
                    (pos for pos in interior if pos not in occupied),
                    key=lambda pos: abs(pos[0] - px) + abs(pos[1] - py),
                )
        self._place_stairs(gx, gy)
        occupied.add((gx, gy))

        if self._has_brass_key_gate:
            key_candidates = [pos for pos in interior if pos not in occupied]
            kx, ky = max(
                key_candidates,
                key=lambda pos: (
                    abs(pos[0] - px)
                    + abs(pos[1] - py)
                    + abs(pos[0] - gx)
                    + abs(pos[1] - gy)
                ),
            )
            self._place_item(kx, ky, BRASS_KEY)
            occupied.add((kx, ky))

        # Traps
        if self._has_traps:
            if self._high_trap_density:
                # 12-18% density of total interior
                interior_count = len(interior)
                n_traps = int(interior_count * 0.15)
                n_traps = max(20, min(n_traps + int(self.rng.integers(-3, 4)), interior_count // 4))
            else:
                n_traps = (
                    int(self.rng.integers(2, 4))  # 2-3
                    if self._room_size <= 5
                    else int(self.rng.integers(5, 9))  # 5-8
                )
            for _ in range(n_traps):
                attempts = 0
                while attempts < 50:
                    idx = int(self.rng.integers(0, len(interior)))
                    tx, ty = interior[idx]
                    if (tx, ty) not in occupied:
                        break
                    attempts += 1
                else:
                    continue
                self._place_trap(tx, ty)
                occupied.add((tx, ty))

        # Monsters
        if self._has_monsters:
            n_monsters = (
                int(self.rng.integers(1, 3))  # 1-2
                if self._room_size <= 5
                else int(self.rng.integers(3, 6))  # 3-5
            )
            monster_types = [RAT, KOBOLD, ORC]
            for i in range(n_monsters):
                while True:
                    idx = int(self.rng.integers(0, len(interior)))
                    mx, my = interior[idx]
                    if (mx, my) not in occupied:
                        break
                ctype = monster_types[i % len(monster_types)]
                self._spawn_creature(ctype, mx, my)
                occupied.add((mx, my))

    def _task_description(self) -> str:
        parts = [
            f"Navigate a {self._room_size}x{self._room_size} "
            f"dungeon room to the stairs (⇣)."
        ]
        if self._has_monsters:
            parts.append(
                "Hostile monsters roam the room -- fight or avoid them."
            )
        if self._has_traps:
            if self._high_trap_density:
                parts.append(
                    "Traps (△) are DENSE (about 15% of cells) -- stepping on "
                    "one deals damage and the trap is sprung."
                )
            else:
                parts.append(
                    "Traps (△) are scattered around -- stepping on one deals damage."
                )
        if self._wind_drift_prob > 0.0:
            parts.append(
                f"Wind: each step has a {int(self._wind_drift_prob * 100)}% "
                "chance of pushing you ±1 cardinal in a random direction. "
                "Plan with slack."
            )
        if self._is_dark:
            parts.append(
                "The room is dark -- you can only see tiles adjacent to you."
            )
        if self._has_brass_key_gate:
            parts.append(
                "A brass key (() is hidden away from the direct route; pick "
                "it up before using the stairs."
            )
        parts.append("Reward: +1 on reaching stairs, -1 on death.")
        return " ".join(parts)

    def _has_key(self) -> bool:
        return any(item.name == BRASS_KEY.name for item in self._inventory)

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        if (
            self._has_brass_key_gate
            and terminated
            and info.get("goal_reached")
            and not self._has_key()
        ):
            terminated = False
            reward = 0.0
            info.pop("goal_reached", None)
            self._message = "The stairs are sealed. Find the brass key first."
            obs = self._render_current_observation()
            return obs, reward, terminated, truncated, info

        # D1 wind drift: with probability _wind_drift_prob, push player ±1 in
        # a random cardinal direction. Drift stops at walls and triggers
        # traps/lava normally.
        if (
            not terminated
            and not truncated
            and self._wind_drift_prob > 0.0
            and self.rng.random() < self._wind_drift_prob
        ):
            cards = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            dx, dy = cards[int(self.rng.integers(0, 4))]
            cx, cy = self._player_pos
            nx, ny = cx + dx, cy + dy
            if self._is_walkable(nx, ny) and self._creature_at(nx, ny) is None:
                self._player_pos = (nx, ny)
                terrain = self._terrain_at(nx, ny)
                msg = " A gust of wind pushes you"
                if terrain == "△":
                    trap_dmg = int(self.rng.integers(1, 4))
                    self._player_hp -= trap_dmg
                    msg += f" into a trap! (-{trap_dmg} HP)"
                    self._grid[ny][nx] = "·"
                else:
                    msg += "."
                self._message = (self._message + msg).strip()
                if self._player_hp <= 0:
                    return self._render_current_observation(), -1.0, True, False, {
                        "cause_of_death": "trap"
                    }
                # Re-check goal after drift
                if self._goal_pos and self._player_pos == self._goal_pos and self._has_key():
                    self._message += " You stagger onto the stairs."
                    return self._render_current_observation(), 1.0, True, False, {
                        "goal_reached": True
                    }
                obs = self._render_current_observation()
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------
# Backward-compat wrapper so existing Room-5x5 tests keep working.
# The old MiniHackRoom5x5Env exposed _agent_x/y, _goal_x/y and
# returned room_size / goal_pos in info.  We patch _step to include
# those extras, and expose the old attributes as properties.
# ------------------------------------------------------------------


class MiniHackRoom5x5Env(_RoomBase):
    """MiniHack Room-5x5: 5x5 interior (7x7 grid)."""

    _room_size = 5
    _random_start = False
    _random_goal = False

    def env_id(self) -> str:
        return "glyphbench/minihack-room-5x5-v0"

    # Backward-compat properties for tests that poke internals
    @property
    def _agent_x(self) -> int:
        return self._player_pos[0]

    @property
    def _agent_y(self) -> int:
        return self._player_pos[1]

    @property
    def _goal_x(self) -> int:
        return self._goal_pos[0] if self._goal_pos else 0

    @property
    def _goal_y(self) -> int:
        return self._goal_pos[1] if self._goal_pos else 0

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Original env never had negative rewards (no monsters/traps)
        info["agent_pos"] = (
            self._player_pos[0] - 1,
            self._player_pos[1] - 1,
        )
        info["goal_pos"] = (
            (self._goal_pos[0] - 1, self._goal_pos[1] - 1)
            if self._goal_pos
            else (-1, -1)
        )
        info["room_size"] = (5, 5)
        info["steps_to_goal"] = self._turn if info.get("goal_reached") else -1
        return obs, reward, terminated, truncated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        # Patch HUD to match old format (includes AC and $)
        floor_items = self._floor_items.get(self._player_pos, [])
        standing_str = ""
        if floor_items:
            item_names = ", ".join(item.name for item in floor_items)
            standing_str = f"    standing on: {item_names}"
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Dlvl: 1    HP: {self._player_hp}/{self._player_max_hp}    "
            f"AC: 10    $: 0"
            f"{standing_str}"
        )
        return GridObservation(
            grid=obs.grid,
            legend=obs.legend,
            hud=hud,
            message=obs.message,
        )


# ------------------------------------------------------------------
# Room-15x15
# ------------------------------------------------------------------


class MiniHackRoom15x15Env(_RoomBase):
    """MiniHack Room-15x15: 15x15 interior (17x17 grid)."""

    _room_size = 15
    _random_start = False
    _random_goal = False

    def env_id(self) -> str:
        return "glyphbench/minihack-room-15x15-v0"


# ------------------------------------------------------------------
# Room-Random variants (explicit seed-varied start and goal)
# ------------------------------------------------------------------


class MiniHackRoomRandom5x5Env(_RoomBase):
    """MiniHack Room-Random-5x5: random player start in a 5x5 room."""

    _room_size = 5
    _random_start = True

    def env_id(self) -> str:
        return "glyphbench/minihack-room-random-5x5-v0"


class MiniHackRoomRandom15x15Env(_RoomBase):
    """MiniHack Room-Random-15x15: random player start in a 15x15 room."""

    _room_size = 15
    _random_start = True

    def env_id(self) -> str:
        return "glyphbench/minihack-room-random-15x15-v0"


# ------------------------------------------------------------------
# Room-Dark variants
# ------------------------------------------------------------------


class MiniHackRoomDark5x5Env(_RoomBase):
    """MiniHack Room-Dark-5x5: dark 5x5 room with limited vision."""

    _room_size = 5
    _random_start = True
    _is_dark = True

    def env_id(self) -> str:
        return "glyphbench/minihack-room-dark-5x5-v0"


class MiniHackRoomDark15x15Env(_RoomBase):
    """MiniHack Room-Dark-15x15: dark 15x15 room with limited vision."""

    _room_size = 15
    _random_start = True
    _is_dark = True

    def env_id(self) -> str:
        return "glyphbench/minihack-room-dark-15x15-v0"


# ------------------------------------------------------------------
# Room-Monster variants
# ------------------------------------------------------------------


class MiniHackRoomMonster5x5Env(_RoomBase):
    """MiniHack Room-Monster-5x5: 5x5 room with 1-2 hostile monsters."""

    _room_size = 5
    _random_start = True
    _has_monsters = True

    def env_id(self) -> str:
        return "glyphbench/minihack-room-monster-5x5-v0"


class MiniHackRoomMonster15x15Env(_RoomBase):
    """MiniHack Room-Monster-15x15: 15x15 room with 3-5 hostile monsters."""

    _room_size = 15
    _random_start = True
    _has_monsters = True

    def env_id(self) -> str:
        return "glyphbench/minihack-room-monster-15x15-v0"


# ------------------------------------------------------------------
# Room-Trap variants
# ------------------------------------------------------------------


class MiniHackRoomTrap5x5Env(_RoomBase):
    """MiniHack Room-Trap-5x5: 5x5 room with 2-3 traps."""

    _room_size = 5
    _random_start = True
    _has_traps = True

    def env_id(self) -> str:
        return "glyphbench/minihack-room-trap-5x5-v0"


class MiniHackRoomTrap15x15Env(_RoomBase):
    """MiniHack Room-Trap-15x15: 15x15 room with HIGH trap density (~15%) +
    D1 wind drift (each step has 25% chance of ±1 cardinal drift) +
    a brass-key gate so the agent can't simply rush diagonally to the
    stairs and ignore the trap field.
    """

    _room_size = 15
    _random_start = True
    _has_traps = True
    _high_trap_density = True
    _wind_drift_prob = 0.25
    _has_brass_key_gate = True

    def env_id(self) -> str:
        return "glyphbench/minihack-room-trap-15x15-v0"


# ------------------------------------------------------------------
# Room-Ultimate variants (monsters + traps + dark)
# ------------------------------------------------------------------


class MiniHackRoomUltimate5x5Env(_RoomBase):
    """MiniHack Room-Ultimate-5x5: dark 5x5 room with monsters and traps."""

    _room_size = 5
    _random_start = True
    _has_monsters = True
    _has_traps = True
    _is_dark = True

    def env_id(self) -> str:
        return "glyphbench/minihack-room-ultimate-5x5-v0"


class MiniHackRoomUltimate15x15Env(_RoomBase):
    """MiniHack Room-Ultimate-15x15: dark 15x15 room with monsters and traps."""

    _room_size = 15
    _random_start = True
    _has_monsters = True
    _has_traps = True
    _is_dark = True

    def env_id(self) -> str:
        return "glyphbench/minihack-room-ultimate-15x15-v0"
