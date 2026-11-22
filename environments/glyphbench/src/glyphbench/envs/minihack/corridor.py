"""MiniHack Corridor environments.

Connected rooms in sequence with 1-cell-wide corridors between them.
Player starts in the first room, must detour to collect a brass key,
unlock the final corridor door, and reach the stairs in the last room.

Mechanic added in phase 3: D3 patrol jitter — each room has 1 patrolling
kobold whose 2-tile patrol path is randomised per seed. Key + stair y
are now also randomised per seed (previously every reset gave the same
trajectory).

Variants:
  * R2: 2 rooms
  * R3: 3 rooms
  * R5: 5 rooms

Each room has a 5x5 interior (7x7 including walls).
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MOVE_VECTORS, MiniHackBase
from glyphbench.envs.minihack.creatures import KOBOLD, Creature
from glyphbench.envs.minihack.items import BRASS_KEY


class _CorridorBase(MiniHackBase):
    """Base class for corridor environments."""

    _num_rooms: int = 2
    _room_interior: int = 5  # interior width/height of each room
    _corridor_len: int = 3  # length of corridor between rooms
    _locked_door_pos: tuple[int, int] = (0, 0)
    _PROGRESS_BUDGET = 0.5
    _GOAL_REWARD = 0.5

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        # patrol routes: each kobold has a list of waypoints to cycle through
        self._patrol_routes: dict[int, list[tuple[int, int]]] = {}
        self._patrol_phase: dict[int, int] = {}
        self._rewarded_milestones: set[str] = set()

    def _generate_level(self, seed: int) -> None:
        ri = self._room_interior
        cl = self._corridor_len
        nr = self._num_rooms

        total_w = nr * (ri + 2) + (nr - 1) * cl
        total_h = ri + 2

        self._init_grid(total_w, total_h)
        self._rewarded_milestones = set()
        for y in range(1, total_h - 1):
            for x in range(1, total_w - 1):
                self._place_wall(x, y)

        # Carve rooms
        for i in range(nr):
            room_x_start = i * (ri + 2 + cl)
            for y in range(1, ri + 1):
                for x in range(room_x_start + 1, room_x_start + ri + 1):
                    if 1 <= x < total_w - 1:
                        self._grid[y][x] = "·"

        # Carve corridors between rooms — randomised vertical position
        # within the room interior so different seeds yield different paths.
        corridor_y = int(self.rng.integers(1, total_h - 1))
        for i in range(nr - 1):
            room_i_right = i * (ri + 2 + cl) + ri + 1
            room_next_left = (i + 1) * (ri + 2 + cl)
            for x in range(room_i_right, room_next_left + 1):
                if 0 < x < total_w - 1:
                    self._grid[corridor_y][x] = "·"

        # Player at random position in first room
        first_room_x0 = 1
        first_room_x1 = ri
        first_room_y0 = 1
        first_room_y1 = ri
        px = int(self.rng.integers(first_room_x0, first_room_x1 + 1))
        py = int(self.rng.integers(first_room_y0, first_room_y1 + 1))
        self._place_player(px, py)

        # Brass key at random non-player position in first room
        key_candidates = [
            (x, y)
            for y in range(first_room_y0, first_room_y1 + 1)
            for x in range(first_room_x0, first_room_x1 + 1)
            if (x, y) != (px, py)
        ]
        idx = int(self.rng.integers(0, len(key_candidates)))
        self._place_item(*key_candidates[idx], BRASS_KEY)

        # Locked door at the boundary of the final room
        final_room_left_wall = (nr - 1) * (ri + 2 + cl)
        self._place_door(final_room_left_wall, corridor_y)
        self._locked_door_pos = (final_room_left_wall, corridor_y)

        # Stairs at random position in last room (not on the door cell)
        last_room_x0 = (nr - 1) * (ri + 2 + cl) + 1
        last_room_x1 = last_room_x0 + ri - 1
        last_room_x1 = min(last_room_x1, total_w - 2)
        last_room_y0 = 1
        last_room_y1 = ri
        stair_candidates = [
            (x, y)
            for y in range(last_room_y0, last_room_y1 + 1)
            for x in range(last_room_x0, last_room_x1 + 1)
        ]
        idx = int(self.rng.integers(0, len(stair_candidates)))
        sx, sy = stair_candidates[idx]
        self._place_stairs(sx, sy)

        # D3: spawn a patrolling kobold per room (skip the start room — the
        # player is already there, and the agent has no defensive items).
        # Patrol = 2-tile waypoint cycle inside the room.
        self._patrol_routes = {}
        self._patrol_phase = {}
        for ri_idx in range(1, nr):
            rx0 = ri_idx * (ri + 2 + cl) + 1
            rx1 = rx0 + ri - 1
            ry0 = 1
            ry1 = ri
            interior = [
                (x, y)
                for y in range(ry0, ry1 + 1)
                for x in range(rx0, rx1 + 1)
                if (x, y) != (sx, sy)
                and self._grid[y][x] == "·"
            ]
            if len(interior) < 2:
                continue
            shuffled = [interior[i] for i in self.rng.permutation(len(interior))]
            wp_a, wp_b = shuffled[0], shuffled[1]
            kobold = Creature.spawn(KOBOLD, wp_a[0], wp_a[1])
            self._creatures.append(kobold)
            ck = id(kobold)
            self._patrol_routes[ck] = [wp_a, wp_b]
            self._patrol_phase[ck] = 0

    def _move_monsters(self) -> None:
        """Patrol jitter AI: kobolds cycle waypoints rather than chasing."""
        for c in self._creatures:
            if c.hp <= 0:
                continue
            ck = id(c)
            route = self._patrol_routes.get(ck)
            if route is None:
                # Fall back to default chase AI for non-patrol creatures
                px, py = self._player_pos
                dx = 0 if c.x == px else (1 if c.x < px else -1)
                dy = 0 if c.y == py else (1 if c.y < py else -1)
                nx, ny = c.x + dx, c.y + dy
                if (nx, ny) == (px, py):
                    self._player_hp -= max(1, c.ctype.damage)
                    self._message += f" The {c.ctype.name} hits you!"
                elif (
                    self._is_walkable_for_monster(nx, ny)
                    and self._creature_at(nx, ny) is None
                ):
                    c.x, c.y = nx, ny
                continue
            target = route[self._patrol_phase[ck]]
            if (c.x, c.y) == target:
                # Switch to next waypoint
                self._patrol_phase[ck] = (self._patrol_phase[ck] + 1) % len(route)
                target = route[self._patrol_phase[ck]]
            tx, ty = target
            dx = 0 if c.x == tx else (1 if c.x < tx else -1)
            dy = 0 if c.y == ty else (1 if c.y < ty else -1)
            nx, ny = c.x + dx, c.y + dy
            # Adjacent to player → attack
            px, py = self._player_pos
            if (nx, ny) == (px, py):
                self._player_hp -= max(1, c.ctype.damage)
                self._message += f" The {c.ctype.name} hits you!"
                continue
            if (
                self._is_walkable_for_monster(nx, ny)
                and self._creature_at(nx, ny) is None
            ):
                c.x, c.y = nx, ny

    def _has_key(self) -> bool:
        return any(item.name == BRASS_KEY.name for item in self._inventory)

    def _milestone_reward(self) -> float:
        return self._PROGRESS_BUDGET / (self._num_rooms + 1)

    def _award_milestone(self, name: str) -> float:
        if name in self._rewarded_milestones:
            return 0.0
        self._rewarded_milestones.add(name)
        return self._milestone_reward()

    def _room_index_at(self, pos: tuple[int, int]) -> int | None:
        x, y = pos
        if not (1 <= y <= self._room_interior):
            return None
        stride = self._room_interior + 2 + self._corridor_len
        for idx in range(self._num_rooms):
            room_x0 = idx * stride + 1
            room_x1 = room_x0 + self._room_interior - 1
            if room_x0 <= x <= room_x1:
                return idx
        return None

    def _collect_progress_reward(self) -> float:
        reward = 0.0
        if self._has_key():
            reward += self._award_milestone("key")
        room_idx = self._room_index_at(self._player_pos)
        if room_idx is not None and room_idx > 0:
            reward += self._award_milestone(f"room-{room_idx + 1}")
        return reward

    def _progress_info(self) -> list[str]:
        return sorted(self._rewarded_milestones)

    def _finish_turn(
        self,
        *,
        reward: float = 0.0,
        terminated: bool = False,
        info: dict[str, Any] | None = None,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info = {} if info is None else info

        if self._player_hp <= 0:
            terminated = True
            self._message = (self._message + " You die.").strip()
            info["cause_of_death"] = (
                "combat" if "hit" in self._message.lower() else "hazard"
            )
            info["progress_milestones"] = self._progress_info()
            return self._render_current_observation(), reward - 1.0, terminated, False, info

        self._move_monsters()

        if self._player_hp <= 0:
            terminated = True
            self._message = (self._message + " You die.").strip()
            info["cause_of_death"] = "monster"
            info["progress_milestones"] = self._progress_info()
            return self._render_current_observation(), reward - 1.0, terminated, False, info

        if self._goal_pos and self._player_pos == self._goal_pos:
            terminated = True
            reward += self._GOAL_REWARD
            self._message = "You reach the stairs. You descend."
            info["goal_reached"] = True

        info["player_pos"] = self._player_pos
        info["hp"] = self._player_hp
        info["progress_milestones"] = self._progress_info()
        return self._render_current_observation(), reward, terminated, False, info

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy
            if self._terrain_at(nx, ny) == "⊞":
                if not self._has_key():
                    self._message = "The corridor door is locked. Find the brass key."
                    return self._finish_turn(
                        info={"player_pos": self._player_pos, "hp": self._player_hp}
                    )
                self._grid[ny][nx] = "·"
                self._player_pos = (nx, ny)
                self._message = "You unlock the corridor door with the brass key."
                reward = self._award_milestone("door")
                return self._finish_turn(
                    reward=reward,
                    info={"player_pos": self._player_pos, "hp": self._player_hp}
                )
        obs, reward, terminated, truncated, info = super()._step(action)
        progress_reward = self._collect_progress_reward()
        if terminated and info.get("goal_reached"):
            reward = reward - 1.0 + self._GOAL_REWARD
        reward += progress_reward
        if progress_reward:
            obs = self._render_current_observation()
        info["progress_milestones"] = self._progress_info()
        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        return (
            f"Navigate through {self._num_rooms} connected rooms to the stairs "
            "(⇣). A locked corridor door blocks the final room: detour to the "
            "brass key (() in the first room, PICKUP it while standing on it, "
            "then unlock the door by moving into it. Each room beyond the "
            "first contains a patrolling kobold (k) cycling between two "
            "tiles — time your moves to avoid being attacked. Reward: +0.5 "
            "total split across first key pickup, first entry into each later "
            "room, and unlocking the door; reaching stairs adds +0.5. Death "
            "adds -1 while preserving earned progress."
        )


class MiniHackCorridorR2Env(_CorridorBase):
    """2-room corridor environment."""

    _num_rooms = 2

    def env_id(self) -> str:
        return "glyphbench/minihack-corridor-r2-v0"


class MiniHackCorridorR3Env(_CorridorBase):
    """3-room corridor environment."""

    _num_rooms = 3

    def env_id(self) -> str:
        return "glyphbench/minihack-corridor-r3-v0"


class MiniHackCorridorR5Env(_CorridorBase):
    """5-room corridor environment."""

    _num_rooms = 5

    def env_id(self) -> str:
        return "glyphbench/minihack-corridor-r5-v0"
