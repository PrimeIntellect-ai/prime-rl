"""Procgen Heist environment.

Maze with colored keys and locked doors. Agent must collect keys to open
matching doors and reach the goal.

Gym ID: glyphbench/procgen-heist-v0
"""

from __future__ import annotations

from collections import deque
from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase

# Key/door color pairs: (key_char, door_char, color_name)
_COLORS: list[tuple[str, str, str]] = [
    ("r", "R", "red"),
    ("b", "B", "blue"),
    ("y", "Y", "yellow"),
]


class HeistEnv(ProcgenBase):
    """Procgen Heist: maze with colored keys and locked doors."""

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
        ),
    )

    MAZE_W = 21
    MAZE_H = 21
    # Reward shaping: +0.4 split across first-time key pickups, +0.6 on goal.
    # Collecting all keys and reaching the goal gives cumulative +1.0.
    _KEY_BUDGET = 0.4
    _GOAL_REWARD = 0.6

    def env_id(self) -> str:
        return "glyphbench/procgen-heist-v0"

    def _generate_level(self, seed: int) -> None:
        w, h = self.MAZE_W, self.MAZE_H
        self._init_world(w, h, fill="\u2588")
        self._gen_maze(w, h)

        # Agent at top-left
        self._agent_x, self._agent_y = 1, 1

        # Collect all open floor cells (excluding agent start)
        floor_cells: list[tuple[int, int]] = []
        for yy in range(1, h - 1):
            for xx in range(1, w - 1):
                if self._world_at(xx, yy) == "\u00b7" and (xx, yy) != (1, 1):
                    floor_cells.append((xx, yy))
        self.rng.shuffle(floor_cells)

        # Place goal
        self._goal_x, self._goal_y = floor_cells.pop()
        self._set_cell(self._goal_x, self._goal_y, "G")

        # Place 3 keys and 3 doors
        self._keys_held: set[str] = set()
        for key_ch, door_ch, _color in _COLORS:
            if len(floor_cells) < 2:
                break
            # Place key on a floor cell
            kx, ky = floor_cells.pop()
            self._set_cell(kx, ky, key_ch)
            # Place door: find a corridor cell (has exactly 2 floor neighbors).
            # Renamed loop var from `dx,dy` (which shadowed the neighbour
            # delta names below) to `door_x,door_y` \u2014 the prior name collision
            # caused intermittent same-cell key+door overwrites.
            door_placed = False
            for i, (door_x, door_y) in enumerate(floor_cells):
                adj_floor = 0
                for ndx, ndy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if self._world_at(door_x + ndx, door_y + ndy) in ("\u00b7", "G", "r", "b", "y"):
                        adj_floor += 1
                if adj_floor == 2:
                    self._set_cell(door_x, door_y, door_ch)
                    floor_cells.pop(i)
                    door_placed = True
                    break
            if not door_placed and floor_cells:
                door_x, door_y = floor_cells.pop()
                self._set_cell(door_x, door_y, door_ch)

        # --- Reachability fix: ensure keys are obtainable in sequence ---
        self._fix_key_reachability(w, h)

    # ------------------------------------------------------------------
    def _bfs_reachable(
        self, start: tuple[int, int], w: int, h: int, passable_doors: set[str]
    ) -> set[tuple[int, int]]:
        """BFS from *start*, treating walls and locked doors as impassable.

        *passable_doors* is the set of door chars (e.g. ``{"R", "B"}``)
        whose matching keys have already been collected, so they can be
        traversed.
        """
        visited: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque()
        queue.append(start)
        visited.add(start)
        while queue:
            cx, cy = queue.popleft()
            for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + ddx, cy + ddy
                if (nx, ny) in visited:
                    continue
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                ch = self._world_at(nx, ny)
                if ch == "\u2588":
                    continue
                # A door that is NOT passable blocks movement.
                if ch in ("R", "B", "Y") and ch not in passable_doors:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
        return visited

    def _fix_key_reachability(self, w: int, h: int) -> None:
        """Ensure each key is reachable without needing to pass through its
        own matching door.  Keys are processed in order; once a key is
        confirmed reachable, its door becomes passable for subsequent
        checks.
        """
        key_door_map = {"r": "R", "b": "B", "y": "Y"}
        start = (self._agent_x, self._agent_y)
        collected_doors: set[str] = set()  # doors passable so far

        for key_ch, door_ch in key_door_map.items():
            # Find the key position on the grid (if it exists).
            key_pos: tuple[int, int] | None = None
            for yy in range(h):
                for xx in range(w):
                    if self._world_at(xx, yy) == key_ch:
                        key_pos = (xx, yy)
                        break
                if key_pos is not None:
                    break
            if key_pos is None:
                continue  # key not placed

            reachable = self._bfs_reachable(start, w, h, collected_doors)
            if key_pos in reachable:
                # Key is already reachable; mark its door as passable.
                collected_doors.add(door_ch)
                continue

            # Key is NOT reachable -- relocate it to a reachable floor cell.
            reachable_floors = [
                (rx, ry)
                for (rx, ry) in reachable
                if self._world_at(rx, ry) == "\u00b7"
                and (rx, ry) != start
            ]
            if reachable_floors:
                idx = int(self.rng.integers(0, len(reachable_floors)))
                nx, ny = reachable_floors[idx]
                # Clear old key position
                self._set_cell(key_pos[0], key_pos[1], "\u00b7")
                self._set_cell(nx, ny, key_ch)
            else:
                carved = False
                for rx, ry in reachable:
                    for ndx, ndy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = rx + ndx, ry + ndy
                        if not (1 <= nx < w - 1 and 1 <= ny < h - 1):
                            continue
                        if self._world_at(nx, ny) != "\u2588":
                            continue
                        self._set_cell(key_pos[0], key_pos[1], "\u00b7")
                        self._set_cell(nx, ny, key_ch)
                        carved = True
                        break
                    if carved:
                        break
            # Mark this door as passable for subsequent keys.
            collected_doors.add(door_ch)

    def _gen_maze(self, w: int, h: int) -> None:
        """Carve a maze using recursive backtracking."""
        self._set_cell(1, 1, "\u00b7")
        stack = [(1, 1)]
        while stack:
            cx, cy = stack[-1]
            neighbors: list[tuple[int, int, int, int]] = []
            for ddx, ddy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + ddx, cy + ddy
                if (
                    1 <= nx < w - 1
                    and 1 <= ny < h - 1
                    and self._world_at(nx, ny) == "\u2588"
                ):
                    neighbors.append((nx, ny, cx + ddx // 2, cy + ddy // 2))
            if neighbors:
                idx = int(self.rng.integers(0, len(neighbors)))
                nx, ny, wx, wy = neighbors[idx]
                self._set_cell(wx, wy, "\u00b7")
                self._set_cell(nx, ny, "\u00b7")
                stack.append((nx, ny))
            else:
                stack.pop()

    def _is_solid(self, x: int, y: int) -> bool:
        ch = self._world_at(x, y)
        # Walls and locked doors are solid
        if ch == "\u2588":
            return True
        if ch in ("R", "B", "Y"):
            # Door is solid unless we have the matching key
            color_map = {"R": "r", "B": "b", "Y": "y"}
            return color_map[ch] not in self._keys_held
        return False

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        if action_name == "LEFT":
            self._try_move(-1, 0)
        elif action_name == "RIGHT":
            self._try_move(1, 0)
        elif action_name == "UP":
            self._try_move(0, -1)
        elif action_name == "DOWN":
            self._try_move(0, 1)

        # Check for key pickup
        cell = self._world_at(self._agent_x, self._agent_y)
        if cell in ("r", "b", "y"):
            self._keys_held.add(cell)
            self._set_cell(self._agent_x, self._agent_y, "\u00b7")
            reward += self._KEY_BUDGET / len(_COLORS)
            color_names = {"r": "red", "b": "blue", "y": "yellow"}
            self._message = f"Picked up {color_names[cell]} key!"

        # Check for door opening (agent walks onto door cell)
        if cell in ("R", "B", "Y"):
            # If we got here, the door was opened via _is_solid check
            self._set_cell(self._agent_x, self._agent_y, "\u00b7")
            color_names = {"R": "red", "B": "blue", "Y": "yellow"}
            self._message = f"Opened {color_names[cell]} door!"

        # Check goal
        if self._agent_x == self._goal_x and self._agent_y == self._goal_y:
            reward += self._GOAL_REWARD
            terminated = True
            self._message = "You reached the goal!"

        info["keys_held"] = list(self._keys_held)
        return reward, terminated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        color_names = {"r": "red", "b": "blue", "y": "yellow"}
        if self._keys_held:
            held = ", ".join(
                color_names.get(k, k)
                for k in sorted(self._keys_held)
            )
        else:
            held = "none"
        # Goal coordinates removed — the gem is visible on the grid via its
        # legend glyph, the agent should locate it visually.
        extra = f"Keys held: {held}"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Navigate the maze, collect colored keys (r/b/y) to open matching "
            "doors (R/B/Y), and reach the goal (G). Reward: +0.4 total split "
            "equally across first-time key pickups, plus +0.6 on the goal."
        )

    def _symbol_meaning(self, ch: str) -> str:
        meanings = {
            "\u00b7": "empty",
            "\u2588": "wall",
            "G": "goal",
            "r": "red key",
            "b": "blue key",
            "y": "yellow key",
            "R": "red door (locked)",
            "B": "blue door (locked)",
            "Y": "yellow door (locked)",
        }
        return meanings.get(ch, super()._symbol_meaning(ch))
