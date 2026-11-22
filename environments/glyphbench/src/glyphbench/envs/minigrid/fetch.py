"""MiniGrid Fetch and PutNear environments.

Fetch (N3 — stake doubling):
  First PICKUP commits. Correct = +reward, wrong = -0.5, episode ends.
  Forces colour-grounding because brute-force-and-retry is no longer free.

PutNear (C2 — forking dependency):
  Pick up the target and drop it adjacent to the goal object, BUT the drop
  position must NOT be within Chebyshev-1 of the target's *original* cell.
  Forces real movement of the target, not a touch-and-release shortcut.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from glyphbench.envs.minigrid.objects import Ball, Key

_COLORS = ["red", "green", "blue", "yellow", "purple"]
_OBJ_TYPES: list[type[Ball | Key]] = [Ball, Key]


class _FetchBase(MiniGridBase):
    _room_size: int = 5
    _num_objects: int = 2

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._target_obj: Ball | Key | None = None
        self._target_desc: str = ""

    def _generate_grid(self, seed: int) -> None:
        size = self._room_size + 2
        self._init_grid(size, size)

        occupied: set[tuple[int, int]] = set()
        objects: list[tuple[int, int, Ball | Key]] = []

        for i in range(self._num_objects):
            while True:
                ox = int(self.rng.integers(1, size - 1))
                oy = int(self.rng.integers(1, size - 1))
                if (ox, oy) not in occupied:
                    break
            occupied.add((ox, oy))
            color = _COLORS[i % len(_COLORS)]
            obj_cls = _OBJ_TYPES[i % len(_OBJ_TYPES)]
            obj = obj_cls(color=color)
            self._place_obj(ox, oy, obj)
            objects.append((ox, oy, obj))

        # Pick target
        target_idx = int(self.rng.integers(0, len(objects)))
        _, _, self._target_obj = objects[target_idx]
        self._target_desc = self._target_obj.legend_name()

        # Agent
        while True:
            ax = int(self.rng.integers(1, size - 1))
            ay = int(self.rng.integers(1, size - 1))
            if (ax, ay) not in occupied:
                break
        self._place_agent(ax, ay, DIR_RIGHT)

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        # Stake doubling: detect a PICKUP attempt BEFORE the parent step
        # consumes it, so we know whether the picked-up object is the target.
        name = self.action_spec.names[action]
        commit: str | None = None
        if name == "PICKUP" and self._carrying is None:
            fx, fy = self._front_pos()
            if 0 <= fx < self._grid_w and 0 <= fy < self._grid_h:
                front_obj = self._get_obj(fx, fy)
                if front_obj is not None and getattr(front_obj, "can_pickup", False):
                    commit = "correct" if front_obj is self._target_obj else "wrong"

        obs, reward, terminated, truncated, info = super()._step(action)
        if commit == "correct":
            terminated = True
            reward = self._reward_on_goal()
            info["target_fetched"] = True
        elif commit == "wrong":
            terminated = True
            # Additive so any reward from super()._step (e.g. step bonuses)
            # survives the wrong-pickup penalty in the cumulative MC return.
            reward += -0.5
            info["target_fetched"] = False
            info["wrong_pickup"] = True

        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        return (
            f"A room with {self._num_objects} coloured pickup-able objects "
            f"(ball/key objects). Find and pick up the {self._target_desc} "
            "with PICKUP. The FIRST PICKUP commits — correct = +reward, "
            "wrong = -0.5 reward and the episode ends. "
            "Reward = 1 - 0.9 * (steps / max_steps) on correct pickup."
        )


class MiniGridFetch5x5N2Env(_FetchBase):
    _room_size = 5
    _num_objects = 2

    def env_id(self) -> str:
        return "glyphbench/minigrid-fetch-5x5-n2-v0"


class MiniGridFetch6x6N2Env(_FetchBase):
    _room_size = 6
    _num_objects = 2

    def env_id(self) -> str:
        return "glyphbench/minigrid-fetch-6x6-n2-v0"


class MiniGridFetch8x8N3Env(_FetchBase):
    _room_size = 8
    _num_objects = 3

    def env_id(self) -> str:
        return "glyphbench/minigrid-fetch-8x8-n3-v0"


class _PutNearBase(MiniGridBase):
    _room_size: int = 6
    _num_objects: int = 2

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._target_obj: Ball | Key | None = None
        self._target_desc: str = ""
        self._target_orig_pos: tuple[int, int] = (0, 0)
        self._goal_pos_putnear: tuple[int, int] = (0, 0)
        self._goal_desc: str = ""

    def _generate_grid(self, seed: int) -> None:
        size = self._room_size + 2
        for _attempt in range(200):
            self._init_grid(size, size)

            occupied: set[tuple[int, int]] = set()
            objects: list[tuple[int, int, Ball | Key]] = []

            for i in range(self._num_objects):
                while True:
                    ox = int(self.rng.integers(1, size - 1))
                    oy = int(self.rng.integers(1, size - 1))
                    if (ox, oy) not in occupied:
                        break
                occupied.add((ox, oy))
                color = _COLORS[i % len(_COLORS)]
                obj_cls = _OBJ_TYPES[i % len(_OBJ_TYPES)]
                obj = obj_cls(color=color)
                self._place_obj(ox, oy, obj)
                objects.append((ox, oy, obj))

            # Pick target to carry. Track its ORIGINAL position so we can
            # reject "drop in same neighbourhood" cheats later.
            target_idx = int(self.rng.integers(0, len(objects)))
            ox, oy, self._target_obj = objects[target_idx]
            self._target_orig_pos = (ox, oy)
            self._target_desc = self._target_obj.legend_name()

            # Pick goal position (where to put near) — the other object
            goal_idx = (target_idx + 1) % len(objects)
            gx, gy, goal_obj = objects[goal_idx]
            self._goal_pos_putnear = (gx, gy)
            self._goal_desc = goal_obj.legend_name()
            if self._legal_putnear_drop_tiles():
                break
        else:  # pragma: no cover - defensive against impossible parameters
            raise RuntimeError("could not sample a solvable PutNear layout")

        # Agent
        while True:
            ax = int(self.rng.integers(1, size - 1))
            ay = int(self.rng.integers(1, size - 1))
            if (ax, ay) not in occupied:
                break
        self._place_agent(ax, ay, DIR_RIGHT)

    def _legal_putnear_drop_tiles(self) -> list[tuple[int, int]]:
        gx, gy = self._goal_pos_putnear
        ox, oy = self._target_orig_pos
        legal: list[tuple[int, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                fx, fy = gx + dx, gy + dy
                if not (0 < fx < self._grid_w - 1 and 0 < fy < self._grid_h - 1):
                    continue
                if abs(fx - ox) <= 1 and abs(fy - oy) <= 1:
                    continue
                cell = self._grid[fy][fx]
                if cell is not None:
                    continue
                legal.append((fx, fy))
        return legal

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        was_carrying = self._carrying
        obs, reward, terminated, truncated, info = super()._step(action)

        # Check if agent just dropped the target object near the goal,
        # AND not within the target's original Chebyshev-1 neighbourhood.
        if was_carrying is self._target_obj and self._carrying is None:
            gx, gy = self._goal_pos_putnear
            ox, oy = self._target_orig_pos
            fx, fy = self._front_pos()
            adj_to_goal = abs(fx - gx) <= 1 and abs(fy - gy) <= 1
            in_orig_zone = abs(fx - ox) <= 1 and abs(fy - oy) <= 1
            if adj_to_goal and not in_orig_zone:
                terminated = True
                reward = self._reward_on_goal()
                info["put_near_success"] = True
            elif adj_to_goal and in_orig_zone:
                # Mark the rejection in info so analysis can distinguish.
                info["put_near_rejected_origin_zone"] = True

        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        return (
            f"A room with {self._num_objects} objects. Pick up the "
            f"{self._target_desc} and drop it adjacent (any of the 8 "
            f"neighbours) to the {self._goal_desc}. Important: the drop "
            f"position must NOT be the target's ORIGINAL cell or any of its "
            f"8 neighbours (the 3x3 box centered on the original cell) — "
            f"remember that original cell before pickup and actually move it. "
            f"Reward = 1 - 0.9 * (steps / max_steps)."
        )


class MiniGridPutNear6x6N2Env(_PutNearBase):
    _room_size = 6
    _num_objects = 2

    def env_id(self) -> str:
        return "glyphbench/minigrid-putnear-6x6-n2-v0"


class MiniGridPutNear8x8N3Env(_PutNearBase):
    _room_size = 8
    _num_objects = 3

    def env_id(self) -> str:
        return "glyphbench/minigrid-putnear-8x8-n3-v0"
