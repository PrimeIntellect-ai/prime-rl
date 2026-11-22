"""Atari Freeway environment.

Guide a chicken across a highway full of traffic.

Gym ID: glyphbench/atari-freeway-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class FreewayEnv(AtariBase):
    """Freeway: guide a chicken across lanes of traffic.

    20x16 grid. Cars move horizontally at various speeds.
    Reach the top to score, then restart at the bottom.

    Actions: NOOP, UP, DOWN
    Pattern A: +1/_WIN_TARGET per successful crossing (full-scope =
    10 crossings). No failure penalty.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN"),
        descriptions=(
            "do nothing",
            "move up one row",
            "move down one row",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 16
    _PLAYER_START_Y = 14
    _GOAL_Y = 1
    _NUM_LANES = 12

    # Pattern A full-scope target: 10 crossings.
    _WIN_TARGET: int = 10

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._lanes: list[list[AtariEntity]] = []
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-freeway-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._lanes = []

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")

        # Goal area (top)
        for x in range(self._WIDTH):
            self._set_cell(x, self._GOAL_Y, "·")

        # Start area (bottom)
        for x in range(self._WIDTH):
            self._set_cell(x, self._PLAYER_START_Y, "·")

        # Create lanes with cars
        rng = self.rng
        for lane_idx in range(self._NUM_LANES):
            lane_y = 2 + lane_idx
            if lane_y >= self._PLAYER_START_Y:
                break

            direction = 1 if lane_idx % 2 == 0 else -1
            # Speed increases with level; at least 1
            speed = 1 + (lane_idx % 3)
            num_cars = 2 + (lane_idx % 2)

            lane_cars: list[AtariEntity] = []
            spacing = self._WIDTH // (num_cars + 1)
            for car_idx in range(num_cars):
                cx = (spacing * (car_idx + 1) + int(rng.integers(0, 3))) % self._WIDTH
                # Cars are 2-3 chars wide; we represent with single char
                car = self._add_entity("car", "C", cx, lane_y, dx=direction)
                car.data["speed"] = speed
                car.data["timer"] = 0
                lane_cars.append(car)
            self._lanes.append(lane_cars)

        # Player at bottom center
        self._player_x = self._WIDTH // 2
        self._player_y = self._PLAYER_START_Y

        self._redraw_lanes()

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Move player
        if action_name == "UP":
            new_y = self._player_y - 1
            if new_y >= self._GOAL_Y:
                self._player_y = new_y
                self._player_dir = (0, -1)
        elif action_name == "DOWN":
            new_y = self._player_y + 1
            if new_y <= self._PLAYER_START_Y:
                self._player_y = new_y
                self._player_dir = (0, 1)

        # Move cars
        for lane in self._lanes:
            for car in lane:
                if not car.alive:
                    continue
                spd = car.data.get("speed", 1)
                car.data["timer"] = car.data.get("timer", 0) + 1
                if car.data["timer"] >= (4 - min(spd, 3)):
                    car.data["timer"] = 0
                    car.x = (car.x + car.dx) % self._WIDTH

        # Check collision with cars
        hit = False
        for lane in self._lanes:
            for car in lane:
                if car.alive and car.x == self._player_x and car.y == self._player_y:
                    hit = True
                    break
            if hit:
                break

        if hit:
            # Push player back to start (no life lost in Freeway)
            self._player_y = self._PLAYER_START_Y
            self._message = "Hit by car! Back to start."

        # Check goal
        if self._player_y <= self._GOAL_Y:
            self._on_point_scored(1)
            if self._progress_count < self._WIN_TARGET:
                reward += 1.0 / self._WIN_TARGET
                self._progress_count += 1
            self._message = "Crossed!"
            self._player_y = self._PLAYER_START_Y

        terminated = False
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            terminated = True
            info["won"] = True
            self._message = "All crossings complete!"

        self._redraw_lanes()
        info["crossings"] = self._score
        return reward, terminated, info

    def _redraw_lanes(self) -> None:
        """Redraw the lane area."""
        # Clear lane area
        for y in range(2, self._PLAYER_START_Y):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")

        # Draw cars
        for lane in self._lanes:
            for car in lane:
                if car.alive and 0 <= car.x < self._WIDTH:
                    self._set_cell(car.x, car.y, "C")

    def _advance_entities(self) -> None:
        # Cars are moved in _game_step
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "border",
            "·": "safe zone",
            "C": "car",
            " ": "road",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        dirs: list[str] = []
        for lane in self._lanes:
            if lane:
                d = lane[0].dx
                dirs.append("→" if d == 1 else "←")
        extra = f"Lane car dirs: {' '.join(dirs)}"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Guide the chicken across the highway. "
            "Press UP to move toward the top. "
            "Avoid cars or you'll be sent back to the start. "
            "Each successful crossing scores +1."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Freeway.\n\n"
            "TASK\n"
            "Guide a chicken from the bottom of a 10-lane highway to "
            "the top without being hit by cars. Each crossing scores "
            "a point; the episode is time-limited, not terminating, "
            "so accumulate as many crossings as possible.\n\n"
            "BOARD\n"
            "20 columns by 16 rows. Borders '-' at top (row 0) and "
            "bottom (row 15). Safe zones '.' run along the top and "
            "bottom of the highway; cars fill the lanes between. "
            "Cars 'C' in each lane alternate direction (even lanes "
            "move right, odd lanes left) with variable speeds (1-3 "
            "cells per few steps). You appear as an arrow glyph.\n\n"
            "MECHANICS\n"
            "UP / DOWN move the chicken 1 row per step (clamped "
            "between rows 1 and 14). Cars advance each step on a "
            "per-lane speed timer and wrap around horizontally. "
            "Colliding with a car (same cell) teleports you back to "
            "the start safe zone; no life lost. Reaching row 1 "
            "awards a crossing and resets you at row 14.\n\n"
            "SCORING\n"
            "+1/10 reward per successful crossing (Pattern A full-"
            "scope = 10 crossings). No penalty for being hit by a "
            "car (only the setback to the start row).\n\n"
            "TERMINATION\n"
            "Episode terminates after 10 crossings (cumulative reward "
            "plateaus at +1.0) or when max_turns is reached.\n\n"
            "HUD\n"
            "Shows score, and per-lane car "
            "direction arrows.\n\n"
            + self.action_spec.render_for_prompt()
        )
