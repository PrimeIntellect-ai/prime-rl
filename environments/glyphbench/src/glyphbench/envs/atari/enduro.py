"""Atari Enduro environment.

Racing game. Pass cars in lanes to score.

Gym ID: glyphbench/atari-enduro-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class EnduroEnv(AtariBase):
    """Enduro: lane-based racing game.

    12x20 grid. Pass other cars to score. Avoid collisions.
    Must pass a target number of cars each day.

    Actions: NOOP, LEFT, RIGHT, ACCELERATE, BRAKE
    Pattern A: +1/_WIN_TARGET per car overtaken (full-scope = 600
    cars). An overtake credits only when (a) the agent is NOT in
    the same lane as the car at the moment it scrolls off the
    bottom AND (b) ACCELERATE is the action chosen on that tick.
    No failure penalty (crashes only slow you down).
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "LEFT", "RIGHT",
            "ACCELERATE", "BRAKE",
        ),
        descriptions=(
            "do nothing",
            "move left one lane",
            "move right one lane",
            "speed up",
            "slow down",
        ),
    )

    _WIDTH = 12
    _HEIGHT = 20
    _NUM_LANES = 3
    _LANE_WIDTH = 3
    _ROAD_LEFT = 1
    _PLAYER_ROW = 17
    _CARS_TARGET = 20

    # Pattern A full-scope target: 600 cars overtaken.
    # Phase 3.5: 3x the original 200 because uniform random play
    # plus the lane gate plus the recent-accel gate still credits
    # ~3.5 cars per 100 steps. Combined with the recent-accel gate
    # AND the same-tick credit window below, random cumulative
    # plateaus well below +1.0 within 10000 turns.
    _WIN_TARGET: int = 600

    # Phase 3.5 active-driving gate: an overtake only counts if
    # ACCELERATE was the action chosen this very tick
    # (tick - last_accel_tick == 0). A skilled agent always
    # ACCELERATEs and credits every lane-mismatch scroll-off;
    # random play uses ACCELERATE only 1/5 of the time, cutting
    # credit rate by a factor of 5.
    _ACCEL_WINDOW: int = 0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._player_lane: int = 1
        self._speed: int = 2
        self._cars_passed: int = 0
        self._day: int = 1
        self._spawn_timer: int = 0
        self._traffic: list[AtariEntity] = []
        self._progress_count: int = 0
        self._tick: int = 0
        # Sentinel: large negative so initial state has no "recent accel".
        self._last_accel_tick: int = -10**9

    def env_id(self) -> str:
        return "glyphbench/atari-enduro-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _lane_x(self, lane: int) -> int:
        return (
            self._ROAD_LEFT
            + 1
            + lane * self._LANE_WIDTH
            + self._LANE_WIDTH // 2
        )

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._traffic = []
        self._player_lane = 1
        self._speed = 2
        self._cars_passed = 0
        self._day = 1
        self._spawn_timer = 0
        self._lives = 1
        self._tick = 0
        self._last_accel_tick = -10**9

        self._player_x = self._lane_x(self._player_lane)
        self._player_y = self._PLAYER_ROW
        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick += 1

        # Player controls
        if action_name == "LEFT":
            if self._player_lane > 0:
                self._player_lane -= 1
                self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            if self._player_lane < self._NUM_LANES - 1:
                self._player_lane += 1
                self._player_dir = (1, 0)
        elif action_name == "ACCELERATE":
            self._speed = min(4, self._speed + 1)
            # Phase 3.5: mark active driving for overtake credit.
            self._last_accel_tick = self._tick
        elif action_name == "BRAKE":
            self._speed = max(1, self._speed - 1)

        self._player_x = self._lane_x(self._player_lane)

        # Spawn traffic
        self._spawn_timer += 1
        if self._spawn_timer >= max(3, 7 - self._speed):
            self._spawn_timer = 0
            rng = self.rng
            lane = int(rng.integers(0, self._NUM_LANES))
            cx = self._lane_x(lane)
            car = self._add_entity(
                "traffic", "C", cx, 1
            )
            car.data["lane"] = lane
            car.data["speed"] = 1
            self._traffic.append(car)

        # Move traffic (relative to player speed).
        # Audit BROKEN fix (Phase 3): a car only counts as 'overtaken'
        # if the agent is NOT in the same lane when it scrolls
        # off-bottom. Previously every off-screen car credited +1
        # regardless of lane, so a stationary random agent passively
        # maxed reward.
        # Phase 3.5: even with the lane gate, ~67% of spawned cars are
        # in adjacent lanes under uniform random spawn over 3 lanes,
        # so random hits +1 within ~2300 steps. Tie credit to active
        # driving: ACCELERATE must be the action chosen this tick
        # (tick - last_accel_tick <= _ACCEL_WINDOW; window=0).
        # Combined with the bumped _WIN_TARGET, random cumulative
        # caps well below +1 over 10000 turns.
        drift = self._speed - 1
        accel_recent = (
            self._tick - self._last_accel_tick <= self._ACCEL_WINDOW
        )
        to_remove: list[AtariEntity] = []
        for car in self._traffic:
            if not car.alive:
                to_remove.append(car)
                continue
            car.y += drift
            # Car passed off bottom
            if car.y >= self._HEIGHT:
                car.alive = False
                same_lane = (
                    car.data["lane"] == self._player_lane
                )
                if not same_lane and accel_recent:
                    self._cars_passed += 1
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                to_remove.append(car)
            # Collision (no fail; just slow down)
            elif (
                car.y >= self._PLAYER_ROW - 1
                and car.y <= self._PLAYER_ROW
                and car.data["lane"] == self._player_lane
            ):
                car.alive = False
                to_remove.append(car)
                self._speed = 1
                self._message = "Crash! Speed reset."

        for c in to_remove:
            if c in self._traffic:
                self._traffic.remove(c)

        # Day progression
        if self._cars_passed >= self._CARS_TARGET * self._day:
            self._day += 1
            self._message = f"Day {self._day}! Keep going!"

        # Win check
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "All cars overtaken!"

        info["cars_passed"] = self._cars_passed
        info["speed"] = self._speed
        info["day"] = self._day
        self._redraw()
        return reward, False, info

    def _redraw(self) -> None:
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")

        # Road borders
        road_r = (
            self._ROAD_LEFT
            + 1
            + self._NUM_LANES * self._LANE_WIDTH
        )
        for y in range(self._HEIGHT):
            self._set_cell(self._ROAD_LEFT, y, "│")
            if road_r < self._WIDTH:
                self._set_cell(road_r, y, "│")

        # Lane dividers (dashed)
        for lane in range(1, self._NUM_LANES):
            lx = self._ROAD_LEFT + 1 + lane * self._LANE_WIDTH
            for y in range(self._HEIGHT):
                if y % 3 == 0:
                    self._set_cell(lx, y, ":")

        # Shoulder/grass
        for y in range(self._HEIGHT):
            for x in range(self._ROAD_LEFT):
                self._set_cell(x, y, "~")
            for x in range(road_r + 1, self._WIDTH):
                self._set_cell(x, y, "~")

        # Traffic cars
        for car in self._traffic:
            if car.alive and 0 <= car.y < self._HEIGHT:
                self._set_cell(car.x, car.y, "C")

    def _advance_entities(self) -> None:
        # Handled in _game_step
        pass

    def _render_current_observation(self, **kw: Any):  # type: ignore[override]
        """Append enduro-specific HUD on top of base HUD.

        Base HUD provides ``Step: T / N`` (X4 cross-cutting).
        """
        from glyphbench.core.observation import GridObservation

        obs = super()._render_current_observation()
        target = self._CARS_TARGET * self._day
        extra = (
            f"Cars: {self._cars_passed}  "
            f"Speed: {self._speed}  "
            f"Day: {self._day}\n"
            f"Target: {target}/day"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid,
            legend=obs.legend,
            hud=new_hud,
            message=obs.message,
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "│": "road edge",
            ":": "lane divider",
            "~": "grass",
            "C": "traffic car",
            " ": "road",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Race down the highway. Steer LEFT/RIGHT to "
            "change lanes. ACCELERATE to go faster, "
            "BRAKE to slow down. Pass cars to score. "
            "Avoid collisions (they reset your speed)."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Enduro.\n\n"
            "TASK\n"
            "Race a car down a 3-lane highway, passing as many cars "
            "as possible. Each day requires reaching the cumulative "
            "target of 20*day cars passed.\n\n"
            "BOARD\n"
            "12 columns by 20 rows. Road runs left-to-right with road "
            "edges '|' and dashed lane dividers ':'. Three lanes of "
            "width 3. Grass '~' on the shoulder. Your car is fixed at "
            "one row near the bottom; traffic cars 'C' approach from "
            "above. You appear as an arrow glyph in your lane.\n\n"
            "MECHANICS\n"
            "LEFT/RIGHT switch lanes (clamped to 3 lanes). ACCELERATE "
            "bumps your speed (1-4); BRAKE decreases it. Traffic "
            "cars move downward relative to you at speed (speed-1) "
            "each step; they spawn at row 1 every max(3, 7-speed) "
            "steps. When a car scrolls off the bottom row in a "
            "different lane than you AND ACCELERATE is the action "
            "you chose on that very tick, it counts as passed. "
            "Coasting (NOOP/LEFT/RIGHT/BRAKE) never credits an "
            "overtake on its own — active acceleration is required.\n\n"
            "SCORING\n"
            "+1/600 reward per car passed (Pattern A full-scope = "
            "600 cars). Colliding with a car (same lane, row 16-17) "
            "only resets your speed to 1; no reward penalty.\n\n"
            "TERMINATION\n"
            "Episode ends after passing 600 cars or after "
            "max_turns. Reaching the day's car target advances "
            "the day (no bonus), with a message "
            "'Day N! Keep going!'.\n\n"
            "HUD\n"
            "Shows score, cars passed, current speed, day, and "
            "target crossings per day.\n\n"
            + self.action_spec.render_for_prompt()
        )
