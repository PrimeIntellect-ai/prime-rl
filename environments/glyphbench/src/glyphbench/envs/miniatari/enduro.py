"""miniatari Enduro.

Identity: Driver overtakes cars on a narrow scrolling highway.
Win condition: overtake 5 cars.
Reward: Pattern B (multi-objective). +0.16 per active overtake (max 5
overtakes -> +0.80) plus a +0.20 no-damage bonus on a fully clean
finish. Total cap = +1.00.

Mechanic (K3 — no-damage bonus): an overtake is credited ONLY when the
agent presses ACCEL on the tick a car drifts past the player row AND
the agent is in a lane adjacent to that car. Pure NOOP drift never
credits an overtake. Two collisions still terminate the run (no -1; the
forfeited bonus is the punishment).

Gym ID: glyphbench/miniatari-enduro-v0
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniEnduroEnv(MiniatariBase):
    """Mini Enduro: 12x10 narrow road, scrolling-highway feel.

    Player car sits at row 8, choice of 3 lanes (x=2,5,8). Opponent cars
    spawn at row 0 and drift down the road; passing one (it leaves the
    bottom of the screen without colliding) counts as an overtake.
    Hitting an opponent makes the player lose progress (no penalty, but
    a near-collision shock). 2 collisions ends the run as a hard cap.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "ACCEL"),
        descriptions=(
            "do nothing (cruise speed)",
            "shift one lane left",
            "shift one lane right",
            "accelerate (advance traffic two cells this tick)",
        ),
    )

    default_max_turns = 300

    _WIDTH = 12
    _HEIGHT = 10
    _PLAYER_Y = 8
    _LANES = (2, 5, 8)
    _WIN_TARGET = 5
    _MAX_COLLISIONS = 2
    _SPAWN_PROB = 0.7
    # Pattern B reward split: 5 overtakes × 0.16 = 0.80 base; +0.20
    # bonus on a clean finish (no collisions). Total max = 1.0.
    _OVERTAKE_REWARD = 0.16
    _NO_DAMAGE_BONUS = 0.20

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._cars: list[list[int]] = []  # per car: [lane_idx, y]
        self._progress: int = 0
        self._collisions: int = 0
        self._lane_idx: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-enduro-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._collisions = 0
        self._cars = []
        self._lane_idx = 1
        self._player_x = self._LANES[self._lane_idx]
        self._player_y = self._PLAYER_Y
        # Pre-seed 1-2 cars on the road
        rng = self.rng
        for _ in range(2):
            li = int(rng.integers(0, len(self._LANES)))
            y = int(rng.integers(0, 4))
            self._cars.append([li, y])

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        rng = self.rng

        # 1. Player lane change
        if action_name == "LEFT" and self._lane_idx > 0:
            self._lane_idx -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._lane_idx < len(self._LANES) - 1:
            self._lane_idx += 1
            self._player_dir = (1, 0)
        self._player_x = self._LANES[self._lane_idx]

        # 2. Determine traffic step (1 or 2 cells). Active ACCEL also
        # gates the overtake credit: only ACCEL ticks can credit an
        # overtake (K3 mechanic).
        accel = action_name == "ACCEL"
        step = 2 if accel else 1

        # 3. Move cars down by `step`. Cars passing the player row count
        # as overtakes ONLY if the agent ACCELed AND was in a lane
        # adjacent to (i.e. one column off from) the car's lane. Cars in
        # the agent's own lane reaching the player row are collisions.
        survivors: list[list[int]] = []
        active_overtakes_this_tick = 0
        collided = False
        for car in self._cars:
            li, y = car
            new_y = y + step
            if new_y > self._PLAYER_Y:
                # Car passed the player row.
                if li == self._lane_idx:
                    collided = True
                elif accel and abs(li - self._lane_idx) == 1:
                    # Active overtake — credited.
                    active_overtakes_this_tick += 1
                # All other cases (passive drift in non-adjacent lane,
                # passive drift in adjacent lane, or non-ACCEL pass) are
                # silent: no credit.
                continue
            if li == self._lane_idx and new_y == self._PLAYER_Y:
                # Collision: car landed on player
                collided = True
                continue
            car[1] = new_y
            survivors.append(car)
        self._cars = survivors

        remaining_overtakes = max(0, self._WIN_TARGET - self._progress)
        credited_overtakes = min(active_overtakes_this_tick, remaining_overtakes)
        for _ in range(credited_overtakes):
            reward += self._OVERTAKE_REWARD
            self._progress += 1
            self._message = f"Overtake! ({self._progress}/{self._WIN_TARGET})"

        if collided:
            self._collisions += 1
            self._message = f"Collision! ({self._collisions}/{self._MAX_COLLISIONS})"
            if self._collisions >= self._MAX_COLLISIONS:
                self._game_over = True
                self._won = False
                info["progress"] = self._progress
                info["collisions"] = self._collisions
                return reward, True, info

        if self._progress >= self._WIN_TARGET:
            if self._collisions == 0:
                reward += self._NO_DAMAGE_BONUS
                self._message = (
                    f"Clean run! +{self._NO_DAMAGE_BONUS:.2f} no-damage bonus."
                )
            self._on_won()
            info["progress"] = self._progress
            info["collisions"] = self._collisions
            return reward, self._game_over, info

        # 4. Spawn new car at row 0 with some probability
        if rng.random() < self._SPAWN_PROB:
            # Pick a lane that doesn't have a car at y=0 already
            occupied = {c[0] for c in self._cars if c[1] == 0}
            free_lanes = [i for i in range(len(self._LANES)) if i not in occupied]
            if free_lanes:
                li = free_lanes[int(rng.integers(0, len(free_lanes)))]
                self._cars.append([li, 0])

        info["progress"] = self._progress
        info["collisions"] = self._collisions
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Road borders
        for y in range(self._HEIGHT):
            grid[y][0] = "│"
            grid[y][self._WIDTH - 1] = "│"
        # Lane stripes
        for y in range(self._HEIGHT):
            for lane_x in (3, 6):
                grid[y][lane_x] = "·"
        # Cars
        for li, y in self._cars:
            x = self._LANES[li]
            if 0 <= x < self._WIDTH and 0 <= y < self._HEIGHT:
                grid[y][x] = "C"
        # Player
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "asphalt",
            "│": "road edge",
            "·": "lane stripe",
            "C": "opponent car",
            "Y": "your car",
        }

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Overtakes: {self._progress}/{self._WIN_TARGET}    "
            f"Collisions: {self._collisions}/{self._MAX_COLLISIONS}    "
            f"Score: {self._score:.3f}    "
            f"Cars: {len(self._cars)}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Enduro on a 12x10 narrow highway with 3 lanes (x=2, 5, "
            "8). Your car (Y) sits at row 8. Opponent cars (C) spawn at "
            "the top and drift down. LEFT/RIGHT shifts you one lane. "
            "ACCEL makes all opponent cars descend 2 cells this tick "
            "(instead of 1). An overtake counts only when you ACCEL on "
            "the tick a car drifts past your row AND you are in a lane "
            "ADJACENT to that car's lane (one off). Passive drift past "
            "you in a far lane gives no credit. If a car ends up in "
            "your lane at your row, that is a collision; 2 collisions "
            "end the run. Overtake 5 cars to win. Reward: +0.16 per "
            "active overtake (5 × 0.16 = 0.80) plus a +0.20 no-damage "
            "bonus on a clean finish (0 collisions); maximum 1.00."
        )
