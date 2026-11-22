"""Atari Robot Tank environment.

Top-down tank combat with damage systems.

Gym ID: glyphbench/atari-robotank-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


def _sign(a: int, b: int) -> int:
    return 1 if a > b else (-1 if a < b else 0)

class RobotankEnv(AtariBase):
    """Robot Tank: tank combat with damage systems.

    20x20 grid. Destroy enemy tanks while managing damage to
    sensors (radar, cannon, treads, video). Each hit disables
    a random system. Lose all sensors = episode ends (single-life).

    Actions: NOOP, LEFT, RIGHT, UP, DOWN, FIRE
    Pattern D: +1/_WIN_TARGET per tank destroyed,
    -1 when all sensors are knocked out (terminates).
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"),
        descriptions=(
            "do nothing", "move left", "move right",
            "move up", "move down",
            "fire in last moved direction",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _SENSORS = ("radar", "cannon", "treads", "video")

    # Pattern D full-scope: 12 tank kills before death.
    _WIN_TARGET: int = 12
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._tanks: list[AtariEntity] = []
        self._bullets: list[AtariEntity] = []
        self._enemy_bullets: list[AtariEntity] = []
        self._step_counter = 0
        self._fire_dx = 0
        self._fire_dy = -1
        self._sensors: dict[str, bool] = {}
        self._kills = 0
        self._bushes: set[tuple[int, int]] = set()
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-robotank-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        self._kills = 0
        self._fire_dx, self._fire_dy = 0, -1
        return super()._reset(seed)

    def _reset_pos(self) -> None:
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT - 3

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._tanks = []
        self._bullets = []
        self._enemy_bullets = []
        self._step_counter = 0
        self._sensors = {s: True for s in self._SENSORS}
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")
        self._bushes = set()
        rng = self.rng
        for _ in range(8):
            bx = int(rng.integers(3, self._WIDTH - 3))
            by = int(rng.integers(3, self._HEIGHT - 3))
            self._bushes.add((bx, by))
        self._reset_pos()
        for _ in range(min(2 + self._level, 5)):
            self._spawn_enemy()
        self._redraw()

    def _spawn_enemy(self) -> None:
        rng = self.rng
        for _ in range(20):
            ex = int(rng.integers(2, self._WIDTH - 2))
            ey = int(rng.integers(2, self._HEIGHT // 2))
            if abs(ex - self._player_x) + abs(
                ey - self._player_y
            ) > 4:
                t = self._add_entity("tank", "T", ex, ey)
                t.data.update(hp=2, timer=0, fire_cd=0)
                self._tanks.append(t)
                return

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1
        can_move = self._sensors["treads"]
        nx, ny = self._player_x, self._player_y
        if can_move:
            if action_name == "LEFT":
                nx -= 1
                self._fire_dx, self._fire_dy = -1, 0
                self._player_dir = (-1, 0)
            elif action_name == "RIGHT":
                nx += 1
                self._fire_dx, self._fire_dy = 1, 0
                self._player_dir = (1, 0)
            elif action_name == "UP":
                ny -= 1
                self._fire_dx, self._fire_dy = 0, -1
                self._player_dir = (0, -1)
            elif action_name == "DOWN":
                ny += 1
                self._fire_dx, self._fire_dy = 0, 1
                self._player_dir = (0, 1)
        elif action_name in ("LEFT", "RIGHT", "UP", "DOWN"):
            self._message = "Treads damaged!"
        if (
            can_move and 0 < nx < self._WIDTH - 1
            and 0 < ny < self._HEIGHT - 1
        ):
            self._player_x, self._player_y = nx, ny
        if action_name == "FIRE":
            if self._sensors["cannon"] and len(self._bullets) < 2:
                b = self._add_entity(
                    "bullet", "*",
                    self._player_x + self._fire_dx,
                    self._player_y + self._fire_dy,
                )
                b.data.update(
                    bdx=self._fire_dx, bdy=self._fire_dy
                )
                self._bullets.append(b)
            elif not self._sensors["cannon"]:
                self._message = "Cannon damaged!"
        # Move player bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.x += b.data["bdx"]
            b.y += b.data["bdy"]
            if not (0 < b.x < self._WIDTH - 1
                    and 0 < b.y < self._HEIGHT - 1):
                b.alive = False
        # Bullet-tank hits
        for b in self._bullets:
            if not b.alive:
                continue
            for t in self._tanks:
                if t.alive and t.x == b.x and t.y == b.y:
                    b.alive = False
                    t.data["hp"] -= 1
                    if t.data["hp"] <= 0:
                        t.alive = False
                        self._kills += 1
                        self._on_point_scored(1)
                        if self._progress_count < self._WIN_TARGET:
                            reward += 1.0 / self._WIN_TARGET
                            self._progress_count += 1
                        self._message = "Tank destroyed!"
                    else:
                        self._message = "Tank hit!"
                    break
        # Enemy AI
        for t in self._tanks:
            if not t.alive:
                continue
            t.data["timer"] += 1
            if t.data["timer"] >= 3:
                t.data["timer"] = 0
                if self._sensors["radar"] or self.rng.random() < 0.3:
                    dx = _sign(self._player_x, t.x)
                    dy = _sign(self._player_y, t.y)
                    if abs(self._player_x - t.x) >= abs(
                        self._player_y - t.y
                    ):
                        ntx, nty = t.x + dx, t.y
                    else:
                        ntx, nty = t.x, t.y + dy
                    if (0 < ntx < self._WIDTH - 1
                            and 0 < nty < self._HEIGHT - 1):
                        t.x, t.y = ntx, nty
            t.data["fire_cd"] += 1
            if t.data["fire_cd"] >= 10:
                dist = abs(t.x - self._player_x) + abs(
                    t.y - self._player_y
                )
                if dist < 10:
                    t.data["fire_cd"] = 0
                    fdx = _sign(self._player_x, t.x)
                    fdy = _sign(self._player_y, t.y)
                    if abs(fdx) >= abs(fdy):
                        fdy = 0
                    else:
                        fdx = 0
                    if fdx or fdy:
                        eb = self._add_entity(
                            "enemy_bullet", "o",
                            t.x + fdx, t.y + fdy,
                        )
                        eb.data.update(bdx=fdx, bdy=fdy)
                        self._enemy_bullets.append(eb)
        # Move enemy bullets
        for eb in self._enemy_bullets:
            if not eb.alive:
                continue
            eb.x += eb.data["bdx"]
            eb.y += eb.data["bdy"]
            if not (0 < eb.x < self._WIDTH - 1
                    and 0 < eb.y < self._HEIGHT - 1):
                eb.alive = False
            elif (eb.x == self._player_x
                  and eb.y == self._player_y):
                eb.alive = False
                self._take_damage()
        # Player-tank collision (rams the tank)
        for t in self._tanks:
            if (t.alive and t.x == self._player_x
                    and t.y == self._player_y):
                t.alive = False
                self._take_damage()
                self._kills += 1
                self._on_point_scored(1)
                if self._progress_count < self._WIN_TARGET:
                    reward += 1.0 / self._WIN_TARGET
                    self._progress_count += 1
        # If all sensors knocked out, the player died
        if self._game_over:
            reward += self._DEATH_PENALTY
            return reward, self._game_over, info

        # Win check
        if self._progress_count >= self._WIN_TARGET:
            self._game_over = True
            info["won"] = True
            self._message = "All tanks destroyed!"
            return reward, self._game_over, info

        # Respawn
        alive = [t for t in self._tanks if t.alive]
        if len(alive) < 2 and self._step_counter % 25 == 0:
            self._spawn_enemy()
        self._bullets = [b for b in self._bullets if b.alive]
        self._enemy_bullets = [
            e for e in self._enemy_bullets if e.alive
        ]
        self._tanks = [t for t in self._tanks if t.alive]
        self._redraw()
        info["sensors"] = dict(self._sensors)
        info["kills"] = self._kills
        return reward, self._game_over, info

    def _take_damage(self) -> None:
        working = [s for s, ok in self._sensors.items() if ok]
        if working:
            s = working[int(self.rng.integers(len(working)))]
            self._sensors[s] = False
            self._message = f"{s.title()} damaged!"
        if not any(self._sensors.values()):
            # Single-life model: total destruction ends the episode.
            self._on_life_lost()
            self._message = "All systems destroyed! Game Over."

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        for bx, by in self._bushes:
            self._set_cell(bx, by, "~")
        for t in self._tanks:
            if t.alive:
                self._set_cell(t.x, t.y, "T")
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "*")
        for eb in self._enemy_bullets:
            if eb.alive:
                self._set_cell(eb.x, eb.y, "o")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall", "│": "wall", "~": "bushes",
            "T": "enemy tank (50pts)", "*": "your bullet",
            "o": "enemy bullet", " ": "ground",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()

        def _st(name: str) -> str:
            return "ok" if self._sensors[name] else "DMG"

        sensors = (
            f"Sensors: radar={_st('radar')}"
            f" cannon={_st('cannon')}"
            f" treads={_st('treads')}"
            f" video={_st('video')}"
        )
        extra = f"{sensors}  Kills: {self._kills}"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Destroy enemy tanks while managing damage. "
            "Hits damage sensors: radar, cannon, treads, video. "
            "Lose all sensors = episode ends."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Robot Tank.\n\n"
            "TASK\n"
            "Pilot an armored tank against enemy tanks while "
            "managing damage across 4 subsystems (radar, cannon, "
            "treads, video). Destroy enemy tanks and survive as "
            "long as possible.\n\n"
            "BOARD\n"
            "20x20 field with wall border. Bushes 'tilde' scatter "
            "through the field as cover. Enemy tanks 'T' (HP = 2 "
            "each). Your bullets '*', enemy bullets 'o'. You start "
            "near one edge as an arrow glyph.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT / UP / DOWN set facing and move 1 cell "
            "(blocked by walls and edges). FIRE launches a bullet "
            "in facing direction (max 2 alive). Enemy tanks update "
            "every 3 steps: move toward you along longer axis, and "
            "fire every 10 steps within distance 10 along dominant "
            "axis. Each hit to you disables a random sensor: "
            "'radar' loss makes enemies less likely to track; "
            "'cannon' stops FIRE; 'treads' stops movement; 'video' "
            "is cosmetic.\n\n"
            "SCORING\n"
            "Pattern D: +1/12 reward per enemy tank destroyed "
            "(takes 2 bullets). -1 reward when all 4 sensors are "
            "knocked out. Cumulative reward bound: [-1, +1].\n\n"
            "TERMINATION\n"
            "Single-life: losing all 4 sensors ends the episode "
            "with reward -1. Destroying 12 tanks ends with "
            "cumulative +1. Episode also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, sensor statuses (ok / DMG), "
            "and kills.\n\n"
            + self.action_spec.render_for_prompt()
        )
