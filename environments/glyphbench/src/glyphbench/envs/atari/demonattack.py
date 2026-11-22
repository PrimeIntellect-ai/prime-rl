"""Atari Demon Attack environment.

Aliens attack from above in waves.

Gym ID: glyphbench/atari-demonattack-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class DemonAttackEnv(AtariBase):
    """Demon Attack: destroy demons attacking from above.

    20x20 grid. Demons fly in patterns, split when hit at
    higher levels. Player at bottom.

    Actions: NOOP, LEFT, RIGHT, FIRE
    Pattern A: +1/_WIN_TARGET per demon hit (full-scope = 5 waves x 8
    demons = 40). No failure penalty.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "fire a bullet upward",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _PLAYER_Y = 18

    # Pattern A full-scope target: 40 demon hits.
    _WIN_TARGET: int = 40

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._demons: list[AtariEntity] = []
        self._bullets: list[AtariEntity] = []
        self._enemy_bullets: list[AtariEntity] = []
        self._step_counter: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-demonattack-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._demons = []
        self._bullets = []
        self._enemy_bullets = []
        self._step_counter = 0

        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")

        self._player_x = self._WIDTH // 2
        self._player_y = self._PLAYER_Y

        n_demons = min(3 + self._level, 8)
        for i in range(n_demons):
            x = 2 + (i * 2) % (self._WIDTH - 4)
            y = 2 + (i % 3) * 2
            d = self._add_entity("demon", "D", x, y)
            d.data["dir"] = 1 if i % 2 == 0 else -1
            d.data["tier"] = min(self._level // 3, 2)
            self._demons.append(d)

        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        if action_name == "LEFT" and self._player_x > 1:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif (
            action_name == "RIGHT"
            and self._player_x < self._WIDTH - 2
        ):
            self._player_x += 1
            self._player_dir = (1, 0)
        elif action_name == "FIRE" and len(self._bullets) < 1:
            b = self._add_entity(
                "bullet", "!", self._player_x,
                self._player_y - 1, dy=-1,
            )
            self._bullets.append(b)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.y += b.dy
            if b.y <= 0:
                b.alive = False

        # Bullet-demon collisions
        new_spawns: list[AtariEntity] = []
        for b in self._bullets:
            if not b.alive:
                continue
            for dem in self._demons:
                if (
                    dem.alive
                    and dem.x == b.x
                    and dem.y == b.y
                ):
                    b.alive = False
                    tier = dem.data.get("tier", 0)
                    if tier > 0:
                        dem.data["tier"] = tier - 1
                        self._on_point_scored(1)
                        if self._progress_count < self._WIN_TARGET:
                            reward += 1.0 / self._WIN_TARGET
                            self._progress_count += 1
                        self._message = "Demon split!"
                        # Spawn a split
                        sx = dem.x + 1
                        if sx >= self._WIDTH - 1:
                            sx = dem.x - 1
                        s = self._add_entity(
                            "demon", "d", sx, dem.y,
                        )
                        s.data["dir"] = -dem.data["dir"]
                        s.data["tier"] = tier - 1
                        new_spawns.append(s)
                        dem.char = "d"
                    else:
                        dem.alive = False
                        self._on_point_scored(2)
                        if self._progress_count < self._WIN_TARGET:
                            reward += 1.0 / self._WIN_TARGET
                            self._progress_count += 1
                        self._message = "Demon destroyed!"
                    break
        self._demons.extend(new_spawns)
        self._bullets = [b for b in self._bullets if b.alive]

        # Move demons
        if self._step_counter % 2 == 0:
            for dem in self._demons:
                if not dem.alive:
                    continue
                d = dem.data.get("dir", 1)
                dem.x += d
                if dem.x <= 1 or dem.x >= self._WIDTH - 2:
                    dem.data["dir"] = -d
                    dem.y += 1

        # Enemy fire
        if self._step_counter % 6 == 0:
            alive = [d for d in self._demons if d.alive]
            if alive and len(self._enemy_bullets) < 3:
                s = alive[int(self.rng.integers(len(alive)))]
                eb = self._add_entity(
                    "enemy_bullet", "↓", s.x, s.y + 1, dy=1,
                )
                self._enemy_bullets.append(eb)

        # Move enemy bullets
        for eb in self._enemy_bullets:
            if not eb.alive:
                continue
            eb.y += eb.dy
            if eb.y >= self._HEIGHT - 1:
                eb.alive = False
            elif (
                eb.x == self._player_x
                and eb.y == self._player_y
            ):
                eb.alive = False
                self._on_life_lost()
                self._message = "Hit! Lost a life."
                self._player_x = self._WIDTH // 2
        self._enemy_bullets = [
            eb for eb in self._enemy_bullets if eb.alive
        ]

        # Demon reaches player
        for dem in self._demons:
            if dem.alive and dem.y >= self._PLAYER_Y:
                dem.alive = False
                self._on_life_lost()
                self._message = "Demon reached you!"
        self._demons = [d for d in self._demons if d.alive]

        # Win check
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "All demons destroyed!"

        # Level clear
        if not self._demons and not self._game_over:
            self._level += 1
            self._message = "Wave cleared!"
            self._generate_level(self._level)

        self._redraw()
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        for dem in self._demons:
            if dem.alive:
                self._set_cell(dem.x, dem.y, dem.char)
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "!")
        for eb in self._enemy_bullets:
            if eb.alive:
                self._set_cell(eb.x, eb.y, "↓")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall", "│": "wall",
            "D": "demon", "d": "small demon",
            "!": "your bullet", "↓": "enemy bullet",
            " ": "empty",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        alive = sum(1 for d in self._demons if d.alive)
        extra = (
            f"Demons: {alive}  Wave: {self._level}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Destroy demons attacking from above. "
            "Higher-tier demons split when hit. "
            "Clear all demons to advance."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Demon Attack.\n\n"
            "TASK\n"
            "Destroy every demon that descends from the top of the "
            "screen. On higher levels a demon may split into a small "
            "demon when hit. Clear all demons to advance.\n\n"
            "BOARD\n"
            "20x20 arena with walls ('-', '|'). You move horizontally "
            "along the bottom. Demons 'D' begin near the top of the "
            "field; split demons are 'd'. Your bullets are '!', "
            "enemy bullets are down-arrows.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT move you 1 cell along row 18. FIRE launches "
            "an upward bullet from your row (only 1 alive at a time). "
            "Demons move every 2 steps, bouncing off side walls and "
            "stepping down one row on each bounce. Enemies fire "
            "downward every 6 steps from a random alive demon (max 3 "
            "enemy bullets). A demon with tier > 0 survives the first "
            "hit but drops to a smaller 'd' and spawns one offspring.\n\n"
            "SCORING\n"
            "+1 progress unit per kill (full demon destroyed or "
            "split child destroyed), regardless of type. No "
            "per-step penalty.\n\n"
            "TERMINATION\n"
            "Single-life: being hit by an enemy bullet, or a "
            "demon reaching your row, ends the episode. The "
            "episode also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, wave/level, and demons alive.\n\n"
            + self.action_spec.render_for_prompt()
        )
