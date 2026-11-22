"""Atari Beam Rider environment.

Enemies on beams; player moves between beam positions and shoots.

Gym ID: glyphbench/atari-beamrider-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class BeamRiderEnv(AtariBase):
    """Beam Rider: shoot enemies descending on beams.

    20x20 grid. Player at bottom moves between beam positions.
    Enemies descend on beams; shoot them or dodge.

    Actions: NOOP, LEFT, RIGHT, FIRE
    Pattern A: +1/_WIN_TARGET per enemy killed (full-scope = 50 enemies
    across sectors). No failure penalty.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move left one beam",
            "move right one beam",
            "fire a torpedo up the beam",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _PLAYER_Y = 18
    _NUM_BEAMS = 9
    _MAX_BULLETS = 2
    _SECTOR_TARGET = 12

    # Pattern A full-scope target: 50 enemies destroyed; multi-sector
    # loop stops accruing reward at the cap. Audit BROKEN fix: raised
    # from 15 because a random spam-FIRE policy reliably scored ~5
    # enemies per ~113 steps, giving cumulative +0.32. Phase 3.5
    # bumped from 30 -> 50 because seed=1 random still occasionally
    # hit +1.0 within ~750 steps; 50 keeps random <= 0.8 across all
    # probed seeds.
    _WIN_TARGET: int = 50

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._bullets: list[AtariEntity] = []
        self._enemies: list[AtariEntity] = []
        self._beam_positions: list[int] = []
        self._player_beam: int = 4
        self._step_counter: int = 0
        self._kills: int = 0
        self._spawn_cd: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-beamrider-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._bullets = []
        self._enemies = []
        self._step_counter = 0
        self._kills = 0
        self._spawn_cd = 0

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")

        # Beam positions (evenly spaced)
        self._beam_positions = []
        spacing = (self._WIDTH - 2) // (self._NUM_BEAMS + 1)
        for i in range(self._NUM_BEAMS):
            self._beam_positions.append(1 + spacing * (i + 1))

        # Player starts at center beam
        self._player_beam = self._NUM_BEAMS // 2
        self._player_x = self._beam_positions[self._player_beam]
        self._player_y = self._PLAYER_Y

        # Spawn initial enemies
        for _ in range(3):
            self._spawn_enemy()

        self._redraw()

    def _spawn_enemy(self) -> None:
        rng = self.rng
        beam = int(rng.integers(0, self._NUM_BEAMS))
        bx = self._beam_positions[beam]
        etype = int(rng.integers(0, 3))
        chars = ("V", "W", "X")
        e = self._add_entity("enemy", chars[etype], bx, 1)
        e.data["beam"] = beam
        e.data["timer"] = 0
        e.data["speed"] = max(2, 4 - self._level // 2)
        self._enemies.append(e)

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Move player between beams
        if action_name == "LEFT" and self._player_beam > 0:
            self._player_beam -= 1
            self._player_x = self._beam_positions[self._player_beam]
            self._player_dir = (-1, 0)
        elif (
            action_name == "RIGHT"
            and self._player_beam < self._NUM_BEAMS - 1
        ):
            self._player_beam += 1
            self._player_x = self._beam_positions[self._player_beam]
            self._player_dir = (1, 0)

        # Fire
        if action_name == "FIRE" and len(self._bullets) < self._MAX_BULLETS:
            b = self._add_entity(
                "bullet", "!", self._player_x,
                self._player_y - 1, dy=-1
            )
            self._bullets.append(b)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.y += b.dy
            if b.y <= 0:
                b.alive = False
                continue
            for e in self._enemies:
                if e.alive and e.x == b.x and e.y == b.y:
                    e.alive = False
                    b.alive = False
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._kills += 1
                    self._message = "Enemy hit!"
                    break
        self._bullets = [b for b in self._bullets if b.alive]

        # Move enemies down beams
        for e in self._enemies:
            if not e.alive:
                continue
            e.data["timer"] += 1
            spd = e.data.get("speed", 3)
            if e.data["timer"] % spd == 0:
                e.y += 1
                if e.y >= self._PLAYER_Y:
                    # Check collision with player
                    if e.x == self._player_x:
                        e.alive = False
                        self._on_life_lost()
                        self._message = "Hit! Lost a life."
                        self._player_beam = self._NUM_BEAMS // 2
                        self._player_x = self._beam_positions[
                            self._player_beam
                        ]
                    elif e.y >= self._HEIGHT - 1:
                        e.alive = False

        # Spawn enemies
        self._spawn_cd -= 1
        alive_count = sum(1 for e in self._enemies if e.alive)
        if self._spawn_cd <= 0 and alive_count < 6:
            self._spawn_enemy()
            self._spawn_cd = max(4, 10 - self._level)

        # Win check
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "Sector target reached!"

        # Sector clear (advance level for HUD; no reward bonus)
        if self._kills >= self._SECTOR_TARGET and not self._game_over:
            self._message = "Sector cleared!"
            self._level += 1
            self._generate_level(self._level)

        self._enemies = [e for e in self._enemies if e.alive]
        self._redraw()
        info["kills"] = self._kills
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")

        # Draw beams
        for bx in self._beam_positions:
            for y in range(1, self._HEIGHT - 1):
                self._set_cell(bx, y, ":")

        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, e.char)
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "!")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall",
            "│": "wall",
            ":": "beam",
            "V": "enemy (type 1)",
            "W": "enemy (type 2)",
            "X": "enemy (type 3)",
            "!": "your torpedo",
            " ": "empty",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        extra = (
            f"Kills: {self._kills}/12  "
            f"Sector: {self._level}  "
            f"Beam: {self._player_beam}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Move between beams with LEFT/RIGHT. FIRE torpedoes "
            "to destroy enemies descending on beams. "
            "Clear 12 enemies to complete the sector."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Beam Rider.\n\n"
            "TASK\n"
            "Pilot a ship at the bottom of 9 vertical beams and shoot "
            "descending enemies. Clear 12 enemies to finish the sector.\n\n"
            "BOARD\n"
            "20x20 field with wall borders. Nine evenly spaced vertical "
            "beams drawn as ':' run from row 1 to row 18. Your ship is "
            "an arrow at the bottom of one beam column. "
            "Enemies are 'V', 'W', or 'X' (type 1/2/3). Your torpedoes "
            "are '!'.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT teleport you one beam over (you cannot be "
            "between beams). FIRE launches a torpedo straight up from "
            "your beam; at most 2 torpedoes alive at once. Enemies spawn "
            "at row 1 on a random beam and descend 1 row every max(2, "
            "4 - level/2) steps. The spawn cooldown is max(4, 10 - "
            "level). Up to 6 enemies alive at once.\n\n"
            "SCORING\n"
            "+1 reward for each enemy shot. +3 reward bonus when you "
            "reach 12 kills (sector clear), which also advances to the "
            "next sector and resets the kill counter.\n\n"
            "TERMINATION\n"
            "Single-life: an enemy that reaches row 18 on your "
            "beam kills you and ends the episode. The episode "
            "also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, sector, kills toward 50, and current "
            "beam index.\n\n"
            + self.action_spec.render_for_prompt()
        )
