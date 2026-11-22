"""Atari Fishing Derby environment.

Two players fishing. Catch fish to score.

Gym ID: glyphbench/atari-fishingderby-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class FishingDerbyEnv(AtariBase):
    """Fishing Derby: catch fish from a lake.

    20x16 grid. Move hook to catch fish, reel them in.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, REEL
    Pattern C: ±1/_WIN_TARGET per fish. First to _WIN_TARGET catches
    wins (full-scope = 30 fish).
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "REEL"),
        descriptions=(
            "do nothing", "move hook up", "move hook down",
            "move hook left", "move hook right", "reel in the line",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 16
    _WATER_TOP = 4
    _WATER_BOT = 14
    _PIER_Y = 3
    _MAX_FISH = 6
    _GAME_TIME = 500

    # Pattern C full-scope target: first to 30 fish caught.
    _WIN_TARGET: int = 30

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._hook_x = self._hook_y = 0
        self._opp_hook_x = self._opp_hook_y = 0
        self._fish: list[AtariEntity] = []
        self._hooked_fish: AtariEntity | None = None
        self._opp_hooked: AtariEntity | None = None
        self._opp_score = self._timer = 0
        self._agent_progress: int = 0
        self._opp_progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-fishingderby-v0"

    def _reset(self, seed: int):
        self._agent_progress = 0
        self._opp_progress = 0
        return super()._reset(seed)

    def _spawn_fish(self) -> None:
        rng = self.rng
        fx = int(rng.integers(2, self._WIDTH - 2))
        fy = int(rng.integers(self._WATER_TOP + 1, self._WATER_BOT))
        dx = 1 if rng.random() < 0.5 else -1
        fish = self._add_entity("fish", "F", fx, fy, dx=dx)
        fish.data["speed_timer"] = 0
        self._fish.append(fish)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._fish = []
        self._hooked_fish = self._opp_hooked = None
        self._opp_score = self._timer = 0
        self._lives = 1
        self._player_x, self._player_y = 3, self._PIER_Y
        self._hook_x, self._hook_y = 5, self._WATER_TOP + 2
        self._opp_hook_x, self._opp_hook_y = 14, self._WATER_TOP + 2
        for _i in range(self._MAX_FISH):
            self._spawn_fish()
        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._timer += 1
        # Clear stale hooks
        if self._hooked_fish and not self._hooked_fish.alive:
            self._hooked_fish = None
        if self._opp_hooked and not self._opp_hooked.alive:
            self._opp_hooked = None
        # Move hook
        if self._hooked_fish is None:
            if action_name == "UP" and self._hook_y > self._WATER_TOP:
                self._hook_y -= 1
                self._player_dir = (0, -1)
            elif action_name == "DOWN" and self._hook_y < self._WATER_BOT - 1:
                self._hook_y += 1
                self._player_dir = (0, 1)
            elif action_name == "LEFT" and self._hook_x > 1:
                self._hook_x -= 1
                self._player_dir = (-1, 0)
            elif action_name == "RIGHT" and self._hook_x < self._WIDTH - 2:
                self._hook_x += 1
                self._player_dir = (1, 0)
        # Check if hook catches a fish
        if self._hooked_fish is None:
            for fish in self._fish:
                if fish.alive and fish.x == self._hook_x and fish.y == self._hook_y:
                    self._hooked_fish = fish
                    self._message = "Fish on the hook!"
                    break
        # Reel in
        if action_name == "REEL" and self._hooked_fish:
            self._hook_y -= 1
            self._hooked_fish.x = self._hook_x
            self._hooked_fish.y = self._hook_y
            if self._hook_y <= self._WATER_TOP:
                self._hooked_fish.alive = False
                if self._hooked_fish in self._fish:
                    self._fish.remove(self._hooked_fish)
                self._hooked_fish = None
                self._on_point_scored(1)
                if self._agent_progress < self._WIN_TARGET:
                    reward += 1.0 / self._WIN_TARGET
                    self._agent_progress += 1
                self._message = "Caught a fish!"
                self._hook_y = self._WATER_TOP + 2
                if len(self._fish) < self._MAX_FISH:
                    self._spawn_fish()
        # Move fish
        for fish in self._fish:
            if not fish.alive or fish is self._hooked_fish:
                continue
            fish.data["speed_timer"] += 1
            if fish.data["speed_timer"] >= 3:
                fish.data["speed_timer"] = 0
                fish.x += fish.dx
                if fish.x <= 1 or fish.x >= self._WIDTH - 2:
                    fish.dx = -fish.dx
                    fish.x += fish.dx
        opp_caught = self._opponent_ai()
        if opp_caught and self._opp_progress < self._WIN_TARGET:
            reward -= 1.0 / self._WIN_TARGET
            self._opp_progress += 1
        # First-to-N termination
        terminated = self._timer >= self._GAME_TIME
        if self._agent_progress >= self._WIN_TARGET:
            terminated = True
            self._message = "You win!"
            info["won"] = True
        elif self._opp_progress >= self._WIN_TARGET:
            terminated = True
            self._message = "Opponent wins!"
        elif terminated:
            if self._score > self._opp_score:
                self._message = "You win!"
            elif self._opp_score > self._score:
                self._message = "Opponent wins!"
            else:
                self._message = "Tie!"
        info["opp_score"] = self._opp_score
        info["time_left"] = self._GAME_TIME - self._timer
        self._redraw()
        return reward, terminated, info

    def _opponent_ai(self) -> bool:
        """Run opponent step. Return True if opponent landed a fish this step."""
        rng = self.rng
        if self._opp_hooked is None:
            best, best_dist = None, 999
            for fish in self._fish:
                if not fish.alive or fish is self._hooked_fish:
                    continue
                d = abs(fish.x - self._opp_hook_x) + abs(fish.y - self._opp_hook_y)
                if d < best_dist:
                    best_dist, best = d, fish
            if best and rng.random() < 0.5:
                if self._opp_hook_x < best.x:
                    self._opp_hook_x += 1
                elif self._opp_hook_x > best.x:
                    self._opp_hook_x -= 1
                if self._opp_hook_y < best.y:
                    self._opp_hook_y += 1
                elif self._opp_hook_y > best.y:
                    self._opp_hook_y -= 1
            for fish in self._fish:
                if not fish.alive or fish is self._hooked_fish:
                    continue
                if fish.x == self._opp_hook_x and fish.y == self._opp_hook_y:
                    self._opp_hooked = fish
                    break
        else:
            self._opp_hook_y -= 1
            self._opp_hooked.x = self._opp_hook_x
            self._opp_hooked.y = self._opp_hook_y
            if self._opp_hook_y <= self._WATER_TOP:
                self._opp_hooked.alive = False
                if self._opp_hooked in self._fish:
                    self._fish.remove(self._opp_hooked)
                self._opp_hooked = None
                self._opp_score += 1
                self._opp_hook_y = self._WATER_TOP + 2
                if len(self._fish) < self._MAX_FISH:
                    self._spawn_fish()
                return True
        return False

    def _redraw(self) -> None:
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")
        for x in range(self._WIDTH):
            self._set_cell(x, self._PIER_Y, "=")
            self._set_cell(x, self._WATER_TOP, "~")
            self._set_cell(x, self._WATER_BOT, "─")
        for y in range(self._WATER_TOP + 1, self._WATER_BOT):
            for x in range(self._WIDTH):
                self._set_cell(x, y, "·")
        self._set_cell(self._hook_x, self._hook_y, "J")
        for y in range(self._PIER_Y + 1, self._hook_y):
            self._set_cell(self._hook_x, y, ":")
        self._set_cell(self._opp_hook_x, self._opp_hook_y, "j")
        self._set_cell(16, self._PIER_Y - 1, "V")
        for fish in self._fish:
            if fish.alive:
                self._set_cell(fish.x, fish.y, "F")

    def _advance_entities(self) -> None:
        pass

    def _render_current_observation(self, **kw: Any):  # type: ignore[override]
        """Append fishing-derby-specific HUD on top of base HUD.

        Base HUD provides ``Step: T / N`` (X4 cross-cutting).
        """
        from glyphbench.core.observation import GridObservation

        obs = super()._render_current_observation()
        left = self._GAME_TIME - self._timer
        extra = f"You: {self._score}  Opp: {self._opp_score}  Time: {left}"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "=": "pier", "~": "water surface", "·": "water",
            "─": "lake bottom", "J": "your hook", "j": "opponent hook",
            ":": "fishing line", "F": "fish", "V": "opponent", " ": "sky",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Fishing derby! Move your hook with "
            "UP/DOWN/LEFT/RIGHT to reach a fish, "
            "then REEL to pull it up. "
            "Catch more fish than your opponent."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Fishing Derby.\n\n"
            "TASK\n"
            "Compete with an AI angler to catch more fish from a lake "
            "in 500 ticks. You and the opponent fish from opposite "
            "piers.\n\n"
            "BOARD\n"
            "20x16 scene. Pier '=' at row 3. Water surface '~' at row "
            "4, water fills rows 5-13 with '.' background, lake bottom "
            "'-' at row 14. Hooks ('J' yours, 'j' theirs) dangle into "
            "the water. Fishing line ':' connects pier to hook. Fish "
            "'F' swim in water. Opponent glyph 'V' sits on the pier. "
            "You control the hook's position.\n\n"
            "MECHANICS\n"
            "When no fish is on your hook: UP/DOWN/LEFT/RIGHT move "
            "the hook 1 cell (constrained to the water). If the hook "
            "shares a cell with a fish, that fish is hooked. When "
            "hooked, movement is frozen; only REEL does anything: "
            "REEL raises the hook by 1 row; when the hook reaches the "
            "water surface, the fish is landed. Fish drift "
            "horizontally every 3 steps and bounce off walls. Up to "
            "6 fish on the lake; a new one spawns after a catch.\n\n"
            "SCORING\n"
            "+1/30 reward per fish you reel out (Pattern C). "
            "-1/30 reward per fish the opponent reels out. "
            "Bound: cumulative reward stays in [-1, +1].\n\n"
            "TERMINATION\n"
            "Episode ends when either side reaches 30 catches "
            "(first-to-30) or after 500 ticks. Final message "
            "reports win / loss / tie.\n\n"
            "HUD\n"
            "Shows your score, opponent score, and time remaining.\n\n"
            + self.action_spec.render_for_prompt()
        )
