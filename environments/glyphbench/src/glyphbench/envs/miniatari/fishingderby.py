"""miniatari Fishing Derby.

Identity: Two anglers race to catch fish from a small lake.
Win condition: agent reels in 3 fish before opponent.
Reward: Pattern C, +1/3 per fish caught, -1/3 per opp fish.
Loss: opponent catches 3 fish first.

Gym ID: glyphbench/miniatari-fishingderby-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=59, mean_return=-0.522
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase, MiniatariEntity


class MiniFishingDerbyEnv(MiniatariBase):
    """Mini Fishing Derby: 14x10 lake, first-to-3 fish.

    Agent moves a fishing hook around the water. When the hook overlaps a
    fish, the fish is hooked. REEL pulls hook upward; reaching the surface
    lands the fish (+1/3). Opponent uses an analogous AI hook on the right
    side. Game over when either side reaches 3 caught fish.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "REEL"),
        descriptions=(
            "do nothing",
            "move hook up",
            "move hook down",
            "move hook left",
            "move hook right",
            "reel in (only meaningful with a hooked fish)",
        ),
    )

    default_max_turns = 200

    _WIDTH = 14
    _HEIGHT = 10
    _PIER_Y = 1
    _SURFACE_Y = 2
    _BOTTOM_Y = 8
    _MAX_FISH = 4
    _WIN_TARGET = 3

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._hook_x: int = 0
        self._hook_y: int = 0
        self._opp_hook_x: int = 0
        self._opp_hook_y: int = 0
        self._fish: list[MiniatariEntity] = []
        self._hooked_fish: MiniatariEntity | None = None
        self._opp_hooked: MiniatariEntity | None = None
        self._agent_score: int = 0
        self._opp_score: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-fishingderby-v0"

    def _spawn_fish(self) -> None:
        rng = self.rng
        fx = int(rng.integers(2, self._WIDTH - 2))
        fy = int(rng.integers(self._SURFACE_Y + 1, self._BOTTOM_Y))
        dx = 1 if rng.random() < 0.5 else -1
        fish = MiniatariEntity(etype="fish", char="F", x=fx, y=fy, dx=dx)
        fish.data["speed_timer"] = 0
        self._fish.append(fish)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._fish = []
        self._hooked_fish = None
        self._opp_hooked = None
        self._agent_score = 0
        self._opp_score = 0
        # Hooks start near surface
        self._hook_x = 3
        self._hook_y = self._SURFACE_Y + 1
        self._opp_hook_x = self._WIDTH - 4
        self._opp_hook_y = self._SURFACE_Y + 1
        # Player position is the pier
        self._player_x = 3
        self._player_y = self._PIER_Y
        self._player_dir = (0, 1)
        for _ in range(self._MAX_FISH):
            self._spawn_fish()

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Clean up dead hooked fish references
        if self._hooked_fish is not None and not self._hooked_fish.alive:
            self._hooked_fish = None
        if self._opp_hooked is not None and not self._opp_hooked.alive:
            self._opp_hooked = None

        # Agent move (only when no fish hooked)
        if self._hooked_fish is None:
            if action_name == "UP" and self._hook_y > self._SURFACE_Y + 1:
                self._hook_y -= 1
                self._player_dir = (0, -1)
            elif action_name == "DOWN" and self._hook_y < self._BOTTOM_Y - 1:
                self._hook_y += 1
                self._player_dir = (0, 1)
            elif action_name == "LEFT" and self._hook_x > 1:
                self._hook_x -= 1
                self._player_dir = (-1, 0)
            elif action_name == "RIGHT" and self._hook_x < self._WIDTH - 2:
                self._hook_x += 1
                self._player_dir = (1, 0)

            # Check if hook lands on a fish
            for fish in self._fish:
                if fish.alive and fish.x == self._hook_x and fish.y == self._hook_y:
                    self._hooked_fish = fish
                    self._message = "Fish on the hook!"
                    break

        # REEL - raise the hook with the fish
        if action_name == "REEL" and self._hooked_fish is not None:
            self._hook_y -= 1
            self._hooked_fish.x = self._hook_x
            self._hooked_fish.y = self._hook_y
            if self._hook_y <= self._SURFACE_Y:
                # Land the fish
                self._hooked_fish.alive = False
                if self._hooked_fish in self._fish:
                    self._fish.remove(self._hooked_fish)
                self._hooked_fish = None
                reward += self._agent_score_reward(self._WIN_TARGET)
                self._agent_score += 1
                self._message = "Caught a fish! +1/3"
                self._hook_y = self._SURFACE_Y + 1
                if self._agent_score >= self._WIN_TARGET:
                    self._on_won()
                else:
                    if len(self._fish) < self._MAX_FISH:
                        self._spawn_fish()

        # Move free fish
        for fish in self._fish:
            if not fish.alive or fish is self._hooked_fish or fish is self._opp_hooked:
                continue
            fish.data["speed_timer"] += 1
            if fish.data["speed_timer"] >= 2:
                fish.data["speed_timer"] = 0
                fish.x += fish.dx
                if fish.x <= 1 or fish.x >= self._WIDTH - 2:
                    fish.dx = -fish.dx
                    fish.x += fish.dx

        # Opponent AI - returns delta opp catches this tick
        if not self._game_over:
            opp_caught = self._opponent_ai()
            for _ in range(opp_caught):
                reward += self._opp_score_reward(self._WIN_TARGET)
                self._opp_score += 1
                if self._opp_score >= self._WIN_TARGET:
                    self._on_life_lost()
                    break

        info["agent_score"] = self._agent_score
        info["opp_score"] = self._opp_score
        return reward, self._game_over, info

    def _opponent_ai(self) -> int:
        """Returns number of fish opp landed this tick (0 or 1)."""
        rng = self.rng
        landed = 0
        if self._opp_hooked is None:
            # Find nearest fish
            best = None
            best_dist = 999
            for fish in self._fish:
                if not fish.alive or fish is self._hooked_fish:
                    continue
                d = abs(fish.x - self._opp_hook_x) + abs(fish.y - self._opp_hook_y)
                if d < best_dist:
                    best_dist = d
                    best = fish
            # Opponent moves slower than agent so a competent agent can win.
            if best is not None and rng.random() < 0.35:
                if self._opp_hook_x < best.x and self._opp_hook_x < self._WIDTH - 2:
                    self._opp_hook_x += 1
                elif self._opp_hook_x > best.x and self._opp_hook_x > 1:
                    self._opp_hook_x -= 1
                elif self._opp_hook_y < best.y and self._opp_hook_y < self._BOTTOM_Y - 1:
                    self._opp_hook_y += 1
                elif self._opp_hook_y > best.y and self._opp_hook_y > self._SURFACE_Y + 1:
                    self._opp_hook_y -= 1
            # Check catch
            for fish in self._fish:
                if (
                    fish.alive
                    and fish is not self._hooked_fish
                    and fish.x == self._opp_hook_x
                    and fish.y == self._opp_hook_y
                ):
                    self._opp_hooked = fish
                    break
        else:
            # Reel up at half speed (every other tick)
            if rng.random() < 0.5:
                self._opp_hook_y -= 1
            self._opp_hooked.x = self._opp_hook_x
            self._opp_hooked.y = self._opp_hook_y
            if self._opp_hook_y <= self._SURFACE_Y:
                self._opp_hooked.alive = False
                if self._opp_hooked in self._fish:
                    self._fish.remove(self._opp_hooked)
                self._opp_hooked = None
                self._message = "Opponent landed a fish."
                self._opp_hook_y = self._SURFACE_Y + 1
                landed = 1
                if len(self._fish) < self._MAX_FISH:
                    self._spawn_fish()
        return landed

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = []
        for y in range(self._HEIGHT):
            row: list[str] = []
            for _x in range(self._WIDTH):
                if y == self._PIER_Y:
                    row.append("=")
                elif y == self._SURFACE_Y:
                    row.append("~")
                elif y == self._BOTTOM_Y:
                    row.append("─")
                elif self._SURFACE_Y < y < self._BOTTOM_Y:
                    row.append("·")
                else:
                    row.append(" ")
            grid.append(row)

        # Fish
        for fish in self._fish:
            if fish.alive and 0 <= fish.x < self._WIDTH and 0 <= fish.y < self._HEIGHT:
                grid[fish.y][fish.x] = "F"

        # Lines
        for y in range(self._PIER_Y + 1, self._hook_y):
            if 0 <= self._hook_x < self._WIDTH:
                grid[y][self._hook_x] = ":"
        for y in range(self._PIER_Y + 1, self._opp_hook_y):
            if 0 <= self._opp_hook_x < self._WIDTH:
                grid[y][self._opp_hook_x] = ";"

        # Hooks
        if 0 <= self._hook_x < self._WIDTH and 0 <= self._hook_y < self._HEIGHT:
            grid[self._hook_y][self._hook_x] = "J"
        if 0 <= self._opp_hook_x < self._WIDTH and 0 <= self._opp_hook_y < self._HEIGHT:
            grid[self._opp_hook_y][self._opp_hook_x] = "j"

        # Anglers on pier
        if 0 <= self._player_x < self._WIDTH:
            grid[self._PIER_Y][self._player_x] = "Y"
        opp_pier_x = self._WIDTH - 4
        if 0 <= opp_pier_x < self._WIDTH:
            grid[self._PIER_Y][opp_pier_x] = "P"

        symbols = {
            "=": "pier", "~": "water surface", "─": "lake bottom",
            "·": "water", " ": "sky", "F": "fish",
            "J": "your hook", "j": "opponent hook",
            ":": "your line", ";": "opponent line",
            "Y": "you (angler)", "P": "opponent angler",
        }

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Caught: You {self._agent_score} - {self._opp_score} Opp    "
            f"First to {self._WIN_TARGET}\n"
            f"Your hook: ({self._hook_x},{self._hook_y})    "
            f"Opp hook: ({self._opp_hook_x},{self._opp_hook_y})    "
            f"Hooked: {'yes' if self._hooked_fish else 'no'}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Two anglers race to catch fish from a 14x10 lake. Your hook (J) "
            "and the opponent's hook (j) dangle into the water. Move your "
            "hook with UP/DOWN/LEFT/RIGHT to overlap a fish (F); the fish "
            "hooks itself automatically. Then REEL repeatedly to raise the "
            "hook to the surface (~) - that lands the fish. Fish drift "
            "horizontally each tick. First to 3 fish wins. Reward: +1/3 "
            "per fish caught, -1/3 per opponent fish."
        )
