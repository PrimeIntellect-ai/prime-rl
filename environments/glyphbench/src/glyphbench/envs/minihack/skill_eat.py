"""MiniHack Eat skill tasks.

Phase 3:
  * eat: randomise food position per seed (was fixed (3,3)).
  * eat-distract: A9 distractor-rich memory — at episode start a colour cue
    names which apple is real food; the others are POISON and take HP. The
    three apples render with distinct colour glyphs (Ⓡ/Ⓖ/Ⓑ) so each is
    individually identifiable from the legend; the cue is erased after the
    first action. Poison damage is fatal: -8 HP per poison eat, so a single
    wrong choice kills (or two poisons together leave the agent below 0 HP).
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MiniHackBase
from glyphbench.envs.minihack.items import FOOD_RATION, Item

# Color tags for the cue (single codepoints). These also serve as the per-
# variant FRUIT GLYPHS in the eat-distract env so the agent can read each
# apple variant individually from the legend.
CUE_COLORS: tuple[str, str, str] = ("Ⓡ", "Ⓖ", "Ⓑ")
CUE_NAMES_FULL: dict[str, str] = {
    "Ⓡ": "red",
    "Ⓖ": "green",
    "Ⓑ": "blue",
}

# Distract-variant per-colour fruit items. The real one (cued colour) is
# the Apple (food); the other two are poisonous (-8 HP on EAT — fatal at
# starting HP).
RED_APPLE = Item("red apple", "Ⓡ", "food", "red fruit")
GREEN_APPLE = Item("green apple", "Ⓖ", "food", "green fruit")
BLUE_APPLE = Item("blue apple", "Ⓑ", "food", "blue fruit")
RED_POISON = Item("red poison-fruit", "Ⓡ", "food", "red fruit")
GREEN_POISON = Item("green poison-fruit", "Ⓖ", "food", "green fruit")
BLUE_POISON = Item("blue poison-fruit", "Ⓑ", "food", "blue fruit")
APPLE_BY_COLOR: dict[str, Item] = {
    "Ⓡ": RED_APPLE,
    "Ⓖ": GREEN_APPLE,
    "Ⓑ": BLUE_APPLE,
}
POISON_BY_COLOR: dict[str, Item] = {
    "Ⓡ": RED_POISON,
    "Ⓖ": GREEN_POISON,
    "Ⓑ": BLUE_POISON,
}
APPLE_NAMES = {RED_APPLE.name, GREEN_APPLE.name, BLUE_APPLE.name}
POISON_NAMES = {RED_POISON.name, GREEN_POISON.name, BLUE_POISON.name}

# Backwards-compat aliases for tests that still reference these names.
POISON_BERRY = RED_POISON  # historical name; tests check membership in poisons
POISON_MUSHROOM = GREEN_POISON
POISON_DAMAGE_HP = 8  # fatal in 1.5 hits at 12 HP starting


class _EatBase(MiniHackBase):
    _distract: bool = False
    _ate_food: bool = False

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._cue_color: str = ""
        self._real_apple_pos: tuple[int, int] | None = None
        self._cue_visible: bool = False

    def _generate_level(self, seed: int) -> None:
        self._init_grid(7, 7)
        self._hunger = 20
        self._ate_food = False
        # Player at random non-corner cell
        interior = [(x, y) for y in range(1, 6) for x in range(1, 6)]
        idx = int(self.rng.integers(0, len(interior)))
        px, py = interior[idx]
        self._place_player(px, py)
        # Stairs at random non-player position
        candidates = [pos for pos in interior if pos != (px, py)]
        sx, sy = candidates[int(self.rng.integers(0, len(candidates)))]
        self._place_stairs(sx, sy)
        candidates = [c for c in candidates if c != (sx, sy)]

        if self._distract:
            # Place 3 distinctly-coloured fruit items (one Ⓡ, one Ⓖ, one Ⓑ).
            # Exactly one — the cued colour — is a real apple; the other two
            # are POISON-FRUIT (-8 HP per EAT, fatal at full HP). The CUE
            # in the HUD names the safe colour; once erased (after the first
            # action) the agent must rely on memory + the per-glyph legend.
            colours = list(CUE_COLORS)
            apple_color_idx = int(self.rng.integers(0, len(colours)))
            self._cue_color = colours[apple_color_idx]
            self._cue_visible = True
            # Place 3 items: shuffle 3 cells from candidates
            shuffled = [candidates[i] for i in self.rng.permutation(len(candidates))]
            colour_order = list(self.rng.permutation(len(colours)))
            self._tagged_apples: dict[tuple[int, int], str] = {}
            for cell, c_idx in zip(shuffled[:3], colour_order, strict=True):
                pos = cell
                colour = colours[c_idx]
                self._tagged_apples[pos] = colour
                if colour == self._cue_color:
                    self._place_item(*pos, APPLE_BY_COLOR[colour])
                    self._real_apple_pos = pos
                else:
                    self._place_item(*pos, POISON_BY_COLOR[colour])
        else:
            # Single food ration at random position
            food_pos = candidates[int(self.rng.integers(0, len(candidates)))]
            self._place_item(*food_pos, FOOD_RATION)

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        # After first action, erase the cue (visible only on initial obs).
        if self._distract:
            self._cue_visible = False

        name = self.action_spec.names[action]
        info: dict[str, Any] = {}
        obs, reward, terminated, truncated, base_info = super()._step(action)
        info.update(base_info)
        if name == "EAT" and "You eat" in self._message:
            self._ate_food = True
        if terminated and info.get("goal_reached") and not self._ate_food:
            terminated = False
            reward = 0.0
            info.pop("goal_reached", None)
            self._message = "You are still starving. Eat real food before descending."
            obs = self._render_current_observation()
        if not terminated:
            self._hunger -= 1
            if self._hunger <= 0:
                self._message = "You starve to death!"
                terminated = True
                # Additive death penalty so any reward earned this step
                # (and earlier in the episode) survives into the
                # cumulative MC return used by GRPO.
                reward += -1.0
                info["cause_of_death"] = "starvation"
            obs = self._render_current_observation()
        return obs, reward, terminated, truncated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        extra = f"Hunger: {self._hunger}"
        if self._distract and self._cue_visible:
            colour_name = CUE_NAMES_FULL[self._cue_color]
            extra += (
                f"    CUE: the safe apple is the {colour_name} "
                f"({self._cue_color}) one."
                "  (cue erased after your first action)"
            )
        elif self._distract:
            extra += "    [cue erased]"
        new_hud = obs.hud + "    " + extra
        return GridObservation(
            grid=obs.grid,
            legend=obs.legend,
            hud=new_hud,
            message=obs.message,
        )

    def _task_description(self) -> str:
        if self._distract:
            return (
                "You are starving. Three fruits — one red (Ⓡ), one green (Ⓖ), "
                "one blue (Ⓑ) — are on the floor. Exactly ONE is a safe apple; "
                "the other two are POISON-FRUIT (-8 HP per EAT, fatal at "
                "starting HP). At episode start a CUE in the HUD names the "
                "safe colour; the cue is erased after your first action, so "
                "memorise it now. The legend lists each fruit by its colour, "
                "so you can map cue → grid cell. PICKUP the safe fruit, EAT "
                "it to refill hunger, then reach the stairs (⇣). Reward: +1 "
                "stairs, -1 death."
            )
        return (
            "You are starving! Find food (%) on the floor, pick it up with PICKUP, "
            "then use EAT to consume it. Then reach the stairs (⇣). "
            "You lose 1 hunger per turn and die at 0. Reward: +1 stairs, -1 death."
        )


class _EatDistractBase(_EatBase):
    """Eat-distract: override EAT to apply fatal poison penalty."""

    _distract = True

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        # Intercept EAT for poison handling BEFORE base consumes it
        if name == "EAT":
            food = next(
                (i for i in self._inventory if i.item_type == "food"), None
            )
            if food and food.name in POISON_NAMES:
                self._inventory.remove(food)
                self._hunger = min(100, self._hunger + 50)
                self._player_hp = max(0, self._player_hp - POISON_DAMAGE_HP)
                self._message = (
                    f"You eat the {food.legend_name()}. Poison! (-{POISON_DAMAGE_HP} HP)"
                )
                self._cue_visible = False
                # Continue with hunger tick / monster turn semantics
                self._hunger -= 1
                if self._player_hp <= 0:
                    return self._render_current_observation(), -1.0, True, False, {
                        "cause_of_death": "poison"
                    }
                if self._hunger <= 0:
                    self._message += " You starve to death!"
                    return self._render_current_observation(), -1.0, True, False, {
                        "cause_of_death": "starvation"
                    }
                return self._render_current_observation(), 0.0, False, False, {}
            if food and food.name in APPLE_NAMES:
                self._inventory.remove(food)
                self._hunger = min(100, self._hunger + 50)
                self._ate_food = True
                self._message = f"You eat the {food.legend_name()}. Yum!"
                self._cue_visible = False
                self._hunger -= 1
                if self._hunger <= 0:
                    return self._render_current_observation(), -1.0, True, False, {
                        "cause_of_death": "starvation"
                    }
                return self._render_current_observation(), 0.0, False, False, {}

        # All other actions: defer to parent
        self._cue_visible = False  # erase after any action
        return super()._step(action)


class MiniHackEatEnv(_EatBase):
    """MiniHack Eat: pick up and eat food before starving."""

    def env_id(self) -> str:
        return "glyphbench/minihack-eat-v0"


class MiniHackEatDistractEnv(_EatDistractBase):
    """MiniHack Eat (Distract): A9 colour-cue memory + poison berries."""

    _distract = True

    def env_id(self) -> str:
        return "glyphbench/minihack-eat-distract-v0"
