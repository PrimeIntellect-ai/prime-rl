"""Submodule-backed Craftax wrapper for GlyphBench evals.

The focused ``glyphbench/craftax-*`` scenarios use the local Python Craftax
port. This module is the registered full-game Craftax wrapper: it drives the
project's pinned public Craftax fork through its stable GlyphBench action and
Unicode rendering API.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation
from glyphbench.envs.craftax.base import craftax_prompt_contract


class _CraftaxRuntime(NamedTuple):
    jax: Any
    jnp: Any
    CraftaxSymbolicEnvNoAutoReset: Any
    EnvParams: Any
    Achievement: Any
    ascii_renderer: Any
    glyphbench_api: Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_craftax_importable() -> None:
    """Prefer the checked-out fork submodule over any installed package."""
    submodule_root = _repo_root() / "third_party" / "Craftax"
    submodule_pkg = submodule_root / "craftax"
    if submodule_pkg.exists():
        submodule_path = str(submodule_root)
        if submodule_path in sys.path:
            sys.path.remove(submodule_path)
        sys.path.insert(0, submodule_path)
        return
    if importlib.util.find_spec("craftax") is not None:
        return


def _load_glyphbench_api() -> Any:
    _ensure_craftax_importable()
    try:
        from craftax.craftax import glyphbench_api
    except ImportError as exc:
        raise ImportError(
            "The Craftax wrapper needs the project's pinned Craftax fork. "
            "Run `git submodule update --init third_party/Craftax`."
        ) from exc
    return glyphbench_api


def _load_runtime() -> _CraftaxRuntime:
    _ensure_craftax_importable()
    try:
        import jax
        import jax.numpy as jnp
        from craftax.craftax import ascii_renderer, glyphbench_api
        from craftax.craftax.constants import Achievement
        from craftax.craftax.craftax_state import EnvParams
        from craftax.craftax.envs.craftax_symbolic_env import (
            CraftaxSymbolicEnvNoAutoReset,
        )
    except ImportError as exc:
        raise ImportError(
            "The Craftax wrapper needs the optional Craftax/JAX dependencies. "
            "Run `uv sync --extra craftax` after initializing submodules."
        ) from exc
    return _CraftaxRuntime(
        jax=jax,
        jnp=jnp,
        CraftaxSymbolicEnvNoAutoReset=CraftaxSymbolicEnvNoAutoReset,
        EnvParams=EnvParams,
        Achievement=Achievement,
        ascii_renderer=ascii_renderer,
        glyphbench_api=glyphbench_api,
    )


CRAFTAX_GLYPHBENCH_API = _load_glyphbench_api()

CRAFTAX_FORK_ACTION_SPEC = ActionSpec(
    names=CRAFTAX_GLYPHBENCH_API.ACTION_NAMES,
    descriptions=CRAFTAX_GLYPHBENCH_API.ACTION_DESCRIPTIONS,
    extra_aliases=CRAFTAX_GLYPHBENCH_API.ACTION_ALIASES,
)

_FLOOR_NAMES = (
    "Overworld",
    "Dungeon",
    "Gnomish Mines",
    "Sewers",
    "Vaults",
    "Troll Mines",
    "Fire Realm",
    "Ice Realm",
    "Graveyard",
)
_FAST_FORWARD_MAX_TICKS = 10000


class CraftaxForkEnv(BaseGlyphEnv):
    """Full Craftax from the project's pinned fork, adapted to GlyphBench."""

    action_spec = CRAFTAX_FORK_ACTION_SPEC
    noop_action_name = "NOOP"
    # Open-ended full game: report the RAW unclamped cumulative Craftax reward
    # (weighted achievement reward + health term), not the [-1,1] benchmark
    # bound. (The craftax-* minigames keep the default clamp.)
    clamp_episode_return = False

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._runtime: _CraftaxRuntime | None = None
        self._env: Any | None = None
        self._params: Any | None = None
        self._state: Any | None = None
        self._jax_key: Any | None = None
        self._message = ""
        self._last_achievements: set[str] = set()
        self._upstream_action_values: tuple[int, ...] | None = None

    def env_id(self) -> str:
        return "glyphbench/craftaxfull-v0"

    def _ensure_runtime(self) -> _CraftaxRuntime:
        if self._runtime is None:
            runtime = _load_runtime()
            self._runtime = runtime
            self._env = runtime.CraftaxSymbolicEnvNoAutoReset()
            self._params = runtime.EnvParams(max_timesteps=int(self.max_turns))
            self._upstream_action_values = runtime.glyphbench_api.action_values()
        return self._runtime

    def _reset(self, seed: int) -> GridObservation:
        runtime = self._ensure_runtime()
        assert self._env is not None
        assert self._params is not None
        self._jax_key = runtime.jax.random.PRNGKey(int(seed))
        self._jax_key, reset_key = runtime.jax.random.split(self._jax_key)
        _obs, self._state = self._env.reset(reset_key, self._params)
        self._message = ""
        self._last_achievements = self._achievement_names()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        runtime = self._ensure_runtime()
        assert self._env is not None
        assert self._params is not None
        assert self._state is not None
        assert self._jax_key is not None
        assert self._upstream_action_values is not None

        spec_name = self.action_spec.names[action]
        upstream_name = runtime.glyphbench_api.UPSTREAM_ACTION_NAMES[action]
        upstream_value = self._upstream_action_values[action]
        before_achievements = self._achievement_names()

        reward_f, done_b, info = self._step_upstream(upstream_value)
        fast_forward_ticks = 0
        fast_forward_capped = False
        if not done_b and self._sleep_rest_active():
            extra_reward, done_b, fast_forward_ticks, fast_forward_capped = (
                self._fast_forward_sleep_rest()
            )
            reward_f += extra_reward

        after_achievements = self._achievement_names()
        unlocked = sorted(after_achievements - before_achievements)
        self._last_achievements = after_achievements
        fast_forward_note = (
            f" Fast-forwarded {fast_forward_ticks} sleep/rest tick"
            f"{'s' if fast_forward_ticks != 1 else ''}."
            if fast_forward_ticks
            else ""
        )
        if fast_forward_capped:
            fast_forward_note += " Fast-forward cap reached; state may still be asleep/resting."
        if unlocked:
            self._message = (
                f"Last action {spec_name} ({upstream_name}) yielded reward "
                f"{reward_f:+.3f}.{fast_forward_note} "
                f"New achievements: {', '.join(unlocked)}."
            )
        else:
            self._message = (
                f"Last action {spec_name} ({upstream_name}) yielded reward "
                f"{reward_f:+.3f}.{fast_forward_note}"
            )

        info_py = self._to_python_info(info)
        info_py.update(
            {
                "action_name": spec_name,
                "upstream_action_name": upstream_name,
                "upstream_action_value": upstream_value,
                "achievements_unlocked": unlocked,
                "num_achievements": len(after_achievements),
                "fast_forward_ticks": fast_forward_ticks,
                "fast_forward_capped": fast_forward_capped,
            }
        )
        return self._render_current_observation(), reward_f, done_b, False, info_py

    def _step_upstream(self, upstream_value: int) -> tuple[float, bool, Any]:
        runtime = self._ensure_runtime()
        assert self._env is not None
        assert self._params is not None
        assert self._state is not None
        assert self._jax_key is not None
        self._jax_key, step_key = runtime.jax.random.split(self._jax_key)
        _obs, self._state, reward, done, info = self._env.step(
            step_key, self._state, upstream_value, self._params
        )
        return float(np.asarray(reward)), bool(np.asarray(done)), info

    def _sleep_rest_active(self) -> bool:
        assert self._state is not None
        return bool(np.asarray(self._state.is_sleeping)) or bool(
            np.asarray(self._state.is_resting)
        )

    def _fast_forward_sleep_rest(self) -> tuple[float, bool, int, bool]:
        assert self._upstream_action_values is not None
        noop_value = self._upstream_action_values[self.action_spec.index_of("NOOP")]
        reward = 0.0
        done = False
        ticks = 0
        capped = False
        while self._sleep_rest_active():
            if ticks >= _FAST_FORWARD_MAX_TICKS:
                capped = True
                break
            step_reward, done, _info = self._step_upstream(noop_value)
            reward += step_reward
            ticks += 1
            if done:
                break
        return reward, done, ticks, capped

    def _render_current_observation(self) -> GridObservation:
        runtime = self._ensure_runtime()
        assert self._state is not None
        rendered = runtime.glyphbench_api.render_craftax_unicode_grid_local(self._state)
        hud = self._hud(np.asarray(rendered.status_data))
        return GridObservation(
            grid=rendered.grid,
            legend=rendered.legend,
            hud=hud,
            message=self._message,
        )

    def _hud(self, status_data: np.ndarray) -> str:
        renderer = self._runtime.ascii_renderer  # type: ignore[union-attr]
        sd = status_data
        floor_idx = int(sd[renderer.S_FLOOR])
        floor_name = (
            _FLOOR_NAMES[floor_idx]
            if 0 <= floor_idx < len(_FLOOR_NAMES)
            else f"Floor {floor_idx}"
        )
        pos_r = int(sd[renderer.S_POS_ROW])
        pos_c = int(sd[renderer.S_POS_COL])
        facing = self._direction_name(int(sd[renderer.S_DIRECTION]))
        hp = f"HP: {int(sd[renderer.S_HEALTH])}/{int(sd[renderer.S_MAX_HEALTH])}"
        food = f"Food: {int(sd[renderer.S_FOOD])}/{int(sd[renderer.S_MAX_FOOD])}"
        drink = f"Drink: {int(sd[renderer.S_DRINK])}/{int(sd[renderer.S_MAX_DRINK])}"
        energy = f"Energy: {int(sd[renderer.S_ENERGY])}/{int(sd[renderer.S_MAX_ENERGY])}"
        mana = f"Mana: {int(sd[renderer.S_MANA])}/{int(sd[renderer.S_MAX_MANA])}"
        light = self._light_name(float(sd[renderer.S_LIGHT_LEVEL]))
        status_flags = [
            f"Sleeping: {'yes' if sd[renderer.S_IS_SLEEPING] > 0.5 else 'no'}",
            f"Resting: {'yes' if sd[renderer.S_IS_RESTING] > 0.5 else 'no'}",
            f"Fireball: {'learned' if sd[renderer.S_LEARNED_FIRE] > 0.5 else 'no'}",
            f"Iceball: {'learned' if sd[renderer.S_LEARNED_ICE] > 0.5 else 'no'}",
        ]
        kills = int(sd[renderer.S_MONSTERS_KILLED])
        required = int(sd[renderer.S_MONSTERS_REQUIRED])
        ladder = "OPEN" if kills >= required else f"CLOSED {kills}/{required}"
        boss = "VULNERABLE" if sd[renderer.S_BOSS_VULNERABLE] > 0.5 else "no"

        inventory = " ".join(
            f"{label}:{int(sd[idx])}"
            for label, idx in (
                ("Wood", renderer.S_INV_WOOD),
                ("Stone", renderer.S_INV_STONE),
                ("Coal", renderer.S_INV_COAL),
                ("Iron", renderer.S_INV_IRON),
                ("Diamond", renderer.S_INV_DIAMOND),
                ("Sapphire", renderer.S_INV_SAPPHIRE),
                ("Ruby", renderer.S_INV_RUBY),
                ("Sapling", renderer.S_INV_SAPLING),
            )
        )
        potions = " ".join(
            f"{label}:{int(sd[renderer.S_POTION_START + i])}"
            for i, label in enumerate(("Red", "Green", "Blue", "Pink", "Cyan", "Yellow"))
        )
        equipment = self._equipment_line(sd, renderer)
        armor = self._armor_line(sd, renderer)
        return "\n".join(
            [
                f"Step: {self._turn} / {self.max_turns}    Upstream timestep: {int(sd[renderer.S_TIMESTEP])}",
                f"{hp}    {food}    {drink}    {energy}    {mana}",
                (
                    f"XP: {int(sd[renderer.S_XP])}    "
                    f"Str: {int(sd[renderer.S_STR])}    "
                    f"Dex: {int(sd[renderer.S_DEX])}    "
                    f"Int: {int(sd[renderer.S_INT])}"
                ),
                (
                    f"Floor: {floor_idx} ({floor_name})    "
                    f"Pos: ({pos_r},{pos_c})    Facing: {facing}    Light: {light}"
                ),
                "[Inventory] " + inventory,
                "[Equipment] " + equipment,
                "[Armor] " + armor,
                "[Potions] " + potions,
                (
                    "[Status] "
                    + " | ".join(status_flags)
                    + f" | Ladder: {ladder} | Boss: {boss} | "
                    f"Achievements: {len(self._last_achievements)}/{self._num_achievements()}"
                ),
            ]
        )

    def _equipment_line(self, sd: np.ndarray, renderer: Any) -> str:
        pickaxe = self._material_name(int(sd[renderer.S_INV_PICKAXE]))
        sword_lvl = int(sd[renderer.S_INV_SWORD])
        sword = (
            f"{self._material_name(sword_lvl)} Sword"
            if sword_lvl > 0
            else "No Sword"
        )
        sword_ench = int(sd[renderer.S_SWORD_ENCH])
        if sword_lvl > 0 and sword_ench > 0:
            sword += f"({self._enchantment_name(sword_ench)})"
        bow = "Bow" if int(sd[renderer.S_INV_BOW]) > 0 else "No Bow"
        bow_ench = int(sd[renderer.S_BOW_ENCH])
        if bow == "Bow" and bow_ench > 0:
            bow += f"({self._enchantment_name(bow_ench)})"
        return " | ".join(
            [
                f"{pickaxe} Pickaxe",
                sword,
                bow,
                f"Arrows:{int(sd[renderer.S_INV_ARROWS])}",
                f"Torches:{int(sd[renderer.S_INV_TORCHES])}",
                f"Books:{int(sd[renderer.S_INV_BOOKS])}",
            ]
        )

    def _armor_line(self, sd: np.ndarray, renderer: Any) -> str:
        parts: list[str] = []
        for idx, slot in enumerate(("Helm", "Chest", "Legs", "Boots")):
            level = int(sd[renderer.S_ARMOR_START + idx])
            enchant = int(sd[renderer.S_ARMOR_ENCH_START + idx])
            if level <= 0:
                continue
            part = f"{self._armor_name(level)} {slot}"
            if enchant > 0:
                part += f"({self._enchantment_name(enchant)})"
            parts.append(part)
        return " | ".join(parts) if parts else "None"

    def _achievement_names(self) -> set[str]:
        if self._state is None or self._runtime is None:
            return set()
        achievements = np.asarray(self._state.achievements)
        out: set[str] = set()
        for achievement in self._runtime.Achievement:
            if bool(achievements[int(achievement.value)]):
                out.add(str(achievement.name).lower())
        return out

    def _num_achievements(self) -> int:
        if self._runtime is None:
            return 0
        return len(tuple(self._runtime.Achievement))

    @staticmethod
    def _direction_name(direction: int) -> str:
        return {0: "none", 1: "left", 2: "right", 3: "up", 4: "down"}.get(
            int(direction), "none"
        )

    @staticmethod
    def _light_name(light_level: float) -> str:
        if light_level < 0.3:
            return "Night"
        if light_level < 0.6:
            return "Dusk"
        return "Day"

    @staticmethod
    def _material_name(level: int) -> str:
        return {0: "None", 1: "Wood", 2: "Stone", 3: "Iron", 4: "Diamond"}.get(
            int(level), "None"
        )

    @staticmethod
    def _armor_name(level: int) -> str:
        return {0: "None", 1: "Iron", 2: "Diamond"}.get(int(level), "None")

    @staticmethod
    def _enchantment_name(level: int) -> str:
        return {0: "None", 1: "Fire", 2: "Ice"}.get(int(level), "None")

    @staticmethod
    def _to_python_info(info: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in dict(info).items():
            arr = np.asarray(value)
            if arr.shape == ():
                out[key] = arr.item()
            else:
                out[key] = arr.tolist()
        return out

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}, the full Craftax roguelike from "
            "the project submodule.\n\n"
            "TASK\n"
            "Survive, explore, craft equipment, clear floors, and defeat the "
            "necromancer. Rewards are the upstream Craftax rewards: newly "
            "unlocked achievements give positive reward, and changes in health "
            "also affect reward. The episode terminates on death, on defeating "
            "the boss, or at the turn limit.\n\n"
            f"{craftax_prompt_contract()}\n\n"
            "OBSERVATION\n"
            "The grid comes from the fork's stable Unicode local renderer. "
            "The view is the 9x11 visible window centered on you; the "
            "HUD carries inventory, equipment, vitals, floor, absolute position, "
            "facing, spell knowledge, ladder progress, and achievements. "
            "Directional arrows mark your current facing. To interact with a "
            "tree, ore, mob, water, chest, fountain, or edible passive creature, "
            "face that cell and use DO. Absolute position is included because "
            "Craftax is long-horizon and navigation-relevant.\n\n"
            "CORE MECHANICS\n"
            "Food, drink, energy, mana, and HP are persistent resources. Food, "
            "drink, and energy decay over time; HP can recover when food, drink, "
            "and energy are positive. Mine trees for wood, place a table, craft "
            "pickaxes and swords, mine stone and ores, place a furnace, then "
            "progress to stronger equipment. Underground floors may need "
            "torches. Kill enough monsters on a floor to open its ladder, then "
            "stand on the ladder and DESCEND. Use ASCEND on upward ladders. "
            "Books teach fireball or iceball. Potions have color-shuffled "
            "effects per run.\n\n"
            "MOVEMENT\n"
            "MOVE_LEFT, MOVE_RIGHT, MOVE_UP, and MOVE_DOWN are absolute moves. "
            "If the target cell is blocked, the move still turns you to face "
            "that direction. DO always applies to the cell you face, not "
            "necessarily the cell you occupy.\n\n"
            + self.action_spec.render_for_prompt()
        )
