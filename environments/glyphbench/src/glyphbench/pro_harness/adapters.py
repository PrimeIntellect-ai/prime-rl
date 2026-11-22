"""Per-game adapters for the Pro harness.

Each adapter teaches the (otherwise game-agnostic) harness:
  * **Observation** — optionally override what the model sees (Craftax renders
    the fork's original LOCAL view beautified to glyphbench's Unicode glyphs —
    a strict 1:1 remap of the ascii char-ids — keeping the original renderer's
    rich HUD for information parity with the validated original harness).
  * **Area identity** — the current floor / dungeon level (Craftax reads it
    from the fork's JAX state; NetHack parses Dlvl from the HUD).
  * **Per-area focus** — the current-floor PRIORITY guide surfaced each turn.
  * **Action synonyms, landmark guidance, initial scratchpad, replay frames.**

Adapter area/observation/focus methods receive the live ``game`` so they can
use authoritative env state (e.g. the Craftax floor) instead of only the text.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


class EnvAdapter(ABC):
    """Game-specific knowledge the harness layers on top of the env."""

    #: Human-readable noun for an "area" (used in prompt wording).
    area_noun: str = "area"

    @abstractmethod
    def area_key(self, game: object, obs_text: str) -> str:
        """Stable identifier for the current area (landmark DB key)."""

    @abstractmethod
    def area_label(self, game: object, obs_text: str) -> str:
        """Human-readable label for the current area."""

    @abstractmethod
    def focus(self, game: object, obs_text: str) -> str:
        """Current-area strategic focus block (may be empty)."""

    def observation(self, game: object, default_obs: str) -> str:
        """What to show the model as the observation. Default: the env's own
        rendered observation. Override to substitute a different renderer."""
        return default_obs

    def native_text_observation(self, game: object, default_obs: str) -> str:
        """Environment-native language observation, when one is available.

        The default preserves compatibility for environments without a
        separate native language renderer.
        """
        return default_obs

    def synonyms(self) -> dict[str, str]:
        """Lowercase phrase -> canonical action name (parser fallback)."""
        return {}

    def landmark_vocabulary(self) -> str:
        """Guidance on which landmarks are worth recording."""
        return (
            "Record only durable, useful locations (stairs, water/food sources, "
            "shops/altars/fountains, crafting stations you placed, rich resource "
            "spots, dangerous zones). Do not record common terrain or moving "
            "creatures."
        )

    def initial_scratchpad(self) -> dict[str, str]:
        return {}

    def render_frame(self, game: object):
        """Return an HxWx3 uint8 frame for the current state, or None if this
        game cannot produce a pixel frame. Used for the replay GIF."""
        return None

    def raw_score(self, info: dict) -> float | None:
        """The env's RAW, un-normalized score from a step ``info`` dict, or
        None if this env has no scalar score."""
        return None

    def raw_score_label(self) -> str:
        return "score"

    def base_system_prompt(self, game: object) -> str | None:
        """Optional override of the env's system prompt. Return None to use
        ``game.system_prompt()``; return a string to replace it."""
        return None


# ---------------------------------------------------------------------------
# Craftax  (glyphbench/craftaxfull-v0 is the JAX fork = real upstream Craftax)
# ---------------------------------------------------------------------------
_CRAFTAX_FLOOR_RE = re.compile(r"Floor:\s*(\d+)\s*\(([^)]*)\)", re.IGNORECASE)


class CraftaxAdapter(EnvAdapter):
    area_noun = "floor"

    _FLOOR_NAMES = (
        "Overworld", "Dungeon", "Gnomish Mines", "Sewers", "Vaults",
        "Troll Mines", "Fire Realm", "Ice Realm", "Graveyard",
    )

    # Original-convention names emitted by the model resolve through the fork's
    # ActionSpec aliases (LEFT->MOVE_LEFT, etc.); DO/PLACE_*/MAKE_*/DESCEND are
    # identical. The fork uses DO for ALL interactions (mine/chop/attack/open/
    # drink/eat) — no separate DRINK_WATER/EAT_PLANT.
    _SYNONYMS = {
        "left": "LEFT", "go left": "LEFT", "west": "LEFT",
        "right": "RIGHT", "go right": "RIGHT", "east": "RIGHT",
        "up": "UP", "go up": "UP", "north": "UP",
        "down": "DOWN", "go down": "DOWN", "south": "DOWN",
        "interact": "DO", "use": "DO", "mine": "DO", "attack": "DO",
        "chop": "DO", "hit": "DO", "dig": "DO", "open": "DO",
        "drink": "DO", "drink water": "DO", "eat": "DO", "eat plant": "DO",
        "descend": "DESCEND", "go downstairs": "DESCEND", "down stairs": "DESCEND",
        "ascend": "ASCEND", "go upstairs": "ASCEND", "up stairs": "ASCEND",
        "shoot": "SHOOT_ARROW", "shoot arrow": "SHOOT_ARROW",
        "fireball": "CAST_FIREBALL", "cast fireball": "CAST_FIREBALL",
        "iceball": "CAST_ICEBALL", "cast iceball": "CAST_ICEBALL",
        "read": "READ_BOOK", "read book": "READ_BOOK",
        "rest": "REST", "sleep": "SLEEP", "wait": "NOOP", "nothing": "NOOP",
    }

    def _floor(self, game: object, obs_text: str) -> tuple[int, str]:
        # Authoritative: read the floor from the fork's JAX state.
        state = getattr(game, "_state", None)
        if state is not None:
            try:
                import numpy as np

                n = int(np.asarray(state.player_level))
                name = self._FLOOR_NAMES[n] if 0 <= n < len(self._FLOOR_NAMES) else f"Floor {n}"
                return n, name
            except Exception:
                pass
        m = _CRAFTAX_FLOOR_RE.search(obs_text or "")
        if m:
            return int(m.group(1)), m.group(2).strip()
        return 0, "Overworld"

    def observation(self, game: object, default_obs: str) -> str:
        """Render the fork's original LOCAL view, beautified to glyphbench's
        Unicode glyph set.

        This is a strict 1:1 beautification of the original ascii renderer:
        ``UNICODE_CHAR_TABLE`` is indexed by the SAME char-ids as the ascii
        ``CHAR_TABLE`` (same 9x11 window, same semantics), so only the glyphs
        change. The original renderer's rich HUD (vitals, inventory, equipment,
        armour, potions, status flags) is reused verbatim — it contains no map
        glyphs — preserving full information parity with the validated original
        harness. The legend is the compact present-glyphs Unicode legend.

        Falls back to the env's default render if the fork renderer is
        unavailable (e.g. before reset)."""
        state = getattr(game, "_state", None)
        runtime = getattr(game, "_runtime", None)
        if state is None or runtime is None:
            return default_obs
        try:
            ar = runtime.ascii_renderer
            api = runtime.glyphbench_api
            char_grid, color_grid, status_data = ar.render_craftax_ascii_grid_local(
                state
            )
            grid = api.unicode_grid_to_string(char_grid)
            legend = api.unicode_legend_from_char_grid(char_grid)
            # Reuse the original ascii HUD verbatim (no map glyphs to remap).
            full = ar.ascii_grid_to_string(
                char_grid, color_grid, status_data,
                include_legend=False, use_color=False,
            )
            lines = full.split("\n")
            sep = lines.index("")  # the grid ends at the first blank line
            hud = "\n".join(lines[sep + 1:]).rstrip()
            return f"{grid}\n\n{hud}\n\n--- Legend ---\n{legend}"
        except Exception:
            return default_obs

    def native_text_observation(self, game: object, default_obs: str) -> str:
        """Return upstream Craftax's coordinate-by-coordinate text rendering.

        This deliberately uses ``render_craftax_text`` unchanged so modality
        experiments measure the renderer supplied by Craftax rather than a
        GlyphBench-authored natural-language conversion.
        """
        state = getattr(game, "_state", None)
        if state is None:
            return default_obs
        try:
            from craftax.craftax.renderer import render_craftax_text

            return str(render_craftax_text(state))
        except Exception:
            return default_obs

    def area_key(self, game: object, obs_text: str) -> str:
        return str(self._floor(game, obs_text)[0])

    def area_label(self, game: object, obs_text: str) -> str:
        n, name = self._floor(game, obs_text)
        return f"Floor {n} ({name})"

    def focus(self, game: object, obs_text: str) -> str:
        from glyphbench.pro_harness import guides_craftax as gc

        n, name = self._floor(game, obs_text)
        return gc.FLOOR_GUIDES.get(n, f"## Floor {n} ({name}) — (no specific guide)")

    def base_system_prompt(self, game: object) -> str:
        # Hand-tuned Craftax guide based on the original fork harness.
        from glyphbench.pro_harness import guides_craftax as gc

        return (
            f"You are playing {game.env_id()}, the full Craftax roguelike.\n\n"
            + gc.GENERAL_GUIDE
            + "\n\n"
            + gc.ACTION_SPACE_TEXT
        )

    def synonyms(self) -> dict[str, str]:
        return dict(self._SYNONYMS)

    def landmark_vocabulary(self) -> str:
        from glyphbench.pro_harness import guides_craftax as gc

        return gc.LANDMARK_INSTRUCTIONS

    def render_frame(self, game: object):
        from glyphbench.craftax_media import render_native_craftax_frame

        return render_native_craftax_frame(game)

    def raw_score(self, info: dict) -> float | None:
        v = info.get("num_achievements")
        return float(v) if v is not None else None

    def raw_score_label(self) -> str:
        return "achievements"

    def initial_scratchpad(self) -> dict[str, str]:
        return {
            "STRATEGY": (
                "Survive, climb the tech tree (wood->stone->iron->diamond tools + "
                "armour), learn both spells, prepare ice & fire enchants, then "
                "descend floors 0->8 and defeat the Necromancer."
            ),
            "PLAN": (
                "1. [ ] Collect >=3 wood; place table (2); craft wood pickaxe (1)\n"
                "2. [ ] Mine coal+iron, build furnace, craft iron tools + armour\n"
                "3. [ ] Keep food/drink/energy up; find the ladder and DESCEND"
            ),
        }


# ---------------------------------------------------------------------------
# NetHack
# ---------------------------------------------------------------------------
_NETHACK_DLVL_RE = re.compile(r"Dlvl:\s*(\d+)", re.IGNORECASE)
_NETHACK_HUNGER_RE = re.compile(r"Hunger:\s*([A-Za-z]+)", re.IGNORECASE)
_NETHACK_COND_RE = re.compile(r"Condition:\s*([^\n]+)", re.IGNORECASE)
_NETHACK_HP_RE = re.compile(r"HP:\s*(-?\d+)\s*/\s*(\d+)", re.IGNORECASE)


class NetHackAdapter(EnvAdapter):
    area_noun = "dungeon level"

    _SYNONYMS = {
        "north": "MOVE_N", "south": "MOVE_S", "east": "MOVE_E", "west": "MOVE_W",
        "up": "MOVE_N", "down": "MOVE_S", "left": "MOVE_W", "right": "MOVE_E",
        "northeast": "MOVE_NE", "northwest": "MOVE_NW",
        "southeast": "MOVE_SE", "southwest": "MOVE_SW",
        "go down": "GO_DOWN", "descend": "GO_DOWN", "down stairs": "GO_DOWN",
        "go up": "GO_UP", "ascend": "GO_UP", "up stairs": "GO_UP",
        "wait": "WAIT", "rest": "WAIT", "search": "SEARCH",
        "pick up": "PICKUP", "pickup": "PICKUP", "grab": "PICKUP",
        "more": "MORE", "continue": "MORE", "escape": "ESC", "cancel": "ESC",
        "yes": "MOVE_NW", "no": "MOVE_SE",
        "pray": "PRAY", "eat": "EAT", "drink": "QUAFF", "quaff": "QUAFF",
        "read": "READ", "wield": "WIELD", "wear": "WEAR", "throw": "THROW",
        "fire": "FIRE", "kick": "KICK", "open": "OPEN", "close": "CLOSE",
        "inventory": "INVENTORY", "look": "LOOK",
    }

    def _dlvl(self, obs_text: str) -> int:
        m = _NETHACK_DLVL_RE.search(obs_text or "")
        return int(m.group(1)) if m else 1

    def area_key(self, game: object, obs_text: str) -> str:
        return str(self._dlvl(obs_text))

    def area_label(self, game: object, obs_text: str) -> str:
        return f"Dlvl {self._dlvl(obs_text)}"

    def focus(self, game: object, obs_text: str) -> str:
        dlvl = self._dlvl(obs_text)
        lines = [f"You are on dungeon level {dlvl}."]
        hunger_m = _NETHACK_HUNGER_RE.search(obs_text or "")
        hunger = hunger_m.group(1) if hunger_m else ""
        if hunger and hunger.lower() not in {"normal", "not", "satiated"}:
            lines.append(
                f"Hunger is '{hunger}': eat soon (EAT) to avoid starving; "
                "if 'Fainting'/'Starved', eat immediately or PRAY."
            )
        elif hunger.lower() == "satiated":
            lines.append("You are Satiated: do not eat more (risk of choking).")
        hp_m = _NETHACK_HP_RE.search(obs_text or "")
        if hp_m and int(hp_m.group(2)) > 0:
            cur, mx = int(hp_m.group(1)), int(hp_m.group(2))
            if cur <= max(1, mx // 5):
                lines.append(
                    "HP is critically low: disengage, retreat to a corridor, "
                    "and consider PRAY (early prayer to your god usually heals)."
                )
        cond_m = _NETHACK_COND_RE.search(obs_text or "")
        cond = cond_m.group(1).strip() if cond_m else ""
        if cond and cond.lower() != "none":
            lines.append(f"Active conditions: {cond} — address them before exploring.")
        lines.append(_nethack_depth_band(dlvl))
        lines.append(
            "If the [Message] shows a --More-- prompt, send MORE; if it shows a "
            "menu or yes/no question, answer it (MOVE_NW='y', MOVE_SE='n', "
            "ESC cancels) before doing anything else."
        )
        return "\n".join(lines)

    def raw_score(self, info: dict) -> float | None:
        for key in ("nethack_blstats_score", "score", "raw_score"):
            if key in info:
                try:
                    return float(info[key])
                except (TypeError, ValueError):
                    return None
        return None

    def raw_score_label(self) -> str:
        return "score"

    def synonyms(self) -> dict[str, str]:
        return dict(self._SYNONYMS)

    def landmark_vocabulary(self) -> str:
        return (
            "Record durable NetHack landmarks, keyed by dungeon level (Dlvl): "
            "down-stairs (>) and up-stairs (<), altars, fountains, shops, "
            "thrones, sinks, vaults, useful item stashes, and known trap/danger "
            "tiles. Do not record corridors, generic floor, or wandering "
            "monsters. Use MAP_REMOVE when a feature is consumed or no longer "
            "relevant."
        )

    def initial_scratchpad(self) -> dict[str, str]:
        return {
            "STRATEGY": (
                "Survive first, then dive deliberately: keep fed, keep HP up, "
                "improve armor/weapon, identify items safely, and descend to "
                "gain score while avoiding lethal fights."
            ),
            "PLAN": (
                "1. [ ] Explore the current level fully; pick up useful items\n"
                "2. [ ] Manage hunger (EAT) and HP (retreat/PRAY when low)\n"
                "3. [ ] Find the down-stairs (>) and descend when reasonably safe"
            ),
        }


def _nethack_depth_band(dlvl: int) -> str:
    if dlvl <= 1:
        return (
            "Early game (Dlvl 1): explore the whole level, pick up your starting "
            "gear and any items, fight only weak monsters, and locate the "
            "down-stairs (>). Eat corpses that are safe (fresh, non-poisonous)."
        )
    if dlvl <= 5:
        return (
            "Shallow dungeon (Dlvl 2-5): keep improving armor/weapon, watch "
            "hunger, and avoid fighting in the open against multiple monsters. "
            "Use corridors to fight one-at-a-time. Pray to your god if you are "
            "starving, very low HP, or otherwise in trouble (works ~once early)."
        )
    if dlvl <= 10:
        return (
            "Mid dungeon (Dlvl 6-10): expect tougher monsters and the Mines/"
            "Sokoban branches. Have a reliable escape (stairs nearby). Keep "
            "emergency food and healing available. Don't over-extend at low HP."
        )
    return (
        "Deep dungeon (Dlvl 11+): only descend with solid AC, healing, and an "
        "escape plan. Many monsters here can kill quickly; prefer ranged/spells "
        "and retreat over risky melee. Preserve life over greed for score."
    )


# ---------------------------------------------------------------------------
def get_adapter(env_id: str) -> EnvAdapter:
    if "craftax" in env_id:
        return CraftaxAdapter()
    if "nethack" in env_id:
        return NetHackAdapter()
    return _GenericAdapter()


class _GenericAdapter(EnvAdapter):
    def area_key(self, game: object, obs_text: str) -> str:
        return "0"

    def area_label(self, game: object, obs_text: str) -> str:
        return "area 0"

    def focus(self, game: object, obs_text: str) -> str:
        return ""
