"""Tests for the Pro harness env adapters."""

from __future__ import annotations

from glyphbench.pro_harness.adapters import (
    CraftaxAdapter,
    NetHackAdapter,
    get_adapter,
)

# The adapter area/observation/focus methods take (game, obs_text). For Craftax
# we pass game=None so it falls back to parsing the floor from the obs text
# (the live path reads game._state.player_level).
CRAFTAX_OBS = (
    "Floor: 3 (Sewers)    Pos: (12,8)    Facing: up    Light: Day\n"
    "HP: 9/9    Food: 9/9\n"
    "@.\n"
)
NETHACK_OBS = (
    "[HUD]\n"
    "HP: 4/16    Pw: 0/0    AC: 7\n"
    "Dlvl: 3    Dungeon: 0    Level: 2    Position: 40,10    Time: 250\n"
    "Hunger: Hungry    Encumbrance: Unencumbered    Condition: None\n"
    "[Grid]\n@\n"
)


def test_get_adapter_routing():
    assert isinstance(get_adapter("glyphbench/craftaxfull-v0"), CraftaxAdapter)
    assert isinstance(get_adapter("glyphbench/nethack-full-v0"), NetHackAdapter)


def test_craftax_area_and_focus():
    ad = CraftaxAdapter()
    assert ad.area_key(None, CRAFTAX_OBS) == "3"
    assert ad.area_label(None, CRAFTAX_OBS) == "Floor 3 (Sewers)"
    focus = ad.focus(None, CRAFTAX_OBS)
    assert "Floor 3" in focus
    # Pulls the verbatim Sewers floor guide.
    assert "Sewers" in focus


def test_craftax_synonyms_map_to_fork_conventions():
    # The JAX fork uses DO for ALL interactions (mine/chop/attack/drink/eat),
    # and original movement names LEFT/RIGHT/UP/DOWN (aliased to MOVE_*).
    syn = CraftaxAdapter().synonyms()
    assert syn["drink"] == "DO"
    assert syn["eat"] == "DO"
    assert syn["mine"] == "DO"
    assert syn["left"] == "LEFT"
    assert syn["descend"] == "DESCEND"


def test_craftax_default_floor_when_missing():
    ad = CraftaxAdapter()
    assert ad.area_key(None, "@") == "0"


def test_craftax_base_prompt_uses_hand_tuned_guide():
    class _FakeGame:
        def env_id(self):
            return "glyphbench/craftaxfull-v0"

    prompt = CraftaxAdapter().base_system_prompt(_FakeGame())
    assert "Craftax Game Guide" in prompt
    assert "## Available Actions" in prompt
    # Original glyph/action conventions.
    assert "DESCEND" in prompt
    assert "MAKE_IRON_ARMOUR" in prompt
    # Long-horizon survival cues remain prompt guidance, not scripted actions.
    compact_prompt = " ".join(prompt.split())
    assert "one helm is only partial protection" in compact_prompt
    assert "Never SLEEP exposed" in compact_prompt
    assert "four-sided stone enclosure" in compact_prompt
    assert "well-spaced lights" in compact_prompt


def test_craftax_floor_two_focus_reinforces_lit_foothold_and_armour():
    obs = CRAFTAX_OBS.replace("Floor: 3 (Sewers)", "Floor: 2 (Gnomish Mines)")
    focus = CraftaxAdapter().focus(None, obs)
    assert "lit, defensible foothold" in focus
    assert "One action makes only one armour piece" in focus


def test_craftax_observation_falls_back_without_state():
    # game=None has no _state/_runtime, so the default obs is returned verbatim.
    ad = CraftaxAdapter()
    assert ad.observation(None, "DEFAULT") == "DEFAULT"
    assert ad.native_text_observation(None, "DEFAULT") == "DEFAULT"


def test_nethack_depth_and_survival_cues():
    ad = NetHackAdapter()
    assert ad.area_key(None, NETHACK_OBS) == "3"
    assert ad.area_label(None, NETHACK_OBS) == "Dlvl 3"
    focus = ad.focus(None, NETHACK_OBS)
    assert "dungeon level 3" in focus
    # Hunger cue surfaced.
    assert "Hunger is 'Hungry'" in focus or "eat" in focus.lower()
    assert "menu" in focus.lower() or "More" in focus


def test_nethack_low_hp_warning():
    obs = NETHACK_OBS.replace("HP: 4/16", "HP: 2/16")
    focus = NetHackAdapter().focus(None, obs)
    assert "HP is critically low" in focus


def test_initial_scratchpads_present():
    assert "STRATEGY" in CraftaxAdapter().initial_scratchpad()
    assert "STRATEGY" in NetHackAdapter().initial_scratchpad()
