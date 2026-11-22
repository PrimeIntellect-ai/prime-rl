"""Regression tests for model-facing HUD policy."""

from __future__ import annotations

import re

import glyphbench  # noqa: F401 - registers all envs
from glyphbench.core.registry import make_env
from glyphbench.core.task_selection import list_task_ids

_REDUNDANT_SPATIAL_PATTERNS = re.compile(
    r"("
    r"\b(?:Pos|Position):|"
    r"\b(?:You|Ship|Tank|Sub|Car|Player):\s*\(|"
    r"\b(?:Ball|Puck):\s*pos=|"
    r"\b(?:Paddle|Cannon|Turret|Bowler|Blaster|Jet|Skier|Gopher)\s+x=|"
    r"\b(?:World pos|Lane idx|On beam|Wave shift=|paddle row|center at row)\b|"
    r"\b(?:Enemies|Robots|Ghosts|Cops|Aliens|Rocks|Raiders|Banks|Segments|"
    r"Demons|Humans|Landers|Phoenixes|Obstacles|Divers|Monsters|Beams|"
    r"Turrets):\s*\("
    r")",
    re.IGNORECASE,
)

_DIRECTIONAL_FACING_HUD_ENVS = (
    "glyphbench/atari-alien-v0",
    "glyphbench/atari-battlezone-v0",
    "glyphbench/atari-berzerk-v0",
    "glyphbench/atari-defender-v0",
    "glyphbench/atari-seaquest-v0",
    "glyphbench/atari-solaris-v0",
    "glyphbench/atari-timepilot-v0",
    "glyphbench/atari-tutankham-v0",
    "glyphbench/atari-venture-v0",
    "glyphbench/atari-wizardofwor-v0",
    "glyphbench/procgen-dodgeball-v0",
    "glyphbench/procgen-ninja-v0",
)


def test_initial_huds_do_not_expose_grid_redundant_positions() -> None:
    offenders: list[str] = []
    # AgenticK is an opt-in structured-task suite whose public task config
    # intentionally includes coordinates (for example graph node positions),
    # rather than a duplicate rendering of a hidden game grid.
    for env_id in list_task_ids(
        exclude_suites=["craftaxfull", "nethack", "agentick"]
    ):
        env = make_env(env_id)
        env.reset(seed=0)
        hud = env.get_observation().hud.replace("\n", " | ")
        if _REDUNDANT_SPATIAL_PATTERNS.search(hud):
            offenders.append(f"{env_id}: {hud}")

    assert offenders == []


def test_directional_agent_facing_not_repeated_in_hud() -> None:
    offenders: list[str] = []
    for env_id in _DIRECTIONAL_FACING_HUD_ENVS:
        env = make_env(env_id)
        env.reset(seed=0)
        hud = env.get_observation().hud
        if "Facing:" in hud:
            offenders.append(f"{env_id}: {hud}")

    assert offenders == []


def test_wave_defense_keeps_non_directional_cannon_facing() -> None:
    env = make_env("glyphbench/classics-wavedefense-v0")
    env.reset(seed=0)
    hud = env.get_observation().hud

    assert "Facing:" in hud


def test_atari_diagonal_ship_facing_only_appears_when_grid_arrow_is_lossy() -> None:
    asteroids = make_env("glyphbench/atari-asteroids-v0")
    asteroids.reset(seed=0)
    assert "Ship facing:" not in asteroids.get_observation().hud
    asteroids.step(asteroids.action_spec.index_of("RIGHT"))
    assert "Ship facing: up-right" in asteroids.get_observation().hud

    gravitar = make_env("glyphbench/atari-gravitar-v0")
    gravitar.reset(seed=0)
    assert "Ship facing:" not in gravitar.get_observation().hud
    gravitar.step(gravitar.action_spec.index_of("RIGHT"))
    assert "Ship facing: up-right" in gravitar.get_observation().hud
