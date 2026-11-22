"""Cross-family regression tests for the additive death-penalty pattern.

For GRPO training on Monte Carlo returns, it is critical that the terminal
death penalty does NOT discard positive rewards earned during the episode.
The fix is to apply -1 (or -X) ADDITIVELY at the death step:

    reward += -1.0   # GOOD: preserves any progress credited this tick

Not as an OVERRIDE:

    reward = -1.0    # BAD: discards same-step progress + masks the signal

These tests cover the canonical envs across all families. Each test
simulates a step in which the agent earns positive shaping reward AND
dies on the same tick, and asserts the resulting step reward is the
progress-this-step minus the death penalty — not a clean -1.
"""
from __future__ import annotations

import pytest


def test_craftax_iron_bootstrap_milestone_credits_even_on_death() -> None:
    from glyphbench.envs.craftax.base import TILE_TABLE
    from glyphbench.envs.craftax.subtasks_extended import CraftaxIronBootstrapEnv

    env = CraftaxIronBootstrapEnv()
    env.reset(seed=0)
    # 4 prior milestones credited; 5th (table) fires this step + die.
    env._milestones = {"wood", "stone", "wood_pickaxe", "stone_pickaxe"}
    env._inventory = {
        "wood": 5, "stone": 5, "wood_pickaxe": 1, "stone_pickaxe": 1,
    }
    env._world[5][5] = TILE_TABLE
    env._hp = 0
    _, reward, terminated, _, info = env.step(env.action_spec.index_of("NOOP"))
    # Step reward = +1/7 (table milestone) + (-1) death = -6/7.
    assert terminated
    assert reward == pytest.approx(1 / 7 - 1.0)
    assert info["subtask_success"] is False


def test_craftax_fight_zombies_final_kill_survives_simultaneous_death() -> None:
    from glyphbench.envs.craftax.subtasks_extended import CraftaxFightZombiesEnv

    env = CraftaxFightZombiesEnv(max_turns=20)
    env.reset(seed=0)
    env._zombie_kills_credited = 2  # 2 zombies already credited
    env._mobs = [{
        "type": "zombie", "x": env._agent_x + 1, "y": env._agent_y,
        "hp": 1, "max_hp": 1, "attack_cooldown": 0,
    }]
    env._inventory = {"stone_sword": 1}
    env._facing = (1, 0)
    env._hp = 0
    _, reward, terminated, _, info = env.step(env.action_spec.index_of("DO"))
    assert terminated
    # Step reward = +1/5 (third of five kills) + (-1) death = -0.8.
    assert reward == pytest.approx(1 / 5 - 1.0)
    assert info.get("subtask_success") is False


def test_procgen_bigfish_eat_credit_survives_being_eaten_same_step() -> None:
    """A small fish eaten + bigger fish eats the agent same step must
    yield reward = +1/N (eat) + (-1) (death), not a clean -1."""
    from glyphbench.envs.procgen.bigfish import BigFishEnv

    env = BigFishEnv()
    env.reset(seed=0)
    # Force a specific state: small fish and big fish both at agent's cell
    # after collision resolution.
    env._entities = []
    env._agent_size = 2
    # Small fish (size 1, eaten by agent for +reward)
    small = env._add_entity("fish", "f", env._agent_x, env._agent_y)
    small.data = {"size": 1}
    # Big fish (size 5, eats agent for -reward)
    big = env._add_entity("fish", "F", env._agent_x, env._agent_y)
    big.data = {"size": 5}
    reward, terminated = env._resolve_fish_collisions()
    # Eat one fish: +1/N. Bigger fish eats agent: -1. Total = 1/N - 1.
    assert terminated
    expected = 1.0 / env._WIN_TARGET + env._DEATH_PENALTY
    assert reward == pytest.approx(expected), (
        f"Expected additive {expected}, got {reward}"
    )


def test_atari_qbert_cube_credit_survives_enemy_collision_same_step() -> None:
    from glyphbench.envs.atari.qbert import QbertEnv

    env = QbertEnv()
    env.reset(seed=0)
    env._spawn_enemy()
    # Seed 0 sends the enemy down-right on its first move.
    _, reward, terminated, _, _ = env.step(env.action_spec.index_of("DOWN_RIGHT"))

    assert terminated
    assert env._progress_count == 1
    assert reward == pytest.approx(1 / env._WIN_TARGET - 1)


def test_atari_robotank_ram_credit_survives_last_sensor_loss() -> None:
    from glyphbench.envs.atari.robotank import RobotankEnv

    env = RobotankEnv()
    env.reset(seed=0)
    env._sensors = {name: name == "video" for name in env._SENSORS}
    env._tanks = []
    tank = env._add_entity("tank", "T", env._player_x, env._player_y)
    tank.data.update(hp=2, timer=0, fire_cd=0)
    env._tanks.append(tank)

    _, reward, terminated, _, _ = env.step(env.action_spec.index_of("NOOP"))

    assert terminated
    assert env._kills == 1
    assert reward == pytest.approx(1 / env._WIN_TARGET - 1)


def test_atari_solaris_kill_credit_survives_enemy_fire_same_step() -> None:
    from glyphbench.envs.atari.solaris import SolarisEnv

    env = SolarisEnv()
    env.reset(seed=0)
    enemy = env._enemies[0]
    bullet = env._add_entity("bullet", "*", enemy.x, enemy.y)
    bullet.data.update(bdx=0, bdy=0)
    env._bullets = [bullet]
    enemy_bullet = env._add_entity("enemy_bullet", "o", env._player_x, env._player_y)
    enemy_bullet.data.update(bdx=0, bdy=0)
    env._enemy_bullets = [enemy_bullet]

    _, reward, terminated, _, _ = env.step(env.action_spec.index_of("NOOP"))

    assert terminated
    assert env._progress_count == 1
    assert reward == pytest.approx(1 / env._WIN_TARGET - 1)
