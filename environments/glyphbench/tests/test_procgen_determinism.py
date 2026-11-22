"""Determinism test for all procgen envs.

Given the same `seed` passed to `env.reset(seed=...)` and the same action
sequence, two fresh env instances must produce byte-identical trajectories
(observation strings, rewards, termination flags).

This guards against RNG leaks where in-step logic calls `random.*` or
`np.random.*` directly rather than routing through the env's seeded
`self.rng` generator.
"""

from __future__ import annotations

import importlib

import pytest

from glyphbench.envs.procgen.base import ProcgenBase

# Map env_id -> (module, class) for all procgen envs.
PROCGEN_ENVS: list[tuple[str, str]] = [
    ("glyphbench.envs.procgen.coinrun", "CoinRunEnv"),
    ("glyphbench.envs.procgen.maze", "MazeEnv"),
    ("glyphbench.envs.procgen.heist", "HeistEnv"),
    ("glyphbench.envs.procgen.leaper", "LeaperEnv"),
    ("glyphbench.envs.procgen.chaser", "ChaserEnv"),
    ("glyphbench.envs.procgen.bigfish", "BigFishEnv"),
    ("glyphbench.envs.procgen.climber", "ClimberEnv"),
    ("glyphbench.envs.procgen.jumper", "JumperEnv"),
    ("glyphbench.envs.procgen.ninja", "NinjaEnv"),
    ("glyphbench.envs.procgen.fruitbot", "FruitBotEnv"),
    ("glyphbench.envs.procgen.miner", "MinerEnv"),
    ("glyphbench.envs.procgen.dodgeball", "DodgeballEnv"),
    ("glyphbench.envs.procgen.caveflyer", "CaveFlyerEnv"),
    ("glyphbench.envs.procgen.starpilot", "StarPilotEnv"),
    ("glyphbench.envs.procgen.plunder", "PlunderEnv"),
    ("glyphbench.envs.procgen.bossfight", "BossFightEnv"),
]


def _load(mod: str, cls: str) -> type[ProcgenBase]:
    return getattr(importlib.import_module(mod), cls)


def _rollout(
    env_cls: type[ProcgenBase], seed: int, actions: list[int]
) -> list[tuple[str, float, bool]]:
    env = env_cls()
    obs, _ = env.reset(seed=seed)
    trajectory: list[tuple[str, float, bool]] = [(obs, 0.0, False)]
    for a in actions:
        # Cap action to env's action space in case some envs have fewer actions.
        a_clamped = a % env.action_spec.n
        obs, reward, terminated, truncated, _info = env.step(a_clamped)
        trajectory.append((obs, reward, terminated))
        if terminated or truncated:
            break
    return trajectory


@pytest.mark.parametrize("mod,cls", PROCGEN_ENVS, ids=[c for _, c in PROCGEN_ENVS])
def test_procgen_env_deterministic(mod: str, cls: str) -> None:
    """Two fresh instances, same seed, same actions -> identical trajectory."""
    env_cls = _load(mod, cls)
    # Mix of movement + interaction actions. Modulo'd against each env's action_spec.n.
    actions = [0, 1, 2, 3, 4, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 3, 2, 1, 0, 4]
    seed = 42

    traj1 = _rollout(env_cls, seed, actions)
    traj2 = _rollout(env_cls, seed, actions)

    assert len(traj1) == len(traj2), (
        f"{cls}: trajectory length differs between runs: "
        f"{len(traj1)} vs {len(traj2)}"
    )
    for step, (a, b) in enumerate(zip(traj1, traj2, strict=False)):
        assert a == b, (
            f"{cls}: divergence at step {step} with seed={seed}.\n"
            f"  run1: reward={a[1]}, terminated={a[2]}\n"
            f"  run2: reward={b[1]}, terminated={b[2]}\n"
            f"  obs match: {a[0] == b[0]}"
        )


# Envs that deliberately contain no randomness (boss patterns are fully scripted).
# These envs produce identical trajectories regardless of seed by design.
_SEED_INSENSITIVE = {"BossFightEnv"}


@pytest.mark.parametrize(
    "mod,cls",
    [e for e in PROCGEN_ENVS if e[1] not in _SEED_INSENSITIVE],
    ids=[c for _, c in PROCGEN_ENVS if c not in _SEED_INSENSITIVE],
)
def test_procgen_env_different_seed_diverges(mod: str, cls: str) -> None:
    """Sanity check: different seeds should produce different trajectories.

    Some envs (e.g. StarPilot, Plunder) use a static initial level and only
    apply RNG during stepping, so we run actions and compare the full
    trajectory (rendered obs + reward sequence) rather than only the reset.
    """
    env_cls = _load(mod, cls)
    actions = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    traj_a = _rollout(env_cls, 42, actions)
    traj_b = _rollout(env_cls, 1337, actions)

    assert traj_a != traj_b, (
        f"{cls}: different seeds produced identical trajectory -- "
        "seed may be ignored."
    )
