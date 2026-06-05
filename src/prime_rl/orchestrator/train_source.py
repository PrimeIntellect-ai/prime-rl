"""TrainSource: weighted env mix over per-env samplers, infinite pull.

Env selection is delegated to an ``EnvMixStrategy`` (default: weighted
round-robin by configured ``ratio`` when all envs set one, else by per-env
dataset size); example selection within an env is delegated to that env's
``SampleStrategy`` (default: a reshuffling cursor). Both are swappable seams.
Returned dicts carry ``env_name`` + ``example_id``.
"""

from __future__ import annotations

from prime_rl.orchestrator.envs import TrainEnv, TrainEnvs
from prime_rl.orchestrator.sampling import WeightedRoundRobin


class TrainSource:
    """``next_example(available_permits)`` picks an env via the mix strategy and
    pulls that env's next example from its sampler (or ``None`` when the env's
    per-call permit cost doesn't fit — the dispatch loop retries when permits
    free up)."""

    def __init__(self, train_envs: TrainEnvs, *, seed: int | None) -> None:
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")
        self._envs_by_name = {env.name: env for env in self.envs}

        # Build each env's sampler (which owns its dataset) and per-env permit
        # cost. Group-scoring envs reserve ``group_size`` permits up front;
        # per-rollout envs need 1. Per-env seeds keep distinct envs from
        # shuffling in lockstep.
        self.env_costs: dict[str, int] = {}
        for i, env in enumerate(self.envs):
            env.build_sampler(seed=(seed + i) if seed is not None else None)
            self.env_costs[env.name] = env.config.group_size if env.requires_group_scoring else 1

        env_names = [env.name for env in self.envs]
        configured_ratios = [env.config.ratio for env in self.envs]
        if all(r is not None for r in configured_ratios):
            weights = [float(r) for r in configured_ratios]  # type: ignore[arg-type]
        else:
            weights = [float(self._dataset_size(env)) for env in self.envs]
        self.env_mix = WeightedRoundRobin(env_names, weights, seed=seed)

    @staticmethod
    def _dataset_size(env: TrainEnv) -> int:
        size = getattr(env.sampler, "dataset_size", None)
        if size is None:
            raise ValueError(
                f"Env {env.name!r} sampler exposes no dataset_size; set explicit per-env ratios to weight the env mix."
            )
        return size

    def next_example(self, available_permits: int) -> dict | None:
        env_name = self.env_mix.pick()
        if self.env_costs[env_name] > available_permits:
            return None
        sampler = self._envs_by_name[env_name].sampler
        assert sampler is not None  # built in __init__
        return sampler.next()
