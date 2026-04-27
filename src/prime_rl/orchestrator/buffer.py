"""Difficulty-based example pool tracking.

After each train group completes, classify the example as `easy` / `normal` /
`hard` from its average reward and (for easy/hard) evict it from sampling.
The scheduler consults `is_evicted` when iterating its dataset to skip
evicted examples on subsequent epochs.
"""

import hashlib
import json
import random
from collections import defaultdict
from typing import Any, Literal, TypeAlias

from prime_rl.configs.orchestrator import BufferConfig
from prime_rl.orchestrator.engine import Group
from prime_rl.utils.logger import get_logger

Pool: TypeAlias = Literal["easy", "normal", "hard"]
POOLS: tuple[Pool, ...] = ("easy", "normal", "hard")


class DifficultyBuffer:
    def __init__(self, config: BufferConfig, env_names: list[str], seed: int | None = None):
        self.config = config
        self.env_names = env_names
        self.logger = get_logger()
        # `seed` falls through from config; orchestrator-level seed is used as
        # a fallback so resume bookkeeping is reproducible.
        self._rng = random.Random(config.seed if config.seed is not None else seed)
        # env_id -> {hash -> stored example dict}. Keep the dict so we can
        # reconstruct hashes on resume even if hash_keys changes meaning we
        # need to re-derive eviction sets.
        self._evicted: dict[str, dict[str, dict]] = defaultdict(dict)
        # hash -> "easy" | "hard"
        self._pool_of: dict[str, Pool] = {}
        # per-step counters reset by metrics()
        self._reset_step_counters()

    @property
    def active(self) -> bool:
        return self.config.easy_threshold is not None or self.config.hard_threshold is not None

    def _reset_step_counters(self) -> None:
        self._step_examples_per_env: dict[str, dict[Pool, int]] = defaultdict(lambda: {p: 0 for p in POOLS})

    def example_hash(self, env_id: str, example: dict) -> str:
        """Hash that survives across runs as long as the hash_keys are stable.
        env_id is always part of the hash so the same prompt under different
        envs is treated as distinct."""
        keys = [k for k in self.config.hash_keys if k in example]
        if not keys:
            raise ValueError(
                f"No hashable keys for example in env {env_id!r} (hash_keys={self.config.hash_keys}, "
                f"example keys={list(example.keys())})"
            )
        payload = [env_id] + [example[k] for k in keys]
        return hashlib.sha256(json.dumps(payload, default=str).encode()).hexdigest()

    def is_evicted(self, env_id: str, example: dict) -> bool:
        if not self.active:
            return False
        return self.example_hash(env_id, example) in self._evicted[env_id]

    def observe(self, group: Group) -> Pool:
        """Classify a completed train group and update pool state. Returns the
        assigned pool for metric aggregation."""
        if not group.rollouts:
            return "normal"
        avg_reward = sum(r.get("reward", 0.0) for r in group.rollouts) / len(group.rollouts)
        pool = self._classify(avg_reward)
        self._step_examples_per_env[group.env_id][pool] += 1
        if pool != "normal":
            h = self.example_hash(group.env_id, group.example)
            if h not in self._evicted[group.env_id]:
                self._evicted[group.env_id][h] = dict(group.example)
                self._pool_of[h] = pool
        return pool

    def _classify(self, avg_reward: float) -> Pool:
        if self.config.easy_threshold is not None and avg_reward >= self.config.easy_threshold:
            return "easy"
        if self.config.hard_threshold is not None and avg_reward <= self.config.hard_threshold:
            return "hard"
        return "normal"

    def state_dict(self) -> dict[str, Any]:
        return {
            "evicted": {env: dict(d) for env, d in self._evicted.items()},
            "pool_of": dict(self._pool_of),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._evicted = defaultdict(dict, {env: dict(d) for env, d in state.get("evicted", {}).items()})
        self._pool_of = dict(state.get("pool_of", {}))
        n_easy = sum(1 for p in self._pool_of.values() if p == "easy")
        n_hard = sum(1 for p in self._pool_of.values() if p == "hard")
        self.logger.info(f"Loaded buffer state: {n_easy} easy + {n_hard} hard evicted example(s)")
        self._apply_resume_fractions()

    def _apply_resume_fractions(self) -> None:
        """On resume, optionally release a fraction of easy/hard examples back
        to the normal pool. Matches the old buffer's `easy_fraction` /
        `hard_fraction` semantics: useful for periodic re-evaluation of
        previously-classified examples."""
        for pool, fraction in (("easy", self.config.easy_fraction), ("hard", self.config.hard_fraction)):
            if fraction <= 0.0:
                continue
            hashes = [h for h, p in self._pool_of.items() if p == pool]
            n = round(len(hashes) * fraction)
            if n <= 0:
                continue
            self._rng.shuffle(hashes)
            released = hashes[:n]
            for h in released:
                self._pool_of.pop(h, None)
                for env in list(self._evicted):
                    self._evicted[env].pop(h, None)
            self.logger.info(f"Resume: released {n}/{len(hashes)} {pool} example(s) back to normal")

    def metrics(self) -> dict[str, float]:
        """Per-step buffer metrics. Resets the per-step counters."""
        out: dict[str, float] = {}
        n_easy = sum(1 for p in self._pool_of.values() if p == "easy")
        n_hard = sum(1 for p in self._pool_of.values() if p == "hard")
        out["buffer/evicted/easy"] = n_easy
        out["buffer/evicted/hard"] = n_hard

        total = sum(sum(c.values()) for c in self._step_examples_per_env.values())
        if total:
            for pool in POOLS:
                pool_total = sum(c[pool] for c in self._step_examples_per_env.values())
                out[f"buffer/step/{pool}_rate"] = pool_total / total
            for env, counts in self._step_examples_per_env.items():
                env_total = sum(counts.values())
                if not env_total:
                    continue
                for pool in POOLS:
                    out[f"buffer/{env}/step/{pool}_rate"] = counts[pool] / env_total
        self._reset_step_counters()
        return out


def setup_buffer(config: BufferConfig, env_names: list[str], seed: int | None) -> DifficultyBuffer | None:
    """Returns a buffer iff at least one threshold is set. Otherwise None
    (scheduler/batcher treat None as disabled)."""
    buf = DifficultyBuffer(config, env_names, seed)
    return buf if buf.active else None
