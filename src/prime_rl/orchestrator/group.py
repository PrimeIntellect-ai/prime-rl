"""Group: the user-facing seam.

A Group owns its dataset, env, sampling, scoring, and any data-side state
(difficulty pools, etc.) — every semantic decision. It also owns the env
**worker** lifecycle: `start()` spawns the subprocess env-server (or
attaches to an external one) and `stop()` tears it down. The orchestrator
calls them around `run_groups`.

The shared mutable state is `Policy`: the watcher mutates `policy.version`
and `policy.model_name`; Groups read them at dispatch time and stamp the
read value into each emitted Trajectory.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Protocol, TypeAlias

import torch
import verifiers as vf
from aiolimiter import AsyncLimiter
from verifiers.serve import ZMQEnvClient

from prime_rl.configs.orchestrator import (
    BufferConfig,
    EvalEnvConfig,
    OrchestratorConfig,
    TrainEnvConfig,
)
from prime_rl.orchestrator.advantage import AdvantageInputs, setup_advantage_fn
from prime_rl.orchestrator.env_workers import EnvWorker, attach_env_client, spawn_env_worker
from prime_rl.orchestrator.vf_utils import get_completion_len
from prime_rl.utils.logger import get_logger

Kind: TypeAlias = Literal["train", "eval"]
Pool: TypeAlias = Literal["easy", "normal", "hard"]
_POOLS: tuple[Pool, ...] = ("easy", "normal", "hard")


@dataclass
class Policy:
    """Mutable shared view of the current policy. Written by the watcher,
    read by Groups. Passed by reference — never copied."""

    version: int = 0
    model_name: str = ""


@dataclass
class Trajectory:
    """One unit of work emitted by a Group. Carries the rollouts, their
    common metadata, and the policy version they were dispatched against."""

    example: dict
    env_id: str
    kind: Kind
    rollouts: list[vf.RolloutOutput]
    policy_version: int
    eval_step: int | None = None  # set for eval Trajectories; None for train


class Group(Protocol):
    """The only seam. Owns dataset + env + worker lifecycle + scoring.
    `do_work()` returns zero or more Trajectories — empty means the Group
    has no work right now (idle backoff is the caller's job)."""

    name: str

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def do_work(self) -> list[Trajectory]: ...


def _difficulty_active(config: BufferConfig | None) -> bool:
    if config is None:
        return False
    return config.easy_threshold is not None or config.hard_threshold is not None


# ── Default implementations ────────────────────────────────────────────────


class GRPOGroup:
    """Train Group: one example × N parallel rollouts × group-relative GRPO.

    Owns its own dataset iteration, its own difficulty-pool state, and its
    own env worker subprocess. Reads `policy.version` + `policy.model_name`
    at dispatch time and stamps the snapshot into the emitted Trajectory.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        env_config: TrainEnvConfig,
        *,
        client: vf.ClientConfig,
        policy: Policy,
        rate_limiter: AsyncLimiter | None = None,
    ):
        self.config = config
        self.env_config = env_config
        self.name = env_config.resolved_name
        self.client = client
        self.policy = policy
        self.rate_limiter = rate_limiter
        self._advantage_fn = setup_advantage_fn(config.advantage)

        self._difficulty: BufferConfig | None = config.buffer if _difficulty_active(config.buffer) else None
        seed = self._difficulty.seed if (self._difficulty and self._difficulty.seed is not None) else config.seed
        self._diff_rng = random.Random(seed)
        self._evicted: dict[str, dict] = {}
        self._pool_of: dict[str, Pool] = {}
        self._step_pool_counts: dict[Pool, int] = {p: 0 for p in _POOLS}

        # Built in start(); None until then.
        self.env: vf.Environment | None = None
        self._worker: EnvWorker | None = None
        self._env_client: ZMQEnvClient | None = None
        self._rows: list[dict] = []
        self._cycle: Iterator[dict] | None = None

    @property
    def _max_rollout_time_seconds(self) -> float | None:
        return self.config.max_rollout_time_minutes * 60.0 if self.config.max_rollout_time_minutes else None

    @property
    def _sampling_args(self) -> dict:
        return self.env_config.sampling.to_sampling_args()

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        self.env = vf.load_environment(self.env_config.stripped_id, **self.env_config.args)
        self._rows = [dict(r) for r in self.env.get_dataset()]
        self._cycle = self._cycle_forever()

        log_level = self.config.log.level or "info"
        self._worker = spawn_env_worker(self.env_config, output_dir=self.config.output_dir, log_level=log_level)
        self._env_client = await attach_env_client(self.env, self._worker.address)
        await self._worker.wait_healthy(self._env_client)
        get_logger().info(f"[{self.name}] env worker healthy at {self._worker.address}")

    async def stop(self) -> None:
        if self._env_client is not None:
            await self._env_client.close()
            self._env_client = None
        if self._worker is not None:
            self._worker.terminate()
            self._worker = None

    async def do_work(self) -> list[Trajectory]:
        assert self._cycle is not None, "Group.do_work() called before start()"
        try:
            example = next(self._cycle)
        except StopIteration:
            return []

        version = self.policy.version
        rollouts = await self._gather(example)
        if not rollouts:
            return []

        self._score(rollouts)
        self._observe(example, rollouts)
        return [
            Trajectory(
                example=example,
                env_id=self.name,
                kind="train",
                rollouts=rollouts,
                policy_version=version,
            )
        ]

    async def _gather(self, example: dict) -> list[vf.RolloutOutput]:
        n = self.config.rollouts_per_example
        timeout = self._max_rollout_time_seconds
        if timeout is None:
            return list(await asyncio.gather(*(self._rollout(example) for _ in range(n))))
        # Race rollouts against the wall clock; survivors run on the smaller
        # (noisier) baseline.
        tasks = [asyncio.create_task(self._rollout(example)) for _ in range(n)]
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        for p in pending:
            p.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        return [t.result() for t in done]

    async def _rollout(self, example: dict) -> vf.RolloutOutput:
        assert self.env is not None
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()
        return await self.env.run_rollout(
            vf.RolloutInput(**example),
            client=self.client,
            model=self.policy.model_name,
            sampling_args=self._sampling_args,
            state_columns=["trajectory", "sampling_args"],
        )

    def _score(self, rollouts: list[vf.RolloutOutput]) -> None:
        rewards = torch.tensor([[r.get("reward", 0.0) for r in rollouts]], dtype=torch.float32)
        lens = torch.tensor([[get_completion_len(r) for r in rollouts]], dtype=torch.int64)
        out = self._advantage_fn(AdvantageInputs(rewards=rewards, completion_lengths=lens))
        for r, a in zip(rollouts, out.advantages[0].tolist()):
            r["advantage"] = a

    # ── dataset iteration with difficulty-based filtering ──────────────────

    def _cycle_forever(self) -> Iterator[dict]:
        i = 0
        while True:
            yielded_this_pass = False
            for row in self._rows:
                r = dict(row)
                r["example_id"] = i
                i += 1
                if self._difficulty is not None and self._is_evicted(r):
                    continue
                yielded_this_pass = True
                yield r
            if not yielded_this_pass:
                get_logger().warning(
                    f"Dataset for env={self.name!r} fully evicted; yielding evicted examples to avoid deadlock"
                )
                for row in self._rows:
                    r = dict(row)
                    r["example_id"] = i
                    i += 1
                    yield r

    # ── difficulty pool ────────────────────────────────────────────────────

    def _example_hash(self, example: dict) -> str:
        assert self._difficulty is not None
        keys = [k for k in self._difficulty.hash_keys if k in example]
        if not keys:
            raise ValueError(
                f"No hashable keys for example in env {self.name!r} "
                f"(hash_keys={self._difficulty.hash_keys}, example keys={list(example.keys())})"
            )
        payload = [self.name] + [example[k] for k in keys]
        return hashlib.sha256(json.dumps(payload, default=str).encode()).hexdigest()

    def _is_evicted(self, example: dict) -> bool:
        return self._example_hash(example) in self._evicted

    def _classify(self, avg_reward: float) -> Pool:
        assert self._difficulty is not None
        if self._difficulty.easy_threshold is not None and avg_reward >= self._difficulty.easy_threshold:
            return "easy"
        if self._difficulty.hard_threshold is not None and avg_reward <= self._difficulty.hard_threshold:
            return "hard"
        return "normal"

    def _observe(self, example: dict, rollouts: list[vf.RolloutOutput]) -> None:
        if self._difficulty is None:
            return
        avg_reward = sum(r.get("reward", 0.0) for r in rollouts) / len(rollouts)
        pool = self._classify(avg_reward)
        self._step_pool_counts[pool] += 1
        if pool == "normal":
            return
        h = self._example_hash(example)
        if h not in self._evicted:
            self._evicted[h] = dict(example)
            self._pool_of[h] = pool

    # ── ckpt + metrics ─────────────────────────────────────────────────────

    def state_dict(self) -> dict[str, Any]:
        if self._difficulty is None:
            return {}
        return {"evicted": dict(self._evicted), "pool_of": dict(self._pool_of)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._difficulty is None or not state:
            return
        self._evicted = dict(state.get("evicted", {}))
        self._pool_of = dict(state.get("pool_of", {}))
        n_easy = sum(1 for p in self._pool_of.values() if p == "easy")
        n_hard = sum(1 for p in self._pool_of.values() if p == "hard")
        get_logger().info(f"[{self.name}] Loaded buffer state: {n_easy} easy + {n_hard} hard evicted example(s)")
        self._apply_resume_fractions()

    def _apply_resume_fractions(self) -> None:
        assert self._difficulty is not None
        for pool, fraction in (
            ("easy", self._difficulty.easy_fraction),
            ("hard", self._difficulty.hard_fraction),
        ):
            if fraction <= 0.0:
                continue
            hashes = [h for h, p in self._pool_of.items() if p == pool]
            n = round(len(hashes) * fraction)
            if n <= 0:
                continue
            self._diff_rng.shuffle(hashes)
            for h in hashes[:n]:
                self._pool_of.pop(h, None)
                self._evicted.pop(h, None)
            get_logger().info(f"[{self.name}] Resume: released {n}/{len(hashes)} {pool} example(s)")

    def metrics(self) -> dict[str, float]:
        if self._difficulty is None:
            return {}
        out: dict[str, float] = {}
        n_easy = sum(1 for p in self._pool_of.values() if p == "easy")
        n_hard = sum(1 for p in self._pool_of.values() if p == "hard")
        out[f"buffer/{self.name}/evicted/easy"] = n_easy
        out[f"buffer/{self.name}/evicted/hard"] = n_hard
        total = sum(self._step_pool_counts.values())
        if total:
            for pool in _POOLS:
                out[f"buffer/{self.name}/step/{pool}_rate"] = self._step_pool_counts[pool] / total
        self._step_pool_counts = {p: 0 for p in _POOLS}
        return out


@dataclass
class _EvalEnvHandle:
    """One eval env's runtime state. Static config lives in `config`."""

    config: EvalEnvConfig
    examples: list[dict] = field(default_factory=list)
    env: vf.Environment | None = None
    worker: EnvWorker | None = None
    env_client: ZMQEnvClient | None = None

    @property
    def name(self) -> str:
        return self.config.resolved_name


class EvalGroup:
    """Eval Group: triggers an epoch every `interval` policy versions, then
    drains it one example at a time. `do_work()` returns [] when no epoch is
    active (run_groups idle-backs off). Owns one env worker per eval env."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        client: vf.ClientConfig,
        policy: Policy,
        resume_step: int | None = None,
    ):
        assert config.eval is not None and config.eval.env, "EvalGroup needs at least one eval env"
        self.name = "eval"
        self.config = config
        self.client = client
        self.policy = policy
        self.last_eval_step = 0
        self._eval_at_zero = config.eval.eval_base_model and resume_step is None
        self._envs: list[_EvalEnvHandle] = [_EvalEnvHandle(config=ec) for ec in config.eval.env]
        self._pending: deque[tuple[int, dict, int]] = deque()
        self._dispatching_step: int | None = None
        self._expected: dict[int, int] = {}

    @property
    def interval(self) -> int | None:
        return self.config.eval.interval if self.config.eval is not None else None

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        log_level = self.config.log.level or "info"
        for h in self._envs:
            h.env = vf.load_environment(h.config.stripped_id, **h.config.args)
            ds = h.env.get_eval_dataset() if hasattr(h.env, "get_eval_dataset") else h.env.get_dataset()
            rows = list(ds)
            if h.config.num_examples != -1:
                rows = rows[: h.config.num_examples]
            h.examples = [dict(r) for r in rows]

            h.worker = spawn_env_worker(h.config, output_dir=self.config.output_dir, log_level=log_level)
            h.env_client = await attach_env_client(h.env, h.worker.address)
            await h.worker.wait_healthy(h.env_client)
            get_logger().info(f"[eval/{h.name}] env worker healthy at {h.worker.address}")

        if self._eval_at_zero and self._envs:
            self._start_epoch(0)

    async def stop(self) -> None:
        for h in self._envs:
            if h.env_client is not None:
                await h.env_client.close()
                h.env_client = None
            if h.worker is not None:
                h.worker.terminate()
                h.worker = None

    # ── epoch + drain ──────────────────────────────────────────────────────

    def expected_eval_count(self, step: int) -> int | None:
        return self._expected.get(step)

    def _start_epoch(self, step: int) -> None:
        entries: list[tuple[int, dict, int]] = []
        for i, h in enumerate(self._envs):
            for j, row in enumerate(h.examples):
                ex = dict(row)
                ex["example_id"] = j
                entries.append((i, ex, step))
        self._pending = deque(entries)
        self._dispatching_step = step
        self._expected[step] = len(entries)

    def _maybe_trigger(self) -> None:
        if not self._envs or self.interval is None:
            return
        if self.policy.version < self.last_eval_step + self.interval:
            return
        if self._dispatching_step is not None:
            return
        self._start_epoch(self.policy.version)
        self.last_eval_step = self.policy.version

    async def do_work(self) -> list[Trajectory]:
        self._maybe_trigger()
        if not self._pending:
            return []

        env_idx, example, eval_step = self._pending.popleft()
        if not self._pending:
            self._dispatching_step = None
        h = self._envs[env_idx]
        version = self.policy.version

        rollouts = await asyncio.gather(
            *(self._rollout(h, example) for _ in range(h.config.rollouts_per_example)),
            return_exceptions=False,
        )
        return [
            Trajectory(
                example=example,
                env_id=h.name,
                kind="eval",
                rollouts=list(rollouts),
                policy_version=version,
                eval_step=eval_step,
            )
        ]

    async def _rollout(self, h: _EvalEnvHandle, example: dict) -> vf.RolloutOutput:
        assert h.env is not None
        return await h.env.run_rollout(
            vf.RolloutInput(**example),
            client=self.client,
            model=self.policy.model_name,
            sampling_args=h.config.sampling.to_sampling_args(),
            state_columns=["trajectory", "sampling_args"],
        )


class WeightedGroup:
    """Composes multiple Groups; each `do_work()` picks one weighted-randomly
    and delegates. Forwards start/stop to all sub-Groups."""

    def __init__(self, *, groups: list[Group], weights: list[float], seed: int | None = None):
        assert len(groups) == len(weights)
        self.name = "weighted"
        self._groups = groups
        self._weights = weights
        self._rng = random.Random(seed)

    async def start(self) -> None:
        await asyncio.gather(*(g.start() for g in self._groups))

    async def stop(self) -> None:
        for g in reversed(self._groups):
            try:
                await g.stop()
            except Exception as exc:
                get_logger().warning(f"[{g.name}] stop() failed: {exc}")

    async def do_work(self) -> list[Trajectory]:
        g = self._rng.choices(self._groups, weights=self._weights, k=1)[0]
        return await g.do_work()


# ── Driver ─────────────────────────────────────────────────────────────────


_IDLE_BACKOFF_SECONDS = 0.1


async def run_groups(
    groups: list[Group],
    out_q: asyncio.Queue[Trajectory],
    max_concurrency: int,
) -> None:
    """The metronome. One worker per Group; all share a global semaphore so
    total in-flight `do_work()` calls never exceed `max_concurrency`. Empty
    returns trigger an idle backoff so EvalGroups (which return [] between
    epochs) don't busy-spin. Caller is responsible for `start()`/`stop()`."""
    sem = asyncio.Semaphore(max_concurrency)
    get_logger().debug(f"run_groups starting with {len(groups)} group(s), concurrency={max_concurrency}")

    async def _worker(g: Group) -> None:
        while True:
            async with sem:
                trajs = await g.do_work()
            if not trajs:
                await asyncio.sleep(_IDLE_BACKOFF_SECONDS)
                continue
            for t in trajs:
                await out_q.put(t)

    await asyncio.gather(*(_worker(g) for g in groups))


# ── Setup ──────────────────────────────────────────────────────────────────


def setup_train_groups(
    config: OrchestratorConfig,
    *,
    client: vf.ClientConfig,
    policy: Policy,
) -> tuple[list[Group], list[GRPOGroup]]:
    """Build one GRPOGroup per train env. Returns `(dispatch_groups,
    train_groups)`. `start()` hasn't been called yet — env load + worker
    spawn happens there."""
    rate_limiter = AsyncLimiter(max_rate=config.tasks_per_minute, time_period=60) if config.tasks_per_minute else None
    train_groups = [
        GRPOGroup(config, env_config, client=client, policy=policy, rate_limiter=rate_limiter)
        for env_config in config.train.env
    ]

    ratios = [env_config.ratio for env_config in config.train.env]
    if all(r is not None for r in ratios) and len(train_groups) > 1:
        named = ", ".join(f"{e.resolved_name}={e.ratio:.2f}" for e in config.train.env)
        get_logger().info(f"Sampling train envs by ratio ({named})")
        return [WeightedGroup(groups=list(train_groups), weights=list(ratios), seed=config.seed)], train_groups
    get_logger().info(f"Sampling train envs round-robin ({len(train_groups)} env(s))")
    return list(train_groups), train_groups


def setup_eval_group(
    config: OrchestratorConfig,
    *,
    client: vf.ClientConfig,
    policy: Policy,
    resume_step: int | None,
) -> EvalGroup | None:
    if config.eval is None or not config.eval.env:
        return None
    return EvalGroup(config, client=client, policy=policy, resume_step=resume_step)


def make_groups_queue(config: OrchestratorConfig, num_groups: int) -> tuple[asyncio.Queue[Trajectory], int]:
    assert config.max_inflight_rollouts is not None
    concurrency = max(1, config.max_inflight_rollouts // config.rollouts_per_example)
    get_logger().info(f"run_groups concurrency: {concurrency} across {num_groups} group(s)")
    queue: asyncio.Queue[Trajectory] = asyncio.Queue(maxsize=concurrency * (config.max_async_level + 1))
    return queue, concurrency
