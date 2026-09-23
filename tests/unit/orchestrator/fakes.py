"""Fakes for driving one orchestrator component at a time.

Every component talks to its neighbours through hooks bound with ``bind``; these
helpers record those calls and stand in for the envs, clients and transports the
components would otherwise reach. Episodes are real ``vf.Episode`` objects stamped
the way the dispatcher stamps them, so sinks and queues see production shapes.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import verifiers.v1 as vf

from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation, TaskRequest, WorkKind
from prime_rl.transports.batch import TrainingSample


class RecordingHooks:
    """Hands out callables that record their calls under a name."""

    def __init__(self) -> None:
        self.calls: dict[str, list[tuple]] = {}

    def record(self, name: str, *, result: Any = None) -> Callable[..., Any]:
        self.calls.setdefault(name, [])

        def hook(*args: Any, **kwargs: Any) -> Any:
            self.calls[name].append(args if not kwargs else (*args, kwargs))
            return result

        return hook

    def record_async(self, name: str, *, result: Any = None) -> Callable[..., Any]:
        self.calls.setdefault(name, [])

        async def hook(*args: Any, **kwargs: Any) -> Any:
            self.calls[name].append(args if not kwargs else (*args, kwargs))
            return result

        return hook

    def __getitem__(self, name: str) -> list[tuple]:
        return self.calls.get(name, [])


class RecordingMonitors:
    """The ``prime_rl.monitors`` surface the components log through."""

    def __init__(self) -> None:
        self.metrics: list[tuple[dict, int | None]] = []
        self.episodes: list[tuple[list, int, str, str]] = []
        self.annotations: list[list] = []
        self.plans: list[tuple[str, int, int]] = []
        self.live: list[list] = []
        self.epochs: list[tuple[str, int, list]] = []

    async def log(self, data, step, kind="train", subset="effective") -> None:
        if isinstance(data, dict):
            self.metrics.append((data, step))
        else:
            self.episodes.append((data if isinstance(data, list) else [data], step, kind, subset))

    async def log_annotations(self, updates) -> None:
        self.annotations.append(list(updates))

    async def log_eval_plan(self, env_name, step, expected) -> None:
        self.plans.append((env_name, step, expected))

    async def log_live(self, events) -> None:
        self.live.append(list(events))

    async def log_eval_epoch(self, env_name, step, episodes) -> None:
        self.epochs.append((env_name, step, list(episodes)))


def make_task(idx: int = 0) -> vf.Task:
    return vf.Task(vf.TaskData(idx=idx, prompt=f"task {idx}"))


def make_episode(
    *,
    env_name: str = "env",
    group_id: str | None = None,
    reward: float = 1.0,
    sampled_tokens: int = 3,
    kind: WorkKind = "train",
    step: int = 1,
    policy: tuple[int, int] | None = (0, 0),
    task: vf.Task | None = None,
    ok: bool = True,
    trainable: bool = True,
) -> vf.Episode:
    """One single-trace episode as the dispatcher would emit it: a user turn, one
    sampled assistant turn of ``sampled_tokens`` tokens, and run provenance."""
    task = task or make_task()
    nodes = [
        vf.MessageNode(
            message=vf.UserMessage(content="q"), token_ids=[0], mask=[False], logprobs=[0.0], sampled=False, parent=None
        ),
        vf.MessageNode(
            message=vf.AssistantMessage(content="a"),
            token_ids=list(range(1, sampled_tokens + 1)),
            mask=[True] * sampled_tokens,
            logprobs=[-0.1] * sampled_tokens,
            sampled=True,
            parent=0,
        ),
    ]
    trace = vf.Trace[vf.TaskData](
        task=vf.TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash),
        agent=vf.AgentInfo(config=vf.AgentConfig(), trainable=trainable),
        nodes=nodes,
        calls=[vf.ModelCall(node=1, usage=vf.Usage(prompt_tokens=1, completion_tokens=sampled_tokens))],
        rewards={"reward": vf.Reward(score=reward)},
        ok=ok,
        is_completed=True,
    )
    if not ok:
        trace.errors.append(vf.Error(type="TestError", message="failed"))
    span = vf.PolicySpan(start=policy[0], end=policy[1]) if policy is not None else None
    work = vf.EvalWorkInfo(step=step, policy=span) if kind == "eval" else vf.TrainWorkInfo(step=step, policy=span)
    episode = vf.Episode(
        env=vf.EnvInfo(id=env_name, name=env_name),
        task=trace.task,
        group=vf.GroupInfo(id=group_id or uuid.uuid4().hex),
        traces=[trace],
        ok=ok,
    )
    episode.record_run(vf.TrainRunInfo(id="run", name="run", work=work))
    return episode


def make_failure(*, env_name: str = "env", group_id: str, kind: WorkKind = "train", step: int = 1) -> DispatchFailure:
    return DispatchFailure(
        kind=kind,
        env_name=env_name,
        group_id=group_id,
        step=step,
        policy_version=0,
        task_type="Task",
        task_key="k",
        task_hash="h",
        error=vf.Error(type="Boom", message="boom"),
    )


def make_cancellation(
    *, env_name: str = "env", group_id: str, count: int, reason: str = "stale", kind: WorkKind = "train", step: int = 1
) -> GroupCancellation:
    return GroupCancellation(kind=kind, env_name=env_name, group_id=group_id, step=step, count=count, reason=reason)


def make_sample(tokens: int = 3, advantage: float = 1.0, env_name: str = "env") -> TrainingSample:
    return TrainingSample(
        token_ids=list(range(tokens)),
        mask=[True] * tokens,
        logprobs=[-0.1] * tokens,
        temperatures=[1.0] * tokens,
        env_name=env_name,
        advantages=[advantage] * tokens,
    )


class FakeAlgorithm:
    """Scores nothing; marks every trace's advantage so ``trace_to_samples`` compiles it."""

    action_loss_type = "rl"
    connected = None

    def __init__(self) -> None:
        self.finalized_episodes = 0
        self.finalized_groups = 0

    async def finalize_episode(self, episode: vf.Episode) -> None:
        self.finalized_episodes += 1

    async def finalize_group(self, episodes: list[vf.Episode]) -> None:
        from prime_rl.orchestrator.algo import assign_advantages

        self.finalized_groups += 1
        for episode in episodes:
            for trace in episode.traces:
                assign_advantages(trace, 1.0)


@dataclass
class FakeEnv:
    """The slice of ``TrainEnv`` / ``EvalEnv`` the components read."""

    name: str
    group_size: int = 2
    algorithm: FakeAlgorithm = field(default_factory=FakeAlgorithm)
    sampling_args: dict = field(default_factory=lambda: {"temperature": 1.0})
    requires_sampling_masks: bool = False
    uses_live_policy: bool = True
    examples: list[vf.Task] = field(default_factory=list)
    run_delay: float = 0.0
    runs: list[dict] = field(default_factory=list)
    fail_every: int | None = None
    """Raise on every N-th ``run`` call to exercise the failure path."""

    def __post_init__(self) -> None:
        self.config = SimpleNamespace(group_size=self.group_size, resolved_name=self.name)
        self.generation_source = SimpleNamespace(uses_live_policy=self.uses_live_policy, clients=None, connected=None)

    async def run(self, *, client, model_name, cache_salt, task_data, on_delta=None) -> vf.Episode:
        self.runs.append({"model": model_name, "cache_salt": cache_salt, "task": task_data})
        if self.fail_every and len(self.runs) % self.fail_every == 0:
            raise RuntimeError("env exploded")
        if self.run_delay:
            await asyncio.sleep(self.run_delay)
        task = vf.Task(vf.TaskData(**task_data))
        return make_episode(env_name=self.name, task=task, policy=None)


class FakeEnvs:
    def __init__(self, *envs: FakeEnv) -> None:
        self._envs = {env.name: env for env in envs}

    @property
    def names(self) -> list[str]:
        return list(self._envs)

    def get(self, name: str) -> FakeEnv:
        return self._envs[name]

    def __iter__(self):
        return iter(self._envs.values())

    def __len__(self) -> int:
        return len(self._envs)


class FakeSource:
    """Hands out one task per call, round-robin over the envs."""

    def __init__(self, envs: FakeEnvs, *, limit: int | None = None) -> None:
        self.envs = envs
        self.limit = limit
        self.served = 0

    def next_task(self, *, step: int) -> TaskRequest | None:
        if self.limit is not None and self.served >= self.limit:
            return None
        env = list(self.envs)[self.served % len(self.envs)]
        self.served += 1
        return TaskRequest(env_name=env.name, task=make_task(self.served), step=step)


class FakeClients:
    """``InferenceClient`` as the dispatcher sees it."""

    model_name = "policy"
    train_client = None
    eval_client = None

    def __init__(self) -> None:
        self.finished: list[list[str]] = []

    async def finish_sessions(self, session_ids: list[str]) -> None:
        self.finished.append(list(session_ids))


class FakeReceiver:
    """A ``WeightReceiver`` whose versions the test publishes by hand."""

    def __init__(self) -> None:
        self.published: set[int] = set()
        self.received: list[int] = []
        self.synced: list[int] = []

    def publish(self, step: int) -> None:
        self.published.add(step)

    def next_version(self, current: int) -> int:
        newer = [step for step in self.published if step > current]
        return min(newer) if newer else current

    async def wait_published(self, step: int, *, cancelled) -> None:
        while step not in self.published:
            await asyncio.sleep(0.01)

    async def receive(self, step: int) -> None:
        self.received.append(step)

    async def sync_startup(self, step: int, timeout: float) -> None:
        self.synced.append(step)
