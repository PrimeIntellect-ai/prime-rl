"""Shared dataclasses, type aliases, and protocols for the orchestrator.

Data carriers only — no behavior. Behavioral modules (dispatcher, sinks,
watcher, metrics, ckpt) import from here so they don't depend on each
other's implementation modules.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal, Protocol

import verifiers as vf

from prime_rl.transport import TrainingSample

# ── policy + checkpoint state ─────────────────────────────────────────────


@dataclass
class Policy:
    """Mutable shared view of the current policy. Passed by reference so
    observers (dispatcher, sinks) see new versions immediately."""

    version: int = 0
    model_name: str = ""


@dataclass
class Progress:
    """Persistent counters. ``step`` is the trainer-aligned step.
    ``last_eval_step_by_env`` records the highest step at which each eval
    env was triggered — read + written by ``EvalSource`` so resumes don't
    re-fire evals that already ran pre-checkpoint."""

    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0
    last_eval_step_by_env: dict[str, int] = field(default_factory=dict)


Kind = Literal["train", "eval"]


@dataclass
class InflightRollout:
    """Per-task scheduling state held by the dispatcher while a rollout (or
    a group, for group-scoring envs) is being generated. One entry per
    in-flight ``run_rollout`` / ``run_group`` task; translates into one or
    more ``FinishedRollout``\\ s on completion."""

    kind: Kind
    env_name: str
    group_id: uuid.UUID
    policy_version: int
    rollout_count: int  # number of inflight permits this task holds
    client_config: vf.ClientConfig | None = None
    off_policy_steps: int = 0  # bumped by ``on_new_version``
    eval_step: int | None = None


@dataclass
class GroupState:
    """Per-group scheduling state in the dispatcher. Completion accumulation
    happens in the sink — the dispatcher only tracks what's needed to keep
    dispatching the remaining rollouts + pin them to the same client (for
    prefix-cache hits)."""

    kind: Kind
    env_name: str
    example: dict
    rollouts_to_schedule: int
    target_rollouts: int  # total rollouts the group will emit to the sink
    emitted: int = 0
    eval_step: int | None = None
    pinned_client: vf.ClientConfig | None = None
    policy_version_at_start: int = 0


@dataclass
class FinishedRollout:
    """A completed rollout the sink receives. ``raw`` is the env's untouched
    ``vf.RolloutOutput``; prime-rl metadata lives on typed fields. Train vs
    eval is discriminated by ``isinstance(r, TrainRollout)`` /
    ``isinstance(r, EvalRollout)``.

    Failures (env errors, empty trajectories, task exceptions, off-policy
    cancellations) flow through with ``raw["error"]`` set; the sinks decide
    drop / partial-train policy.

    ``rollout_id`` is the only safe key for tracing one rollout through its
    lifecycle — ``(env_name, example_id)`` collides on re-sampling and
    ``group_id`` covers a whole group."""

    raw: vf.RolloutOutput
    env_name: str
    example_id: int | str
    group_id: uuid.UUID
    policy_version: int  # snapshot at dispatch
    off_policy_steps: int  # at completion
    rollout_id: uuid.UUID = field(default_factory=uuid.uuid4)

    @property
    def error(self) -> dict | None:
        return self.raw.get("error")

    @property
    def reward(self) -> float:
        return float(self.raw.get("reward", 0.0))

    @property
    def is_truncated(self) -> bool:
        return bool(self.raw.get("is_truncated", False))

    def to_dict(self) -> vf.RolloutOutput:
        """``raw`` + metadata merged into a single dict for I/O
        (``save_rollouts``, ``monitor.log_samples`` / ``log_eval_samples``).
        Returns a shallow copy; never mutates ``self.raw``."""
        out: vf.RolloutOutput = dict(self.raw)  # type: ignore[assignment]
        out["rollout_id"] = str(self.rollout_id)
        out["group_id"] = str(self.group_id)
        out["env_name"] = self.env_name
        out["example_id"] = self.example_id
        out["policy_version"] = self.policy_version
        out["off_policy_steps"] = self.off_policy_steps
        return out


@dataclass
class TrainRollout(FinishedRollout):
    """Train-only fields populated by ``TrainSink``: ``samples`` in
    ``process_rollout``, ``advantage`` / ``is_filtered`` / ``filter_results``
    in ``process_group`` / ``process_batch``."""

    samples: list[TrainingSample] = field(default_factory=list)
    advantage: float | None = None
    is_filtered: bool = False
    filter_results: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> vf.RolloutOutput:
        out = super().to_dict()
        if self.advantage is not None:
            out["advantage"] = self.advantage
        out["is_filtered"] = self.is_filtered
        out["filters"] = dict(self.filter_results)
        return out


@dataclass
class EvalRollout(FinishedRollout):
    """``eval_step`` is the policy version at which the epoch was triggered
    — the bucket key the sink groups by."""

    eval_step: int = 0

    def to_dict(self) -> vf.RolloutOutput:
        out = super().to_dict()
        out["eval_step"] = self.eval_step
        return out


@dataclass
class TrainBatchMetrics:
    """Per-batch aggregates that ``TrainSink.process_batch`` extracts from
    the cohort; the orchestrator hands this to ``MetricsBuilder.build``
    along with post-barrier timing scalars."""

    n_trainable: int
    num_prefill_tokens: int
    num_decode_tokens: int
    rollout_prefill_lens: list[int]
    rollout_decode_lens: list[int]
    samples_per_rollout: list[int]
    samples_shipped: int
    # Errored rollouts are dropped at the group level (don't reach
    # ``TrainBatch.rollouts``), so these counters surface the per-batch
    # error rate in the success log.
    arrivals_by_env: dict[str, int] = field(default_factory=dict)
    errors_by_env: dict[str, int] = field(default_factory=dict)


@dataclass
class TrainBatch:
    """- ``samples``: trainer-bound payload (post-filter survivors only).
    - ``rollouts``: full cohort kept for orchestrator-side I/O and metrics.
    - ``metrics``: typed counter view; wandb dict assembled at log time."""

    rollouts: list[TrainRollout]
    samples: list[TrainingSample]
    metrics: TrainBatchMetrics


@dataclass
class EvalBatchMetrics:
    """Typed per-batch metrics built by ``EvalSink.process_batch``. Final
    wandb dict derived via ``to_wandb_dict`` at log time."""

    n_rollouts: int
    n_cancelled: int
    n_errored: int
    valid_rate: float
    n_examples: int = 0
    group_size: int = 1
    reward_mean: float = 0.0
    completion_len_mean: float = 0.0
    completion_len_max: float = 0.0
    completion_len_min: float = 0.0
    truncation_rate: float = 0.0
    no_response_rate: float = 0.0
    num_turns_mean: float = 0.0
    pass_at_k: dict[str, float] = field(default_factory=dict)

    def to_wandb_dict(self, *, env_name: str, step: int) -> dict[str, float]:
        prefix = f"eval/{env_name}"
        out: dict[str, float] = {
            "step": float(step),
            f"{prefix}/n_rollouts": float(self.n_rollouts),
            f"{prefix}/cancelled_count": float(self.n_cancelled),
            f"{prefix}/errored_count": float(self.n_errored),
            f"{prefix}/valid_rate": self.valid_rate,
        }
        if self.n_examples > 0:
            out[f"{prefix}/n_examples"] = float(self.n_examples)
            out[f"{prefix}/reward/mean"] = self.reward_mean
            out[f"{prefix}/completion_len/mean"] = self.completion_len_mean
            out[f"{prefix}/completion_len/max"] = self.completion_len_max
            out[f"{prefix}/completion_len/min"] = self.completion_len_min
            out[f"{prefix}/is_truncated/mean"] = self.truncation_rate
            out[f"{prefix}/no_response/mean"] = self.no_response_rate
            out[f"{prefix}/num_turns/mean"] = self.num_turns_mean
            for k, v in self.pass_at_k.items():
                out[f"{prefix}/{k}"] = v
        return out


@dataclass
class EvalBatch:
    """One env's eval epoch. ``rollouts`` is the raw cohort for save / log;
    ``metrics`` is the typed per-env view built by
    ``EvalSink.process_batch``."""

    env_name: str
    step: int
    rollouts: list[EvalRollout]
    metrics: EvalBatchMetrics


class VersionObserver(Protocol):
    """Notified after each successful policy update. Walked synchronously
    by the watcher *after* mutating ``Policy``, so observers see the new
    version immediately."""

    async def on_new_version(self, step: int) -> None: ...
