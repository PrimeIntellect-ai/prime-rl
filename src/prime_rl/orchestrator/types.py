"""Shared dataclasses for the orchestrator. Data carriers only; no behavior."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, fields
from typing import Literal, Protocol

import verifiers as vf

from prime_rl.transport import TrainingSample


@dataclass
class Policy:
    """Mutable shared view of the policy. Passed by reference so observers
    see new versions immediately."""

    version: int = 0
    model_name: str = ""


@dataclass
class Progress:
    """Persistent counters; ``step`` is the trainer-aligned step.
    ``last_eval_step_by_env`` is read/written by ``EvalSource`` to suppress
    duplicate eval epochs on resume."""

    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0
    last_eval_step_by_env: dict[str, int] = field(default_factory=dict)


Kind = Literal["train", "eval"]


@dataclass
class InflightRollout:
    """Per-task scheduling state in the dispatcher; one entry per in-flight
    ``run_rollout`` / ``run_group`` task."""

    kind: Kind
    env_name: str
    group_id: uuid.UUID
    policy_version: int
    rollout_count: int
    client_config: vf.ClientConfig | None = None
    off_policy_steps: int = 0
    eval_step: int | None = None


@dataclass
class GroupState:
    """Per-group dispatcher state: what's left to schedule + the pinned
    client (for prefix-cache hits)."""

    kind: Kind
    env_name: str
    example: dict
    rollouts_to_schedule: int
    target_rollouts: int
    emitted: int = 0
    eval_step: int | None = None
    pinned_client: vf.ClientConfig | None = None
    policy_version_at_start: int = 0


@dataclass
class FinishedRollout:
    """A completed rollout the sink receives. ``raw`` is the env's untouched
    ``vf.RolloutOutput``; prime-rl metadata lives on typed fields. Train vs
    eval is discriminated via ``isinstance``. ``rollout_id`` is the only
    safe key for tracing one rollout — ``(env_name, example_id)`` collides
    on re-sampling and ``group_id`` covers a whole group."""

    raw: vf.RolloutOutput
    env_name: str
    example_id: int | str
    group_id: uuid.UUID
    policy_version: int
    off_policy_steps: int
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
        """``raw`` + metadata merged for I/O (``save_rollouts``,
        ``monitor.log_samples``). Shallow copy; never mutates ``self.raw``."""
        out: vf.RolloutOutput = dict(self.raw)  # type: ignore[assignment]
        for f in fields(self):
            if f.name in ("raw", "samples"):
                continue
            val = getattr(self, f.name)
            if f.name == "filter_results":
                out["filters"] = dict(val)
                continue
            out[f.name] = str(val) if isinstance(val, uuid.UUID) else val
        return out


@dataclass
class TrainRollout(FinishedRollout):
    samples: list[TrainingSample] = field(default_factory=list)
    advantage: float | None = None
    is_filtered: bool = False
    filter_results: dict[str, bool] = field(default_factory=dict)


@dataclass
class EvalRollout(FinishedRollout):
    eval_step: int = 0


@dataclass
class TrainBatchMetrics:
    """Per-batch aggregates from ``TrainSink.process_batch``; consumed by
    ``MetricsBuilder.build``. ``arrivals_by_env`` / ``errors_by_env`` count
    rollouts at the sink (errored ones get dropped at the group level before
    reaching ``TrainBatch.rollouts``)."""

    n_trainable: int
    num_prefill_tokens: int
    num_decode_tokens: int
    rollout_prefill_lens: list[int]
    rollout_decode_lens: list[int]
    samples_per_rollout: list[int]
    samples_shipped: int
    arrivals_by_env: dict[str, int] = field(default_factory=dict)
    errors_by_env: dict[str, int] = field(default_factory=dict)


@dataclass
class TrainBatch:
    """``samples`` is the trainer-bound payload (post-filter survivors);
    ``rollouts`` is the full cohort kept for orchestrator-side I/O."""

    rollouts: list[TrainRollout]
    samples: list[TrainingSample]
    metrics: TrainBatchMetrics


@dataclass
class EvalBatchMetrics:
    """Typed per-batch metrics from ``EvalSink.process_batch``. Final wandb
    dict derived via ``to_wandb_dict`` at log time."""

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
    """One env's eval epoch. ``metrics`` is the typed view from
    ``EvalSink.process_batch``."""

    env_name: str
    step: int
    rollouts: list[EvalRollout]
    metrics: EvalBatchMetrics


class VersionObserver(Protocol):
    """Notified after each policy update; walked by the watcher after it
    mutates ``Policy``."""

    async def on_new_version(self, step: int) -> None: ...
