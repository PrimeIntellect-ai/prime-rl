"""Shared dataclasses for the orchestrator. Data carriers only; no behavior."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Generic, Literal, Protocol

import verifiers.v1 as vf
from pydantic import ConfigDict, Field
from verifiers.v1.task import TaskT

from prime_rl.transport import TrainingSample


@dataclass
class Policy:
    """Mutable shared view of the policy. Passed by reference so observers
    see new versions immediately."""

    version: int = 0
    model_name: str = ""


@dataclass
class Progress:
    """Persistent counters; ``step`` is the trainer-aligned step."""

    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0


RolloutKind = Literal["train", "eval"]


@dataclass
class InflightRollout:
    """Per-task scheduling state in the dispatcher; one entry per in-flight
    ``run_rollout`` / ``run_group`` task."""

    kind: RolloutKind
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

    kind: RolloutKind
    env_name: str
    task_idx: int
    rollouts_to_schedule: int
    target_rollouts: int
    emitted: int = 0
    eval_step: int | None = None
    pinned_client: vf.ClientConfig | None = None
    policy_version_at_start: int = 0


class Rollout(vf.Trace[TaskT], Generic[TaskT]):
    """A completed rollout. The env's typed ``vf.Trace`` is the rollout; prime-rl
    scheduling and training metadata is attached as excluded pydantic fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: RolloutKind = Field(default="train", exclude=True)
    env_name: str = Field(default="", exclude=True)
    group_id: uuid.UUID = Field(default_factory=uuid.uuid4, exclude=True)
    policy_version: int = Field(default=0, exclude=True)
    off_policy_steps: int = Field(default=0, exclude=True)
    samples: list[TrainingSample] = Field(default_factory=list, exclude=True)
    advantages: list[float] | None = Field(default=None, exclude=True)
    is_filtered: bool = Field(default=False, exclude=True)
    filter_results: dict[str, bool] = Field(default_factory=dict, exclude=True)
    eval_step: int | None = Field(default=None, exclude=True)

    @property
    def advantage(self) -> float | None:
        if not self.advantages:
            return None
        return sum(self.advantages) / len(self.advantages)

    def to_dict(self) -> dict:
        out = self.model_dump(mode="json")
        out.update(
            {
                "kind": self.kind,
                "env_name": self.env_name,
                "group_id": str(self.group_id),
                "policy_version": self.policy_version,
                "off_policy_steps": self.off_policy_steps,
                "filters": dict(self.filter_results),
            }
        )
        if self.eval_step is not None:
            out["eval_step"] = self.eval_step
        if self.advantage is not None:
            out["advantage"] = self.advantage
        return out


@dataclass(frozen=True)
class RolloutView:
    """A finalized rollout as a writable handle — the single currency the
    scoring hooks operate on. Exposes what the env produced (``raw``), the
    samples interleaving built (``samples``, carrying ``obs_spans``), and the
    rollout's identity/reward; credit is written through
    :meth:`assign_advantages`, which spreads over the samples' completion
    tokens. Deliberately does *not* expose pipeline-internal lifecycle fields
    (``is_filtered``, ``filter_results``, ``group_id``) or not-yet-assigned
    credit (``advantages``) — a hook can only touch what is valid at its
    stage."""

    _rollout: Rollout

    @property
    def trace(self) -> Rollout:
        return self._rollout

    @property
    def samples(self) -> list[TrainingSample]:
        return self._rollout.samples

    @property
    def reward(self) -> float:
        return self._rollout.reward

    @property
    def env_name(self) -> str:
        return self._rollout.env_name

    @property
    def example_id(self) -> int | str:
        return self._rollout.task.idx

    @property
    def completion_len(self) -> int:
        return self._rollout.completion_len

    @property
    def tool_response_len(self) -> int:
        return sum(len(node.token_ids) for node in self._rollout.nodes if getattr(node.message, "role", None) == "tool")

    @property
    def num_turns(self) -> int:
        return self._rollout.num_turns

    def assign_advantages(self, values: float | list[float]) -> None:
        """Write the rl advantage stream: a scalar broadcast over the
        rollout's completion tokens, or a per-token list aligned to them
        (concatenated across samples in step order). Prompt positions are
        padded at stamping; a rollout never assigned ships no advantage
        stream."""
        total = sum(len(sample.completion_ids) for sample in self._rollout.samples)
        if isinstance(values, (int, float)):
            self._rollout.advantages = [float(values)] * total
            return
        if len(values) != total:
            raise ValueError(
                f"per-token advantages must align with the rollout's completion tokens: "
                f"got {len(values)}, expected {total} (env '{self._rollout.env_name}')."
            )
        self._rollout.advantages = [float(v) for v in values]


@dataclass
class TrainBatchMetrics:
    """Per-batch aggregates from ``TrainSink.process_batch``; consumed by
    ``MetricsBuilder.build``. ``arrivals_by_env`` / ``errors_by_env`` count
    rollouts at the sink."""

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

    rollouts: list[Rollout]
    samples: list[TrainingSample]
    metrics: TrainBatchMetrics


@dataclass
class EvalBatchMetrics:
    """Typed per-batch metrics from ``EvalSink.process_batch``. Final wandb
    dict derived via ``to_wandb_dict`` at log time."""

    n_rollouts: int
    n_cancelled: int
    n_errored: int
    n_examples: int = 0
    group_size: int = 1
    reward_mean: float = 0.0
    completion_len_mean: float = 0.0
    completion_len_max: float = 0.0
    completion_len_min: float = 0.0
    truncation_rate: float = 0.0
    no_response_rate: float = 0.0
    num_turns_mean: float = 0.0
    num_turns_min: float = 0.0
    num_turns_max: float = 0.0
    pass_at_k: dict[str, float] = field(default_factory=dict)

    def to_wandb_dict(self, *, env_name: str, step: int) -> dict[str, float]:
        prefix = f"eval/{env_name}"
        out: dict[str, float] = {
            "step": float(step),
            f"{prefix}/cancelled_count": float(self.n_cancelled),
            f"{prefix}/errored_count": float(self.n_errored),
        }
        if self.n_examples > 0:
            out[f"{prefix}/avg@{self.group_size}"] = self.reward_mean
            out[f"{prefix}/completion_len/mean"] = self.completion_len_mean
            out[f"{prefix}/completion_len/max"] = self.completion_len_max
            out[f"{prefix}/completion_len/min"] = self.completion_len_min
            out[f"{prefix}/is_truncated/mean"] = self.truncation_rate
            out[f"{prefix}/no_response/mean"] = self.no_response_rate
            out[f"{prefix}/num_turns/mean"] = self.num_turns_mean
            out[f"{prefix}/num_turns/min"] = self.num_turns_min
            out[f"{prefix}/num_turns/max"] = self.num_turns_max
            for k, v in self.pass_at_k.items():
                out[f"{prefix}/{k}"] = v
        return out


@dataclass
class EvalBatch:
    """One env's eval epoch. ``metrics`` is the typed view from
    ``EvalSink.process_batch``."""

    env_name: str
    step: int
    rollouts: list[Rollout]
    metrics: EvalBatchMetrics


class VersionObserver(Protocol):
    """Notified after each policy update; walked by the watcher after it
    mutates ``Policy``."""

    async def on_new_version(self, step: int) -> None: ...
