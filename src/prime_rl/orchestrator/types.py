"""Shared graph-native data carriers for the orchestrator."""

from __future__ import annotations

import traceback
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import verifiers.v1 as vf
from pydantic import ConfigDict, Field, SerializeAsAny

from prime_rl.transport import TrainingSample

if TYPE_CHECKING:
    from prime_rl.orchestrator.metrics import EvalGraphs, TrainGraphs


@dataclass
class Policy:
    """Mutable shared view of the policy. Passed by reference so observers
    see new versions immediately."""

    version: int = 0
    model_name: str = ""


@dataclass
class Progress:
    """Persistent counters; ``step`` is the trainer-aligned step (1-indexed)."""

    step: int = 1
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0


GraphKind = Literal["train", "eval"]


@dataclass
class InflightGraph:
    """Scheduling metadata for one in-flight topology invocation."""

    kind: GraphKind
    env_name: str
    group_id: uuid.UUID
    policy_version: int
    client_config: vf.ClientConfig | None = None
    off_policy_steps: int = 0
    eval_step: int | None = None


@dataclass
class GroupState:
    """Per-group dispatcher state: what's left to schedule + the pinned
    client (for prefix-cache hits)."""

    kind: GraphKind
    env_name: str
    task_idx: int
    graphs_to_schedule: int
    target_graphs: int
    emitted: int = 0
    eval_step: int | None = None
    pinned_client: vf.ClientConfig | None = None
    policy_version_at_start: int = 0


class TrainingTrace(vf.WireTrace):
    """A graph trace with Prime's transient training state."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # ``samples`` holds msgspec structs

    samples: list[TrainingSample] = Field(default_factory=list, exclude=True)
    # Per-token rl advantage stream, full-length-N (= len(token_ids)) per
    # sample, concatenated across the trace's samples in order; 0.0 on
    # non-trainable positions. None = no credit assigned (advantage-based
    # filters skip it; the wire ships no advantage stream).
    advantages: list[float] | None = Field(default=None, exclude=True)

    def assign_advantages(self, values: float | list[float]) -> None:
        """Write the rl advantage stream: a scalar broadcast over the
        trace's trainable (mask-True) tokens (0.0 elsewhere), or a per-token
        list already aligned full-length to the samples' concatenated
        ``token_ids``. A trace never assigned ships no advantage stream."""
        total = sum(len(sample.token_ids) for sample in self.samples)
        if isinstance(values, (int, float)):
            self.advantages = [
                float(values) if trainable else 0.0 for sample in self.samples for trainable in sample.mask
            ]
            return
        if len(values) != total:
            raise ValueError(
                f"per-token advantages must align with the trace's tokens: "
                f"got {len(values)}, expected {total} (trace '{self.id}')."
            )
        self.advantages = [float(v) for v in values]

    def scalar_advantage(self) -> float | None:
        """Scalar view of the per-token advantage stream for monitoring: the
        mean over assigned (non-zero) positions — exact for the uniform GRPO
        case, 0.0 for a zero-advantage group, None when no credit was assigned."""
        if not self.advantages:
            return None
        nonzero = [a for a in self.advantages if a != 0.0]
        return sum(nonzero) / len(nonzero) if nonzero else 0.0

    @property
    def is_trainable(self) -> bool:
        """Whether the trace carries a training signal — a nonzero advantage on some token. A
        uniform-reward GRPO group (all-zero advantages) or an unscored trace has no gradient."""
        return bool(self.advantages) and any(a != 0.0 for a in self.advantages)


class AgentGraph(vf.AgentGraph):
    """One topology invocation plus Prime's transient orchestration state.

    The first training contract intentionally supports exactly one trainable trace. The
    graph remains the algorithm and persistence unit; supporting multiple trainable traces
    later only requires algorithms that assign credit across those traces.
    """

    traces: list[SerializeAsAny[TrainingTrace]] = Field(default_factory=list)
    kind: GraphKind = Field(default="train", exclude=True)
    env_name: str = Field(default="", exclude=True)
    group_id: uuid.UUID = Field(default_factory=uuid.uuid4, exclude=True)
    policy_version: int = Field(default=0, exclude=True)
    off_policy_steps: int = Field(default=0, exclude=True)
    is_filtered: bool = Field(default=False, exclude=True)
    filter_results: dict[str, bool] = Field(default_factory=dict, exclude=True)
    eval_step: int | None = Field(default=None, exclude=True)

    @classmethod
    def from_wire(cls, graph: vf.AgentGraph) -> "AgentGraph":
        return cls.model_validate(
            {
                **graph.model_dump(exclude={"task", "traces"}),
                "task": graph.task,
                "traces": [TrainingTrace.model_validate(trace.model_dump(mode="python")) for trace in graph.traces],
            }
        )

    @property
    def trainable_traces(self) -> list[TrainingTrace]:
        return [trace for trace in self.traces if trace.trainable]

    @property
    def training_trace(self) -> TrainingTrace:
        traces = self.trainable_traces
        if len(traces) != 1:
            raise ValueError(
                f"topology graph {self.id!r} must contain exactly one trainable trace; found {len(traces)}"
            )
        return traces[0]

    @property
    def training_trace_or_none(self) -> TrainingTrace | None:
        traces = self.trainable_traces
        return traces[0] if len(traces) == 1 else None

    @property
    def has_error(self) -> bool:
        traces = self.trainable_traces
        return self.error is not None or (bool(traces) and all(trace.has_error for trace in traces))

    @property
    def failure(self) -> vf.Error | None:
        if self.error is not None:
            return self.error
        traces = self.trainable_traces
        failed = [trace.error for trace in traces if trace.error is not None]
        return failed[0] if failed and len(failed) == len(traces) else None

    def capture_error(self, error: Exception) -> None:
        self.error = vf.Error(
            type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
        )

    def to_record(self) -> dict:
        return super().to_record() | {
            "kind": self.kind,
            "env_name": self.env_name,
            "group_id": str(self.group_id),
            "policy_version": self.policy_version,
            "eval_step": self.eval_step,
        }

    @property
    def reward(self) -> float:
        trace = self.training_trace_or_none
        return trace.reward if trace is not None else 0.0

    @property
    def rewards(self) -> dict[str, float]:
        trace = self.training_trace_or_none
        return trace.rewards if trace is not None else {}

    @property
    def metrics(self) -> dict[str, float]:
        trace = self.training_trace_or_none
        return trace.metrics if trace is not None else {}

    @property
    def timing(self) -> vf.Timing:
        trace = self.training_trace_or_none
        return trace.timing if trace is not None else vf.Timing()

    @property
    def num_input_tokens(self) -> int:
        return sum(trace.num_input_tokens for trace in self.trainable_traces)

    @property
    def num_output_tokens(self) -> int:
        return sum(trace.num_output_tokens for trace in self.trainable_traces)

    @property
    def num_total_tokens(self) -> int:
        return sum(trace.num_total_tokens for trace in self.trainable_traces)

    @property
    def num_turns(self) -> int:
        return sum(trace.num_turns for trace in self.trainable_traces)

    @property
    def num_branches(self) -> int:
        return sum(trace.num_branches for trace in self.trainable_traces)

    @property
    def is_truncated(self) -> bool:
        return any(trace.is_truncated for trace in self.trainable_traces)

    @property
    def is_completed(self) -> bool:
        traces = self.trainable_traces
        return bool(traces) and all(trace.is_completed for trace in traces)

    @property
    def stop_condition(self) -> str | None:
        trace = self.training_trace_or_none
        return trace.stop_condition if trace is not None else None

    @property
    def is_trainable(self) -> bool:
        return any(trace.is_trainable for trace in self.trainable_traces)

    def scalar_advantage(self) -> float | None:
        trace = self.training_trace_or_none
        return trace.scalar_advantage() if trace is not None else None


@dataclass
class TrainBatch:
    """``graphs`` is the full arrival window since the last ship (errored + filtered included; its
    ``.effective`` / ``.metrics`` views drive logging). ``samples`` is the trainer-bound payload (the
    shipped cohort's post-filter survivors) — an empty list means nothing ships, which would stall the
    trainer. Trainable counts derive from ``graphs`` and token totals from
    ``samples``, so neither is carried as a field."""

    graphs: TrainGraphs
    samples: list[TrainingSample]


@dataclass
class EvalBatch:
    """One env's eval epoch. ``graphs`` is the full returned cohort (errored included); its
    ``.effective`` / ``.metrics`` views drive logging."""

    env_name: str
    step: int
    graphs: EvalGraphs


class VersionObserver(Protocol):
    """Notified around each policy update; walked by the watcher.

    ``on_version_pending`` fires *before* the inference engines are paused for
    the weight update; ``on_new_version`` fires *after* the new weights are live
    and ``Policy`` has been mutated."""

    async def on_version_pending(self, step: int) -> None: ...

    async def on_new_version(self, step: int) -> None: ...
