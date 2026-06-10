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
    example: dict
    rollouts_to_schedule: int
    target_rollouts: int
    emitted: int = 0
    eval_step: int | None = None
    pinned_client: vf.ClientConfig | None = None
    policy_version_at_start: int = 0
    dispatch_ids: list[str] = field(default_factory=list)
    scheduled: int = 0


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
    source_rollout_id: uuid.UUID | None = None


@dataclass
class EvalRollout(FinishedRollout):
    eval_step: int = 0


@dataclass
class RAEStats:
    """Instrumentation snapshot of the multi-agent advantage path (RAE).

    Derived strictly from the estimator's inputs/outputs (rewards in,
    baseline values out) so the estimator implementation can be swapped
    without touching this. ``baseline_abs_delta_sum`` accumulates per-update
    baseline movement ``|after - before|``, using the estimator's effective
    prior (0.0) as the before-value for cold keys. A "cold" update is a
    baseline key never seen before. Window semantics match the sink's other
    counters: everything since the last ship, regardless of which batch the
    affected rows land in."""

    updates: int = 0
    cold_updates: int = 0
    baseline_abs_delta_sum: float = 0.0
    baseline_sum_by_member: dict[str, float] = field(default_factory=dict)
    updates_by_member: dict[str, int] = field(default_factory=dict)
    baseline_keys_total: int = 0

    def merge(self, other: RAEStats) -> None:
        self.updates += other.updates
        self.cold_updates += other.cold_updates
        self.baseline_abs_delta_sum += other.baseline_abs_delta_sum
        for member_id, baseline_sum in other.baseline_sum_by_member.items():
            self.baseline_sum_by_member[member_id] = self.baseline_sum_by_member.get(member_id, 0.0) + baseline_sum
        for member_id, count in other.updates_by_member.items():
            self.updates_by_member[member_id] = self.updates_by_member.get(member_id, 0) + count
        # The baseline table only grows; the freshest snapshot is the largest.
        self.baseline_keys_total = max(self.baseline_keys_total, other.baseline_keys_total)


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
    rae_stats: RAEStats | None = None


@dataclass
class TrainBatch:
    """``samples`` is the trainer-bound payload (post-filter survivors);
    ``rollouts`` is the trainer-unit cohort used for train metrics."""

    rollouts: list[TrainRollout]
    samples: list[TrainingSample]
    metrics: TrainBatchMetrics
    episode_rollouts: list[TrainRollout] | None = None


def rollouts_for_logging(batch: TrainBatch) -> list[TrainRollout]:
    """Use source episodes for multi-agent member rollouts, otherwise keep
    the trainer-unit rollout. Mixed batches preserve batch order and dedupe
    repeated members from the same episode."""
    if batch.episode_rollouts is None:
        return batch.rollouts

    episodes_by_id = {r.rollout_id: r for r in batch.episode_rollouts}
    seen_episodes: set[uuid.UUID] = set()
    out: list[TrainRollout] = []
    for rollout in batch.rollouts:
        source_id = rollout.source_rollout_id
        if source_id is None or source_id not in episodes_by_id:
            out.append(rollout)
            continue
        if source_id in seen_episodes:
            continue
        seen_episodes.add(source_id)
        out.append(episodes_by_id[source_id])
    return out


@dataclass
class EvalBatchMetrics:
    """Typed per-batch metrics from ``EvalSink.process_batch``. Final wandb
    dict derived via ``to_wandb_dict`` at log time.

    ``mar_metrics`` / ``winner_counts`` form the multi-agent MARScore panel
    (per-member rewards + metrics, judge-winner distribution); both empty for
    single-agent envs. ``inert_scalar`` is the env's structural declaration
    that its episode scalar is identically zero by construction (symmetric
    zero-sum debate, ``truth_member=None`` — see ``episode_scalar_is_inert``):
    ``avg@k`` / ``pass@k`` would be constant-zero panels there, so
    ``to_wandb_dict`` omits those keys and the MARScore panel carries the
    signal instead. Envs with a live scalar — including ``truth_member``
    debate packs whose batch honestly scored 0.0 — keep them."""

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
    mar_metrics: dict[str, float] = field(default_factory=dict)
    winner_counts: dict[str, int] = field(default_factory=dict)
    inert_scalar: bool = False
    # Key-agnostic means of numeric env-emitted rollout metrics
    # (``raw["metrics"]``), e.g. debate diagnostics.
    env_metrics: dict[str, float] = field(default_factory=dict)

    def to_wandb_dict(self, *, env_name: str, step: int) -> dict[str, float]:
        prefix = f"eval/{env_name}"
        out: dict[str, float] = {
            "step": float(step),
            f"{prefix}/cancelled_count": float(self.n_cancelled),
            f"{prefix}/errored_count": float(self.n_errored),
        }
        if self.n_examples > 0:
            if not self.inert_scalar:
                out[f"{prefix}/avg@{self.group_size}"] = self.reward_mean
                for k, v in self.pass_at_k.items():
                    out[f"{prefix}/{k}"] = v
            out[f"{prefix}/completion_len/mean"] = self.completion_len_mean
            out[f"{prefix}/completion_len/max"] = self.completion_len_max
            out[f"{prefix}/completion_len/min"] = self.completion_len_min
            out[f"{prefix}/is_truncated/mean"] = self.truncation_rate
            out[f"{prefix}/no_response/mean"] = self.no_response_rate
            out[f"{prefix}/num_turns/mean"] = self.num_turns_mean
            out[f"{prefix}/num_turns/min"] = self.num_turns_min
            out[f"{prefix}/num_turns/max"] = self.num_turns_max
            for k, v in self.mar_metrics.items():
                out[f"{prefix}/mar/{k}"] = v
            n_winners = sum(self.winner_counts.values())
            for value, count in sorted(self.winner_counts.items()):
                out[f"{prefix}/winner_count/{value}"] = float(count)
                out[f"{prefix}/winner_share/{value}"] = count / n_winners
            for key, value in self.env_metrics.items():
                out[f"{prefix}/metrics/{key}"] = value
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
