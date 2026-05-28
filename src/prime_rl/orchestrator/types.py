"""Shared dataclasses, type aliases, and protocols for the orchestrator.

Data carriers and small types live here so behavioral modules (dispatcher,
sinks, watcher, metrics, ckpt) can import without depending on each other's
implementation modules.

Sections (in dependency order; no module here imports another orchestrator
module):

- ``Policy``: the single mutable view of the current trainer weights.
- ``Progress``: persistent counters owned by the checkpoint manager.
- ``Kind``: dispatch primitive (the train/eval discriminator).
- ``InflightRollout`` / ``GroupState``: the dispatcher's per-task scheduling
  state.
- ``FinishedRollout`` → ``TrainRollout`` / ``EvalRollout``: the atomic units
  flowing on the dispatcher's output queue. Each wraps the raw
  ``vf.RolloutOutput`` (env-produced data; never mutated by prime-rl) +
  prime-rl bookkeeping on typed fields.
- ``TrainBatch`` / ``EvalBatch`` and their ``TrainBatchMetrics`` /
  ``EvalBatchMetrics``: per-batch payloads the sinks return to the
  orchestrator.
- ``VersionObserver``: the watcher → dispatcher notification interface.
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
    """Mutable shared view of the current policy.

    The ``WeightWatcher`` writes ``version`` (and ``model_name`` after a LoRA
    swap) when a new checkpoint becomes available; the ``RolloutDispatcher``
    and all in-flight rollout meta read these fields at dispatch time. Passed
    by reference — never copied — so observers see new versions immediately.
    """

    version: int = 0
    model_name: str = ""


@dataclass
class Progress:
    """Persistent counters for the orchestrator.

    ``step`` is the trainer-aligned step (== ``policy.version`` after every
    successful weight update).

    ``last_eval_step_by_env`` records, per eval env name, the highest
    ``step`` at which an eval epoch has been triggered. The
    ``EvalSource`` reads + writes this dict so that on resume from a
    checkpoint, we don't re-fire evals that already ran pre-checkpoint
    (interval-aligned envs would otherwise duplicate at the resume step).
    """

    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0
    last_eval_step_by_env: dict[str, int] = field(default_factory=dict)


# ── dispatcher primitives ─────────────────────────────────────────────────


Kind = Literal["train", "eval"]


@dataclass
class InflightRollout:
    """Per-task scheduling state held by the dispatcher while a rollout (or
    a group of rollouts, for group-scoring envs) is being generated.

    One entry per in-flight ``run_rollout`` / ``run_group`` task.
    Translates into one or more ``FinishedRollout``\\ s on completion.
    """

    kind: Kind
    env_name: str
    group_id: uuid.UUID
    policy_version: int
    rollout_count: int  # number of inflight permits this task holds
    client_config: vf.ClientConfig | None = None
    off_policy_steps: int = 0  # incremented on every ``on_new_version``
    eval_step: int | None = None


@dataclass
class GroupState:
    """Per-group scheduling state for the dispatcher.

    Tracks only what's needed to keep dispatching the remaining rollouts of
    a group and to keep them pinned to the same inference client (for prefix-
    cache hits). Completion accumulation lives in the sink, not here — each
    finished rollout is emitted immediately by ``handle_completed_rollout``.

    For group-scoring envs ``rollouts_to_schedule`` collapses to 0 after the
    single ``run_group`` task is queued; otherwise it's decremented per
    rollout. The dispatcher drops the entry from ``self.groups`` once every
    member has been emitted.
    """

    kind: Kind
    env_name: str
    example: dict
    rollouts_to_schedule: int
    target_rollouts: int  # total rollouts the group will emit to the sink
    emitted: int = 0  # # of rollouts emitted to ``out_q`` so far
    eval_step: int | None = None
    pinned_client: vf.ClientConfig | None = None
    policy_version_at_start: int = 0


# ── finished rollouts (what the dispatcher emits) ─────────────────────────


@dataclass
class FinishedRollout:
    """A completed rollout the sink receives. ``raw`` is the env's untouched
    ``vf.RolloutOutput``; prime-rl bookkeeping lives on typed fields directly
    on this dataclass (not stamped into ``raw``). Discriminate ``train`` vs
    ``eval`` by ``isinstance(r, TrainRollout)`` / ``isinstance(r, EvalRollout)``.

    Invariant — every rollout the dispatcher acquires an inflight permit for
    eventually arrives at the corresponding sink exactly once, success or
    failure. Failures (env-reported errors, empty trajectories, task
    exceptions, off-policy cancellations) flow through with ``raw["error"]``
    set; the sinks decide drop / partial-train policy.

    ``rollout_id`` is a unique per-rollout UUID generated at construction —
    the only safe key for referencing this exact rollout through its
    lifecycle (dispatch → sink → batch → JSONL on disk). Neither
    ``(env_name, example_id)`` nor ``group_id`` is unique: the same example
    can be re-sampled as multiple groups, and a group contains
    ``group_size`` rollouts. ``group_id`` is the dispatcher's UUID for the
    group this rollout belongs to; it's how the sink groups rollouts for
    GRPO advantages.
    """

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
        """Materialize ``raw`` + prime-rl metadata into a single dict for I/O
        boundaries (``save_rollouts``, ``monitor.log_samples`` /
        ``log_eval_samples``). Returns a shallow copy of ``raw`` with
        metadata merged in — never mutates ``self.raw``."""
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
    """Train-side rollout. Train-only fields are populated by ``TrainSink``:
    ``samples`` in ``process_rollout`` (after tokenization), ``advantage`` +
    ``is_filtered`` + ``filter_results`` in ``process_group`` /
    ``process_batch``."""

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
    """Eval-side rollout. Carries ``eval_step`` (the policy version at which
    the eval epoch was triggered — the bucket key the sink groups by)."""

    eval_step: int = 0

    def to_dict(self) -> vf.RolloutOutput:
        out = super().to_dict()
        out["eval_step"] = self.eval_step
        return out


# ── sink payloads ─────────────────────────────────────────────────────────


@dataclass
class TrainBatchMetrics:
    """Per-batch counters/aggregates that ``TrainSink.process_batch``
    extracts from the rollout cohort. The orchestrator passes this to
    ``MetricsBuilder.build`` (along with post-barrier timing scalars) to
    assemble the wandb dict — keeps the heavy per-rollout walk out of the
    log-time critical path."""

    n_trainable: int
    num_prefill_tokens: int
    num_decode_tokens: int
    rollout_prefill_lens: list[int]
    rollout_decode_lens: list[int]
    samples_per_rollout: list[int]
    samples_shipped: int
    # Per-env arrival/error counts accumulated by the sink between ships.
    # Errored rollouts are dropped at the group level (they don't reach
    # ``TrainBatch.rollouts``), so we need a separate counter to surface
    # the per-batch error rate in the success log.
    arrivals_by_env: dict[str, int] = field(default_factory=dict)
    errors_by_env: dict[str, int] = field(default_factory=dict)


@dataclass
class TrainBatch:
    """One training batch ready to ship.

    - ``samples`` is the trainer-bound payload (post-filter survivors only)
      that goes to ``sender.send``.
    - ``rollouts`` is the full cohort (including post-filter dropouts) kept
      for orchestrator-side I/O (``save_rollouts``, ``log_samples``,
      ``log_distributions``, ``offload_images_to_disk``) and as input to
      ``MetricsBuilder``.
    - ``metrics`` is the extracted per-batch counter view (no derived wandb
      dict here — that's the orchestrator's job at log time once it knows
      step_time / teacher_logprobs_time / save_ckpt_time).
    """

    rollouts: list[TrainRollout]
    samples: list[TrainingSample]
    metrics: TrainBatchMetrics


@dataclass
class EvalBatchMetrics:
    """Per-batch typed metrics built by ``EvalSink.process_batch``. Final
    wandb dict is derived via ``to_wandb_dict`` at log time, keeping the
    dataclass typed while still preserving wandb-friendly keys downstream."""

    n_rollouts: int
    n_cancelled: int
    n_errored: int
    valid_rate: float
    # Survivor-derived. Zeros when no valid rollouts (all errored).
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
    """One env's eval epoch. ``rollouts`` is the raw cohort kept for
    ``save_rollouts`` + ``monitor.log_eval_samples``; ``metrics`` is the
    typed per-env metrics view built by ``EvalSink.process_batch``. The
    orchestrator turns ``metrics`` into the wandb dict at log time via
    ``metrics.to_wandb_dict(env_name=…, step=…)``."""

    env_name: str
    step: int
    rollouts: list[EvalRollout]
    metrics: EvalBatchMetrics


# ── watcher → observer interface ──────────────────────────────────────────


class VersionObserver(Protocol):
    """Notified after each successful policy update.

    The watcher walks the observer list synchronously in registration order
    *after* mutating ``Policy``. Observers see the freshly-installed version
    immediately and can use the call to invalidate caches, cancel stale
    in-flight work, or trigger evals.
    """

    async def on_new_version(self, step: int) -> None: ...
