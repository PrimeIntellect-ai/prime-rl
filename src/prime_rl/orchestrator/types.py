"""Shared dataclasses, type aliases, and protocols for the orchestrator.

Data carriers and small types live here so behavioral modules (dispatcher,
sinks, watcher, metrics, ckpt) can import without depending on each other's
implementation modules.

Sections (in dependency order; no module here imports another orchestrator
module):

- ``Policy``: the single mutable view of the current trainer weights.
- ``Progress``: persistent counters owned by the checkpoint manager.
- ``Kind``: dispatch primitive (the train/eval discriminator).
- ``Rollout`` / ``RolloutMeta`` / ``GroupState``: the dispatcher's in-flight
  bookkeeping + the atomic unit flowing on its output queue.
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
    successful weight update). The eval boundary is a strict function of
    ``step`` + ``eval.interval`` + ``skip_first_step``,
    so we don't track a separate ``last_eval_step``.
    """

    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0


# ── dispatcher primitives ─────────────────────────────────────────────────


Kind = Literal["train", "eval"]


@dataclass
class Rollout:
    """The atomic unit emitted by the dispatcher — one completed rollout.

    Invariant — every rollout the dispatcher acquires an inflight permit for
    eventually arrives at the corresponding sink exactly once, success or
    failure. Group/batch boundaries are sink-derived by counting arrivals up
    to ``group_size`` (and ``num_examples * group_size`` for eval epochs);
    the dispatcher does not stamp boundary flags because "the last rollout
    in a group" is whichever straggler happens to finish last — not knowable
    at dispatch time.

    Failures (env-reported errors, empty trajectories, task exceptions,
    off-policy cancellations) are carried via ``raw["error"]`` rather than
    silently dropped. Sinks check that field to decide drop / partial-train.

    ``group_id`` is the dispatcher's UUID for the dispatched group this
    rollout belongs to. The sink uses it as the ``pending_groups`` key —
    ``(env_name, example_id)`` isn't unique because the same example can
    be re-sampled while an earlier group is still in flight, especially on
    small datasets. ``env_name`` / ``example_id`` are still available on
    ``raw["env_name"]`` and ``raw["example_id"]`` for logging / aggregation.

    ``policy_version`` is the snapshot at dispatch time; the train sink
    uses it for per-rollout off-policy metrics. For eval rollouts, the
    dispatcher also stamps ``raw["_eval_step"]`` with the policy version at
    which the eval epoch was triggered (used by the eval sink to bucket
    groups into epochs).
    """

    kind: Kind
    group_id: uuid.UUID
    raw: vf.RolloutOutput
    policy_version: int


@dataclass
class RolloutMeta:
    """Per-task bookkeeping. One entry per in-flight ``run_rollout`` / ``run_group``."""

    kind: Kind
    env_name: str
    group_id: uuid.UUID
    policy_version: int
    rollout_count: int  # number of inflight permits this task holds
    client_config: vf.ClientConfig | None = None
    off_policy_steps: int = 0  # incremented on every ``on_new_version``; train only
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

    rollouts: list[vf.RolloutOutput]
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
            out[f"{prefix}/avg@{self.group_size}"] = self.reward_mean
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
    rollouts: list[vf.RolloutOutput]
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
