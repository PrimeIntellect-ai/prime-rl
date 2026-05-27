"""Shared dataclasses, type aliases, and protocols for the orchestrator.

Data carriers and small types live here so behavioral modules (dispatcher,
sinks, watcher, metrics, ckpt) can import without depending on each other's
implementation modules.

Sections (in dependency order; no module here imports another orchestrator
module):

- ``Policy``: the single mutable view of the current trainer weights.
- ``Progress``: persistent counters owned by the checkpoint manager.
- ``Kind`` / ``SchedMode``: dispatch primitives.
- ``Rollout`` / ``RolloutMeta`` / ``GroupState``: the dispatcher's in-flight
  bookkeeping + the atomic unit flowing on its output queue.
- ``TrainBatch`` / ``EvalBatch`` / ``ProcessResult``: per-batch payloads
  the sinks return to the orchestrator.
- ``VersionObserver``: the watcher → dispatcher notification interface.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal, Protocol

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
    ``step`` + ``eval.interval`` + ``skip_first_step`` + ``skip_eval_on_resume``,
    so we don't track a separate ``last_eval_step``.
    """

    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0


# ── dispatcher primitives ─────────────────────────────────────────────────


Kind = Literal["train", "eval"]


class SchedMode(Enum):
    """Which kind of work the dispatcher will schedule next.

    Transitions are level-triggered (driven by the eval queue's emptiness), so
    in-flight rollouts of the opposite kind drain naturally on both sides of
    every eval boundary — the overlap mechanism.
    """

    PREFER_TRAIN = auto()
    PREFER_EVAL = auto()


@dataclass
class Rollout:
    """The atomic unit emitted by the dispatcher — one completed rollout.

    Invariant — every rollout the dispatcher acquires a semaphore permit for
    eventually arrives at the corresponding sink exactly once, success or
    failure. Group/batch boundaries are sink-derived by counting arrivals up
    to ``group_size`` (and ``num_examples * group_size`` for eval epochs);
    the dispatcher does not stamp boundary flags because "the last rollout
    in a group" is whichever straggler happens to finish last — not knowable
    at dispatch time.

    Failures (env-reported errors, empty trajectories, task exceptions,
    off-policy cancellations) are carried via ``raw["error"]`` rather than
    silently dropped. Sinks check that field to decide drop / partial-train.

    ``policy_version`` is the snapshot at dispatch time; the train sink
    uses it for per-rollout off-policy metrics. ``eval_step`` is set only
    for eval rollouts (the policy version at which the eval epoch was
    triggered).
    """

    kind: Kind
    env_name: str
    example_id: int
    raw: vf.RolloutOutput
    policy_version: int
    eval_step: int | None = None


@dataclass
class RolloutMeta:
    """Per-task bookkeeping. One entry per in-flight ``run_rollout`` / ``run_group``."""

    kind: Kind
    env_name: str
    group_id: uuid.UUID
    policy_version: int
    rollout_count: int  # number of semaphore permits this task holds
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
class ProcessResult:
    """Per-batch counters the metrics builder reads. Produced by
    ``TrainSink.process_batch`` and passed back via ``TrainBatch.result``."""

    n_trainable: int
    num_prefill_tokens: int
    num_decode_tokens: int
    rollout_prefill_lens: list[int]
    rollout_decode_lens: list[int]
    samples_per_rollout: list[int]
    samples_shipped: int


@dataclass
class TrainBatch:
    """Raw payload the orchestrator hands back to ``MetricsBuilder.build`` /
    ``sender.send``. The ``samples`` list is the trainer-bound subset
    (post-filter survivors only); ``rollouts`` is the full cohort kept for
    metric aggregation. Metrics are NOT pre-baked here — they're a derived
    view computed at log time by the orchestrator with up-to-date timings.
    """

    rollouts: list[vf.RolloutOutput]
    samples: list[TrainingSample]
    result: ProcessResult


@dataclass
class EvalBatch:
    """One env's eval epoch. ``metrics`` is built in ``EvalSink.process_batch``
    — the natural per-batch processing hook — from ``rollouts``; the
    orchestrator just hands it to ``monitor.log`` along with the raw
    rollouts (for samples + save_rollouts)."""

    env_name: str
    step: int
    rollouts: list[vf.RolloutOutput]
    metrics: dict[str, Any]


# ── watcher → observer interface ──────────────────────────────────────────


@dataclass
class DispatcherMetrics:
    """Per-poll counters the dispatcher exposes to ``IntervalLogger``.

    Split into two groups:

    - *Gauges* (read by ``gauges()``): point-in-time snapshots — no reset.
    - *Drain counters* (``drained()``): monotonic per-poll counters; the
      logger consumes them with ``drained()`` which clears each one to
      zero so the next poll measures only what happened since.
    """

    # Drain counters (reset each ``drained()`` call).
    cancelled_train_rollouts: int = 0
    cancelled_eval_rollouts: int = 0
    empty_rollouts_by_env: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errored_rollouts_by_env: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_rollouts_by_env: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Monotonic gauges (not drained — running totals over the run).
    eval_epochs_started: int = 0
    mode_transitions: int = 0

    def record_cancellation(self, *, kind: Literal["train", "eval"], n: int) -> None:
        if kind == "train":
            self.cancelled_train_rollouts += n
        else:
            self.cancelled_eval_rollouts += n

    def record_error(self, env_name: str, error_type: str) -> None:
        self.errored_rollouts_by_env[env_name] += 1
        self.errors_by_type[error_type] += 1

    def record_empty(self, env_name: str) -> None:
        self.empty_rollouts_by_env[env_name] += 1

    def record_arrivals(self, env_name: str, n: int) -> None:
        self.total_rollouts_by_env[env_name] += n

    def drained(self) -> dict[str, float]:
        """Return current drain counters as wandb-shaped metrics + clear them.

        Gauges (live snapshots like ``inflight_*``) are reported separately by
        the dispatcher's ``gauges()`` — those don't need reset semantics.
        """
        out: dict[str, float] = {
            "dispatcher/cancelled_train_rollouts": float(self.cancelled_train_rollouts),
            "dispatcher/cancelled_eval_rollouts": float(self.cancelled_eval_rollouts),
        }
        for env, total in self.total_rollouts_by_env.items():
            errored = self.errored_rollouts_by_env.get(env, 0)
            empty = self.empty_rollouts_by_env.get(env, 0)
            if total > 0:
                out[f"rollouts/{env}/errored_rate"] = errored / total
                out[f"rollouts/{env}/empty_rate"] = empty / total
        for err_type, count in self.errors_by_type.items():
            out[f"errors/{err_type}"] = float(count)

        self.cancelled_train_rollouts = 0
        self.cancelled_eval_rollouts = 0
        self.empty_rollouts_by_env.clear()
        self.errored_rollouts_by_env.clear()
        self.errors_by_type.clear()
        self.total_rollouts_by_env.clear()
        return out


class VersionObserver(Protocol):
    """Notified after each successful policy update.

    The watcher walks the observer list synchronously in registration order
    *after* mutating ``Policy``. Observers see the freshly-installed version
    immediately and can use the call to invalidate caches, cancel stale
    in-flight work, or trigger evals.
    """

    async def on_new_version(self, step: int) -> None: ...
