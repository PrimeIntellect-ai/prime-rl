"""Shared dataclasses, type aliases, and protocols for orchestrator v2.

All of v2's data carriers and small types live here so behavioral modules
(dispatcher, sinks, watcher, metrics, ckpt) can import without depending on
each other's implementation modules.

Sections (in dependency order, but no module here imports another v2 module):

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

from dataclasses import dataclass, field
from enum import Enum, auto
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
    """Persistent counters for the v2 orchestrator.

    ``step`` is the trainer-aligned step (== ``policy.version`` after every
    successful weight update). The eval boundary is a strict function of
    ``step`` + ``eval.interval`` + ``eval_base_model`` + ``skip_eval_on_resume``,
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

    Carries two boundary signals that sinks use to drive ``process_rollout``
    (always) / ``process_group`` (on ``is_group_done``) / ``process_batch``
    (on ``is_batch_done`` for eval, or sink-derived for train):

    - ``is_group_done``: last rollout of the ``(env, example_id)`` GRPO
      group. Set for both train (last in GRPO group) and eval (last in
      per-example group).
    - ``is_batch_done``: last rollout of the natural "batch" unit. Set by
      the dispatcher for eval (last rollout of the env's epoch). Always
      ``False`` for train — the train sink determines batch boundaries
      itself from the configured ``batch_size``/``token_batch_size``.

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
    is_group_done: bool
    is_batch_done: bool
    eval_step: int | None = None


@dataclass
class RolloutMeta:
    """Per-task bookkeeping. One entry per in-flight ``run_rollout`` / ``run_group``."""

    kind: Kind
    env_name: str
    group_id: int
    policy_version: int
    rollout_count: int  # number of semaphore permits this task holds
    client_config: vf.ClientConfig | None = None
    off_policy_steps: int = 0  # incremented on every ``on_new_version``; train only
    eval_step: int | None = None


@dataclass
class GroupState:
    """Accumulator for one rollout group across N independent ``run_rollout`` tasks.

    For group-scoring envs ``rollouts_to_schedule`` collapses to 0 after the
    single ``run_group`` task is queued; otherwise it's decremented per rollout.
    """

    kind: Kind
    env_name: str
    example: dict
    rollouts_to_schedule: int
    target_rollouts: int  # total rollouts expected for this group
    completed_rollouts: list[vf.RolloutOutput] = field(default_factory=list)
    failed_rollouts: int = 0
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
    """One env's eval epoch — the raw rollouts the orchestrator hands back
    to ``EvalSink.build_metrics`` and to the monitor (samples + save_rollouts)."""

    env_name: str
    step: int
    rollouts: list[vf.RolloutOutput]


# ── watcher → observer interface ──────────────────────────────────────────


class VersionObserver(Protocol):
    """Notified after each successful policy update.

    The watcher walks the observer list synchronously in registration order
    *after* mutating ``Policy``. Observers see the freshly-installed version
    immediately and can use the call to invalidate caches, cancel stale
    in-flight work, or trigger evals.
    """

    async def on_new_version(self, step: int) -> None: ...
