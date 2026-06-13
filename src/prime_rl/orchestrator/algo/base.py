"""The per-env algorithm runtime: base class and pipeline phase functions.

Each named class in this package *is* one training algorithm, one module per
algorithm: it owns the algorithm's two hooks directly —
``assign_advantages`` (group-time credit + per-token weights) and
``query_references`` (ship-time reference queries) — and declares what it
needs (``action_loss_type``, a ``model_role`` like "teacher"). Reading a
module top to bottom reads the algorithm; writing your own is subclassing
:class:`Algorithm` and overriding the same hooks. Shared math (group
normalization, prefill alignment) lives as plain functions in
``advantage.py``; duplication of orchestration between similar algorithms
(e.g. OPD and OPSD) is accepted so each module stays self-contained.

The two hooks are pinned by the filter barrier: everything the orchestrator
can compute locally happens in ``assign_advantages`` *before* filtering
(filters read the advantage streams), and everything that queries a model
happens in ``query_references`` *after* filtering (so dropped rollouts never
pay reference compute). How rollouts are *produced* is not the algorithm's
concern: that is the env's :class:`~prime_rl.orchestrator.sampler.Sampler`,
and sample construction (interleaving, with observation-token provenance
recorded as ``obs_spans``) is pure pipeline.

The pipeline (dispatcher, train sink, orchestrator) calls the module-level
phase functions (:func:`finalize_group` at group completion,
:func:`finalize_batch` at batch ship) and reads the class
declarations; it never branches on algorithm config fields or model roles —
liveness of a reference is the only runtime distinction. prime-rl hosts
exactly one model — the trainable policy, whose pool is passed in; every
frozen model reference is an external endpoint the algorithm *connects to*
(never launches) in :meth:`Algorithm.setup`.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from prime_rl.configs.algorithm import ActionLossType, AdvantageConfig, FrozenModelConfig, ModelReference
from prime_rl.orchestrator.algo.routing import stamp_advantages, stamp_loss_routing
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.envs import TrainEnvs
    from prime_rl.orchestrator.types import TrainRollout
    from prime_rl.utils.client import InferencePool


async def connect_frozen_pool(config: FrozenModelConfig) -> InferencePool:
    """Connect a client pool to an inline frozen model and wait for it to be
    ready. The endpoint is externally hosted — prime-rl connects and waits,
    never launches."""
    from prime_rl.utils.client import setup_inference_pool

    get_logger().info(f"Initializing frozen model pool (model={config.name}, base_url={', '.join(config.base_url)})")
    pool = await setup_inference_pool(config, model_name=config.name)
    await pool.wait_for_ready(config.name)
    return pool


class Algorithm:
    """Base class for one env's training algorithm — the runtime of the
    bundle's ``advantage`` component (its sibling :class:`Sampler` interprets
    ``sampling``).

    Everything on this class is yours to override; the pipeline drives the
    compilation through the module-level phase functions below
    (:func:`finalize_group` / :func:`finalize_batch`) and never calls
    anything else. The surface is:

    - declarations — which loss component the action tokens feed
      (``action_loss_type``) and what the algorithm calls its reference
      model, if it has one (``model_role``, e.g. "teacher");
    - lifecycle — :meth:`setup` connects client pools to the frozen models
      the algorithm declares, resolving each reference via :meth:`connect`;
    - the two hooks, pinned by the filter barrier:

      - :meth:`assign_advantages` — per group, at finalization, *before*
        filtering (filters read the streams): write each rollout's per-token
        advantage stream, plus any observation ce weights (echo reads the
        ``obs_spans`` provenance interleaving recorded on the samples).
        Default: nothing — rollouts keep ``advantages=None``, so
        advantage-based filters skip them.
      - :meth:`query_references` — per batch, at ship time, *after*
        filtering, async: query the algorithm's own reference pool (e.g.
        ``self.teacher_pool``, connected in :meth:`setup`) and attach
        per-token results. Survivors only — dropped rollouts never cost
        reference compute.

    Constructed with the advantage component it interprets plus the two
    host-owned resources: the policy pool and the policy's renderer (the
    canonical messages → token ids path; ``None`` under MITO)."""

    action_loss_type: ClassVar[ActionLossType] = "rl"
    model_role: ClassVar[str | None] = None

    def __init__(self, advantage: AdvantageConfig, policy_pool: InferencePool, renderer: Renderer | None):
        self.advantage = advantage
        self.policy_pool = policy_pool
        self.renderer = renderer
        self.connected_pools: list[InferencePool] = []  # client pools connected in setup(); closed at shutdown

    async def setup(self) -> None:
        """Connect client pools to the algorithm's frozen models — override
        and resolve each reference via :meth:`connect`. The base has nothing
        to connect."""

    async def connect(self, reference: ModelReference) -> InferencePool:
        """Resolve a model reference to a client pool: the live policy's own
        pool, or a freshly connected pool to a frozen endpoint. Only the
        latter is tracked in ``connected_pools`` — the host closes what the
        algorithm opened, and nothing else, at shutdown."""
        if reference == "policy":
            return self.policy_pool
        pool = await connect_frozen_pool(reference)
        self.connected_pools.append(pool)
        return pool

    def assign_advantages(self, rollouts: list[TrainRollout]) -> None:
        """Write each rollout's per-token advantage stream
        (``rollout.advantages``) — and any observation ce weights — for one
        finalized group."""

    async def query_references(self, rollouts: list[TrainRollout]) -> None:
        """Query the algorithm's reference models and attach per-token
        results to a batch of rollouts at ship time."""


def finalize_group(algorithm: Algorithm, rollouts: list[TrainRollout]) -> None:
    """Group phase: assign credit via the algorithm's
    :meth:`~Algorithm.assign_advantages`, then stamp each sample's wire fields
    (the advantage stream + loss routing). After this the records are frozen —
    groups die at stamping."""
    algorithm.assign_advantages(rollouts)
    for rollout in rollouts:
        stamp_advantages(rollout)
        for sample in rollout.samples:
            sample.reward = rollout.reward
            sample.env_name = rollout.env_name
            stamp_loss_routing(sample, algorithm.action_loss_type)


async def finalize_batch(train_envs: TrainEnvs, rollouts: list[TrainRollout]) -> None:
    """Ship phase: run each env's ``query_references`` over its unfiltered
    rollouts, concurrently across envs. Per-env concurrency is bounded by the
    algorithm's own config; envs without references return immediately."""
    by_env: dict[str, list[TrainRollout]] = defaultdict(list)
    for rollout in rollouts:
        if not rollout.is_filtered:
            by_env[rollout.env_name].append(rollout)
    await asyncio.gather(
        *(
            train_envs.get(env_name).algorithm.query_references(env_rollouts)
            for env_name, env_rollouts in by_env.items()
        )
    )
