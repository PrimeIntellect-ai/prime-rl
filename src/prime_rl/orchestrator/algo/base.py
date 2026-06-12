"""The per-env algorithm runtime: base class and pipeline hooks.

Each named class in this package *is* one training algorithm, one module per
algorithm: it owns the algorithm's methods directly — ``assign`` (group-time
credit) and ``score`` (ship-time reference scoring) — and declares what it
needs (``action_loss_type``, a ``model_role`` like "teacher"). Reading a
module top to bottom reads the algorithm; writing your own is subclassing
:class:`Algorithm` and overriding the same methods. Shared math (group
normalization, prefill alignment) lives as plain functions in
``advantage.py``; duplication of orchestration between similar algorithms
(e.g. OPD and OPSD) is accepted so each module stays self-contained.

How rollouts are *produced* is not the algorithm's concern: that is the env's
:class:`~prime_rl.orchestrator.sampler.Sampler`. The algorithm consumes
finalized rollouts and compiles them into the per-token component weight
streams the trainer executes — credit assignment and loss routing are two
phases of that one compilation.

The pipeline (dispatcher, train sink, orchestrator) calls the base-class hooks
and reads its properties; it never branches on algorithm config fields or
model roles — liveness of a reference is the only runtime distinction.
prime-rl hosts exactly one model — the trainable policy, whose pool is passed
in; every frozen model reference is an external endpoint the algorithm
*connects to* (never launches) in :meth:`Algorithm.setup`.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from prime_rl.configs.algorithm import ActionLossType, AlgorithmConfig, FrozenModelConfig
from prime_rl.orchestrator.algo.routing import stamp_advantages, stamp_loss_routing
from prime_rl.orchestrator.trajectories import interleave_rollout
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    import verifiers as vf
    from renderers.base import Renderer

    from prime_rl.orchestrator.envs import TrainEnvs
    from prime_rl.orchestrator.types import TrainRollout
    from prime_rl.transport import TrainingSample
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
    """Base class for one env's training algorithm — the interpreter of the
    bundle's ``advantage`` component (its sibling :class:`Sampler` interprets
    ``sampling``).

    The algorithm is one compilation — finalized rollouts in, per-token
    component weight streams out — split over the pipeline's three phases,
    each driven by one method the pipeline calls:

    - :meth:`build_samples` — per rollout, as it arrives: interleave the
      trajectory into training samples, weighting env-provided observation
      tokens via the :meth:`observation_weights` hook.
    - :meth:`finalize_group` — per group, at finalization: assign credit via
      the :meth:`assign` hook, then stamp the wire fields (advantage + loss
      routing).
    - :meth:`score` — per batch, at ship time, async: attach per-token
      reference data by querying ``self.reference_pool``. Runs on batch
      survivors only, so filtered rollouts never cost reference compute.

    Subclasses override the hooks — :meth:`observation_weights` (default:
    ``None``, observations stay masked), :meth:`assign` (default: nothing —
    rollouts keep ``advantages=None``, so advantage-based filters skip them),
    and :meth:`score` (driver and hook in one; the default scores nothing).

    Class-level declarations say what the algorithm needs: which loss
    component its action tokens feed (``action_loss_type``) and what it calls
    its reference model, if it has one (``model_role``, e.g. "teacher").
    Constructed with the policy pool and the policy's renderer (the canonical
    messages → token ids path; ``None`` under MITO); connects a client pool to
    an inline frozen reference model in :meth:`setup`."""

    action_loss_type: ClassVar[ActionLossType] = "rl"
    model_role: ClassVar[str | None] = None

    def __init__(self, config: AlgorithmConfig, policy_pool: InferencePool, renderer: Renderer | None):
        self.config = config
        self.policy_pool = policy_pool
        self.renderer = renderer
        self.reference_pool: InferencePool | None = None  # resolved in setup() when the algorithm declares a model
        self.connected_pools: list[InferencePool] = []  # client pools connected in setup(); closed at shutdown

    async def setup(self) -> None:
        """Connect a client pool to the algorithm's frozen reference model and
        wait for readiness. Must run before scoring."""
        reference = getattr(self.config.advantage, "model", None)
        if reference is not None:
            if reference == "policy":
                self.reference_pool = self.policy_pool
            else:
                self.reference_pool = await connect_frozen_pool(reference)
                self.connected_pools.append(self.reference_pool)

    def assign(self, rollouts: list[TrainRollout]) -> None:
        """Assign credit to one finalized group of rollouts."""

    async def score(self, rollouts: list[TrainRollout]) -> None:
        """Attach per-token reference data to a batch of rollouts at ship time."""

    def observation_weights(self, output: vf.RolloutOutput) -> list[list[float]] | None:
        """Per-token ce weights for env-provided observation tokens: one list
        per trajectory step, each spanning that step's ``prompt_ids`` +
        ``completion_ids``. :meth:`build_samples` aligns the spans onto the
        merged samples; algorithms that train on observations (echo) override
        this. ``None`` (the default) masks every observation token out."""
        return None

    def build_samples(
        self,
        output: vf.RolloutOutput,
        *,
        env_name: str,
        mm_token_type_ids_mapping: dict[int, int] | None = None,
    ) -> list[TrainingSample] | None:
        """Compile one finalized rollout into training samples: best-effort
        interleaving of the trajectory steps, with this algorithm's
        :meth:`observation_weights` deciding what env-provided tokens train."""
        return interleave_rollout(
            output,
            mm_token_type_ids_mapping=mm_token_type_ids_mapping,
            env_name=env_name,
            obs_weights=self.observation_weights(output),
        )

    def finalize_group(self, rollouts: list[TrainRollout]) -> None:
        """Score one finalized group: assign credit, then stamp each sample's
        wire fields (the advantage stream + loss routing)."""
        self.assign(rollouts)
        for rollout in rollouts:
            stamp_advantages(rollout)
            for sample in rollout.samples:
                sample.reward = rollout.reward
                sample.env_name = rollout.env_name
                stamp_loss_routing(sample, self.action_loss_type)

    def _reference_pool(self) -> InferencePool:
        pool = self.reference_pool
        assert pool is not None, f"{self.model_role or 'reference'} pool not set — Algorithm.setup() must run first"
        return pool


async def score_train_batch(train_envs: TrainEnvs, rollouts: list[TrainRollout]) -> None:
    """Run each env's ``score`` over its unfiltered rollouts, concurrently
    across envs. Per-env concurrency is bounded by the algorithm's own
    config; envs without reference scoring return immediately."""
    by_env: dict[str, list[TrainRollout]] = defaultdict(list)
    for rollout in rollouts:
        if not rollout.is_filtered:
            by_env[rollout.env_name].append(rollout)
    await asyncio.gather(
        *(train_envs.get(env_name).algorithm.score(env_rollouts) for env_name, env_rollouts in by_env.items())
    )
