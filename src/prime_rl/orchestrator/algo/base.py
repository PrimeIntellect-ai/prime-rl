"""The per-env algorithm runtime: the :class:`Algorithm` base class.

Each named class in this package *is* one training algorithm, one module per
algorithm: it owns the algorithm's two scoring hooks directly —
``score_rollout`` (per arrival) and ``score_group`` (per group) — and declares
which loss component its action tokens feed (``action_loss_type``). Reading a
module top to bottom reads the algorithm; writing your own is subclassing
:class:`Algorithm` and overriding the hooks its signal needs. Shared math (group
normalization, prefill alignment) lives as plain functions in ``advantage.py``;
duplication of orchestration between similar algorithms (e.g. OPD and OPSD) is
accepted so each module stays self-contained.

The two hooks are one scope-and-timing ladder — the wider scope is unlocked by
a later barrier, so the two axes coincide. Both are ``async`` (either may do
I/O); a hook that only does advantage math never awaits:

- ``score_rollout(rollout)`` — one rollout, on arrival: rollout-local signals
  (raw reward, process rewards, echo's observation weighting) *and* per-rollout
  I/O against another model — an inference pool the algorithm connected in
  ``setup()`` (a frozen teacher) or the live policy (opsd's self-distillation),
  queried with bounded concurrency. No siblings.
- ``score_group(group)`` — the cohort, on group completion, *before* filtering
  (filters read the streams): group-relative credit (GRPO/MaxRL baselines).

How rollouts are *produced* is not the algorithm's concern: that is the env's
:class:`~prime_rl.orchestrator.sampler.Sampler`, and sample construction
(interleaving, with observation-token provenance via structural node
attribution) is pure pipeline.

The pipeline (train sink) drives each algorithm through its non-virtual
:meth:`Algorithm.finalize_rollout` / :meth:`Algorithm.finalize_group` methods
and reads the class declarations; it never branches on algorithm config fields
or model roles — liveness of a reference is the only runtime distinction.
prime-rl hosts exactly one model — the trainable policy, whose pool every
algorithm is handed (``self.policy_pool``): use it for anything that scores
against the live model (opsd's self-distillation teacher, an LLM judge, ...).
Every *frozen* model an algorithm needs is an external endpoint it *connects to*
(never launches) and owns, in :meth:`Algorithm.setup`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from prime_rl.configs.algorithm import ActionLossType, AlgoConfig, FrozenModelConfig
from prime_rl.orchestrator.algo.routing import stamp_advantages, stamp_loss_routing
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from renderers import RendererConfig

    from prime_rl.orchestrator.types import Rollout
    from prime_rl.transport import TrainingSample
    from prime_rl.utils.client import InferencePool


async def connect_frozen_pool(
    config: FrozenModelConfig, *, renderer_config: RendererConfig | None = None
) -> InferencePool:
    """Connect a client pool to an inline frozen model and wait for it to be
    ready. The endpoint is externally hosted — prime-rl connects and waits,
    never launches.

    When ``renderer_config`` is set, the pool's train client is the renderer
    (token-in/out) client — required when the frozen model *generates* rollouts
    (sft), so the rollout carries tokens. Left as plain chat-completions
    otherwise (opd/opsd read teacher logprobs via prefill, where the train
    client type is moot)."""
    from prime_rl.utils.client import setup_inference_pool

    get_logger().info(f"Initializing frozen model pool (model={config.name}, base_url={', '.join(config.base_url)})")
    if renderer_config is not None:
        pool = await setup_inference_pool(
            config, model_name=config.name, train_client_type="renderer", renderer_config=renderer_config
        )
    else:
        pool = await setup_inference_pool(config, model_name=config.name)
    await pool.wait_for_ready(config.name)
    return pool


class Algorithm:
    """Base class for one env's training algorithm — the runtime of the
    algorithm config's per-token training signal (its sibling :class:`Sampler`
    interprets the ``sampling`` half).

    Everything on this class is yours to override; the pipeline drives the
    compilation through the non-virtual :meth:`finalize_rollout` /
    :meth:`finalize_group` methods and never calls anything else. The surface is:

    - declarations — which loss component the action tokens feed
      (``action_loss_type``);
    - lifecycle — :meth:`setup` connects client pools to the frozen models
      the algorithm declares, resolving each reference via :meth:`connect`;
    - the two scoring hooks, each ``async`` and given the :class:`Rollout`
      directly — read the trace, write credit via
      :meth:`Rollout.assign_advantages`. They are
      async so either stage may do I/O — e.g. a process-reward model or a
      teacher at arrival, or a judge at group time whose signal a pre-batch
      filter then reads; a hook that only does advantage math simply never
      awaits.

      - :meth:`score_rollout` — one rollout, on arrival: rollout-local credit,
        observation ce weights, or per-token results from a model the algorithm
        connected in :meth:`setup` (e.g. teacher reference logprobs). Default:
        nothing.
      - :meth:`score_group` — the cohort, *before* filtering (filters read the
        streams): group-relative credit. Default: nothing — rollouts keep
        ``advantages=None``, so advantage-based filters skip them.

    Model I/O lives in :meth:`score_rollout`: it runs at arrival, *before* the
    pre-batch filters, so it pays compute on rollouts that may then be filtered
    out — accepted for the simpler one-rollout-at-a-time shape.

    Constructed with the algorithm config it interprets plus the live policy
    pool (``self.policy_pool`` — never closed by the algorithm). An algorithm that needs to tokenize (e.g. opsd's demonstration
    hint) builds its own renderer in :meth:`setup` from its config; the policy's
    renderer is not threaded in."""

    action_loss_type: ClassVar[ActionLossType] = "rl"

    def __init__(self, config: AlgoConfig, policy_pool: InferencePool):
        self.policy_pool = policy_pool
        self.connected_pools: list[InferencePool] = []  # frozen pools connected in setup(); closed at shutdown

    async def setup(self) -> None:
        """Connect client pools to the algorithm's frozen models — override
        and resolve each reference via :meth:`connect`. The base has nothing
        to connect."""

    async def connect(self, reference: FrozenModelConfig) -> InferencePool:
        """Connect a client pool to a frozen model endpoint and track it in
        ``connected_pools`` — the host closes what the algorithm opened, at
        shutdown. The live policy is never connected here; opsd receives the
        policy pool directly."""
        pool = await connect_frozen_pool(reference)
        self.connected_pools.append(pool)
        return pool

    async def score_rollout(self, rollout: Rollout) -> None:
        """Arrival phase, one rollout, before its group is complete: write
        rollout-local credit (``rollout.assign_advantages``), observation ce
        weights (echo), or per-token results from a model — an inference pool
        connected in :meth:`setup`, or the live policy (opsd). No siblings, no
        group stats."""

    async def score_group(self, group: list[Rollout]) -> None:
        """Group phase, the finalized cohort, before filtering: write
        group-relative credit."""

    @classmethod
    def route_sample(cls, sample: TrainingSample) -> None:
        """Route one already-built sample through this algorithm's loss."""
        stamp_loss_routing(sample, cls.action_loss_type)

    async def finalize_rollout(self, rollout: Rollout) -> None:
        """Arrival phase (non-virtual): rollout-local scoring as each rollout is
        tokenized."""
        if rollout.samples:
            await self.score_rollout(rollout)

    async def finalize_group(self, rollouts: list[Rollout]) -> None:
        """Group phase (non-virtual): group-relative scoring, then stamp each
        sample's wire fields (the advantage stream + loss routing). After this
        the records are frozen — groups die at stamping."""
        await self.score_group(rollouts)
        for rollout in rollouts:
            stamp_advantages(rollout)
            for sample in rollout.samples:
                self.route_sample(sample)
