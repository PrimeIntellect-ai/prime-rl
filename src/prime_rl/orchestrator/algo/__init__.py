"""Orchestrator-side algorithm runtime.

The config side (``prime_rl.configs.algorithm``) defines *what* an algorithm
is — a bundle of sampling and the per-token training signal. This package
turns the signal half into runtime objects (the sampling half is the env's
:class:`~prime_rl.orchestrator.sampler.Sampler`):

- one module per algorithm (``grpo``, ``echo``, ``max_rl``, ``opd``,
  ``opsd``, ``sft``, ``reward``, ``custom``) — each named class owns its
  hooks (``observation_weights`` / ``assign_advantages`` / ``score``) and
  declares what it needs (loss component, a "teacher", ...). One instance
  per env, built by :func:`build_algorithm`. Custom credit assignment plugs
  in through the ``custom`` advantage type (:class:`CustomAlgorithm` imports
  a user function by path).
- ``base`` — the :class:`Algorithm` base class and the pipeline phase
  functions (:func:`build_samples` / :func:`finalize_group` /
  :func:`score_train_batch`).
- ``advantage`` — pure advantage math: the custom-function interface
  (:class:`AdvantageInputs`, per-token advantages out) and the default
  group-norm computation.
- ``routing`` — wire-field stamping: per-token component weight streams
  (rl / ce / ref_kl) and the per-token advantage stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.orchestrator.algo.advantage import (
    AdvantageFn,
    AdvantageInputs,
    apply_advantage_fn,
    default_advantage_fn,
    max_rl_advantage_fn,
)
from prime_rl.orchestrator.algo.base import (
    Algorithm,
    build_samples,
    connect_frozen_pool,
    finalize_group,
    score_train_batch,
)
from prime_rl.orchestrator.algo.custom import CustomAlgorithm
from prime_rl.orchestrator.algo.echo import EchoAlgorithm
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.max_rl import MaxRLAlgorithm
from prime_rl.orchestrator.algo.opd import OPDAlgorithm
from prime_rl.orchestrator.algo.opsd import OPSDAlgorithm
from prime_rl.orchestrator.algo.reward import RewardAlgorithm
from prime_rl.orchestrator.algo.routing import stamp_advantages, stamp_loss_routing
from prime_rl.orchestrator.algo.sft import SFTDistillAlgorithm

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.configs.algorithm import AlgorithmConfig
    from prime_rl.utils.client import InferencePool

# Runtime dispatch is keyed on the advantage type — it names the algorithm,
# and each config class's defaults are its vetted parameterization.
ALGORITHM_CLASSES: dict[str, type[Algorithm]] = {
    "grpo": GRPOAlgorithm,
    "echo": EchoAlgorithm,
    "max_rl": MaxRLAlgorithm,
    "opd": OPDAlgorithm,
    "opsd": OPSDAlgorithm,
    "sft": SFTDistillAlgorithm,
    "reward": RewardAlgorithm,
    "custom": CustomAlgorithm,
}


def build_algorithm(config: AlgorithmConfig, policy_pool: InferencePool, renderer: Renderer | None) -> Algorithm:
    cls = ALGORITHM_CLASSES[config.advantage.type]
    assert cls.action_loss_type == config.advantage.action_loss_type  # config and runtime declare in two places
    # The bundle dissolves at construction: the Algorithm is the advantage
    # component's runtime (its sibling Sampler interprets the sampling half).
    return cls(config.advantage, policy_pool, renderer)


__all__ = [
    "AdvantageFn",
    "AdvantageInputs",
    "Algorithm",
    "CustomAlgorithm",
    "EchoAlgorithm",
    "GRPOAlgorithm",
    "MaxRLAlgorithm",
    "OPDAlgorithm",
    "OPSDAlgorithm",
    "RewardAlgorithm",
    "SFTDistillAlgorithm",
    "apply_advantage_fn",
    "build_algorithm",
    "build_samples",
    "connect_frozen_pool",
    "default_advantage_fn",
    "finalize_group",
    "max_rl_advantage_fn",
    "score_train_batch",
    "stamp_advantages",
    "stamp_loss_routing",
]
