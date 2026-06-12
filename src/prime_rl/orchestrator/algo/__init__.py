"""Orchestrator-side algorithm runtime.

The config side (``prime_rl.configs.algorithm``) defines *what* an algorithm
is — a bundle of sampling and the per-token training signal. This package
turns the signal half into runtime objects (the sampling half is the env's
:class:`~prime_rl.orchestrator.sampler.Sampler`):

- one module per algorithm (``grpo``, ``echo``, ``max_rl``, ``opd``,
  ``opsd``, ``rlcsd``, ``sft``, ``reward``, ``custom``) — each named class owns its
  ``assign`` (group-time credit) and ``score`` (ship-time reference scoring)
  methods and declares what it needs (loss component, a "teacher", ...).
  One instance per env, built by :func:`build_algorithm`. Custom credit
  assignment plugs in through the ``custom`` advantage type
  (:class:`CustomAlgorithm` imports a user function by path).
- ``base`` — the :class:`Algorithm` base class and the pipeline hooks
  (frozen-pool connection, batch scoring).
- ``advantage`` — pure advantage math: the custom-function interface
  (:class:`AdvantageInputs` / :class:`AdvantageOutputs`) and the default
  group-norm computation.
- ``routing`` — wire-field stamping: per-token component weight streams
  (rl / ce / ref_kl) and per-token advantage spreading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.orchestrator.algo.advantage import (
    AdvantageFn,
    AdvantageInputs,
    AdvantageOutputs,
    assign_advantages,
    default_advantage_fn,
    max_rl_advantage_fn,
)
from prime_rl.orchestrator.algo.base import Algorithm, connect_frozen_pool, score_train_batch
from prime_rl.orchestrator.algo.custom import CustomAlgorithm
from prime_rl.orchestrator.algo.echo import EchoAlgorithm
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.max_rl import MaxRLAlgorithm
from prime_rl.orchestrator.algo.opd import OPDAlgorithm
from prime_rl.orchestrator.algo.opsd import OPSDAlgorithm
from prime_rl.orchestrator.algo.reward import RewardAlgorithm
from prime_rl.orchestrator.algo.rlcsd import RLCSDAlgorithm
from prime_rl.orchestrator.algo.routing import spread_token_advantages, stamp_loss_routing
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
    "rlcsd": RLCSDAlgorithm,
    "sft": SFTDistillAlgorithm,
    "reward": RewardAlgorithm,
    "custom": CustomAlgorithm,
}


def build_algorithm(config: AlgorithmConfig, policy_pool: InferencePool, renderer: Renderer | None) -> Algorithm:
    cls = ALGORITHM_CLASSES[config.advantage.type]
    assert cls.action_loss_type == config.advantage.action_loss_type  # config and runtime declare in two places
    return cls(config, policy_pool, renderer)


__all__ = [
    "AdvantageFn",
    "AdvantageInputs",
    "AdvantageOutputs",
    "Algorithm",
    "CustomAlgorithm",
    "EchoAlgorithm",
    "GRPOAlgorithm",
    "MaxRLAlgorithm",
    "OPDAlgorithm",
    "OPSDAlgorithm",
    "RLCSDAlgorithm",
    "RewardAlgorithm",
    "SFTDistillAlgorithm",
    "assign_advantages",
    "build_algorithm",
    "connect_frozen_pool",
    "default_advantage_fn",
    "max_rl_advantage_fn",
    "score_train_batch",
    "spread_token_advantages",
    "stamp_loss_routing",
]
