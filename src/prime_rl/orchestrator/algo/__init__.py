"""Orchestrator-side algorithm runtime.

The config side (``prime_rl.configs.algorithm``) defines *what* an algorithm
is — a preset of sampling, advantage, and loss routing. This package turns
that declaration into runtime objects:

- ``algorithm`` — the named algorithm classes (:class:`GRPOAlgorithm`,
  :class:`OPDAlgorithm`, :class:`OPSDAlgorithm`, ...), each owning its
  ``assign`` (group-time credit) and ``score`` (ship-time reference scoring)
  methods and declaring what it needs (loss component, a "teacher", ...).
  One instance per env, built by :func:`build_algorithm`; the only
  orchestrator components that interpret ``AlgorithmConfig``. Subclass
  :class:`Algorithm` to write your own.
- ``advantage`` — pure advantage math: the custom-function interface
  (:class:`AdvantageInputs` / :class:`AdvantageOutputs`) and the default
  group-norm computation.
- ``routing`` — wire-field stamping: per-token component weight streams
  (rl / ce / ref_kl) and per-token advantage spreading.
"""

from prime_rl.orchestrator.algo.advantage import (
    AdvantageFn,
    AdvantageInputs,
    AdvantageOutputs,
    assign_advantages,
    default_advantage_fn,
)
from prime_rl.orchestrator.algo.algorithm import (
    ALGORITHM_CLASSES,
    Algorithm,
    CustomAlgorithm,
    GRPOAlgorithm,
    OPDAlgorithm,
    OPSDAlgorithm,
    RewardAlgorithm,
    SFTDistillAlgorithm,
    build_algorithm,
    connect_frozen_pool,
    score_train_batch,
)
from prime_rl.orchestrator.algo.routing import spread_token_advantages, stamp_loss_routing

__all__ = [
    "ALGORITHM_CLASSES",
    "AdvantageFn",
    "AdvantageInputs",
    "AdvantageOutputs",
    "Algorithm",
    "CustomAlgorithm",
    "GRPOAlgorithm",
    "OPDAlgorithm",
    "OPSDAlgorithm",
    "RewardAlgorithm",
    "SFTDistillAlgorithm",
    "assign_advantages",
    "build_algorithm",
    "connect_frozen_pool",
    "default_advantage_fn",
    "score_train_batch",
    "spread_token_advantages",
    "stamp_loss_routing",
]
