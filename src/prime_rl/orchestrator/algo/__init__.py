"""Orchestrator-side algorithm runtime.

The config side (``prime_rl.configs.algorithm``) defines *what* an algorithm
is — a bundle of sampling and the per-token training signal. This package
turns the signal half into runtime objects (the sampling half is the env's
:class:`~prime_rl.orchestrator.sampler.Sampler`):

- ``algorithm`` — the named algorithm classes (:class:`GRPOAlgorithm`,
  :class:`OPDAlgorithm`, :class:`OPSDAlgorithm`, ...), each owning its
  ``assign`` (group-time credit) and ``score`` (ship-time reference scoring)
  methods and declaring what it needs (loss component, a "teacher", ...).
  One instance per env, built by :func:`build_algorithm`. Custom credit
  assignment plugs in through the ``custom`` advantage type
  (:class:`CustomAlgorithm` imports a user function by path).
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
    max_rl_advantage_fn,
)
from prime_rl.orchestrator.algo.algorithm import (
    Algorithm,
    CustomAlgorithm,
    EchoAlgorithm,
    GRPOAlgorithm,
    MaxRLAlgorithm,
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
