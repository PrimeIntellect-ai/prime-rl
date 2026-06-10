"""Orchestrator-side algorithm runtime.

The config side (``prime_rl.configs.algorithm``) defines *what* an algorithm
is — a preset of sampling, advantage, and loss routing. This package turns
that declaration into runtime objects:

- ``algorithm`` — :class:`Algorithm`, one strategy object per env and the only
  orchestrator component that interprets ``AlgorithmConfig``; connects client
  pools to inline frozen model references (prime-rl never hosts them).
- ``strategies`` — one :class:`AdvantageStrategy` per advantage union member,
  owning both execution points of the training signal: group-time assignment
  and ship-time reference scoring (run via :func:`score_train_batch`).
- ``advantage`` — pure advantage math: the custom-function interface
  (:class:`AdvantageInputs` / :class:`AdvantageOutputs`) and the default
  group-norm computation.
- ``routing`` — wire-field stamping: per-token loss routing and per-token
  advantage spreading.
"""

from prime_rl.orchestrator.algo.advantage import (
    AdvantageFn,
    AdvantageInputs,
    AdvantageOutputs,
    assign_advantages,
    default_advantage_fn,
)
from prime_rl.orchestrator.algo.algorithm import Algorithm, connect_frozen_pool, score_train_batch
from prime_rl.orchestrator.algo.routing import ACTION_LOSS_TYPES, spread_token_advantages, stamp_loss_routing
from prime_rl.orchestrator.algo.strategies import (
    AdvantageStrategy,
    CustomAdvantage,
    DemoRefKLAdvantage,
    GroupNormAdvantage,
    RefKLAdvantage,
    RewardAdvantage,
    SupervisedAdvantage,
    setup_advantage_strategy,
)

__all__ = [
    "ACTION_LOSS_TYPES",
    "AdvantageFn",
    "AdvantageInputs",
    "AdvantageOutputs",
    "AdvantageStrategy",
    "Algorithm",
    "CustomAdvantage",
    "DemoRefKLAdvantage",
    "GroupNormAdvantage",
    "RefKLAdvantage",
    "RewardAdvantage",
    "SupervisedAdvantage",
    "assign_advantages",
    "connect_frozen_pool",
    "default_advantage_fn",
    "score_train_batch",
    "setup_advantage_strategy",
    "spread_token_advantages",
    "stamp_loss_routing",
]
