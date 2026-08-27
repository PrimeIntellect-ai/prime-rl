"""Orchestrator-side algorithm runtime.

The config side (``prime_rl.configs.algorithm``) defines *what* an algorithm
is — a bundle of sampling and the per-token training signal. This package
turns the signal half into runtime objects (the sampling half is the env's
:class:`~prime_rl.orchestrator.generation_source.GenerationSource`):

- one module per algorithm (``grpo``, ``echo``, ``max_rl``, ``rae``,
  ``hierarchical_grpo``, ``opd``, ``opsd``, ``sft``) — each named class owns
  its scoring hooks
  (``score_episode`` / ``score_group``) and declares what it needs (loss
  component, a "teacher", ...). One instance per env, built by
  :func:`build_algorithm`. A new credit-assignment scheme is a new named class:
  subclass :class:`Algorithm`, assign advantages in the hook whose timing fits,
  and register it below.
- ``base`` — the :class:`Algorithm` base class. Algorithms annotate native
  verifier graphs; transport samples are compiled only after admission.
- ``routing`` — graph annotation and final transport loss routing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.orchestrator.algo.base import Algorithm, connect_frozen_client
from prime_rl.orchestrator.algo.echo import EchoAlgorithm
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.hierarchical_grpo import HierarchicalGRPOAlgorithm
from prime_rl.orchestrator.algo.max_rl import MaxRLAlgorithm
from prime_rl.orchestrator.algo.opd import OPDAlgorithm
from prime_rl.orchestrator.algo.opsd import OPSDAlgorithm
from prime_rl.orchestrator.algo.rae import RAEAlgorithm
from prime_rl.orchestrator.algo.routing import assign_advantages, stamp_loss_routing
from prime_rl.orchestrator.algo.sft import SFTDistillAlgorithm

if TYPE_CHECKING:
    from prime_rl.configs.algorithm import AlgoConfig
    from prime_rl.orchestrator.clients import InferenceClient

# Runtime dispatch is keyed on ``algo.type`` — it names the algorithm, and
# each config class's defaults are its vetted parameterization.
ALGORITHM_CLASSES: dict[str, type[Algorithm]] = {
    "grpo": GRPOAlgorithm,
    "echo": EchoAlgorithm,
    "max_rl": MaxRLAlgorithm,
    "rae": RAEAlgorithm,
    "hierarchical_grpo": HierarchicalGRPOAlgorithm,
    "opd": OPDAlgorithm,
    "opsd": OPSDAlgorithm,
    "sft": SFTDistillAlgorithm,
}


def build_algorithm(config: AlgoConfig, clients: InferenceClient) -> Algorithm:
    cls = ALGORITHM_CLASSES[config.type]
    assert cls.action_loss_type == config.action_loss_type  # config and runtime declare in two places
    # The Algorithm is the runtime of the algorithm config's training signal
    # (its sibling GenerationSource interprets the sampling half). Every algorithm is
    # handed the live policy pool — opsd self-distills against it, others may
    # judge against it or ignore it. Other models (a frozen teacher, a hint
    # renderer) are built from the algorithm's own config in setup().
    return cls(config, clients)


__all__ = [
    "Algorithm",
    "EchoAlgorithm",
    "GRPOAlgorithm",
    "HierarchicalGRPOAlgorithm",
    "MaxRLAlgorithm",
    "OPDAlgorithm",
    "OPSDAlgorithm",
    "RAEAlgorithm",
    "SFTDistillAlgorithm",
    "build_algorithm",
    "connect_frozen_client",
    "assign_advantages",
    "stamp_loss_routing",
]
