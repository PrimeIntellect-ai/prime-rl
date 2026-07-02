"""Orchestrator-side algorithm runtime.

The config side (``prime_rl.configs.algorithm``) defines *what* an algorithm
is — a bundle of sampling and the per-token training signal. This package
turns the signal half into runtime objects (the sampling half is the env's
:class:`~prime_rl.orchestrator.sampler.Sampler`):

- one module per algorithm (``grpo``, ``echo``, ``max_rl``, ``opd``,
  ``opsd``, ``sft``) — each named class owns its scoring hooks
  (``score_rollout`` / ``score_group``) and declares what it needs (loss
  component, a "teacher", ...). One instance per env, built by
  :func:`build_algorithm`. A new credit-assignment scheme is a new named class:
  subclass :class:`Algorithm`, assign advantages in the hook whose timing fits,
  and register it below.
- ``base`` — the :class:`Algorithm` base class, whose non-virtual
  ``finalize_rollout`` / ``finalize_group`` methods the pipeline drives.
  Advantages are per-token everywhere they are stored or shipped — there is no
  scalar advantage in the pipeline. An algorithm assigns credit in its scoring
  hook via ``Rollout.assign_advantages``: a scalar that is *broadcast* over the
  rollout's completion tokens (uniform credit, the common case), or an explicit
  full-length-N per-token list aligned to the concatenated sample token_ids
  (0.0 off-mask).
- ``routing`` — wire-field stamping: per-token component weight streams
  (rl / ce / ref_kl) and the per-token advantage stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.orchestrator.algo.base import Algorithm, connect_frozen_pool
from prime_rl.orchestrator.algo.echo import EchoAlgorithm
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.max_rl import MaxRLAlgorithm
from prime_rl.orchestrator.algo.opd import OPDAlgorithm
from prime_rl.orchestrator.algo.opsd import OPSDAlgorithm
from prime_rl.orchestrator.algo.routing import stamp_advantages, stamp_loss_routing
from prime_rl.orchestrator.algo.sft import SFTAlgorithm
from prime_rl.orchestrator.types import Rollout

if TYPE_CHECKING:
    from prime_rl.configs.algorithm import AlgoConfig
    from prime_rl.utils.client import InferencePool

# Runtime dispatch is keyed on ``algo.type`` — it names the algorithm, and
# each config class's defaults are its vetted parameterization.
ALGORITHM_CLASSES: dict[str, type[Algorithm]] = {
    "grpo": GRPOAlgorithm,
    "echo": EchoAlgorithm,
    "max_rl": MaxRLAlgorithm,
    "opd": OPDAlgorithm,
    "opsd": OPSDAlgorithm,
    "sft": SFTAlgorithm,
}


def build_algorithm(config: AlgoConfig, policy_pool: InferencePool) -> Algorithm:
    cls = ALGORITHM_CLASSES[config.type]
    assert cls.action_loss_type == config.action_loss_type  # config and runtime declare in two places
    # The Algorithm is the runtime of the algorithm config's training signal
    # (its sibling Sampler interprets the sampling half). Every algorithm is
    # handed the live policy pool — opsd self-distills against it, others may
    # judge against it or ignore it. Other models (a frozen teacher, a hint
    # renderer) are built from the algorithm's own config in setup().
    return cls(config, policy_pool)


__all__ = [
    "Algorithm",
    "EchoAlgorithm",
    "GRPOAlgorithm",
    "MaxRLAlgorithm",
    "OPDAlgorithm",
    "OPSDAlgorithm",
    "Rollout",
    "SFTAlgorithm",
    "build_algorithm",
    "connect_frozen_pool",
    "stamp_advantages",
    "stamp_loss_routing",
]
