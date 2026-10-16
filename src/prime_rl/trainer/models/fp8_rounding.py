"""Deterministic FP8 weight-rounding streams shared by model replicas."""

import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from prime_rl.utils.logger import get_logger

FROZEN_STEP = 1


def _stream_id(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little") & ((1 << 63) - 1)


def seed_for_step(base_seed: int, step: int) -> int:
    """The rounding seed that ``set_fp8_rounding_step(model, step)`` installs, without touching any state."""
    return _stream_id(f"{base_seed}:{step}")


@dataclass
class FP8RoundingState:
    base_seed: int
    seed: torch.Tensor
    frozen: bool = False

    def set_step(self, step: int) -> None:
        # Keep each forward's seed immutable until its backward has consumed it.
        self.seed = torch.tensor(seed_for_step(self.base_seed, step), device=self.seed.device, dtype=torch.int64)


def setup_fp8_rounding(model: nn.Module, base_seed: int = 0, frozen: bool = False) -> None:
    from prime_rl.trainer.models.layers.expert_compute import DeepGemmFP8ExpertCompute
    from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseLinear
    from prime_rl.trainer.models.layers.moe import GroupedExperts

    state = None
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    for module in model.modules():
        # Projections emptied by a runtime fusion have no weight of their own to round.
        dense = isinstance(module, Float8BlockwiseLinear) and module.stochastic_rounding and module.weight is not None
        grouped = (
            isinstance(module, GroupedExperts)
            and isinstance(module.compute, DeepGemmFP8ExpertCompute)
            and module.compute.stochastic_rounding
        )
        if not (dense or grouped):
            continue
        if state is None:
            state = FP8RoundingState(base_seed, torch.empty((), device="cuda", dtype=torch.int64), frozen)
            state.set_step(FROZEN_STEP if frozen else 0)
        module.fp8_rounding = state
        # Tensor stream IDs let identical blocks share a compiled graph.
        if dense:
            module.rounding_salt = torch.tensor(
                _stream_id(parameter_names[id(module.weight)]), device=state.seed.device, dtype=torch.int64
            )
        else:
            module.rounding_salts = {
                name: torch.tensor(
                    _stream_id(parameter_names[id(parameter)]), device=state.seed.device, dtype=torch.int64
                )
                for name, parameter in module.named_parameters(recurse=False)
            }
            if not hasattr(module, "fp8_expert_offset"):
                module.fp8_expert_offset = 0
    model.fp8_rounding = state
    if frozen:
        assert state is not None, "frozen FP8 rounding needs stochastically rounded FP8 modules"
        get_logger().info(
            f"FP8 weight rounding frozen at stream {base_seed}:{FROZEN_STEP} (seed {seed_for_step(base_seed, FROZEN_STEP)})"
        )


def set_fp8_rounding_step(model: nn.Module, step: int) -> None:
    state = model.fp8_rounding
    if state is not None and not state.frozen:
        state.set_step(step)
