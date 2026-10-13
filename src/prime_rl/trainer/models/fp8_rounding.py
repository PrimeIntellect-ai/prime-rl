"""Deterministic FP8 weight-rounding streams shared by model replicas."""

import hashlib
from dataclasses import dataclass

import torch
from torch import nn


def _stream_id(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little") & ((1 << 63) - 1)


@dataclass
class FP8RoundingState:
    base_seed: int
    seed: torch.Tensor

    def set_step(self, step: int) -> None:
        # Keep each forward's seed immutable until its backward has consumed it.
        self.seed = torch.tensor(_stream_id(f"{self.base_seed}:{step}"), device=self.seed.device, dtype=torch.int64)


def setup_fp8_rounding(model: nn.Module, base_seed: int = 0) -> None:
    from prime_rl.trainer.models.layers.expert_compute import DeepGemmFP8ExpertCompute
    from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseLinear
    from prime_rl.trainer.models.layers.moe import GroupedExperts

    state = None
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    for module in model.modules():
        dense = isinstance(module, Float8BlockwiseLinear) and module.stochastic_rounding
        grouped = (
            isinstance(module, GroupedExperts)
            and isinstance(module.compute, DeepGemmFP8ExpertCompute)
            and module.compute.stochastic_rounding
        )
        if not (dense or grouped):
            continue
        if state is None:
            state = FP8RoundingState(base_seed, torch.empty((), device="cuda", dtype=torch.int64))
            state.set_step(0)
        module.fp8_rounding = state
        if dense:
            module.rounding_salt = _stream_id(parameter_names[id(module.weight)])
        else:
            module.rounding_salts = {
                name: _stream_id(parameter_names[id(parameter)])
                for name, parameter in module.named_parameters(recurse=False)
            }
            if not hasattr(module, "fp8_expert_offset"):
                module.fp8_expert_offset = 0
    model.fp8_rounding = state


def set_fp8_rounding_step(model: nn.Module, step: int) -> None:
    state = model.fp8_rounding
    if state is not None:
        state.set_step(step)
