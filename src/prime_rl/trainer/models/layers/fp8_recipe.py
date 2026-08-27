from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from prime_rl.trainer.models.layers.fp8_grouped_gemm import (
    cast_grouped_input_to_fp8,
    compute_grouped_layout,
    grouped_fp8_gemm,
)
from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseLinear
from prime_rl.trainer.models.layers.lowprecision import (
    LinearRecipe,
    MoEExpertKernel,
    PreparedActivations,
    PreparedWeights,
)


class Fp8LinearRecipe(LinearRecipe):
    name = "fp8_blockwise"

    def linear_cls(self) -> type[nn.Linear]:
        return Float8BlockwiseLinear

    def is_linear_shape_supported(self, in_features: int, out_features: int) -> bool:
        return in_features % 128 == 0 and out_features % 128 == 0

    def convert_linear(self, mod: nn.Linear) -> nn.Linear:
        return Float8BlockwiseLinear.from_linear(mod)


@dataclass(frozen=True, eq=False)
class Fp8PreparedWeights(PreparedWeights):
    w1: Tensor
    w2: Tensor
    w3: Tensor | None


@dataclass(frozen=True, eq=False)
class Fp8PreparedActivations(PreparedActivations):
    x: Tensor
    layout: tuple
    offs: Tensor
    x_fp8_cache: tuple[Tensor, Tensor]
    orig_dtype: torch.dtype


class Fp8MoEExpertKernel(MoEExpertKernel):
    name = "fp8_blockwise"

    def preprocess_weights(self, w1: Tensor, w2: Tensor, w3: Tensor | None) -> PreparedWeights:
        return Fp8PreparedWeights(
            w1=w1.bfloat16().transpose(-2, -1),
            w2=w2.bfloat16().transpose(-2, -1),
            w3=w3.bfloat16().transpose(-2, -1) if w3 is not None else None,
        )

    def preprocess_activations(self, x: Tensor, num_tokens_per_expert: Tensor) -> PreparedActivations:
        offs = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        x_bf16 = x.bfloat16()
        layout = compute_grouped_layout(offs, x_bf16.size(0))
        x_fp8_cache = cast_grouped_input_to_fp8(x_bf16, layout)
        return Fp8PreparedActivations(x=x_bf16, layout=layout, offs=offs, x_fp8_cache=x_fp8_cache, orig_dtype=x.dtype)

    def compute(self, weights: PreparedWeights, activations: PreparedActivations) -> Tensor:
        assert isinstance(weights, Fp8PreparedWeights)
        assert isinstance(activations, Fp8PreparedActivations)
        x, layout, offs = activations.x, activations.layout, activations.offs
        if weights.w3 is not None:
            h = F.silu(grouped_fp8_gemm(x, weights.w1, offs, layout=layout, x_fp8_cache=activations.x_fp8_cache))
            h = h * grouped_fp8_gemm(x, weights.w3, offs, layout=layout, x_fp8_cache=activations.x_fp8_cache)
        else:
            h = (grouped_fp8_gemm(x, weights.w1, offs, layout=layout, x_fp8_cache=activations.x_fp8_cache)).relu() ** 2
        return grouped_fp8_gemm(h, weights.w2, offs, layout=layout)

    def postprocess_activations(self, out: Tensor, activations: PreparedActivations) -> Tensor:
        assert isinstance(activations, Fp8PreparedActivations)
        return out.to(activations.orig_dtype)



FP8_MOE_EXPERT_KERNEL = Fp8MoEExpertKernel()
