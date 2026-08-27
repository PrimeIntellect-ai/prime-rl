from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from prime_rl.trainer.models.layers.fp8_grouped_gemm import (
    cast_grouped_input_to_fp8,
    compute_grouped_layout,
    grouped_fp8_gemm,
)
from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseLinear
from prime_rl.trainer.models.layers.lowprecision import GroupedLayout, LowPrecisionRecipe, QuantizedActivation


@dataclass(frozen=True, eq=False)
class Fp8GroupedLayout(GroupedLayout):
    layout: tuple
    offs: Tensor
    total_m: int


@dataclass(frozen=True, eq=False)
class Fp8QuantizedActivation(QuantizedActivation):
    data: Tensor
    scale: Tensor


class Fp8BlockwiseRecipe(LowPrecisionRecipe):
    name = "fp8_blockwise"

    def linear_cls(self) -> type[nn.Linear]:
        return Float8BlockwiseLinear

    def is_linear_shape_supported(self, in_features: int, out_features: int) -> bool:
        return in_features % 128 == 0 and out_features % 128 == 0

    def convert_linear(self, mod: nn.Linear) -> nn.Linear:
        return Float8BlockwiseLinear.from_linear(mod)

    def build_grouped_layout(self, offs: Tensor, total_m: int) -> GroupedLayout:
        return Fp8GroupedLayout(layout=compute_grouped_layout(offs, total_m), offs=offs, total_m=total_m)

    def quantize_grouped_activation(self, x: Tensor, layout: GroupedLayout) -> QuantizedActivation:
        assert isinstance(layout, Fp8GroupedLayout)
        data, scale = cast_grouped_input_to_fp8(x, layout.layout)
        return Fp8QuantizedActivation(data=data, scale=scale)

    def grouped_gemm(
        self,
        x: Tensor,
        weight: Tensor,
        layout: GroupedLayout,
        x_q: QuantizedActivation | None = None,
    ) -> Tensor:
        assert isinstance(layout, Fp8GroupedLayout)
        x_fp8_cache = (x_q.data, x_q.scale) if isinstance(x_q, Fp8QuantizedActivation) else None
        return grouped_fp8_gemm(x, weight, layout.offs, layout=layout.layout, x_fp8_cache=x_fp8_cache)


FP8_BLOCKWISE_RECIPE = Fp8BlockwiseRecipe()
