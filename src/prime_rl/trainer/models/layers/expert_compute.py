# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import weakref
from collections.abc import Callable
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

import torch
from torch.distributed.tensor import DTensor

from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.trainer.models.layers.moe import GroupedExperts


def to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class ExpertCompute(Protocol):
    token_group_alignment: int

    def validate(self, experts: "GroupedExperts") -> None: ...

    def __call__(
        self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor
    ) -> torch.Tensor: ...


class FusedDispatchExpertCompute(Protocol):
    """Expert compute that also dispatches tokens to their experts and combines the results.

    It takes this rank's undispatched tokens and their routing, paired with a ``FusedTokenDispatcher``.
    """

    token_group_alignment: int

    def validate(self, experts: "GroupedExperts") -> None: ...

    def __call__(
        self,
        experts: "GroupedExperts",
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> torch.Tensor: ...


def broadcast_expert_bias(
    bias: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    target_rows: int,
) -> torch.Tensor:
    repeats = num_tokens_per_expert.to(torch.int64)
    padding_rows = repeats.new_tensor(target_rows) - repeats.sum()
    return torch.repeat_interleave(
        torch.cat((bias, bias.new_zeros((1, bias.shape[1])))),
        torch.cat((repeats, padding_rows.unsqueeze(0))),
        dim=0,
        output_size=target_rows,
    )


class GroupedGemmExpertCompute:
    def __init__(
        self,
        gemm: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        token_group_alignment: int = 8,
    ) -> None:
        self.gemm = gemm
        self.token_group_alignment = token_group_alignment

    def validate(self, experts: "GroupedExperts") -> None:
        """The shared forward handles the experts' activation, biases, and weight layout."""

    def __call__(self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        x_bf16 = x.bfloat16()

        if experts.gate_up_proj is None:
            up_proj = to_local(experts.up_proj).transpose(-2, -1)
            up = self.gemm(x_bf16, up_proj.bfloat16(), offsets)

            gate = None
            if experts.gate_proj is not None:
                gate_proj = to_local(experts.gate_proj).transpose(-2, -1)
                gate = self.gemm(x_bf16, gate_proj.bfloat16(), offsets)
        else:
            gate_up_proj = to_local(experts.gate_up_proj).transpose(-2, -1)
            gate_up = self.gemm(x_bf16, gate_up_proj.bfloat16(), offsets)
            gate, up = gate_up.chunk(2, dim=-1)

        if experts.up_proj_bias is not None:
            up_proj_bias = to_local(experts.up_proj_bias)
            up = up + broadcast_expert_bias(up_proj_bias, num_tokens_per_expert, up.shape[0]).bfloat16()

        if gate is not None and experts.gate_proj_bias is not None:
            gate_proj_bias = to_local(experts.gate_proj_bias)
            gate = gate + broadcast_expert_bias(gate_proj_bias, num_tokens_per_expert, gate.shape[0]).bfloat16()

        hidden = experts.activation.apply(gate, up)
        down_proj = to_local(experts.down_proj).transpose(-2, -1)
        output = self.gemm(hidden, down_proj.bfloat16(), offsets)
        if experts.down_proj_bias is not None:
            down_proj_bias = to_local(experts.down_proj_bias)
            output = output + broadcast_expert_bias(down_proj_bias, num_tokens_per_expert, output.shape[0]).bfloat16()
        return output.type_as(x)


class BF16ExpertCompute(GroupedGemmExpertCompute):
    def __init__(self) -> None:
        super().__init__(torch._grouped_mm)


class DeepGemmFP8ExpertCompute(GroupedGemmExpertCompute):
    def __init__(self) -> None:
        from prime_rl.trainer.models.kernels.fp8_utils import GROUP_ALIGNMENT
        from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm

        super().__init__(grouped_fp8_gemm, token_group_alignment=GROUP_ALIGNMENT)


_WEIGHT_CACHES: "weakref.WeakSet[NVFP4ExpertCompute]" = weakref.WeakSet()


_cache_size_logged = False


def clear_quantized_weight_caches() -> None:
    """Drop every cached quantized expert weight. The train loops call it after each optimizer step, the only place
    trainer weights change (CPU-offload optimizers update weights without ``torch.optim.Optimizer.step`` hooks). The
    first time, log what the caches held: the memory ``cache_weights`` costs."""
    global _cache_size_logged
    if not _cache_size_logged and _WEIGHT_CACHES:
        held = sum(
            tensor.nbytes
            for compute in _WEIGHT_CACHES
            for slots in compute.caches.values()
            for cache in slots
            for value in cache.values()
            for tensor in _tensors(value)
        )
        get_logger().info(f"NVFP4 weight cache held {held / 2**30:.1f} GiB of quantized expert weights this step")
        _cache_size_logged = True
    for compute in _WEIGHT_CACHES:
        compute.caches.clear()


def _tensors(value) -> list[torch.Tensor]:
    """The tensors in a cached quantized weight: a tensor, a tuple of them, or a dataclass of them."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for item in value for t in _tensors(item)]
    return [t for t in vars(value).values() if isinstance(t, torch.Tensor)] if hasattr(value, "__dict__") else []


class NVFP4ExpertCompute(GroupedGemmExpertCompute):
    """With ``cache_weights``, each expert weight is quantized once per optimizer step instead of on every
    micro-batch and recompute: the caches, one per GEMM of each experts module, live until
    ``clear_quantized_weight_caches``."""

    def __init__(self, kernel: ModuleType, backward: str, four_over_six: bool, cache_weights: bool) -> None:
        def gemm(x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
            cache = next(self.slots) if self.slots is not None else None
            return kernel.grouped_gemm(
                x, weight, offs=offsets, backward=backward, four_over_six=four_over_six, weight_cache=cache
            )

        super().__init__(gemm, token_group_alignment=kernel.TOKEN_GROUP_ALIGNMENT)
        self.kernel = kernel
        self.slots = None
        self.caches: weakref.WeakKeyDictionary | None = None
        if cache_weights:
            self.caches = weakref.WeakKeyDictionary()
            _WEIGHT_CACHES.add(self)

    def __call__(self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        if self.caches is None:
            return super().__call__(experts, x, num_tokens_per_expert)
        # The shared forward runs the module's GEMMs in a fixed order: the n-th GEMM uses the n-th cache
        self.slots = iter(self.caches.setdefault(experts, [{}, {}, {}]))
        try:
            return super().__call__(experts, x, num_tokens_per_expert)
        finally:
            self.slots = None

    def validate(self, experts: "GroupedExperts") -> None:
        """Gate and up projections share one set of weight scales, so they must be fused; the kernel tiles in
        multiples of 32."""
        if experts.gate_proj is not None:
            raise ValueError("NVFP4 gated experts require the gate_up runtime fusion for shared weight scales.")
        reason = self.kernel.unsupported_shape_reason(experts.down_proj.shape[-2], experts.hidden_dim)
        if reason is not None:
            raise ValueError(reason)


class MXFP8ExpertCompute(GroupedGemmExpertCompute):
    def __init__(self, kernel: ModuleType, high_precision_wgrad: bool) -> None:
        super().__init__(
            partial(kernel.grouped_gemm, high_precision_wgrad=high_precision_wgrad),
            token_group_alignment=kernel.TOKEN_GROUP_ALIGNMENT,
        )
