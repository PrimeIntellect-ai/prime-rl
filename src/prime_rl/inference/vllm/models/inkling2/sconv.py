# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 short convolutions with their state kept in a sliding-window cache.

Each decoder layer owns one `InklingConvState` holding its four conv streams
(K, V, attn-out, mlp-out) packed head-major. K/V are TP-sharded with the KV
heads; the attn-/mlp-output streams are replicated full-width on every rank
(`replicate_hidden=True`).
"""

from typing import cast

import torch
from torch import nn
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.model_executor.utils import set_weight_attrs

from .conv_state import InklingConvState, InklingSconvMetadata
from .ops.linear import linear_k_groups
from .ops.sconv import short_conv

K_STREAM, V_STREAM, ATTN_STREAM, MLP_STREAM = range(4)


def conv_metadata(owner: InklingConvState) -> InklingSconvMetadata | None:
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict) or owner.kv_cache.numel() == 0:
        return None
    return cast(InklingSconvMetadata | None, attn_metadata.get(owner.prefix))


class Inkling2ShortConv(nn.Module):
    """Causal depthwise conv (+ residual, scaled) over one conv-state stream.

    Static: `weight` of shape `(dim, 1, W)`. Dynamic: per-token taps
    `static_w (W, dim) + weight_proj(dynamic_input)` per head. The K/V convs
    only hold weights; the attention prologue kernel runs both.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        owner: InklingConvState,
        stream: int,
        *,
        dynamic_input_size: int | None = None,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.owner = owner
        self.stream = stream
        self.dynamic = dynamic_input_size is not None
        self.tp_rank = get_tensor_model_parallel_rank()
        if dynamic_input_size is not None:
            self.static_w = nn.Parameter(torch.empty(kernel_size, dim), requires_grad=False)
            self.weight_proj = nn.Parameter(
                torch.empty(num_heads * kernel_size, dynamic_input_size),
                requires_grad=False,
            )
            set_weight_attrs(self.static_w, {"weight_loader": self._load_dim1})
            set_weight_attrs(self.weight_proj, {"weight_loader": self._load_dim0})
        else:
            self.weight = nn.Parameter(torch.empty(dim, 1, kernel_size), requires_grad=False)
            set_weight_attrs(self.weight, {"weight_loader": self._load_dim0})

    def _load(self, param: nn.Parameter, loaded: torch.Tensor, dim: int) -> None:
        size = param.shape[dim]
        if loaded.shape[dim] != size:
            loaded = loaded.narrow(dim, self.tp_rank * size, size)
        param.data.copy_(loaded)

    def _load_dim0(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load(param, loaded, 0)

    def _load_dim1(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load(param, loaded, 1)

    @property
    def shard_width(self) -> int:
        """Hidden channels this rank's conv state covers (its KV heads' slice)."""
        offset, width = self.owner.stream_ranges[self.stream]
        return width * self.owner.num_kv_heads

    @property
    def conv_weight(self) -> torch.Tensor:
        """Static `(dim, W)` taps, or `(W, dim)` static part of dynamic taps."""
        return self.static_w if self.dynamic else self.weight.squeeze(1)

    def dynamic_taps(self, x: torch.Tensor) -> torch.Tensor | None:
        """Per-token tap deltas `(T, num_heads * W)` projected from `x`."""
        if not self.dynamic:
            return None
        # Same dispatch as the reference ``linear``: fp32 1024-wide K groups.
        if self.weight_proj.shape[1] >= 2560 and self.weight_proj.shape[0] not in (
            1,
            2,
            4,
        ):
            return linear_k_groups(x.contiguous(), self.weight_proj, x.dtype)
        return nn.functional.linear(x, self.weight_proj)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        addend: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        owner: InklingConvState | None = None,
    ) -> torch.Tensor:
        """Static conv of `x`; with `addend`, returns `(residual + addend)
        + conv(x)` accumulated in fp32 (written to `out` if given).

        `owner` is the conv state to run on: a loop pass after the first has
        its own (default: the layer's)."""
        assert not self.dynamic
        owner = owner or self.owner
        m = conv_metadata(owner)
        if m is None:
            # No conv state (profile run): skip the conv, keep the residual sum.
            if addend is not None:
                acc = addend.float() if residual is None else residual + addend.float()
                x = (acc + x.to(addend.dtype).float()).to(addend.dtype)
            return x if out is None else out.copy_(x)
        offset, width = owner.stream_ranges[self.stream]
        return short_conv(
            x,
            self.conv_weight,
            owner.kv_cache,
            positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            m.query_start,
            offset,
            width,
            addend,
            residual,
            out,
        )
