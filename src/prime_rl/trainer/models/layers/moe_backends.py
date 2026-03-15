# Adapted in part from StepTronOSS MoE utilities:
# - `SteptronOss/steptronoss/model/utils/moe_utils.py`
# The backend-selection layer in this file is Prime-RL-specific.
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Literal

import torch

logger = logging.getLogger(__name__)

try:
    from prime_rl.trainer.models.layers.moe_triton import (
        triton_histogram,
        triton_index_compute,
        triton_index_scatter,
        triton_moe_weighted_gather,
    )
    _TRITON_IMPORT_ERROR: Exception | None = None
except Exception as error:
    triton_histogram = None
    triton_index_compute = None
    triton_index_scatter = None
    triton_moe_weighted_gather = None
    _TRITON_IMPORT_ERROR = error

MoERoutingBackendName = Literal["torch", "triton"]
MoEScatterBackendName = Literal["torch", "triton"]
MoEGatherBackendName = Literal["torch", "triton"]
MoEGroupedFFNBackendName = Literal["torch", "triton"]

_WARNED_BACKENDS: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class MoEBackendSelection:
    routing: MoERoutingBackendName = "torch"
    scatter: MoEScatterBackendName = "torch"
    gather: MoEGatherBackendName = "torch"
    grouped_ffn: MoEGroupedFFNBackendName = "torch"

    @classmethod
    def from_config(cls, config) -> "MoEBackendSelection":
        raw = getattr(config, "moe_backends", None)
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(mode="python")
        if not isinstance(raw, dict):
            raise TypeError("config.moe_backends must be a mapping or MoEBackendSelection")
        return cls(**raw)


@dataclass(frozen=True)
class MoEDispatchInfo:
    top_scores: torch.Tensor
    selected_experts_indices: torch.Tensor
    num_tokens_per_expert: torch.Tensor
    top_scores_experts_sorted: torch.Tensor
    token_indices_experts_sorted: torch.Tensor
    scatter_index: torch.Tensor | None = None


def _torch_histogram(top_k_rank: torch.Tensor, expert_num: int) -> torch.Tensor:
    if expert_num == 0:
        return torch.zeros((0,), device=top_k_rank.device, dtype=torch.int32)

    flat = top_k_rank.reshape(-1)
    if flat.numel() == 0:
        return torch.zeros((expert_num,), device=top_k_rank.device, dtype=torch.int32)

    valid = flat >= 0
    counts = torch.bincount(flat[valid].to(torch.int64), minlength=expert_num)
    return counts.to(torch.int32)


def _torch_index_compute(indices: torch.Tensor, expert_histogram: torch.Tensor) -> torch.Tensor:
    out = indices.to(torch.int32, copy=True)
    if out.numel() == 0:
        return out

    flat = indices.reshape(-1)
    valid = flat >= 0
    if not valid.any():
        return out

    hist = expert_histogram.to(torch.int64)
    base_offset = torch.cumsum(hist, dim=0) - hist

    positions = torch.arange(flat.numel(), device=indices.device, dtype=torch.int64)
    valid_pos = positions[valid]
    valid_expert = flat[valid].to(torch.int64)

    sort_key = valid_expert * flat.numel() + valid_pos
    order = torch.argsort(sort_key)
    sorted_pos = valid_pos[order]
    sorted_expert = valid_expert[order]

    idx = torch.arange(sorted_expert.numel(), device=indices.device, dtype=torch.int64)
    group_change = torch.ones_like(sorted_expert, dtype=torch.bool)
    group_change[1:] = sorted_expert[1:] != sorted_expert[:-1]
    group_start = torch.zeros_like(sorted_expert, dtype=torch.int64)
    group_start[group_change] = idx[group_change]
    group_start = torch.cummax(group_start, dim=0).values
    within = idx - group_start

    scatter_pos_sorted = base_offset[sorted_expert] + within

    flat_out = out.reshape(-1).to(torch.int64)
    flat_out[sorted_pos] = scatter_pos_sorted
    return flat_out.reshape_as(out).to(torch.int32)


class MoEBackends:
    def __init__(self, selection: MoEBackendSelection, num_experts: int, top_k: int):
        self.selection = selection
        self.num_experts = num_experts
        self.top_k = top_k

    def histogram(self, selected_experts_indices: torch.Tensor) -> torch.Tensor:
        if self._resolve_backend("routing", selected_experts_indices) == "triton":
            assert triton_histogram is not None
            return triton_histogram(selected_experts_indices, self.num_experts)
        return _torch_histogram(selected_experts_indices, self.num_experts)

    def use_grouped_ffn(self, x: torch.Tensor) -> bool:
        if self.selection.grouped_ffn != "triton":
            return False
        if triton_histogram is None or triton_index_scatter is None or triton_moe_weighted_gather is None:
            reason = "Triton grouped_ffn backend requires Triton histogram/scatter/gather kernels"
            if _TRITON_IMPORT_ERROR is not None:
                reason = f"{reason}: {_TRITON_IMPORT_ERROR}"
            self._warn_fallback("grouped_ffn", reason)
            return False
        if not x.is_cuda:
            self._warn_fallback("grouped_ffn", "Triton grouped_ffn backend requires CUDA tensors")
            return False
        return True

    def index_compute(self, indices: torch.Tensor, expert_histogram: torch.Tensor) -> torch.Tensor:
        if self._resolve_backend("routing", indices) == "triton":
            assert triton_index_compute is not None
            return triton_index_compute(indices, expert_histogram)
        return _torch_index_compute(indices, expert_histogram)

    def build_dispatch(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> MoEDispatchInfo:
        routing_backend = self._resolve_backend("routing", selected_experts_indices)
        if routing_backend == "triton":
            assert triton_histogram is not None
            assert triton_index_compute is not None
            num_tokens_per_expert = triton_histogram(selected_experts_indices, self.num_experts)
            scatter_index = triton_index_compute(selected_experts_indices, num_tokens_per_expert)
            top_scores_experts_sorted, token_indices_experts_sorted = self._scatter_index_to_expert_order(
                top_scores, scatter_index
            )
            return MoEDispatchInfo(
                top_scores=top_scores,
                selected_experts_indices=selected_experts_indices,
                num_tokens_per_expert=num_tokens_per_expert,
                top_scores_experts_sorted=top_scores_experts_sorted,
                token_indices_experts_sorted=token_indices_experts_sorted,
                scatter_index=scatter_index,
            )

        num_tokens_per_expert = _torch_histogram(selected_experts_indices, self.num_experts)
        flat_selected_experts = selected_experts_indices.reshape(-1)
        token_indices_experts_sorted = torch.argsort(flat_selected_experts, stable=True)
        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k
        return MoEDispatchInfo(
            top_scores=top_scores,
            selected_experts_indices=selected_experts_indices,
            num_tokens_per_expert=num_tokens_per_expert,
            top_scores_experts_sorted=top_scores_experts_sorted,
            token_indices_experts_sorted=token_indices_experts_sorted,
        )

    def ensure_scatter_index(self, dispatch_info: MoEDispatchInfo) -> MoEDispatchInfo:
        if dispatch_info.scatter_index is not None:
            return dispatch_info

        scatter_index = self.index_compute(
            dispatch_info.selected_experts_indices,
            dispatch_info.num_tokens_per_expert,
        )
        return self._with_scatter_index(dispatch_info, scatter_index)

    def dispatch_tokens(
        self,
        x: torch.Tensor,
        dispatch_info: MoEDispatchInfo,
    ) -> tuple[torch.Tensor, MoEDispatchInfo]:
        if self._resolve_backend("scatter", x) == "triton":
            assert triton_index_scatter is not None
            routed_input, scatter_index = triton_index_scatter(
                x,
                dispatch_info.selected_experts_indices,
                dispatch_info.num_tokens_per_expert,
            )
            dispatch_info = self._with_scatter_index(dispatch_info, scatter_index)
            return routed_input, dispatch_info

        if dispatch_info.token_indices_experts_sorted.numel() == 0:
            return x.new_zeros((0, x.shape[-1])), dispatch_info

        token_indices = dispatch_info.token_indices_experts_sorted.reshape(-1, 1).expand(-1, x.shape[-1])
        routed_input = torch.gather(x, dim=0, index=token_indices)
        return routed_input, dispatch_info

    def combine_outputs(
        self,
        routed_output: torch.Tensor,
        dispatch_info: MoEDispatchInfo,
        token_count: int,
        hidden_dim: int,
        score_before_experts: bool,
    ) -> torch.Tensor:
        if self._resolve_backend("gather", routed_output) == "triton":
            dispatch_info = self.ensure_scatter_index(dispatch_info)
            assert dispatch_info.scatter_index is not None
            assert triton_moe_weighted_gather is not None
            weights = torch.ones_like(dispatch_info.top_scores) if score_before_experts else dispatch_info.top_scores
            return triton_moe_weighted_gather(routed_output, dispatch_info.scatter_index, weights)

        if not score_before_experts and dispatch_info.top_scores_experts_sorted.numel() > 0:
            routed_output = (
                routed_output.to(torch.float32) * dispatch_info.top_scores_experts_sorted.reshape(-1, 1)
            ).to(routed_output.dtype)

        expert_output = routed_output.new_zeros((token_count, hidden_dim))
        if dispatch_info.token_indices_experts_sorted.numel() == 0:
            return expert_output

        token_indices = dispatch_info.token_indices_experts_sorted.reshape(-1, 1).expand(-1, hidden_dim)
        expert_output.scatter_add_(0, token_indices, routed_output)
        return expert_output

    def run_routed_experts(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        experts: torch.nn.Module,
        score_before_experts: bool,
    ) -> torch.Tensor:
        dispatch_info = self.build_dispatch(top_scores, selected_experts_indices)
        routed_input, dispatch_info = self.dispatch_tokens(x, dispatch_info)

        if score_before_experts and dispatch_info.top_scores_experts_sorted.numel() > 0:
            routed_input = (
                routed_input.to(torch.float32) * dispatch_info.top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        routed_output = experts(routed_input, dispatch_info.num_tokens_per_expert)
        return self.combine_outputs(
            routed_output=routed_output,
            dispatch_info=dispatch_info,
            token_count=x.shape[0],
            hidden_dim=x.shape[1],
            score_before_experts=score_before_experts,
        )

    def run_grouped_ffn(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        experts: torch.nn.Module,
        score_before_experts: bool,
    ) -> torch.Tensor:
        assert self.use_grouped_ffn(x)
        assert triton_histogram is not None
        assert triton_index_scatter is not None
        assert triton_moe_weighted_gather is not None

        num_tokens_per_expert = triton_histogram(selected_experts_indices, self.num_experts)
        routed_input, scatter_index = triton_index_scatter(x, selected_experts_indices, num_tokens_per_expert)

        if score_before_experts:
            top_scores_experts_sorted, _ = self._scatter_index_to_expert_order(top_scores, scatter_index)
            if top_scores_experts_sorted.numel() > 0:
                routed_input = (
                    routed_input.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)
                ).to(x.dtype)
            weights = torch.ones_like(top_scores)
        else:
            weights = top_scores

        routed_output = experts.forward_grouped_ffn(routed_input, num_tokens_per_expert)
        return triton_moe_weighted_gather(routed_output, scatter_index, weights)

    def _resolve_backend(self, kind: Literal["routing", "scatter", "gather"], tensor: torch.Tensor) -> str:
        requested_backend = getattr(self.selection, kind)
        if requested_backend != "triton":
            return "torch"

        missing_kernels = {
            "routing": triton_histogram is None or triton_index_compute is None,
            "scatter": triton_index_scatter is None,
            "gather": triton_moe_weighted_gather is None,
        }
        if missing_kernels[kind]:
            reason = "Triton kernels are unavailable"
            if _TRITON_IMPORT_ERROR is not None:
                reason = f"{reason}: {_TRITON_IMPORT_ERROR}"
            self._warn_fallback(kind, reason)
            return "torch"

        if not tensor.is_cuda:
            self._warn_fallback(kind, "Triton backends require CUDA tensors")
            return "torch"
        return "triton"

    def _scatter_index_to_expert_order(
        self,
        top_scores: torch.Tensor,
        scatter_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_scatter_index = scatter_index.reshape(-1)
        valid = flat_scatter_index >= 0
        out_size = int(valid.sum().item())
        if out_size == 0:
            return (
                top_scores.new_zeros((0,)),
                torch.zeros((0,), device=top_scores.device, dtype=torch.long),
            )

        token_num = top_scores.shape[0]
        flat_top_scores = top_scores.reshape(-1)
        flat_token_indices = torch.arange(token_num, device=top_scores.device, dtype=torch.long).repeat_interleave(
            self.top_k
        )

        scatter_pos = flat_scatter_index[valid].to(torch.int64)
        top_scores_experts_sorted = top_scores.new_empty((out_size,))
        token_indices_experts_sorted = torch.empty((out_size,), device=top_scores.device, dtype=torch.long)
        top_scores_experts_sorted[scatter_pos] = flat_top_scores[valid]
        token_indices_experts_sorted[scatter_pos] = flat_token_indices[valid]
        return top_scores_experts_sorted, token_indices_experts_sorted

    def _with_scatter_index(self, dispatch_info: MoEDispatchInfo, scatter_index: torch.Tensor) -> MoEDispatchInfo:
        top_scores_experts_sorted, token_indices_experts_sorted = self._scatter_index_to_expert_order(
            dispatch_info.top_scores,
            scatter_index,
        )
        return replace(
            dispatch_info,
            scatter_index=scatter_index,
            top_scores_experts_sorted=top_scores_experts_sorted,
            token_indices_experts_sorted=token_indices_experts_sorted,
        )

    def _warn_fallback(self, kind: str, reason: str) -> None:
        key = (kind, reason)
        if key in _WARNED_BACKENDS:
            return
        _WARNED_BACKENDS.add(key)
        logger.warning("Falling back to torch MoE %s backend: %s", kind, reason)
