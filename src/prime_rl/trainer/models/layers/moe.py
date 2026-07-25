# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.configs.trainer import EPCommBackend
from prime_rl.trainer.distributed.expert_parallel import expert_parallel


@dataclass
class SwiGLuMoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1

    # router
    score_func: Literal["softmax", "sigmoid", "topk_softmax"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True

    # token-choice
    top_k: int = 1
    load_balance_coeff: float | None = 1e-3
    fp8: bool = False  # use FP8 grouped GEMM via DeepGEMM (requires SM90)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def fused_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return swiglu(gate, up)


def interleaved_clamped_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(max=7.0)
    up = up.clamp(min=-7.0, max=7.0)
    return (up + 1) * gate * torch.sigmoid(gate * 1.702)


@torch.compile(dynamic=True)
def relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


class FeedForward(nn.Module):
    """Dense feed-forward layer with named input and output projections."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        input_projection_names: tuple[str, ...] = ("w1", "w3"),
        output_projection_name: str = "w2",
        activation_fn: Callable[..., torch.Tensor] = swiglu,
    ) -> None:
        super().__init__()
        self.input_projection_names = input_projection_names
        self.output_projection_name = output_projection_name
        self.activation_fn = activation_fn
        for name in input_projection_names:
            self.add_module(name, nn.Linear(dim, hidden_dim, bias=False))
        self.add_module(output_projection_name, nn.Linear(hidden_dim, dim, bias=False))

    @staticmethod
    def remove_weight_suffix_for_state_dict(
        module: "FeedForward",
        state_dict: dict[str, Any],
        prefix: str,
        _local_metadata: dict[str, Any],
    ) -> None:
        names = (*module.input_projection_names, module.output_projection_name)
        for name in names:
            state_dict[prefix + name] = state_dict.pop(prefix + name + ".weight")

    @staticmethod
    def restore_weight_suffix_from_state_dict(
        module: "FeedForward",
        state_dict: dict[str, Any],
        prefix: str,
        _local_metadata: dict[str, Any],
        _strict: bool,
        _missing_keys: list[str],
        _unexpected_keys: list[str],
        _error_msgs: list[str],
    ) -> None:
        names = (*module.input_projection_names, module.output_projection_name)
        for name in names:
            state_dict[prefix + name + ".weight"] = state_dict.pop(prefix + name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = [getattr(self, name)(x) for name in self.input_projection_names]
        return getattr(self, self.output_projection_name)(self.activation_fn(*projected))

    def init_weights(self, init_std: float = 0.02):
        input_projections = [getattr(self, name) for name in self.input_projection_names]
        nn.init.trunc_normal_(input_projections[0].weight, mean=0.0, std=0.02)
        output_projection = getattr(self, self.output_projection_name)
        for linear in (*input_projections[1:], output_projection):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


def broadcast_expert_bias(
    bias: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    target_rows: int,
) -> torch.Tensor:
    bias = torch.repeat_interleave(bias, num_tokens_per_expert, dim=0)
    if bias.shape[0] < target_rows:
        bias = F.pad(bias, (0, 0, 0, target_rows - bias.shape[0]))
    return bias


@expert_parallel
def _run_experts_grouped_mm(
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    *,
    grouped_mm_fn: Callable[..., torch.Tensor],
    activation_fn: Callable[[torch.Tensor], torch.Tensor],
    input_bias: torch.Tensor | None = None,
    output_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _run_experts_grouped_mm_impl(
        input_weight,
        output_weight,
        x,
        num_tokens_per_expert,
        grouped_mm_fn=grouped_mm_fn,
        activation_fn=activation_fn,
        input_bias=input_bias,
        output_bias=output_bias,
    )


def _run_experts_grouped_mm_impl(
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    *,
    grouped_mm_fn: Callable[..., torch.Tensor],
    activation_fn: Callable[[torch.Tensor], torch.Tensor],
    input_bias: torch.Tensor | None = None,
    output_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    assert x.dim() == 2

    projected = grouped_mm_fn(x.bfloat16(), input_weight.bfloat16(), offs=offsets)

    if input_bias is not None:
        projected = projected + broadcast_expert_bias(input_bias, num_tokens_per_expert, projected.shape[0]).bfloat16()

    hidden = activation_fn(projected)
    out = grouped_mm_fn(hidden, output_weight.bfloat16(), offs=offsets)

    if output_bias is not None:
        out = out + broadcast_expert_bias(output_bias, num_tokens_per_expert, out.shape[0]).bfloat16()
    return out.type_as(x)


class GroupedExperts(nn.Module):
    @staticmethod
    def export_weights_to_state_dict(
        module: "GroupedExperts",
        state_dict: dict[str, Any],
        prefix: str,
        _local_metadata: dict[str, Any],
    ) -> None:
        input_weights = state_dict.pop(prefix + "input_weight").split(module.input_weight_sizes, dim=1)
        output_weight = state_dict.pop(prefix + "w2")
        if module.transpose_weights_for_state_dict:
            input_weights = tuple(weight.transpose(-2, -1) for weight in input_weights)
            output_weight = output_weight.transpose(-2, -1)
        for name, weight in zip(module.input_weight_names, input_weights, strict=True):
            state_dict[prefix + name] = weight
        state_dict[prefix + module.output_weight_name] = output_weight
        if module.input_bias_name is not None:
            state_dict[prefix + module.input_bias_name] = state_dict.pop(prefix + "input_bias")
        if module.output_bias_name is not None:
            state_dict[prefix + module.output_bias_name] = state_dict.pop(prefix + "output_bias")

    @staticmethod
    def import_weights_from_state_dict(
        module: "GroupedExperts",
        state_dict: dict[str, Any],
        prefix: str,
        _local_metadata: dict[str, Any],
        _strict: bool,
        _missing_keys: list[str],
        _unexpected_keys: list[str],
        _error_msgs: list[str],
    ) -> None:
        input_weights = [state_dict.pop(prefix + name) for name in module.input_weight_names]
        output_weight = state_dict.pop(prefix + module.output_weight_name)
        if module.transpose_weights_for_state_dict:
            input_weights = [weight.transpose(-2, -1) for weight in input_weights]
            output_weight = output_weight.transpose(-2, -1)
        state_dict[prefix + "input_weight"] = torch.cat(input_weights, dim=1)
        state_dict[prefix + "w2"] = output_weight
        if module.input_bias_name is not None:
            state_dict[prefix + "input_bias"] = state_dict.pop(prefix + module.input_bias_name)
        if module.output_bias_name is not None:
            state_dict[prefix + "output_bias"] = state_dict.pop(prefix + module.output_bias_name)

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        *,
        input_weight_names: tuple[str, ...] = ("w1", "w3"),
        input_weight_sizes: tuple[int, ...] | None = None,
        output_weight_name: str = "w2",
        input_bias_name: str | None = None,
        output_bias_name: str | None = None,
        transpose_weights_for_state_dict: bool = False,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = fused_swiglu,
        grouped_mm_fn: Callable[..., torch.Tensor] = torch._grouped_mm,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.input_weight_names = input_weight_names
        self.input_weight_sizes = input_weight_sizes or (hidden_dim,) * len(input_weight_names)
        self.output_weight_name = output_weight_name
        self.input_bias_name = input_bias_name
        self.output_bias_name = output_bias_name
        self.transpose_weights_for_state_dict = transpose_weights_for_state_dict
        self.input_weight = nn.Parameter(torch.empty(num_experts, sum(self.input_weight_sizes), dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.input_bias = (
            nn.Parameter(torch.empty(num_experts, sum(self.input_weight_sizes)))
            if input_bias_name is not None
            else None
        )
        self.output_bias = nn.Parameter(torch.empty(num_experts, dim)) if output_bias_name is not None else None
        self._activation_fn = activation_fn
        self._grouped_mm_fn = grouped_mm_fn
        self.ep_comm_backend: EPCommBackend = "torch"
        self.register_state_dict_post_hook(self.export_weights_to_state_dict)
        self.register_load_state_dict_pre_hook(self.import_weights_from_state_dict)

    @property
    def w1(self) -> torch.Tensor:
        index = self.input_weight_names.index("w1")
        return self.input_weight.split(self.input_weight_sizes, dim=1)[index]

    @property
    def w3(self) -> torch.Tensor:
        index = self.input_weight_names.index("w3")
        return self.input_weight.split(self.input_weight_sizes, dim=1)[index]

    @property
    def gate_up_proj(self) -> torch.Tensor:
        index = self.input_weight_names.index("gate_up_proj")
        weight = self.input_weight.split(self.input_weight_sizes, dim=1)[index]
        return weight.transpose(-2, -1) if self.transpose_weights_for_state_dict else weight

    @property
    def down_proj(self) -> torch.Tensor:
        return self.w2.transpose(-2, -1) if self.transpose_weights_for_state_dict else self.w2

    @property
    def gate_up_proj_bias(self) -> torch.Tensor | None:
        return self.input_bias

    @property
    def down_proj_bias(self) -> torch.Tensor | None:
        return self.output_bias

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend

    def _forward_deepep(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        return _run_experts_grouped_mm_impl(
            self.input_weight.to_local().transpose(-2, -1),
            self.w2.to_local().transpose(-2, -1),
            x,
            num_tokens_per_expert,
            grouped_mm_fn=self._grouped_mm_fn,
            activation_fn=self._activation_fn,
            input_bias=self.input_bias.to_local() if self.input_bias is not None else None,
            output_bias=self.output_bias.to_local() if self.output_bias is not None else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if self.ep_comm_backend == "deepep":
            return self._forward_deepep(x, num_tokens_per_expert)

        return _run_experts_grouped_mm(
            self.input_weight.transpose(-2, -1),
            self.w2.transpose(-2, -1),
            x,
            num_tokens_per_expert,
            grouped_mm_fn=self._grouped_mm_fn,
            activation_fn=self._activation_fn,
            input_bias=self.input_bias,
            output_bias=self.output_bias,
        )

    def init_weights(self, init_std: float):
        input_weights = self.input_weight.split(self.input_weight_sizes, dim=1)
        nn.init.trunc_normal_(input_weights[0], mean=0.0, std=0.02)
        for weight in (*input_weights[1:], self.w2):
            nn.init.trunc_normal_(weight, mean=0.0, std=init_std)
        if self.input_bias is not None:
            nn.init.zeros_(self.input_bias)
        if self.output_bias is not None:
            nn.init.zeros_(self.output_bias)


def _selected_probability_mass_sum(
    scores: torch.Tensor, top_scores: torch.Tensor, score_func: Literal["softmax", "sigmoid"]
) -> torch.Tensor:
    with torch.no_grad():
        if score_func == "softmax":
            return top_scores.sum()
        selected_prob_mass = top_scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        return selected_prob_mass.sum(dim=-1).sum()


class TokenChoiceTopKRouter(nn.Module):
    """Route each token to its top-k experts.

    Args:
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        score_func (Literal["softmax", "sigmoid", "topk_softmax"]): Score transform. ``topk_softmax``
            selects experts from the logits and normalizes only the selected logits.
        route_norm (bool): Whether to normalize the routing scores when using sigmoid.
        route_scale (float): Scaling factor applied to the routing scores.
        gate_bias (bool): Whether the gate has a trainable logit bias.
        weight_state_dict_name (str): Checkpoint/state-dict name for the gate weight.
        bias_state_dict_name (str): Checkpoint/state-dict name for the gate bias.
        selection_bias_state_dict_name (str | None): Checkpoint/state-dict name for a persistent
            selection-only bias. When set, the bias affects expert selection but not routing weights.
    """

    @staticmethod
    def export_parameters_to_state_dict(
        module: "TokenChoiceTopKRouter",
        state_dict: dict[str, Any],
        prefix: str,
        _local_metadata: dict[str, Any],
    ) -> None:
        for runtime_name, state_dict_name in module.runtime_to_state_dict_names.items():
            state_dict[prefix + state_dict_name] = state_dict.pop(prefix + runtime_name)

    @staticmethod
    def import_parameters_from_state_dict(
        module: "TokenChoiceTopKRouter",
        state_dict: dict[str, Any],
        prefix: str,
        _local_metadata: dict[str, Any],
        _strict: bool,
        _missing_keys: list[str],
        _unexpected_keys: list[str],
        _error_msgs: list[str],
    ) -> None:
        for runtime_name, state_dict_name in module.runtime_to_state_dict_names.items():
            state_dict[prefix + runtime_name] = state_dict.pop(prefix + state_dict_name)

    def _fqn_modifiers(self) -> dict[str, str]:
        modifiers = {}
        for runtime_name, state_dict_name in self.runtime_to_state_dict_names.items():
            runtime_parent, runtime_attribute = runtime_name.rsplit(".", 1)
            if state_dict_name == runtime_attribute:
                modifiers[state_dict_name] = runtime_parent
        return modifiers

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid", "topk_softmax"],
        route_norm: bool,
        route_scale: float,
        *,
        gate_bias: bool = False,
        weight_state_dict_name: str = "gate.weight",
        bias_state_dict_name: str = "gate.bias",
        selection_bias_state_dict_name: str | None = None,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=gate_bias)
        self.selection_bias_state_dict_name = selection_bias_state_dict_name
        if selection_bias_state_dict_name is not None:
            self.register_buffer(selection_bias_state_dict_name, torch.zeros(num_experts))
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.runtime_to_state_dict_names = {"gate.weight": weight_state_dict_name}
        if gate_bias:
            self.runtime_to_state_dict_names["gate.bias"] = bias_state_dict_name
        self.force_balanced = False
        # Set via model.moe_router_dtype='float32': the gate weight is kept in fp32
        # (exempt from FSDP bf16 casting) and the gate GEMM runs in fp32.
        self.fp32_gate = False
        self.register_state_dict_post_hook(self.export_parameters_to_state_dict)
        self.register_load_state_dict_pre_hook(self.import_parameters_from_state_dict)

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None, routed_experts: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.
            routed_experts (torch.Tensor | None, optional): Optional tensor with shape ``(bs * slen, top_k)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
                - routing_confidence_sum (torch.Tensor):
                    Sum over tokens of the selected-expert probability mass before route normalization/scaling.
        """
        # scores shape (bs*slen, num_experts)
        assert routed_experts is None or routed_experts.shape[-1] == self.top_k, (
            f"routed_experts shape: {routed_experts.shape}, top_k: {self.top_k}"
        )
        if self.fp32_gate:
            gate_bias = self.gate.bias.float() if self.gate.bias is not None else None
            logits = F.linear(x.float(), self.gate.weight.float(), gate_bias)
        else:
            logits = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(logits.float())
        elif self.score_func == "softmax":
            scores = F.softmax(logits.float(), dim=1)
        elif self.score_func == "topk_softmax":
            scores = logits
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        # top scores shape (bs*slen, top_k)
        # NOTE: selection biases are only used for routing. The gating value
        #       top_scores is still derived from the original scores/logits.

        if routed_experts is not None:
            top_scores = scores.gather(dim=1, index=routed_experts)
            selected_experts_indices = routed_experts
        elif self.force_balanced:
            num_tokens = scores.shape[0]
            arange = torch.arange(num_tokens * self.top_k, device=scores.device)
            selected_experts_indices = (arange % self.num_experts).view(num_tokens, self.top_k)
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            selection_scores = scores
            if self.selection_bias_state_dict_name is not None:
                selection_scores = selection_scores + self.get_buffer(self.selection_bias_state_dict_name)
            if expert_bias is not None:
                selection_scores = selection_scores + expert_bias
            _, selected_experts_indices = torch.topk(selection_scores, k=self.top_k, dim=1)
            top_scores = scores.gather(dim=1, index=selected_experts_indices)

        if self.score_func == "topk_softmax":
            top_scores = F.softmax(top_scores, dim=-1, dtype=top_scores.dtype)
            with torch.no_grad():
                routing_confidence_sum = top_scores.sum()
        else:
            routing_confidence_sum = _selected_probability_mass_sum(scores, top_scores, self.score_func)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.reshape(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_scores, selected_experts_indices, num_tokens_per_expert, routing_confidence_sum

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)


# NOTE: the reason we make this a stateless module is to support
#       expert_tensor_parallel_degree=1 with consistent TP/EP APIs.
class TokenReorderer(nn.Module):
    """
    This module reorders token indices to match the order of experts, enabling
    efficient parallel processing of tokens by experts.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of experts each token will be routed to.
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reorders token indices to match the order of experts for MoE routing.

        Args:
            top_scores (torch.Tensor): Routing scores for selected experts,
                shape (batch_size*seq_len, top_k)
            selected_experts_indices (torch.Tensor): Expert indices selected for each token,
                shape (batch_size*seq_len, top_k)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores_experts_sorted: Scores reordered to match expert ordering
                - token_indices_experts_sorted: Token indices reordered to match expert ordering
                - num_tokens_per_expert: Number of tokens assigned to each expert
        """
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        selected_experts_indices = selected_experts_indices.reshape(-1)
        num_tokens_per_expert = torch.histc(
            selected_experts_indices,
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(selected_experts_indices, stable=True)

        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )


class MoE(nn.Module):
    """Token-choice MoE runtime composed from a router, grouped experts, and optional projections."""

    @classmethod
    def from_swiglu(cls, args: SwiGLuMoEArgs, dim: int, hidden_dim: int) -> "MoE":
        grouped_mm_fn = torch._grouped_mm
        if args.fp8:
            from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm

            grouped_mm_fn = grouped_fp8_gemm

        experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=args.num_experts,
            grouped_mm_fn=grouped_mm_fn,
        )
        router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=args.num_experts,
            top_k=args.top_k,
            score_func=args.score_func,
            route_norm=args.route_norm,
            route_scale=args.route_scale,
        )
        shared_expert = (
            FeedForward(dim=dim, hidden_dim=hidden_dim * args.num_shared_experts)
            if args.num_shared_experts > 0
            else None
        )
        if shared_expert is not None:
            shared_expert.register_state_dict_post_hook(FeedForward.remove_weight_suffix_for_state_dict)
            shared_expert.register_load_state_dict_pre_hook(FeedForward.restore_weight_suffix_from_state_dict)

        return cls(
            router=router,
            experts=experts,
            shared_expert=shared_expert,
            score_before_experts=args.score_before_experts,
            load_balance_coeff=args.load_balance_coeff,
        )

    def __init__(
        self,
        *,
        router: TokenChoiceTopKRouter,
        experts: GroupedExperts,
        shared_expert: FeedForward | None,
        score_before_experts: bool,
        load_balance_coeff: float | None,
        routed_input_projection: tuple[str, nn.Module] | None = None,
        routed_output_projection: tuple[str, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.router = router
        self.experts = experts
        self.shared_expert = shared_expert
        self.score_before_experts = score_before_experts
        self.routed_input_projection_name = None
        if routed_input_projection is not None:
            self.routed_input_projection_name, projection = routed_input_projection
            self.add_module(self.routed_input_projection_name, projection)
        self.routed_output_projection_name = None
        if routed_output_projection is not None:
            self.routed_output_projection_name, projection = routed_output_projection
            self.add_module(self.routed_output_projection_name, projection)

        self.ep_comm_backend: EPCommBackend = "torch"
        self.experts.set_ep_comm_backend(self.ep_comm_backend)
        self.reorderer = TokenReorderer(num_experts=experts.num_experts, top_k=router.top_k)
        self.deepep_token_chunk_size: int | None = None

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(experts.num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(experts.num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("routing_confidence_sum", torch.tensor(0.0, dtype=torch.float32), persistent=False)

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend
        self.experts.set_ep_comm_backend(backend)

    def set_deepep_token_chunk_size(self, chunk_size: int | None) -> None:
        self.deepep_token_chunk_size = chunk_size

    def _run_local_routed_experts(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        return self.experts(x, num_tokens_per_expert)

    def _run_routed_experts(
        self,
        x: torch.Tensor,
        token_indices_experts_sorted: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        top_scores_experts_sorted: torch.Tensor,
    ) -> torch.Tensor:
        dim = x.shape[-1]
        routed_indices = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
        routed_input = torch.gather(x, dim=0, index=routed_indices)

        if self.routed_input_projection_name is not None:
            routed_input = getattr(self, self.routed_input_projection_name)(routed_input)

        if self.score_before_experts:
            routed_input = (routed_input.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)).to(x.dtype)

        routed_output = self.experts(routed_input, num_tokens_per_expert)

        if not self.score_before_experts:
            routed_output = (routed_output.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)).to(x.dtype)

        if self.routed_output_projection_name is not None:
            routed_output = getattr(self, self.routed_output_projection_name)(routed_output)

        return routed_output

    def _run_deepep_routed_experts(
        self,
        x: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> torch.Tensor:
        from prime_rl.trainer.distributed.deepep import (
            combine_tokens,
            dispatch_tokens_async,
            finalize_dispatch_tokens,
            sync_combine,
        )
        from prime_rl.trainer.distributed.expert_parallel import get_ep_group

        if x.shape[0] == 0:
            shared_output = self.shared_expert(x) if self.shared_expert is not None else None
            return x.new_zeros(x.shape) if shared_output is None else shared_output

        group = get_ep_group(self.experts)
        routed_input = x
        if self.routed_input_projection_name is not None:
            routed_input = getattr(self, self.routed_input_projection_name)(routed_input)
        chunk_size = min(self.deepep_token_chunk_size or routed_input.shape[0], routed_input.shape[0])

        def dispatch_chunk(start: int, end: int):
            return dispatch_tokens_async(
                routed_input[start:end],
                selected_experts_indices[start:end],
                top_scores[start:end],
                num_experts=self.experts.num_experts,
                group=group,
                score_before_experts=self.score_before_experts,
            )

        def run_pending_chunk(pending_state):
            hidden_states, num_tokens_per_expert, dispatch_state = finalize_dispatch_tokens(pending_state)
            routed_output = self._run_local_routed_experts(hidden_states, num_tokens_per_expert)
            # Keep combine outside the checkpointed routed-expert region so
            # selective AC only recomputes local expert matmuls.
            return combine_tokens(routed_output, dispatch_state)

        pending_state = dispatch_chunk(0, chunk_size)
        routed_outputs: list[torch.Tensor] = []

        for chunk_start in range(chunk_size, routed_input.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, routed_input.shape[0])
            next_pending_state = dispatch_chunk(chunk_start, chunk_end)
            routed_outputs.append(run_pending_chunk(pending_state))
            pending_state = next_pending_state

        routed_outputs.append(run_pending_chunk(pending_state))

        shared_output = self.shared_expert(x) if self.shared_expert is not None else None
        sync_combine()
        routed_output = routed_outputs[0] if len(routed_outputs) == 1 else torch.cat(routed_outputs, dim=0)
        if self.routed_output_projection_name is not None:
            routed_output = getattr(self, self.routed_output_projection_name)(routed_output)
        return routed_output if shared_output is None else shared_output + routed_output

    def forward(
        self,
        x: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.
            routed_experts (torch.Tensor | None, optional): Optional tensor with shape ``(bs, slen, top_k)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        if routed_experts is not None:
            _, _, top_k = routed_experts.shape
            routed_experts = routed_experts.reshape(
                -1, top_k
            )  # we have to reshape here because the original is non-contiguous

        # top_scores and selected_experts_indices shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
            routing_confidence_sum,
        ) = self.router(x, self.expert_bias, routed_experts=routed_experts)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # Full block checkpointing can double count tokens_per_expert because it reruns the router
        # in backward. The selective MoE path avoids that by checkpointing only the
        # routed expert compute below.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)
            self.routing_confidence_sum.add_(routing_confidence_sum)

        if self.ep_comm_backend == "deepep":
            routed_output = self._run_deepep_routed_experts(x, selected_experts_indices, top_scores)
            return routed_output.reshape(bs, slen, dim)

        # top_scores and token_indices_experts_sorted shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        # NOTE: the reason we need to compute num_tokens_per_expert again is:
        #       1st computation in router is to update self.tokens_per_expert
        #       which would be the same across all TP ranks.
        #       2nd computation in reorderer is for the actual routing and experts computation
        #       which would be sharded over TP ranks if expert_tensor_parallel_degree==1.
        #       If tensor_paralllel_degree == expert_tensor_parallel_degree, they agree.
        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        routed_output = self._run_routed_experts(
            x,
            token_indices_experts_sorted,
            num_tokens_per_expert,
            top_scores_experts_sorted,
        )
        if self.shared_expert is not None:
            out = self.shared_expert(x)
        else:
            out = torch.zeros_like(x)

        routed_indices = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
        out = out.scatter_add(dim=0, index=routed_indices, src=routed_output)
        out = out.reshape(bs, slen, dim)
        return out

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_expert is not None:
            self.shared_expert.init_weights(init_std)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(self.experts.num_experts, dtype=torch.float32)
            self.routing_confidence_sum = torch.tensor(0.0, dtype=torch.float32)
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(self.experts.num_experts, dtype=torch.float32)
