"""Shared eager Qwen3/GLM4 MoE arithmetic for EP1/TP1 mismatch experiments."""

from functools import lru_cache
from types import MethodType

import torch
from torch import Tensor

from prime_rl.trainer.models.layers.dense_alignment import _LogSoftmax, linear_forward
from prime_rl.trainer.models.layers.inference_swiglu import InferenceSilu


@lru_cache(maxsize=1)
def _router_kernel():
    import prime_kernels

    return prime_kernels.load("mismatch_router")


def router_linear(x: Tensor, weight: Tensor) -> Tensor:
    return _router_kernel().linear(x, weight)


class _RouterLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return router_linear(x, weight)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        return (grad @ weight.float()).to(x.dtype), (grad.T @ x.float()).to(weight.dtype)


def route(
    logits: Tensor,
    top_k: int,
    normalize: bool,
    *,
    score_func: str = "softmax",
    selection_bias: Tensor | None = None,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    if score_func == "softmax":
        scores = _LogSoftmax.apply(logits).exp()
    elif score_func == "sigmoid":
        scores = logits.sigmoid()
    else:
        raise ValueError(f"Unsupported aligned routing function: {score_func}")
    selection_scores = scores if selection_bias is None else scores + selection_bias
    indices = selection_scores.argsort(dim=-1, descending=True, stable=True)[:, :top_k]
    indices = indices.sort(dim=-1).values
    selected = scores.gather(1, indices)
    with torch.no_grad():
        confidence = selected.sum() if score_func == "softmax" else (selected / scores.sum(-1, keepdim=True)).sum()
    if normalize:
        denominator = selected[:, :1]
        for slot in range(1, top_k):
            denominator = denominator + selected[:, slot : slot + 1]
        selected = selected / denominator
    return selected * scale, indices, confidence


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, offsets):
        boundaries = [0, *offsets.tolist()]
        output = x.new_empty((x.shape[0], weight.shape[-1]))
        for expert, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
            if stop > start:
                output[start:stop] = linear_forward(x[start:stop], weight[expert].T)
        ctx.save_for_backward(x, weight)
        ctx.boundaries = boundaries
        return output

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight)
        for expert, (start, stop) in enumerate(zip(ctx.boundaries, ctx.boundaries[1:])):
            if stop > start:
                dx[start:stop] = grad[start:stop] @ weight[expert].T
                dw[expert] = x[start:stop].T @ grad[start:stop]
        return dx, dw, None


class AlignedGroupedGemm:
    token_group_alignment = 1

    def __call__(self, x, weight, *, offs):
        return _GroupedLinear.apply(x, weight, offs)


class AlignedLocalDispatcher:
    def __init__(self, num_experts):
        self.num_experts = num_experts

    def run(self, x, top_scores, selected_experts_indices, experts, *, score_before_experts):
        if score_before_experts:
            raise ValueError("Aligned Qwen3 MoE requires scores after experts")
        top_k = selected_experts_indices.shape[1]
        permutation = selected_experts_indices.flatten().argsort(stable=True)
        counts = torch.bincount(selected_experts_indices.flatten(), minlength=self.num_experts)
        routed = experts(x[permutation // top_k], counts)
        inverse = permutation.argsort()
        routed = routed[inverse].reshape(x.shape[0], top_k, x.shape[-1])
        output = routed[:, 0].float() * top_scores[:, :1]
        for slot in range(1, top_k):
            output = output + routed[:, slot].float() * top_scores[:, slot : slot + 1]
        return output.to(x.dtype)

    def synchronize(self):
        return None


def _router_forward(self, x, routed_experts=None):
    if routed_experts is not None or self.force_balanced or not self.fp32_gate:
        raise ValueError("Aligned Qwen3 MoE requires its live FP32 router")
    if self.gate.bias is not None:
        raise ValueError("Aligned MoE requires router logits without a trainable bias")
    scores, indices, confidence = route(
        _RouterLinear.apply(x, self.gate.weight),
        self.top_k,
        self.route_norm,
        score_func=self.score_func,
        selection_bias=self.selection_bias,
        scale=self.route_scale,
    )
    counts = torch.bincount(indices.flatten(), minlength=self.num_experts)
    return scores, indices, counts, confidence


def enable_trainer_moe_alignment(model):
    from prime_rl.trainer.models.layers.moe import MoE

    if model.config.model_type not in ("qwen3_moe", "glm4_moe"):
        raise ValueError("MoE alignment supports Qwen3 MoE and GLM4 MoE")
    if getattr(model.config, "n_group", 1) != 1:
        raise ValueError("MoE alignment requires a single expert group")
    original_keep_fp32 = model.keep_in_fp32_for_weight_transfer
    model.keep_in_fp32_for_weight_transfer = lambda name: (
        name.endswith("mlp.router.gate.weight") or original_keep_fp32(name)
    )
    for module in model.modules():
        if isinstance(module, MoE):
            module.router.forward = MethodType(_router_forward, module.router)
            module.experts.grouped_gemm = AlignedGroupedGemm()
            module.experts.activation = InferenceSilu
            module.token_dispatcher = AlignedLocalDispatcher(module.experts.num_experts)


def enable_serving_moe_alignment():
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    initialize = Qwen3MoeSparseMoeBlock.__init__

    def init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        if self.tp_size != 1 or self.ep_size != 1 or self.shared_expert is not None or self.enable_eplb:
            raise ValueError("Aligned Qwen3 MoE requires TP1/EP1 without shared experts or load balancing")
        backend = self.experts.routed_experts.quant_method.unquantized_backend
        if backend.name != "TRITON":
            raise ValueError("Aligned Qwen3 MoE requires unquantized Triton weight layouts")
        self.gate.weight.data = self.gate.weight.data.float()
        self._alignment_dispatcher = AlignedLocalDispatcher(self.n_routed_experts)

    def forward(self, hidden_states):
        if hidden_states.shape[0] == 0:
            return hidden_states
        layer = self.experts.routed_experts
        scores, indices, _ = route(
            router_linear(hidden_states, self.gate.weight),
            self.experts.moe_config.experts_per_token,
            self.experts.router.renormalize,
        )

        def experts(x, counts):
            offsets = counts.cumsum(0)
            gate_up = _GroupedLinear.apply(x, layer.w13_weight.transpose(-1, -2), offsets)
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = InferenceSilu.apply(gate, up)
            return _GroupedLinear.apply(hidden, layer.w2_weight.transpose(-1, -2), offsets)

        return self._alignment_dispatcher.run(hidden_states, scores, indices, experts, score_before_experts=False)

    Qwen3MoeSparseMoeBlock.__init__ = init
    Qwen3MoeSparseMoeBlock.forward = forward


def enable_serving_glm_alignment():
    from vllm.model_executor.models.glm4_moe import Glm4MoE

    initialize = Glm4MoE.__init__

    def init(self, config, *args, **kwargs):
        initialize(self, config, *args, **kwargs)
        if self.tp_size != 1 or self.ep_size != 1 or self.enable_eplb or config.n_group != 1:
            raise ValueError("Aligned GLM requires TP1/EP1, one expert group, and no serving load balancing")
        if self.experts.routed_experts.quant_method.unquantized_backend.name != "TRITON":
            raise ValueError("Aligned GLM requires unquantized Triton weight layouts")
        self._alignment_dispatcher = AlignedLocalDispatcher(self.n_routed_experts)
        self._alignment_top_k = config.num_experts_per_tok
        self._alignment_normalize = config.norm_topk_prob

    def forward(self, hidden_states):
        if hidden_states.shape[0] == 0:
            return hidden_states
        layer = self.experts.routed_experts
        scores, indices, _ = route(
            router_linear(hidden_states, self.gate.weight),
            self._alignment_top_k,
            self._alignment_normalize,
            score_func="sigmoid",
            selection_bias=self.gate.e_score_correction_bias,
            scale=self.routed_scaling_factor,
        )

        def experts(x, counts):
            offsets = counts.cumsum(0)
            gate_up = _GroupedLinear.apply(x, layer.w13_weight.transpose(-1, -2), offsets)
            gate, up = gate_up.chunk(2, dim=-1)
            return _GroupedLinear.apply(InferenceSilu.apply(gate, up), layer.w2_weight.transpose(-1, -2), offsets)

        output = self._alignment_dispatcher.run(hidden_states, scores, indices, experts, score_before_experts=False)
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)
        return output

    Glm4MoE.__init__ = init
    Glm4MoE.forward = forward
