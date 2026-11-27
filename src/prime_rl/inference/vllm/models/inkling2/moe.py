# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 mixture-of-experts layer on a stock vLLM FusedMoE.

Routed experts only (`n_shared_experts == 0`). The router is a sigmoid-scored
top-8 over 384 experts with a selection-only bias (`mlp.gate.bias`, fp32),
renormalized weights times `route_scale = 8`, computed from fp32 logits. The
experts apply the soft-capped SiTU activation
``4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25)``.

A looped model routes once per layer visit. vLLM binds one routed-experts row
per MoE layer, so each visit of a layer has its own capture source and the
router writes to the row of the visit it runs in.
"""

import torch
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .configs import Inkling2ModelConfig


class Inkling2RoutingCapture(nn.Module):
    """Receives vLLM's capture callback for one layer visit (a
    `RoutedExpertsCaptureSource`): `layer_id` is the visit's routed-experts row."""

    def __init__(self, visit_id: int) -> None:
        super().__init__()
        self.layer_id = visit_id
        self.capture_fn = None


class Inkling2MoE(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
        visit_ids: tuple[int, ...] | None = None,
    ) -> None:
        """`visit_ids` are the routed-experts rows of the layer's loop passes;
        without them, vLLM captures to the row of the layer."""
        super().__init__()
        assert config.n_shared_experts == 0
        self.routing_captures = nn.ModuleList(Inkling2RoutingCapture(visit_id) for visit_id in visit_ids or ())
        vllm_config = get_current_vllm_config()
        # The router keeps bf16 weights and produces fp32 logits.
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=torch.float32,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.expert_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.experts = FusedMoEFactory(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            renormalize=True,
            quant_config=quant_config,
            activation=MoEActivation.SITU.value,
            activation_situ_beta=config.mlp_gate_softcap,
            activation_situ_linear_beta=config.mlp_up_softcap,
            scoring_func="sigmoid",
            routed_scaling_factor=config.route_scale,
            e_score_correction_bias=self.gate.expert_bias,
            router_logits_dtype=torch.float32,
            prefix=f"{prefix}.experts",
        )

    def forward(self, x: torch.Tensor, pass_idx: int = 0) -> torch.Tensor:
        num_tokens, hidden = x.shape
        if self.routing_captures and (capture_fn := self.routing_captures[pass_idx].capture_fn) is not None:
            # Capture to the row of this visit. Under CUDA graphs this runs at
            # capture time, like the router's own capture call.
            self.experts.router.set_capture_fn(capture_fn)
            if (sink := getattr(self.experts.routed_experts, "routing_sink", None)) is not None:
                sink.capture_fn = capture_fn
        router_logits, _ = self.gate(x.view(-1, hidden))
        out = self.experts(hidden_states=x, router_logits=router_logits)
        return out.view(num_tokens, hidden)

    def refresh_router_bias(self) -> None:
        """Re-point the fused router at the loaded bias Parameter.

        Layerwise reload may materialize a new Parameter object for the gate;
        the router holds the object it was built with.
        """
        bias = self.gate.expert_bias
        self.experts.routed_experts.e_score_correction_bias = bias
        router = getattr(self.experts, "router", None)
        if router is not None and hasattr(router, "e_score_correction_bias"):
            router.e_score_correction_bias = bias
