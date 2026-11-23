# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 mixture-of-experts layer on a stock vLLM FusedMoE.

Routed experts only (`n_shared_experts == 0`). The router is a sigmoid-scored
top-8 over 384 experts with a selection-only bias (`mlp.gate.bias`, fp32),
renormalized weights times `route_scale = 8`, computed from fp32 logits. The
experts apply the soft-capped SiTU activation
``4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25)``.
"""

import torch
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .configs import Inkling2ModelConfig


class Inkling2MoE(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        assert config.n_shared_experts == 0
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden = x.shape
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
