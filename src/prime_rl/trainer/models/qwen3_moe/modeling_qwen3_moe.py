# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import Tensor, nn

from prime_rl.trainer.models.base import PrimeModel
from prime_rl.trainer.models.layers.attn import ATTN_IMPL2CLASS, AttentionConfig
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding
from prime_rl.trainer.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from prime_rl.trainer.models.qwen3_moe.converting_qwen3_moe import conversion_chain
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            is_causal=True,
            attention_bias=config.attention_bias,
            use_qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
        )
        # TODO: Sliding window support
        self.self_attn = ATTN_IMPL2CLASS[config.attn_implementation](attn_config)

        moe_args = MoEArgs(
            num_experts=config.num_experts,
            expert_type="gated",
            activation=config.hidden_act,
            score_func="softmax",
            route_norm=config.norm_topk_prob,
            route_scale=1.0,
            score_before_experts=False,
            top_k=config.num_experts_per_tok,
            load_balance_coeff=config.load_balance_coeff,
        )
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = MoE.from_args(
                moe_args,
                dim=config.hidden_size,
                hidden_dim=config.moe_intermediate_size,
                shared_expert=None,
            )
        else:
            self.mlp = FeedForward(
                dim=config.hidden_size,
                hidden_dim=config.intermediate_size,
                activation=config.hidden_act,
            )

        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        cu_seqlens: Tensor,
        max_seqlen: int,
        routed_experts: Tensor | None = None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, routed_experts=routed_experts)
        return residual + hidden_states


class Qwen3MoeModel(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.rotary_emb = RotaryEmbedding(config.rope_parameters, config.head_dim, config.max_position_embeddings)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
        routed_experts: Tensor | None = None,
    ) -> Tensor:
        """``routed_experts`` (``[batch, seq, num_layers, top_k]``) replays the inference router's choices."""
        hidden_states = self.embed_tokens(input_ids)
        cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
            seq_lens.to(device=hidden_states.device),
            total_tokens=None if seq_lens_are_pre_shard else hidden_states.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_routed_experts = routed_experts[:, :, layer_idx, :] if routed_experts is not None else None
            hidden_states = decoder_layer(
                hidden_states, position_embeddings, cu_seqlens, max_seqlen, routed_experts=layer_routed_experts
            )
        return self.norm(hidden_states)


class Qwen3MoeForCausalLM(PrimeModel):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.model = Qwen3MoeModel(config)
        self.lm_head = VanillaOutputLinear(config.hidden_size, config.vocab_size)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        # Per-expert keys, or the fused expert keys transformers 5 writes.
        return any("mlp.experts.1.up_proj" in name or "mlp.experts.gate_up_proj" in name for name in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.gate_proj" in name for name in state_dict)

    @classmethod
    def conversion_chain(cls, config: Qwen3MoeConfig):
        return conversion_chain(config)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        *,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
        labels: Tensor | None = None,
        temperature: Tensor | None = None,
        sampling_mask: Tensor | None = None,
        routed_experts: Tensor | None = None,
    ) -> PrimeLmOutput:
        hidden_states = self.model(input_ids, position_ids, seq_lens, seq_lens_are_pre_shard, routed_experts)
        return self.lm_head(hidden_states, labels, temperature=temperature, sampling_mask=sampling_mask)
