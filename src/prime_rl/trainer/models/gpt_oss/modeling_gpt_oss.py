import torch
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutput

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from prime_rl.trainer.models.gpt_oss.converting_gpt_oss import (
    conversion_chain,
    is_hf_state_dict,
    is_prime_state_dict,
)
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig, apply_rotary_pos_emb
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


class GptOssRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class GptOssAttention(nn.Module):
    def __init__(self, config: GptOssConfig, layer_idx: int) -> None:
        super().__init__()
        if config._attn_implementation != "flash_attention_4":
            raise ValueError("The custom GPT-OSS implementation requires FlashAttention 4")
        if config.attention_dropout != 0:
            raise ValueError("The custom GPT-OSS implementation does not support attention dropout")

        from flash_attn.cute import flash_attn_varlen_func

        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))
        self.flash_attn = torch._dynamo.disable(flash_attn_varlen_func)

    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        window_size = (self.sliding_window - 1, 0) if self.sliding_window is not None else (None, None)
        output = self.flash_attn(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.scaling,
            causal=True,
            window_size=window_size,
            learnable_sink=self.sinks,
        )
        return output[0] if isinstance(output, tuple) else output

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        if batch_size != 1:
            raise ValueError(f"Custom GPT-OSS expects one packed row, got batch size {batch_size}")

        query = self.q_proj(hidden_states).view(batch_size, sequence_length, self.num_attention_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(
            query.transpose(1, 2),
            key.transpose(1, 2),
            cos,
            sin,
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)

        output = self.compute_attention(query[0], key[0], value[0], cu_seqlens, max_seqlen)
        return self.o_proj(output.contiguous().view(batch_size, sequence_length, -1))


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config: GptOssConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = GptOssAttention(config, layer_idx)

        router = TokenChoiceTopKRouter(
            dim=config.hidden_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            score_func="topk_softmax",
            route_norm=False,
            route_scale=1.0,
            gate_bias=True,
        )
        experts = GroupedExperts(
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            num_experts=config.num_local_experts,
            expert_type="gated",
            activation="clamped_swiglu",
            bias=True,
        )
        self.mlp = MoE(
            router=router,
            experts=experts,
            shared_expert=None,
            score_before_experts=False,
            load_balance_coeff=None,
        )
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        routed_experts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, cu_seqlens, max_seqlen)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, routed_experts=routed_experts)
        return residual + hidden_states


class GptOssPreTrainedModel(PreTrainedModelPrimeRL):
    config: GptOssConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GptOssDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _keep_in_fp32_modules = ["post_attention_layernorm", "input_layernorm", "norm"]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, GptOssAttention):
            nn.init.normal_(module.sinks, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, GroupedExperts):
            module.init_weights(self.config.initializer_range)
        elif isinstance(module, TokenChoiceTopKRouter):
            module.init_weights(self.config.initializer_range)
        elif isinstance(module, GptOssRMSNorm):
            nn.init.ones_(module.weight)
        else:
            super()._init_weights(module)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_hf_state_dict(state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_prime_state_dict(state_dict)

    @classmethod
    def conversion_chain(cls, config: GptOssConfig):
        return conversion_chain(config)


class GptOssModel(GptOssPreTrainedModel):
    def __init__(self, config: GptOssConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            GptOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = GptOssRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            RotaryEmbeddingConfig(
                max_position_embeddings=config.max_position_embeddings,
                rope_type=config.rope_parameters["rope_type"],
                model_config=config,
            )
        )
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if position_ids is None:
            raise ValueError("position_ids are required for custom GPT-OSS")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
            seq_lens.to(device=inputs_embeds.device),
            total_tokens=None if seq_lens_are_pre_shard else inputs_embeds.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_routed_experts = routed_experts[:, :, layer_idx] if routed_experts is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings,
                cu_seqlens,
                max_seqlen,
                routed_experts=layer_routed_experts,
            )
        return BaseModelOutput(last_hidden_state=self.norm(hidden_states))


class GptOssForCausalLM(GptOssPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__(config)
        self.model = GptOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = VanillaOutputLinear(config.hidden_size, config.vocab_size)
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        temperature: torch.Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> PrimeLmOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )
        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            slice_indices = logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature[:, slice_indices] if temperature is not None else None,
        )

    def init_buffers_post_meta(self) -> None:
        rotary_emb = self.model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
            rotary_emb.config,
            rotary_emb.inv_freq.device,
        )
        rotary_emb.inv_freq.copy_(inv_freq)
        for module in self.modules():
            if isinstance(module, MoE):
                module.tokens_per_expert.zero_()
                module.routing_confidence_sum.zero_()


def substitute_gpt_oss_ring_attention(
    process_group: torch.distributed.ProcessGroup,
    heads_k_stride: int,
) -> None:
    from ring_flash_attn.adapters.hf_adapter import DATA_PARAMS

    from prime_rl.trainer.distributed.collectives import all_gather

    def ring_attention(self, query, key, value, _cu_seqlens, _max_seqlen):
        key_value_groups = self.num_attention_heads // self.num_key_value_heads
        local_k_slice = DATA_PARAMS["local_k_slice"]
        window_size = (self.sliding_window - 1, 0) if self.sliding_window is not None else (None, None)
        outputs = []

        for key_head_start in range(0, self.num_key_value_heads, heads_k_stride):
            key_head_stop = min(key_head_start + heads_k_stride, self.num_key_value_heads)
            query_head_start = key_head_start * key_value_groups
            query_head_stop = key_head_stop * key_value_groups

            gathered_key = all_gather(key[:, key_head_start:key_head_stop], 0, process_group)[local_k_slice]
            gathered_value = all_gather(value[:, key_head_start:key_head_stop], 0, process_group)[local_k_slice]
            output = self.flash_attn(
                query[:, query_head_start:query_head_stop],
                gathered_key,
                gathered_value,
                cu_seqlens_q=DATA_PARAMS["cu_seqlens_q"],
                cu_seqlens_k=DATA_PARAMS["cu_seqlens_k"],
                max_seqlen_q=DATA_PARAMS["max_seqlen_q"],
                max_seqlen_k=DATA_PARAMS["max_seqlen_k"],
                softmax_scale=self.scaling,
                causal=True,
                window_size=window_size,
                learnable_sink=self.sinks[query_head_start:query_head_stop],
            )
            outputs.append(output[0] if isinstance(output, tuple) else output)

        return torch.cat(outputs, dim=1)

    GptOssAttention.compute_attention = ring_attention


def substitute_gpt_oss_ulysses_attention(
    process_group: torch.distributed.ProcessGroup,
    flash_attn,
) -> None:
    from prime_rl.trainer.models.layers.ulysses_attn import ULYSSES_PARAMS, ulysses_flash_attn_varlen_func

    cp_size = torch.distributed.get_world_size(process_group)

    def ulysses_attention(self, query, key, value, _cu_seqlens, _max_seqlen):
        window_size = (self.sliding_window - 1, 0) if self.sliding_window is not None else (-1, -1)
        return ulysses_flash_attn_varlen_func(
            flash_attn,
            query,
            key,
            value,
            cu_seqlens_q=ULYSSES_PARAMS["cu_seqlens"],
            cu_seqlens_k=ULYSSES_PARAMS["cu_seqlens"],
            max_seqlen_q=ULYSSES_PARAMS["max_seqlen"],
            max_seqlen_k=ULYSSES_PARAMS["max_seqlen"],
            causal=True,
            cp_group=process_group,
            cp_size=cp_size,
            flash_attn_version=4,
            window_size=window_size,
            softmax_scale=self.scaling,
            learnable_sink=self.sinks,
        )

    GptOssAttention.compute_attention = ulysses_attention


__all__ = [
    "GptOssAttention",
    "GptOssDecoderLayer",
    "GptOssForCausalLM",
    "GptOssModel",
    "GptOssPreTrainedModel",
    "GptOssRMSNorm",
    "substitute_gpt_oss_ring_attention",
    "substitute_gpt_oss_ulysses_attention",
]
