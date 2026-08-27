# GPT-OSS model with PrimeRL grouped experts. Attention, rotary embedding, and
# RMSNorm are reused from HuggingFace.

from typing import Union

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from prime_rl.trainer.models.gpt_oss.converting_gpt_oss import (
    conversion_chain,
    is_hf_state_dict,
    is_prime_state_dict,
)
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.moe import (
    GroupedExperts,
    MoE,
    TokenChoiceTopKRouter,
)


class GptOssDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        assert not getattr(config, "fp8", False), "FP8 training is not supported for GPT-OSS"
        self.hidden_size = config.hidden_size
        self.self_attn = GptOssAttention(config=config, layer_idx=layer_idx)
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
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class GptOssPreTrainedModel(PreTrainedModelPrimeRL):
    config: GptOssConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GptOssDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _keep_in_fp32_modules = ["post_attention_layernorm", "input_layernorm", "norm"]
    _compatible_flash_implementations = ["kernels-community/vllm-flash-attn3", "flash_attention_4"]
    _can_record_outputs = {"hidden_states": GptOssDecoderLayer}

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_hf_state_dict(state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_prime_state_dict(state_dict)

    @classmethod
    def conversion_chain(cls, config):
        return conversion_chain(config)


@auto_docstring
class GptOssModel(GptOssPreTrainedModel):
    _no_split_modules = ["GptOssDecoderLayer"]

    def __init__(self, config: GptOssConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GptOssRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


@auto_docstring
class GptOssForCausalLM(GptOssPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: GptOssConfig):
        super().__init__(config)
        self.model = GptOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: torch.Tensor | None = None,
        # Document boundaries derive from position_ids (HF packed-sequence masks);
        # seq_lens is accepted to satisfy the trainer's universal contract.
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        r"""
        seq_lens (`torch.LongTensor` of shape `(num_documents,)`):
            Per-document lengths of the packed row (PrimeRL packed-batch contract).
        seq_lens_are_pre_shard (`bool`, *optional*, defaults to `False`):
            Whether `seq_lens` holds pre-CP-shard (global) document boundaries.
        temperature (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token temperatures for logprobs/entropy computation when `labels` are provided.
        """
        assert use_cache is None or not use_cache, "use_cache is not supported for custom GPT-OSS"
        assert past_key_values is None, "past_key_values is not supported for custom GPT-OSS"

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        buffer_names = [name for name, _ in self.named_buffers()]
        if "model.rotary_emb.inv_freq" in buffer_names:
            rotary_emb = self.model.rotary_emb
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            rope_init_fn = (
                ROPE_INIT_FUNCTIONS[rotary_emb.rope_type]
                if rotary_emb.rope_type != "default"
                else rotary_emb.compute_default_rope_parameters
            )
            inv_freq, rotary_emb.attention_scaling = rope_init_fn(rotary_emb.config, rotary_emb.inv_freq.device)
            rotary_emb.inv_freq.copy_(inv_freq)
            if "model.rotary_emb.original_inv_freq" in buffer_names:
                rotary_emb.original_inv_freq.copy_(inv_freq)


__all__ = [
    "GptOssForCausalLM",
    "GptOssModel",
    "GptOssPreTrainedModel",
]
