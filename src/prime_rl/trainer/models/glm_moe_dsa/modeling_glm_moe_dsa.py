import warnings
from typing import Optional, Union

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.deprecation import deprecate_kwarg

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import (
    convert_hf_layer_to_tt,
    convert_hf_to_tt_moe,
    convert_tt_layer_to_hf,
    convert_tt_layer_to_vllm_kernel,
    convert_tt_to_hf_moe,
)
from prime_rl.trainer.models.glm_moe_dsa.sparse_mla_attention import GlmMoeDsaAttention, SparseMlaAttentionArgs
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import MLP, MLPConfig
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig


def _sparse_mla_attention_args(config: GlmMoeDsaConfig) -> SparseMlaAttentionArgs:
    if config.q_lora_rank is None:
        raise ValueError("Sparse MLA attention requires q_lora_rank to be set")
    return SparseMlaAttentionArgs(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        kv_lora_rank=config.kv_lora_rank,
        q_lora_rank=config.q_lora_rank,
        qk_rope_head_dim=config.qk_rope_head_dim,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_head_dim=config.qk_head_dim,
        v_head_dim=config.v_head_dim,
        attention_bias=config.attention_bias,
        rms_norm_eps=config.rms_norm_eps,
        index_n_heads=config.index_n_heads,
        index_head_dim=config.index_head_dim,
        index_topk=config.index_topk,
    )


class GlmMoeDsaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GlmMoeDsaAttention(_sparse_mla_attention_args(config))

        moe_args = MoEArgs(
            num_experts=config.n_routed_experts,
            num_shared_experts=config.n_shared_experts,
            score_func="sigmoid",
            route_norm=config.norm_topk_prob,
            route_scale=config.routed_scaling_factor,
            score_before_experts=False,
            top_k=config.num_experts_per_tok,
            load_balance_coeff=1e-3,
            use_grouped_mm=config.use_grouped_mm,
        )
        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            gate_act=config.hidden_act,
            bias=False,
        )

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = MoE(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)
        else:
            self.mlp = MLP(mlp_config)

        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        ks: Optional[torch.Tensor] = None,
        ke: Optional[torch.Tensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            ks=ks,
            ke=ke,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, routed_experts=routed_experts)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class GlmMoeDsaPreTrainedModel(PreTrainedModelPrimeRL):
    config: GlmMoeDsaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GlmMoeDsaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": GlmMoeDsaDecoderLayer,
    }

    def _init_weights(self, module):
        super()._init_weights(module)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.1.up_proj" in name or "mlp.experts.gate_up_proj" in name for name in state_dict.keys())

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.w1" in module_name for module_name in state_dict.keys())

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_tt_to_hf_moe(state_dict)
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_hf_to_tt_moe(state_dict)
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_tt_layer_to_hf(state_dict, layer_idx)
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_hf_layer_to_tt(state_dict, layer_idx)
        return state_dict

    @classmethod
    def convert_layer_to_vllm_kernel(
        cls, state_dict: dict[str, Tensor], layer_idx: int, quantize_fp8: bool = False
    ) -> dict[str, Tensor]:
        return convert_tt_layer_to_vllm_kernel(state_dict, layer_idx, quantize_fp8=quantize_fp8)


@auto_docstring
class GlmMoeDsaModel(GlmMoeDsaPreTrainedModel):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GlmMoeDsaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_type = rope_parameters.get("rope_type", "default") if isinstance(rope_parameters, dict) else "default"
        rotary_config = RotaryEmbeddingConfig(
            max_position_embeddings=config.max_position_embeddings,
            rope_type=rope_type,
            model_config=config,
        )
        self.rotary_emb = RotaryEmbedding(rotary_config)
        self.gradient_checkpointing = False

        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutputWithPast:
        """
        routed_experts (`torch.LongTensor` of shape `(batch_size, sequence_length, num_hidden_layers, num_experts_per_tok)`, *optional*):
            Routed experts for each token in the sequence. Only used for router replay.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        flat_position_ids = position_ids.view(-1)
        S = flat_position_ids.shape[0]
        ks = torch.arange(S, dtype=torch.int32, device=flat_position_ids.device) - flat_position_ids.to(torch.int32)
        ke = torch.arange(1, S + 1, dtype=torch.int32, device=flat_position_ids.device)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            routed_experts_layer = routed_experts[:, :, layer_idx, :] if routed_experts is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                ks=ks,
                ke=ke,
                routed_experts=routed_experts_layer,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


@auto_docstring
class GlmMoeDsaForCausalLM(GlmMoeDsaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = GlmMoeDsaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        warnings.warn("GlmMoeDsaForCausalLM is experimental, higher trainer<->inference KL mismatch may be observed.")
        warnings.warn("`model.attn` is ignored, GlmMoeDsa uses only sparse attention.")

        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Optional[torch.Tensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels used by PrimeRL's wrapped LM head to optionally compute per-token logprobs/entropy.
        temperature (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token temperatures for logprobs/entropy computation when `labels` are provided.
        routed_experts (`torch.LongTensor` of shape `(batch_size, sequence_length, num_hidden_layers, num_experts_per_tok)`, *optional*):
            Routed experts for each token in the sequence. Only used for router replay.
        """
        assert use_cache is None, "use_cache is not supported for custom glm_moe_dsa for now"
        assert past_key_values is None, "past_key_values is not supported for custom glm_moe_dsa for now"

        if position_ids is None:
            if inputs_embeds is not None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            routed_experts=routed_experts,
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
            inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
                rotary_emb.config, rotary_emb.inv_freq.device
            )
            rotary_emb.inv_freq.copy_(inv_freq)


__all__ = ["GlmMoeDsaConfig", "GlmMoeDsaPreTrainedModel", "GlmMoeDsaModel", "GlmMoeDsaForCausalLM"]
