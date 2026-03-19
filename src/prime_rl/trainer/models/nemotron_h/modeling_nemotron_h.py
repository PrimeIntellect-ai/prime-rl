"""PrimeRL implementation of NemotronH (Nemotron-3-Super-120B-A12B).

Hybrid Mamba-Transformer-MoE architecture with three distinct layer types:
- Mamba-2 layers (using NemotronHMamba2Mixer from HF transformers)
- LatentMoE layers (non-gated experts with latent projections)
- Attention layers (using shared FlashAttention/SDPA from prime-rl)
"""

from typing import Optional

import torch
from torch import Tensor, nn
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.nemotron_h.modular_nemotron_h import NemotronHMamba2Mixer
from transformers.utils import auto_docstring, logging

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.attn import ATTN_IMPL2CLASS, AttentionConfig
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.moe import LatentMoE, NemotronHRouter, NonGatedGroupedExperts
from prime_rl.trainer.models.layers.rms_norm import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig
from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from prime_rl.trainer.models.nemotron_h.converting_nemotron_h import (
    convert_hf_layer_to_prime,
    convert_hf_to_prime,
    convert_prime_layer_to_hf,
    convert_prime_to_hf,
)

logger = logging.get_logger(__name__)


class NemotronHMambaLayer(GradientCheckpointingLayer):
    """Mamba-2 SSM layer: norm -> NemotronHMamba2Mixer -> residual."""

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.layer_norm_epsilon))
        self.mamba = NemotronHMamba2Mixer(config, layer_idx=layer_idx)
        self.mlp = None  # No MoE in this layer type

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        return residual + hidden_states


class NemotronHMoELayer(GradientCheckpointingLayer):
    """MoE layer: norm -> LatentMoE -> residual."""

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.layer_norm_epsilon))
        self.mlp = LatentMoE(
            dim=config.hidden_size,
            latent_dim=config.moe_latent_size,
            moe_intermediate_size=config.moe_intermediate_size,
            shared_expert_intermediate_size=config.moe_shared_expert_intermediate_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            n_group=config.n_group,
            topk_group=config.topk_group,
            norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor,
            use_grouped_mm=config.use_grouped_mm,
            load_balance_coeff=config.load_balance_coeff,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class NemotronHAttentionLayer(GradientCheckpointingLayer):
    """Attention layer: norm -> FlashAttention/SDPA -> residual."""

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.layer_norm_epsilon))
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            is_causal=True,
            attention_bias=config.attention_bias,
            use_qk_norm=False,
            rms_norm_eps=config.layer_norm_epsilon,
        )
        self.self_attn = ATTN_IMPL2CLASS[config._attn_implementation](attn_config)
        self.mlp = None  # No MoE in this layer type

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return residual + hidden_states


BLOCK_TYPE_MAP = {
    "mamba": NemotronHMambaLayer,
    "moe": NemotronHMoELayer,
    "attention": NemotronHAttentionLayer,
}


def _build_layer(config: NemotronHConfig, layer_idx: int):
    layer_type = config.layers_block_type[layer_idx]
    cls = BLOCK_TYPE_MAP[layer_type]
    if layer_type == "mamba":
        return cls(config, layer_idx=layer_idx)
    return cls(config)


@auto_docstring
class NemotronHPreTrainedModel(PreTrainedModelPrimeRL):
    config: NemotronHConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NemotronHMambaLayer", "NemotronHMoELayer", "NemotronHAttentionLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (NonGatedGroupedExperts, NemotronHRouter)):
            module.init_weights(std)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mixer." in name for name in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mamba." in name or "mlp.experts.w1" in name for name in state_dict)

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        # Need config to know layer types; infer from state dict keys
        layers_block_type = _infer_layers_block_type(state_dict)
        convert_prime_to_hf(state_dict, layers_block_type)
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        layers_block_type = _infer_layers_block_type_from_hf(state_dict)
        convert_hf_to_prime(state_dict, layers_block_type)
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        layer_type = _infer_layer_type_prime(state_dict, layer_idx)
        convert_prime_layer_to_hf(state_dict, layer_idx, layer_type)
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        layer_type = _infer_layer_type_hf(state_dict, layer_idx)
        convert_hf_layer_to_prime(state_dict, layer_idx, layer_type)
        return state_dict


def _infer_layer_type_hf(state_dict: dict[str, Tensor], layer_idx: int) -> str:
    """Infer layer type from HF state dict keys for a given layer."""
    prefix = f"model.layers.{layer_idx}.mixer."
    layer_keys = [k for k in state_dict if k.startswith(prefix)]
    if not layer_keys:
        return "mamba"  # fallback
    for k in layer_keys:
        suffix = k[len(prefix) :]
        if suffix.startswith("gate.") or suffix.startswith("experts."):
            return "moe"
        if suffix.startswith("q_proj") or suffix.startswith("k_proj"):
            return "attention"
    return "mamba"


def _infer_layer_type_prime(state_dict: dict[str, Tensor], layer_idx: int) -> str:
    """Infer layer type from PrimeRL state dict keys for a given layer."""
    prefix = f"model.layers.{layer_idx}."
    for k in state_dict:
        if not k.startswith(prefix):
            continue
        suffix = k[len(prefix) :]
        if suffix.startswith("mlp."):
            return "moe"
        if suffix.startswith("self_attn."):
            return "attention"
        if suffix.startswith("mamba."):
            return "mamba"
    return "mamba"


def _infer_layers_block_type_from_hf(state_dict: dict[str, Tensor]) -> list[str]:
    """Infer full layers_block_type list from HF state dict."""
    max_layer = max(int(k.split(".")[2]) for k in state_dict if k.startswith("model.layers.")) + 1
    return [_infer_layer_type_hf(state_dict, i) for i in range(max_layer)]


def _infer_layers_block_type(state_dict: dict[str, Tensor]) -> list[str]:
    """Infer full layers_block_type list from PrimeRL state dict."""
    max_layer = max(int(k.split(".")[2]) for k in state_dict if k.startswith("model.layers.")) + 1
    return [_infer_layer_type_prime(state_dict, i) for i in range(max_layer)]


class NemotronHModel(NemotronHPreTrainedModel):
    def __init__(self, config: NemotronHConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([_build_layer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.layer_norm_epsilon))

        # RoPE is only needed for attention layers
        has_attention = any(t == "attention" for t in config.layers_block_type)
        if has_attention:
            rope_type = "default"
            if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
                rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
            rotary_config = RotaryEmbeddingConfig(
                max_position_embeddings=config.max_position_embeddings,
                rope_type=rope_type,
                model_config=config,
            )
            self.rotary_emb = RotaryEmbedding(rotary_config)
        else:
            self.rotary_emb = None

        self.gradient_checkpointing = False
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Compute cu_seqlens and max_seqlen for flash attention
        if self.config._attn_implementation in ("flash_attention_2", "flash_attention_3", "fa4"):
            flat_position_ids = position_ids.view(-1)
            seqlens = torch.cat(
                [
                    flat_position_ids[0:1],
                    flat_position_ids[:-1][(flat_position_ids == 0)[1:]] + 1,
                    flat_position_ids[-1:] + 1,
                ]
            )
            max_seqlen = seqlens.max().item()
            cu_seqlens = seqlens.cumsum(dim=0, dtype=torch.int32)
            torch._dynamo.mark_dynamic(cu_seqlens, 0)
        else:
            max_seqlen = None
            cu_seqlens = None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids) if self.rotary_emb is not None else None

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


@auto_docstring
class NemotronHForCausalLM(NemotronHPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: NemotronHConfig):
        super().__init__(config)
        self.model = NemotronHModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        temperature: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> PrimeLmOutput:
        if position_ids is None:
            if inputs_embeds is not None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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


__all__ = [
    "NemotronHForCausalLM",
    "NemotronHModel",
    "NemotronHPreTrainedModel",
]
