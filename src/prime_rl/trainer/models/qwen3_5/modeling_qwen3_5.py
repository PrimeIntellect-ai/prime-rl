import functools
from typing import Optional, Union

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5PreTrainedModel as HFQwen3_5PreTrainedModel,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import MLP, MLPConfig
from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedAttentionConfig,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeGatedFlashAttention,
    Qwen3_5MoeGatedSDPAAttention,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeRotaryEmbedding,
    normalize_qwen3_5_attn_implementation,
)
from prime_rl.trainer.models.qwen3_5_moe.mrope import build_qwen3_5_mrope_position_ids
from prime_rl.utils.cp import setup_cp_attention_params, shard_for_cp, shard_position_ids_for_cp
from prime_rl.utils.sequence import get_cu_seqlens_from_position_ids, get_cu_seqlens_from_seq_lens


def _build_text_config(composite_config: PretrainedConfig) -> Qwen3_5TextConfig:
    """Build dense Qwen3.5 text config from HF's composite VLM config."""
    text_dict = composite_config.text_config.to_dict()
    text_config = Qwen3_5TextConfig(**text_dict)
    attn_impl = getattr(
        composite_config.text_config,
        "_attn_implementation",
        getattr(composite_config, "_attn_implementation", None),
    )
    if attn_impl is not None:
        text_config._attn_implementation = attn_impl
    return text_config


class Qwen3_5GatedSDPAAttention(Qwen3_5MoeGatedSDPAAttention):
    pass


class Qwen3_5GatedFlashAttention(Qwen3_5MoeGatedFlashAttention):
    pass


QWEN35_ATTN_IMPL2CLASS = {
    "sdpa": Qwen3_5GatedSDPAAttention,
    "flash_attention_2": functools.partial(Qwen3_5GatedFlashAttention, flash_attn_version=2),
    "flash_attention_3": functools.partial(Qwen3_5GatedFlashAttention, flash_attn_version=3),
    "fa4": functools.partial(Qwen3_5GatedFlashAttention, flash_attn_version=4),
}


def _get_gated_attention(config: Qwen3_5TextConfig) -> nn.Module:
    attn_config = Qwen3_5MoeGatedAttentionConfig(
        hidden_size=config.hidden_size,
        head_dim=config.head_dim,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        rms_norm_eps=config.rms_norm_eps,
        attention_bias=config.attention_bias,
        attention_dropout=config.attention_dropout,
    )

    attn_impl = normalize_qwen3_5_attn_implementation(config._attn_implementation)
    config._attn_implementation = attn_impl

    if attn_impl not in QWEN35_ATTN_IMPL2CLASS:
        supported = list(QWEN35_ATTN_IMPL2CLASS.keys())
        raise ValueError(
            f"Qwen3.5 attention does not support '{config._attn_implementation}'. "
            f"Supported implementations: {supported}."
        )

    return QWEN35_ATTN_IMPL2CLASS[attn_impl](attn_config)


def _create_rotary_emb(config: Qwen3_5TextConfig) -> Qwen3_5MoeRotaryEmbedding:
    return Qwen3_5MoeRotaryEmbedding(config)


class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config)
        elif self.layer_type == "full_attention":
            self.self_attn = _get_gated_attention(config)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type}")

        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            gate_act=config.hidden_act,
            bias=False,
        )
        self.mlp = MLP(mlp_config)
        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
        cu_seqlens_are_global: bool = False,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states, cu_seqlens=cu_seqlens, cu_seqlens_are_global=cu_seqlens_are_global
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3_5PreTrainedModel(PreTrainedModelPrimeRL, HFQwen3_5PreTrainedModel):
    config_class = Qwen3_5TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _supports_attention_backend = True
    _can_compile_fullgraph = False
    _can_record_outputs = {
        "hidden_states": Qwen3_5DecoderLayer,
    }

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False, allow_all_kernels: bool = False
    ) -> str:
        attn_impl = normalize_qwen3_5_attn_implementation(attn_implementation or "sdpa")
        if attn_impl not in QWEN35_ATTN_IMPL2CLASS:
            supported = list(QWEN35_ATTN_IMPL2CLASS.keys())
            raise ValueError(
                f"Qwen3.5 attention does not support '{attn_implementation}'. Supported implementations: {supported}."
            )
        return attn_impl

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict


class Qwen3_5Model(Qwen3_5PreTrainedModel):
    def __init__(self, config: Qwen3_5TextConfig):
        config._attn_implementation = normalize_qwen3_5_attn_implementation(config._attn_implementation)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _create_rotary_emb(config)
        self.gradient_checkpointing = False

        self.post_init()

    def set_context_parallel_attributes(self, cp_group, cp_rank: int, cp_world_size: int) -> None:
        self._cp_group = cp_group
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size
        for layer in self.layers:
            if getattr(layer, "layer_type", None) == "linear_attention":
                layer.linear_attn.cp_group = cp_group
                layer.linear_attn.cp_rank = cp_rank
                layer.linear_attn.cp_world_size = cp_world_size

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        seq_lens: Optional[torch.LongTensor] = None,
        seq_lens_are_global: bool = False,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        flash_attn_enabled = self.config._attn_implementation in ("flash_attention_2", "flash_attention_3", "fa4")
        if flash_attn_enabled:
            if seq_lens_are_global and seq_lens is not None:
                cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(seq_lens.to(device=inputs_embeds.device))
            elif position_ids.ndim == 3:
                if inputs_embeds.shape[0] != 1:
                    raise ValueError("3D Qwen3.5 MRoPE positions require batch size 1 for varlen attention")
                seq_len = inputs_embeds.shape[1]
                if seq_lens is None:
                    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=inputs_embeds.device)
                    max_seqlen = seq_len
                else:
                    cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
                        seq_lens.to(device=inputs_embeds.device),
                        total_tokens=seq_len,
                    )
            else:
                cu_seqlens, max_seqlen = get_cu_seqlens_from_position_ids(position_ids)
            torch._dynamo.mark_dynamic(cu_seqlens, 0)
        elif seq_lens is not None:
            # seq_lens is only populated for batch size 1 (see cat_collate), so this
            # covers both 2D text-only packs and 3D MRoPE packs identically.
            seq_lens = seq_lens.to(device=inputs_embeds.device)
            if seq_lens.numel() > 1 and "full_attention" in self.config.layer_types:
                raise ValueError("Packed Qwen3.5 batches with full_attention layers require flash attention")
            cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
                seq_lens,
                total_tokens=None if seq_lens_are_global else inputs_embeds.shape[1],
            )
        else:
            cu_seqlens = None
            max_seqlen = None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        cu_seqlens_are_global = seq_lens_are_global and seq_lens is not None

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_are_global=cu_seqlens_are_global,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Qwen3_5VLMModel(nn.Module):
    """Composite VLM body: HF vision encoder + custom PrimeRL dense text model."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.visual = Qwen3_5VisionModel._from_config(config.vision_config)
        self.language_model = Qwen3_5Model(_build_text_config(config))

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def set_context_parallel_attributes(self, cp_group, cp_rank: int, cp_world_size: int) -> None:
        self.language_model.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)

    def _dummy_vision_inputs(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        vcfg = self.config.vision_config
        m = vcfg.spatial_merge_size
        num_patches = m * m
        patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size
        pixel_values = torch.zeros(num_patches, patch_dim, device=device, dtype=self.visual.dtype)
        grid_thw = torch.tensor([[1, m, m]], dtype=torch.long, device=device)
        return pixel_values, grid_thw

    def prepare_inputs_embeds_and_position_ids(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        seq_lens: torch.LongTensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids are required when inputs_embeds are not provided")
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        if image_grid_thw is not None and input_ids is None:
            raise ValueError("input_ids are required to compute Qwen3.5 multimodal MRoPE positions")
        if image_grid_thw is not None and mm_token_type_ids is None:
            raise ValueError(
                "Qwen3.5 multimodal forward requires mm_token_type_ids to compute MRoPE positions correctly"
            )
        if image_grid_thw is not None and position_ids is not None and position_ids.ndim != 3:
            raise ValueError(
                f"Qwen3.5 multimodal forward requires 3D MRoPE position_ids; got shape={tuple(position_ids.shape)}"
            )

        has_images = pixel_values is not None
        vision_grid_thw = image_grid_thw
        if has_images:
            if input_ids is None:
                raise ValueError("input_ids are required when scattering Qwen3.5 image features")
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required when pixel_values are provided")
            pixel_values = pixel_values.type(self.visual.dtype)
        else:
            pixel_values, vision_grid_thw = self._dummy_vision_inputs(inputs_embeds.device)

        vision_output = self.visual(pixel_values, grid_thw=vision_grid_thw, return_dict=True)
        image_embeds = vision_output.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)

        if has_images:
            image_mask = input_ids == self.config.image_token_id
            image_token_count = int(image_mask.sum().item())
            image_feature_count = int(image_embeds.shape[0])
            if image_token_count != image_feature_count:
                raise ValueError(
                    "Qwen VLM image token/feature mismatch before scatter: "
                    f"image_token_id={self.config.image_token_id}, "
                    f"image_tokens={image_token_count}, image_features={image_feature_count}, "
                    f"input_ids_shape={tuple(input_ids.shape)}, "
                    f"pixel_values_shape={tuple(pixel_values.shape)}, "
                    f"image_grid_thw_shape={tuple(image_grid_thw.shape) if image_grid_thw is not None else None}"
                )
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        else:
            inputs_embeds = inputs_embeds + image_embeds.sum() * 0.0

        if position_ids is None:
            if image_grid_thw is not None:
                position_ids = build_qwen3_5_mrope_position_ids(
                    input_ids=input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    spatial_merge_size=self.config.vision_config.spatial_merge_size,
                    seq_lens=seq_lens,
                )
            else:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        return inputs_embeds, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        seq_lens: torch.LongTensor | None = None,
        seq_lens_are_global: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        inputs_embeds, position_ids = self.prepare_inputs_embeds_and_position_ids(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            seq_lens=seq_lens,
        )

        cp_group = getattr(self.language_model, "_cp_group", None)
        if image_grid_thw is not None and cp_group is not None:
            cp_rank = self.language_model._cp_rank
            cp_world_size = self.language_model._cp_world_size
            setup_cp_attention_params(position_ids, cp_group=cp_group, cp_style="ulysses", seq_lens=seq_lens)
            inputs_embeds = shard_for_cp(inputs_embeds, cp_rank=cp_rank, cp_world_size=cp_world_size)
            position_ids = shard_position_ids_for_cp(position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)
            seq_lens_are_global = True

        return self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            seq_lens=seq_lens,
            seq_lens_are_global=seq_lens_are_global,
        )


def _has_vlm_keys(state_dict: dict[str, Tensor]) -> bool:
    return any(k.startswith("model.language_model.") for k in state_dict)


def _remap_lm_keys(state_dict: dict[str, Tensor], to_flat: bool = True) -> None:
    src = "model.language_model." if to_flat else "model."
    dst = "model." if to_flat else "model.language_model."
    for k in [k for k in list(state_dict.keys()) if k.startswith(src) and not k.startswith("model.visual.")]:
        state_dict[dst + k[len(src) :]] = state_dict.pop(k)


class Qwen3_5ForCausalLM(Qwen3_5PreTrainedModel, GenerationMixin):
    """Unified dense Qwen3.5 model for both text-only and VLM configs."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _checkpoint_conversion_mapping = {}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._is_vlm = hasattr(config, "vision_config")

        if self._is_vlm:
            self.model = Qwen3_5VLMModel(config)
            text_config = config.text_config
            self._tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
        else:
            self.model = Qwen3_5Model(config)
            text_config = config

        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        if self._is_vlm:
            return self.model.get_input_embeddings()
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        if self._is_vlm:
            self.model.set_input_embeddings(value)
        else:
            self.model.embed_tokens = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def set_context_parallel_attributes(self, cp_group, cp_rank: int, cp_world_size: int) -> None:
        self.model.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        if _has_vlm_keys(state_dict):
            _remap_lm_keys(state_dict, to_flat=True)
            _remap_lm_keys(state_dict, to_flat=False)
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        if _has_vlm_keys(state_dict):
            _remap_lm_keys(state_dict, to_flat=True)
            _remap_lm_keys(state_dict, to_flat=False)
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return cls.convert_to_hf(state_dict)

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return cls.convert_to_prime(state_dict)

    def prime_forward_kwargs(
        self,
        *,
        seq_lens: Tensor | None = None,
        seq_lens_are_global: bool = False,
    ) -> dict[str, object]:
        if seq_lens is None:
            return {}
        return {"seq_lens": seq_lens, "seq_lens_are_global": seq_lens_are_global}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Union[torch.Tensor, None] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
        seq_lens: Optional[torch.LongTensor] = None,
        seq_lens_are_global: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        assert use_cache is None, "use_cache is not supported for custom qwen3_5 for now"
        assert past_key_values is None, "past_key_values is not supported for custom qwen3_5 for now"

        has_vlm_image_inputs = self._is_vlm and image_grid_thw is not None
        if position_ids is None and not has_vlm_image_inputs:
            if inputs_embeds is not None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            elif input_ids is not None:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        if self._is_vlm:
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                seq_lens=seq_lens,
                seq_lens_are_global=seq_lens_are_global,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                seq_lens=seq_lens,
                seq_lens_are_global=seq_lens_are_global,
            )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        if self._is_vlm:
            lm_rope = self.model.language_model.rotary_emb
        else:
            lm_rope = self.model.rotary_emb

        if hasattr(lm_rope, "rope_init_fn"):
            inv_freq, lm_rope.attention_scaling = lm_rope.rope_init_fn(lm_rope.config, lm_rope.inv_freq.device)
            lm_rope.inv_freq.copy_(inv_freq)

        if self._is_vlm:
            vis_rope = self.model.visual.rotary_pos_emb
            if hasattr(vis_rope, "inv_freq"):
                dim = vis_rope.inv_freq.shape[0]
                inv_freq = 1.0 / (
                    10000.0
                    ** (torch.arange(0, dim * 2, 2, dtype=torch.float32, device=vis_rope.inv_freq.device) / (dim * 2))
                )
                vis_rope.inv_freq.copy_(inv_freq)


__all__ = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5GatedFlashAttention",
    "Qwen3_5Model",
    "Qwen3_5PreTrainedModel",
]
