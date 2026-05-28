import math
import re

import torch
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioAttention,
    Gemma4AudioRelPositionalEncoding,
    Gemma4CausalLMOutputWithPast,
    Gemma4ClippableLinear,
    Gemma4ModelOutputWithPast,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
    Gemma4VisionRotaryEmbedding,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM as HFGemma4ForCausalLM,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForConditionalGeneration as HFGemma4ForConditionalGeneration,
)
from transformers.utils import TransformersKwargs, can_return_tuple
from typing_extensions import Unpack

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput

_CLIPPABLE_BUFFER_NAMES = frozenset({"input_min", "input_max", "output_min", "output_max"})


def _identity_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return state_dict


def _init_text_rotary_embedding(module: Gemma4TextRotaryEmbedding) -> None:
    for layer_type, rope_init_fn in module.rope_init_fns.items():
        rope_init_fn_kwargs = {"device": getattr(module, f"{layer_type}_inv_freq").device, "layer_type": layer_type}
        if layer_type == "full_attention" and module.rope_type[layer_type] == "proportional":
            rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

        inv_freq, attention_scaling = rope_init_fn(module.config, **rope_init_fn_kwargs)
        getattr(module, f"{layer_type}_inv_freq").copy_(inv_freq)
        getattr(module, f"{layer_type}_original_inv_freq").copy_(inv_freq)
        setattr(module, f"{layer_type}_attention_scaling", attention_scaling)


def _init_vision_rotary_embedding(module: Gemma4VisionRotaryEmbedding) -> None:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    rope_init_fn = (
        ROPE_INIT_FUNCTIONS[module.rope_type]
        if module.rope_type != "default"
        else module.compute_default_rope_parameters
    )
    inv_freq, attention_scaling = rope_init_fn(module.config, module.inv_freq.device)
    module.inv_freq.copy_(inv_freq)
    module.original_inv_freq.copy_(inv_freq)
    module.attention_scaling = attention_scaling


def _init_audio_relative_position(module: Gemma4AudioRelPositionalEncoding) -> None:
    min_timescale = 1.0
    max_timescale = 10000.0
    num_timescales = module.hidden_size // 2
    log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, device=module.inv_timescales.device) * -log_timescale_increment
    )
    module.inv_timescales.copy_(inv_timescales.unsqueeze(0).unsqueeze(0))


def _init_gemma4_nonpersistent_buffers(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, Gemma4TextRotaryEmbedding):
            _init_text_rotary_embedding(module)
        elif isinstance(module, Gemma4VisionRotaryEmbedding):
            _init_vision_rotary_embedding(module)
        elif isinstance(module, Gemma4TextScaledWordEmbedding):
            module.embed_scale.fill_(module.scalar_embed_scale)
        elif isinstance(module, Gemma4AudioRelPositionalEncoding):
            _init_audio_relative_position(module)
        elif isinstance(module, Gemma4AudioAttention):
            module.softcap.fill_(module.attention_logits_soft_cap)
        elif isinstance(module, Gemma4ClippableLinear) and module.use_clipped_linears:
            module.input_min.fill_(-float("inf"))
            module.input_max.fill_(float("inf"))
            module.output_min.fill_(-float("inf"))
            module.output_max.fill_(float("inf"))


def _mark_gemma4_runtime_buffers_nonpersistent(model: torch.nn.Module) -> None:
    """Match Gemma4 Hub checkpoints: clipping ranges are runtime constants."""
    for module in model.modules():
        if isinstance(module, Gemma4ClippableLinear) and module.use_clipped_linears:
            module._non_persistent_buffers_set.update(_CLIPPABLE_BUFFER_NAMES)


class _Gemma4PrimeRLMixin:
    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return False

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return _identity_state_dict(state_dict)

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return _identity_state_dict(state_dict)

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return _identity_state_dict(state_dict)

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return _identity_state_dict(state_dict)

    @classmethod
    def convert_adapter_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            out[re.sub(r"(\.layers\.\d+)\.experts\.", r"\1.moe.experts.", key)] = value
        return out

    def init_buffers_post_meta(self) -> None:
        _init_gemma4_nonpersistent_buffers(self)


class Gemma4ForCausalLM(_Gemma4PrimeRLMixin, HFGemma4ForCausalLM, PreTrainedModelPrimeRL):
    config: Gemma4TextConfig

    def __init__(self, config: Gemma4TextConfig, **kwargs):
        super().__init__(config)
        _mark_gemma4_runtime_buffers_nonpersistent(self)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        temperature: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        assert use_cache is None or use_cache is False, "use_cache is not supported for custom Gemma4"
        assert past_key_values is None, "past_key_values is not supported for custom Gemma4"

        outputs: BaseModelOutputWithPast = self.model(
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


class Gemma4ForConditionalGeneration(_Gemma4PrimeRLMixin, HFGemma4ForConditionalGeneration, PreTrainedModelPrimeRL):
    config: Gemma4Config

    def __init__(self, config: Gemma4Config, **kwargs):
        super().__init__(config)
        _mark_gemma4_runtime_buffers_nonpersistent(self)
        final_logit_softcapping = config.get_text_config().final_logit_softcapping
        if final_logit_softcapping is not None:
            config.final_logit_softcapping = final_logit_softcapping

    def _prime_rl_forward_backbone(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast | Gemma4ModelOutputWithPast:
        if pixel_values is None and pixel_values_videos is None and input_features is None:
            model_kwargs = dict(kwargs)
            model_kwargs.pop("return_dict", None)
            return self.model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                return_dict=True,
                **model_kwargs,
            )

        model_kwargs = dict(kwargs)
        model_kwargs.pop("return_dict", None)
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            image_position_ids=image_position_ids,
            video_position_ids=video_position_ids,
            return_dict=True,
            **model_kwargs,
        )

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        temperature: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput | Gemma4CausalLMOutputWithPast:
        assert use_cache is None or use_cache is False, "use_cache is not supported for custom Gemma4"
        assert past_key_values is None, "past_key_values is not supported for custom Gemma4"

        outputs: Gemma4ModelOutputWithPast = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            image_position_ids=image_position_ids,
            video_position_ids=video_position_ids,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )


__all__ = ["Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"]
