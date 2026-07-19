"""Nemotron-VL: frozen CRADIO vision tower grafted onto a frozen NemotronH backbone
through a trainable InternVL-style MLP projector (`mlp1`).

Composition (mirrors Nano Omni's graft, with the LM swapped for Super/Ultra):
  pixel tiles -> RadioVisionModel (frozen ViT-H/16) -> 2x2 pixel shuffle
  -> mlp1: RMSNorm -> Linear(5120, 20480) -> ReLU^2 -> Linear(20480, d_lm)
  -> scattered into the token embedding at `<image>` (img_context_token_id) positions
  -> NemotronHModel (frozen hybrid Mamba2/Attention/LatentMoE backbone).

Follows the qwen3_5 VLM structure (`model.visual` / `model.language_model`,
masked_scatter, dummy vision forward for text-only batches so FSDP collectives
stay consistent across DP ranks). NemotronH has no RoPE, so none of Qwen's
MRoPE position machinery is needed — positions are plain arange.

Context parallelism: NemotronH supports ulysses CP only (ring is unsupported for
hybrid-Mamba; Mamba layers cap cp at n_groups=8). For batches containing images
the trainer defers CP sharding to this model (`defers_vlm_cp_to_model`): the
scatter runs on the full sequence, then hidden states and positions are sharded
here, same as qwen3_5.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.moe import NemotronHRouter, NonGatedGroupedExperts
from prime_rl.trainer.models.nemotron_h.modeling_nemotron_h import NemotronHModel
from prime_rl.trainer.models.nemotron_vl.configuration_nemotron_vl import NemotronVLConfig
from prime_rl.trainer.models.nemotron_vl.converting_nemotron_vl import (
    convert_nemotron_vl_hf_to_prime,
    convert_nemotron_vl_prime_to_hf,
)
from prime_rl.trainer.models.nemotron_vl.modeling_radio import RadioVisionModel
from prime_rl.utils.cp import setup_cp_attention_params, shard_for_cp, shard_position_ids_for_cp


class SquaredReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.pow(F.relu(x), 2)


class ProjectorRMSNorm(nn.Module):
    """RMSNorm exactly as in Nano Omni's projector (fp32 compute, eps 1e-5)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


def _build_projector(config: NemotronVLConfig) -> nn.Sequential:
    # Sequential indices match the HF checkpoint keys mlp1.{0,1,3} (activation at 2 has no params).
    in_dim = config.vit_hidden_size * int(1 / config.downsample_ratio) ** 2
    return nn.Sequential(
        ProjectorRMSNorm(in_dim, eps=1e-5),
        nn.Linear(in_dim, config.projector_hidden_size, bias=False),
        SquaredReLU(),
        nn.Linear(config.projector_hidden_size, config.text_config.hidden_size, bias=False),
    )


class NemotronVLPreTrainedModel(PreTrainedModelPrimeRL):
    config: NemotronVLConfig
    config_class = NemotronVLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NemotronHMambaLayer", "NemotronHMoELayer", "NemotronHAttentionLayer", "RadioBlock"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _can_compile_fullgraph = False

    def _init_weights(self, module):
        std = self.config.text_config.initializer_range
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
        return any(name.startswith(("vision_model.", "language_model.vision_model.")) for name in state_dict) or any(
            name.startswith("backbone.") for name in state_dict
        )

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(name.startswith("model.visual.") for name in state_dict)

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return convert_nemotron_vl_hf_to_prime(state_dict)

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return convert_nemotron_vl_prime_to_hf(state_dict)


class NemotronVLModel(nn.Module):
    """Composite VLM body: frozen RADIO tower + trainable mlp1 + frozen NemotronH backbone."""

    def __init__(self, config: NemotronVLConfig):
        super().__init__()
        self.config = config
        # get_model sets attn_implementation on the composite config; recent transformers
        # propagate it to sub-configs, but NemotronH layers read it off text_config directly,
        # so forward it explicitly to be robust across versions.
        config.text_config._attn_implementation = config._attn_implementation
        self.visual = RadioVisionModel(config.vision_config)
        self.mlp1 = _build_projector(config)
        self.language_model = NemotronHModel(config.text_config)
        self._cp_group = None
        self._cp_rank = 0
        self._cp_world_size = 1

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def set_context_parallel_attributes(self, cp_group, cp_rank: int, cp_world_size: int) -> None:
        self._cp_group = cp_group
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size
        self.language_model.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)

    def pixel_shuffle(self, x: Tensor) -> Tensor:
        """InternVL pixel shuffle (ps_version v2), verbatim from Nano Omni."""
        scale_factor = self.config.downsample_ratio
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        return x.permute(0, 2, 1, 3).contiguous()

    def extract_feature(self, pixel_values: Tensor) -> Tensor:
        """(num_tiles, 3, H, W) normalized tiles -> (num_tiles, tokens_per_tile, d_lm)."""
        if pixel_values.ndim != 4:
            raise ValueError(f"pixel_values must be (num_tiles, 3, H, W); got shape {tuple(pixel_values.shape)}")
        pixel_values = pixel_values.to(self.visual.dtype)
        vit_embeds = self.visual(pixel_values)
        patch_size = self.config.vision_config.patch_size
        num_tiles, _, height, width = pixel_values.shape
        vit_embeds = vit_embeds.reshape(num_tiles, height // patch_size, width // patch_size, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds)
        vit_embeds = vit_embeds.reshape(num_tiles, -1, vit_embeds.shape[-1])
        return self.mlp1(vit_embeds)

    def _dummy_pixel_values(self, device: torch.device) -> Tensor:
        # Smallest tile the pipeline supports: 2x2 patches -> 1 image token after pixel shuffle.
        side = 2 * self.config.vision_config.patch_size
        return torch.zeros(1, 3, side, side, device=device, dtype=self.visual.dtype)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids are required when inputs_embeds are not provided")
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("input_ids are required to scatter image features")

            image_embeds = self.extract_feature(pixel_values)
            image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

            image_mask = input_ids == self.config.img_context_token_id
            image_token_count = int(image_mask.sum().item())
            if image_token_count != image_embeds.shape[0]:
                raise ValueError(
                    "Nemotron-VL image token/feature mismatch before scatter: "
                    f"img_context_token_id={self.config.img_context_token_id}, "
                    f"image_tokens={image_token_count}, image_features={image_embeds.shape[0]}, "
                    f"input_ids_shape={tuple(input_ids.shape)}, "
                    f"pixel_values_shape={tuple(pixel_values.shape)}"
                )
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # With images the trainer defers CP sharding to the model (the scatter above
            # needs the full sequence); shard embeds/positions here, same as qwen3_5.
            if self._cp_group is not None:
                if position_ids is None:
                    raise ValueError("position_ids are required for Nemotron-VL image batches under CP")
                setup_cp_attention_params(position_ids, cp_group=self._cp_group, cp_style="ulysses", seq_lens=seq_lens)
                inputs_embeds = shard_for_cp(inputs_embeds, cp_rank=self._cp_rank, cp_world_size=self._cp_world_size)
                position_ids = shard_position_ids_for_cp(
                    position_ids, cp_rank=self._cp_rank, cp_world_size=self._cp_world_size
                )
                seq_lens_are_pre_shard = True
        else:
            # Keep vision tower + projector in the autograd graph on text-only batches so
            # FSDP collectives line up across DP ranks (same trick as qwen3_5).
            image_embeds = self.extract_feature(self._dummy_pixel_values(inputs_embeds.device))
            inputs_embeds = inputs_embeds + image_embeds.sum() * 0.0

        return self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )


class NemotronVLForCausalLM(NemotronVLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    @property
    def defers_vlm_cp_to_model(self) -> bool:
        """Image batches under CP must reach the model unsharded; it shards after the scatter."""
        return True

    def __init__(self, config: NemotronVLConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = NemotronVLModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def set_context_parallel_attributes(self, cp_group, cp_rank: int, cp_world_size: int) -> None:
        self.model.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        temperature: Optional[torch.Tensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
        pixel_values: Optional[Tensor] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,  # renderer-supplied, unused (no MRoPE)
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
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
            pixel_values=pixel_values,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        pass


__all__ = ["NemotronVLForCausalLM", "NemotronVLModel", "NemotronVLPreTrainedModel"]
