import torch
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutput

from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.nemotron_h.backbone import NemotronHModel
from prime_rl.trainer.models.nemotron_h.base import NemotronHPreTrainedModel
from prime_rl.trainer.models.nemotron_h_omni.configuration_nemotron_h_omni import NemotronHOmniConfig
from prime_rl.trainer.models.nemotron_h_omni.converting_nemotron_h_omni import (
    conversion_chain,
    is_hf_state_dict,
    is_prime_state_dict,
)
from prime_rl.trainer.models.nemotron_h_omni.vision import RadioVisionModel


class NemotronOmniProjector(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int, output_size: int, bias: bool = False) -> None:
        super().__init__()
        self.norm = RMSNorm(RMSNormConfig(input_size, eps=1e-5))
        self.linear1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear2 = nn.Linear(intermediate_size, output_size, bias=bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.linear2(torch.relu(self.linear1(self.norm(hidden_states))).square())


class NemotronHOmniModel(nn.Module):
    def __init__(self, config: NemotronHOmniConfig) -> None:
        super().__init__()
        self.config = config
        if config.downsample_ratio <= 0 or config.downsample_ratio > 1:
            raise ValueError("downsample_ratio must be the reciprocal of a positive integer")
        self.spatial_merge_size = round(1 / config.downsample_ratio)
        if config.downsample_ratio != 1 / self.spatial_merge_size:
            raise ValueError("downsample_ratio must be the reciprocal of a positive integer")
        if getattr(config, "ps_version", "v2") != "v2":
            raise ValueError("Nemotron Omni image execution requires ps_version='v2'")
        self.language_model = NemotronHModel(config.llm_config)
        self.visual = RadioVisionModel(config.vision_config, config.norm_mean, config.norm_std)
        self.vision_projector = NemotronOmniProjector(
            config.vit_hidden_size * self.spatial_merge_size**2,
            config.projector_hidden_size,
            config.llm_config.hidden_size,
        )

    def get_image_features(self, pixel_values: Tensor) -> Tensor:
        patch_size = self.config.vision_config.patch_size
        merge_size = self.spatial_merge_size
        if any(size % (patch_size * merge_size) for size in pixel_values.shape[-2:]):
            raise ValueError("Image dimensions must be divisible by patch_size * spatial_merge_size")
        image_features = self.visual(pixel_values)
        batch_size, _, hidden_size = image_features.shape
        rows, cols = (size // patch_size for size in pixel_values.shape[-2:])
        image_features = image_features.reshape(batch_size, rows, cols // merge_size, hidden_size * merge_size)
        image_features = image_features.permute(0, 2, 1, 3).reshape(
            batch_size, cols // merge_size, rows // merge_size, hidden_size * merge_size**2
        )
        image_features = image_features.transpose(1, 2).reshape(batch_size, -1, hidden_size * merge_size**2)
        return self.vision_projector(image_features)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        pixel_values: Tensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        routed_experts: torch.Tensor | None = None,
    ) -> BaseModelOutput:
        inputs_embeds = None
        if pixel_values is not None:
            if self.language_model.cp_context.cp_enabled:
                raise NotImplementedError("Nemotron Omni image context parallelism is not supported yet")
            image_features = self.get_image_features(pixel_values)
            image_mask = input_ids == self.config.img_context_token_id
            if image_mask.sum() != image_features.shape[0] * image_features.shape[1]:
                raise ValueError("Image placeholder count must match the number of projected image tokens")
            inputs_embeds = self.language_model.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1), image_features.to(inputs_embeds.dtype)
            )
        return self.language_model(
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
            routed_experts=routed_experts,
        )


class NemotronHOmniForCausalLM(NemotronHPreTrainedModel):
    config_class = NemotronHOmniConfig
    supports_packed_multimodal_training = False

    def __init__(self, config: NemotronHOmniConfig) -> None:
        super().__init__(config)
        self.model = NemotronHOmniModel(config)
        self.lm_head = VanillaOutputLinear(config.llm_config.hidden_size, config.llm_config.vocab_size)

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return super().keep_in_fp32_for_weight_transfer(name) or name.endswith(("visual.norm_mean", "visual.norm_std"))

    is_hf_state_dict = staticmethod(is_hf_state_dict)
    is_prime_state_dict = staticmethod(is_prime_state_dict)
    conversion_chain = staticmethod(conversion_chain)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.model.language_model.embed_tokens = embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        temperature: torch.Tensor | None = None,
        sampling_mask: torch.Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        pixel_values: Tensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> PrimeLmOutput:
        hidden_states = self.model(
            input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
            routed_experts=routed_experts,
        )
        return self.lm_head(
            hidden_states.last_hidden_state, labels, temperature=temperature, sampling_mask=sampling_mask
        )

    @classmethod
    def convert_adapter_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        state_dict = super().convert_adapter_to_hf(state_dict)
        return {
            name.replace("model.language_model.", "language_model.backbone.", 1): tensor
            for name, tensor in state_dict.items()
        }
