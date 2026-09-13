import importlib
import re
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor, nn
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_outputs import BaseModelOutput

from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.models.nemotron_h.modeling_nemotron_h import (
    NemotronHModel,
    NemotronHPreTrainedModel,
)
from prime_rl.trainer.models.nemotron_h_omni.converting_nemotron_h_omni import (
    conversion_chain,
    is_hf_state_dict,
    is_prime_state_dict,
)
from prime_rl.utils.cp import (
    setup_cp_attention_params,
    shard_for_cp,
    shard_position_ids_for_cp,
)

_REMOTE_PROJECTOR_CLASS = "NemotronH_Omni_Reasoning_V3VisionProjector"
_EXPECTED_MODEL_TYPE = "nemotron_h_omni"
_EXPECTED_AUTO_MODEL = "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3"
_IMMUTABLE_REVISION = re.compile(r"[0-9a-fA-F]{40}")


def _local_snapshot_revision(model_path: str) -> str | None:
    path = Path(model_path).expanduser()
    if not path.is_dir():
        return None
    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        if _IMMUTABLE_REVISION.fullmatch(parent.name):
            return parent.name
    return None


@lru_cache
def _load_remote_model_module(
    class_reference: str,
    model_path: str,
    revision: str | None,
):
    remote_model_class = get_class_from_dynamic_module(
        class_reference,
        model_path,
        revision=revision,
    )
    return importlib.import_module(remote_model_class.__module__)


def _load_remote_vision_components(
    config,
    *,
    trust_remote_code: bool,
) -> tuple[type[nn.Module], type[nn.Module]]:
    if trust_remote_code is not True:
        raise ValueError("Nemotron-H Omni requires trust_remote_code=True to load checkpoint vision code")
    if getattr(config, "model_type", None) != _EXPECTED_MODEL_TYPE:
        raise ValueError(f"Nemotron-H Omni remote vision code requires model_type={_EXPECTED_MODEL_TYPE!r}")

    auto_map = getattr(config, "auto_map", None) or {}
    class_reference = auto_map.get("AutoModelForImageTextToText")
    if class_reference != _EXPECTED_AUTO_MODEL:
        raise ValueError(
            "Nemotron-H Omni requires the expected AutoModelForImageTextToText "
            f"class reference {_EXPECTED_AUTO_MODEL!r}"
        )

    model_path = str(getattr(config, "_name_or_path", "") or "")
    if not model_path:
        raise ValueError("Nemotron-H Omni requires a checkpoint model path")
    revision = getattr(config, "_commit_hash", None)
    local_path = Path(model_path).expanduser()
    local_snapshot_revision = _local_snapshot_revision(model_path)
    if local_path.is_dir() and local_snapshot_revision is None:
        raise ValueError(
            "Nemotron-H Omni local checkpoint must resolve under a full immutable 40-hex snapshot directory"
        )
    if not local_path.is_dir() and (not isinstance(revision, str) or _IMMUTABLE_REVISION.fullmatch(revision) is None):
        raise ValueError("Nemotron-H Omni remote vision code requires a full immutable commit revision")
    module = _load_remote_model_module(
        class_reference,
        model_path,
        revision,
    )
    return module.RadioModel, getattr(module, _REMOTE_PROJECTOR_CLASS)


def merge_image_embeddings(
    input_ids: torch.LongTensor,
    inputs_embeds: Tensor,
    image_embeds: Tensor,
    *,
    image_token_id: int,
) -> Tensor:
    image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
    if image_embeds.shape[-1] != inputs_embeds.shape[-1]:
        raise ValueError(
            "Nemotron-H Omni projected image embedding hidden size does not match "
            f"token embeddings: {image_embeds.shape[-1]} != {inputs_embeds.shape[-1]}"
        )
    image_mask = input_ids == image_token_id
    image_token_count = int(image_mask.sum().item())
    if image_token_count != image_embeds.shape[0]:
        raise ValueError(
            "Nemotron-H Omni image token count does not match projected image embeddings: "
            f"{image_token_count} tokens != {image_embeds.shape[0]} embeddings"
        )
    return inputs_embeds.masked_scatter(
        image_mask.unsqueeze(-1).expand_as(inputs_embeds),
        image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
    )


def _is_integer_tensor(value: Tensor) -> bool:
    return value.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }


def _normalize_pixel_values(
    pixel_values: Tensor | None,
    imgs_sizes: torch.LongTensor | None,
    *,
    num_channels: int,
) -> Tensor | list[Tensor] | None:
    if pixel_values is None:
        return None
    if not torch.is_floating_point(pixel_values):
        raise ValueError("Nemotron-H Omni pixel_values must use a floating-point dtype")
    if pixel_values.ndim == 4:
        return pixel_values
    if pixel_values.ndim != 1:
        raise ValueError("Nemotron-H Omni pixel_values must be a flat wire tensor or a 4D image tensor")
    if not isinstance(num_channels, int) or isinstance(num_channels, bool) or num_channels <= 0:
        raise ValueError("Nemotron-H Omni vision channel count must be positive")
    if imgs_sizes is None or imgs_sizes.ndim != 2 or imgs_sizes.shape[1] != 2:
        raise ValueError("Nemotron-H Omni flat pixel_values require imgs_sizes with shape (images, 2)")
    if not _is_integer_tensor(imgs_sizes):
        raise ValueError("Nemotron-H Omni imgs_sizes must use an integer dtype")
    if imgs_sizes.numel() == 0 or not torch.all(imgs_sizes > 0):
        raise ValueError("Nemotron-H Omni imgs_sizes values must be positive")

    image_sizes = imgs_sizes.tolist()
    expected_values = sum(num_channels * height * width for height, width in image_sizes)
    if pixel_values.numel() != expected_values:
        raise ValueError("Nemotron-H Omni flat pixel_values element count does not match imgs_sizes")

    images = []
    offset = 0
    for height, width in image_sizes:
        num_values = num_channels * height * width
        images.append(pixel_values[offset : offset + num_values].reshape(1, num_channels, height, width))
        offset += num_values
    return images


def _validate_image_inputs(
    input_ids: torch.LongTensor,
    pixel_values: Tensor | list[Tensor] | None,
    imgs_sizes: torch.LongTensor | None,
    num_tokens: torch.LongTensor | None,
    num_patches: torch.LongTensor | None,
    mm_token_type_ids: torch.LongTensor | None,
    *,
    image_token_id: int,
    num_channels: int,
) -> None:
    image_mask = input_ids == image_token_id
    if mm_token_type_ids is not None:
        if mm_token_type_ids.shape != input_ids.shape:
            raise ValueError("Nemotron-H Omni mm_token_type_ids must match input_ids")
        if not torch.equal(mm_token_type_ids == 1, image_mask):
            raise ValueError("Nemotron-H Omni image markers do not match image tokens")

    metadata = {
        "imgs_sizes": imgs_sizes,
        "num_tokens": num_tokens,
        "num_patches": num_patches,
    }
    if pixel_values is None:
        if any(value is not None for value in metadata.values()):
            raise ValueError("Nemotron-H Omni image metadata requires pixel_values")
        return
    if any(value is not None for value in metadata.values()) and any(value is None for value in metadata.values()):
        raise ValueError("Nemotron-H Omni image metadata must be supplied together")

    images = list(pixel_values) if isinstance(pixel_values, list) else list(pixel_values.split(1))
    if not images or any(
        image.ndim != 4
        or image.shape[0] != 1
        or image.shape[1] != num_channels
        or image.shape[-2] <= 0
        or image.shape[-1] <= 0
        for image in images
    ):
        raise ValueError(
            f"Nemotron-H Omni image tensors must have shape (1, {num_channels}, positive height, positive width)"
        )
    if any(not torch.is_floating_point(image) for image in images):
        raise ValueError("Nemotron-H Omni pixel_values must use a floating-point dtype")

    num_images = len(images)
    if imgs_sizes is not None:
        if not _is_integer_tensor(imgs_sizes):
            raise ValueError("Nemotron-H Omni imgs_sizes must use an integer dtype")
        if imgs_sizes.shape != (num_images, 2):
            raise ValueError("Nemotron-H Omni imgs_sizes must have shape (images, 2)")
        if not torch.all(imgs_sizes > 0):
            raise ValueError("Nemotron-H Omni imgs_sizes values must be positive")
        for image, (height, width) in zip(images, imgs_sizes.tolist(), strict=True):
            if image.shape[-2:] != (height, width):
                raise ValueError("Nemotron-H Omni imgs_sizes do not match pixel_values")
    for name, value in (("num_tokens", num_tokens), ("num_patches", num_patches)):
        if value is None:
            continue
        if not _is_integer_tensor(value):
            raise ValueError(f"Nemotron-H Omni {name} must use an integer dtype")
        if value.shape != (num_images,):
            raise ValueError(f"Nemotron-H Omni {name} must have shape (images,)")
        if not torch.all(value > 0):
            raise ValueError(f"Nemotron-H Omni {name} values must be positive")
    if num_patches is not None and not torch.all(num_patches == 1):
        raise ValueError("Nemotron-H Omni image inputs require one patch group per image")
    if num_tokens is not None and int(num_tokens.sum().item()) != int(image_mask.sum().item()):
        raise ValueError("Nemotron-H Omni num_tokens do not match image tokens")


def _replace_unit_radio_layer_scales(module: nn.Module, value: float) -> None:
    if value != 1.0:
        raise ValueError(
            "Nemotron-H Omni requires RADIO layerscale_value=1.0 because the "
            "checkpoint does not contain LayerScale parameters"
        )
    for name, child in tuple(module.named_children()):
        if child.__class__.__name__ == "RadioLayerScale":
            setattr(module, name, nn.Identity())
        else:
            _replace_unit_radio_layer_scales(child, value)


class NemotronHOmniModel(nn.Module):
    def __init__(
        self,
        config,
        *,
        trust_remote_code: bool,
    ) -> None:
        super().__init__()
        self.config = config
        vision_model_class, vision_projector_class = _load_remote_vision_components(
            config,
            trust_remote_code=trust_remote_code,
        )
        self.language_model = NemotronHModel(config.llm_config)
        self.vision_model = vision_model_class(config.vision_config)
        _replace_unit_radio_layer_scales(
            self.vision_model,
            config.vision_config.layerscale_value,
        )
        self.vision_model.make_preprocessor_external()
        self.vision_projector = vision_projector_class(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.language_model.embed_tokens = embeddings

    def set_context_parallel_attributes(
        self,
        process_group,
        rank: int,
        world_size: int,
    ) -> None:
        self.language_model.context_parallel_group = process_group
        self.language_model.context_parallel_rank = rank
        self.language_model.context_parallel_world_size = world_size
        self.language_model.set_context_parallel_attributes(
            process_group,
            rank,
            world_size,
        )

    def _dummy_pixel_values(self, inputs_embeds: Tensor) -> Tensor:
        image_size = self.config.patch_size * int(1 / self.config.downsample_ratio)
        num_channels = getattr(self.config.vision_config, "num_channels", 3)
        vision_parameter = next(self.vision_model.parameters())
        return torch.zeros(
            1,
            num_channels,
            image_size,
            image_size,
            device=inputs_embeds.device,
            dtype=vision_parameter.dtype,
        )

    def prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Tensor | None,
        image_flags: torch.LongTensor | None,
        imgs_sizes: torch.LongTensor | None,
        num_tokens: torch.LongTensor | None,
        num_patches: torch.LongTensor | None,
        mm_token_type_ids: torch.LongTensor | None,
    ) -> Tensor:
        has_images = pixel_values is not None
        num_channels = getattr(self.config.vision_config, "num_channels", 3)
        pixel_values = _normalize_pixel_values(
            pixel_values,
            imgs_sizes,
            num_channels=num_channels,
        )
        _validate_image_inputs(
            input_ids,
            pixel_values,
            imgs_sizes,
            num_tokens,
            num_patches,
            mm_token_type_ids,
            image_token_id=self.config.img_context_token_id,
            num_channels=num_channels,
        )
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        if pixel_values is None:
            pixel_values = self._dummy_pixel_values(inputs_embeds)

        if isinstance(pixel_values, list):
            image_embeds = torch.cat(
                [self.vision_projector(image, self.vision_model).flatten(0, 1) for image in pixel_values],
                dim=0,
            )
        else:
            image_embeds = self.vision_projector(pixel_values, self.vision_model)
        if has_images:
            if image_flags is not None:
                image_embeds = image_embeds[image_flags.squeeze(-1) == 1]
            return merge_image_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                image_token_id=self.config.img_context_token_id,
            )
        return inputs_embeds + image_embeds.sum() * 0.0

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        pixel_values: Tensor | None = None,
        image_flags: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        imgs_sizes: torch.LongTensor | None = None,
        num_tokens: torch.LongTensor | None = None,
        num_patches: torch.LongTensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutput:
        has_images = pixel_values is not None
        inputs_embeds = self.prepare_inputs(
            input_ids,
            pixel_values,
            image_flags,
            imgs_sizes,
            num_tokens,
            num_patches,
            mm_token_type_ids,
        )

        process_group = getattr(self.language_model, "context_parallel_group", None)
        if has_images and process_group is not None:
            if position_ids is None:
                position_ids = torch.arange(
                    inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                ).unsqueeze(0)
            rank = self.language_model.context_parallel_rank
            world_size = self.language_model.context_parallel_world_size
            setup_cp_attention_params(
                position_ids,
                cp_group=process_group,
                cp_style="ulysses",
                seq_lens=seq_lens,
            )
            inputs_embeds = shard_for_cp(
                inputs_embeds,
                cp_rank=rank,
                cp_world_size=world_size,
            )
            position_ids = shard_position_ids_for_cp(
                position_ids,
                cp_rank=rank,
                cp_world_size=world_size,
            )
            if routed_experts is not None:
                routed_experts = shard_for_cp(
                    routed_experts,
                    cp_rank=rank,
                    cp_world_size=world_size,
                )
            seq_lens_are_pre_shard = True

        return self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )


class NemotronHOmniForCausalLM(NemotronHPreTrainedModel):
    def __init__(
        self,
        config,
        *,
        _prime_trust_remote_code: bool = False,
    ) -> None:
        super().__init__(config)
        self.model = NemotronHOmniModel(
            config,
            trust_remote_code=_prime_trust_remote_code,
        )
        self.lm_head = VanillaOutputLinear(
            config.llm_config.hidden_size,
            config.llm_config.vocab_size,
        )
        self.supports_packed_multimodal_training = True

    @classmethod
    def from_config(cls, config, trust_remote_code: bool = False, **kwargs):
        return cls._from_config(
            config,
            _prime_trust_remote_code=trust_remote_code,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            trust_remote_code=trust_remote_code,
            _prime_trust_remote_code=trust_remote_code,
            **kwargs,
        )

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_hf_state_dict(state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_prime_state_dict(state_dict)

    @classmethod
    def conversion_chain(cls, config) -> list:
        return conversion_chain(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.model.set_input_embeddings(embeddings)

    def set_context_parallel_attributes(
        self,
        process_group,
        rank: int,
        world_size: int,
    ) -> None:
        self.model.set_context_parallel_attributes(process_group, rank, world_size)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        temperature: Tensor | None = None,
        sampling_mask: Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        pixel_values: Tensor | None = None,
        image_flags: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        imgs_sizes: torch.LongTensor | None = None,
        num_tokens: torch.LongTensor | None = None,
        num_patches: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> PrimeLmOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_flags=image_flags,
            mm_token_type_ids=mm_token_type_ids,
            imgs_sizes=imgs_sizes,
            num_tokens=num_tokens,
            num_patches=num_patches,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )
        return self.lm_head(
            outputs.last_hidden_state,
            labels,
            temperature=temperature,
            sampling_mask=sampling_mask,
        )

    @torch.no_grad()
    def init_buffers_post_meta(self) -> None:
        self.model.vision_model.summary_idxs.copy_(
            torch.tensor(
                self.config.vision_config.summary_idxs,
                device=self.model.vision_model.summary_idxs.device,
            )
        )
        for module in self.modules():
            if isinstance(module, MoE):
                module.tokens_per_expert.zero_()
                module.routing_confidence_sum.zero_()


__all__ = [
    "NemotronHOmniForCausalLM",
    "NemotronHOmniModel",
    "merge_image_embeddings",
]
