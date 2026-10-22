"""VLMs are the ``_CUSTOM_VLM_MAPPING`` models. They keep the vision encoder at ``model.model.visual``
and the decoder at ``model.model.language_model``; text models keep the decoder at ``model.model``."""

import base64
from io import BytesIO
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from prime_rl.configs.trainer import ModelConfig
from prime_rl.transports.batch import MMRefs
from prime_rl.utils.logger import get_logger


def get_vision_encoder(model: nn.Module) -> nn.Module | None:
    return getattr(model.model, "visual", None)


def get_language_model(model: nn.Module) -> nn.Module:
    return getattr(model.model, "language_model", model.model)


def get_layer_prefix(model: nn.Module) -> str:
    """Weight key prefix of the decoder layers."""
    return "model.language_model.layers." if hasattr(model.model, "language_model") else "model.layers."


def setup_processor(config: ModelConfig):
    """Load an ``AutoProcessor`` for VLM models. Returns ``None`` for text-only models."""
    from transformers import AutoProcessor

    logger = get_logger()
    try:
        processor = AutoProcessor.from_pretrained(config.name, trust_remote_code=config.trust_remote_code)
    except (ValueError, OSError, KeyError) as e:
        logger.debug(f"No AutoProcessor available for {config.name} ({type(e).__name__}); treating as text-only.")
        return None
    if not (getattr(processor, "image_processor", None) or getattr(processor, "video_processor", None)):
        logger.debug(f"AutoProcessor for {config.name} has no image/video processor; treating as text-only.")
        return None
    logger.info(f"Loaded multimodal processor: {type(processor).__name__}")
    return processor


def _load_image(data_url: str) -> Image.Image:
    header, separator, payload = data_url.partition(",")
    if not separator or not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError("Multimodal training requires base64 data image URLs")
    with Image.open(BytesIO(base64.b64decode(payload, validate=True))) as image:
        return image.convert("RGB")


def materialize_images(refs: MMRefs, processor: Any) -> dict[str, torch.Tensor]:
    """Decode the image refs of a micro batch into the Qwen3.5 vision forward kwargs."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("Multimodal samples require a model image processor")
    images = [_load_image(ref.url) for ref in refs.images]
    processed = image_processor(images=images, return_tensors="pt")
    kwargs = {key: torch.as_tensor(processed[key]).contiguous() for key in ("pixel_values", "image_grid_thw")}
    merge_size = int(image_processor.merge_size)
    # HF Qwen-VL / renderer pad count: T*H*W / merge_size^2.
    lengths = [int(grid.prod()) // (merge_size * merge_size) for grid in kwargs["image_grid_thw"].reshape(-1, 3)]
    expected = [ref.length for ref in refs.images]
    if lengths != expected:
        raise ValueError(f"Image placeholder lengths differ from vLLM: expected {expected}, got {lengths}")
    return kwargs
