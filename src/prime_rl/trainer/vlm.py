"""VLMs are the ``_CUSTOM_VLM_MAPPING`` models. They keep the vision encoder at ``model.model.visual``
and the decoder at ``model.model.language_model``; text models keep the decoder at ``model.model``."""

import torch.nn as nn

from prime_rl.configs.trainer import ModelConfig
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
