from typing import Any

import torch

from prime_rl.trainer.vlm import load_image
from prime_rl.transports.batch import MMRefs


def materialize_qwen_vl_images(refs: MMRefs, processor: Any) -> dict[str, torch.Tensor]:
    """Decode the image refs of a micro batch into Qwen-VL vision forward kwargs."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("Multimodal samples require a model image processor")
    images = [load_image(ref.url) for ref in refs.images]
    processed = image_processor(images=images, return_tensors="pt")
    kwargs = {key: torch.as_tensor(processed[key]).contiguous() for key in ("pixel_values", "image_grid_thw")}
    merge_size = int(image_processor.merge_size)
    # HF Qwen-VL / renderer pad count: T*H*W / merge_size^2.
    lengths = [int(grid.prod()) // (merge_size * merge_size) for grid in kwargs["image_grid_thw"].reshape(-1, 3)]
    expected = [ref.length for ref in refs.images]
    if lengths != expected:
        raise ValueError(f"Image placeholder lengths differ from vLLM: expected {expected}, got {lengths}")
    return kwargs
