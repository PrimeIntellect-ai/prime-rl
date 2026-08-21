from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import torch
from PIL import Image

from prime_rl.multimodal import MaterializedMM, MultimodalAdapter
from prime_rl.transports.rollouts import MMRefs


def _load_image(data_url: str) -> Image.Image:
    header, separator, payload = data_url.partition(",")
    if not separator or not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError("Multimodal training requires base64 data image URLs")
    with Image.open(BytesIO(base64.b64decode(payload, validate=True))) as image:
        return image.convert("RGB")


def materialize_mm_refs(refs: MMRefs, processor: Any, adapter: MultimodalAdapter) -> MaterializedMM:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("Multimodal samples require a model image processor")
    images = [_load_image(ref.url) for ref in refs.images]
    return adapter.materialize(image_processor, images, [ref.length for ref in refs.images])


def stage_mm_refs(
    refs: MMRefs,
    processor: Any,
    adapter: MultimodalAdapter,
    device: torch.device | str,
) -> MaterializedMM:
    """Materialize one microbatch on ``device`` without retaining its CPU tensors."""
    materialized = materialize_mm_refs(refs, processor, adapter)
    device_kwargs = {key: value.to(device) for key, value in materialized.kwargs.items()}

    # ``Tensor.to`` is synchronous here, so the destination owns usable storage
    # before the CPU processor outputs are released. If this copy becomes
    # non-blocking, the source lifetime must instead be tied to a CUDA event.
    materialized.kwargs.clear()
    return MaterializedMM(kwargs=device_kwargs, forward_policy=materialized.forward_policy)


def release_staged_mm(materialized: MaterializedMM) -> None:
    """Drop the final Python references to one microbatch's device payload."""
    materialized.kwargs.clear()
