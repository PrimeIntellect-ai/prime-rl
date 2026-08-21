from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

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
