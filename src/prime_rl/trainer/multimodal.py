from __future__ import annotations

import base64
import threading
from io import BytesIO
from typing import Any

import torch
from cachetools import LRUCache
from PIL import Image

from prime_rl.multimodal import MaterializedMM, MultimodalAdapter
from prime_rl.transports.batch import MMRefs

_GIB = 1024**3

class ImageCache:
    """
    Trainer side processed image cache similar to
    https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/cache/base.py
    """
    def __init__(self, max_gb: float):
        self._cache = LRUCache[tuple[str, int], dict[str, torch.Tensor]] = LRUCache(
            maxsize=int(max_gb * _GIB),
            getsizeof=lambda kwargs: sum(tensor.nbytes for tensor in kwargs.values()),
        )
        # trainer is multi-process so we use lock to protect
        # cache access
        self._lock = threading.Lock()
        self._hits = 0
        self._lookups = 0

    def get(self, key: tuple[str, int]) -> dict[str, torch.Tensor] | None:
        with self._lock:
            kwargs = self._cache.get(key)
            self._lookups += 1
            self._hits += kwargs is not None
            return kwargs

    def put(self, key: tuple[str, int], kwargs: dict[str, torch.Tensor]) -> None:
        with self._lock:
            if self._cache.getsizeof(kwargs) <= self._cache.maxsize:
                self._cache[key] = kwargs

    def pop_stats(self) -> tuple[int, int, int]:
        with self._lock:
            stats = (self._hits, self._lookups, int(self._cache.currsize))
            self._hits = 0
            self._lookups = 0
            return stats


def _load_image(data_url: str) -> Image.Image:
    header, separator, payload = data_url.partition(",")
    if not separator or not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError("Multimodal training requires base64 data image URLs")
    with Image.open(BytesIO(base64.b64decode(payload, validate=True))) as image:
        return image.convert("RGB")


def materialize_mm_refs(
    refs: MMRefs, processor: Any, adapter: MultimodalAdapter, cache: ImageCache | None = None
) -> MaterializedMM:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("Multimodal samples require a model image processor")
    if cache is None:
        images = [_load_image(ref.url) for ref in refs.images]
        return adapter.materialize(image_processor, images, [ref.length for ref in refs.images])
    per_image: list[dict[str, torch.Tensor]] = []
    for ref in refs.images:
        key = (ref.url, ref.length)
        kwargs = cache.get(key)
        if kwargs is None:
            kwargs = adapter.materialize(image_processor, [_load_image(ref.url)], [ref.length]).kwargs
            cache.put(key, kwargs)
        per_image.append(kwargs)
    # torch.cat copies, so cached tensors are never handed to training loop
    kwargs = {name: torch.cat([item[name] for item in per_image]) for name in per_image[0]}
    return MaterializedMM(kwargs=kwargs, forward_policy=adapter.forward_policy)



