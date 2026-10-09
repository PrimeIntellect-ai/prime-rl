from __future__ import annotations

import base64
import threading
from io import BytesIO
from typing import Any

import torch
from cachetools import LRUCache
from PIL import Image

from prime_rl.multimodal import MaterializedMM, MultimodalAdapter
from prime_rl.transports.batch import MMImageRef, MMRefs

_GIB = 1024**3

CacheKey = tuple[str, int] # image data url, placeholder length
ProcessedImage = dict[str, torch.Tensor] # processor output, pixel_values and image_grid_thw
CacheStats = tuple[int, int, int] # hits, lookups, current size

class ImageCache:
    """
    Trainer side processed image cache similar to
    https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/cache/base.py
    """
    def __init__(self, max_gb: float, dtype: torch.dtype):
        self._dtype = dtype
        self._cache: LRUCache[CacheKey, ProcessedImage] = LRUCache(
            maxsize=int(max_gb * _GIB),
            getsizeof=lambda image: sum(tensor.nbytes for tensor in image.values()),
        )
        # micro batches are prepared on worker threads that share this cache,
        # so we use lock to protect cache access
        self._lock = threading.Lock()
        # some stats to track cache hit rate
        self._hits = 0
        self._lookups = 0

    def get(self, key: CacheKey) -> ProcessedImage | None:
        with self._lock:
            image = self._cache.get(key)
            self._lookups += 1
            self._hits += image is not None
            return image

    def put(self, key: CacheKey, image: ProcessedImage) -> None:
        with self._lock:
            if self._cache.getsizeof(image) <= self._cache.maxsize:
                self._cache[key] = image

    def pop_stats(self) -> CacheStats:
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

def process_image(
    ref: MMImageRef,
    image_processor: Any,
    adapter: MultimodalAdapter,
    cache: ImageCache,
) -> ProcessedImage:
    key = (ref.url, ref.length)
    image = cache.get(key)
    if image is None:
        image = adapter.materialize(image_processor, [_load_image(ref.url)], [ref.length]).kwargs
        for name, tensor in image.items():
            if tensor.is_floating_point():
                image[name] = tensor.to(cache.dtype)
        cache.put(key, image)
    return image

def materialize_mm_refs(
    refs: MMRefs, processor: Any, adapter: MultimodalAdapter, cache: ImageCache | None = None
) -> MaterializedMM:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("Multimodal samples require a model image processor")
    if cache is None:
        images = [_load_image(ref.url) for ref in refs.images]
        return adapter.materialize(image_processor, images, [ref.length for ref in refs.images])
    processed = [process_image(ref, image_processor, adapter, cache) for ref in refs.images]
    batch = {name: torch.cat([item[name] for item in processed]) for name in processed[0]}
    return MaterializedMM(kwargs=batch, forward_policy=adapter.forward_policy)
