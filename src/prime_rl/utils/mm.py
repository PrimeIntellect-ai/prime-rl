from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from io import BytesIO
from typing import Any

from renderers.mm_store import decode_data_image_url

from prime_rl.multimodal.adapters.base import MaterializedMM, MultimodalAdapter
from prime_rl.multimodal.registry import get_multimodal_adapter
from prime_rl.multimodal.schema import RawMMItem, parse_raw_mm_item
from prime_rl.transport.types import MMImageRef, MMRefs

IMAGE_MODALITY = "image"
SUPPORTED_MODALITIES = {IMAGE_MODALITY}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _validate_modalities(mm_items: Mapping[str, list[Any]]) -> None:
    unsupported = sorted(
        modality for modality, items in mm_items.items() if items and modality not in SUPPORTED_MODALITIES
    )
    if unsupported:
        raise NotImplementedError(
            "v1 multimodal training currently supports raw image refs only; "
            f"unsupported modalities: {', '.join(unsupported)}"
        )


def _raw_item_dicts(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    item_dicts: list[dict[str, Any]] = []
    for item in items:
        item_dict = dict(item)
        parse_raw_mm_item(item_dict)
        item_dicts.append(item_dict)
    return item_dicts


def _placeholder_bounds(placeholder: Any) -> tuple[int, int]:
    offset = _field(placeholder, "offset")
    length = _field(placeholder, "length")
    if not isinstance(offset, int) or not isinstance(length, int):
        raise ValueError(f"Raw image placeholder must have integer offset/length, got {placeholder!r}")
    if offset < 0 or length <= 0:
        raise ValueError(f"Raw image placeholder must have offset >= 0 and length > 0, got {placeholder!r}")
    return offset, length


def build_mm_refs(multi_modal_data: Any) -> MMRefs | None:
    mm_items = _field(multi_modal_data, "mm_items", None)
    if not mm_items:
        return None
    _validate_modalities(mm_items)

    image_items = _raw_item_dicts(mm_items.get(IMAGE_MODALITY, []))
    if not image_items:
        return None

    mm_hashes = _field(multi_modal_data, "mm_hashes", {}) or {}
    image_hashes = list(mm_hashes.get(IMAGE_MODALITY, []))
    mm_placeholders = _field(multi_modal_data, "mm_placeholders", {}) or {}
    image_placeholders = list(mm_placeholders.get(IMAGE_MODALITY, []))
    if len(image_hashes) != len(image_items) or len(image_placeholders) != len(image_items):
        raise ValueError(
            "Raw image descriptor/hash/placeholder mismatch: "
            f"descriptors={len(image_items)}, hashes={len(image_hashes)}, placeholders={len(image_placeholders)}"
        )

    images: list[MMImageRef] = []
    prev_end = 0
    for item, image_hash, placeholder in zip(image_items, image_hashes, image_placeholders, strict=True):
        offset, length = _placeholder_bounds(placeholder)
        # Truncation cuts at a prefix of ``images``, which is only sound if
        # placeholders arrive in token order without overlap.
        if offset < prev_end:
            raise ValueError(f"Raw image placeholders must be sorted and non-overlapping, got {image_placeholders!r}")
        prev_end = offset + length
        images.append(MMImageRef(item=item, hash=image_hash, offset=offset, length=length))
    return MMRefs(images=images)


def sha256_32(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def _parse_image_refs(refs: MMRefs) -> list[RawMMItem]:
    return [parse_raw_mm_item(image.item) for image in refs.images]


def _single_family_adapter(items: list[RawMMItem]) -> MultimodalAdapter:
    families = {item.family for item in items}
    if len(families) != 1:
        raise ValueError(f"Raw multimodal refs must use exactly one adapter family, got {sorted(families)}")
    return get_multimodal_adapter(next(iter(families)))


def _validate_processor_layout(adapter: MultimodalAdapter, image_processor: Any, image_items: list[RawMMItem]) -> None:
    actual_fingerprint = adapter.processor_fingerprint(image_processor)
    for item in image_items:
        if item.layout_fingerprint != actual_fingerprint:
            raise ValueError(
                f"Raw image layout fingerprint mismatch: expected {item.layout_fingerprint}, got {actual_fingerprint}"
            )


def _load_verified_images(image_refs: list[MMImageRef], image_items: list[RawMMItem]) -> list[Any]:
    from PIL import Image

    images = []
    for ref, item in zip(image_refs, image_items, strict=True):
        raw = decode_data_image_url(item.raw_image_data)
        actual_hash = sha256_32(raw)
        if actual_hash != ref.hash:
            raise ValueError(f"Raw image hash mismatch: expected {ref.hash}, got {actual_hash}")
        with Image.open(BytesIO(raw)) as image:
            images.append(image.convert("RGB"))
    return images


class RawImageMaterializer:
    """Materialize raw image refs with the trainer model's HF image processor."""

    def __init__(self, model_name: str, *, trust_remote_code: bool):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self._image_processor = None

    @property
    def image_processor(self):
        if self._image_processor is None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
            image_processor = getattr(processor, "image_processor", None)
            if image_processor is None:
                raise ValueError(f"{self.model_name!r} does not expose an image_processor")
            self._image_processor = image_processor
        return self._image_processor

    def materialize(self, refs: MMRefs) -> MaterializedMM | None:
        image_items = _parse_image_refs(refs)
        if not image_items:
            return None

        image_processor = self.image_processor
        adapter = _single_family_adapter(image_items)
        _validate_processor_layout(adapter, image_processor, image_items)
        images = _load_verified_images(refs.images, image_items)
        return adapter.materialize_for_trainer(image_processor, image_items, images)
