from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from prime_rl.multimodal.adapters.base import MaterializedMM, MultimodalAdapter
from prime_rl.multimodal.registry import get_multimodal_adapter
from prime_rl.multimodal.schema import RawMMItem, parse_raw_mm_item
from prime_rl.transport.types import MMRefs

IMAGE_MODALITY = "image"
SUPPORTED_MODALITIES = {IMAGE_MODALITY}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def file_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Raw multimodal image refs must be file:// URIs, got {uri!r}")
    if parsed.netloc not in ("", "localhost"):
        raise ValueError(f"file:// multimodal refs must be local paths, got {uri!r}")
    return Path(unquote(parsed.path))


def missing_file_uris(uris: Iterable[str]) -> list[str]:
    """Return missing local ``file://`` image refs; non-file refs are ignored."""
    missing: list[str] = []
    for uri in uris:
        if urlparse(uri).scheme != "file":
            continue
        if not file_uri_to_path(uri).exists():
            missing.append(uri)
    return missing


def image_uris_from_messages(messages: Iterable[Any]) -> list[str]:
    uris: list[str] = []
    for message in messages:
        content = _field(message, "content")
        if not isinstance(content, list):
            continue
        for part in content:
            if _field(part, "type") != "image_url":
                continue
            image_url = _field(part, "image_url")
            url = image_url if isinstance(image_url, str) else _field(image_url, "url")
            if isinstance(url, str):
                uris.append(url)
    return uris


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
    item_dicts = [dict(item) for item in items]
    for item in item_dicts:
        parse_raw_mm_item(item)
    return item_dicts


def build_mm_refs(multi_modal_data: Any, messages: Iterable[Any]) -> MMRefs | None:
    mm_items = _field(multi_modal_data, "mm_items", None)
    if not mm_items:
        return None
    _validate_modalities(mm_items)

    image_items = _raw_item_dicts(mm_items.get(IMAGE_MODALITY, []))
    if not image_items:
        return None

    mm_hashes = _field(multi_modal_data, "mm_hashes", {}) or {}
    image_hashes = list(mm_hashes.get(IMAGE_MODALITY, []))
    if len(image_hashes) != len(image_items):
        raise ValueError(
            "Raw image descriptor/hash mismatch: "
            f"{len(image_items)} image descriptors but {len(image_hashes)} image hashes"
        )

    uris = image_uris_from_messages(messages)
    if len(uris) != len(image_items):
        raise ValueError(
            "Raw image URI/descriptor mismatch: "
            f"{len(uris)} image refs in messages but {len(image_items)} image descriptors"
        )
    for uri in uris:
        file_uri_to_path(uri)

    return MMRefs(
        descriptor={
            "mm_items": {IMAGE_MODALITY: image_items},
            "mm_hashes": {IMAGE_MODALITY: image_hashes},
        },
        uris=uris,
    )


def sha256_32(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def _parse_image_refs(refs: MMRefs) -> tuple[list[RawMMItem], list[str]]:
    image_item_dicts = refs.descriptor.get("mm_items", {}).get(IMAGE_MODALITY, [])
    image_hashes = list(refs.descriptor.get("mm_hashes", {}).get(IMAGE_MODALITY, []))
    if not image_item_dicts:
        return [], []
    if len(refs.uris) != len(image_item_dicts) or len(image_hashes) != len(image_item_dicts):
        raise ValueError(
            "Raw image refs must have matching URI, descriptor, and hash counts "
            f"(uris={len(refs.uris)}, descriptors={len(image_item_dicts)}, hashes={len(image_hashes)})"
        )
    return [parse_raw_mm_item(item) for item in image_item_dicts], image_hashes


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


def _load_verified_images(uris: list[str], expected_hashes: list[str]) -> list[Any]:
    from PIL import Image

    images = []
    for uri, expected_hash in zip(uris, expected_hashes, strict=True):
        raw = file_uri_to_path(uri).read_bytes()
        actual_hash = sha256_32(raw)
        if actual_hash != expected_hash:
            raise ValueError(f"Raw image hash mismatch for {uri}: expected {expected_hash}, got {actual_hash}")
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
        image_items, image_hashes = _parse_image_refs(refs)
        if not image_items:
            return None

        image_processor = self.image_processor
        adapter = _single_family_adapter(image_items)
        _validate_processor_layout(adapter, image_processor, image_items)
        images = _load_verified_images(refs.uris, image_hashes)
        return adapter.materialize_for_trainer(image_processor, image_items, images)

    def synthesize_placeholder(self, refs: MMRefs) -> MaterializedMM | None:
        """Build zero-valued multimodal tensors via the owning adapter."""
        image_items, _ = _parse_image_refs(refs)
        if not image_items:
            return None
        image_processor = self.image_processor
        adapter = _single_family_adapter(image_items)
        return adapter.synthesize_placeholder(image_processor, image_items)
