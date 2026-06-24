from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from prime_rl.multimodal.adapters.base import MaterializedMM
from prime_rl.multimodal.registry import get_multimodal_adapter
from prime_rl.multimodal.schema import (
    RawMMItem,
    contains_processed_payload_key,
    parse_raw_mm_item,
)
from prime_rl.transport.types import MMRefs

IMAGE_MODALITY = "image"
SUPPORTED_MODALITIES = {IMAGE_MODALITY}
PROCESSED_MM_KEYS = {"pixel_values", "image_embeds", "image_features"}


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
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            continue
        if not Path(unquote(parsed.path)).exists():
            missing.append(uri)
    return missing


def image_file_uris_from_messages(messages: Iterable[Any]) -> list[str]:
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


def _normalize_json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple):
        return [_normalize_json_value(v, f"{path}[]") for v in value]
    if isinstance(value, list):
        return [_normalize_json_value(v, f"{path}[]") for v in value]
    if isinstance(value, Mapping):
        return {str(k): _normalize_json_value(v, f"{path}.{k}") for k, v in value.items()}
    raise TypeError(
        f"v1 multimodal sidecars must be JSON-safe raw image descriptors; {path} has unsupported {type(value).__name__}"
    )


def validate_raw_mm_item(item: Mapping[str, Any]) -> dict[str, Any]:
    if contains_processed_payload_key(item):
        raise TypeError(
            "v1 multimodal sidecars must be raw image descriptors, not processed payloads "
            f"({', '.join(sorted(PROCESSED_MM_KEYS))})"
        )
    normalized = {str(k): _normalize_json_value(v, str(k)) for k, v in item.items()}
    parse_raw_mm_item(normalized)
    return normalized


def _validate_modalities(mm_items: Mapping[str, list[Any]]) -> None:
    unsupported = sorted(
        modality for modality, items in mm_items.items() if items and modality not in SUPPORTED_MODALITIES
    )
    if unsupported:
        raise NotImplementedError(
            "v1 multimodal training currently supports raw image refs only; "
            f"unsupported modalities: {', '.join(unsupported)}"
        )


def _placeholder_dict(placeholder: Any) -> dict[str, int]:
    return {
        "offset": int(_field(placeholder, "offset")),
        "length": int(_field(placeholder, "length")),
    }


def build_mm_refs(multi_modal_data: Any, messages: Iterable[Any]) -> MMRefs | None:
    mm_items = _field(multi_modal_data, "mm_items", None)
    if not mm_items:
        return None
    _validate_modalities(mm_items)

    image_items = [validate_raw_mm_item(item) for item in mm_items.get(IMAGE_MODALITY, [])]
    if not image_items:
        return None

    mm_hashes = _field(multi_modal_data, "mm_hashes", {}) or {}
    image_hashes = list(mm_hashes.get(IMAGE_MODALITY, []))
    if len(image_hashes) != len(image_items):
        raise ValueError(
            "Raw image descriptor/hash mismatch: "
            f"{len(image_items)} image descriptors but {len(image_hashes)} image hashes"
        )

    mm_placeholders = _field(multi_modal_data, "mm_placeholders", {}) or {}
    image_placeholders = [_placeholder_dict(p) for p in mm_placeholders.get(IMAGE_MODALITY, [])]
    if image_placeholders and len(image_placeholders) != len(image_items):
        raise ValueError(
            "Raw image placeholder/descriptor mismatch: "
            f"{len(image_placeholders)} placeholders but {len(image_items)} image descriptors"
        )

    uris = image_file_uris_from_messages(messages)
    if len(uris) != len(image_items):
        raise ValueError(
            "Raw image URI/descriptor mismatch: "
            f"{len(uris)} file refs in messages but {len(image_items)} image descriptors"
        )
    for uri in uris:
        file_uri_to_path(uri)

    return MMRefs(
        descriptor={
            "mm_items": {IMAGE_MODALITY: image_items},
            "mm_hashes": {IMAGE_MODALITY: image_hashes},
            "mm_placeholders": {IMAGE_MODALITY: image_placeholders},
        },
        uris=uris,
    )


def sha256_32(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def _single_family_adapter(items: list[RawMMItem]):
    families = {item.family for item in items}
    if len(families) != 1:
        raise ValueError(f"Raw multimodal refs must use exactly one adapter family, got {sorted(families)}")
    return get_multimodal_adapter(next(iter(families)))


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

    def materialize(self, refs: MMRefs | None) -> MaterializedMM | None:
        if refs is None:
            return None

        image_item_dicts = refs.descriptor.get("mm_items", {}).get(IMAGE_MODALITY, [])
        image_hashes = refs.descriptor.get("mm_hashes", {}).get(IMAGE_MODALITY, [])
        if not image_item_dicts:
            return None
        if len(refs.uris) != len(image_item_dicts) or len(image_hashes) != len(image_item_dicts):
            raise ValueError(
                "Raw image refs must have matching URI, descriptor, and hash counts "
                f"(uris={len(refs.uris)}, descriptors={len(image_item_dicts)}, hashes={len(image_hashes)})"
            )
        image_items = [parse_raw_mm_item(validate_raw_mm_item(item)) for item in image_item_dicts]
        adapter = _single_family_adapter(image_items)
        actual_fingerprint = adapter.processor_fingerprint(self.image_processor)
        for item in image_items:
            if item.layout_fingerprint != actual_fingerprint:
                raise ValueError(
                    "Raw image layout fingerprint mismatch: "
                    f"expected {item.layout_fingerprint}, got {actual_fingerprint}"
                )

        from PIL import Image

        images = []
        for uri, expected_hash in zip(refs.uris, image_hashes, strict=True):
            raw = file_uri_to_path(uri).read_bytes()
            actual_hash = sha256_32(raw)
            if actual_hash != expected_hash:
                raise ValueError(f"Raw image hash mismatch for {uri}: expected {expected_hash}, got {actual_hash}")
            images.append(Image.open(BytesIO(raw)).convert("RGB"))

        return adapter.materialize_for_trainer(self.image_processor, image_items, images)

    def synthesize_placeholder(self, refs: MMRefs | None) -> MaterializedMM | None:
        """Build zero-valued multimodal tensors via the owning adapter."""
        if refs is None:
            return None

        image_item_dicts = refs.descriptor.get("mm_items", {}).get(IMAGE_MODALITY, [])
        image_hashes = refs.descriptor.get("mm_hashes", {}).get(IMAGE_MODALITY, [])
        if not image_item_dicts:
            return None
        if len(refs.uris) != len(image_item_dicts) or len(image_hashes) != len(image_item_dicts):
            raise ValueError(
                "Raw image refs must have matching URI, descriptor, and hash counts "
                f"(uris={len(refs.uris)}, descriptors={len(image_item_dicts)}, hashes={len(image_hashes)})"
            )
        image_items = [parse_raw_mm_item(validate_raw_mm_item(item)) for item in image_item_dicts]
        adapter = _single_family_adapter(image_items)
        return adapter.synthesize_placeholder(self.image_processor, image_items)
