from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from renderers.mm_store import RAW_MM_ITEM_KIND, RAW_MM_ITEM_VERSION


@dataclass(frozen=True)
class RawMMItem:
    modality: str
    family: str
    layout_fingerprint: str
    raw_image_uri: str
    payload: dict[str, Any]
    raw_ref: str | None = None
    vllm_modality: str | None = None


def _descriptor_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"v1 multimodal sidecars must be raw descriptor dicts, got {type(value).__name__}")


def _required_str(value: Mapping[str, Any], field: str) -> str:
    item = value.get(field)
    if isinstance(item, str) and item:
        return item
    raise ValueError(f"raw multimodal descriptor is missing {field}")


def _optional_str(value: Mapping[str, Any], field: str) -> str | None:
    item = value.get(field)
    if item is None or isinstance(item, str):
        return item
    raise ValueError(f"raw multimodal descriptor {field} must be a string when present")


def _payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("raw multimodal descriptor payload must be a dict")
    return {str(k): v for k, v in payload.items()}


def _validate_envelope(value: Mapping[str, Any]) -> None:
    if value.get("kind") != RAW_MM_ITEM_KIND:
        raise ValueError("raw multimodal descriptor is missing the common envelope kind")
    if value.get("version") != RAW_MM_ITEM_VERSION:
        raise ValueError(f"unsupported raw multimodal descriptor version: {value.get('version')!r}")


def parse_raw_mm_item(value: Any) -> RawMMItem:
    descriptor = _descriptor_mapping(value)
    _validate_envelope(descriptor)
    return RawMMItem(
        modality=_required_str(descriptor, "modality"),
        family=_required_str(descriptor, "family"),
        layout_fingerprint=_required_str(descriptor, "layout_fingerprint"),
        raw_image_uri=_required_str(descriptor, "raw_image_uri"),
        payload=_payload(descriptor),
        raw_ref=_optional_str(descriptor, "raw_ref"),
        vllm_modality=_optional_str(descriptor, "vllm_modality"),
    )
