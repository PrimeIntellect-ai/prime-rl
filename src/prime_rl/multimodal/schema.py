from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

RAW_MM_ITEM_KIND = "prime_raw_mm_item"
RAW_MM_ITEM_VERSION = 1
PROCESSED_MM_KEYS = {"pixel_values", "image_embeds", "image_features"}


@dataclass(frozen=True)
class RawMMItem:
    modality: str
    family: str
    layout_fingerprint: str
    payload: dict[str, Any]
    raw_ref: str | None = None
    vllm_modality: str | None = None


def contains_processed_payload_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(PROCESSED_MM_KEYS.intersection(value)) or any(
            contains_processed_payload_key(v) for v in value.values()
        )
    if isinstance(value, list | tuple):
        return any(contains_processed_payload_key(v) for v in value)
    return False


def parse_raw_mm_item(value: Any) -> RawMMItem:
    if not isinstance(value, Mapping):
        raise TypeError(f"v1 multimodal sidecars must be raw descriptor dicts, got {type(value).__name__}")
    if contains_processed_payload_key(value):
        raise TypeError("v1 multimodal sidecars must not carry processed multimodal payloads")
    if value.get("kind") != RAW_MM_ITEM_KIND:
        raise ValueError("raw multimodal descriptor is missing the common envelope kind")
    if int(value.get("version", -1)) != RAW_MM_ITEM_VERSION:
        raise ValueError(f"unsupported raw multimodal descriptor version: {value.get('version')!r}")
    modality = value.get("modality")
    family = value.get("family")
    layout_fingerprint = value.get("layout_fingerprint")
    payload = value.get("payload")
    if not isinstance(modality, str) or not modality:
        raise ValueError("raw multimodal descriptor is missing modality")
    if not isinstance(family, str) or not family:
        raise ValueError("raw multimodal descriptor is missing family")
    if not isinstance(layout_fingerprint, str) or not layout_fingerprint:
        raise ValueError("raw multimodal descriptor is missing layout_fingerprint")
    if not isinstance(payload, Mapping):
        raise ValueError("raw multimodal descriptor payload must be a dict")
    raw_ref = value.get("raw_ref")
    if raw_ref is not None and not isinstance(raw_ref, str):
        raise ValueError("raw multimodal descriptor raw_ref must be a string when present")
    vllm_modality = value.get("vllm_modality")
    if vllm_modality is not None and not isinstance(vllm_modality, str):
        raise ValueError("raw multimodal descriptor vllm_modality must be a string when present")
    return RawMMItem(
        modality=modality,
        family=family,
        layout_fingerprint=layout_fingerprint,
        payload={str(k): v for k, v in payload.items()},
        raw_ref=raw_ref,
        vllm_modality=vllm_modality,
    )
