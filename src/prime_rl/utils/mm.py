"""Shared multimodal helpers used by both the orchestrator (flag-off path,
materialize pixels then ship heavy mm_kwargs) and the trainer (flag-on path,
ship lightweight mm_refs then materialize pixels in the data loader).

Factoring these here keeps the two paths byte-identical and avoids a
trainer→orchestrator import dependency.
"""

from typing import Any

import torch

from prime_rl.transport.types import EncodedTensor, MMRefs


def reconstruct_mm_pixels(renderer: Any, descriptor: dict, messages: list) -> Any:
    """Re-attach ``pixel_values`` to a descriptor-only union mm_data.

    Delegates to the renderer's ``materialize_pixels`` (hash-matched reprocess
    of the window images, with a ``grid_thw`` assert). The descriptor's
    ``image_grid_thw`` is decoded from its msgpack wire shape back to numpy
    first, so the renderer's numpy-vs-numpy grid assert holds after transport.
    """
    from renderers.base import MultiModalData
    from verifiers.utils.serve_utils import decode_tensor_payload

    items = descriptor.get("mm_items") or {}
    decoded_items: dict[str, list] = {}
    for modality, lst in items.items():
        new_lst: list[dict[str, Any]] = []
        for item in lst or []:
            item = dict(item)
            grid = item.get("image_grid_thw")
            if item.get("pixel_values") is None and grid is not None:
                item["image_grid_thw"] = decode_tensor_payload(grid, to_torch=False)
            new_lst.append(item)
        decoded_items[modality] = new_lst

    md = MultiModalData(
        mm_hashes=descriptor.get("mm_hashes") or {},
        mm_placeholders={},
        mm_items=decoded_items,
    )
    return renderer.materialize_pixels(md, messages)


def pack_mm_kwargs_tensors(mm_data: Any) -> "dict[str, torch.Tensor] | None":
    """Batch the renderer's per-image ``mm_items`` into model-agnostic forward
    kwargs, returning torch tensors (not encoded bytes).

    ``mm_data`` may arrive as a ``MultiModalData`` instance (in-process for
    tests) or as a plain dict (after msgpack round-trip from the env-worker).
    Each item is a dict keyed by the names the model's ``forward`` expects
    (``pixel_values`` + ``image_grid_thw`` for Qwen3-VL, just ``pixel_values``
    for Gemma3-VL, etc.). We batch by ``torch.cat(..., dim=0)`` per key —
    generic because every HF VLM processor emits a leading batch/patch
    dimension, and the renderer always processes one image per call.

    Returns a dict of torch tensors keyed by kwarg name, or ``None`` when no
    multimodal data is present.
    """
    from verifiers.utils.serve_utils import decode_tensor_payload

    mm_items = mm_data.mm_items if hasattr(mm_data, "mm_items") else (mm_data or {}).get("mm_items") or {}
    # Flatten across modalities into one kwarg dict — the model's forward
    # signature is the schema. ``mm_items`` is typically ``{"image": [...],
    # "video": [...]}`` but each modality's keys don't collide for any HF VLM
    # we ship today.
    per_kwarg: dict[str, list] = {}
    for _modality, items in mm_items.items():
        for item in items or []:
            for key, payload in item.items():
                # ``decode_tensor_payload`` rehydrates the encoded wire shape to
                # torch but passes already-rehydrated numpy through unchanged.
                # ``as_tensor`` normalizes both to torch so ``torch.cat`` below
                # is uniform.
                per_kwarg.setdefault(key, []).append(torch.as_tensor(decode_tensor_payload(payload)))
    if not per_kwarg:
        return None
    out: dict[str, torch.Tensor] = {}
    for key, tensors in per_kwarg.items():
        out[key] = torch.cat(tensors, dim=0).contiguous()
    return out


def encode_mm_kwargs(tensors: dict[str, torch.Tensor]) -> dict[str, EncodedTensor]:
    """Encode packed torch tensors into ``EncodedTensor`` wire payloads."""
    out: dict[str, EncodedTensor] = {}
    for key, cat in tensors.items():
        arr = cat.detach().cpu().numpy()
        out[key] = EncodedTensor(dtype=str(arr.dtype), shape=list(arr.shape), data=arr.tobytes())
    return out


def build_image_messages(uris: list[str]) -> list[dict]:
    """Minimal messages ``materialize_pixels`` hash-matches against. Order and
    duplicates are harmless — matching dedups by hash."""
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}} for u in uris]}]


def image_uris_from_messages(messages: list) -> list[str]:
    """Collect every image URI from message ``content`` lists. Keeps order;
    duplicates are fine. Accepts ``file://`` (offloaded) and ``data:image``
    (in-process/non-offloaded)."""
    uris: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url", "")
            if isinstance(url, str) and (url.startswith("file://") or url.startswith("data:image")):
                uris.append(url)
    return uris


def materialize_mm_refs(renderer: Any, refs: MMRefs) -> "dict[str, torch.Tensor] | None":
    """Trainer entry point: reconstruct pixels from refs and pack into forward
    kwargs (torch tensors)."""
    return pack_mm_kwargs_tensors(reconstruct_mm_pixels(renderer, refs.descriptor, build_image_messages(refs.uris)))


def make_defer_mm_validation_hook(trainer_defers: bool, trainer_renderer: Any):
    """Build a MultiRunManager config-validation hook that vets each discovered
    run's orchestrator config against the trainer for deferred materialization.

    On failure the run is rejected at discovery (``get_orchestrator_config`` writes
    ``config_validation_error.txt`` and returns None, so the run is never registered
    or packed — this is rejection, not ``evicted.txt`` eviction). That keeps one
    misconfigured run from either crashing all ranks later inside ``get_batch`` or
    silently materializing with the wrong image processor.
    """

    def validate(orch_config: Any) -> "tuple[bool, str]":
        if not getattr(orch_config, "defer_mm_materialization", False):
            return True, ""  # run ships pixels (mm_kwargs) — trainer handles regardless
        if not trainer_defers:
            return False, (
                "run sets defer_mm_materialization=true but the trainer does not — the trainer has no "
                "renderer to materialize mm_refs and the run's batches would fail in get_batch. "
                "Enable defer_mm_materialization and set [renderer] on the trainer."
            )
        # Both defer: the trainer materializes every run's images with its single
        # renderer, so the families must match. Auto resolves against the (shared)
        # base model → compatible. Within-family processor drift is backstopped by
        # the grid skew-assert in materialize_pixels.
        orch_r = getattr(orch_config, "renderer", None)
        if (
            orch_r is not None
            and type(orch_r).__name__ != "AutoRendererConfig"
            and type(orch_r) is not type(trainer_renderer)
        ):
            return False, (
                f"run renderer {type(orch_r).__name__} is a different family than the trainer renderer "
                f"{type(trainer_renderer).__name__}; deferred materialization would use the wrong image processor."
            )
        return True, ""

    return validate
