from __future__ import annotations

import json
import math
import os
import socket
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

_TRUE_VALUES = {"1", "true", "yes", "on"}
_DEFAULT_SELECTED_SUBSTRINGS = (
    "embed",
    "lm_head",
    "norm",
    "mamba",
    "mixer",
    "attention",
    "attn",
    "router",
    "gate",
    "expert",
)


def weight_transfer_stats_enabled() -> bool:
    return os.getenv("PRIME_RL_WEIGHT_TRANSFER_STATS", "").lower() in _TRUE_VALUES


def parse_step_from_weight_dir(weight_dir: str | Path | None) -> int | None:
    if weight_dir is None:
        return None
    name = Path(weight_dir).name
    if not name.startswith("step_"):
        return None
    try:
        return int(name.removeprefix("step_"))
    except ValueError:
        return None


def derive_weight_transfer_stats_dir(
    *,
    weight_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path | None:
    if not weight_transfer_stats_enabled():
        return None

    explicit = os.getenv("PRIME_RL_WEIGHT_TRANSFER_STATS_DIR")
    if explicit:
        return Path(explicit)

    if weight_dir is not None:
        path = Path(weight_dir)
        if path.parent.name == "broadcasts":
            # .../<output>/run_default/broadcasts/step_N
            return path.parent.parent.parent / "weight_transfer_stats"
        return path.parent / "weight_transfer_stats"

    if output_dir is not None:
        return Path(output_dir) / "weight_transfer_stats"

    return None


def record_state_dict_stats(
    boundary: str,
    state_dict: dict[str, Tensor],
    *,
    layer_id: int | None = None,
    step: int | None = None,
    weight_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    rank: int | None = None,
) -> None:
    stats_dir = derive_weight_transfer_stats_dir(weight_dir=weight_dir, output_dir=output_dir)
    if stats_dir is None:
        return

    records = []
    for index, (name, tensor) in enumerate(state_dict.items()):
        if not _should_record_tensor(name, index, model_tensor=False):
            continue
        records.append(_tensor_stats(boundary, name, tensor, layer_id=layer_id, step=step, rank=rank))
    _append_records(stats_dir, boundary, records)


def trace_state_iter(
    state_iter: Iterable[tuple[str, Tensor]],
    *,
    boundary: str,
    layer_id: int | None = None,
    step: int | None = None,
    weight_dir: str | Path | None = None,
    rank: int | None = None,
) -> Iterator[tuple[str, Tensor]]:
    stats_dir = derive_weight_transfer_stats_dir(weight_dir=weight_dir)
    if stats_dir is None:
        yield from state_iter
        return

    for index, (name, tensor) in enumerate(state_iter):
        if _should_record_tensor(name, index, model_tensor=False):
            _append_records(
                stats_dir,
                boundary,
                [_tensor_stats(boundary, name, tensor, layer_id=layer_id, step=step, rank=rank)],
            )
        yield name, tensor


def record_model_tensor_stats(
    boundary: str,
    model: Module,
    *,
    step: int | None = None,
    weight_dir: str | Path | None = None,
    rank: int | None = None,
) -> None:
    stats_dir = derive_weight_transfer_stats_dir(weight_dir=weight_dir)
    if stats_dir is None:
        return

    records = []
    for index, (name, tensor) in enumerate(_iter_model_tensors(model)):
        if not _should_record_tensor(name, index, model_tensor=True):
            continue
        records.append(_tensor_stats(boundary, name, tensor, step=step, rank=rank))
    _append_records(stats_dir, boundary, records)


def _iter_model_tensors(model: Module) -> Iterator[tuple[str, Tensor]]:
    yield from model.named_parameters()
    if os.getenv("PRIME_RL_WEIGHT_TRANSFER_STATS_INCLUDE_BUFFERS", "1").lower() in _TRUE_VALUES:
        for name, tensor in model.named_buffers():
            yield f"{name}#buffer", tensor


def _should_record_tensor(name: str, index: int, *, model_tensor: bool) -> bool:
    limit_env = (
        os.getenv("PRIME_RL_WEIGHT_TRANSFER_MODEL_STATS_MAX_TENSORS")
        if model_tensor
        else os.getenv("PRIME_RL_WEIGHT_TRANSFER_STATS_MAX_TENSORS")
    )
    if limit_env:
        try:
            if index >= int(limit_env):
                return False
        except ValueError:
            pass

    scope = (
        os.getenv("PRIME_RL_WEIGHT_TRANSFER_MODEL_STATS_SCOPE")
        if model_tensor
        else os.getenv("PRIME_RL_WEIGHT_TRANSFER_STATS_SCOPE")
    )
    scope = (scope or "selected").lower()
    if scope == "all":
        return True

    selectors = (
        os.getenv("PRIME_RL_WEIGHT_TRANSFER_MODEL_STATS_SELECTORS")
        if model_tensor
        else os.getenv("PRIME_RL_WEIGHT_TRANSFER_STATS_SELECTORS")
    )
    selected_substrings = tuple(
        selector.strip() for selector in selectors.split(",") if selector.strip()
    ) if selectors else _DEFAULT_SELECTED_SUBSTRINGS
    return any(selector in name for selector in selected_substrings)


def _tensor_stats(
    boundary: str,
    name: str,
    tensor: Tensor,
    *,
    layer_id: int | None = None,
    step: int | None = None,
    rank: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "time": time.time(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "rank": rank,
        "env_rank": os.getenv("RANK"),
        "env_local_rank": os.getenv("LOCAL_RANK"),
        "boundary": boundary,
        "step": step,
        "layer_id": layer_id,
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
    }
    try:
        with torch.no_grad():
            detached = tensor.detach()
            record.update(_finite_stats(detached))
            record.update(_sample_stats(detached))
    except Exception as exc:
        record["stats_error"] = repr(exc)
    return record


def _finite_stats(tensor: Tensor) -> dict[str, Any]:
    numel = tensor.numel()
    if numel == 0:
        return {"finite_count": 0, "nan_count": 0, "posinf_count": 0, "neginf_count": 0}

    if not (tensor.is_floating_point() or tensor.is_complex()):
        return {"finite_count": numel, "nan_count": 0, "posinf_count": 0, "neginf_count": 0}

    isfinite = torch.isfinite(tensor)
    finite_count = int(isfinite.sum().item())
    result: dict[str, Any] = {
        "finite_count": finite_count,
        "nan_count": int(torch.isnan(tensor).sum().item()),
        "posinf_count": int(torch.isposinf(tensor).sum().item()) if tensor.is_floating_point() else None,
        "neginf_count": int(torch.isneginf(tensor).sum().item()) if tensor.is_floating_point() else None,
    }
    if finite_count == numel and not tensor.is_complex():
        result["min"] = _safe_float(tensor.min().item())
        result["max"] = _safe_float(tensor.max().item())
        result["absmax"] = _safe_float(tensor.abs().max().item())
        result["mean"] = _safe_float(tensor.float().mean().item())
    return result


def _sample_stats(tensor: Tensor) -> dict[str, Any]:
    numel = tensor.numel()
    if numel == 0:
        return {"sample_count": 0}

    sample_size = _sample_size()
    flat = tensor.reshape(-1)
    stride = max(numel // sample_size, 1)
    sample = flat[::stride][:sample_size]
    if tensor.is_complex():
        sample = sample.real
    sample = sample.float()
    return {
        "sample_count": int(sample.numel()),
        "sample_stride": stride,
        "sample_sum": _safe_float(sample.sum().item()),
        "sample_absmax": _safe_float(sample.abs().max().item()),
    }


def _sample_size() -> int:
    try:
        return max(int(os.getenv("PRIME_RL_WEIGHT_TRANSFER_STATS_SAMPLE_SIZE", "4096")), 1)
    except ValueError:
        return 4096


def _safe_float(value: float) -> float | None:
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def _append_records(stats_dir: Path, boundary: str, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    stats_dir.mkdir(parents=True, exist_ok=True)
    path = stats_dir / f"{boundary}_pid{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
