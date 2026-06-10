from __future__ import annotations

import json
import math
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any


FALSE_VALUES = {"", "0", "false", "False", "no", "off"}


def enabled() -> bool:
    return os.environ.get("PRIME_NAN_TRACE", "") not in FALSE_VALUES


def weight_trace_enabled() -> bool:
    return enabled() and os.environ.get("PRIME_WEIGHT_UPDATE_TRACE", "") not in FALSE_VALUES


def heavy_trace_enabled() -> bool:
    return enabled() and os.environ.get("PRIME_NAN_TRACE_HEAVY", "") not in FALSE_VALUES


def fail_fast_enabled() -> bool:
    return os.environ.get("PRIME_NAN_FAIL_FAST", "") not in FALSE_VALUES


def trace_dir() -> Path:
    configured = os.environ.get("PRIME_NAN_TRACE_DIR")
    if configured:
        return Path(configured)
    run_root = os.environ.get("QWEN_RUN_ROOT")
    if run_root:
        return Path(run_root) / "trace"
    return Path("/tmp/prime-nan-trace")


def _json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return repr(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in value[:4096]]
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return tensor_summary(value)
    except Exception:
        pass
    return repr(value)


def write_event(kind: str, **fields: Any) -> None:
    if not enabled():
        return
    path = trace_dir()
    path.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "kind": kind,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        **fields,
    }
    with open(path / f"{kind}.jsonl", "a") as f:
        f.write(json.dumps(_json_safe(record), allow_nan=False, sort_keys=True) + "\n")


def tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    detached = tensor.detach()
    summary: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
    }
    if detached.numel() == 0:
        summary.update({"finite": True, "finite_count": 0, "nan_count": 0, "posinf_count": 0, "neginf_count": 0})
        return summary
    if not (torch.is_floating_point(detached) or torch.is_complex(detached)):
        return summary | {"finite": True}

    finite_mask = torch.isfinite(detached)
    nan_mask = torch.isnan(detached)
    posinf_mask = torch.isposinf(detached)
    neginf_mask = torch.isneginf(detached)
    finite_count = int(finite_mask.sum().item())
    summary.update(
        {
            "finite": finite_count == detached.numel(),
            "finite_count": finite_count,
            "nan_count": int(nan_mask.sum().item()),
            "posinf_count": int(posinf_mask.sum().item()),
            "neginf_count": int(neginf_mask.sum().item()),
        }
    )
    if finite_count:
        finite_values = detached[finite_mask]
        summary["finite_min"] = float(finite_values.min().item())
        summary["finite_max"] = float(finite_values.max().item())
        summary["finite_mean"] = float(finite_values.float().mean().item())
    bad = (~finite_mask).nonzero(as_tuple=False)
    if bad.numel():
        summary["bad_indices"] = bad[:16].detach().cpu().tolist()
    return summary


def _scan_jsonish(value: Any, path: str, issues: list[dict[str, Any]], depth: int = 0) -> None:
    if depth > 10 or len(issues) >= 64:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            issues.append({"path": path, "value": repr(value)})
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _scan_jsonish(item, f"{path}.{key}", issues, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value[:4096]):
            _scan_jsonish(item, f"{path}[{idx}]", issues, depth + 1)


def check_finite(name: str, value: Any, **context: Any) -> bool:
    """Return True when non-finite values are found and write a trace record."""
    if not enabled():
        return False

    issues: list[dict[str, Any]] = []
    summary: Any = None
    try:
        import torch

        if isinstance(value, torch.Tensor):
            summary = tensor_summary(value)
            if not summary.get("finite", True):
                issues.append({"path": name, "tensor": summary})
        else:
            _scan_jsonish(value, name, issues)
    except Exception as exc:
        issues.append({"path": name, "trace_error": repr(exc)})

    if not issues:
        return False

    write_event(
        "nonfinite",
        name=name,
        issues=issues,
        summary=summary,
        context=context,
        stack="".join(traceback.format_stack(limit=24)),
    )
    if fail_fast_enabled():
        raise FloatingPointError(f"Non-finite value detected in {name}")
    return True


def trace_state_dict(prefix: str, state_dict: dict[str, Any], **context: Any) -> None:
    if not weight_trace_enabled():
        return
    write_event("weight_state_dict", prefix=prefix, num_tensors=len(state_dict), keys=list(state_dict)[:64], context=context)
    for name, tensor in state_dict.items():
        check_finite(f"{prefix}.{name}", tensor, **context)
