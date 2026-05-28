from __future__ import annotations

import json
import logging
import math
import os
import socket
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_INSTALL_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()
_OUTPUT_PATCHED = False
_SCHEDULER_PATCHED = False
_CUDAGRAPH_DISPATCH_PATCHED = False
_MODEL_RUNNER_PATCHED = False
_COUNTS: dict[tuple[int, str], int] = defaultdict(int)
_MODEL_RUNNER_LOCAL = threading.local()


def install_vllm_nan_trace() -> None:
    """Install env-gated traces for hard-to-replay vLLM non-finite bugs."""
    if not _trace_dir():
        return

    with _INSTALL_LOCK:
        _patch_output_processor()
        _patch_scheduler()
        _patch_cudagraph_dispatcher()
        _patch_model_runner()


def _trace_dir() -> Path | None:
    value = os.environ.get("PRIME_RL_VLLM_NAN_TRACE_DIR", "").strip()
    return Path(value) if value else None


def _truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _trace_limit(kind: str, default: int) -> int:
    env_name = "PRIME_RL_VLLM_NAN_TRACE_" + kind.upper() + "_LIMIT"
    return _env_int(env_name, default)


def _take_record(kind: str, default_limit: int) -> bool:
    limit = _trace_limit(kind, default_limit)
    if limit < 0:
        return True
    key = (os.getpid(), kind)
    with _WRITE_LOCK:
        count = _COUNTS[key]
        if count >= limit:
            return False
        _COUNTS[key] = count + 1
    return True


def _write_jsonl(kind: str, payload: dict[str, Any], *, default_limit: int = -1) -> None:
    trace_dir = _trace_dir()
    if trace_dir is None:
        return
    if not _take_record(kind, default_limit):
        return

    record = {
        "schema": f"prime_rl.vllm_nan_trace.{kind}.v1",
        "created_unix": time.time(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        **payload,
    }
    path = trace_dir / f"{kind}.{os.getpid()}.jsonl"
    line = json.dumps(_json_safe(record), sort_keys=True, allow_nan=False)
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 12:
        return repr(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return repr(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "name") and value.__class__.__module__ == "enum":
        return value.name
    if isinstance(value, dict):
        return {str(k): _json_safe(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, depth=depth + 1) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"), depth=depth + 1)
        except Exception:
            return repr(value)
    return repr(value)


def _enum_summary(value: Any) -> Any:
    if value is None:
        return None
    return getattr(value, "name", repr(value))


def _object_attrs(value: Any, attrs: tuple[str, ...]) -> dict[str, Any] | None:
    if value is None:
        return None
    return {attr: getattr(value, attr, None) for attr in attrs}


def _slice_list(values: Any, limit: int) -> list[Any]:
    if values is None:
        return []
    try:
        result = list(values)
    except Exception:
        return [repr(values)]
    if limit >= 0:
        return result[:limit]
    return result


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def _tensor_summary(tensor: Any, *, name: str, include_rows: bool = False) -> dict[str, Any]:
    if tensor is None:
        return {"name": name, "present": False}
    try:
        import torch
    except Exception:
        return {"name": name, "present": True, "repr": repr(tensor)}
    if not isinstance(tensor, torch.Tensor):
        return {"name": name, "present": True, "type": type(tensor).__name__, "repr": repr(tensor)}

    summary: dict[str, Any] = {
        "name": name,
        "present": True,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0:
        summary.update({"finite": True, "nan_count": 0, "posinf_count": 0, "neginf_count": 0, "nonfinite_count": 0})
        return summary
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return summary

    try:
        finite = torch.isfinite(tensor)
        nonfinite = ~finite
        nonfinite_count = int(nonfinite.sum().item())
        nan_count = int(torch.isnan(tensor).sum().item())
        posinf_count = int(torch.isposinf(tensor).sum().item())
        neginf_count = int(torch.isneginf(tensor).sum().item())
        summary.update(
            {
                "finite": nonfinite_count == 0,
                "nan_count": nan_count,
                "posinf_count": posinf_count,
                "neginf_count": neginf_count,
                "nonfinite_count": nonfinite_count,
            }
        )
        if include_rows and tensor.ndim >= 2 and nonfinite_count:
            row_mask = nonfinite.reshape(tensor.shape[0], -1).any(dim=1)
            row_indices = row_mask.nonzero(as_tuple=False).flatten()[: _env_int("PRIME_RL_VLLM_NAN_TRACE_BAD_ROW_LIMIT", 16)]
            rows = []
            for row_index_tensor in row_indices:
                row_index = int(row_index_tensor.item())
                row = tensor[row_index].reshape(-1)
                row_nonfinite = ~torch.isfinite(row)
                rows.append(
                    {
                        "row": row_index,
                        "nonfinite_count": int(row_nonfinite.sum().item()),
                        "nan_count": int(torch.isnan(row).sum().item()),
                        "posinf_count": int(torch.isposinf(row).sum().item()),
                        "neginf_count": int(torch.isneginf(row).sum().item()),
                    }
                )
            summary["bad_rows"] = rows
    except Exception as exc:
        summary["finite_check_error"] = repr(exc)
    return summary


def _tensor_has_nonfinite(tensor: Any) -> bool:
    return bool(_tensor_summary(tensor, name="check").get("nonfinite_count", 0))


def _numpy_summary(value: Any, *, name: str, include_rows: bool = False) -> dict[str, Any] | None:
    try:
        import numpy as np
    except Exception:
        return None
    if not isinstance(value, np.ndarray):
        return None

    summary: dict[str, Any] = {
        "name": name,
        "present": True,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.size),
    }
    if value.size == 0:
        summary.update({"finite": True, "nan_count": 0, "posinf_count": 0, "neginf_count": 0, "nonfinite_count": 0})
        return summary
    if not (np.issubdtype(value.dtype, np.floating) or np.issubdtype(value.dtype, np.complexfloating)):
        return summary

    try:
        finite = np.isfinite(value)
        nonfinite = ~finite
        nan_count = int(np.isnan(value).sum())
        posinf_count = int(np.isposinf(value).sum())
        neginf_count = int(np.isneginf(value).sum())
        nonfinite_count = int(nonfinite.sum())
        summary.update(
            {
                "finite": nonfinite_count == 0,
                "nan_count": nan_count,
                "posinf_count": posinf_count,
                "neginf_count": neginf_count,
                "nonfinite_count": nonfinite_count,
            }
        )
        if include_rows and value.ndim >= 2 and nonfinite_count:
            bad_rows = np.flatnonzero(nonfinite.reshape(value.shape[0], -1).any(axis=1))
            rows = []
            for row_index in bad_rows[: _env_int("PRIME_RL_VLLM_NAN_TRACE_BAD_ROW_LIMIT", 16)]:
                row = value[int(row_index)].reshape(-1)
                rows.append(
                    {
                        "row": int(row_index),
                        "nonfinite_count": int((~np.isfinite(row)).sum()),
                        "nan_count": int(np.isnan(row).sum()),
                        "posinf_count": int(np.isposinf(row).sum()),
                        "neginf_count": int(np.isneginf(row).sum()),
                    }
                )
            summary["bad_rows"] = rows
    except Exception as exc:
        summary["finite_check_error"] = repr(exc)
    return summary


def _payload_summary(value: Any, *, name: str, include_rows: bool = False, depth: int = 0) -> dict[str, Any]:
    if value is None:
        return {"name": name, "present": False}
    if depth > 6:
        return {"name": name, "present": True, "type": type(value).__name__, "repr": repr(value)}

    try:
        import torch

        if isinstance(value, torch.Tensor):
            return _tensor_summary(value, name=name, include_rows=include_rows)
    except Exception:
        pass

    numpy_summary = _numpy_summary(value, name=name, include_rows=include_rows)
    if numpy_summary is not None:
        return numpy_summary

    if isinstance(value, (float, int)) and not isinstance(value, bool):
        is_finite = math.isfinite(value)
        return {
            "name": name,
            "present": True,
            "type": type(value).__name__,
            "finite": is_finite,
            "nan_count": int(isinstance(value, float) and math.isnan(value)),
            "posinf_count": int(value == math.inf),
            "neginf_count": int(value == -math.inf),
            "nonfinite_count": int(not is_finite),
            "value": value if is_finite else repr(value),
        }

    asdict_fn = getattr(value, "_asdict", None)
    if callable(asdict_fn):
        try:
            value = asdict_fn()
        except Exception:
            pass

    if isinstance(value, dict):
        fields = {
            str(key): _payload_summary(field_value, name=str(key), include_rows=include_rows, depth=depth + 1)
            for key, field_value in list(value.items())[: _env_int("PRIME_RL_VLLM_NAN_TRACE_PAYLOAD_FIELD_LIMIT", 32)]
        }
        return {
            "name": name,
            "present": True,
            "type": "dict",
            "len": len(value),
            "fields": fields,
            "nonfinite_count": _summary_nonfinite_count(fields),
        }

    if isinstance(value, (list, tuple)):
        sequence_summary = _numeric_sequence_summary(value, name=name)
        if sequence_summary["scanned_numeric_count"]:
            return sequence_summary
        limit = _env_int("PRIME_RL_VLLM_NAN_TRACE_PAYLOAD_ITEM_LIMIT", 8)
        items = [
            _payload_summary(item, name=f"{name}[{index}]", include_rows=include_rows, depth=depth + 1)
            for index, item in enumerate(list(value)[:limit])
        ]
        return {
            "name": name,
            "present": True,
            "type": type(value).__name__,
            "len": len(value),
            "items": items,
            "nonfinite_count": _summary_nonfinite_count(items),
        }

    return {"name": name, "present": True, "type": type(value).__name__, "repr": repr(value)}


def _numeric_sequence_summary(values: Any, *, name: str) -> dict[str, Any]:
    limit = _env_int("PRIME_RL_VLLM_NAN_TRACE_SEQUENCE_SCAN_LIMIT", 200000)
    stats: dict[str, Any] = {
        "name": name,
        "present": True,
        "type": type(values).__name__,
        "len": _safe_len(values),
        "scanned_numeric_count": 0,
        "finite": True,
        "nan_count": 0,
        "posinf_count": 0,
        "neginf_count": 0,
        "nonfinite_count": 0,
        "truncated": False,
        "bad_values": [],
    }

    def scan(value: Any, path: str, depth: int) -> None:
        if stats["scanned_numeric_count"] >= limit:
            stats["truncated"] = True
            return
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            stats["scanned_numeric_count"] += 1
            if math.isfinite(value):
                return
            stats["finite"] = False
            stats["nonfinite_count"] += 1
            stats["nan_count"] += int(isinstance(value, float) and math.isnan(value))
            stats["posinf_count"] += int(value == math.inf)
            stats["neginf_count"] += int(value == -math.inf)
            if len(stats["bad_values"]) < _env_int("PRIME_RL_VLLM_NAN_TRACE_BAD_VALUE_LIMIT", 16):
                stats["bad_values"].append({"path": path, "value": repr(value)})
            return
        if depth > 8:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                scan(item, f"{path}.{key}", depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                scan(item, f"{path}[{index}]", depth + 1)

    scan(values, name, 0)
    return stats


def _summary_nonfinite_count(value: Any) -> int:
    if isinstance(value, dict):
        count = value.get("nonfinite_count")
        if isinstance(count, int):
            return count
        return sum(_summary_nonfinite_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_summary_nonfinite_count(item) for item in value)
    return 0


def _sampled_token_ids(sampler_output: Any) -> list[list[int]]:
    token_tensor = getattr(sampler_output, "sampled_token_ids", None)
    if token_tensor is None:
        return []
    try:
        return token_tensor.detach().cpu().tolist()
    except Exception:
        return []


def _contains_token(sampled_token_ids: list[list[int]], token_id: int | None) -> bool:
    if token_id is None:
        return False
    return any(token_id in row for row in sampled_token_ids)


def _trace_token_id() -> int | None:
    value = os.environ.get("PRIME_RL_VLLM_NAN_TRACE_TOKEN_ID", "0").strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return 0


def _scheduler_output_summary(scheduler_output: Any) -> dict[str, Any]:
    num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
    scheduled_new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None) or []
    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
    return {
        "total_num_scheduled_tokens": getattr(scheduler_output, "total_num_scheduled_tokens", None),
        "num_scheduled_tokens_len": len(num_scheduled),
        "num_scheduled_tokens_sample": dict(list(num_scheduled.items())[:32]),
        "scheduled_new_reqs": [_request_like_summary(req) for req in scheduled_new_reqs[:32]],
        "scheduled_cached_reqs": _cached_request_data_summary(cached_reqs),
        "finished_req_ids": _slice_list(getattr(scheduler_output, "finished_req_ids", None), 32),
        "preempted_req_ids": _slice_list(getattr(scheduler_output, "preempted_req_ids", None), 32),
        "num_common_prefix_blocks": _slice_list(getattr(scheduler_output, "num_common_prefix_blocks", None), 32),
        "has_structured_output_requests": getattr(scheduler_output, "has_structured_output_requests", None),
        "pending_structured_output_tokens": getattr(scheduler_output, "pending_structured_output_tokens", None),
    }


def _request_like_summary(req: Any) -> dict[str, Any]:
    prompt_token_ids = getattr(req, "prompt_token_ids", None)
    return {
        "req_id": getattr(req, "req_id", getattr(req, "request_id", None)),
        "prompt_len": _safe_len(prompt_token_ids),
        "max_tokens": getattr(getattr(req, "sampling_params", None), "max_tokens", None),
        "lora_request": repr(getattr(req, "lora_request", None)),
    }


def _cached_request_data_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    req_ids = getattr(value, "req_ids", None)
    num_computed = getattr(value, "num_computed_tokens", None)
    return {
        "req_ids_len": _safe_len(req_ids),
        "req_ids_sample": _slice_list(req_ids, 32),
        "num_computed_tokens_sample": _slice_list(num_computed, 32),
    }


def _input_batch_summary(model_runner: Any, *, include_tokens: bool = False) -> dict[str, Any]:
    input_batch = getattr(model_runner, "input_batch", None)
    if input_batch is None:
        return {}
    req_ids = list(getattr(input_batch, "req_ids", []) or [])
    num_reqs = int(getattr(input_batch, "num_reqs", len(req_ids)) or 0)
    limit = _env_int("PRIME_RL_VLLM_NAN_TRACE_ACTIVE_REQ_LIMIT", 32)
    rows = []
    for index, req_id in enumerate(req_ids[:limit]):
        row: dict[str, Any] = {
            "index": index,
            "req_id": req_id,
            "num_prompt_tokens": _array_value(getattr(input_batch, "num_prompt_tokens", None), index),
            "num_computed_tokens": _array_value(getattr(input_batch, "num_computed_tokens_cpu", None), index),
            "num_tokens_no_spec": _array_value(getattr(input_batch, "num_tokens_no_spec", None), index),
            "lora_mapping": _array_value(getattr(input_batch, "request_lora_mapping", None), index),
        }
        if include_tokens:
            row["token_tail"] = _input_batch_token_tail(input_batch, index)
        rows.append(row)
    return {"num_reqs": num_reqs, "req_ids_len": len(req_ids), "rows": rows}


def _array_value(array: Any, index: int) -> Any:
    try:
        value = array[index]
    except Exception:
        return None
    try:
        return int(value)
    except Exception:
        return _json_safe(value)


def _input_batch_token_tail(input_batch: Any, index: int) -> list[int]:
    limit = _env_int("PRIME_RL_VLLM_NAN_TRACE_ACTIVE_OUTPUT_TOKENS", 512)
    try:
        end = int(input_batch.num_tokens_no_spec[index])
        start = max(0, end - limit)
        return [int(v) for v in input_batch.token_ids_cpu[index, start:end].tolist()]
    except Exception:
        return []


def _batch_execution_summary(result: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    cudagraph_mode = batch_desc = should_ubatch = num_tokens_across_dp = cudagraph_stats = None
    try:
        cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, cudagraph_stats = result
    except Exception:
        pass
    return {
        "inputs": {
            "num_tokens": kwargs.get("num_tokens", args[0] if len(args) > 0 else None),
            "num_reqs": kwargs.get("num_reqs", args[1] if len(args) > 1 else None),
            "max_num_scheduled_tokens": kwargs.get("max_num_scheduled_tokens"),
            "use_cascade_attn": kwargs.get("use_cascade_attn"),
        },
        "cudagraph_mode": _enum_summary(cudagraph_mode),
        "batch_descriptor": _object_attrs(
            batch_desc,
            ("num_tokens", "num_reqs", "uniform_decode", "uniform", "has_lora", "num_active_loras"),
        ),
        "should_ubatch": should_ubatch,
        "num_tokens_across_dp": num_tokens_across_dp,
        "cudagraph_stats": repr(cudagraph_stats),
    }


def _model_runner_context(model_runner: Any, execute_index: int | None = None) -> dict[str, Any]:
    return {
        "execute_index": execute_index,
        "rank": getattr(model_runner, "rank", None),
        "local_rank": getattr(model_runner, "local_rank", None),
        "dp_rank": getattr(getattr(model_runner, "parallel_config", None), "data_parallel_rank", None),
        "tp_rank": getattr(getattr(model_runner, "parallel_config", None), "tensor_parallel_rank", None),
        "input_batch": _input_batch_summary(model_runner, include_tokens=_truthy("PRIME_RL_VLLM_NAN_TRACE_ACTIVE_STATES", True)),
        "batch_execution": getattr(model_runner, "_prime_rl_nan_trace_batch_execution", None),
    }


def _next_execute_index(model_runner: Any) -> int:
    value = int(getattr(model_runner, "_prime_rl_nan_trace_execute_index", 0) or 0) + 1
    setattr(model_runner, "_prime_rl_nan_trace_execute_index", value)
    return value


def _patch_output_processor() -> None:
    global _OUTPUT_PATCHED
    if _OUTPUT_PATCHED:
        return
    try:
        from vllm.v1.engine.output_processor import OutputProcessor
    except Exception:
        logger.exception("Failed to import vLLM OutputProcessor for NaN trace")
        return

    original = OutputProcessor.process_outputs

    def _patched_process_outputs(self, engine_core_outputs, engine_core_timestamp=None, iteration_stats=None):
        try:
            _trace_engine_core_outputs(self, engine_core_outputs)
        except Exception:
            logger.exception("Failed to trace vLLM EngineCoreOutput")
        return original(self, engine_core_outputs, engine_core_timestamp, iteration_stats)

    _patched_process_outputs._prime_rl_vllm_nan_trace = True
    OutputProcessor.process_outputs = _patched_process_outputs
    _OUTPUT_PATCHED = True


def _trace_engine_core_outputs(output_processor: Any, engine_core_outputs: list[Any]) -> None:
    token_id = _trace_token_id()
    triggered = []
    compact_enabled = _truthy("PRIME_RL_VLLM_NAN_TRACE_OUTPUT_COMPACT", False)
    for output in engine_core_outputs:
        new_token_ids = list(getattr(output, "new_token_ids", []) or [])
        num_nans = int(getattr(output, "num_nans_in_logits", 0) or 0)
        new_logprobs = _payload_summary(getattr(output, "new_logprobs", None), name="new_logprobs", include_rows=True)
        prompt_logprobs = _payload_summary(
            getattr(output, "new_prompt_logprobs_tensors", None),
            name="new_prompt_logprobs_tensors",
            include_rows=True,
        )
        logprob_nonfinite_count = _summary_nonfinite_count(new_logprobs) + _summary_nonfinite_count(prompt_logprobs)
        saw_token = token_id is not None and token_id in new_token_ids
        if num_nans or logprob_nonfinite_count or saw_token or compact_enabled:
            triggered.append(
                {
                    "request_id": getattr(output, "request_id", None),
                    "new_token_ids": new_token_ids[:64],
                    "num_new_token_ids": len(new_token_ids),
                    "finish_reason": _enum_summary(getattr(output, "finish_reason", None)),
                    "stop_reason": getattr(output, "stop_reason", None),
                    "num_nans_in_logits": num_nans,
                    "logprob_nonfinite_count": logprob_nonfinite_count,
                    "new_logprobs": new_logprobs,
                    "new_prompt_logprobs_tensors": prompt_logprobs,
                    "saw_trace_token": saw_token,
                }
            )
    if not triggered:
        return
    _write_jsonl(
        "engine_core_output",
        {
            "outputs": triggered,
            "active_request_states": _request_states_summary(output_processor),
        },
        default_limit=2048,
    )


def _request_states_summary(output_processor: Any) -> list[dict[str, Any]]:
    if not _truthy("PRIME_RL_VLLM_NAN_TRACE_ACTIVE_STATES", True):
        return []
    states = getattr(output_processor, "request_states", None) or {}
    limit = _env_int("PRIME_RL_VLLM_NAN_TRACE_ACTIVE_REQ_LIMIT", 32)
    rows = []
    for req_id, state in list(states.items())[:limit]:
        detokenizer = getattr(state, "detokenizer", None)
        output_token_ids = getattr(detokenizer, "output_token_ids", None)
        rows.append(
            {
                "request_id": req_id,
                "is_prefilling": getattr(state, "is_prefilling", None),
                "num_cached_tokens": getattr(state, "num_cached_tokens", None),
                "output_token_count": _safe_len(output_token_ids),
                "output_token_tail": _tail_list(output_token_ids, _env_int("PRIME_RL_VLLM_NAN_TRACE_ACTIVE_OUTPUT_TOKENS", 512)),
                "sampling_params": _sampling_summary(getattr(state, "sampling_params", None)),
            }
        )
    return rows


def _tail_list(values: Any, limit: int) -> list[Any]:
    if values is None:
        return []
    try:
        result = list(values[-limit:])
    except Exception:
        result = _slice_list(values, limit)
    return result


def _sampling_summary(params: Any) -> dict[str, Any] | None:
    if params is None:
        return None
    return {
        "max_tokens": getattr(params, "max_tokens", None),
        "temperature": getattr(params, "temperature", None),
        "top_p": getattr(params, "top_p", None),
        "top_k": getattr(params, "top_k", None),
        "logprobs": getattr(params, "logprobs", None),
        "skip_special_tokens": getattr(params, "skip_special_tokens", None),
    }


def _patch_scheduler() -> None:
    global _SCHEDULER_PATCHED
    if _SCHEDULER_PATCHED:
        return
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception:
        logger.exception("Failed to import vLLM Scheduler for NaN trace")
        return

    original = Scheduler.schedule

    def _patched_schedule(self):
        output = original(self)
        if _truthy("PRIME_RL_VLLM_NAN_TRACE_SCHEDULER", True):
            try:
                compact = _truthy("PRIME_RL_VLLM_NAN_TRACE_SCHEDULER_COMPACT", True)
                if compact:
                    _write_jsonl("scheduler", _scheduler_output_summary(output), default_limit=20000)
            except Exception:
                logger.exception("Failed to trace vLLM SchedulerOutput")
        return output

    _patched_schedule._prime_rl_vllm_nan_trace = True
    Scheduler.schedule = _patched_schedule
    _SCHEDULER_PATCHED = True


def _patch_cudagraph_dispatcher() -> None:
    global _CUDAGRAPH_DISPATCH_PATCHED
    if _CUDAGRAPH_DISPATCH_PATCHED:
        return
    try:
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    except Exception:
        logger.exception("Failed to import vLLM CudagraphDispatcher for NaN trace")
        return

    original = CudagraphDispatcher.dispatch

    def _patched_dispatch(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if _truthy("PRIME_RL_VLLM_NAN_TRACE_CUDAGRAPH_DISPATCH", True):
            try:
                mode, batch_desc = result
                _write_jsonl(
                    "cudagraph_dispatch",
                    {
                        "requested_num_tokens": kwargs.get("num_tokens", args[0] if args else None),
                        "uniform_decode": kwargs.get("uniform_decode", args[1] if len(args) > 1 else None),
                        "has_lora": kwargs.get("has_lora", args[2] if len(args) > 2 else None),
                        "num_active_loras": kwargs.get("num_active_loras", args[3] if len(args) > 3 else None),
                        "mode": _enum_summary(mode),
                        "batch_descriptor": _object_attrs(
                            batch_desc,
                            ("num_tokens", "num_reqs", "uniform_decode", "uniform", "has_lora", "num_active_loras"),
                        ),
                    },
                    default_limit=20000,
                )
            except Exception:
                logger.exception("Failed to trace vLLM cudagraph dispatch")
        return result

    _patched_dispatch._prime_rl_vllm_nan_trace = True
    CudagraphDispatcher.dispatch = _patched_dispatch
    _CUDAGRAPH_DISPATCH_PATCHED = True


def _patch_model_runner() -> None:
    global _MODEL_RUNNER_PATCHED
    if _MODEL_RUNNER_PATCHED:
        return
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:
        logger.exception("Failed to import vLLM GPUModelRunner for NaN trace")
        return

    original_execute_model = GPUModelRunner.execute_model
    original_sample = GPUModelRunner._sample
    original_determine_batch = getattr(GPUModelRunner, "_determine_batch_execution_and_padding", None)

    if original_determine_batch is not None:

        def _patched_determine_batch_execution_and_padding(self, *args, **kwargs):
            result = original_determine_batch(self, *args, **kwargs)
            try:
                self._prime_rl_nan_trace_batch_execution = _batch_execution_summary(result, args, kwargs)
                _MODEL_RUNNER_LOCAL.context = _model_runner_context(self)
            except Exception:
                logger.exception("Failed to record vLLM batch execution summary")
            return result

        _patched_determine_batch_execution_and_padding._prime_rl_vllm_nan_trace = True
        GPUModelRunner._determine_batch_execution_and_padding = _patched_determine_batch_execution_and_padding

    def _patched_execute_model(self, *args, **kwargs):
        execute_index = _next_execute_index(self)
        previous_context = getattr(_MODEL_RUNNER_LOCAL, "context", None)
        _MODEL_RUNNER_LOCAL.context = _model_runner_context(self, execute_index=execute_index)
        try:
            result = original_execute_model(self, *args, **kwargs)
            if _truthy("PRIME_RL_VLLM_NAN_TRACE_MODEL_RUNNER", True):
                try:
                    state = getattr(self, "execute_model_state", None)
                    if state is not None:
                        _trace_model_runner_state(self, state, phase="execute_model")
                except Exception:
                    logger.exception("Failed to trace vLLM model runner execute_model state")
            return result
        finally:
            _MODEL_RUNNER_LOCAL.context = previous_context

    def _patched_sample(self, logits, spec_decode_metadata):
        if not _truthy("PRIME_RL_VLLM_NAN_TRACE_MODEL_RUNNER", True):
            return original_sample(self, logits, spec_decode_metadata)
        before = _tensor_summary(logits, name="logits_before_sample", include_rows=True)
        sampler_output = original_sample(self, logits, spec_decode_metadata)
        try:
            sampled = _sampled_token_ids(sampler_output)
            logprobs = _payload_summary(getattr(sampler_output, "logprobs_tensors", None), name="sampler_logprobs", include_rows=True)
            saw_token = _contains_token(sampled, _trace_token_id())
            nonfinite = bool(before.get("nonfinite_count", 0))
            logprob_nonfinite_count = _summary_nonfinite_count(logprobs)
            batch_trace = _truthy("PRIME_RL_VLLM_NAN_TRACE_BATCH", False)
            if nonfinite or logprob_nonfinite_count or saw_token or batch_trace:
                after = _tensor_summary(logits, name="logits_after_sample", include_rows=True)
                _write_jsonl(
                    "model_runner_sample",
                    {
                        "trigger": {
                            "nonfinite_logits_before": nonfinite,
                            "nonfinite_logits_after": bool(after.get("nonfinite_count", 0)),
                            "logprob_nonfinite_count": logprob_nonfinite_count,
                            "saw_trace_token": saw_token,
                            "trace_token_id": _trace_token_id(),
                        },
                        "context": getattr(_MODEL_RUNNER_LOCAL, "context", None) or _model_runner_context(self),
                        "spec_decode": spec_decode_metadata is not None,
                        "sampled_token_ids": sampled[:64],
                        "logits_before": before,
                        "logits_after": after,
                        "logprobs": logprobs,
                    },
                    default_limit=2048,
                )
        except Exception:
            logger.exception("Failed to trace vLLM model runner sample")
        return sampler_output

    _patched_execute_model._prime_rl_vllm_nan_trace = True
    _patched_sample._prime_rl_vllm_nan_trace = True
    GPUModelRunner.execute_model = _patched_execute_model
    GPUModelRunner._sample = _patched_sample
    _MODEL_RUNNER_PATCHED = True


def _trace_model_runner_state(model_runner: Any, state: Any, *, phase: str) -> None:
    logits = getattr(state, "logits", None)
    sample_hidden_states = getattr(state, "sample_hidden_states", None)
    hidden_states = getattr(state, "hidden_states", None)
    logits_summary = _tensor_summary(logits, name="logits", include_rows=True)
    sample_hidden_summary = _tensor_summary(sample_hidden_states, name="sample_hidden_states", include_rows=True)
    hidden_summary = _tensor_summary(hidden_states, name="hidden_states")
    padded_hidden_only = (
        bool(hidden_summary.get("nonfinite_count", 0))
        and not bool(sample_hidden_summary.get("nonfinite_count", 0))
        and not bool(logits_summary.get("nonfinite_count", 0))
    )
    triggered = (
        bool(logits_summary.get("nonfinite_count", 0))
        or bool(sample_hidden_summary.get("nonfinite_count", 0))
        or (bool(hidden_summary.get("nonfinite_count", 0)) and _truthy("PRIME_RL_VLLM_NAN_TRACE_PADDED_HIDDEN", False))
        or _truthy("PRIME_RL_VLLM_NAN_TRACE_BATCH", False)
    )
    if not triggered:
        if padded_hidden_only:
            _write_jsonl(
                "padded_hidden_state",
                {
                    "phase": phase,
                    "context": _model_runner_context(model_runner),
                    "hidden_states": hidden_summary,
                    "sample_hidden_states": sample_hidden_summary,
                    "logits": logits_summary,
                },
                default_limit=64,
            )
        return
    _write_jsonl(
        "model_runner_state",
        {
            "phase": phase,
            "context": _model_runner_context(model_runner),
            "scheduler_output": _scheduler_output_summary(getattr(state, "scheduler_output", None)),
            "logits": logits_summary,
            "sample_hidden_states": sample_hidden_summary,
            "hidden_states": hidden_summary,
            "spec_decode": getattr(state, "spec_decode_metadata", None) is not None,
            "cudagraph_stats": repr(getattr(state, "cudagraph_stats", None)),
        },
        default_limit=2048,
    )
