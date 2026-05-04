from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_INSTALL_LOCK = threading.Lock()
_OUTPUT_PATCHED = False
_SCHEDULER_PATCHED = False
_MODEL_RUNNER_PATCHED = False
_WRITE_LOCK = threading.Lock()
_ACTIVE_STATE_DUMPED: set[str] = set()


def install_vllm_nan_trace() -> None:
    """Install env-gated vLLM tracing hooks for Nemotron NaN investigation."""
    if not _trace_dir():
        return

    with _INSTALL_LOCK:
        _patch_output_processor()
        _patch_scheduler()
        _patch_model_runner()


def _trace_dir() -> str | None:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_DIR")


def _trace_token_id() -> int | None:
    value = os.environ.get("PRIME_RL_VLLM_NAN_TRACE_TOKEN_ID", "0").strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return 0


def _scheduler_trace_enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_SCHEDULER", "1") != "0"


def _model_runner_trace_enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_MODEL_RUNNER", "1") != "0"


def _active_state_trace_enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_ACTIVE_STATES", "1") != "0"


def _active_output_token_limit() -> int:
    value = os.environ.get("PRIME_RL_VLLM_NAN_TRACE_ACTIVE_OUTPUT_TOKENS", "4096")
    try:
        return int(value)
    except ValueError:
        return 4096


def _patch_output_processor() -> None:
    global _OUTPUT_PATCHED
    if _OUTPUT_PATCHED:
        return

    try:
        from vllm.v1.engine.output_processor import OutputProcessor
    except Exception:
        logger.exception("Failed to import vLLM OutputProcessor for NaN tracing")
        return

    original_process_outputs = OutputProcessor.process_outputs

    def _patched_process_outputs(
        self,
        engine_core_outputs,
        engine_core_timestamp=None,
        iteration_stats=None,
    ):
        try:
            _trace_engine_core_outputs(self, engine_core_outputs, engine_core_timestamp)
        except Exception:
            logger.exception("Failed to trace vLLM EngineCoreOutput")
        return original_process_outputs(
            self,
            engine_core_outputs,
            engine_core_timestamp,
            iteration_stats,
        )

    OutputProcessor.process_outputs = _patched_process_outputs
    _OUTPUT_PATCHED = True


def _patch_scheduler() -> None:
    global _SCHEDULER_PATCHED
    if _SCHEDULER_PATCHED:
        return

    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception:
        logger.exception("Failed to import vLLM Scheduler for NaN tracing")
        return

    original_schedule = Scheduler.schedule

    def _patched_schedule(self):
        scheduler_output = original_schedule(self)
        if _scheduler_trace_enabled():
            try:
                _trace_scheduler_output(self, scheduler_output)
            except Exception:
                logger.exception("Failed to trace vLLM scheduler output")
        return scheduler_output

    Scheduler.schedule = _patched_schedule
    _SCHEDULER_PATCHED = True


def _patch_model_runner() -> None:
    global _MODEL_RUNNER_PATCHED
    if _MODEL_RUNNER_PATCHED:
        return

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:
        logger.exception("Failed to import vLLM GPUModelRunner for NaN tracing")
        return

    original_execute_model = GPUModelRunner.execute_model
    original_sample = GPUModelRunner._sample

    def _patched_execute_model(self, *args, **kwargs):
        result = original_execute_model(self, *args, **kwargs)
        if not _model_runner_trace_enabled():
            return result

        try:
            state = getattr(self, "execute_model_state", None)
            if state is not None:
                _trace_model_runner_logits_state(
                    req_ids=list(getattr(self.input_batch, "req_ids", []) or []),
                    logits=getattr(state, "logits", None),
                    sample_hidden_states=getattr(state, "sample_hidden_states", None),
                    spec_decode=getattr(state, "spec_decode_metadata", None) is not None,
                )
        except Exception:
            logger.exception("Failed to trace vLLM model runner logits state")
        return result

    def _patched_sample(self, logits, spec_decode_metadata):
        if not _model_runner_trace_enabled() or logits is None:
            return original_sample(self, logits, spec_decode_metadata)

        req_ids = list(getattr(self.input_batch, "req_ids", []) or [])
        before = _logits_row_counts(logits)
        sampler_output = original_sample(self, logits, spec_decode_metadata)
        try:
            sampled_token_ids = _sampled_token_ids(sampler_output)
            after = _logits_row_counts(logits)
            _trace_model_runner_sample(
                req_ids=req_ids,
                logits=logits,
                before=before,
                after=after,
                sampled_token_ids=sampled_token_ids,
                spec_decode=spec_decode_metadata is not None,
            )
        except Exception:
            logger.exception("Failed to trace vLLM model runner sample")
        return sampler_output

    GPUModelRunner.execute_model = _patched_execute_model
    GPUModelRunner._sample = _patched_sample
    _MODEL_RUNNER_PATCHED = True


def _trace_engine_core_outputs(output_processor: Any, engine_core_outputs: list[Any], timestamp: float | None) -> None:
    watched_token_id = _trace_token_id()
    for output in engine_core_outputs:
        req_state = output_processor.request_states.get(output.request_id)
        if req_state is None:
            continue

        output_len_before = 0
        if req_state.detokenizer is not None:
            output_len_before = req_state.detokenizer.num_output_tokens()

        logprob_summary = _summarize_logprobs(
            output.new_logprobs,
            watched_token_id=watched_token_id,
            output_len_before=output_len_before,
        )
        generated_watch_positions = [
            output_len_before + idx
            for idx, token_id in enumerate(output.new_token_ids or [])
            if watched_token_id is not None and token_id == watched_token_id
        ]
        has_nans_in_logits = getattr(output, "num_nans_in_logits", 0) > 0
        if not (has_nans_in_logits or logprob_summary["events"] or generated_watch_positions):
            continue

        event = {
            "schema": "prime_rl.vllm_output_nan_trace.v1",
            "created_unix": time.time(),
            "engine_core_timestamp": timestamp,
            "pid": os.getpid(),
            "internal_request_id": output.request_id,
            "external_request_id": req_state.external_req_id,
            "prompt_len": req_state.prompt_len,
            "max_tokens": req_state.max_tokens_param,
            "temperature": req_state.temperature,
            "top_p": req_state.top_p,
            "n": req_state.n,
            "num_cached_tokens": output.num_cached_tokens,
            "num_external_computed_tokens": getattr(output, "num_external_computed_tokens", None),
            "num_nans_in_logits": getattr(output, "num_nans_in_logits", None),
            "output_len_before": output_len_before,
            "output_token_ids_before": _token_ids_snapshot(
                _request_state_output_token_ids(req_state),
                limit=_active_output_token_limit(),
            ),
            "new_token_ids": list(output.new_token_ids or []),
            "generated_watch_token_positions": generated_watch_positions,
            "finish_reason": _json_safe(getattr(output, "finish_reason", None)),
            "stop_reason": _json_safe(getattr(output, "stop_reason", None)),
            "logprobs": logprob_summary,
        }

        _write_jsonl(
            "output_events",
            event,
        )
        if has_nans_in_logits and _active_state_trace_enabled():
            _trace_active_request_states_once(output_processor, event)


def _trace_active_request_states_once(output_processor: Any, trigger: dict[str, Any]) -> None:
    dump_key = str(trigger.get("internal_request_id") or trigger.get("external_request_id"))
    with _WRITE_LOCK:
        if dump_key in _ACTIVE_STATE_DUMPED:
            return
        _ACTIVE_STATE_DUMPED.add(dump_key)

    states = []
    for internal_request_id, req_state in sorted(output_processor.request_states.items()):
        states.append(
            {
                "internal_request_id": internal_request_id,
                "external_request_id": getattr(req_state, "external_req_id", None),
                "prompt_len": getattr(req_state, "prompt_len", None),
                "max_tokens": getattr(req_state, "max_tokens_param", None),
                "temperature": getattr(req_state, "temperature", None),
                "top_p": getattr(req_state, "top_p", None),
                "n": getattr(req_state, "n", None),
                "num_cached_tokens": getattr(req_state, "num_cached_tokens", None),
                "output_token_ids": _token_ids_snapshot(
                    _request_state_output_token_ids(req_state),
                    limit=_active_output_token_limit(),
                ),
            }
        )

    _write_jsonl(
        "active_request_states",
        {
            "schema": "prime_rl.vllm_active_request_states.v1",
            "created_unix": time.time(),
            "pid": os.getpid(),
            "trigger": trigger,
            "request_count": len(states),
            "requests": states,
        },
    )


def _trace_scheduler_output(scheduler: Any, scheduler_output: Any) -> None:
    running_by_id = {req.request_id: req for req in getattr(scheduler, "running", [])}
    cached = scheduler_output.scheduled_cached_reqs
    cached_reqs = [
        _scheduler_cached_request_summary(
            req_id=req_id,
            request=running_by_id.get(req_id),
            num_scheduled_tokens=scheduler_output.num_scheduled_tokens.get(req_id),
            num_computed_tokens=cached.num_computed_tokens[idx],
            num_output_tokens=cached.num_output_tokens[idx],
            new_block_ids=cached.new_block_ids[idx],
            resumed=req_id in cached.resumed_req_ids,
        )
        for idx, req_id in enumerate(cached.req_ids)
    ]
    new_reqs = [
        _scheduler_new_request_summary(
            new_req,
            request=running_by_id.get(new_req.req_id),
            num_scheduled_tokens=scheduler_output.num_scheduled_tokens.get(new_req.req_id),
        )
        for new_req in scheduler_output.scheduled_new_reqs
    ]

    _write_jsonl(
        "scheduler_steps",
        {
            "schema": "prime_rl.vllm_scheduler_nan_trace.v1",
            "created_unix": time.time(),
            "pid": os.getpid(),
            "running_count": len(getattr(scheduler, "running", [])),
            "waiting_count": len(getattr(scheduler, "waiting", [])),
            "skipped_waiting_count": len(getattr(scheduler, "skipped_waiting", [])),
            "num_waiting_for_streaming_input": getattr(scheduler, "num_waiting_for_streaming_input", None),
            "total_num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
            "num_common_prefix_blocks": scheduler_output.num_common_prefix_blocks,
            "new_block_ids_to_zero": _block_list_summary(scheduler_output.new_block_ids_to_zero),
            "finished_req_ids": sorted(scheduler_output.finished_req_ids),
            "preempted_req_ids": sorted(scheduler_output.preempted_req_ids or []),
            "scheduled_new_reqs": new_reqs,
            "scheduled_cached_reqs": cached_reqs,
            "num_scheduled_tokens": dict(sorted(scheduler_output.num_scheduled_tokens.items())),
        },
    )


def _trace_model_runner_sample(
    *,
    req_ids: list[str],
    logits: Any,
    before: dict[str, Any],
    after: dict[str, Any],
    sampled_token_ids: list[int | None],
    spec_decode: bool,
) -> None:
    watched_token_id = _trace_token_id()
    selected_rows: set[int] = set()

    for row_idx, count in enumerate(before["nan_counts"]):
        if count:
            selected_rows.add(row_idx)
    for row_idx, count in enumerate(after["nan_counts"]):
        if count:
            selected_rows.add(row_idx)
    if watched_token_id is not None:
        for row_idx, token_id in enumerate(sampled_token_ids):
            if token_id == watched_token_id:
                selected_rows.add(row_idx)

    if not selected_rows:
        return

    rows = []
    for row_idx in sorted(selected_rows):
        req_id = req_ids[row_idx] if row_idx < len(req_ids) else None
        rows.append(
            {
                "row_index": row_idx,
                "request_id": req_id,
                "external_request_id_guess": _external_request_id_from_internal(req_id) if req_id else None,
                "sampled_token_id": sampled_token_ids[row_idx] if row_idx < len(sampled_token_ids) else None,
                "before": _row_count_at(before, row_idx),
                "after": _row_count_at(after, row_idx),
            }
        )

    _write_jsonl(
        "model_runner_samples",
        {
            "schema": "prime_rl.vllm_model_runner_sample_trace.v1",
            "created_unix": time.time(),
            "pid": os.getpid(),
            "shape": list(logits.shape),
            "dtype": str(getattr(logits, "dtype", "")),
            "spec_decode": spec_decode,
            "watched_token_id": watched_token_id,
            "num_rows": len(req_ids),
            "rows": rows,
        },
    )


def _trace_model_runner_logits_state(
    *,
    req_ids: list[str],
    logits: Any,
    sample_hidden_states: Any,
    spec_decode: bool,
) -> None:
    if logits is None or sample_hidden_states is None:
        return

    hidden_counts = _tensor_row_counts(sample_hidden_states)
    logits_counts = _tensor_row_counts(logits)
    selected_rows: set[int] = set()
    for row_idx, count in enumerate(hidden_counts["nan_counts"]):
        if count:
            selected_rows.add(row_idx)
    for row_idx, count in enumerate(logits_counts["nan_counts"]):
        if count:
            selected_rows.add(row_idx)
    for row_idx, count in enumerate(hidden_counts["posinf_counts"]):
        if count:
            selected_rows.add(row_idx)
    for row_idx, count in enumerate(hidden_counts["neginf_counts"]):
        if count:
            selected_rows.add(row_idx)
    for row_idx, count in enumerate(logits_counts["posinf_counts"]):
        if count:
            selected_rows.add(row_idx)
    for row_idx, count in enumerate(logits_counts["neginf_counts"]):
        if count:
            selected_rows.add(row_idx)

    if not selected_rows:
        return

    rows = []
    for row_idx in sorted(selected_rows):
        req_id = req_ids[row_idx] if row_idx < len(req_ids) else None
        rows.append(
            {
                "row_index": row_idx,
                "request_id": req_id,
                "external_request_id_guess": _external_request_id_from_internal(req_id) if req_id else None,
                "hidden": _row_count_at(hidden_counts, row_idx),
                "logits": _row_count_at(logits_counts, row_idx),
            }
        )

    _write_jsonl(
        "model_runner_logits",
        {
            "schema": "prime_rl.vllm_model_runner_logits_trace.v1",
            "created_unix": time.time(),
            "pid": os.getpid(),
            "hidden_shape": list(sample_hidden_states.shape),
            "hidden_dtype": str(getattr(sample_hidden_states, "dtype", "")),
            "logits_shape": list(logits.shape),
            "logits_dtype": str(getattr(logits, "dtype", "")),
            "spec_decode": spec_decode,
            "num_rows": len(req_ids),
            "rows": rows,
        },
    )


def _scheduler_new_request_summary(
    new_req: Any,
    *,
    request: Any | None,
    num_scheduled_tokens: int | None,
) -> dict[str, Any]:
    sampling_params = getattr(new_req, "sampling_params", None)
    return {
        "request_id": new_req.req_id,
        "external_request_id_guess": _external_request_id_from_internal(new_req.req_id),
        "kind": "new",
        "prompt_len": len(new_req.prompt_token_ids or []),
        "max_tokens": getattr(sampling_params, "max_tokens", None),
        "temperature": getattr(sampling_params, "temperature", None),
        "top_p": getattr(sampling_params, "top_p", None),
        "num_scheduled_tokens": num_scheduled_tokens,
        "num_computed_tokens": new_req.num_computed_tokens,
        "num_output_tokens": getattr(request, "num_output_tokens", None),
        "output_token_ids_tail": _token_ids_tail(_request_output_token_ids(request)),
        "num_cached_tokens": getattr(request, "num_cached_tokens", None),
        "status": str(getattr(request, "status", "")) if request is not None else None,
        "block_ids": _block_ids_summary(new_req.block_ids),
    }


def _scheduler_cached_request_summary(
    *,
    req_id: str,
    request: Any | None,
    num_scheduled_tokens: int | None,
    num_computed_tokens: int,
    num_output_tokens: int,
    new_block_ids: Any,
    resumed: bool,
) -> dict[str, Any]:
    return {
        "request_id": req_id,
        "external_request_id_guess": _external_request_id_from_internal(req_id),
        "kind": "resumed" if resumed else "cached",
        "prompt_len": getattr(request, "num_prompt_tokens", None),
        "max_tokens": getattr(request, "max_tokens", None),
        "num_scheduled_tokens": num_scheduled_tokens,
        "num_computed_tokens": num_computed_tokens,
        "num_output_tokens": num_output_tokens,
        "output_token_ids_tail": _token_ids_tail(_request_output_token_ids(request)),
        "num_cached_tokens": getattr(request, "num_cached_tokens", None),
        "num_preemptions": getattr(request, "num_preemptions", None),
        "status": str(getattr(request, "status", "")) if request is not None else None,
        "new_block_ids": _block_ids_summary(new_block_ids),
    }


def _summarize_logprobs(
    logprobs_lists: Any | None,
    *,
    watched_token_id: int | None,
    output_len_before: int,
) -> dict[str, Any]:
    if logprobs_lists is None:
        return {"events": []}

    token_ids_lst, logprobs_lst, ranks_lst, _ = logprobs_lists
    events: list[dict[str, Any]] = []
    for step_index, (token_ids_raw, logprobs_raw, ranks_raw) in enumerate(zip(token_ids_lst, logprobs_lst, ranks_lst)):
        token_ids = _as_list(token_ids_raw)
        logprobs = _as_list(logprobs_raw)
        ranks = _as_list(ranks_raw)
        candidates = []
        sampled_token_id = token_ids[0] if token_ids else None
        for candidate_index, logprob in enumerate(logprobs):
            logprob_float = _as_float(logprob)
            token_id = token_ids[candidate_index] if candidate_index < len(token_ids) else None
            rank = ranks[candidate_index] if candidate_index < len(ranks) else None
            if not math.isfinite(logprob_float) or token_id == watched_token_id:
                candidates.append(
                    {
                        "candidate_index": candidate_index,
                        "token_id": token_id,
                        "rank": rank,
                        "logprob": _json_safe(logprob_float),
                    }
                )
        if candidates or sampled_token_id == watched_token_id:
            events.append(
                {
                    "step_index": step_index,
                    "output_token_offset": output_len_before + step_index,
                    "sampled_token_id": sampled_token_id,
                    "sampled_logprob": _json_safe(_as_float(logprobs[0])) if logprobs else None,
                    "sampled_rank": ranks[0] if ranks else None,
                    "candidates": candidates,
                }
            )
    return {"events": events}


def _block_ids_summary(block_ids: Any) -> Any:
    if block_ids is None:
        return None
    groups = list(block_ids)
    return [_block_list_summary(group) for group in groups]


def _logits_row_counts(logits: Any) -> dict[str, list[int]]:
    return _tensor_row_counts(logits)


def _tensor_row_counts(tensor: Any) -> dict[str, list[int]]:
    if len(tensor.shape) == 0:
        rows = tensor.reshape(1, 1)
    elif len(tensor.shape) == 1:
        rows = tensor.reshape(tensor.shape[0], 1)
    else:
        rows = tensor.reshape(tensor.shape[0], -1)

    finite = rows.isfinite()
    nan_counts = rows.isnan().sum(dim=-1).detach().cpu().tolist()
    posinf_counts = rows.isposinf().sum(dim=-1).detach().cpu().tolist()
    neginf_counts = rows.isneginf().sum(dim=-1).detach().cpu().tolist()
    finite_counts = finite.sum(dim=-1).detach().cpu().tolist()
    return {
        "nan_counts": [int(value) for value in nan_counts],
        "posinf_counts": [int(value) for value in posinf_counts],
        "neginf_counts": [int(value) for value in neginf_counts],
        "finite_counts": [int(value) for value in finite_counts],
    }


def _row_count_at(counts: dict[str, list[int]], row_idx: int) -> dict[str, int | None]:
    return {
        key.removesuffix("_counts"): values[row_idx] if row_idx < len(values) else None
        for key, values in counts.items()
    }


def _sampled_token_ids(sampler_output: Any) -> list[int | None]:
    sampled = getattr(sampler_output, "sampled_token_ids", None)
    if sampled is None:
        return []
    try:
        values = sampled.detach().cpu().view(-1).tolist()
    except Exception:
        return []
    return [int(value) if value is not None else None for value in values]


def _request_state_output_token_ids(req_state: Any) -> list[int]:
    detokenizer = getattr(req_state, "detokenizer", None)
    return _as_int_list(getattr(detokenizer, "output_token_ids", None))


def _request_output_token_ids(request: Any | None) -> list[int]:
    return _as_int_list(getattr(request, "output_token_ids", None))


def _token_ids_tail(token_ids: list[int], limit: int = 16) -> dict[str, Any]:
    return {
        "count": len(token_ids),
        "tail": token_ids[-limit:] if limit > 0 else [],
    }


def _token_ids_snapshot(token_ids: list[int], *, limit: int) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "count": len(token_ids),
        "head": token_ids[:16],
        "tail": token_ids[-64:],
    }
    if limit < 0 or len(token_ids) <= limit:
        snapshot["token_ids"] = token_ids
    else:
        snapshot["truncated"] = True
        snapshot["limit"] = limit
    return snapshot


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        return [int(item) for item in list(value)]
    except Exception:
        return []


def _block_list_summary(block_ids: Any) -> dict[str, Any] | None:
    if block_ids is None:
        return None
    values = list(block_ids)
    non_null = [value for value in values if value is not None]
    return {
        "count": len(values),
        "none_count": len(values) - len(non_null),
        "head": non_null[:8],
        "tail": non_null[-8:],
        "min": min(non_null) if non_null else None,
        "max": max(non_null) if non_null else None,
    }


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _write_jsonl(stem: str, record: dict[str, Any]) -> None:
    dump_dir = _trace_dir()
    if not dump_dir:
        return
    path = Path(dump_dir).expanduser() / f"{stem}.{os.getpid()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def _external_request_id_from_internal(request_id: str) -> str:
    # vLLM appends an internal child suffix for async generation; prime-rl's
    # external /generate request id is the `gen-<hex>` prefix.
    if not request_id.startswith("gen-"):
        return request_id
    parts = request_id.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return request_id


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)
