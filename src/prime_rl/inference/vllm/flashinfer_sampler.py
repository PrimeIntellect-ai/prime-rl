"""Opt-in finite-top-k sampled-logprob fast path for vLLM 0.22.0.

PrimeRL's rollout paths ask vLLM for processed sampled-token logprobs, but do
not consume top alternatives or token ranks. The renderer path forces
``SamplingParams.logprobs=1`` to make completion logprobs appear in its internal
response, but Prime still only reads the sampled token's scalar logprob. vLLM's
native processed path materializes full-vocabulary processed logprobs before
gathering that scalar. This patch handles the narrow finite-top-k case directly:

    processed logits -> top-K slice -> top-p in K-space -> sample -> scalar logprob

The current implementation uses FlashInfer for value/index top-k selection and
a small K-space tail for sampling/logprob. It is environment-gated and
default-off. Enable with ``PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB=1``.
The older ``PRIME_RL_ENABLE_FLASHINFER_SAMPLED_LOGPROB=1`` name remains a
compatibility alias for existing experiment launchers.
"""

from __future__ import annotations

import datetime
import importlib
import json
import os
import socket
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from prime_rl.inference.vllm import sampler_perf

_ENABLE_ENV = "PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB"
_ENABLE_LEGACY_ENV = "PRIME_RL_ENABLE_FLASHINFER_SAMPLED_LOGPROB"
_LOG_FALLBACK_ENV = "PRIME_RL_LOG_FINITE_TOPK_SAMPLED_LOGPROB_FALLBACK"
_LOG_FALLBACK_LEGACY_ENV = "PRIME_RL_LOG_FLASHINFER_SAMPLER_FALLBACK"
_HIT_LOG_LIMIT_ENV = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_HIT_LOG_LIMIT"
_HIT_LOG_LIMIT_LEGACY_ENV = "PRIME_RL_FLASHINFER_SAMPLER_HIT_LOG_LIMIT"
_STATS_INTERVAL_ENV = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_STATS_INTERVAL"
_STATS_INTERVAL_LEGACY_ENV = "PRIME_RL_FLASHINFER_SAMPLER_STATS_INTERVAL"
_STATS_LOG_ENV = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_STATS_LOG"
_STATS_LOG_DIR_ENV = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_STATS_LOG_DIR"
_TOKEN_STATS_INTERVAL_ENV = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_TOKEN_STATS_INTERVAL"
_TOKEN_STATS_INTERVAL_LEGACY_ENV = "PRIME_RL_FLASHINFER_SAMPLER_TOKEN_STATS_INTERVAL"
_DENSE_PRESENCE_ENV = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_DENSE_PRESENCE"
_DENSE_PRESENCE_LEGACY_ENV = "PRIME_RL_FLASHINFER_SAMPLER_DENSE_PRESENCE"
_TAIL_ENV = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_TAIL"
_TAIL_LEGACY_ENV = "PRIME_RL_FLASHINFER_SAMPLER_TAIL"
_PRECOMPILE_TAIL_ENV = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TAIL"
_PRECOMPILE_TAIL_TOP_K_ENV = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TOP_K"
_PRECOMPILE_TAIL_TOP_P_ENV = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TOP_P"
_PRECOMPILE_TAIL_VOCAB_ENV = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_VOCAB"
_PRECOMPILE_TAIL_BATCH_ENV = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCH"
_PRECOMPILE_TAIL_BATCHES_ENV = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCHES"
_PATCH_MARKER = "_prime_rl_finite_topk_sampled_logprob"
_INIT_PATCH_MARKER = "_prime_rl_finite_topk_sampled_logprob_init"
_SUPPORTED_FLASHINFER = "0.6.11.post2"
_SAMPLING_EPS = 1e-5
_NO_INPUT_BATCH = object()
_INELIGIBLE_INPUT_BATCH = object()


def _env_value(name: str, legacy_name: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is not None:
        return value
    if legacy_name is not None:
        return os.environ.get(legacy_name)
    return None


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def _env_enabled() -> bool:
    return _truthy(_env_value(_ENABLE_ENV, _ENABLE_LEGACY_ENV))


def _fallback_logging_enabled() -> bool:
    return _truthy(_env_value(_LOG_FALLBACK_ENV, _LOG_FALLBACK_LEGACY_ENV))


def _hit_log_limit() -> int:
    raw = _env_value(_HIT_LOG_LIMIT_ENV, _HIT_LOG_LIMIT_LEGACY_ENV)
    if raw is None:
        return 1
    return max(0, int(raw))


def _stats_interval() -> int:
    raw = _env_value(_STATS_INTERVAL_ENV, _STATS_INTERVAL_LEGACY_ENV)
    if raw is None:
        return 0
    return max(0, int(raw))


def _token_stats_interval() -> int:
    raw = _env_value(_TOKEN_STATS_INTERVAL_ENV, _TOKEN_STATS_INTERVAL_LEGACY_ENV)
    if raw is None:
        return 0
    return max(0, int(raw))


def _dense_presence_enabled() -> bool:
    return _truthy(_env_value(_DENSE_PRESENCE_ENV, _DENSE_PRESENCE_LEGACY_ENV))


def _use_triton_tail() -> bool:
    raw = _env_value(_TAIL_ENV, _TAIL_LEGACY_ENV)
    return (raw or "triton").lower() != "torch"


def _precompile_tail_enabled() -> bool:
    return _truthy(os.environ.get(_PRECOMPILE_TAIL_ENV))


def _precompile_tail_top_k() -> int:
    return int(os.environ.get(_PRECOMPILE_TAIL_TOP_K_ENV, "20"))


def _precompile_tail_top_p() -> float:
    return float(os.environ.get(_PRECOMPILE_TAIL_TOP_P_ENV, "0.95"))


def _precompile_tail_top_p_values() -> list[float]:
    return [_precompile_tail_top_p()]


def _precompile_tail_vocab_size() -> int:
    return int(os.environ.get(_PRECOMPILE_TAIL_VOCAB_ENV, "248320"))


def _precompile_tail_batch_size() -> int:
    return int(os.environ.get(_PRECOMPILE_TAIL_BATCH_ENV, "1"))


def _precompile_tail_batch_sizes() -> list[int]:
    raw = os.environ.get(_PRECOMPILE_TAIL_BATCHES_ENV)
    if raw is None:
        return [_precompile_tail_batch_size()]

    sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return list(dict.fromkeys(sizes))


def _load_flashinfer() -> Any:
    flashinfer = importlib.import_module("flashinfer")
    if getattr(flashinfer, "__version__", None) != _SUPPORTED_FLASHINFER:
        raise RuntimeError(
            "PrimeRL finite-top-k sampled-logprob patch is pinned to "
            f"FlashInfer {_SUPPORTED_FLASHINFER}; found "
            f"{getattr(flashinfer, '__version__', None)!r}."
        )
    if not hasattr(flashinfer, "top_k"):
        raise RuntimeError("Installed FlashInfer does not expose flashinfer.top_k.")
    return flashinfer


def _debug_tensor_prefix(tensor: torch.Tensor | None) -> list[float | int] | None:
    if tensor is None:
        return None
    return tensor.detach().cpu()[:8].tolist()


def _debug_argmax_invariant_processors(logitsprocs: Any) -> list[str]:
    processors = []
    for processor in logitsprocs.argmax_invariant:
        min_p_count = getattr(processor, "min_p_count", None)
        if min_p_count is None:
            processors.append(processor.__class__.__name__)
        else:
            processors.append(f"{processor.__class__.__name__}(min_p_count={min_p_count})")
    return processors


def _batch_bucket(batch_size: int) -> str:
    if batch_size <= 16:
        return str(batch_size)
    upper = 1 << (batch_size - 1).bit_length()
    lower = upper >> 1
    return f"{lower + 1}-{upper}"


def _format_counter(counter: Counter[str]) -> dict[str, int]:
    def sort_key(item: tuple[str, int]) -> tuple[int, str]:
        key, _ = item
        if key.isdigit():
            return int(key), key
        return int(key.split("-", 1)[-1]), key

    return {key: value for key, value in sorted(counter.items(), key=sort_key)}


def _format_reason_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: value for key, value in sorted(counter.items())}


def _runtime_identity() -> dict[str, str | int | None]:
    return {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "infer_node_rank": os.environ.get("INFER_NODE_RANK"),
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _stats_log_path() -> Path | None:
    explicit = os.environ.get(_STATS_LOG_ENV)
    if explicit:
        return Path(explicit)

    log_dir = os.environ.get(_STATS_LOG_DIR_ENV)
    if not log_dir:
        return None

    host = socket.gethostname().split(".", 1)[0]
    rank = os.environ.get("INFER_NODE_RANK") or os.environ.get("SLURM_PROCID") or os.environ.get("RANK") or "unknown"
    local_rank = os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID") or "unknown"
    return Path(log_dir) / (f"finite_topk_sampler_stats_{host}_rank{rank}_local{local_rank}_pid{os.getpid()}.jsonl")


def _append_stats_record(logger, record: dict[str, Any]) -> None:
    path = _stats_log_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        logger.warning(
            "Failed to write finite-top-k sampled-logprob stats to %s: %s",
            path,
            exc,
        )


def _percentile(sorted_values: list[int], numerator: int, denominator: int) -> int:
    if not sorted_values:
        return 0
    index = (len(sorted_values) - 1) * numerator // denominator
    return sorted_values[index]


def _summarize_ints(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0.0}
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "p50": _percentile(sorted_values, 1, 2),
        "p90": _percentile(sorted_values, 9, 10),
        "max": sorted_values[-1],
        "mean": sum(sorted_values) / len(sorted_values),
    }


def _maybe_log_output_token_stats(forward, logger, sampling_metadata: Any) -> None:
    interval = _token_stats_interval()
    if interval == 0:
        return

    calls = getattr(forward, "_prime_rl_flashinfer_token_stats_calls", 0) + 1
    setattr(forward, "_prime_rl_flashinfer_token_stats_calls", calls)
    if calls % interval != 0:
        return

    output_token_ids = getattr(sampling_metadata, "output_token_ids", None)
    if not isinstance(output_token_ids, Sequence):
        logger.warning(
            "PrimeRL finite-top-k sampled-logprob output-token stats: calls=%d output_token_ids_unavailable.",
            calls,
        )
        return

    lengths = [len(tokens) for tokens in output_token_ids]
    # This is intentionally opt-in because set construction is CPU work. It is
    # used to choose the sparse-presence algorithm, not during normal training.
    unique_counts = [len(set(tokens)) for tokens in output_token_ids]
    logger.warning(
        "PrimeRL finite-top-k sampled-logprob output-token stats: calls=%d batch=%d lengths=%s unique_counts=%s.",
        calls,
        len(lengths),
        _summarize_ints(lengths),
        _summarize_ints(unique_counts),
    )


def _allclose_cpu(values: Any, target: float, eps: float = _SAMPLING_EPS) -> bool:
    return bool((abs(values - target) <= eps).all())


def _record_stats(
    forward,
    logger,
    batch_size: int,
    fast_path: bool,
    fallback_reason: str | None = None,
    traffic_class: str = "learner",
) -> None:
    interval = _stats_interval()
    if interval == 0:
        return

    stats = getattr(forward, "_prime_rl_flashinfer_sampled_logprob_stats", None)
    if stats is None:
        stats = {
            "calls": 0,
            "fast_calls": 0,
            "fallback_calls": 0,
            "fast_rows": 0,
            "fallback_rows": 0,
            "learner_fast_rows": 0,
            "learner_fallback_rows": 0,
            "warmup_or_profiling_fallback_rows": 0,
            "learner_fast_calls": 0,
            "learner_fallback_calls": 0,
            "warmup_or_profiling_fallback_calls": 0,
            "fast_batch_hist": Counter(),
            "fallback_batch_hist": Counter(),
            "fallback_reason_hist": Counter(),
            "fallback_traffic_hist": Counter(),
            "fallback_reason_by_traffic": Counter(),
        }
        setattr(forward, "_prime_rl_flashinfer_sampled_logprob_stats", stats)

    stats["calls"] += 1
    bucket = _batch_bucket(batch_size)
    if fast_path:
        stats["fast_calls"] += 1
        stats["fast_rows"] += batch_size
        stats["fast_batch_hist"][bucket] += 1
        if traffic_class == "learner":
            stats["learner_fast_calls"] += 1
            stats["learner_fast_rows"] += batch_size
    else:
        stats["fallback_calls"] += 1
        stats["fallback_rows"] += batch_size
        stats["fallback_batch_hist"][bucket] += 1
        reason = fallback_reason or "unknown"
        stats["fallback_reason_hist"][reason] += 1
        stats["fallback_traffic_hist"][traffic_class] += 1
        stats["fallback_reason_by_traffic"][f"{traffic_class}:{reason}"] += 1
        if traffic_class == "warmup_or_profiling":
            stats["warmup_or_profiling_fallback_calls"] += 1
            stats["warmup_or_profiling_fallback_rows"] += batch_size
        else:
            stats["learner_fallback_calls"] += 1
            stats["learner_fallback_rows"] += batch_size

    if stats["calls"] % interval != 0:
        return

    calls = stats["calls"]
    rows = stats["fast_rows"] + stats["fallback_rows"]
    learner_rows = stats["learner_fast_rows"] + stats["learner_fallback_rows"]
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event": "finite_topk_sampled_logprob_stats",
        **_runtime_identity(),
        "calls": calls,
        "fast_calls": stats["fast_calls"],
        "fallback_calls": stats["fallback_calls"],
        "hit_rate": stats["fast_calls"] / calls,
        "rows": rows,
        "fast_rows": stats["fast_rows"],
        "fallback_rows": stats["fallback_rows"],
        "row_hit_rate": stats["fast_rows"] / rows if rows else 0.0,
        "learner_fast_calls": stats["learner_fast_calls"],
        "learner_fallback_calls": stats["learner_fallback_calls"],
        "learner_fast_rows": stats["learner_fast_rows"],
        "learner_fallback_rows": stats["learner_fallback_rows"],
        "learner_row_hit_rate": (stats["learner_fast_rows"] / learner_rows if learner_rows else 0.0),
        "warmup_or_profiling_fallback_calls": stats["warmup_or_profiling_fallback_calls"],
        "warmup_or_profiling_fallback_rows": stats["warmup_or_profiling_fallback_rows"],
        "fast_batch_hist": _format_counter(stats["fast_batch_hist"]),
        "fallback_batch_hist": _format_counter(stats["fallback_batch_hist"]),
        "fallback_reason_hist": _format_reason_counter(stats["fallback_reason_hist"]),
        "fallback_traffic_hist": _format_reason_counter(stats["fallback_traffic_hist"]),
        "fallback_reason_by_traffic": _format_reason_counter(stats["fallback_reason_by_traffic"]),
    }
    _append_stats_record(logger, record)
    logger.warning(
        "PrimeRL finite-top-k sampled-logprob stats: calls=%d fast_calls=%d "
        "fallback_calls=%d hit_rate=%.6f rows=%d fast_rows=%d "
        "fallback_rows=%d row_hit_rate=%.6f learner_row_hit_rate=%.6f "
        "warmup_or_profiling_fallback_calls=%d fast_batch_hist=%s "
        "fallback_batch_hist=%s fallback_traffic_hist=%s "
        "fallback_reason_hist=%s.",
        calls,
        stats["fast_calls"],
        stats["fallback_calls"],
        stats["fast_calls"] / calls,
        rows,
        stats["fast_rows"],
        stats["fallback_rows"],
        stats["fast_rows"] / rows if rows else 0.0,
        record["learner_row_hit_rate"],
        stats["warmup_or_profiling_fallback_calls"],
        _format_counter(stats["fast_batch_hist"]),
        _format_counter(stats["fallback_batch_hist"]),
        _format_reason_counter(stats["fallback_traffic_hist"]),
        _format_reason_counter(stats["fallback_reason_hist"]),
    )


def _argmax_invariant_processors_supported(logitsprocs: Any) -> bool:
    for processor in logitsprocs.argmax_invariant:
        if processor.__class__.__name__ != "MinPLogitsProcessor":
            return False
        if getattr(processor, "min_p_count", None) != 0:
            return False
    return True


def _log_first_fallback(
    forward,
    logger,
    logits: torch.Tensor,
    sampling_metadata: Any,
    logprobs_mode: str,
    predict_bonus_token: bool,
    reason: str | None,
) -> None:
    if not _fallback_logging_enabled():
        return
    if getattr(forward, "_prime_rl_flashinfer_sampled_logprob_rollout_fallback_logged", False):
        return
    setattr(forward, "_prime_rl_flashinfer_sampled_logprob_rollout_fallback_logged", True)

    input_batch = _input_batch_cpu_fields(logits.shape[0])
    top_k_cpu = None
    temperature_cpu = None
    if input_batch is not None:
        top_k_cpu = input_batch.top_k_cpu[: logits.shape[0]][:8].tolist()
        temperature_cpu = input_batch.temperature_cpu[: logits.shape[0]][:8].tolist()

    logger.warning(
        "PrimeRL finite-top-k sampled-logprob fast path fallback: "
        "reason=%s device=%s logprobs_mode=%s predict_bonus_token=%s "
        "max_num_logprobs=%s has_logprob_token_ids=%s all_random=%s "
        "all_greedy=%s temperature=%s top_k=%s top_p=%s "
        "argmax_invariant=%s input_batch=%s top_k_cpu=%s temperature_cpu=%s.",
        reason or "unknown",
        logits.device,
        logprobs_mode,
        predict_bonus_token,
        sampling_metadata.max_num_logprobs,
        bool(sampling_metadata.logprob_token_ids),
        sampling_metadata.all_random,
        sampling_metadata.all_greedy,
        _debug_tensor_prefix(sampling_metadata.temperature),
        _debug_tensor_prefix(sampling_metadata.top_k),
        _debug_tensor_prefix(sampling_metadata.top_p),
        _debug_argmax_invariant_processors(sampling_metadata.logitsprocs),
        input_batch is not None,
        top_k_cpu,
        temperature_cpu,
    )


def _input_batch_cpu_fields(num_rows: int):
    ref = sampler_perf._INPUT_BATCH_REF
    if ref is None:
        return None
    input_batch = ref()
    if input_batch is None or num_rows > len(input_batch.req_ids):
        return None
    return input_batch


def _fallback_traffic_class(
    logits: torch.Tensor,
    sampling_metadata: Any,
    rejection_reason: str | None,
) -> str:
    """Classify fallbacks so synthetic warmup does not poison learner hit rate."""

    # vLLM's profiling/warmup sampler calls use SamplingMetadata without a live
    # InputBatch and without the OpenAI chat width-1 logprob request shape. The
    # debate learner traffic has a live InputBatch once real requests arrive.
    if (
        rejection_reason == "max_num_logprobs_not_width1"
        and sampling_metadata.max_num_logprobs is None
        and _input_batch_cpu_fields(logits.shape[0]) is None
    ):
        return "warmup_or_profiling"
    return "learner"


def _can_use_presence_only_logits_processors(
    logits: torch.Tensor,
    sampling_metadata: Any,
    predict_bonus_token: bool,
) -> bool:
    return (
        _presence_only_logits_processors_reason(
            logits,
            sampling_metadata,
            predict_bonus_token,
        )
        is None
    )


def _non_argmax_invariant_processors_reason(logitsprocs: Any) -> str | None:
    for processor in logitsprocs.non_argmax_invariant:
        name = processor.__class__.__name__
        if name == "MinTokensLogitsProcessor":
            min_toks = getattr(processor, "min_toks", None)
            if min_toks is None:
                return "min_tokens_state_missing"
            if min_toks:
                return f"min_tokens_active:{len(min_toks)}"
            continue
        if name == "LogitBiasLogitsProcessor":
            biases = getattr(processor, "biases", None)
            if biases is None:
                return "logit_bias_state_missing"
            if biases:
                return f"logit_bias_active:{len(biases)}"
            continue
        return f"unsupported_non_argmax_invariant:{name}"
    return None


def _thinking_budget_state_count(sampling_metadata: Any) -> int:
    holder = sampling_metadata.thinking_budget_state_holder
    if holder is None or not holder.has_tracked_requests():
        return 0
    holder_state = getattr(holder, "_state", None)
    if isinstance(holder_state, dict):
        return len(holder_state)
    return -1


def _presence_only_logits_processors_reason(
    logits: torch.Tensor,
    sampling_metadata: Any,
    predict_bonus_token: bool,
) -> str | None:
    if not _dense_presence_enabled():
        return "disabled"
    if predict_bonus_token:
        return "predict_bonus_token"
    if sampling_metadata.allowed_token_ids_mask is not None:
        return "allowed_token_ids_mask"
    if sampling_metadata.bad_words_token_ids:
        return "bad_words"
    non_argmax_reason = _non_argmax_invariant_processors_reason(sampling_metadata.logitsprocs)
    if non_argmax_reason is not None:
        return non_argmax_reason
    if not isinstance(sampling_metadata.output_token_ids, Sequence):
        return f"output_token_ids_type:{type(sampling_metadata.output_token_ids).__name__}"
    if len(sampling_metadata.output_token_ids) != logits.shape[0]:
        return f"output_token_ids_len:{len(sampling_metadata.output_token_ids)}"
    if sampling_metadata.no_penalties:
        return None

    input_batch = _input_batch_cpu_fields(logits.shape[0])
    if input_batch is not None:
        num_rows = logits.shape[0]
        if not _allclose_cpu(input_batch.frequency_penalties_cpu[:num_rows], 0.0):
            return "frequency_penalty_nonzero"
        if not _allclose_cpu(input_batch.repetition_penalties_cpu[:num_rows], 1.0):
            return "repetition_penalty_not_one"
        return None

    frequency = sampling_metadata.frequency_penalties
    repetition = sampling_metadata.repetition_penalties
    if not torch.all(torch.abs(frequency) <= _SAMPLING_EPS).item():
        return "frequency_penalty_nonzero"
    if not torch.all(torch.abs(repetition - 1.0) <= _SAMPLING_EPS).item():
        return "repetition_penalty_not_one"
    return None


def _build_output_tokens_for_dense_presence(
    logits: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor | None:
    vocab_size = logits.shape[1]
    input_batch = _input_batch_cpu_fields(logits.shape[0])
    if input_batch is not None and sampler_perf._staging is not None:
        return sampler_perf.build_output_tokens_fast(
            input_batch,
            sampler_perf._staging,
            sampling_metadata.output_token_ids,
            vocab_size,
            logits.device,
        )

    # Synthetic probes do not have a live InputBatch. Use the upstream tensor
    # builder there; live unrecognized InputBatch rows fall back above.
    from vllm.utils.platform_utils import is_pin_memory_available
    from vllm.utils.torch_utils import make_tensor_with_pad

    output_tokens = make_tensor_with_pad(
        sampling_metadata.output_token_ids,
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(),
    ).to(logits.device, non_blocking=True)
    output_tokens.masked_fill_(output_tokens == -1, vocab_size)
    return output_tokens


def _apply_presence_only_logits_processors(
    logits: torch.Tensor,
    sampling_metadata: Any,
    predict_bonus_token: bool,
) -> torch.Tensor | None:
    if sampling_metadata.no_penalties:
        output_tokens_for_holder = sampling_metadata.output_token_ids
    else:
        output_tokens = _build_output_tokens_for_dense_presence(logits, sampling_metadata)
        if output_tokens is None:
            return None
        output_tokens_for_holder = sampling_metadata.output_token_ids
        if output_tokens.numel() != 0:
            from vllm.model_executor.layers.utils import get_token_bin_counts_and_mask

            _, output_mask = get_token_bin_counts_and_mask(
                output_tokens,
                logits.shape[1],
                logits.shape[0],
            )
            logits.sub_(sampling_metadata.presence_penalties.unsqueeze(1) * output_mask)
    holder = sampling_metadata.thinking_budget_state_holder
    if holder is not None and holder.has_tracked_requests():
        holder.update_state(
            output_tokens_for_holder,
            sampling_metadata.spec_token_ids,
            repeat_indices=None,
        )
        logits = holder.apply_to_logits(
            logits,
            predict_bonus_token,
            sampling_metadata.spec_token_ids,
        )
    return logits


def _can_skip_unit_temperature(logits: torch.Tensor, sampling_metadata: Any) -> bool:
    if sampling_metadata.temperature is None:
        return False
    input_batch = _input_batch_cpu_fields(logits.shape[0])
    if input_batch is not None:
        return _allclose_cpu(input_batch.temperature_cpu[: logits.shape[0]], 1.0)
    return bool(torch.all(torch.abs(sampling_metadata.temperature - 1.0) <= _SAMPLING_EPS).item())


def _resolve_sampling_from_input_batch_with_reason(
    logits: torch.Tensor,
) -> tuple[tuple[int, float] | object, str | None]:
    input_batch = _input_batch_cpu_fields(logits.shape[0])
    if input_batch is None:
        return _NO_INPUT_BATCH, "input_batch_missing"

    num_rows = logits.shape[0]
    vocab_size = logits.shape[-1]
    top_k_rows = input_batch.top_k_cpu[:num_rows]
    top_k = int(top_k_rows[0])
    if top_k <= 0 or top_k > 64 or top_k >= vocab_size:
        return _INELIGIBLE_INPUT_BATCH, "input_batch_top_k_ineligible"
    if not (top_k_rows == top_k).all():
        return _INELIGIBLE_INPUT_BATCH, "input_batch_mixed_top_k"
    top_p_rows = input_batch.top_p_cpu[:num_rows]
    top_p = float(top_p_rows[0])
    if top_p <= 0.0 or top_p > 1.0:
        return _INELIGIBLE_INPUT_BATCH, "input_batch_top_p_ineligible"
    if not (top_p_rows == top_p).all():
        return _INELIGIBLE_INPUT_BATCH, "input_batch_mixed_top_p"
    if (input_batch.temperature_cpu[:num_rows] < _SAMPLING_EPS).any():
        return _INELIGIBLE_INPUT_BATCH, "input_batch_zero_temperature"
    return (top_k, top_p), None


def _resolve_sampling_from_input_batch(logits: torch.Tensor) -> tuple[int, float] | object:
    sampling, _ = _resolve_sampling_from_input_batch_with_reason(logits)
    return sampling


def _resolve_sampling_from_metadata_with_reason(
    logits: torch.Tensor,
    sampling_metadata: Any,
) -> tuple[tuple[int, float] | None, str | None]:
    top_k = sampling_metadata.top_k
    top_p = sampling_metadata.top_p
    assert top_k is not None
    assert top_p is not None

    vocab_size = logits.shape[-1]
    if torch.any(top_k <= 0) or torch.any(top_k > 64) or torch.any(top_k >= vocab_size):
        return None, "metadata_top_k_ineligible"
    if torch.any(top_p <= 0) or torch.any(top_p > 1):
        return None, "metadata_top_p_ineligible"
    # FlashInfer's top_k primitive takes one scalar K. Mixed-K batches stay on
    # the native vLLM path until we add a segmented/row-wise value+index kernel.
    if not torch.all(top_k == top_k[0]):
        return None, "metadata_mixed_top_k"
    if not torch.all(top_p == top_p[0]):
        return None, "metadata_mixed_top_p"
    if torch.any(sampling_metadata.temperature < _SAMPLING_EPS):
        return None, "metadata_zero_temperature"
    return (int(top_k[0].item()), float(top_p[0].item())), None


def _resolve_sampling_from_metadata(
    logits: torch.Tensor,
    sampling_metadata: Any,
) -> tuple[int, float] | None:
    sampling, _ = _resolve_sampling_from_metadata_with_reason(logits, sampling_metadata)
    return sampling


def _resolve_fast_path_sampling_with_reason(
    logits: torch.Tensor,
    sampling_metadata: Any,
    logprobs_mode: str,
    predict_bonus_token: bool,
) -> tuple[tuple[int, float] | None, str | None]:
    if predict_bonus_token:
        return None, "predict_bonus_token"
    if logits.device.type != "cuda":
        return None, "non_cuda_logits"
    if logprobs_mode != "processed_logprobs":
        return None, "logprobs_mode_not_processed_logprobs"
    if sampling_metadata.max_num_logprobs not in {0, 1}:
        return None, "max_num_logprobs_not_width1"
    if sampling_metadata.logprob_token_ids:
        return None, "explicit_logprob_token_ids"
    if sampling_metadata.generators:
        # vLLM's optimized samplers also fall back when per-request generators
        # are present. The Triton tail uses distribution-equivalent uniforms,
        # not per-row generator bitstreams.
        return None, "per_request_generators"
    if not sampling_metadata.all_random or sampling_metadata.all_greedy:
        return None, "not_all_random"
    if sampling_metadata.temperature is None:
        return None, "temperature_missing"
    if sampling_metadata.top_k is None or sampling_metadata.top_p is None:
        return None, "top_k_or_top_p_missing"
    if not _argmax_invariant_processors_supported(sampling_metadata.logitsprocs):
        # The target debate configs set min_p=0. vLLM still installs the
        # MinPLogitsProcessor, but its apply() is a no-op when min_p_count == 0.
        return None, "unsupported_argmax_invariant_processor"

    sampling, input_batch_reason = _resolve_sampling_from_input_batch_with_reason(logits)
    if sampling is _INELIGIBLE_INPUT_BATCH:
        return None, input_batch_reason
    if sampling is not _NO_INPUT_BATCH:
        return sampling, None
    sampling, metadata_reason = _resolve_sampling_from_metadata_with_reason(
        logits,
        sampling_metadata,
    )
    if sampling is None:
        return None, metadata_reason
    return sampling, None


def _resolve_fast_path_sampling(
    logits: torch.Tensor,
    sampling_metadata: Any,
    logprobs_mode: str,
    predict_bonus_token: bool,
) -> tuple[int, float] | None:
    sampling, _ = _resolve_fast_path_sampling_with_reason(
        logits,
        sampling_metadata,
        logprobs_mode,
        predict_bonus_token,
    )
    return sampling


def _metadata_can_use_fast_path(
    logits: torch.Tensor,
    sampling_metadata: Any,
    logprobs_mode: str,
    predict_bonus_token: bool,
) -> bool:
    return (
        _resolve_fast_path_sampling(
            logits,
            sampling_metadata,
            logprobs_mode,
            predict_bonus_token,
        )
        is not None
    )


@triton.jit
def _k_tail_uniform_kernel(
    vals,
    ids,
    uniforms,
    out_sampled_ids,
    out_sampled_logprobs,
    K,
    top_p,
    K_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, K_BLOCK)
    mask = offsets < K

    row_vals = tl.load(vals + row * K + offsets, mask=mask, other=-float("inf"))
    row_ids = tl.load(ids + row * K + offsets, mask=mask, other=0)

    max_val = tl.max(row_vals, axis=0)
    weights = tl.exp(row_vals - max_val)
    weights = tl.where(mask, weights, 0.0)
    prefix = tl.cumsum(weights, 0)
    total_weight = tl.sum(weights, axis=0)
    keep = mask & ((prefix - weights) < top_p * total_weight)
    kept_weights = tl.where(keep, weights, 0.0)
    kept_prefix = tl.cumsum(kept_weights, 0)
    support_sum = tl.sum(kept_weights, axis=0)

    threshold = tl.load(uniforms + row) * support_sum
    sample_rank = tl.min(
        tl.where((kept_prefix >= threshold) & keep, offsets, K_BLOCK),
        axis=0,
    )
    sampled_id = tl.max(tl.where(offsets == sample_rank, row_ids, 0), axis=0)
    sampled_val = tl.max(
        tl.where(offsets == sample_rank, row_vals, -float("inf")),
        axis=0,
    )

    tl.store(out_sampled_ids + row, sampled_id)
    tl.store(out_sampled_logprobs + row, sampled_val - max_val - tl.log(support_sum))


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _flashinfer_support(
    flashinfer: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert sampling_metadata.top_p is not None

    vals, ids = flashinfer.top_k(logits.contiguous(), top_k, sorted=False)
    vals, order = vals.sort(dim=-1, descending=True)
    ids = ids.gather(-1, order)

    weights = torch.softmax(vals, dim=-1, dtype=torch.float32)
    prefix = weights.cumsum(dim=-1)
    keep = (prefix - weights) < sampling_metadata.top_p.unsqueeze(1)
    support_vals = vals.masked_fill(~keep, -float("inf"))
    support_logprobs = support_vals - torch.logsumexp(
        support_vals,
        dim=-1,
        keepdim=True,
    )
    return ids, support_vals, support_logprobs


def _sample_with_logprob_torch(
    flashinfer: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.v1.sample.ops.topk_topp_sampler import random_sample

    ids, support_vals, support_logprobs = _flashinfer_support(
        flashinfer,
        logits,
        sampling_metadata,
        top_k,
    )
    probs = torch.softmax(support_vals, dim=-1, dtype=torch.float32)
    sampled_in_topk = random_sample(probs, sampling_metadata.generators)
    sampled = ids.gather(1, sampled_in_topk.unsqueeze(1)).squeeze(1).long()
    sampled_logprobs = support_logprobs.gather(
        1,
        sampled_in_topk.unsqueeze(1),
    ).squeeze(1)
    return sampled, sampled_logprobs


def _sample_with_logprob_triton(
    flashinfer: Any,
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    vals, ids = flashinfer.top_k(logits.contiguous(), top_k, sorted=False)
    vals, order = vals.sort(dim=-1, descending=True)
    ids = ids.gather(-1, order)

    batch_size = vals.shape[0]
    uniforms = torch.rand(batch_size, device=vals.device, dtype=torch.float32)
    sampled = torch.empty(batch_size, device=vals.device, dtype=torch.int64)
    sampled_logprobs = torch.empty(batch_size, device=vals.device, dtype=torch.float32)
    _k_tail_uniform_kernel[(batch_size,)](
        vals,
        ids,
        uniforms,
        sampled,
        sampled_logprobs,
        top_k,
        top_p,
        K_BLOCK=_next_power_of_two(top_k),
        num_warps=1,
    )
    return sampled, sampled_logprobs


def _maybe_precompile_triton_tail(logger) -> None:
    if not _precompile_tail_enabled() or not _use_triton_tail():
        return
    if getattr(_maybe_precompile_triton_tail, "_done", False):
        return
    if not torch.cuda.is_available():
        logger.warning(
            "PrimeRL finite-top-k sampled-logprob Triton tail precompile skipped: "
            "CUDA is not available in this process."
        )
        return

    top_k = _precompile_tail_top_k()
    top_p_values = _precompile_tail_top_p_values()
    vocab_size = _precompile_tail_vocab_size()
    batch_sizes = _precompile_tail_batch_sizes()
    if top_k <= 0 or top_k > 64 or top_k >= vocab_size:
        logger.warning(
            "PrimeRL finite-top-k sampled-logprob Triton tail precompile skipped: invalid top_k=%d for vocab_size=%d.",
            top_k,
            vocab_size,
        )
        return
    invalid_top_p_values = [top_p for top_p in top_p_values if top_p <= 0.0 or top_p > 1.0]
    if invalid_top_p_values:
        logger.warning(
            "PrimeRL finite-top-k sampled-logprob Triton tail precompile skipped: invalid top_p_values=%s.",
            top_p_values,
        )
        return
    invalid_batch_sizes = [size for size in batch_sizes if size <= 0]
    if not batch_sizes or invalid_batch_sizes:
        logger.warning(
            "PrimeRL finite-top-k sampled-logprob Triton tail precompile skipped: invalid batch_sizes=%s.",
            batch_sizes,
        )
        return

    flashinfer = _load_flashinfer()
    device = torch.device("cuda", torch.cuda.current_device())
    try:
        with torch.inference_mode():
            for batch_size in batch_sizes:
                logits = torch.zeros(
                    (batch_size, vocab_size),
                    device=device,
                    dtype=torch.float32,
                )
                for top_p in top_p_values:
                    _sample_with_logprob_triton(flashinfer, logits, top_k, top_p)
            torch.cuda.synchronize(device)
    except Exception as exc:
        logger.warning(
            "PrimeRL finite-top-k sampled-logprob Triton tail precompile failed: %s",
            exc,
        )
        return

    setattr(_maybe_precompile_triton_tail, "_done", True)
    logger.warning(
        "Precompiled PrimeRL finite-top-k sampled-logprob Triton tail "
        "for batches=%s vocab=%d top_k=%d top_p_values=%s.",
        batch_sizes,
        vocab_size,
        top_k,
        top_p_values,
    )


def apply_flashinfer_sampled_logprob_patch() -> None:
    if not _env_enabled():
        return

    import vllm
    from vllm.logger import init_logger
    from vllm.v1.outputs import LogprobsTensors, SamplerOutput
    from vllm.v1.sample.sampler import Sampler

    logger = init_logger(__name__)

    if vllm.__version__ != sampler_perf.SUPPORTED_VLLM:
        raise RuntimeError(
            "PrimeRL finite-top-k sampled-logprob patch is pinned to "
            f"vLLM {sampler_perf.SUPPORTED_VLLM}; found {vllm.__version__}."
        )
    if getattr(Sampler.forward, _PATCH_MARKER, False):
        return

    flashinfer = _load_flashinfer()
    original_forward = Sampler.forward
    original_init = Sampler.__init__

    if not getattr(Sampler.__init__, _INIT_PATCH_MARKER, False):

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            _maybe_precompile_triton_tail(logger)

        setattr(__init__, _INIT_PATCH_MARKER, True)
        Sampler.__init__ = __init__

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override=None,
    ) -> SamplerOutput:
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        sampling, rejection_reason = _resolve_fast_path_sampling_with_reason(
            logits,
            sampling_metadata,
            logprobs_mode,
            predict_bonus_token,
        )
        if sampling is None:
            traffic_class = _fallback_traffic_class(
                logits,
                sampling_metadata,
                rejection_reason,
            )
            _record_stats(
                forward,
                logger,
                logits.shape[0],
                fast_path=False,
                fallback_reason=rejection_reason,
                traffic_class=traffic_class,
            )
            _log_first_fallback(
                forward,
                logger,
                logits,
                sampling_metadata,
                logprobs_mode,
                predict_bonus_token,
                rejection_reason,
            )
            return original_forward(
                self,
                logits,
                sampling_metadata,
                predict_bonus_token=predict_bonus_token,
                logprobs_mode_override=logprobs_mode_override,
            )
        top_k, top_p = sampling
        _record_stats(forward, logger, logits.shape[0], fast_path=True)
        _maybe_log_output_token_stats(forward, logger, sampling_metadata)

        logits = logits.to(torch.float32)
        dense_presence_reason = _presence_only_logits_processors_reason(
            logits,
            sampling_metadata,
            predict_bonus_token,
        )
        dense_presence = dense_presence_reason is None
        if dense_presence:
            processed_logits = _apply_presence_only_logits_processors(
                logits,
                sampling_metadata,
                predict_bonus_token,
            )
            if processed_logits is None:
                dense_presence = False
                dense_presence_reason = "output_tokens_unavailable"
            else:
                logits = processed_logits
        if not dense_presence:
            logits = self.apply_logits_processors(
                logits,
                sampling_metadata,
                predict_bonus_token,
            )
        skip_temperature = _can_skip_unit_temperature(logits, sampling_metadata)
        if not skip_temperature:
            logits = self.apply_temperature(
                logits,
                sampling_metadata.temperature,
                sampling_metadata.all_random,
            )
        hit_count = getattr(forward, "_prime_rl_flashinfer_sampled_logprob_hit_count", 0)
        if hit_count < _hit_log_limit():
            setattr(forward, "_prime_rl_flashinfer_sampled_logprob_hit_count", hit_count + 1)
            logger.warning(
                "PrimeRL finite-top-k sampled-logprob fast path hit: "
                "hit=%d batch=%d vocab=%d top_k=%d top_p=%.6g tail=%s "
                "dense_presence=%s dense_presence_reason=%s "
                "thinking_budget_state=%d skip_temperature=%s "
                "max_num_logprobs=%s.",
                hit_count + 1,
                logits.shape[0],
                logits.shape[-1],
                top_k,
                top_p,
                "triton" if _use_triton_tail() else "torch",
                dense_presence,
                dense_presence_reason,
                _thinking_budget_state_count(sampling_metadata),
                skip_temperature,
                sampling_metadata.max_num_logprobs,
            )
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)
        if _use_triton_tail():
            sampled, sampled_logprobs = _sample_with_logprob_triton(
                flashinfer,
                logits,
                top_k,
                top_p,
            )
        else:
            sampled, sampled_logprobs = _sample_with_logprob_torch(
                flashinfer,
                logits,
                sampling_metadata,
                top_k,
            )
        sampled_i32 = sampled.to(torch.int32)
        logprobs_tensors = LogprobsTensors(
            logprob_token_ids=sampled_i32.unsqueeze(-1),
            logprobs=sampled_logprobs.to(torch.float32).unsqueeze(-1),
            selected_token_ranks=torch.ones_like(sampled_i32, dtype=torch.int32),
        )
        return SamplerOutput(
            sampled_token_ids=sampled_i32.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )

    setattr(forward, _PATCH_MARKER, True)
    Sampler.forward = forward
    logger.warning("Enabled PrimeRL finite-top-k sampled-logprob fast path for vLLM processed sampled-token logprobs.")
