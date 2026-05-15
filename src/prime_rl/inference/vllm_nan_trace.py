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
_NEMOTRON_H_LAYERS_PATCHED = False
_NEMOTRON_H_MIXERS_PATCHED = False
_ATTENTION_OPS_PATCHED = False
_MAMBA_MIXER2_OPS_PATCHED = False
_CUDAGRAPH_REPLAY_PATCHED = False
_WRITE_LOCK = threading.Lock()
_ACTIVE_STATE_DUMPED: set[str] = set()
_TARGET_BATCH_TRACE_COUNT: dict[int, int] = {}
_CUDAGRAPH_REPLAY_TRACE_COUNT: dict[int, int] = {}
_MODEL_RUNNER_TRACE_LOCAL = threading.local()
_MAMBA_TRACE_LOCAL = threading.local()


def install_vllm_nan_trace() -> None:
    """Install env-gated vLLM tracing hooks for Nemotron NaN investigation."""
    if not _trace_dir():
        return

    with _INSTALL_LOCK:
        _patch_output_processor()
        _patch_scheduler()
        _patch_model_runner()
        _patch_cudagraph_replay()
        _patch_nemotron_h_layers()
        _patch_nemotron_h_mixers()
        _patch_attention_ops()
        _patch_mamba_mixer2_ops()


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


def _scheduler_compact_trace_enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_SCHEDULER_COMPACT", "0") != "0"


def _model_runner_trace_enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_MODEL_RUNNER", "1") != "0"


def _model_runner_batch_trace_enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_BATCH", "0") != "0"


def _model_runner_batch_gpu_trace_enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_BATCH_GPU", "0") != "0"


def _target_batch_trace_enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_TARGET_BATCH", "0") != "0"


def _target_batch_trace_only_full() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_TARGET_BATCH_ONLY_FULL", "1") != "0"


def _target_batch_trace_only_uniform() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_TARGET_BATCH_ONLY_UNIFORM", "1") != "0"


def _nemotron_h_layer_trace_enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_NEMOTRON_H_LAYERS", "0") != "0"


def _nemotron_h_mixer_trace_enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_NEMOTRON_H_MIXERS", "0") != "0"


def _attention_op_trace_enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_ATTENTION_OPS", "0") != "0"


def _mamba_op_trace_enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_MAMBA_OPS", "0") != "0"


def _mamba_op_trace_requires_request_ids() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_MAMBA_OPS_REQUIRE_REQUEST_IDS", "1") != "0"


def _cudagraph_replay_trace_enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_CUDAGRAPH_REPLAY", "0") != "0"


def _cudagraph_replay_trace_only_full() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_CUDAGRAPH_REPLAY_ONLY_FULL", "1") != "0"


def _cudagraph_replay_trace_fail_fast() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_CUDAGRAPH_REPLAY_FAIL_FAST", "1") != "0"


def _cudagraph_replay_tensor_limit() -> int:
    return _env_int("PRIME_RL_VLLM_NAN_TRACE_CUDAGRAPH_REPLAY_TENSOR_LIMIT", 32) or 32


def _cudagraph_replay_numel_limit() -> int:
    return _env_int("PRIME_RL_VLLM_NAN_TRACE_CUDAGRAPH_REPLAY_NUMEL_LIMIT", 50_000_000) or 50_000_000


def _active_state_trace_enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_NAN_TRACE_ACTIVE_STATES", "1") != "0"


def _torch_is_compiling() -> bool:
    try:
        import torch

        compiler = getattr(torch, "compiler", None)
        if compiler is not None and compiler.is_compiling():
            return True
        dynamo = getattr(torch, "_dynamo", None)
        if dynamo is not None and dynamo.is_compiling():
            return True
    except Exception:
        return False
    return False


def _active_output_token_limit() -> int:
    value = os.environ.get("PRIME_RL_VLLM_NAN_TRACE_ACTIVE_OUTPUT_TOKENS", "4096")
    try:
        return int(value)
    except ValueError:
        return 4096


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


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
    original_determine_batch = getattr(GPUModelRunner, "_determine_batch_execution_and_padding", None)

    if original_determine_batch is not None:

        def _patched_determine_batch_execution_and_padding(self, *args, **kwargs):
            result = original_determine_batch(self, *args, **kwargs)
            if _model_runner_trace_enabled() or _model_runner_batch_trace_enabled():
                try:
                    self._prime_rl_nan_trace_batch_execution = _batch_execution_summary(result, args, kwargs)
                    context = _current_model_runner_context() or {}
                    _MODEL_RUNNER_TRACE_LOCAL.context = _model_runner_execution_context(
                        self,
                        execute_index=context.get("execute_index"),
                    )
                except Exception:
                    logger.exception("Failed to trace vLLM model runner batch execution decision")
            return result

        GPUModelRunner._determine_batch_execution_and_padding = _patched_determine_batch_execution_and_padding

    def _patched_execute_model(self, *args, **kwargs):
        execute_index = _next_model_runner_execute_index(self)
        previous_context = getattr(_MODEL_RUNNER_TRACE_LOCAL, "context", None)
        _MODEL_RUNNER_TRACE_LOCAL.context = _model_runner_execution_context(self, execute_index=execute_index)
        try:
            result = original_execute_model(self, *args, **kwargs)
            if _model_runner_batch_trace_enabled():
                try:
                    state = getattr(self, "execute_model_state", None)
                    if state is not None:
                        _trace_model_runner_batch_state(
                            self,
                            state,
                            getattr(self, "_prime_rl_nan_trace_batch_execution", None),
                        )
                except Exception:
                    logger.exception("Failed to trace vLLM model runner batch state")

            if _target_batch_trace_enabled():
                try:
                    state = getattr(self, "execute_model_state", None)
                    if state is not None:
                        _trace_model_runner_target_batch_state(
                            self,
                            state,
                            getattr(self, "_prime_rl_nan_trace_batch_execution", None),
                        )
                except Exception:
                    logger.exception("Failed to trace targeted vLLM model runner batch state")

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
                        batch_execution=getattr(self, "_prime_rl_nan_trace_batch_execution", None),
                    )
            except Exception:
                logger.exception("Failed to trace vLLM model runner logits state")
            return result
        finally:
            _MODEL_RUNNER_TRACE_LOCAL.context = previous_context

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
                batch_execution=getattr(self, "_prime_rl_nan_trace_batch_execution", None),
            )
        except Exception:
            logger.exception("Failed to trace vLLM model runner sample")
        return sampler_output

    GPUModelRunner.execute_model = _patched_execute_model
    GPUModelRunner._sample = _patched_sample
    _MODEL_RUNNER_PATCHED = True


def _patch_cudagraph_replay() -> None:
    global _CUDAGRAPH_REPLAY_PATCHED
    if _CUDAGRAPH_REPLAY_PATCHED:
        return
    if not _cudagraph_replay_trace_enabled():
        return

    try:
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.forward_context import get_forward_context, is_forward_context_available
    except Exception:
        logger.exception("Failed to import vLLM CUDA graph wrapper for NaN tracing")
        return

    original_call = CUDAGraphWrapper.__call__

    def _patched_call(self, *args, **kwargs):
        trace_info = None
        try:
            trace_info = _cudagraph_replay_trace_info(
                self,
                get_forward_context=get_forward_context,
                is_forward_context_available=is_forward_context_available,
            )
        except Exception:
            logger.exception("Failed to inspect vLLM CUDA graph replay context")

        if trace_info is None:
            return original_call(self, *args, **kwargs)

        pre_summary = _tensor_tree_summary(
            {"args": args, "kwargs": kwargs},
            max_tensors=_cudagraph_replay_tensor_limit(),
            max_numel=_cudagraph_replay_numel_limit(),
        )
        output = original_call(self, *args, **kwargs)
        post_summary = _tensor_tree_summary(
            output,
            max_tensors=_cudagraph_replay_tensor_limit(),
            max_numel=_cudagraph_replay_numel_limit(),
        )

        if pre_summary["nonfinite_tensor_count"] or post_summary["nonfinite_tensor_count"]:
            _trace_cudagraph_replay_nonfinite(
                trace_info=trace_info,
                pre_summary=pre_summary,
                post_summary=post_summary,
            )
            if _cudagraph_replay_trace_fail_fast() and post_summary["nonfinite_tensor_count"]:
                raise RuntimeError(
                    "Prime-RL vLLM NaN trace: FULL CUDA graph replay produced non-finite output"
                )
        return output

    CUDAGraphWrapper.__call__ = _patched_call
    _CUDAGRAPH_REPLAY_PATCHED = True


def _cudagraph_replay_trace_info(
    cudagraph_wrapper: Any,
    *,
    get_forward_context: Any,
    is_forward_context_available: Any,
) -> dict[str, Any] | None:
    if not is_forward_context_available():
        return None

    forward_context = get_forward_context()
    batch_descriptor = getattr(forward_context, "batch_descriptor", None)
    runtime_mode = getattr(forward_context, "cudagraph_runtime_mode", None)
    wrapper_mode = getattr(cudagraph_wrapper, "runtime_mode", None)
    if batch_descriptor is None or runtime_mode is None or runtime_mode != wrapper_mode:
        return None

    mode_name = _enum_name(runtime_mode)
    if _cudagraph_replay_trace_only_full() and "FULL" not in mode_name:
        return None

    entries = getattr(cudagraph_wrapper, "concrete_cudagraph_entries", {}) or {}
    entry = entries.get(batch_descriptor)
    if entry is None or getattr(entry, "cudagraph", None) is None:
        return None
    if not _cudagraph_replay_trace_take_record():
        return None

    return {
        "runtime_mode": _enum_summary(runtime_mode),
        "wrapper_runtime_mode": _enum_summary(wrapper_mode),
        "batch_descriptor": _object_attr_summary(
            batch_descriptor,
            ("num_tokens", "num_reqs", "uniform", "has_lora", "num_active_loras"),
        ),
        "entry_batch_descriptor": _object_attr_summary(
            getattr(entry, "batch_descriptor", None),
            ("num_tokens", "num_reqs", "uniform", "has_lora", "num_active_loras"),
        ),
        "input_addresses": _sequence_summary(getattr(entry, "input_addresses", None)),
        "slot_mapping": _slot_mappings_summary(getattr(forward_context, "slot_mapping", None)),
        "model_runner_context": _current_model_runner_context(),
    }


def _cudagraph_replay_trace_take_record() -> bool:
    limit = _env_int("PRIME_RL_VLLM_NAN_TRACE_CUDAGRAPH_REPLAY_LIMIT", 0)
    if limit is not None and limit <= 0:
        return True
    pid = os.getpid()
    with _WRITE_LOCK:
        count = _CUDAGRAPH_REPLAY_TRACE_COUNT.get(pid, 0)
        if limit is not None and count >= limit:
            return False
        _CUDAGRAPH_REPLAY_TRACE_COUNT[pid] = count + 1
    return True


def _trace_cudagraph_replay_nonfinite(
    *,
    trace_info: dict[str, Any],
    pre_summary: dict[str, Any],
    post_summary: dict[str, Any],
) -> None:
    _write_jsonl(
        "cudagraph_replay_nonfinite",
        {
            "schema": "prime_rl.vllm_cudagraph_replay_nonfinite.v1",
            "created_unix": time.time(),
            "pid": os.getpid(),
            "trace_info": trace_info,
            "pre": pre_summary,
            "post": post_summary,
        },
    )


def _patch_nemotron_h_layers() -> None:
    global _NEMOTRON_H_LAYERS_PATCHED
    if _NEMOTRON_H_LAYERS_PATCHED:
        return
    if not _nemotron_h_layer_trace_enabled():
        return

    try:
        from vllm.model_executor.models import nemotron_h
    except Exception:
        logger.exception("Failed to import vLLM NemotronH layers for NaN tracing")
        return

    for layer_type_name in (
        "NemotronHMLPDecoderLayer",
        "NemotronHMoEDecoderLayer",
        "NemotronHMambaDecoderLayer",
        "NemotronHAttentionDecoderLayer",
    ):
        layer_type = getattr(nemotron_h, layer_type_name, None)
        if layer_type is None:
            continue
        _patch_nemotron_h_layer_type(layer_type, layer_type_name)

    _NEMOTRON_H_LAYERS_PATCHED = True


def _patch_nemotron_h_layer_type(layer_type: Any, layer_type_name: str) -> None:
    original_init = layer_type.__init__
    original_forward = layer_type.forward

    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        layer_idx = kwargs.get("layer_idx")
        if layer_idx is None and len(args) >= 2:
            layer_idx = args[1]
        self._prime_nan_trace_layer_idx = layer_idx
        self._prime_nan_trace_layer_type = layer_type_name

    def _patched_forward(self, *args, **kwargs):
        hidden_in, residual_in = _nemotron_h_forward_inputs(
            layer_type_name=layer_type_name,
            args=args,
            kwargs=kwargs,
        )
        output = original_forward(self, *args, **kwargs)
        try:
            _trace_nemotron_h_layer_output(
                layer_idx=getattr(self, "_prime_nan_trace_layer_idx", None),
                layer_type=getattr(self, "_prime_nan_trace_layer_type", layer_type_name),
                hidden_in=hidden_in,
                residual_in=residual_in,
                output=output,
            )
        except Exception:
            logger.exception("Failed to trace NemotronH layer output")
        return output

    layer_type.__init__ = _patched_init
    layer_type.forward = _patched_forward


def _nemotron_h_forward_inputs(
    *,
    layer_type_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    hidden_in = kwargs.get("hidden_states")
    residual_in = kwargs.get("residual")

    if hidden_in is None:
        if "Attention" in layer_type_name:
            if len(args) >= 2:
                hidden_in = args[1]
            if residual_in is None and len(args) >= 3:
                residual_in = args[2]
        else:
            if len(args) >= 1:
                hidden_in = args[0]
            if residual_in is None and len(args) >= 2:
                residual_in = args[1]

    return hidden_in, residual_in


def _patch_nemotron_h_mixers() -> None:
    global _NEMOTRON_H_MIXERS_PATCHED
    if _NEMOTRON_H_MIXERS_PATCHED:
        return
    if not _nemotron_h_mixer_trace_enabled():
        return

    try:
        from vllm.model_executor.models import nemotron_h
    except Exception:
        logger.exception("Failed to import vLLM NemotronH mixers for NaN tracing")
        return

    mixer_specs = (
        ("NemotronHAttention", "attention"),
        ("NemotronHMoE", "moe"),
        ("NemotronHMLP", "mlp"),
    )
    for type_name, mixer_kind in mixer_specs:
        mixer_type = getattr(nemotron_h, type_name, None)
        if mixer_type is None:
            continue
        _patch_nemotron_h_mixer_type(mixer_type, type_name, mixer_kind)

    _NEMOTRON_H_MIXERS_PATCHED = True


def _patch_nemotron_h_mixer_type(mixer_type: Any, type_name: str, mixer_kind: str) -> None:
    original_forward = mixer_type.forward

    def _patched_forward(self, *args, **kwargs):
        hidden_in = kwargs.get("hidden_states")
        if hidden_in is None and args:
            hidden_in = args[0]

        output = original_forward(self, *args, **kwargs)
        try:
            if _mamba_trace_context_allows_event() and (
                _tensor_has_nonfinite(hidden_in) or _tensor_has_nonfinite(output)
            ):
                _trace_nemotron_h_mixer_event(
                    mixer_type=type_name,
                    mixer_kind=mixer_kind,
                    layer_name=_nemotron_mixer_layer_name(self),
                    hidden_in=hidden_in,
                    output=output,
                )
        except Exception:
            logger.exception("Failed to trace NemotronH mixer output")
        return output

    mixer_type.forward = _patched_forward


def _nemotron_mixer_layer_name(mixer: Any) -> str | None:
    if hasattr(mixer, "prefix"):
        return getattr(mixer, "prefix", None)
    attn = getattr(mixer, "attn", None)
    if attn is not None and hasattr(attn, "layer_name"):
        return getattr(attn, "layer_name", None)
    gate = getattr(mixer, "gate", None)
    if gate is not None and hasattr(gate, "prefix"):
        return getattr(gate, "prefix", None)
    return None


def _patch_attention_ops() -> None:
    global _ATTENTION_OPS_PATCHED
    if _ATTENTION_OPS_PATCHED:
        return
    if not _attention_op_trace_enabled():
        return

    try:
        from vllm.model_executor.layers.attention.attention import Attention
    except Exception:
        logger.exception("Failed to import vLLM Attention for NaN tracing")
        return

    original_forward = Attention.forward

    def _patched_forward(self, query, key, value, output_shape=None):
        output = original_forward(self, query, key, value, output_shape=output_shape)
        try:
            if _mamba_trace_context_allows_event() and (
                _tensor_has_nonfinite(query)
                or _tensor_has_nonfinite(key)
                or _tensor_has_nonfinite(value)
                or _tensor_has_nonfinite(output)
            ):
                _trace_attention_op_event(
                    layer_name=getattr(self, "layer_name", None),
                    query=query,
                    key=key,
                    value=value,
                    output=output,
                )
        except Exception:
            logger.exception("Failed to trace vLLM attention op")
        return output

    Attention.forward = _patched_forward
    _ATTENTION_OPS_PATCHED = True


def _patch_mamba_mixer2_ops() -> None:
    global _MAMBA_MIXER2_OPS_PATCHED
    if _MAMBA_MIXER2_OPS_PATCHED:
        return
    if not _mamba_op_trace_enabled():
        return

    try:
        from vllm.model_executor.layers.mamba import mamba_mixer2
    except Exception:
        logger.exception("Failed to import vLLM MambaMixer2 for NaN tracing")
        return

    mixer_type = getattr(mamba_mixer2, "MambaMixer2", None)
    if mixer_type is not None:
        original_conv_ssm_forward = mixer_type.conv_ssm_forward

        def _patched_conv_ssm_forward(self, *args, **kwargs):
            previous_context = getattr(_MAMBA_TRACE_LOCAL, "context", None)
            _MAMBA_TRACE_LOCAL.context = {
                "layer_name": getattr(self, "prefix", None),
                "tp_size": getattr(self, "tp_size", None),
                "num_heads": getattr(self, "num_heads", None),
                "head_dim": getattr(self, "head_dim", None),
                "n_groups": getattr(self, "n_groups", None),
                "ssm_state_size": getattr(self, "ssm_state_size", None),
                "conv_kernel_size": getattr(self, "conv_kernel_size", None),
                "projected_states": args[0] if args else None,
            }
            try:
                return original_conv_ssm_forward(self, *args, **kwargs)
            finally:
                _MAMBA_TRACE_LOCAL.context = previous_context

        mixer_type.conv_ssm_forward = _patched_conv_ssm_forward

    original_causal_conv1d_update = getattr(mamba_mixer2, "causal_conv1d_update", None)
    if original_causal_conv1d_update is not None:

        def _patched_causal_conv1d_update(*args, **kwargs):
            input_tensor = args[0] if args else None
            result = original_causal_conv1d_update(*args, **kwargs)
            try:
                if _mamba_trace_context_allows_event() and (
                    _tensor_has_nonfinite(input_tensor) or _tensor_has_nonfinite(result)
                ):
                    _trace_mamba_op_event(
                        op_name="causal_conv1d_update",
                        before={
                            "hidden_states_B_C_d": _tensor_total_counts(input_tensor),
                            "projected_states": _tensor_total_counts(_mamba_projected_states_from_context()),
                            "conv_state": _tensor_meta(args[1]) if len(args) > 1 else None,
                        },
                        after={
                            "hidden_states_B_C_d": _tensor_total_counts(result),
                        },
                        metadata={
                            "conv_state_indices": _tensor_or_sequence_summary(kwargs.get("conv_state_indices")),
                            "block_idx_last_scheduled_token": _tensor_or_sequence_summary(
                                kwargs.get("block_idx_last_scheduled_token")
                            ),
                            "initial_state_idx": _tensor_or_sequence_summary(kwargs.get("initial_state_idx")),
                            "num_accepted_tokens": _tensor_or_sequence_summary(kwargs.get("num_accepted_tokens")),
                            "query_start_loc": _tensor_or_sequence_summary(kwargs.get("query_start_loc")),
                            "max_query_len": kwargs.get("max_query_len"),
                        },
                    )
            except Exception:
                logger.exception("Failed to trace Mamba2 causal_conv1d_update")
            return result

        mamba_mixer2.causal_conv1d_update = _patched_causal_conv1d_update

    original_selective_state_update = getattr(mamba_mixer2, "selective_state_update", None)
    if original_selective_state_update is not None:

        def _patched_selective_state_update(*args, **kwargs):
            ssm_state = args[0] if len(args) > 0 else None
            hidden_states_d = args[1] if len(args) > 1 else None
            out_tensor = kwargs.get("out")
            result = original_selective_state_update(*args, **kwargs)
            try:
                if _mamba_trace_context_allows_event() and (
                    _tensor_has_nonfinite(hidden_states_d)
                    or _tensor_has_nonfinite(out_tensor)
                    or _tensor_has_nonfinite(result)
                ):
                    _trace_mamba_op_event(
                        op_name="selective_state_update",
                        before={
                            "hidden_states_d": _tensor_total_counts(hidden_states_d),
                            "ssm_state": _tensor_meta(ssm_state),
                        },
                        after={
                            "out": _tensor_total_counts(out_tensor),
                            "result": _tensor_total_counts(result),
                        },
                        metadata={
                            "state_batch_indices": _tensor_or_sequence_summary(kwargs.get("state_batch_indices")),
                            "dst_state_batch_indices": _tensor_or_sequence_summary(
                                kwargs.get("dst_state_batch_indices")
                            ),
                            "num_accepted_tokens": _tensor_or_sequence_summary(kwargs.get("num_accepted_tokens")),
                            "cu_seqlens": _tensor_or_sequence_summary(kwargs.get("cu_seqlens")),
                            "dt": _tensor_total_counts(args[2]) if len(args) > 2 else None,
                            "B": _tensor_total_counts(args[4]) if len(args) > 4 else None,
                            "C": _tensor_total_counts(args[5]) if len(args) > 5 else None,
                        },
                    )
            except Exception:
                logger.exception("Failed to trace Mamba2 selective_state_update")
            return result

        mamba_mixer2.selective_state_update = _patched_selective_state_update

    _MAMBA_MIXER2_OPS_PATCHED = True


def _mamba_projected_states_from_context() -> Any:
    context = getattr(_MAMBA_TRACE_LOCAL, "context", None)
    if not isinstance(context, dict):
        return None
    return context.get("projected_states")


def _trace_nemotron_h_mixer_event(
    *,
    mixer_type: str,
    mixer_kind: str,
    layer_name: str | None,
    hidden_in: Any,
    output: Any,
) -> None:
    hidden_counts = _tensor_row_counts(hidden_in) if hasattr(hidden_in, "shape") else None
    output_counts = _tensor_row_counts(output) if hasattr(output, "shape") else None
    selected_rows: set[int] = set()
    if hidden_counts is not None:
        _add_nonfinite_rows(selected_rows, hidden_counts)
    if output_counts is not None:
        _add_nonfinite_rows(selected_rows, output_counts)

    rows = []
    for row_idx in sorted(selected_rows):
        rows.append(
            {
                "row_index": row_idx,
                "hidden_in": _row_count_at(hidden_counts, row_idx) if hidden_counts else None,
                "output": _row_count_at(output_counts, row_idx) if output_counts else None,
            }
        )

    _write_jsonl(
        "nemotron_h_mixer_states",
        {
            "schema": "prime_rl.vllm_nemotron_h_mixer_trace.v1",
            "created_unix": time.time(),
            "pid": os.getpid(),
            "mixer_type": mixer_type,
            "mixer_kind": mixer_kind,
            "layer_name": layer_name,
            "hidden_in": _tensor_meta(hidden_in),
            "output": _tensor_meta(output),
            "model_runner_context": _current_model_runner_context(),
            "rows": rows,
        },
    )


def _trace_attention_op_event(
    *,
    layer_name: str | None,
    query: Any,
    key: Any,
    value: Any,
    output: Any,
) -> None:
    tensors = {
        "query": query,
        "key": key,
        "value": value,
        "output": output,
    }
    counts = {
        name: _tensor_row_counts(tensor) if hasattr(tensor, "shape") else None for name, tensor in tensors.items()
    }
    selected_rows: set[int] = set()
    for item_counts in counts.values():
        if item_counts is not None:
            _add_nonfinite_rows(selected_rows, item_counts)

    rows = []
    for row_idx in sorted(selected_rows):
        rows.append(
            {
                "row_index": row_idx,
                **{
                    name: _row_count_at(item_counts, row_idx) if item_counts else None
                    for name, item_counts in counts.items()
                },
            }
        )

    _write_jsonl(
        "attention_ops",
        {
            "schema": "prime_rl.vllm_attention_op_trace.v1",
            "created_unix": time.time(),
            "pid": os.getpid(),
            "layer_name": layer_name,
            "query": _tensor_meta(query),
            "key": _tensor_meta(key),
            "value": _tensor_meta(value),
            "output": _tensor_meta(output),
            "model_runner_context": _current_model_runner_context(),
            "rows": rows,
        },
    )


def _trace_mamba_op_event(
    *,
    op_name: str,
    before: dict[str, Any],
    after: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    _write_jsonl(
        "mamba_mixer2_ops",
        {
            "schema": "prime_rl.vllm_mamba_mixer2_op_trace.v1",
            "created_unix": time.time(),
            "pid": os.getpid(),
            "op_name": op_name,
            "layer_context": _mamba_layer_context_summary(),
            "model_runner_context": _current_model_runner_context(),
            "before": before,
            "after": after,
            "metadata": metadata,
        },
    )


def _mamba_layer_context_summary() -> dict[str, Any] | None:
    context = getattr(_MAMBA_TRACE_LOCAL, "context", None)
    if not isinstance(context, dict):
        return None
    return {key: value for key, value in context.items() if key != "projected_states"}


def _current_model_runner_context() -> dict[str, Any] | None:
    return getattr(_MODEL_RUNNER_TRACE_LOCAL, "context", None)


def _mamba_trace_context_allows_event() -> bool:
    if not _mamba_op_trace_requires_request_ids():
        return True

    context = _current_model_runner_context()
    if context is None:
        return False
    return bool(context.get("req_id_count", 0))


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
    if _scheduler_compact_trace_enabled():
        _trace_scheduler_shape(scheduler, scheduler_output)
        return

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


def _trace_scheduler_shape(scheduler: Any, scheduler_output: Any) -> None:
    scheduled_tokens = list((scheduler_output.num_scheduled_tokens or {}).values())
    scheduled_req_count = len(scheduled_tokens)
    uniform_decode = scheduled_req_count > 0 and all(value == 1 for value in scheduled_tokens)
    _write_jsonl(
        "scheduler_shapes",
        {
            "schema": "prime_rl.vllm_scheduler_shape_trace.v1",
            "created_unix": time.time(),
            "pid": os.getpid(),
            "running_count": len(getattr(scheduler, "running", [])),
            "waiting_count": len(getattr(scheduler, "waiting", [])),
            "skipped_waiting_count": len(getattr(scheduler, "skipped_waiting", [])),
            "scheduled_req_count": scheduled_req_count,
            "scheduled_cached_req_count": len(getattr(scheduler_output.scheduled_cached_reqs, "req_ids", [])),
            "scheduled_new_req_count": len(scheduler_output.scheduled_new_reqs),
            "total_num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
            "min_scheduled_tokens": min(scheduled_tokens) if scheduled_tokens else None,
            "max_scheduled_tokens": max(scheduled_tokens) if scheduled_tokens else None,
            "uniform_decode": uniform_decode,
            "finished_req_count": len(scheduler_output.finished_req_ids),
            "preempted_req_count": len(scheduler_output.preempted_req_ids or []),
            "new_block_ids_to_zero_count": len(scheduler_output.new_block_ids_to_zero or []),
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
    batch_execution: dict[str, Any] | None,
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
            "batch_execution": batch_execution,
            "model_runner_context": _current_model_runner_context(),
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
    batch_execution: dict[str, Any] | None,
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
            "batch_execution": batch_execution,
            "model_runner_context": _current_model_runner_context(),
            "num_rows": len(req_ids),
            "rows": rows,
        },
    )


def _next_model_runner_execute_index(model_runner: Any) -> int:
    execute_index = int(getattr(model_runner, "_prime_rl_nan_trace_execute_index", 0)) + 1
    model_runner._prime_rl_nan_trace_execute_index = execute_index
    return execute_index


def _model_runner_execution_context(model_runner: Any, *, execute_index: int | None = None) -> dict[str, Any]:
    input_batch = getattr(model_runner, "input_batch", None)
    req_ids = list(getattr(input_batch, "req_ids", []) or []) if input_batch is not None else []
    num_reqs = getattr(input_batch, "num_reqs", len(req_ids)) if input_batch is not None else len(req_ids)
    num_reqs_int = _safe_int(num_reqs, len(req_ids)) or len(req_ids)
    return {
        "execute_index": execute_index,
        "req_id_count": len(req_ids),
        "req_ids_head": req_ids[:16],
        "req_ids_tail": req_ids[-16:] if len(req_ids) > 16 else [],
        "num_reqs": _json_safe(num_reqs),
        "input_batch": _input_batch_summary(input_batch, num_reqs_int),
        "batch_execution": getattr(model_runner, "_prime_rl_nan_trace_batch_execution", None),
    }


def _batch_execution_summary(result: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    names = (
        "num_tokens",
        "num_reqs",
        "num_scheduled_tokens_np",
        "max_num_scheduled_tokens",
        "use_cascade_attn",
        "allow_microbatching",
        "force_eager",
        "force_uniform_decode",
        "force_has_lora",
        "force_num_active_loras",
        "num_encoder_reqs",
    )
    inputs = {name: args[idx] for idx, name in enumerate(names) if idx < len(args)}
    inputs.update(kwargs)

    try:
        cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, cudagraph_stats = result
    except Exception:
        return {
            "result": _json_safe(result),
            "inputs": _batch_execution_inputs_summary(inputs),
        }

    return {
        "inputs": _batch_execution_inputs_summary(inputs),
        "cudagraph_mode": _enum_summary(cudagraph_mode),
        "batch_descriptor": _object_attr_summary(
            batch_desc,
            ("num_tokens", "num_reqs", "uniform", "has_lora", "num_active_loras"),
        ),
        "should_ubatch": should_ubatch,
        "num_tokens_across_dp": _tensor_or_sequence_summary(num_tokens_across_dp),
        "cudagraph_stats": _object_attr_summary(
            cudagraph_stats,
            ("num_unpadded_tokens", "num_padded_tokens", "num_paddings", "runtime_mode"),
        ),
    }


def _batch_execution_inputs_summary(inputs: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in inputs.items():
        if key == "num_scheduled_tokens_np":
            summary[key] = _sequence_summary(value)
        else:
            summary[key] = _json_safe(value)
    return summary


def _trace_model_runner_batch_state(model_runner: Any, state: Any, batch_execution: dict[str, Any] | None) -> None:
    input_batch = getattr(model_runner, "input_batch", None)
    scheduler_output = getattr(state, "scheduler_output", None)
    req_ids = list(getattr(input_batch, "req_ids", []) or [])
    num_reqs = getattr(input_batch, "num_reqs", len(req_ids)) if input_batch is not None else len(req_ids)
    try:
        num_reqs = int(num_reqs)
    except Exception:
        num_reqs = len(req_ids)
    req_ids = req_ids[:num_reqs]

    scheduled_tokens_by_req = {}
    if scheduler_output is not None:
        scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
        scheduled_tokens_by_req = {req_id: scheduled_tokens.get(req_id) for req_id in req_ids}

    record = {
        "schema": "prime_rl.vllm_model_runner_batch_trace.v1",
        "created_unix": time.time(),
        "pid": os.getpid(),
        "num_reqs": num_reqs,
        "req_ids": req_ids,
        "external_request_id_guesses": [_external_request_id_from_internal(req_id) for req_id in req_ids],
        "batch_execution": batch_execution,
        "state_cudagraph_stats": _object_attr_summary(
            getattr(state, "cudagraph_stats", None),
            ("num_unpadded_tokens", "num_padded_tokens", "num_paddings", "runtime_mode"),
        ),
        "scheduler_output": _model_runner_scheduler_output_summary(scheduler_output, scheduled_tokens_by_req),
        "input_batch": _input_batch_summary(input_batch, num_reqs),
        "logits": _tensor_meta(getattr(state, "logits", None)),
        "sample_hidden_states": _tensor_meta(getattr(state, "sample_hidden_states", None)),
        "spec_decode": getattr(state, "spec_decode_metadata", None) is not None,
    }
    if _model_runner_batch_gpu_trace_enabled():
        record["gpu_inputs"] = _model_runner_gpu_input_summary(model_runner, state)

    _write_jsonl("model_runner_batches", record)


def _trace_model_runner_target_batch_state(
    model_runner: Any,
    state: Any,
    batch_execution: dict[str, Any] | None,
) -> None:
    input_batch = getattr(model_runner, "input_batch", None)
    if input_batch is None or not _target_batch_trace_matches(batch_execution):
        return
    if not _target_batch_trace_take_record():
        return

    req_ids = list(getattr(input_batch, "req_ids", []) or [])
    num_reqs = _safe_int(getattr(input_batch, "num_reqs", len(req_ids)), len(req_ids))
    req_ids = req_ids[:num_reqs]
    query_start_locs = _sequence_values(getattr(input_batch, "query_start_loc_np", None))
    if not query_start_locs:
        query_start_locs = _sequence_values(getattr(input_batch, "query_start_loc", None))

    record = {
        "schema": "prime_rl.vllm_model_runner_target_batch_trace.v1",
        "created_unix": time.time(),
        "pid": os.getpid(),
        "model_runner_context": _current_model_runner_context(),
        "num_reqs": num_reqs,
        "batch_execution": batch_execution,
        "state_cudagraph_stats": _object_attr_summary(
            getattr(state, "cudagraph_stats", None),
            ("num_unpadded_tokens", "num_padded_tokens", "num_paddings", "runtime_mode"),
        ),
        "input_batch": _input_batch_summary(input_batch, num_reqs),
        "rows": _target_batch_rows_summary(input_batch, req_ids, query_start_locs),
        "slot_mappings_by_row": _slot_mappings_rows_summary(
            getattr(state, "slot_mappings", None),
            query_start_locs,
            num_reqs,
        ),
        "logits": _tensor_meta(getattr(state, "logits", None)),
        "sample_hidden_states": _tensor_meta(getattr(state, "sample_hidden_states", None)),
        "spec_decode": getattr(state, "spec_decode_metadata", None) is not None,
    }
    _write_jsonl("model_runner_target_batches", record)


def _target_batch_trace_matches(batch_execution: dict[str, Any] | None) -> bool:
    if not isinstance(batch_execution, dict):
        return False

    mode_name = _batch_execution_mode_name(batch_execution)
    if _target_batch_trace_only_full() and "FULL" not in mode_name:
        return False
    if _target_batch_trace_only_uniform() and not _batch_execution_uniform_decode(batch_execution):
        return False

    actual_num_reqs = _safe_int(
        (batch_execution.get("inputs") or {}).get("num_reqs"),
        None,
    )
    padded_num_reqs = _safe_int(
        (batch_execution.get("batch_descriptor") or {}).get("num_reqs"),
        None,
    )
    return _value_in_range(
        actual_num_reqs,
        low=_env_int("PRIME_RL_VLLM_NAN_TRACE_TARGET_BATCH_ACTUAL_MIN", 192),
        high=_env_int("PRIME_RL_VLLM_NAN_TRACE_TARGET_BATCH_ACTUAL_MAX", 224),
    ) or _value_in_range(
        padded_num_reqs,
        low=_env_int("PRIME_RL_VLLM_NAN_TRACE_TARGET_BATCH_PADDED_MIN", 208),
        high=_env_int("PRIME_RL_VLLM_NAN_TRACE_TARGET_BATCH_PADDED_MAX", 224),
    )


def _target_batch_trace_take_record() -> bool:
    limit = _env_int("PRIME_RL_VLLM_NAN_TRACE_TARGET_BATCH_LIMIT", 512)
    if limit is not None and limit <= 0:
        return False
    pid = os.getpid()
    with _WRITE_LOCK:
        count = _TARGET_BATCH_TRACE_COUNT.get(pid, 0)
        if limit is not None and count >= limit:
            return False
        _TARGET_BATCH_TRACE_COUNT[pid] = count + 1
    return True


def _batch_execution_mode_name(batch_execution: dict[str, Any]) -> str:
    mode = batch_execution.get("cudagraph_mode")
    if not isinstance(mode, dict):
        return str(mode)
    return str(mode.get("name") or mode.get("string") or mode.get("value") or "")


def _batch_execution_uniform_decode(batch_execution: dict[str, Any]) -> bool:
    descriptor = batch_execution.get("batch_descriptor") or {}
    if descriptor.get("uniform") is True:
        return True
    scheduled = (batch_execution.get("inputs") or {}).get("num_scheduled_tokens_np") or {}
    return scheduled.get("min") == 1 and scheduled.get("max") == 1


def _value_in_range(value: int | None, *, low: int | None, high: int | None) -> bool:
    if value is None:
        return False
    if low is not None and value < low:
        return False
    if high is not None and value > high:
        return False
    return True


def _model_runner_scheduler_output_summary(
    scheduler_output: Any | None,
    scheduled_tokens_by_req: dict[str, Any],
) -> dict[str, Any] | None:
    if scheduler_output is None:
        return None
    return {
        "total_num_scheduled_tokens": getattr(scheduler_output, "total_num_scheduled_tokens", None),
        "num_common_prefix_blocks": getattr(scheduler_output, "num_common_prefix_blocks", None),
        "finished_req_ids": sorted(getattr(scheduler_output, "finished_req_ids", []) or []),
        "preempted_req_ids": sorted(getattr(scheduler_output, "preempted_req_ids", []) or []),
        "num_scheduled_tokens_by_active_req": scheduled_tokens_by_req,
    }


def _input_batch_summary(input_batch: Any | None, num_reqs: int) -> dict[str, Any] | None:
    if input_batch is None:
        return None
    return {
        "max_num_reqs": getattr(input_batch, "max_num_reqs", None),
        "max_model_len": getattr(input_batch, "max_model_len", None),
        "max_num_batched_tokens": getattr(input_batch, "max_num_batched_tokens", None),
        "num_prompt_tokens": _sequence_summary(getattr(input_batch, "num_prompt_tokens", None), count=num_reqs),
        "num_tokens_no_spec": _sequence_summary(getattr(input_batch, "num_tokens_no_spec", None), count=num_reqs),
        "num_computed_tokens_cpu": _sequence_summary(
            getattr(input_batch, "num_computed_tokens_cpu", None),
            count=num_reqs,
        ),
        "num_accepted_tokens_cpu": _sequence_summary(
            getattr(input_batch, "num_accepted_tokens_cpu", None),
            count=num_reqs,
        ),
    }


def _model_runner_gpu_input_summary(model_runner: Any, state: Any) -> dict[str, Any]:
    return {
        "input_ids": _tensor_or_sequence_summary(getattr(model_runner, "input_ids", None)),
        "positions": _tensor_or_sequence_summary(getattr(model_runner, "positions", None)),
        "slot_mappings": _slot_mappings_summary(getattr(state, "slot_mappings", None)),
    }


def _slot_mappings_summary(slot_mappings: Any) -> dict[str, Any] | None:
    if slot_mappings is None:
        return None
    if isinstance(slot_mappings, list):
        return {
            "type": "list",
            "count": len(slot_mappings),
            "items": [_slot_mappings_summary(item) for item in slot_mappings[:4]],
            "truncated": len(slot_mappings) > 4,
        }
    if not isinstance(slot_mappings, dict):
        return {
            "type": type(slot_mappings).__name__,
            "value": _tensor_or_sequence_summary(slot_mappings),
        }

    groups: dict[str, dict[str, Any]] = {}
    for layer_name, tensor in slot_mappings.items():
        key = _tensor_identity(tensor)
        group = groups.setdefault(
            key,
            {
                "layer_count": 0,
                "layer_names": [],
                "tensor": _tensor_or_sequence_summary(tensor),
            },
        )
        group["layer_count"] += 1
        if len(group["layer_names"]) < 8:
            group["layer_names"].append(layer_name)

    return {
        "type": "dict",
        "layer_count": len(slot_mappings),
        "unique_tensor_count": len(groups),
        "groups": list(groups.values())[:8],
        "truncated": len(groups) > 8,
    }


def _target_batch_rows_summary(
    input_batch: Any,
    req_ids: list[str],
    query_start_locs: list[Any],
) -> list[dict[str, Any]]:
    num_reqs = len(req_ids)
    scheduled_tokens = _sequence_values(getattr(input_batch, "num_scheduled_tokens", None), count=num_reqs)
    prompt_tokens = _sequence_values(getattr(input_batch, "num_prompt_tokens", None), count=num_reqs)
    tokens_no_spec = _sequence_values(getattr(input_batch, "num_tokens_no_spec", None), count=num_reqs)
    computed_tokens = _sequence_values(getattr(input_batch, "num_computed_tokens_cpu", None), count=num_reqs)
    accepted_tokens = _sequence_values(getattr(input_batch, "num_accepted_tokens_cpu", None), count=num_reqs)
    seq_lens = _sequence_values(getattr(input_batch, "seq_lens", None), count=num_reqs)
    seq_lens_upper = _sequence_values(getattr(input_batch, "seq_lens_cpu_upper_bound", None), count=num_reqs)
    is_prefilling = _sequence_values(getattr(input_batch, "is_prefilling_np", None), count=num_reqs)
    idx_mapping = _sequence_values(getattr(input_batch, "idx_mapping_np", None), count=num_reqs)
    if not idx_mapping:
        idx_mapping = _sequence_values(getattr(input_batch, "idx_mapping", None), count=num_reqs)
    expanded_local_pos = _sequence_values(getattr(input_batch, "expanded_local_pos", None), count=num_reqs)

    rows = []
    for row_idx, req_id in enumerate(req_ids):
        query_start = _row_value(query_start_locs, row_idx)
        query_end = _row_value(query_start_locs, row_idx + 1)
        rows.append(
            {
                "row_index": row_idx,
                "request_id": req_id,
                "external_request_id_guess": _external_request_id_from_internal(req_id),
                "num_scheduled_tokens": _row_value(scheduled_tokens, row_idx),
                "num_prompt_tokens": _row_value(prompt_tokens, row_idx),
                "num_tokens_no_spec": _row_value(tokens_no_spec, row_idx),
                "num_computed_tokens_cpu": _row_value(computed_tokens, row_idx),
                "num_accepted_tokens_cpu": _row_value(accepted_tokens, row_idx),
                "seq_len": _row_value(seq_lens, row_idx),
                "seq_len_cpu_upper_bound": _row_value(seq_lens_upper, row_idx),
                "is_prefilling": _row_value(is_prefilling, row_idx),
                "idx_mapping": _row_value(idx_mapping, row_idx),
                "expanded_local_pos": _row_value(expanded_local_pos, row_idx),
                "query_start": query_start,
                "query_end": query_end,
                "input_ids": _vector_slice_summary(
                    getattr(input_batch, "input_ids", None),
                    query_start,
                    query_end,
                ),
                "positions": _vector_slice_summary(
                    getattr(input_batch, "positions", None),
                    query_start,
                    query_end,
                ),
            }
        )
    return rows


def _slot_mappings_rows_summary(
    slot_mappings: Any,
    query_start_locs: list[Any],
    num_reqs: int,
) -> dict[str, Any] | None:
    if slot_mappings is None:
        return None
    token_indices = [_safe_int(value, None) for value in query_start_locs[:num_reqs]]
    if isinstance(slot_mappings, list):
        return {
            "type": "list",
            "count": len(slot_mappings),
            "items": [_slot_mappings_rows_summary(item, query_start_locs, num_reqs) for item in slot_mappings[:4]],
            "truncated": len(slot_mappings) > 4,
        }
    if not isinstance(slot_mappings, dict):
        return {
            "type": type(slot_mappings).__name__,
            "values_by_row": _indexed_values(slot_mappings, token_indices),
        }

    groups: dict[str, dict[str, Any]] = {}
    for layer_name, tensor in slot_mappings.items():
        key = _tensor_identity(tensor)
        group = groups.setdefault(
            key,
            {
                "layer_count": 0,
                "layer_names": [],
                "tensor": _tensor_or_sequence_summary(tensor),
                "values_by_row": _indexed_values(tensor, token_indices),
            },
        )
        group["layer_count"] += 1
        if len(group["layer_names"]) < 8:
            group["layer_names"].append(layer_name)

    return {
        "type": "dict",
        "layer_count": len(slot_mappings),
        "unique_tensor_count": len(groups),
        "groups": list(groups.values())[:8],
        "truncated": len(groups) > 8,
    }


def _vector_slice_summary(vector: Any, start: Any, end: Any, *, limit: int = 8) -> dict[str, Any] | None:
    start_int = _safe_int(start, None)
    end_int = _safe_int(end, None)
    if vector is None or start_int is None or end_int is None:
        return None
    if end_int < start_int:
        return {"start": start_int, "end": end_int, "error": "end_before_start"}
    try:
        sliced = vector[start_int:end_int]
        values = _sequence_values(sliced)
    except Exception:
        return {"start": start_int, "end": end_int, "error": "slice_failed"}
    return {
        "start": start_int,
        "end": end_int,
        "count": len(values),
        "head": values[:limit],
        "tail": values[-limit:] if len(values) > limit else [],
    }


def _indexed_values(vector: Any, indices: list[int | None]) -> dict[str, Any] | None:
    if vector is None:
        return None
    values = []
    for index in indices:
        if index is None:
            values.append(None)
            continue
        try:
            item = vector[index]
            if hasattr(item, "detach"):
                item = item.detach().cpu()
            if hasattr(item, "item"):
                item = item.item()
            values.append(_json_safe(item))
        except Exception:
            values.append("index_failed")
    return {
        "count": len(values),
        "values": values,
    }


def _trace_nemotron_h_layer_output(
    *,
    layer_idx: int | None,
    layer_type: str,
    hidden_in: Any,
    residual_in: Any,
    output: Any,
) -> None:
    if not _mamba_trace_context_allows_event():
        return
    if not isinstance(output, tuple) or len(output) < 2:
        return

    hidden_out, residual_out = output[0], output[1]
    hidden_out_bad = _tensor_has_nonfinite(hidden_out)
    residual_out_bad = _tensor_has_nonfinite(residual_out)
    if not (hidden_out_bad or residual_out_bad):
        return

    hidden_out_counts = _tensor_row_counts(hidden_out)
    residual_out_counts = _tensor_row_counts(residual_out) if residual_out is not None else None
    hidden_in_counts = _tensor_row_counts(hidden_in) if hidden_in is not None else None
    residual_in_counts = _tensor_row_counts(residual_in) if residual_in is not None else None

    selected_rows: set[int] = set()
    _add_nonfinite_rows(selected_rows, hidden_out_counts)
    if residual_out_counts is not None:
        _add_nonfinite_rows(selected_rows, residual_out_counts)

    rows = []
    for row_idx in sorted(selected_rows):
        rows.append(
            {
                "row_index": row_idx,
                "hidden_in": _row_count_at(hidden_in_counts, row_idx) if hidden_in_counts else None,
                "residual_in": _row_count_at(residual_in_counts, row_idx) if residual_in_counts else None,
                "hidden_out": _row_count_at(hidden_out_counts, row_idx),
                "residual_out": _row_count_at(residual_out_counts, row_idx) if residual_out_counts else None,
            }
        )

    _write_jsonl(
        "nemotron_h_layer_states",
        {
            "schema": "prime_rl.vllm_nemotron_h_layer_trace.v1",
            "created_unix": time.time(),
            "pid": os.getpid(),
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "hidden_in": _tensor_meta(hidden_in),
            "residual_in": _tensor_meta(residual_in),
            "hidden_out": _tensor_meta(hidden_out),
            "residual_out": _tensor_meta(residual_out),
            "model_runner_context": _current_model_runner_context(),
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


def _tensor_has_nonfinite(tensor: Any) -> bool:
    if tensor is None or not hasattr(tensor, "isfinite"):
        return False
    if _torch_is_compiling():
        return False
    return not bool(tensor.isfinite().all().detach().cpu().item())


def _add_nonfinite_rows(selected_rows: set[int], counts: dict[str, list[int]]) -> None:
    for key in ("nan_counts", "posinf_counts", "neginf_counts"):
        for row_idx, count in enumerate(counts[key]):
            if count:
                selected_rows.add(row_idx)


def _tensor_meta(tensor: Any) -> dict[str, Any] | None:
    if tensor is None or not hasattr(tensor, "shape"):
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(getattr(tensor, "dtype", "")),
    }


def _tensor_total_counts(tensor: Any) -> dict[str, Any] | None:
    meta = _tensor_meta(tensor)
    if meta is None or not hasattr(tensor, "isfinite"):
        return meta
    if _torch_is_compiling():
        return meta
    try:
        finite = tensor.isfinite()
        return {
            **meta,
            "numel": int(tensor.numel()),
            "finite": int(finite.sum().detach().cpu().item()),
            "nan": int(tensor.isnan().sum().detach().cpu().item()),
            "posinf": int(tensor.isposinf().sum().detach().cpu().item()),
            "neginf": int(tensor.isneginf().sum().detach().cpu().item()),
        }
    except Exception:
        return meta


def _tensor_tree_summary(value: Any, *, max_tensors: int, max_numel: int) -> dict[str, Any]:
    tensors: list[dict[str, Any]] = []
    seen: set[int] = set()
    truncated = False

    def walk(item: Any, path: str, depth: int) -> None:
        nonlocal truncated
        if item is None or depth > 8:
            return
        if len(tensors) >= max_tensors:
            truncated = True
            return

        item_id = id(item)
        if item_id in seen:
            return
        seen.add(item_id)

        if hasattr(item, "shape") and hasattr(item, "numel"):
            tensors.append(_tensor_finiteness_summary(item, path=path, max_numel=max_numel))
            return

        if isinstance(item, dict):
            for key, child in list(item.items())[:64]:
                walk(child, f"{path}.{key}", depth + 1)
            if len(item) > 64:
                truncated = True
            return

        if isinstance(item, (list, tuple)):
            for idx, child in enumerate(item[:64]):
                walk(child, f"{path}[{idx}]", depth + 1)
            if len(item) > 64:
                truncated = True
            return

        intermediate_tensors = getattr(item, "tensors", None)
        if isinstance(intermediate_tensors, dict):
            walk(intermediate_tensors, f"{path}.tensors", depth + 1)

    walk(value, "$", 0)
    nonfinite_tensor_count = sum(1 for tensor in tensors if tensor.get("nonfinite"))
    return {
        "tensor_count": len(tensors),
        "nonfinite_tensor_count": nonfinite_tensor_count,
        "truncated": truncated,
        "tensors": tensors,
    }


def _tensor_finiteness_summary(tensor: Any, *, path: str, max_numel: int) -> dict[str, Any]:
    meta = {
        "path": path,
        "shape": list(getattr(tensor, "shape", [])),
        "dtype": str(getattr(tensor, "dtype", "")),
        "device": str(getattr(tensor, "device", "")),
    }
    try:
        numel = int(tensor.numel())
    except Exception:
        return meta

    meta["numel"] = numel
    if numel == 0:
        meta["nonfinite"] = False
        return meta
    if numel > max_numel:
        meta["skipped_finiteness"] = f"numel>{max_numel}"
        return meta

    try:
        if hasattr(tensor, "is_floating_point") and not tensor.is_floating_point():
            meta["nonfinite"] = False
            return meta
    except Exception:
        pass

    if not hasattr(tensor, "isfinite") or _torch_is_compiling():
        return meta

    try:
        finite = tensor.isfinite()
        nan_count = int(tensor.isnan().sum().detach().cpu().item())
        posinf_count = int(tensor.isposinf().sum().detach().cpu().item())
        neginf_count = int(tensor.isneginf().sum().detach().cpu().item())
        nonfinite = nan_count + posinf_count + neginf_count
        meta.update(
            {
                "finite": int(finite.sum().detach().cpu().item()),
                "nan": nan_count,
                "posinf": posinf_count,
                "neginf": neginf_count,
                "nonfinite": nonfinite > 0,
            }
        )
        if nonfinite:
            meta["bad_rows"] = _tensor_bad_rows(tensor)
    except Exception as exc:
        meta["finiteness_error"] = str(exc)
    return meta


def _tensor_bad_rows(tensor: Any, *, limit: int = 16) -> list[dict[str, Any]]:
    try:
        if len(tensor.shape) == 0:
            rows = tensor.reshape(1, 1)
        elif len(tensor.shape) == 1:
            rows = tensor.reshape(tensor.shape[0], 1)
        else:
            rows = tensor.reshape(tensor.shape[0], -1)
        bad_by_row = ~rows.isfinite().all(dim=-1)
        row_indices = bad_by_row.nonzero(as_tuple=False).reshape(-1)[:limit].detach().cpu().tolist()
        counts = _tensor_row_counts(tensor)
        return [
            {
                "row_index": int(row_idx),
                **_row_count_at(counts, int(row_idx)),
            }
            for row_idx in row_indices
        ]
    except Exception as exc:
        return [{"error": str(exc)}]


def _tensor_or_sequence_summary(value: Any, *, limit: int = 16, count: int | None = None) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "detach") and hasattr(value, "numel"):
        return _tensor_vector_summary(value, limit=limit, count=count)
    return _sequence_summary(value, limit=limit, count=count)


def _tensor_vector_summary(tensor: Any, *, limit: int = 16, count: int | None = None) -> dict[str, Any] | None:
    try:
        flat = tensor.detach().reshape(-1)
        if count is not None:
            flat = flat[:count]
        total = int(flat.numel())
        head = flat[: min(limit, total)].detach().cpu().tolist()
        tail = flat[max(0, total - limit) :].detach().cpu().tolist() if total > limit else []
        summary = {
            "shape": list(tensor.shape),
            "dtype": str(getattr(tensor, "dtype", "")),
            "count": total,
            "head": head,
            "tail": tail,
        }
        if total:
            try:
                summary["min"] = flat.min().detach().cpu().item()
                summary["max"] = flat.max().detach().cpu().item()
            except Exception:
                pass
            try:
                summary["minus_one_count"] = int((flat == -1).sum().detach().cpu().item())
            except Exception:
                pass
        return summary
    except Exception:
        return _tensor_meta(tensor)


def _sequence_values(value: Any, *, count: int | None = None) -> list[Any]:
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "reshape"):
            value = value.reshape(-1)
        if hasattr(value, "tolist"):
            values = value.tolist()
        else:
            values = list(value)
    except Exception:
        return []
    if count is not None:
        values = values[:count]
    return [_json_safe(item) for item in values]


def _sequence_summary(value: Any, *, limit: int = 16, count: int | None = None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        if hasattr(value, "tolist"):
            values = value.tolist()
        else:
            values = list(value)
    except Exception:
        return {"value": _json_safe(value)}
    if count is not None:
        values = values[:count]
    non_null = [item for item in values if item is not None]
    summary: dict[str, Any] = {
        "count": len(values),
        "head": values[:limit],
        "tail": values[-limit:] if len(values) > limit else [],
    }
    try:
        summary["min"] = min(non_null) if non_null else None
        summary["max"] = max(non_null) if non_null else None
    except Exception:
        pass
    return summary


def _enum_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "type": type(value).__name__,
        "name": getattr(value, "name", None),
        "value": getattr(value, "value", None),
        "string": str(value),
    }


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", None) or getattr(value, "value", None) or value)


def _object_attr_summary(value: Any, attrs: tuple[str, ...]) -> dict[str, Any] | None:
    if value is None:
        return None
    return {attr: _json_safe(getattr(value, attr)) for attr in attrs if hasattr(value, attr)}


def _tensor_identity(value: Any) -> str:
    try:
        return f"{value.device}:{value.data_ptr()}:{tuple(value.shape)}"
    except Exception:
        return str(id(value))


def _row_count_at(counts: dict[str, list[int]], row_idx: int) -> dict[str, int | None]:
    return {
        key.removesuffix("_counts"): values[row_idx] if row_idx < len(values) else None
        for key, values in counts.items()
    }


def _row_value(values: list[Any], row_idx: int) -> Any:
    if row_idx < 0 or row_idx >= len(values):
        return None
    return _json_safe(values[row_idx])


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        if hasattr(value, "item"):
            value = value.item()
        return int(value)
    except Exception:
        return default


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
