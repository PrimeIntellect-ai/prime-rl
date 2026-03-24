def transformers_v5_compat():
    """vLLM general plugin: patch transformers v5 config attrs that vLLM 0.16 still expects.

    Registered as a ``vllm.general_plugins`` entry-point so it runs automatically
    in every vLLM process, including spawned workers.
    """
    from transformers import Qwen3VLMoeTextConfig

    if not hasattr(Qwen3VLMoeTextConfig, "tie_word_embeddings"):
        Qwen3VLMoeTextConfig.tie_word_embeddings = False

    _patch_qwen35_lora()


def _patch_qwen35_lora():
    """Fix Qwen3.5 LoRA: align packed_modules_mapping with output_sizes.

    Qwen3.5's GDN layers use create_qkvz_proj with 4 output_sizes (q, k, v, z)
    but packed_modules_mapping only lists 2 entries, causing an IndexError
    during LoRA initialization.

    Also generalizes MergedColumnParallelLinearWithLoRA.can_replace_layer
    to accept any number of packed modules (not just 2), and generalizes
    MergedColumnParallelLinearWithShardedLoRA.slice_lora_a to handle N
    subloras instead of the hardcoded 2 (needed for fully_sharded_loras=True).

    Upstream: https://github.com/vllm-project/vllm/issues/36372
    """
    from vllm.lora.layers.column_parallel_linear import (
        MergedColumnParallelLinearWithLoRA,
        MergedColumnParallelLinearWithShardedLoRA,
    )
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLMBase,
        Qwen3_5ForConditionalGeneration,
    )

    qkvz_fix = ["in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z"]

    Qwen3_5ForCausalLMBase.packed_modules_mapping["in_proj_qkvz"] = qkvz_fix
    Qwen3_5ForConditionalGeneration.packed_modules_mapping["in_proj_qkvz"] = qkvz_fix

    from vllm.lora.layers.utils import _not_fully_sharded_can_replace

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(cls, source_layer, lora_config, packed_modules_list, model_config=None):
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear

        return type(source_layer) is MergedColumnParallelLinear and len(packed_modules_list) == len(
            source_layer.output_sizes
        )

    MergedColumnParallelLinearWithLoRA.can_replace_layer = can_replace_layer

    def slice_lora_a(self, lora_a):
        output_shard_size = self.lora_a_stacked[0].shape[2]
        output_start_idx = self.tp_rank * output_shard_size
        return [
            a[output_start_idx : output_start_idx + output_shard_size, :] if a is not None else None for a in lora_a
        ]

    MergedColumnParallelLinearWithShardedLoRA.slice_lora_a = slice_lora_a


# Monkeypatch PrometheusStatLogger to avoid NotImplementedError for LoRA in DP mode
def monkey_patch_prometheus_stat_logger_for_lora_in_dp_mode():
    from vllm.v1.metrics import loggers as vllm_metrics_loggers

    _original_prometheus_stat_logger_init = vllm_metrics_loggers.PrometheusStatLogger.__init__

    def _patched_prometheus_stat_logger_init(self, vllm_config, engine_indexes=None):
        """Patched init that temporarily disables lora_config to skip the DP mode check."""
        original_lora_config = vllm_config.lora_config
        vllm_config.lora_config = None
        try:
            _original_prometheus_stat_logger_init(self, vllm_config, engine_indexes)
        finally:
            vllm_config.lora_config = original_lora_config
        # Re-initialize LoRA metrics if needed (after the DP check is bypassed)
        if original_lora_config is not None:
            self.labelname_max_lora = "max_lora"
            self.labelname_waiting_lora_adapters = "waiting_lora_adapters"
            self.labelname_running_lora_adapters = "running_lora_adapters"
            self.max_lora = original_lora_config.max_loras
            self.gauge_lora_info = vllm_metrics_loggers.PrometheusStatLogger._gauge_cls(
                name="vllm:lora_requests_info",
                documentation="Running stats on lora requests.",
                multiprocess_mode="sum",
                labelnames=[
                    self.labelname_max_lora,
                    self.labelname_waiting_lora_adapters,
                    self.labelname_running_lora_adapters,
                ],
            )

    vllm_metrics_loggers.PrometheusStatLogger.__init__ = _patched_prometheus_stat_logger_init


# Monkeypatch LoadLoRAAdapter to allow loading the same adapter multiple times
def monkey_patch_load_lora_adapter():
    from http import HTTPStatus

    from vllm.entrypoints.openai.engine.protocol import ErrorResponse
    from vllm.entrypoints.openai.models.serving import (
        OpenAIServingModels,
        create_error_response,
    )
    from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest
    from vllm.logger import init_logger
    from vllm.lora.request import LoRARequest

    logger = init_logger(__name__)

    async def _patched_load_lora_adapter(
        self: OpenAIServingModels, request: LoadLoRAAdapterRequest, base_model_name: str | None = None
    ) -> ErrorResponse | str:
        lora_name = request.lora_name

        # Ensure atomicity based on the lora name
        async with self.lora_resolver_lock[lora_name]:
            lora_path = request.lora_path
            ## START PATCHED CODE
            if lora_name in self.lora_requests:
                lora_request = self.lora_requests[lora_name]
                lora_request.lora_path = lora_path
            else:
                unique_id = self.lora_id_counter.inc(1)
                lora_request = LoRARequest(lora_name=lora_name, lora_int_id=unique_id, lora_path=lora_path)
            ## END PATCHED CODE
            if base_model_name is not None and self.is_base_model(base_model_name):
                lora_request.base_model_name = base_model_name

            # Validate that the adapter can be loaded into the engine
            # This will also preload it for incoming requests
            try:
                await self.engine_client.add_lora(lora_request)
            except Exception as e:
                error_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                if "No adapter found" in str(e):
                    error_type = "NotFoundError"
                    status_code = HTTPStatus.NOT_FOUND

                return create_error_response(message=str(e), err_type=error_type, status_code=status_code)

            self.lora_requests[lora_name] = lora_request
            logger.info("Loaded new LoRA adapter: name '%s', path '%s'", lora_name, lora_path)
            return f"Success: LoRA adapter '{lora_name}' added successfully."

    OpenAIServingModels.load_lora_adapter = _patched_load_lora_adapter


# Monkeypatch LRUCacheWorkerLoRAManager to allow loading adapter inplace without doing it every request
def monkey_patch_LRUCacheWorkerLoRAManager():
    from vllm.lora.worker_manager import LoRARequest, LRUCacheLoRAModelManager, LRUCacheWorkerLoRAManager

    # The dunder is intended. It's a private method that we're patching.
    def _patched__apply_adapters(self: LRUCacheWorkerLoRAManager, lora_requests: set[LoRARequest]) -> None:
        loras_map = {lora_request.lora_int_id: lora_request for lora_request in lora_requests if lora_request}
        if len(loras_map) > self._adapter_manager.lora_slots:
            raise RuntimeError(
                f"Number of requested LoRAs ({len(loras_map)}) is greater "
                "than the number of GPU LoRA slots "
                f"({self._adapter_manager.lora_slots})."
            )
        for lora in loras_map.values():
            ## START PATCHED CODE
            self.add_adapter(lora, force_load=False)
            ## END PATCHED CODE

    def _patched_add_adapter(
        self: LRUCacheWorkerLoRAManager, lora_request: LoRARequest, force_load: bool = True
    ) -> bool:
        # Note that this method is not thread-safe. It may be invoked multiple
        # times for the same adapter when using multiple API servers.
        # This is ok because it's currently only called from
        # the single-threaded core engine loop.

        ## START PATCHED CODE
        if lora_request.lora_int_id not in self.list_adapters() or force_load:
            ## END PATCHED CODE
            # Load the new adapter first to ensure it is actually valid, before
            # evicting any existing adapters.
            # This may cause the # of loaded lora adapters to very temporarily
            # exceed `--max-cpu-loras`.
            lora = self._load_adapter(lora_request)
            ## START PATCHED CODE
            self._adapter_manager.remove_adapter(lora.id)
            ## END PATCHED CODE

            # Loading succeeded, now check if we will exceed cache capacity and
            # evict if the oldest adapter if so
            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                assert isinstance(self._adapter_manager, LRUCacheLoRAModelManager)
                self._adapter_manager.remove_oldest_adapter()
            # Then add the new adapter to the cache
            loaded = self._adapter_manager.add_adapter(lora)
        else:
            # If the lora is already loaded, just touch it to
            # update its position in the caches
            loaded = self._adapter_manager.get_adapter(lora_request.lora_int_id) is not None
        self._adapter_manager.activate_adapter(lora_request.lora_int_id)
        return loaded

    LRUCacheWorkerLoRAManager._apply_adapters = _patched__apply_adapters
    LRUCacheWorkerLoRAManager.add_adapter = _patched_add_adapter


# Monkeypatch TokenizeParams to fix overly conservative validation
def monkey_patch_tokenize_params_validation():
    """
    Patch TokenizeParams validation to only reject requests where the prompt
    itself exceeds max_model_len, not where prompt + max_tokens > max_model_len.

    Original behavior:
        - Rejects if prompt_len > (max_model_len - max_tokens)

    Patched behavior:
        - Only rejects if prompt_len > max_model_len
        - Lets the engine naturally cap generation at max_model_len
    """
    from vllm.exceptions import VLLMValidationError
    from vllm.renderers.params import TokenizeParams

    def _patched_token_len_check(self, tokenizer, tokens):
        """Only validate that prompt fits in max_model_len, not prompt+max_tokens"""
        if self.max_total_tokens is not None and len(tokens) > self.max_total_tokens:
            raise VLLMValidationError(
                f"The prompt is {len(tokens)} tokens, which exceeds the "
                f"model's maximum context length of {self.max_total_tokens} tokens. "
                f"Please reduce the length of the input prompt.",
                parameter="input_tokens",
                value=len(tokens),
            )
        return tokens

    def _patched_text_len_check(self, tokenizer, text):
        """Only validate text length against max_model_len, not max_input_tokens"""
        if self.max_total_tokens is None or tokenizer is None:
            return text

        if self.truncate_prompt_tokens is None:
            max_chars = self.max_total_tokens * tokenizer.max_chars_per_token
            if len(text) > max_chars:
                raise VLLMValidationError(
                    f"You passed {len(text)} input characters. "
                    f"However, the model's context length is only "
                    f"{self.max_total_tokens} tokens "
                    f"(at most {max_chars} characters). "
                    f"Please reduce the length of the input prompt.",
                    parameter="input_text",
                    value=len(text),
                )
        return text

    def _patched_get_encode_kwargs(self):
        """Use max_total_tokens (max_model_len) instead of max_input_tokens for HF tokenizer truncation.

        The original uses max_input_tokens (= max_model_len - max_tokens) + 1, which causes HuggingFace's
        tokenizer.encode() to left-truncate prompts before _token_len_check even runs.
        """
        max_length = self.truncate_prompt_tokens
        if max_length is not None and max_length < 0:
            max_length = self.max_total_tokens
        elif max_length is None and self.max_total_tokens is not None:
            max_length = self.max_total_tokens + 1

        return dict(
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=self.add_special_tokens,
        )

    TokenizeParams._token_len_check = _patched_token_len_check
    TokenizeParams._text_len_check = _patched_text_len_check
    TokenizeParams.get_encode_kwargs = _patched_get_encode_kwargs


def monkey_patch_hermes_tool_parser_thread_safety():
    """Patch Hermes2ProToolParser to cache tokenizer encode/decode results.

    The original __init__ calls tokenizer.encode() and tokenizer.decode() on
    every instantiation. Under concurrent load, the shared HuggingFace tokenizer's
    Rust backend panics with ``RuntimeError: Already borrowed`` because multiple
    threads mutably borrow the same internal state simultaneously.

    Fix: run the first __init__ (which calls encode/decode) under a lock, cache
    the results, and reuse them for all subsequent instantiations without ever
    touching the tokenizer again.
    """
    import threading

    import regex as re
    from vllm.tool_parsers.abstract_tool_parser import ToolParser
    from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

    _original_init = Hermes2ProToolParser.__init__
    _cache: dict[int, dict] = {}
    _lock = threading.Lock()

    def _patched_init(self, tokenizer):
        from vllm.tokenizers.mistral import MistralTokenizer

        # Resolve the actual tokenizer that __init__ will use for encode/decode
        actual_tokenizer = tokenizer.tokenizer if isinstance(tokenizer, MistralTokenizer) else tokenizer
        key = id(actual_tokenizer)

        if key in _cache:
            # Fast path: skip encode/decode entirely, set up instance from cache
            ToolParser.__init__(self, tokenizer)
            if isinstance(tokenizer, MistralTokenizer):
                self.model_tokenizer = tokenizer.tokenizer
            self.current_tool_name_sent = False
            self.prev_tool_call_arr = []
            self.current_tool_id = -1
            self.streamed_args_for_tool = []
            self.tool_call_start_token = "<tool_call>"
            self.tool_call_end_token = "</tool_call>"
            self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
            self.scratch_pad_regex = re.compile(r"<scratch_pad>(.*?)</scratch_pad>", re.DOTALL)
            cached = _cache[key]
            self.tool_call_start_token_ids = cached["start_ids"]
            self.tool_call_end_token_ids = cached["end_ids"]
            self.tool_call_start_token_array = cached["start_array"]
            self.tool_call_end_token_array = cached["end_array"]
            self.buffered_delta_text = ""
            return

        # Slow path: first instantiation for this tokenizer, run under lock
        with _lock:
            if key in _cache:
                # Another thread populated it while we waited
                _patched_init(self, tokenizer)
                return
            _original_init(self, tokenizer)
            _cache[key] = {
                "start_ids": self.tool_call_start_token_ids,
                "end_ids": self.tool_call_end_token_ids,
                "start_array": self.tool_call_start_token_array,
                "end_array": self.tool_call_end_token_array,
            }

    Hermes2ProToolParser.__init__ = _patched_init


def monkey_patch_tokenizer_thread_safety():
    """Patch HuggingFace tokenizer to make _encode_plus thread-safe.

    Under concurrent request load, vLLM's API server calls _encode_plus from
    multiple async handlers simultaneously. _encode_plus mutates the Rust
    tokenizer's internal state via set_truncation_and_padding (enable_truncation/
    enable_padding) and encode_special_tokens. The Rust backend uses RefCell-style
    borrow tracking (PyO3), and concurrent mutable borrows cause it to panic
    with ``RuntimeError: Already borrowed``.

    Fix: wrap the entire _encode_plus method in a per-tokenizer threading lock
    so that state mutation and the subsequent encode call are atomic.
    """
    import threading

    from transformers import PreTrainedTokenizerFast

    _original_encode_plus = PreTrainedTokenizerFast._encode_plus
    _locks: dict[int, threading.Lock] = {}
    _meta_lock = threading.Lock()

    def _get_lock(tokenizer_id: int) -> threading.Lock:
        if tokenizer_id not in _locks:
            with _meta_lock:
                if tokenizer_id not in _locks:
                    _locks[tokenizer_id] = threading.Lock()
        return _locks[tokenizer_id]

    def _patched_encode_plus(self, *args, **kwargs):
        lock = _get_lock(id(self._tokenizer))
        with lock:
            return _original_encode_plus(self, *args, **kwargs)

    PreTrainedTokenizerFast._encode_plus = _patched_encode_plus


def monkey_patch_minimax_m2_for_lora():
    """Patch vLLM's MiniMaxM2 model for LoRA compatibility.

    These patches are only needed when using LoRA with MiniMax M2 but are safe
    to apply unconditionally (verified with non-LoRA runs). We apply them at
    import time because the worker __init__ runs before the vLLM config is
    available, so we can't check if LoRA is enabled.

    Problem 1 — Gate dtype mismatch:
        vLLM's MiniMaxM2MoE creates the gate (router) with params_dtype=float32
        and casts inputs to float32. When LoRA is enabled, vLLM wraps ALL
        ReplicatedLinear layers (including the gate) with LoRA support. Even
        though our adapter has no gate LoRA weights, the LoRA Triton kernel
        still runs for all wrapped layers when any adapter is active — and it
        asserts inputs are float16/bfloat16. Qwen3 MoE doesn't have this
        problem because its gate uses the model dtype.
        Fix: recreate the gate in model dtype and remove the float32 cast.
        FusedMoE already has router_logits_dtype=float32, so routing precision
        is preserved inside the expert dispatch.

    Problem 2 — Adapter key naming mismatch:
        PrimeRL saves adapter keys using its internal naming convention
        (mlp.experts.{j}.gate_proj/down_proj/up_proj), which matches Qwen3 MoE
        but not MiniMax M2. vLLM's MiniMax M2 model expects HF-style keys
        (block_sparse_moe.experts.{j}.w1/w2/w3). For full model weights this
        is handled by vLLM's load_weights(), but LoRA adapters are loaded
        through a separate path (LoRAModel.from_local_checkpoint) that doesn't
        have model-specific key translation.
        Fix: set hf_to_vllm_mapper on the model class so vLLM remaps adapter
        keys during LoRA loading. This attribute is only read by _load_adapter
        in the LoRA worker manager — it has no effect without LoRA.
    """
    from vllm.model_executor.models.minimax_m2 import MiniMaxM2ForCausalLM, MiniMaxM2MoE
    from vllm.model_executor.models.utils import WeightsMapper

    # --- Gate dtype fix (only matters with LoRA, safe without) ---
    _original_init = MiniMaxM2MoE.__init__

    def _patched_init(self, config, quant_config=None, prefix=""):
        _original_init(self, config, quant_config, prefix)
        from vllm.model_executor.layers.linear import ReplicatedLinear

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

    def _patched_forward(self, hidden_states):
        from vllm.distributed import tensor_model_parallel_all_reduce

        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_dim)

    MiniMaxM2MoE.__init__ = _patched_init
    MiniMaxM2MoE.forward = _patched_forward

    # --- Adapter key remapping (only read by vLLM's LoRA adapter loader) ---
    MiniMaxM2ForCausalLM.hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".mlp.experts.": ".block_sparse_moe.experts.",
            ".gate_proj.": ".w1.",
            ".down_proj.": ".w2.",
            ".up_proj.": ".w3.",
        },
    )


def monkey_patch_harmony_stop_token_propagation():
    """Fix: vLLM 0.17.0 doesn't merge harmony stop tokens into per-request SamplingParams.

    The harmony mode sets stop_token_ids (including <|call|> and <|return|>) in
    default_sampling_params at server init, but ChatCompletionRequest.to_sampling_params()
    ignores them, using only self.stop_token_ids (which defaults to []).

    Upstream: https://github.com/vllm-project/vllm/issues/22519
    """
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    _original_to_sampling_params = ChatCompletionRequest.to_sampling_params

    def _patched_to_sampling_params(self, max_tokens, default_sampling_params):
        params = _original_to_sampling_params(self, max_tokens, default_sampling_params)
        # Merge harmony stop tokens from default_sampling_params
        default_stop_ids = default_sampling_params.get("stop_token_ids", [])
        if default_stop_ids:
            existing = set(params.stop_token_ids or [])
            merged = list(existing | set(default_stop_ids))
            params.stop_token_ids = merged
        return params

    ChatCompletionRequest.to_sampling_params = _patched_to_sampling_params


def monkey_patch_fused_moe_lora_dp():
    """Fix: LoRA + MoE + DP>1 produces corrupted output in vLLM 0.17.0.

    Two bugs:
    1. LoRA injection sets supports_internal_mk=True (via moe_kernel not None),
       causing the MoE runner to skip DP dispatch/combine. But the LoRA kernel
       uses NoDPEP and doesn't handle DP internally.
    2. LoRA decorators capture full-batch tensors but fire per-chunk inside the
       kernel's chunk loop. At DP>=3, dispatched batch > CHUNK_SIZE causes
       shape mismatches.

    Fix: Replace _inject_lora_into_fused_moe with a version that:
    (a) sets moe_kernel=None so the runner correctly dispatches
    (b) makes decorators chunk-aware by tracking chunk offsets

    Upstream: https://github.com/vllm-project/vllm/issues/23244
    """
    import types

    from vllm import envs
    from vllm.distributed.utils import divide
    from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
    from vllm.model_executor.layers.fused_moe.config import _get_config_dtype_str
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import MarlinExperts
    from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import FusedMoEModularMethod
    from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import UnfusedOAITritonExperts
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.model_executor.layers.fused_moe.prepare_finalize import MoEPrepareAndFinalizeNoDPEPModular

    def _fixed_inject(self):
        moe_state_dict = {}
        top_k = self.base_layer.top_k

        self.base_layer.ensure_moe_quant_config_init()
        quant_config = self.base_layer.quant_method.moe_quant_config

        if getattr(self.base_layer.quant_method, "supports_internal_mk", False):
            m_fused_moe_fn = self.base_layer.quant_method.moe_kernel
            m_fused_moe_fn.shared_experts = None
        else:
            prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular()
            m_fused_moe_fn = FusedMoEKernel(
                prepare_finalize,
                self.base_layer.quant_method.select_gemm_impl(prepare_finalize, self.base_layer),
            )

        if quant_config.use_mxfp4_w4a16:
            assert isinstance(m_fused_moe_fn.impl.fused_experts, (MarlinExperts, UnfusedOAITritonExperts))
        else:
            assert isinstance(m_fused_moe_fn.impl.fused_experts, TritonExperts)

        # --- Decorators (chunk-aware) ---

        def fwd_decorator(layer, func):
            def wrapper(*args, **kwargs):
                moe_state_dict["hidden_states"] = kwargs["hidden_states"]
                moe_state_dict["topk_ids"] = kwargs["topk_ids"]
                moe_state_dict["topk_weights"] = kwargs["topk_weights"]
                moe_state_dict["expert_map"] = kwargs["expert_map"]
                moe_state_dict["apply_router_weight_on_input"] = kwargs["apply_router_weight_on_input"]
                moe_state_dict["_chunk_offset"] = 0
                return func(*args, **kwargs)

            return wrapper

        def act_decorator(layer, func):
            def wrapper(*args, **kwargs):
                _, output, input = args
                chunk_M = input.view(-1, top_k, input.shape[-1]).shape[0]
                chunk_offset = moe_state_dict.get("_chunk_offset", 0)
                hidden_states = moe_state_dict["hidden_states"][chunk_offset : chunk_offset + chunk_M]
                topk_weights = moe_state_dict["topk_weights"][chunk_offset : chunk_offset + chunk_M]
                curr_topk_ids = moe_state_dict["topk_ids"][chunk_offset : chunk_offset + chunk_M]
                expert_map = moe_state_dict["expert_map"]
                config_dtype = _get_config_dtype_str(
                    dtype=hidden_states.dtype, use_fp8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False
                )
                num_tokens = hidden_states.size(0)
                M = min(num_tokens, envs.VLLM_FUSED_MOE_CHUNK_SIZE)
                max_lora_rank = self.w13_lora_a_stacked[0].shape[-2]
                shrink_config, expand_config = self._get_lora_moe_configs(
                    op_prefix="w13",
                    num_loras=self.max_loras,
                    rank=max_lora_rank,
                    num_slices=self._w13_slices,
                    M=M,
                    layer=layer,
                    top_k=top_k,
                    config_dtype=config_dtype,
                )
                SPARSITY_FACTOR = 8
                naive_block_assignment = (
                    expert_map is None
                    and num_tokens * top_k * SPARSITY_FACTOR <= self.base_layer.local_num_experts * self.max_loras
                )
                token_lora_mapping, sorted_token_ids_lora, expert_ids_lora, num_tokens_post_padded_lora = (
                    self.punica_wrapper.moe_lora_align_block_size(
                        curr_topk_ids,
                        num_tokens,
                        shrink_config["BLOCK_SIZE_M"],
                        self.base_layer.local_num_experts,
                        self.max_loras,
                        self.adapter_enabled,
                        expert_map,
                        naive_block_assignment=naive_block_assignment,
                    )
                )
                moe_state_dict["sorted_token_ids_lora"] = sorted_token_ids_lora
                moe_state_dict["expert_ids_lora"] = expert_ids_lora
                moe_state_dict["num_tokens_post_padded_lora"] = num_tokens_post_padded_lora
                moe_state_dict["token_lora_mapping"] = token_lora_mapping
                if sorted_token_ids_lora is not None:
                    expert_ids_lora = expert_ids_lora.view(self.max_loras, -1)
                    sorted_token_ids_lora = sorted_token_ids_lora.view(self.max_loras, -1)
                self.punica_wrapper.add_lora_fused_moe(
                    input.view(-1, top_k, input.shape[-1]),
                    hidden_states,
                    self.w13_lora_a_stacked,
                    self.w13_lora_b_stacked,
                    topk_weights,
                    sorted_token_ids_lora,
                    expert_ids_lora,
                    num_tokens_post_padded_lora,
                    max_lora_rank,
                    top_k,
                    shrink_config,
                    expand_config,
                    self.adapter_enabled,
                    fully_sharded=self.fully_sharded,
                    token_lora_mapping=token_lora_mapping,
                )
                result = func(*args, **kwargs)
                moe_state_dict["intermediate_cache2"] = output
                moe_state_dict["_chunk_M"] = chunk_M
                return result

            return wrapper

        def moe_sum_decorator(layer, func):
            def wrapper(*args, **kwargs):
                chunk_offset = moe_state_dict.get("_chunk_offset", 0)
                chunk_M = moe_state_dict.get("_chunk_M", moe_state_dict["hidden_states"].size(0))
                hidden_states = moe_state_dict["hidden_states"][chunk_offset : chunk_offset + chunk_M]
                topk_weights = moe_state_dict["topk_weights"][chunk_offset : chunk_offset + chunk_M]
                config_dtype = _get_config_dtype_str(
                    dtype=hidden_states.dtype, use_fp8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False
                )
                num_tokens = hidden_states.size(0)
                M = min(num_tokens, envs.VLLM_FUSED_MOE_CHUNK_SIZE)
                max_lora_rank = self.w2_lora_a_stacked[0].shape[-2]
                shrink_config, expand_config = self._get_lora_moe_configs(
                    op_prefix="w2",
                    num_loras=self.max_loras,
                    rank=max_lora_rank,
                    num_slices=1,
                    M=M,
                    layer=layer,
                    top_k=top_k,
                    config_dtype=config_dtype,
                )
                sorted_token_ids_lora = moe_state_dict["sorted_token_ids_lora"]
                expert_ids_lora = moe_state_dict["expert_ids_lora"]
                num_tokens_post_padded_lora = moe_state_dict["num_tokens_post_padded_lora"]
                token_lora_mapping = moe_state_dict.get("token_lora_mapping")
                if sorted_token_ids_lora is not None:
                    expert_ids_lora = expert_ids_lora.view(self.max_loras, -1)
                    sorted_token_ids_lora = sorted_token_ids_lora.view(self.max_loras, -1)
                intermediate_cache2 = moe_state_dict["intermediate_cache2"]
                intermediate_cache3 = args[0]
                shard_size_w2 = divide(self.base_layer.hidden_size, self.tp_size)
                self.punica_wrapper.add_lora_fused_moe(
                    intermediate_cache3,
                    intermediate_cache2,
                    self.w2_lora_a_stacked,
                    self.w2_lora_b_stacked,
                    topk_weights,
                    sorted_token_ids_lora,
                    expert_ids_lora,
                    num_tokens_post_padded_lora,
                    max_lora_rank,
                    top_k,
                    shrink_config,
                    expand_config,
                    self.adapter_enabled,
                    True,
                    fully_sharded=self.fully_sharded,
                    offset=shard_size_w2 * self.tp_rank if self.fully_sharded else 0,
                    token_lora_mapping=token_lora_mapping,
                )
                result = func(*args, **kwargs)
                moe_state_dict["_chunk_offset"] = chunk_offset + chunk_M
                return result

            return wrapper

        # --- Install decorators and replace quant method ---

        fused_experts = m_fused_moe_fn.impl.fused_experts
        m_fused_moe_fn.apply = fwd_decorator(self.base_layer, m_fused_moe_fn.apply)
        fused_experts.activation = act_decorator(self.base_layer, fused_experts.activation)
        fused_experts.moe_sum = moe_sum_decorator(self.base_layer, fused_experts.moe_sum)

        new_method = FusedMoEModularMethod(self.base_layer.quant_method, m_fused_moe_fn)

        # Bug 1 fix: NoDPEP kernel makes supports_internal_mk=True, causing the
        # runner to skip DP dispatch/combine. Set moe_kernel=None and patch apply().
        if isinstance(m_fused_moe_fn.prepare_finalize, MoEPrepareAndFinalizeNoDPEPModular):
            saved_kernel = new_method.moe_kernel
            saved_disable_expert_map = new_method.disable_expert_map
            new_method.moe_kernel = None

            def _apply_with_saved_kernel(self, layer, x, topk_weights, topk_ids, shared_experts_input):
                return saved_kernel.apply(
                    hidden_states=x,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    activation=layer.activation,
                    global_num_experts=layer.global_num_experts,
                    apply_router_weight_on_input=layer.apply_router_weight_on_input,
                    expert_map=None if saved_disable_expert_map else layer.expert_map,
                    shared_experts_input=shared_experts_input,
                )

            new_method.apply = types.MethodType(_apply_with_saved_kernel, new_method)

        self.base_layer._replace_quant_method(new_method)

    FusedMoEWithLoRA._inject_lora_into_fused_moe = _fixed_inject
