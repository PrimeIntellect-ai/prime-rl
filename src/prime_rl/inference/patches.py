def transformers_v5_compat():
    """vLLM general plugin: patch transformers v5 config attrs that vLLM 0.16 still expects.

    Registered as a ``vllm.general_plugins`` entry-point so it runs automatically
    in every vLLM process, including spawned workers.
    """
    from transformers import Qwen3VLMoeTextConfig

    if not hasattr(Qwen3VLMoeTextConfig, "tie_word_embeddings"):
        Qwen3VLMoeTextConfig.tie_word_embeddings = False


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

    TokenizeParams._token_len_check = _patched_token_len_check
    TokenizeParams._text_len_check = _patched_text_len_check


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


def monkey_patch_per_message_tokenization():
    """Patch vLLM to cache AR-generated token IDs and reuse them in subsequent turns.

    Problem: in multi-turn conversations via the OpenAI chat API, assistant
    responses are sent back as text.  vLLM re-tokenizes the whole conversation
    from scratch, but BPE encoding of that text can produce different token IDs
    than what the model originally generated during AR decoding (non-canonical
    tokenization).  Different token IDs → prefix-cache miss → full re-prefill
    of the entire conversation history every turn.

    Fix — two patches working together:

    1. **Cache population** (``chat_completion_full_generator`` wrapper):
       after every non-streaming generation, store ``text → token_ids`` for
       each completion output.

    2. **Cache lookup** (``HfRenderer.render_messages`` override):
       when building the prompt for a new request, split the rendered text at
       message-content boundaries.  For assistant messages whose content is
       found in the cache, splice in the original AR token IDs instead of
       re-encoding.  All other segments (template markup, user/system content)
       are tokenized normally.
    """
    import hashlib
    from collections import OrderedDict

    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.renderers.hf import (
        HfRenderer,
        parse_chat_messages,
        resolve_chat_template_content_format,
        safe_apply_chat_template,
    )
    from vllm.renderers.inputs import DictPrompt
    from vllm.renderers.params import ChatParams

    # ── Bounded LRU cache: response text → AR token IDs ──────────────

    _AR_TOKEN_CACHE: OrderedDict[str, list[int]] = OrderedDict()
    _AR_CACHE_MAX = 4096

    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _cache_put(text: str, token_ids: list[int]) -> None:
        key = _cache_key(text)
        _AR_TOKEN_CACHE[key] = token_ids
        _AR_TOKEN_CACHE.move_to_end(key)
        while len(_AR_TOKEN_CACHE) > _AR_CACHE_MAX:
            _AR_TOKEN_CACHE.popitem(last=False)

    def _cache_get(text: str) -> list[int] | None:
        key = _cache_key(text)
        ids = _AR_TOKEN_CACHE.get(key)
        if ids is not None:
            _AR_TOKEN_CACHE.move_to_end(key)
        return ids

    # ── Patch 1: populate cache after generation ─────────────────────

    _original_full_generator = OpenAIServingChat.chat_completion_full_generator

    async def _caching_full_generator(self, request, result_generator, *args, **kwargs):
        last_res = None

        async def _capturing_generator():
            nonlocal last_res
            async for res in result_generator:
                last_res = res
                yield res

        response = await _original_full_generator(self, request, _capturing_generator(), *args, **kwargs)

        if last_res is not None:
            for output in last_res.outputs:
                if output.text and output.token_ids:
                    _cache_put(output.text, list(output.token_ids))

        return response

    OpenAIServingChat.chat_completion_full_generator = _caching_full_generator

    # ── Patch 2: use cached AR tokens during tokenization ────────────

    _original_render_messages = HfRenderer.render_messages

    def _per_message_render(
        self: HfRenderer,
        messages: list,
        params: ChatParams,
    ) -> tuple[list, DictPrompt]:
        model_config = self.config
        tokenizer = self.get_tokenizer()

        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            model_config,
            content_format=resolve_chat_template_content_format(
                chat_template=params.chat_template,
                tools=params.chat_template_kwargs.get("tools"),
                given_format=params.chat_template_content_format,
                tokenizer=tokenizer,
                model_config=model_config,
            ),
        )

        if mm_data is not None:
            return _original_render_messages(self, messages, params)

        template_kwargs = params.get_apply_chat_template_kwargs()

        full_text = safe_apply_chat_template(model_config, tokenizer, conversation, tokenize=False, **template_kwargs)

        # Build a set of assistant content strings for cache-eligible lookup.
        assistant_contents: set[str] = set()
        for msg in conversation:
            if msg.get("role") == "assistant" and msg.get("content"):
                assistant_contents.add(msg["content"])

        # Split rendered text at message-content boundaries, tokenize each
        # segment independently, and splice in cached AR tokens for assistant
        # messages when available.
        all_token_ids: list[int] = []
        last_end = 0

        for msg in conversation:
            content = msg.get("content", "")
            if not content:
                continue
            pos = full_text.find(content, last_end)
            if pos == -1:
                continue

            # Template markup before this content
            if pos > last_end:
                all_token_ids.extend(tokenizer.encode(full_text[last_end:pos], add_special_tokens=False))

            # Message content: use cached AR tokens for assistant messages
            cached = _cache_get(content) if content in assistant_contents else None
            if cached is not None:
                all_token_ids.extend(cached)
            else:
                all_token_ids.extend(tokenizer.encode(content, add_special_tokens=False))

            last_end = pos + len(content)

        # Remaining template text (im_end, generation prompt, etc.)
        if last_end < len(full_text):
            all_token_ids.extend(tokenizer.encode(full_text[last_end:], add_special_tokens=False))

        prompt: DictPrompt = {"prompt_token_ids": all_token_ids, "prompt": full_text}  # type: ignore[assignment]
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids  # type: ignore[typeddict-unknown-key]

        return conversation, prompt

    HfRenderer.render_messages = _per_message_render


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
