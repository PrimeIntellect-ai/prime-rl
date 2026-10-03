import os


def apply_shared_vllm_patches():
    """vLLM general plugin and the single place prime-rl applies its vLLM patches; vLLM runs it once in every process.

    vLLM swallows plugin load failures (``load_plugins_by_group`` logs and continues), so a broken
    entry-point target silently skips ALL of these patches.
    """
    from prime_rl.inference.vllm.gpt_oss_weight_loading import patch_gpt_oss_weight_loading
    from prime_rl.inference.vllm.qwen38_weight_loading import patch_qwen38_weight_loading

    patch_gpt_oss_weight_loading()
    patch_qwen38_weight_loading()
    monkey_patch_nano_v3_reasoning_parser()
    monkey_patch_minimax_m2_think_end_passthrough()
    monkey_patch_return_routed_experts_with_nixl_connector()
    monkey_patch_kv_xfer_finished_tolerate_freed()
    monkey_patch_online_fp8_parameter_cast()
    monkey_patch_deepseek_v4_allowed_layer_types()
    monkey_patch_deepseek_v4_request_tools_placement()
    monkey_patch_tokenize_params_validation()
    monkey_patch_strip_routed_experts_from_chat()
    monkey_patch_dp_coordinator_startup_timeout()


def monkey_patch_deepseek_v4_allowed_layer_types():
    """Let vLLM's `DeepseekV4Config` construct under a transformers pin below 5.15.

    `vllm/transformers_utils/configs/deepseek_v4.py` leaves `layer_types` to
    `PretrainedConfig`, whose validator checks it against a vocabulary that only gained V4's
    compressed attention names in transformers 5.15, so `AutoConfig.from_pretrained` on any
    DeepSeek V4 checkpoint raises. vLLM declares `transformers>=5.5.3`, so this is a gap in
    upstream's own version floor rather than something prime-rl introduced.

    `EngineArgs.__post_init__` loads general plugins before `create_model_config`, which is why
    patching here is early enough. The trainer applies the same shim on its own import path.
    """
    from prime_rl.utils.transformers_compat import allow_deepseek_v4_layer_types

    allow_deepseek_v4_layer_types()


def monkey_patch_deepseek_v4_request_tools_placement():
    """Attach request-level tools to the first existing DSV4 system message.

    vLLM 0.29.0's Python DeepSeek-V4 tokenizer always prepends a synthetic
    system message for request-level tools. That puts the tool schema before
    an existing system prompt, unlike DeepSeek's reference encoder, vLLM's
    Rust renderer, and prime-rl's training renderer. Upstream fixed this in
    https://github.com/vllm-project/vllm/pull/51856 (commit 2909ad8f).
    The fix is expected to ship in vLLM 0.31.

    Wrap the tokenizer factory so existing-system requests take the corrected
    path through the stock implementation: shallow-copy the conversation,
    attach tools to its first system message, and suppress the stock synthetic
    insertion. Requests without a system message retain the stock behavior.
    Remove this patch once the vLLM pin includes the upstream fix (likely 0.31).
    """
    import copy

    from vllm.tokenizers import deepseek_v4 as dsv4_tokenizer

    original_get_tokenizer = dsv4_tokenizer.get_deepseek_v4_tokenizer
    if getattr(original_get_tokenizer, "_prime_rl_places_request_tools", False):
        return

    def _get_deepseek_v4_tokenizer(tokenizer):
        wrapped = original_get_tokenizer(tokenizer)
        tokenizer_cls = wrapped.__class__
        original_apply_chat_template = tokenizer_cls.apply_chat_template

        # Each factory call creates a fresh dynamic tokenizer subclass, but be
        # defensive if vLLM starts caching that class in a future release.
        if getattr(original_apply_chat_template, "_prime_rl_places_request_tools", False):
            return wrapped

        def _apply_chat_template(self, messages, tools=None, **kwargs):
            if tools:
                conversation = kwargs.get("conversation", messages)
                system_idx = next(
                    (i for i, message in enumerate(conversation) if message.get("role") == "system"),
                    None,
                )
                if system_idx is not None:
                    conversation = conversation.copy()
                    conversation[system_idx] = copy.copy(conversation[system_idx])
                    conversation[system_idx]["tools"] = tools
                    kwargs["conversation"] = conversation
                    tools = None

            return original_apply_chat_template(self, messages, tools=tools, **kwargs)

        _apply_chat_template._prime_rl_places_request_tools = True
        tokenizer_cls.apply_chat_template = _apply_chat_template
        return wrapped

    _get_deepseek_v4_tokenizer._prime_rl_places_request_tools = True
    dsv4_tokenizer.get_deepseek_v4_tokenizer = _get_deepseek_v4_tokenizer


def monkey_patch_online_fp8_parameter_cast():
    """Pass plain tensors to vLLM's compiled block-FP8 caster.

    Layerwise reload restores ``ModelWeightParameter`` objects before online
    quantization. TorchDynamo recursively dispatches that tensor subclass while
    tracing ``per_block_cast_to_fp8``. ``Parameter.data`` shares storage but is
    a plain tensor, matching the input accepted during the initial model load.
    """
    from torch.nn import Parameter
    from vllm.model_executor.layers.quantization.online import fp8

    original_cast = fp8.per_block_cast_to_fp8
    if getattr(original_cast, "_prime_rl_unwraps_parameters", False):
        return

    def _per_block_cast_to_fp8(x, *args, **kwargs):
        if isinstance(x, Parameter):
            x = x.data
        return original_cast(x, *args, **kwargs)

    _per_block_cast_to_fp8._prime_rl_unwraps_parameters = True
    fp8.per_block_cast_to_fp8 = _per_block_cast_to_fp8


def monkey_patch_kv_xfer_finished_tolerate_freed():
    """Tolerate KV-transfer finish notifications for already-freed requests.

    In disaggregated P/D (NIXL, optionally + a KV store connector) a request can
    be finished — most often ``FINISHED_ABORTED`` from an off-policy cancel, a
    client disconnect, or a request timeout — while it still has in-flight KV
    transfers. When such a request's ``finished_recving`` and ``finished_sending``
    both land in the same ``Scheduler.update_from_output`` step, the stock
    ``_update_from_kv_xfer_finished`` frees it in the recving branch
    (``_free_blocks`` -> ``del self.requests[req_id]``) and then the sending
    branch hits ``assert req_id in self.requests`` and kills the EngineCore. On a
    DP deployment that one death cascades to every rank via the gloo finish-state
    all-reduce, taking down the whole inference pool.

    The trigger is the abort itself, not weight-update pause/resume: it reproduces
    during normal stepping whenever an aborted request's recv and send complete in
    the same step (observed with zero off-policy cancellations, driven only by
    incidental client-side aborts). Skip already-freed request ids instead of
    asserting — their blocks are freed either way, so dropping the stale
    notification is safe.

    Upstream issue: https://github.com/vllm-project/vllm/issues/46240
    """
    from vllm.logger import init_logger
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    logger = init_logger("vllm.v1.core.sched.scheduler")

    if getattr(Scheduler._update_from_kv_xfer_finished, "_prime_rl_tolerates_freed", False):
        return

    def _update_from_kv_xfer_finished(self, kv_connector_output):
        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            # Stale notification for a request freed earlier this step (e.g. an
            # aborted request whose send completion freed it). Nothing to do.
            if req_id not in self.requests:
                continue
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            # See above: the recving branch may have already freed an aborted
            # request whose send also completed this step.
            if req_id not in self.requests:
                continue
            self._free_blocks(self.requests[req_id])

    _update_from_kv_xfer_finished._prime_rl_tolerates_freed = True
    Scheduler._update_from_kv_xfer_finished = _update_from_kv_xfer_finished
    logger.warning("Patched Scheduler._update_from_kv_xfer_finished to tolerate freed (aborted) KV-transfer reqs.")


def monkey_patch_nano_v3_reasoning_parser():
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
    from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

    class NanoV3ReasoningParser(DeepSeekR1ReasoningParser):
        def extract_reasoning(self, model_output, request):
            reasoning_content, final_content = super().extract_reasoning(model_output, request)
            chat_template_kwargs = getattr(request, "chat_template_kwargs", None)

            if chat_template_kwargs and chat_template_kwargs.get("enable_thinking") is False and final_content is None:
                reasoning_content, final_content = final_content, reasoning_content

            return reasoning_content, final_content

    ReasoningParserManager.register_module("nano_v3", module=NanoV3ReasoningParser)


def monkey_patch_minimax_m2_think_end_passthrough():
    """Keep the literal ``</think>`` in MiniMax-M2 content on tool-calling turns.

    prime-rl serves MiniMax-M2 with ``reasoning=minimax_m2_append_think``, which
    returns content as ``<think>`` + the full completion so think tags round-trip
    through multi-turn re-serialization. vLLM 0.24's minimax_m2 parser engine
    added a ``(CONTENT, THINK_END) -> no-events`` transition that silently
    swallows the ``</think>`` (0.23's regex tool parser passed it through
    untouched), and it also ``.strip()``s content whenever tool calls are
    present. Drop the transition — the engine emits unmatched terminals as plain
    state content — and disable the content strip.
    """
    import dataclasses
    import functools

    from vllm.parser import minimax_m2
    from vllm.parser.engine.parser_engine_config import ParserState

    original_config = minimax_m2.minimax_m2_config

    @functools.cache
    def _patched_config():
        config = original_config()
        transitions = dict(config.transitions)
        del transitions[(ParserState.CONTENT, "THINK_END")]
        return dataclasses.replace(
            config,
            transitions=transitions,
            strip_content_whitespace_with_tools=False,
        )

    minimax_m2.minimax_m2_config = _patched_config


def monkey_patch_return_routed_experts_with_nixl_connector():
    from vllm.config.vllm import VllmConfig
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    original_post_init = VllmConfig.__post_init__

    if getattr(original_post_init, "_prime_rl_allows_nixl_routed_experts", False):
        return

    def _is_nixl_routed_experts_pd_config(config: VllmConfig) -> bool:
        kv_transfer_config = config.kv_transfer_config
        return (
            config.model_config is not None
            and config.model_config.enable_return_routed_experts
            and kv_transfer_config is not None
            and kv_transfer_config.kv_connector == "NixlConnector"
            and kv_transfer_config.is_kv_transfer_instance
        )

    def _post_init(config: VllmConfig):
        if not _is_nixl_routed_experts_pd_config(config):
            return original_post_init(config)

        if config.parallel_config.pipeline_parallel_size > 1:
            raise ValueError("--enable-return-routed-experts is incompatible with pipeline parallelism (PP > 1).")
        if config.use_v2_model_runner:
            raise ValueError(
                "Routed-expert capture with NIXL requires the V1 model runner. Set VLLM_USE_V2_MODEL_RUNNER=0."
            )

        # vLLM rejects every KV connector, but our P/D path uses NIXL and
        # stitches prefill/decode routed experts in the router. CPU KV offload
        # remains rejected by prime-rl config validation.
        config.model_config.enable_return_routed_experts = False
        try:
            return original_post_init(config)
        finally:
            config.model_config.enable_return_routed_experts = True

    _post_init._prime_rl_allows_nixl_routed_experts = True
    VllmConfig.__post_init__ = _post_init
    logger.warning("Enabled vLLM routed-experts capture with NIXL connector patch.")


def monkey_patch_strip_routed_experts_from_chat():
    """Drop routed_experts from chat-completions responses.

    routed_experts are only consumed via the serialized ``/generate``
    (serving_tokens) path used for router-replay training, which encodes them as a
    ``{data, shape, start}`` object the PD router can merge. The stock
    chat-completions path instead encodes them as a base64 ``np.save`` *string*,
    which the PD router rejects ("prefill routed_experts must be an object with
    base64 data and shape") and fails every eval rollout (evals go through chat
    completions). ``enable_return_routed_experts`` is a server-wide model-config
    flag with no per-request toggle, so strip the field on the chat path here.
    """
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    if getattr(OpenAIServingChat.chat_completion_full_generator, "_prime_rl_strips_routed_experts", False):
        return

    _original = OpenAIServingChat.chat_completion_full_generator

    async def _strip(result_generator):
        async for res in result_generator:
            for output in res.outputs:
                output.routed_experts = None
            yield res

    async def _patched(self, request, result_generator, *args, **kwargs):
        return await _original(self, request, _strip(result_generator), *args, **kwargs)

    _patched._prime_rl_strips_routed_experts = True
    OpenAIServingChat.chat_completion_full_generator = _patched
    logger.info(
        "Stripped routed_experts from chat-completions responses (PD router merges only the /generate object form)."
    )


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


def monkey_patch_dp_coordinator_startup_timeout():
    """Raise the DP coordinator startup timeout from vLLM's hard-coded 120s.

    The coordinator child process is spawned on the DP-rank-0 API server while
    every engine-core rank on the node is importing and loading weights, so its
    own spawn-time re-import can exceed the hard-coded timeout under that CPU/IO
    contention (seen on multi-node disaggregated GLM-5.1 launches). Configurable
    via PRIME_DP_COORDINATOR_STARTUP_TIMEOUT (seconds, default 300).
    """
    import multiprocessing.connection

    from vllm.v1.engine.coordinator import DPCoordinator

    timeout = float(os.environ.get("PRIME_DP_COORDINATOR_STARTUP_TIMEOUT", "300"))

    def _patched_wait_for_zmq_addrs(self, zmq_addr_pipe):
        try:
            ready = multiprocessing.connection.wait([zmq_addr_pipe, self.proc.sentinel], timeout=timeout)
            if not ready:
                raise RuntimeError(
                    f"DP Coordinator process failed to report ZMQ addresses within {timeout}s during startup."
                )
            try:
                return zmq_addr_pipe.recv()
            except EOFError:
                raise RuntimeError("DP Coordinator process failed during startup.") from None
        finally:
            zmq_addr_pipe.close()

    DPCoordinator._wait_for_zmq_addrs = _patched_wait_for_zmq_addrs
