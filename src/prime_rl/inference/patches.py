import torch

from prime_rl.inference.vllm.padded_input_scrub import monkey_patch_vllm_padded_input_scrub


def apply_shared_vllm_patches():
    """vLLM general plugin: prime-rl patches that must run in every vLLM process.

    Registered as a ``vllm.general_plugins`` entry-point so it runs automatically
    in every vLLM process, including spawned workers. Note vLLM swallows plugin
    load failures (``load_plugins_by_group`` logs and continues), so a broken
    entry-point target silently skips ALL of these patches.
    """
    _patch_lora_key_prefix()
    _patch_qwen35_moe_lora_format()
    monkey_patch_nano_v3_reasoning_parser()
    monkey_patch_qwen3_coder_param_newline_trim()
    monkey_patch_minimax_m2_think_end_passthrough()
    monkey_patch_vllm_padded_input_scrub()
    monkey_patch_return_routed_experts_with_nixl_connector()
    monkey_patch_kv_xfer_finished_tolerate_freed()


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


def monkey_patch_qwen3_coder_param_newline_trim():
    """Restore vLLM 0.23's single-newline trim for qwen3_coder tool parameters.

    vLLM 0.24's parser engine applies a full ``.strip()`` to every parameter
    value in ``_qwen3_arg_converter``; 0.23's ``Qwen3CoderToolParser`` trimmed
    exactly one leading and one trailing newline. A full strip corrupts
    whitespace-significant string parameters (str_replace-style
    ``old_str``/``new_str``, file content with an indented first line, values
    with intentional trailing newlines), silently changing tool execution and
    rewards for agentic runs. Copy of ``_qwen3_arg_converter`` with only the
    trim changed. Also covers ``nemotron_v3``, which reuses ``qwen3_config``.
    """
    import json

    from vllm.parser import qwen3

    def _trim_one_newline(value: str) -> str:
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        return value

    def _patched_arg_converter(raw_args: str, partial: bool) -> str:
        params: dict[str, object] = {}

        for match in qwen3._PARAM_RE.finditer(raw_args):
            name = match.group(1)
            value = match.group(2)
            ## START PATCHED CODE (upstream: value.strip())
            params[name] = _trim_one_newline(value)
            ## END PATCHED CODE

        if partial:
            remaining = qwen3._PARAM_RE.sub("", raw_args)
            m = qwen3._PARTIAL_PARAM_RE.search(remaining)
            if m:
                name = m.group(1)
                value = m.group(2)
                if name:
                    ## START PATCHED CODE (upstream: value.strip())
                    params[name] = _trim_one_newline(value)
                    ## END PATCHED CODE

        return json.dumps(params, ensure_ascii=False)

    qwen3._qwen3_arg_converter = _patched_arg_converter
    # qwen3_config captures the converter when first built; drop any cached configs.
    qwen3.qwen3_config.cache_clear()


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
    from vllm import envs
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
        if envs.VLLM_USE_V2_MODEL_RUNNER:
            raise ValueError("VLLM_USE_V2_MODEL_RUNNER does not yet support: routed experts capture")

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


def _patch_qwen35_moe_lora_format():
    """Force Qwen3.5-MoE onto vLLM's 2D per-expert LoRA format.

    vLLM 0.24.0 still defaults ``Qwen3_5MoeForConditionalGeneration.is_3d_moe_weight = True``,
    which makes the LoRA loader expect 3D stacked-expert adapters
    (``base_layer.lora_{A,B}.weight`` / ``lora_{A,B}.weight``, experts folded into the
    rank dim; see ``_stack_moe_lora_weights``). Our trainer instead emits the 2D
    per-expert layout (``{expert_id}.gate_proj.lora_A.weight`` ...) from
    ``MultiLoRAGroupedExperts.state_dict_for_adapter`` -- vLLM only consults that layout
    when ``is_3d_moe_weight`` is False (or ``enable_mixed_moe_lora_format=True``).
    Without this override the adapters fail to load with key/shape mismatches.

    The rest of the old Qwen3.5 LoRA shim (the in_proj_qkvz packed-mapping fix and the
    N-slice ``can_replace_layer`` / ``slice_lora_a`` generalizations for vllm#36372) is
    handled natively by 0.23.0 and was dropped. Remove this too once we either adopt the
    3D stacked save format (like gpt-oss) or start the engine with
    ``enable_mixed_moe_lora_format=True``.
    """
    from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration

    Qwen3_5MoeForConditionalGeneration.is_3d_moe_weight = False


def _patch_lora_key_prefix():
    """Accept both bare-suffix and fully-qualified expert module names in LoRA adapters.

    Copy of vLLM 0.24.0's ``LoRAModel.from_local_checkpoint`` with one change: the
    ``.experts`` branch of ``check_unexpected_modules`` accepts either the bare suffix
    (``down_proj``) or the qualified per-expert name (``experts.N.down_proj``), where
    upstream only accepts the qualified form. Our trainer's 2D per-expert adapters
    (Qwen3.5-MoE) carry names whose qualified form is not in the expected set while
    the bare suffix is; Qwen3-30B-A3B adapters go the other way. Upstream fix
    vllm-project/vllm#38522 was closed unmerged, so this stays.
    """
    from vllm.lora.lora_model import (
        LoRAModel,
        MoEEPLoadSpec,
        PEFTHelper,
        TensorizerConfig,
        WeightsMapper,
        _is_remote_expert_key,
        get_lora_id,
        is_base_embedding_weights,
        os,
        parse_fine_tuned_lora_name,
        safetensors,
    )

    def _patched_from_local_checkpoint(
        cls,
        lora_dir: str,
        expected_lora_modules: set[str],
        peft_helper: PEFTHelper,
        *,
        lora_model_id: int | None = None,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        model_vocab_size: int | None = None,
        weights_mapper: WeightsMapper | None = None,
        tensorizer_config_dict: dict | None = None,
        skip_prefixes: list[str] | None = None,
        moe_ep_spec: MoEEPLoadSpec | None = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a local checkpoint.

        Args:
            lora_dir: The local path that has lora data.
            expected_lora_modules: Name of modules that are expected to be
                replaced by lora.
            peft_helper: Loaded lora configuration information.
            lora_model_id: LoRA model id. If not given, automatically set by
                a global counter.
            device: Device where the lora model is loaded.
            dtype: dtype of the lora model weights.
            skip_prefixes: List of module name prefixes to skip during loading.
                Models can define this to skip modules not used in inference
                (e.g., MTP layers). Format: ["mtp."]
            moe_ep_spec: When 2D FusedMoE LoRA modules are present with
                expert parallelism enabled, the (ep_rank, local, global)
                slicing metadata shared across all MoE layers. Non-local
                expert weights are skipped at read time instead of being
                loaded and discarded later.

        Returns:
            Loaded LoRA Model.
        """
        lora_tensor_path = os.path.join(lora_dir, "adapter_model.safetensors")
        lora_bin_file_path = os.path.join(lora_dir, "adapter_model.bin")
        lora_pt_file_path = os.path.join(lora_dir, "adapter_model.pt")

        tensors: dict[str, torch.Tensor] = {}
        unexpected_modules: list[list[str] | str] = []

        def check_unexpected_modules(modules: dict):
            for lora_module in modules.keys():  # noqa
                if is_base_embedding_weights(lora_module):
                    continue
                # Handle PEFT file format where experts.base_layer is the
                # gate_up_proj and experts is the down_proj
                if "base_layer" in lora_module:
                    continue
                # Skip modules based on model-defined prefixes
                if skip_prefixes and cls._should_skip_module(lora_module, skip_prefixes):
                    continue
                module_name, _ = parse_fine_tuned_lora_name(lora_module, weights_mapper)
                # Case for expert lora weights.
                ## START PATCHED CODE (upstream only accepts the qualified form)
                if ".experts" in module_name:
                    expert_suffix = module_name.split(".")[-1]
                    experts_qualified = "experts" + module_name.split(".experts", 1)[-1]
                    if expert_suffix not in expected_lora_modules and experts_qualified not in expected_lora_modules:
                        unexpected_modules.append(module_name)
                ## END PATCHED CODE

                elif module_name.rsplit(".", 1)[-1] not in expected_lora_modules:
                    unexpected_modules.append(module_name)

            if unexpected_modules:
                raise ValueError(
                    f"While loading {lora_dir}, expected"
                    f" target modules in {expected_lora_modules}"
                    f" but received {unexpected_modules}."
                    f" Please verify that the loaded LoRA module is correct"
                )

        if tensorizer_config_dict:
            from tensorizer import TensorDeserializer

            tensorizer_config = TensorizerConfig(**tensorizer_config_dict)
            tensorizer_dir = tensorizer_config.tensorizer_dir
            if tensorizer_dir is None:
                raise ValueError("tensorizer_dir must be set in tensorizer config.")
            lora_tensor_path = os.path.join(tensorizer_dir, "adapter_model.tensors")
            tensorizer_args = tensorizer_config._construct_tensorizer_args()
            tensors = TensorDeserializer(
                lora_tensor_path,
                dtype=tensorizer_config.dtype,
                device=device,
                **tensorizer_args.deserialization_kwargs,
            )
            check_unexpected_modules(tensors)

        elif os.path.isfile(lora_tensor_path):
            # Find unexpected modules.
            # Use safetensor key as a source of truth to find expected modules.
            # in peft if you have target_modules A, B, C and C does not exist
            # in the model it won’t error and model will be trained with A, B
            # loraified. C won’t exist in the safetensor but it will exist in
            # the target_modules of the adapter_config.json.
            unexpected_modules = []
            with safetensors.safe_open(lora_tensor_path, framework="pt") as f:  # type: ignore
                # Load tensors if there are only expected modules.
                check_unexpected_modules(f)
                for module in f.keys():  # noqa
                    if moe_ep_spec is not None and _is_remote_expert_key(module, moe_ep_spec):
                        continue
                    tensors[module] = f.get_tensor(module)
        elif os.path.isfile(lora_bin_file_path) or os.path.isfile(lora_pt_file_path):
            lora_file_path = lora_bin_file_path if os.path.isfile(lora_bin_file_path) else lora_pt_file_path
            tensors = torch.load(lora_file_path, map_location=device, weights_only=True)
            check_unexpected_modules(tensors)
            if moe_ep_spec is not None:
                # `.bin`/`.pt` adapters can't be lazy-loaded, but pruning
                # the dict here still frees the non-local expert tensors
                # before the dtype cast / pin_memory work that follows.
                tensors = {k: v for k, v in tensors.items() if not _is_remote_expert_key(k, moe_ep_spec)}
        else:
            raise ValueError(f"{lora_dir} doesn't contain tensors")

        return cls.from_lora_tensors(
            lora_model_id=get_lora_id() if lora_model_id is None else lora_model_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device=device,
            dtype=dtype,
            model_vocab_size=model_vocab_size,
            weights_mapper=weights_mapper,
            skip_prefixes=skip_prefixes,
        )

    LoRAModel.from_local_checkpoint = classmethod(_patched_from_local_checkpoint)


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
        Fix: rebuild the gate as GateLinear with a bf16 weight (out_dtype=float32
        keeps fp32 router logits). vLLM 0.24.0's own forward already drops the
        float32 input cast. FusedMoE also has router_logits_dtype=float32, so
        routing precision is preserved inside the expert dispatch.

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
        from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

        # vLLM 0.24.0 builds the gate as GateLinear with a float32 weight; rebuild it
        # with a bf16 weight (model dtype) so the LoRA Triton kernel's float16/bfloat16
        # assertion passes, keeping out_dtype=float32 so router logits stay fp32 (the
        # GateLinear bf16xbf16->fp32 path).
        self.gate = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

    MiniMaxM2MoE.__init__ = _patched_init

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
    """Fix: vLLM doesn't merge harmony stop tokens into per-request SamplingParams.

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


def monkey_patch_no_moe_lora():
    """This disables LoRA for MoE layers and makes them pick better kernels.

    Otherwise, the oracle will always try to pick TritonExperts.
    For blackwells, we want TRTLLMFlashInfer.
    """
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

    original_post_init = FusedMoEConfig.__post_init__

    def _patched__post_init__(self: FusedMoEConfig):
        original_post_init(self)
        # Disable LoRA for MoE layers. `is_lora_enabled` is only read later during
        # kernel selection (modular_kernel / unquantized oracle), never inside
        # `__post_init__`, so flipping it after the original runs is sufficient.
        self.is_lora_enabled = False

    FusedMoEConfig.__post_init__ = _patched__post_init__


def monkey_patch_fp32_lm_head():
    """Run the lm_head projection in fp32, via a native bf16xbf16 -> fp32 GEMM.

    Uses ``torch.mm(..., out_dtype=torch.float32)`` (PyTorch >= 2.10) so the
    matmul accumulates and emits fp32 directly without zero-padding the bf16
    operands or maintaining a separate fp32 weight copy. This avoids the
    epilogue truncation to bf16 that `F.linear(bf16, bf16)` does, which is
    where lm_head precision actually leaks before the sampler's softmax.

    Activated by setting ``additional_config["fp32_lm_head"] = True`` on the
    vLLM namespace; the launcher does this when ``inference.enable_fp32_lm_head``
    is set. The flag is captured once on ``LogitsProcessor.__init__`` (where
    vLLM guarantees a ``set_current_vllm_config()`` context) and stored on the
    instance — reading it from ``_get_logits`` during serving doesn't work
    because vLLM doesn't keep the context set during forwards.

    Tracks vllm-project/vllm#24567 (which uses the operand-upcast approach).
    Per @Jackmin801 on PR #2438, native ``out_dtype=fp32`` mm is more efficient
    and just as correct.
    """
    import torch
    from vllm.config import get_current_vllm_config
    from vllm.logger import init_logger
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    logger = init_logger(__name__)

    _original_init = LogitsProcessor.__init__
    _original_get_logits = LogitsProcessor._get_logits

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        vllm_config = get_current_vllm_config()
        additional_config = vllm_config.additional_config or {}
        self._fp32_lm_head_enabled = additional_config.get("fp32_lm_head", False)
        if self._fp32_lm_head_enabled:
            logger.warning("fp32 lm_head ENABLED for this LogitsProcessor instance.")

    def _patched_get_logits(self, hidden_states, lm_head, embedding_bias):
        if not getattr(self, "_fp32_lm_head_enabled", False):
            return _original_get_logits(self, hidden_states, lm_head, embedding_bias)

        # Native bf16xbf16 -> fp32 GEMM. torch.mm requires 2D inputs; vLLM v1's
        # generative path passes 2D [num_tokens, hidden_size] hidden_states, but
        # flatten defensively in case some future caller passes 3D.
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = torch.mm(flat, lm_head.weight.t(), out_dtype=torch.float32)
        if embedding_bias is not None:
            logits = logits + embedding_bias.to(torch.float32)
        if hidden_states.dim() > 2:
            logits = logits.reshape(*hidden_states.shape[:-1], -1)

        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    LogitsProcessor.__init__ = _patched_init
    LogitsProcessor._get_logits = _patched_get_logits
    logger.info("Installed fp32 lm_head patch (native out_dtype=fp32 mm).")


def monkey_patch_dp_coordinator_startup_timeout():
    """Raise the DP coordinator startup timeout from vLLM's hard-coded 120s.

    The coordinator child process is spawned on the DP-rank-0 API server while
    every engine-core rank on the node is importing and loading weights, so its
    own spawn-time re-import can exceed the hard-coded timeout under that CPU/IO
    contention (seen on multi-node disaggregated GLM-5.1 launches). Configurable
    via PRIME_DP_COORDINATOR_STARTUP_TIMEOUT (seconds, default 300).
    """
    import multiprocessing.connection
    import os

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
