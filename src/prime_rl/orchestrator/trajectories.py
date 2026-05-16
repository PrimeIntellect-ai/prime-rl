import base64
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import verifiers as vf
from PIL import Image
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.trainer import RoleLossMaskConfig
from prime_rl.transport import TrainingSample
from prime_rl.utils.chat_template import (
    IncrementalTokenizationError,
    build_incremental_token_mask,
    common_prefix_len,
    deserialize_tool_calls,
    normalize_messages,
    render_messages,
    strip_message_content,
)
from prime_rl.utils.logger import get_logger

# We use list() instead of deepcopy() for flat lists (token IDs, logprobs) - safe because
# primitives are immutable. pixel_values/image_grid_thw are not mutated after creation.


def _align_routed_experts(
    routed_experts: list[list[list[int]]] | None,
    expected_len: int,
) -> list[list[list[int]]] | None:
    """Align routed_experts length with the expected token count.

    VLLM's capturer uses `num_tokens - 1` slot mappings because the final
    generated token was never fed as input to a forward pass and has no
    routing decision. Append zero-filled entries for the missing positions.
    """
    if routed_experts is None or not routed_experts:
        return routed_experts
    deficit = expected_len - len(routed_experts)
    if deficit <= 0:
        return routed_experts
    num_layers = len(routed_experts[0])
    topk = len(routed_experts[0][0])
    zero_entry = [[0] * topk for _ in range(num_layers)]
    return routed_experts + [zero_entry for _ in range(deficit)]


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    return common_prefix_len(a, b)


def _normalize_messages(messages: Any, default_role: str) -> list[dict[str, Any]]:
    return normalize_messages(messages, default_role)


def _deserialize_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return deserialize_tool_calls(messages)


def _strip_message_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return strip_message_content(messages)


def _render_messages(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = False,
    tools: list[dict[str, Any]] | None = None,
    processor=None,
) -> list[int]:
    return render_messages(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        processor=processor,
    )


def _should_mask_role(message: dict[str, Any], config: RoleLossMaskConfig) -> bool:
    role = message.get("role")
    match role:
        case "system":
            return config.system
        case "user":
            return config.user
        case "assistant":
            return config.assistant
        case "tool":
            return config.tool
        case _:
            raise ValueError(f"Invalid message role: {role}")


def _tool_call_function_name(tc: dict[str, Any] | Any) -> str | None:
    """Extract the function name from a tool_call, handling both verifiers' flat
    {id, name, arguments} shape and OpenAI's nested {id, type, function: {name, arguments}}.
    """
    if isinstance(tc, dict):
        name = tc.get("name")
        if name is None:
            fn = tc.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
        return name
    name = getattr(tc, "name", None)
    if name is None:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None) if fn is not None else None
    return name


def _tool_call_names_in_order(
    prompt: list[dict[str, Any]] | None,
    completion: list[dict[str, Any]] | None,
) -> list[str | None]:
    """Return the function names of the originating tool_calls for each tool-role
    message in `prompt + completion`, in conversation order.

    Walks the assistant messages first to build a `tool_call_id -> name` map,
    then returns one entry per tool-role message — `None` for any tool message
    whose `tool_call_id` we can't resolve (older trajectories, malformed data).
    The caller should treat `None` as "name unknown" and apply its own policy.
    """
    id_to_name: dict[str, str | None] = {}
    for messages in (prompt or [], completion or []):
        for m in messages:
            if not isinstance(m, dict) or m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id is None:
                    continue
                id_to_name[tc_id] = _tool_call_function_name(tc)

    names: list[str | None] = []
    for messages in (prompt or [], completion or []):
        for m in messages:
            if not isinstance(m, dict) or m.get("role") != "tool":
                continue
            tc_id = m.get("tool_call_id")
            names.append(id_to_name.get(tc_id) if tc_id is not None else None)
    return names


def _build_role_loss_mask_from_token_stream(
    full_ids: list[int],
    tokenizer: PreTrainedTokenizer,
    loss_mask_config: RoleLossMaskConfig,
    tool_call_names_in_order: list[str | None] | None = None,
) -> list[bool]:
    """Parse a Qwen-style ChatML token stream into a per-token role mask.

    Scans `<|im_start|>ROLE\\n ... <|im_end|>` spans directly in the vLLM
    token stream — no chat-template re-rendering. This sidesteps every
    vLLM/template divergence (pretty-print tool_call JSON, JSON-schema key
    ordering, trailing whitespace, etc.) that the older
    `build_incremental_token_mask` path tripped on.

    Special-case for Qwen3's tool messages: the template wraps role="tool"
    inside a user block (`<|im_start|>user\\n<tool_response>…`), so on a
    `user` header we peek at the next token: if it's the `<tool_response>`
    special token, we classify the span as `tool`, otherwise as `user`.

    Assumes:
    - `<|im_start|>`, `<|im_end|>`, `<tool_response>` are single
      vocabulary tokens (they are in Qwen3).
    - Role names (`system`, `user`, `assistant`, `tool`) encode as one or
      more text tokens that decode back to the role string.
    - The ChatML delimiter between header and body is `\\n` (token 198 in
      Qwen3).
    """
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    tool_response = tokenizer.convert_tokens_to_ids("<tool_response>")
    tool_response_close = tokenizer.convert_tokens_to_ids("</tool_response>")
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline_ids) != 1:
        raise RuntimeError(
            "Expected tokenizer to encode '\\n' as a single token for ChatML "
            f"parsing, got {newline_ids!r}"
        )
    newline = newline_ids[0]

    _KNOWN_ROLES = {"system", "user", "assistant", "tool"}

    allowlist: set[str] | None = (
        set(loss_mask_config.tool_name_allowlist)
        if loss_mask_config.tool_name_allowlist is not None
        else None
    )

    def _mask_role(role: str | None) -> bool:
        # Defensive: an unrecognised role shouldn't mask anything. The token
        # stream on a live run should only contain the four ChatML roles; if
        # it doesn't we'd rather be silent than crash the whole batch.
        if role not in _KNOWN_ROLES:
            return False
        return _should_mask_role({"role": role}, loss_mask_config)

    def _mask_for_tool_span(span_idx: int) -> bool:
        # Conservative: when the allowlist is set but we lack the metadata
        # needed to decide (no precomputed names, cursor past the end, or
        # name unresolved), mask the span out. Silent inclusion would defeat
        # the experimental control the allowlist exists for.
        if not _mask_role("tool") or allowlist is None:
            return _mask_role("tool")
        if tool_call_names_in_order is None or span_idx >= len(tool_call_names_in_order):
            return False
        name = tool_call_names_in_order[span_idx]
        return name in allowlist if name is not None else False

    mask = [False] * len(full_ids)
    current_role: str | None = None
    current_span_mask = False
    tool_span_idx = 0
    i = 0
    n = len(full_ids)
    while i < n:
        tok = full_ids[i]
        if tok == im_start:
            # Scan ahead for the `\n` terminating the role header.
            j = i + 1
            while j < n and full_ids[j] != newline:
                j += 1
            role_tokens = full_ids[i + 1 : j]
            role_str = tokenizer.decode(role_tokens).strip() if role_tokens else ""
            # Qwen3's tool messages live inside a `user` block; disambiguate
            # by peeking at the first body token.
            if role_str == "user" and j + 1 < n and full_ids[j + 1] == tool_response:
                role_str = "tool"
            current_role = role_str
            current_span_mask = (
                _mask_for_tool_span(tool_span_idx) if role_str == "tool" else _mask_role(role_str)
            )
            span_end = min(j, n - 1)
            for k in range(i, span_end + 1):
                mask[k] = current_span_mask
            i = j + 1
        elif tok == im_end and current_role is not None:
            mask[i] = current_span_mask
            i += 1
            # `\n` separator after <|im_end|> (if present) stays with the
            # current role, matching the behavior of build_incremental_token_mask.
            if i < n and full_ids[i] == newline:
                mask[i] = current_span_mask
                i += 1
            if current_role == "tool":
                tool_span_idx += 1
            current_role = None
            current_span_mask = False
        else:
            mask[i] = current_span_mask if current_role is not None else False
            i += 1

    if loss_mask_config.tool_content_only and loss_mask_config.tool:
        # Second pass: zero the chat-template envelope tokens on each tool span,
        # leaving only the actual content tokens inside `<tool_response>…</tool_response>`
        # contributing to the SFT loss. The structural tokens (`<|im_start|>user\n`,
        # `<tool_response>\n`, `\n</tool_response>`, `<|im_end|>\n`) are the same
        # for every tool message regardless of content; training to generate
        # them gives a degenerate target that the model can satisfy by
        # hallucinating the envelope outside real tool calls (cf. cmb-all step-700
        # coherence collapse in 2026-05-14 forth-lang 1k writeup).
        i = 0
        while i < n:
            if full_ids[i] != im_start:
                i += 1
                continue
            j = i + 1
            while j < n and full_ids[j] != newline:
                j += 1
            is_tool_span = (
                j + 1 < n
                and full_ids[j + 1] == tool_response
                and tokenizer.decode(full_ids[i + 1 : j]).strip() == "user"
            )
            if not is_tool_span:
                i = j + 1
                continue
            # Find the matching <|im_end|>.
            k = j + 1
            while k < n and full_ids[k] != im_end:
                k += 1
            if k >= n:
                break  # malformed trailing span; nothing more to do
            # Header: <|im_start|> user \n (indices i..j inclusive)
            for p in range(i, j + 1):
                mask[p] = False
            # Opening envelope: <tool_response> \n (indices j+1, j+2)
            mask[j + 1] = False
            if j + 2 < n and full_ids[j + 2] == newline:
                mask[j + 2] = False
            # Closing envelope: \n </tool_response> (indices k-2, k-1)
            if k - 1 >= 0 and full_ids[k - 1] == tool_response_close:
                mask[k - 1] = False
                if k - 2 >= 0 and full_ids[k - 2] == newline:
                    mask[k - 2] = False
            # Footer: <|im_end|> \n (indices k, k+1)
            mask[k] = False
            if k + 1 < n and full_ids[k + 1] == newline:
                mask[k + 1] = False
            i = k + 1

    return mask


def _extract_tool_content_spans(
    full_ids: list[int],
    tokenizer: PreTrainedTokenizer,
    tool_call_names_in_order: list[str | None],
) -> list[tuple[int, int, str]]:
    """Return ``[(content_start, content_end, tool_name), ...]`` for each
    tool message in the token stream.

    Span is **content-only**: the indices cover just the actual tool-output
    tokens *between* ``<tool_response>\\n`` and ``\\n</tool_response>``,
    excluding the chat-template envelope. Half-open Python slice semantics
    (``full_ids[content_start:content_end]`` gives the content tokens).

    Tool messages whose originating tool_call name we couldn't resolve
    (``None`` entries in ``tool_call_names_in_order``) are skipped — there's
    no useful aggregation bucket for them.

    Used for capturing per-tool token-level logprobs after a rollout (see
    ``OrchestratorConfig.capture_tool_logprobs``).
    """
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    tool_response = tokenizer.convert_tokens_to_ids("<tool_response>")
    tool_response_close = tokenizer.convert_tokens_to_ids("</tool_response>")
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline_ids) != 1:
        raise RuntimeError(
            "Expected tokenizer to encode '\\n' as a single token for ChatML "
            f"parsing, got {newline_ids!r}"
        )
    newline = newline_ids[0]

    spans: list[tuple[int, int, str]] = []
    tool_span_idx = 0
    n = len(full_ids)
    i = 0
    while i < n:
        if full_ids[i] != im_start:
            i += 1
            continue
        j = i + 1
        while j < n and full_ids[j] != newline:
            j += 1
        is_tool_span = (
            j + 1 < n
            and full_ids[j + 1] == tool_response
            and tokenizer.decode(full_ids[i + 1 : j]).strip() == "user"
        )
        if not is_tool_span:
            i = j + 1
            continue
        # Find matching <|im_end|>.
        k = j + 1
        while k < n and full_ids[k] != im_end:
            k += 1
        if k >= n:
            break

        # Content start: skip <tool_response> and the optional separating \n.
        content_start = j + 2
        if content_start < n and full_ids[content_start] == newline:
            content_start += 1
        # Content end: back off </tool_response> (and any preceding \n that
        # isn't BPE-fused into the last content token).
        content_end = k
        if k - 1 >= 0 and full_ids[k - 1] == tool_response_close:
            content_end = k - 1
            if k - 2 >= 0 and full_ids[k - 2] == newline:
                content_end = k - 2

        if (
            tool_call_names_in_order is not None
            and tool_span_idx < len(tool_call_names_in_order)
        ):
            name = tool_call_names_in_order[tool_span_idx]
            if name is not None and content_end > content_start:
                spans.append((content_start, content_end, name))
        tool_span_idx += 1
        i = k + 1

    return spans


async def annotate_tool_nll_metrics(
    rollouts: list[vf.RolloutOutput],
    *,
    clients: list[Any],
    model_name: str,
    tokenizer: PreTrainedTokenizer,
) -> dict[str, int]:
    """For each rollout, score the model's own log-probabilities on every
    tool-response *content* token via a vllm `prompt_logprobs` forward pass,
    and write per-tool aggregated metrics into ``rollout["metrics"]``:

      * ``tool_nll_<tool_name>``        — mean negative log-likelihood per
                                          tool-content token (across all tool
                                          messages of that name in this
                                          rollout).
      * ``tool_nll_token_count_<name>`` — number of tokens that went into
                                          the above mean. Useful as a sanity
                                          / sample-size denominator.

    Mutates the rollouts in place. The last tool message of a terminating
    rollout (typically ``submit_code``'s response) is not scored because it
    never appears in any subsequent prompt; this is documented in the
    config field's docstring.

    Returns a stats dict with skip-reason counters so the caller can log
    visibility into how many rollouts contributed to the metrics:
    ``{"scored": N, "no_trajectory": N, "no_spans": N, "annotated": N}``.
    """
    # Lazy import to keep utils.py the only place importing httpx/openai.
    from prime_rl.orchestrator.utils import compute_prompt_logprobs

    stats = {"scored": 0, "no_trajectory": 0, "no_spans": 0, "annotated": 0}

    # Build the score targets and per-rollout span maps.
    score_token_seqs: list[list[int]] = []
    per_rollout_spans: list[list[tuple[int, int, str]]] = []
    for rollout in rollouts:
        trajectory = rollout.get("trajectory") or []
        last_step = trajectory[-1] if trajectory else None
        tokens = last_step.get("tokens") if last_step else None
        if not tokens or not tokens.get("prompt_ids"):
            per_rollout_spans.append([])
            score_token_seqs.append([])
            stats["no_trajectory"] += 1
            continue
        # Score the last step's full prompt — this contains every tool
        # response except the terminating one (which is never re-prompted).
        full_ids = list(tokens["prompt_ids"])
        tool_call_names = _tool_call_names_in_order(
            rollout.get("prompt"), rollout.get("completion")
        )
        spans = _extract_tool_content_spans(full_ids, tokenizer, tool_call_names)
        per_rollout_spans.append(spans)
        if spans:
            score_token_seqs.append(full_ids)
        else:
            score_token_seqs.append([])
            stats["no_spans"] += 1

    # Run a single batched scoring pass for rollouts that actually have spans.
    scorable_idx = [i for i, ids in enumerate(score_token_seqs) if ids]
    if not scorable_idx:
        return stats
    scoring_inputs = [score_token_seqs[i] for i in scorable_idx]
    logprobs_lists = await compute_prompt_logprobs(
        clients=clients, model_name=model_name, token_sequences=scoring_inputs
    )
    stats["scored"] = len(scorable_idx)

    # Aggregate per tool name, write into rollout metrics.
    for i, logprobs in zip(scorable_idx, logprobs_lists):
        spans = per_rollout_spans[i]
        if not spans or not logprobs:
            continue
        per_tool_sum: dict[str, float] = {}
        per_tool_count: dict[str, int] = {}
        for content_start, content_end, name in spans:
            content_end = min(content_end, len(logprobs))
            if content_start >= content_end:
                continue
            seg = logprobs[content_start:content_end]
            per_tool_sum[name] = per_tool_sum.get(name, 0.0) + sum(-lp for lp in seg)
            per_tool_count[name] = per_tool_count.get(name, 0) + len(seg)
        metrics = rollouts[i].setdefault("metrics", {})
        wrote_any = False
        for name, count in per_tool_count.items():
            if count > 0:
                metrics[f"tool_nll_{name}"] = per_tool_sum[name] / count
                metrics[f"tool_nll_token_count_{name}"] = float(count)
                wrote_any = True
        if wrote_any:
            stats["annotated"] += 1
    return stats


def _build_role_loss_masks_for_step(
    step: vf.TrajectoryStep,
    tokenizer: PreTrainedTokenizer,
    loss_mask_config: RoleLossMaskConfig,
    tools: list[dict[str, Any]] | None = None,
    processor=None,
) -> tuple[list[bool], list[bool]] | None:
    """Derive per-token role loss masks for a single trajectory step.

    Parses the vLLM token stream directly when the step carries
    `prompt_ids`/`completion_ids` (the normal RL path). Falls back to a
    re-render of the messages when tokens aren't available (e.g. offline
    SFT-data synthesis).
    """
    tool_call_names = _tool_call_names_in_order(step.get("prompt"), step.get("completion"))
    tokens = step.get("tokens")
    if tokens is not None:
        full_ids = list(tokens["prompt_ids"]) + list(tokens["completion_ids"])
        full_loss_mask = _build_role_loss_mask_from_token_stream(
            full_ids, tokenizer, loss_mask_config, tool_call_names_in_order=tool_call_names
        )
        split_idx = len(tokens["prompt_ids"])
        return full_loss_mask[:split_idx], full_loss_mask[split_idx:]

    # Fallback: no saved tokens (offline SFT synthesis path). Re-render and
    # derive the split from the rendered prompt length.
    prompt = _normalize_messages(step.get("prompt"), default_role="user")
    completion = _normalize_messages(step.get("completion"), default_role="assistant")
    if processor is not None:
        prompt = _prepare_messages_for_processor(prompt)
        completion = _prepare_messages_for_processor(completion)
    messages = prompt + completion
    allowlist = (
        set(loss_mask_config.tool_name_allowlist)
        if loss_mask_config.tool_name_allowlist is not None
        else None
    )
    id_to_name: dict[str, str | None] = {}
    if allowlist is not None:
        for m in messages:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id is not None:
                    id_to_name[tc_id] = _tool_call_function_name(tc)

    def _role_to_mask(message: dict[str, Any]) -> bool:
        base = _should_mask_role(message, loss_mask_config)
        if not base or message.get("role") != "tool" or allowlist is None:
            return base
        name = id_to_name.get(message.get("tool_call_id"))
        return name in allowlist if name is not None else False

    try:
        full_ids, full_loss_mask = build_incremental_token_mask(
            tokenizer,
            messages,
            role_to_mask=_role_to_mask,
            tools=tools,
            collapse_consecutive_tool_messages=True,
            processor=processor,
        )
    except IncrementalTokenizationError as e:
        get_logger().warning(f"Skipping rollout step due to unstable incremental tokenization: {e}")
        return None
    prompt_has_assistant_completion = len(completion) > 0 and completion[0].get("role") == "assistant"
    prompt_ids = _render_messages(
        tokenizer,
        prompt,
        add_generation_prompt=prompt_has_assistant_completion,
        tools=tools,
        processor=processor,
    )
    split_idx = _common_prefix_len(prompt_ids, full_ids)
    return full_loss_mask[:split_idx], full_loss_mask[split_idx:]


def _prepare_messages_for_processor(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert messages to the format expected by the VLM processor.

    - Converts image_url items to image items with loaded PIL Images
    - Strips extra fields (e.g. image_url on text items) that confuse the processor
    - Ensures all message content is in list format (processor requires this)
    """
    prepared = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            prepared.append({**msg, "content": [{"type": "text", "text": content}]})
            continue

        if not isinstance(content, list):
            prepared.append(msg)
            continue

        new_content = []
        for item in content:
            if item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url.startswith(_FILE_URL_PREFIX):
                    img = _load_file_image(url)
                elif url.startswith("data:image"):
                    b64_data = url.split(",", 1)[1]
                    img = Image.open(BytesIO(base64.b64decode(b64_data)))
                else:
                    new_content.append(item)
                    continue
                new_content.append({"type": "image", "image": img})
            elif item.get("type") == "text":
                new_content.append({"type": "text", "text": item.get("text", "")})
            else:
                new_content.append(item)
        prepared.append({**msg, "content": new_content})

    return prepared


def _tokenize_step_from_messages(
    step: vf.TrajectoryStep,
    tokenizer: PreTrainedTokenizer,
    tools: list[dict[str, Any]] | None = None,
    processor=None,
) -> dict[str, Any]:
    prompt = _normalize_messages(step.get("prompt"), default_role="user")
    completion = _normalize_messages(step.get("completion"), default_role="assistant")

    prompt = _strip_message_content(_deserialize_tool_calls(prompt))
    completion = _strip_message_content(_deserialize_tool_calls(completion))

    assert all(m.get("role") == "assistant" for m in completion), (
        "Expected all completion messages to be assistant role for SFT distillation, "
        f"got roles: {[m.get('role') for m in completion]}"
    )

    if processor is not None:
        prompt = _prepare_messages_for_processor(prompt)
        completion = _prepare_messages_for_processor(completion)

    all_messages = prompt + completion
    prompt_has_assistant_completion = len(completion) > 0 and completion[0].get("role") == "assistant"
    prompt_ids = _render_messages(
        tokenizer,
        prompt,
        add_generation_prompt=prompt_has_assistant_completion,
        tools=tools,
        processor=processor,
    )
    full_ids = _render_messages(
        tokenizer,
        all_messages,
        tools=tools,
        processor=processor,
    )

    split_idx = _common_prefix_len(prompt_ids, full_ids)
    original_prompt_len = len(prompt_ids)

    prompt_ids = full_ids[:split_idx]
    completion_ids = full_ids[split_idx:]
    completion_mask = [True] * len(completion_ids)
    completion_logprobs = [0.0] * len(completion_ids)

    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": [False] * len(prompt_ids),
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "completion_logprobs": completion_logprobs,
        "routed_experts": None,
        "prompt_prefix_len": split_idx,
        "original_prompt_len": original_prompt_len,
    }


def _sort_dict_keys_recursively(obj: Any) -> Any:
    """Recursively sort all dict keys alphabetically."""
    if isinstance(obj, dict):
        return {k: _sort_dict_keys_recursively(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_sort_dict_keys_recursively(v) for v in obj]
    return obj


def _convert_tools_to_oai_format(tool_defs: list) -> list[dict[str, Any]] | None:
    """Convert verifiers Tool objects or dicts to OAI function-calling format.

    The OpenAI-compatible inference server (vLLM) serializes each tool's JSON
    Schema `parameters` with alphabetically sorted keys (schema normalization
    via Pydantic). The chat template then renders that sorted dict directly via
    `tojson`, producing e.g. `"parameters": {"additionalProperties": ...,
    "properties": ...}`. Python dict insertion order gives the opposite here,
    which breaks the role-mask length check against vLLM's prompt_ids. The
    outer `{"type", "function"}` wrapper and the inner function fields
    (`{"name", "description", "parameters"}`) are kept in insertion order,
    because vLLM preserves those.
    """
    if not tool_defs:
        return None

    def _get(tool: Any, key: str) -> Any:
        if isinstance(tool, dict):
            return tool.get(key)
        return getattr(tool, key, None)

    return [
        {
            "type": "function",
            "function": {
                "name": _get(tool, "name"),
                "description": _get(tool, "description"),
                "parameters": _sort_dict_keys_recursively(_get(tool, "parameters")),
                **({} if _get(tool, "strict") is None else {"strict": _get(tool, "strict")}),
            },
        }
        for tool in tool_defs
    ]


def pretokenize_rollout_trajectory(
    output: vf.RolloutOutput,
    tokenizer: PreTrainedTokenizer,
    processor=None,
) -> bool:
    """Populate missing step tokens from prompt/completion messages."""
    logger = get_logger()
    tools = _convert_tools_to_oai_format(output.get("tool_defs", []))

    for step_idx, step in enumerate(output["trajectory"]):
        if step["tokens"] is not None:
            continue

        reconstructed = _tokenize_step_from_messages(step, tokenizer, tools=tools, processor=processor)
        if reconstructed["prompt_prefix_len"] < reconstructed["original_prompt_len"]:
            logger.debug(
                f"Prompt tokenization was non-prefix for example {output['example_id']} step {step_idx}. "
                f"Using longest common prefix length {reconstructed['prompt_prefix_len']} "
                f"(original prompt had {reconstructed['original_prompt_len']} tokens)."
            )

        reconstructed.pop("prompt_prefix_len")
        reconstructed.pop("original_prompt_len")
        step["tokens"] = reconstructed

    return True


def interleave_rollout(
    output: vf.RolloutOutput,
    vlm_cache: "VLMImageCache | None" = None,
    cache_key: int | None = None,
    mm_token_type_ids_mapping: dict[int, int] | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    loss_mask_config: RoleLossMaskConfig | None = None,
    sft_loss_mask_config: RoleLossMaskConfig | None = None,
    processor=None,
) -> list[TrainingSample] | None:
    """
    Convert vf.RolloutOutput to trainable rollouts by interleaving trajectory steps
    where the extension property holds.

    When consecutive steps share token prefixes (extension property), they are
    merged into a single sample. When extension breaks (e.g., due to context
    compaction or a change in control-flow), a new sample is started.

    Supports multi-prefix matching to handle interleaved agents. For example,
    [agent1-step1, agent1-step2, agent2-step1, agent1-step3] produces two samples:
    agent1 steps merged together, agent2 step separate.

    Returns a list of samples - could be 1 (extension always held) or up to T
    (extension never held).

    For VLM models, pass vlm_cache to attach cumulative pixel_values per sample.
    Each sample gets the images accumulated up to its last merged step.

    Args:
        output: vf.RolloutOutput containing trajectory data
        vlm_cache: Pre-computed VLM image cache for multimodal training
        cache_key: Cache key to use when retrieving images from the VLM cache
    """
    logger = get_logger()

    trajectory = output["trajectory"]
    if len(trajectory) == 0:
        error = output.get("error")
        stop = output.get("stop_condition")
        logger.warning(
            f"No trajectory steps for example {output['example_id']} (error={error}, stop={stop}). Skipping rollout."
        )
        return None

    has_error = output["error"] is not None
    # this field should be guaranteed because we set temperature in get_sampling_args
    temperature = output["sampling_args"]["temperature"]
    tools = _convert_tools_to_oai_format(output.get("tool_defs", []))
    use_custom_loss_mask = loss_mask_config is not None and not loss_mask_config.is_completion_only()
    # The auxiliary SFT mask is built only when an explicit config is provided.
    # We build it regardless of whether it happens to be "completion-only"; its
    # presence in the config is what signals the trainer to use it for SFT.
    use_sft_loss_mask = sft_loss_mask_config is not None

    def prepare_step_tokens(step: vf.TrajectoryStep, step_idx: int) -> dict[str, Any] | None:
        tokens = step["tokens"]
        if tokens is not None:
            prepared = {
                "prompt_ids": list(tokens["prompt_ids"]),
                "prompt_mask": [bool(i) for i in tokens["prompt_mask"]],
                "completion_ids": list(tokens["completion_ids"]),
                "completion_mask": [bool(i) for i in tokens["completion_mask"]],
                "completion_logprobs": list(tokens["completion_logprobs"]),
                "routed_experts": tokens.get("routed_experts"),
            }
            if use_custom_loss_mask:
                assert tokenizer is not None
                role_masks = _build_role_loss_masks_for_step(
                    step, tokenizer, loss_mask_config, tools=tools, processor=processor
                )
                if role_masks is None:
                    return None
                prepared["prompt_loss_mask"], prepared["completion_loss_mask"] = role_masks
            else:
                prepared["prompt_loss_mask"] = list(prepared["prompt_mask"])
                prepared["completion_loss_mask"] = list(prepared["completion_mask"])
            if use_sft_loss_mask:
                assert tokenizer is not None
                sft_role_masks = _build_role_loss_masks_for_step(
                    step, tokenizer, sft_loss_mask_config, tools=tools, processor=processor
                )
                if sft_role_masks is None:
                    return None
                prepared["sft_prompt_loss_mask"], prepared["sft_completion_loss_mask"] = sft_role_masks
            return prepared

        logger.warning(f"Missing rollout tokens for example {output['example_id']} step {step_idx}.")
        return None

    prepared_steps: list[dict[str, Any]] = []
    for step_idx, step in enumerate(trajectory):
        prepared = prepare_step_tokens(step, step_idx)
        if prepared is None:
            return None
        prepared_steps.append(prepared)

    def make_sample(tokens: dict[str, Any]) -> TrainingSample:
        """Create a new TrainingSample from a trajectory step."""
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
            prompt_loss_mask = [False] * len(tokens["prompt_loss_mask"])
            completion_loss_mask = [False] * len(tokens["completion_loss_mask"])
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]
            prompt_loss_mask = [bool(i) for i in tokens["prompt_loss_mask"]]
            completion_loss_mask = [bool(i) for i in tokens["completion_loss_mask"]]
        # Auxiliary SFT masks, if built (optional — only present when the
        # orchestrator's `sft_loss_mask` config is set).
        if use_sft_loss_mask:
            if has_error:
                sft_prompt_loss_mask = [False] * len(tokens["sft_prompt_loss_mask"])
                sft_completion_loss_mask = [False] * len(tokens["sft_completion_loss_mask"])
            else:
                sft_prompt_loss_mask = [bool(i) for i in tokens["sft_prompt_loss_mask"]]
                sft_completion_loss_mask = [bool(i) for i in tokens["sft_completion_loss_mask"]]
        else:
            sft_prompt_loss_mask = None
            sft_completion_loss_mask = None
        completion_ids = list(tokens["completion_ids"])

        routed_experts = _align_routed_experts(
            tokens.get("routed_experts"),
            len(tokens["prompt_ids"]) + len(tokens["completion_ids"]),
        )
        prompt_ids = list(tokens["prompt_ids"])
        return TrainingSample(
            prompt_ids=prompt_ids,
            prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
            prompt_loss_mask=prompt_loss_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_loss_mask=completion_loss_mask,
            sft_prompt_loss_mask=sft_prompt_loss_mask,
            sft_completion_loss_mask=sft_completion_loss_mask,
            completion_logprobs=list(tokens["completion_logprobs"]),
            completion_temperatures=[temperature] * len(completion_ids),
            teacher_logprobs=None,
            advantage=None,
            routed_experts=routed_experts,
            mm_token_type_ids=None,
        )

    def extend_sample(sample: TrainingSample, prefix_len: int, step_idx: int) -> None:
        """Extend an existing sample with a new trajectory step (extension property holds)."""
        tokens = prepared_steps[step_idx]

        # Extend with new prompt tokens (mask=False, no gradient)
        new_prompt_ids = tokens["prompt_ids"][prefix_len:]
        sample.completion_ids.extend(new_prompt_ids)
        sample.completion_mask.extend([False] * len(new_prompt_ids))
        assert sample.completion_loss_mask is not None
        sample.completion_loss_mask.extend(tokens["prompt_loss_mask"][prefix_len:])
        if use_sft_loss_mask:
            assert sample.sft_completion_loss_mask is not None
            sample.sft_completion_loss_mask.extend(tokens["sft_prompt_loss_mask"][prefix_len:])
        sample.completion_logprobs.extend([0.0] * len(new_prompt_ids))
        sample.completion_temperatures.extend([temperature] * len(new_prompt_ids))

        # Extend with new completion tokens
        completion_ids = tokens["completion_ids"]
        sample.completion_ids.extend(completion_ids)
        if has_error:
            sample.completion_mask.extend([False] * len(tokens["completion_mask"]))
            sample.completion_loss_mask.extend([False] * len(tokens["completion_loss_mask"]))
            if use_sft_loss_mask:
                assert sample.sft_completion_loss_mask is not None
                sample.sft_completion_loss_mask.extend([False] * len(tokens["sft_completion_loss_mask"]))
        else:
            sample.completion_mask.extend(bool(i) for i in tokens["completion_mask"])
            sample.completion_loss_mask.extend(bool(i) for i in tokens["completion_loss_mask"])
            if use_sft_loss_mask:
                assert sample.sft_completion_loss_mask is not None
                sample.sft_completion_loss_mask.extend(bool(i) for i in tokens["sft_completion_loss_mask"])
        sample.completion_logprobs.extend(tokens["completion_logprobs"])
        sample.completion_temperatures.extend([temperature] * len(completion_ids))

        if tokens.get("routed_experts") is not None and sample.routed_experts is not None:
            step_routed = tokens["routed_experts"]
            # The previous step's last routing entry was zero-padded by _align_routed_experts
            # (vLLM only captures num_tokens-1 routings per request). This step actually
            # processed that boundary token as part of its prompt, so replace the zero-fill
            # with the real routing decision before appending new entries.
            if prefix_len > 0 and prefix_len <= len(step_routed):
                sample.routed_experts[prefix_len - 1] = step_routed[prefix_len - 1]
            sample.routed_experts.extend(step_routed[prefix_len:])
            expected_len = len(sample.prompt_ids) + len(sample.completion_ids)
            sample.routed_experts = _align_routed_experts(sample.routed_experts, expected_len)

    # Track [prefix_tokens, sample, last_step_idx] per active sample
    active_samples: list[tuple[list[int], TrainingSample, int]] = []

    first_tokens = prepared_steps[0]
    first_prefix = first_tokens["prompt_ids"] + first_tokens["completion_ids"]
    active_samples.append((first_prefix, make_sample(first_tokens), 0))

    for step_idx, _step in enumerate(trajectory[1:], start=1):
        tokens = prepared_steps[step_idx]
        step_prompt_ids = tokens["prompt_ids"]

        # Check if this step extends ANY active prefix
        matched_idx = None
        for idx, (prefix_tokens, _, _) in enumerate(active_samples):
            if step_prompt_ids[: len(prefix_tokens)] == prefix_tokens:
                matched_idx = idx
                break

        if matched_idx is not None:
            # Extension holds - merge into matched sample
            prefix_tokens, sample, _ = active_samples[matched_idx]
            extend_sample(sample, len(prefix_tokens), step_idx=step_idx)
            active_samples[matched_idx] = (tokens["prompt_ids"] + tokens["completion_ids"], sample, step_idx)
        else:
            # No prefix matches - start a new sample
            logger.debug(
                f"Extension property broke at step {step_idx + 1} for example {output['example_id']}. "
                f"Starting new sample (active_prefixes={len(active_samples)}, step_prompt_len={len(step_prompt_ids)})."
            )
            new_prefix = tokens["prompt_ids"] + tokens["completion_ids"]
            active_samples.append((new_prefix, make_sample(tokens), step_idx))

    # Attach images once per sample using only the last merged step
    if vlm_cache is not None:
        key = output["example_id"] if cache_key is None else cache_key
        for _, sample, last_step_idx in active_samples:
            pv, shape, grids = vlm_cache.get_for_step(key, last_step_idx)
            sample.pixel_values = pv
            sample.pixel_values_shape = shape
            sample.image_grid_thw = grids
            if mm_token_type_ids_mapping is not None:
                sample.mm_token_type_ids = [
                    mm_token_type_ids_mapping.get(token_id, 0) for token_id in sample.prompt_ids + sample.completion_ids
                ]

    return [sample for _, sample, _ in active_samples]


# =============================================================================
# VLM-specific functions
# =============================================================================


_FILE_URL_PREFIX = "file://"


def offload_images_to_disk(rollouts: list[vf.RolloutOutput], output_dir: Path) -> int:
    """Replace base64 image data in rollout trajectories with file paths on disk.

    Scans all trajectory step prompts for data:image URLs, writes the decoded
    image bytes to ``{output_dir}/assets/images/{hash}.png``, and replaces the
    URL in-place with ``file://{path}``.  Deduplicates by content hash so each
    unique image is written only once.

    Returns the number of unique images written to disk.
    """
    images_dir = output_dir / "assets" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    written: set[str] = set()

    for output in rollouts:
        for step in output.get("trajectory", []):
            prompt = step.get("prompt")
            if not prompt or not isinstance(prompt, list):
                continue
            for msg in prompt:
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                for item in content:
                    if item.get("type") != "image_url":
                        continue
                    url = item.get("image_url", {}).get("url", "")
                    if not url.startswith("data:image"):
                        continue
                    b64_data = url.split(",", 1)[1]
                    content_hash = hashlib.sha256(b64_data.encode()).hexdigest()[:16]
                    path = images_dir / f"{content_hash}.png"
                    if content_hash not in written:
                        if not path.exists():
                            path.write_bytes(base64.b64decode(b64_data))
                        written.add(content_hash)
                    item["image_url"]["url"] = f"{_FILE_URL_PREFIX}{path}"

    return len(written)


def _load_file_image(path_str: str) -> Image.Image:
    """Load an image from a file:// path."""
    return Image.open(path_str.removeprefix(_FILE_URL_PREFIX))


def _extract_images_from_messages(messages: list) -> list[tuple[Image.Image, str]]:
    """Extract (image, key) pairs from OpenAI-style chat messages.

    Handles both base64 data URLs and file:// paths from disk offloading.
    """
    images = []
    if not messages or not isinstance(messages, list):
        return images

    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith(_FILE_URL_PREFIX):
                        img = _load_file_image(url)
                        images.append((img, url))
                    elif url.startswith("data:image"):
                        b64_data = url.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_data)
                        img = Image.open(BytesIO(img_bytes))
                        images.append((img, b64_data))
    return images


def _collect_image_keys_from_messages(messages: list) -> list[str]:
    """Extract image keys from OpenAI-style chat messages without decoding.

    Handles both base64 data URLs and file:// paths from disk offloading.
    """
    keys = []
    if not messages or not isinstance(messages, list):
        return keys
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        keys.append(url.split(",", 1)[1])
                    elif url.startswith(_FILE_URL_PREFIX):
                        keys.append(url)
    return keys


def _decode_image(key: str) -> Image.Image:
    """Decode an image from a base64 string or load from a file:// path."""
    if key.startswith(_FILE_URL_PREFIX):
        return _load_file_image(key)
    return Image.open(BytesIO(base64.b64decode(key)))


_PARALLEL_DECODE_THRESHOLD = 4


_IMAGE_STRIPPED_PLACEHOLDER = "[preprocessed image]"


def strip_base64_images(examples: list[tuple[int, vf.RolloutOutput]]) -> None:
    """Strip image data from rollout prompts to free memory.

    Handles both base64 data URLs and file:// paths from disk offloading.
    The images have been decoded and indexed; the original data is no longer needed.
    """
    for _, output in examples:
        for step in output.get("trajectory", []):
            prompt = step.get("prompt")
            if not prompt or not isinstance(prompt, list):
                continue
            for msg in prompt:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if url.startswith("data:image") or url.startswith(_FILE_URL_PREFIX):
                                item["image_url"]["url"] = _IMAGE_STRIPPED_PLACEHOLDER


def _extract_images_from_examples(
    examples: list[tuple[int, vf.RolloutOutput]],
) -> tuple[list[Image.Image], dict[int, list[list[int]]]]:
    """
    Extract images from all trajectory steps of each example.

    Two-pass approach: first collects unique base64 keys (fast, string-only),
    then decodes unique images in parallel via ThreadPoolExecutor.

    Args:
        examples: List of (cache_key, output) tuples where output contains a "trajectory"
            list with steps that have "prompt" messages in OpenAI chat format.

    Returns:
        Tuple of (all_images, step_image_indices_per_example)
        - all_images: deduplicated flat list of decoded PIL images
        - step_image_indices_per_example: dict mapping cache_key to per-step lists of
          indices into all_images (e.g., [[0], [0, 1], [1]] for the decreasing-images case)
    """
    # Pass 1: collect unique b64 keys and build step indices
    unique_keys: list[str] = []
    key_to_index: dict[str, int] = {}
    step_image_indices_per_example: dict[int, list[list[int]]] = {}

    for eid, output in examples:
        trajectory = output.get("trajectory", [])
        if not trajectory:
            step_image_indices_per_example[eid] = []
            continue

        step_image_indices = []
        for step in trajectory:
            prompt = step.get("prompt")
            image_keys = _collect_image_keys_from_messages(prompt)
            indices = []
            for key in image_keys:
                if key not in key_to_index:
                    key_to_index[key] = len(unique_keys)
                    unique_keys.append(key)
                indices.append(key_to_index[key])
            step_image_indices.append(indices)

        step_image_indices_per_example[eid] = step_image_indices

    # Pass 2: decode unique images (parallel when worthwhile)
    if len(unique_keys) > _PARALLEL_DECODE_THRESHOLD:
        with ThreadPoolExecutor(max_workers=min(len(unique_keys), 16)) as pool:
            all_images = list(pool.map(_decode_image, unique_keys))
    else:
        all_images = [_decode_image(k) for k in unique_keys]
    del unique_keys, key_to_index

    strip_base64_images(examples)

    return all_images, step_image_indices_per_example


_DEFAULT_IMAGE_CHUNK_SIZE = 32


class _ImageStore:
    """Holds per-unique-image data, assembled lazily on demand.

    Instead of duplicating pixel bytes for every step that references an image,
    we store each image's bytes once and assemble the concatenation at retrieval time.
    """

    def __init__(
        self,
        image_bytes: list[bytes],
        image_num_patches: list[int],
        patch_dim: int,
        image_grids: list[list[int]],
    ):
        self.image_bytes = image_bytes
        self.image_num_patches = image_num_patches
        self.patch_dim = patch_dim
        self.image_grids = image_grids
        self._cache: dict[tuple[int, ...], tuple[bytes, list[int], list[list[int]]]] = {}

    def assemble(self, indices: list[int]) -> tuple[bytes, list[int], list[list[int]]]:
        """Assemble pixel bytes, shape, and grids for a set of image indices.

        Results are cached by index tuple — multi-turn rollouts with the same
        cumulative image set (common across rollouts of the same example) hit
        the cache and skip the join.
        """
        cache_key = tuple(indices)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        total_patches = sum(self.image_num_patches[i] for i in indices)
        pixel_bytes = b"".join(self.image_bytes[i] for i in indices)
        shape = [total_patches, self.patch_dim]
        grids = [self.image_grids[i] for i in indices]
        result = (pixel_bytes, shape, grids)
        self._cache[cache_key] = result
        return result


def _preprocess_images_batched(
    images: list[Image.Image],
    step_image_indices_per_example: dict[int, list[list[int]]],
    processor,
    chunk_size: int = _DEFAULT_IMAGE_CHUNK_SIZE,
) -> tuple["_ImageStore | None", dict[int, list[list[int]]]]:
    """
    Preprocess all images in chunked batches, returning an _ImageStore and step indices.

    Images are processed in chunks to avoid OOM on large batches. Per-image bytes are
    stored once in the _ImageStore and assembled lazily at retrieval time.

    Returns:
        Tuple of (_ImageStore or None, step_image_indices_per_example).
        The store is None when there are no images or no processor.
    """
    if not images or processor is None:
        return None, step_image_indices_per_example

    logger = get_logger()
    image_sizes = [(img.width, img.height) for img in images]

    # Process images in chunks to avoid OOM, parallelized across threads
    # (PIL/numpy release the GIL so threads give real concurrency here)
    chunks = [images[i : i + chunk_size] for i in range(0, len(images), chunk_size)]

    def _process_chunk(chunk: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        processed = processor.image_processor(images=chunk, return_tensors="pt")
        return processed["pixel_values"], processed["image_grid_thw"]

    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=min(len(chunks), 8)) as pool:
            results = list(pool.map(_process_chunk, chunks))
    else:
        results = [_process_chunk(chunks[0])]

    # Free PIL images now that preprocessing is done
    del chunks
    images.clear()

    all_pixel_values_list = [r[0] for r in results]
    all_grid_thw_list = [r[1] for r in results]

    all_pixel_values = torch.cat(all_pixel_values_list, dim=0)
    all_grid_thw = torch.cat(all_grid_thw_list, dim=0)
    del all_pixel_values_list, all_grid_thw_list, results

    logger.debug(
        f"VLM image processing: {len(image_sizes)} images, sizes={image_sizes}, "
        f"pixel_values={all_pixel_values.shape}, grid_thw={all_grid_thw.tolist()}"
    )

    # Pre-compute patch start offset for each image
    patch_starts = [0]
    for g in all_grid_thw:
        patch_starts.append(patch_starts[-1] + int(g[0] * g[1] * g[2]))

    patch_dim = all_pixel_values.shape[1]

    # Convert to bytes per-image and free the tensor immediately after
    image_bytes_list: list[bytes] = []
    image_num_patches_list: list[int] = []
    image_grids_list: list[list[int]] = []
    for i in range(len(image_sizes)):
        img_slice = all_pixel_values[patch_starts[i] : patch_starts[i + 1]]
        image_bytes_list.append(img_slice.numpy().tobytes())
        image_num_patches_list.append(img_slice.shape[0])
        image_grids_list.append(all_grid_thw[i].tolist())
    del all_pixel_values, all_grid_thw

    store = _ImageStore(
        image_bytes=image_bytes_list,
        image_num_patches=image_num_patches_list,
        patch_dim=patch_dim,
        image_grids=image_grids_list,
    )

    return store, step_image_indices_per_example


class VLMImageCache:
    """Result of building VLM image cache with per-step image data."""

    def __init__(
        self,
        cache: dict[int, list[tuple[bytes | None, list[int] | None, list[list[int]] | None]]],
        num_unique_examples: int,
        extract_time: float,
        preprocess_time: float,
    ):
        self._store: _ImageStore | None = None
        self._step_indices: dict[int, list[list[int]]] | None = None
        self.cache = cache
        self.num_unique_examples = num_unique_examples
        self.num_unique_images = 0
        self.extract_time = extract_time
        self.preprocess_time = preprocess_time

    @classmethod
    def from_store(
        cls,
        store: _ImageStore | None,
        step_indices: dict[int, list[list[int]]],
        num_unique_examples: int,
        num_unique_images: int,
        extract_time: float,
        preprocess_time: float,
    ) -> "VLMImageCache":
        """Create a store-backed cache that assembles bytes lazily."""
        obj = cls.__new__(cls)
        obj._store = store
        obj._step_indices = step_indices
        obj.cache = {}
        obj.num_unique_examples = num_unique_examples
        obj.num_unique_images = num_unique_images
        obj.extract_time = extract_time
        obj.preprocess_time = preprocess_time
        return obj

    def _assemble(self, indices: list[int]) -> tuple[bytes | None, list[int] | None, list[list[int]] | None]:
        if not indices:
            return (None, None, None)
        return self._store.assemble(indices)

    def get_for_step(
        self, cache_key: int, step_idx: int
    ) -> tuple[bytes | None, list[int] | None, list[list[int]] | None]:
        """Get cumulative images up to and including the given step."""
        if self._store is not None:
            steps = self._step_indices.get(cache_key, [])
            if not steps or step_idx >= len(steps):
                return (None, None, None)
            return self._assemble(steps[step_idx])

        steps = self.cache.get(cache_key, [])
        if not steps or step_idx >= len(steps):
            return (None, None, None)
        return steps[step_idx]

    def get_all(self, cache_key: int) -> tuple[bytes | None, list[int] | None, list[list[int]] | None]:
        """Get all images for the cache key (last step's cumulative images)."""
        if self._store is not None:
            steps = self._step_indices.get(cache_key, [])
            if not steps:
                return (None, None, None)
            return self._assemble(steps[-1])

        steps = self.cache.get(cache_key, [])
        if not steps:
            return (None, None, None)
        return steps[-1]


def build_vlm_image_cache(rollouts: list[vf.RolloutOutput], processor) -> VLMImageCache:
    """
    Build image cache for VLM training by extracting and preprocessing images.

    Caches per rollout to keep images aligned with divergent multi-turn trajectories.
    """
    examples = [(idx, rollout) for idx, rollout in enumerate(rollouts)]
    unique_example_ids = {rollout["example_id"] for rollout in rollouts}

    # Extract images (also strips base64 data from rollout prompts to free memory)
    extract_start = time.perf_counter()
    all_images, images_per_example = _extract_images_from_examples(examples)
    num_unique_images = len(all_images)
    extract_time = time.perf_counter() - extract_start

    # Preprocess images (clears PIL image list when done)
    preprocess_start = time.perf_counter()
    store, step_indices = _preprocess_images_batched(all_images, images_per_example, processor)
    preprocess_time = time.perf_counter() - preprocess_start

    return VLMImageCache.from_store(
        store=store,
        step_indices=step_indices,
        num_unique_examples=len(unique_example_ids),
        num_unique_images=num_unique_images,
        extract_time=extract_time,
        preprocess_time=preprocess_time,
    )
