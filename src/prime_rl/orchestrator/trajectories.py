import base64
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pybase64
import torch
import verifiers as vf
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.orchestrator import EchoConfig
from prime_rl.transport import RoutedExperts, TrainingSample
from prime_rl.utils.chat_template import (
    common_prefix_len,
    deserialize_tool_calls,
    normalize_messages,
    render_messages,
    strip_message_content,
)
from prime_rl.utils.logger import get_logger

# We use list() instead of deepcopy() for flat lists (token IDs, logprobs) - safe because
# primitives are immutable. mm_kwargs payloads are not mutated after creation.


def align_routed_experts(
    routed_experts: np.ndarray | None,
    expected_len: int,
) -> np.ndarray | None:
    """Align routed_experts length with the expected token count.

    VLLM's capturer uses `num_tokens - 1` slot mappings because the final
    generated token was never fed as input to a forward pass and has no
    routing decision. Append zero-filled entries for the missing positions.
    """
    if routed_experts is None:
        return routed_experts
    assert routed_experts.ndim == 3
    if routed_experts.shape[0] > expected_len:
        return np.ascontiguousarray(routed_experts[:expected_len])
    deficit = expected_len - routed_experts.shape[0]
    if deficit <= 0:
        return routed_experts
    padding = np.zeros((deficit, routed_experts.shape[1], routed_experts.shape[2]), dtype=routed_experts.dtype)
    return np.concatenate((routed_experts, padding), axis=0)


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
) -> list[int]:
    return render_messages(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
    )


def _tokenize_step_from_messages(
    step: vf.TrajectoryStep,
    tokenizer: PreTrainedTokenizer,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prompt = _normalize_messages(step.get("prompt"), default_role="user")
    completion = _normalize_messages(step.get("completion"), default_role="assistant")

    prompt = _strip_message_content(_deserialize_tool_calls(prompt))
    completion = _strip_message_content(_deserialize_tool_calls(completion))

    assert all(m.get("role") == "assistant" for m in completion), (
        "Expected all completion messages to be assistant role for SFT distillation, "
        f"got roles: {[m.get('role') for m in completion]}"
    )

    all_messages = prompt + completion
    prompt_has_assistant_completion = len(completion) > 0 and completion[0].get("role") == "assistant"
    prompt_ids = _render_messages(
        tokenizer,
        prompt,
        add_generation_prompt=prompt_has_assistant_completion,
        tools=tools,
    )
    full_ids = _render_messages(
        tokenizer,
        all_messages,
        tools=tools,
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


def _convert_tools_to_oai_format(tool_defs: list) -> list[dict[str, Any]] | None:
    """Convert verifiers Tool objects or dicts to OAI function-calling format."""
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
                "parameters": _get(tool, "parameters"),
                **({} if _get(tool, "strict") is None else {"strict": _get(tool, "strict")}),
            },
        }
        for tool in tool_defs
    ]


def _tokenize_step_with_renderer(
    step: vf.TrajectoryStep,
    renderer,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Tokenize a trajectory step using a Renderer."""
    from renderers.base import build_trajectory_step

    prompt = _normalize_messages(step.get("prompt"), default_role="user")
    completion = _normalize_messages(step.get("completion"), default_role="assistant")
    prompt = _strip_message_content(_deserialize_tool_calls(prompt))
    completion = _strip_message_content(_deserialize_tool_calls(completion))
    return build_trajectory_step(renderer, prompt, completion, tools=tools)


def backfill_rollout_tokens(
    output: vf.RolloutOutput,
    tokenizer: PreTrainedTokenizer,
    renderer=None,
) -> bool:
    """Populate missing step tokens from prompt/completion messages.

    When a renderer is provided, uses it for tokenization (faster, deterministic).
    Otherwise falls back to the tokenizer + apply_chat_template path.
    """
    if all(step["tokens"] is not None for step in output["trajectory"]):
        return True

    logger = get_logger()
    tools = _convert_tools_to_oai_format(output.get("tool_defs", []))

    for step_idx, step in enumerate(output["trajectory"]):
        if step["tokens"] is not None:
            continue

        if renderer is not None:
            step["tokens"] = _tokenize_step_with_renderer(step, renderer, tools=tools)
        else:
            reconstructed = _tokenize_step_from_messages(step, tokenizer, tools=tools)
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


def _step_echo_alpha(
    prompt_attribution: dict | None,
    prompt_len: int,
    completion_len: int,
    echo_config: EchoConfig | None,
    filter_mask: list[bool] | None = None,
) -> list[float | None]:
    """Per-token echo alpha for one trajectory step, length ``prompt_len +
    completion_len``. Each entry is ``None`` (not echoed — RL applies) or a
    ``float`` (echoed at that alpha, overriding the RL advantage and flipping
    loss_mask=True in ``prepare_sample``); ``alpha=0`` is a real kill-RL value,
    distinct from ``None``.

    Prompt-side positions resolve their role via
    ``message_indices[k] → message_roles[mi]`` and take that role's alpha;
    ``is_content[k]`` excludes template scaffold, and tool positions also match
    ``tool_names``. Completion-side positions are always assistant-role.
    ``filter_mask`` (optional, same length) narrows the baseline by AND — False
    drops a position to ``None`` (never adds echo); a wrong length raises.

    Returns all-None when echo is disabled or attribution is missing.
    """
    expected_total_len = prompt_len + completion_len
    if filter_mask is not None and len(filter_mask) != expected_total_len:
        raise ValueError(
            f"filter_mask length {len(filter_mask)} does not match prompt_len + completion_len = {expected_total_len}"
        )

    def _build_baseline() -> list[float | None]:
        """Build the role-level echo_alpha baseline. Multiple early returns;
        the outer scope then applies ``filter_mask`` at a single exit point."""
        out: list[float | None] = [None] * expected_total_len
        if echo_config is None:
            return out

        # Completion-side first (independent of prompt_attribution — the
        # completion is always assistant-role by construction). When
        # assistant echo is enabled, every completion position carries
        # that alpha.
        if echo_config.assistant is not None:
            assistant_alpha = echo_config.assistant.alpha
            for k in range(prompt_len, expected_total_len):
                out[k] = assistant_alpha

        # Prompt-side requires the renderer attribution.
        if prompt_attribution is None:
            return out

        # prompt_attribution arrives as a dict through the verifiers
        # env-server JSON boundary even though the renderer emits a
        # RenderedTokens object.
        message_roles = prompt_attribution.get("message_roles")
        if message_roles is None:
            return out

        message_indices = prompt_attribution.get("message_indices")
        is_content = prompt_attribution.get("is_content")
        # Defensive: if the renderer didn't populate is_content (DefaultRenderer
        # leaves it empty) or sizes don't match, we can't tell body from
        # scaffold — bail to the completion-side-only mask.
        if not is_content or len(is_content) != prompt_len:
            return out
        if not message_indices or len(message_indices) != prompt_len:
            return out

        # Resolve per-role alphas once.
        system_alpha = echo_config.system.alpha if echo_config.system is not None else None
        user_alpha = echo_config.user.alpha if echo_config.user is not None else None
        assistant_alpha_prompt = echo_config.assistant.alpha if echo_config.assistant is not None else None
        tool_role_config = echo_config.tool
        tool_alpha = tool_role_config.alpha if tool_role_config is not None else None
        enabled_tools = (
            set(tool_role_config.tool_names) if tool_role_config is not None and tool_role_config.tool_names else None
        )

        # Tool-role check needs the per-message function name; safe-get
        # since message_tool_names may be absent on non-tool-aware renderers.
        message_tool_names = prompt_attribution.get("message_tool_names") or []

        for k in range(prompt_len):
            mi = message_indices[k]
            if mi < 0 or not is_content[k]:
                continue
            if mi >= len(message_roles):
                continue
            role = message_roles[mi]
            if role == "system":
                if system_alpha is not None:
                    out[k] = system_alpha
            elif role == "user":
                if user_alpha is not None:
                    out[k] = user_alpha
            elif role == "assistant":
                if assistant_alpha_prompt is not None:
                    out[k] = assistant_alpha_prompt
            elif role == "tool":
                if tool_alpha is None:
                    continue
                # Per-tool-name filter: enabled_tools=None means "all tools".
                name = message_tool_names[mi] if mi < len(message_tool_names) else None
                if enabled_tools is None or (name is not None and name in enabled_tools):
                    out[k] = tool_alpha
            # Unknown roles silently skipped.
        return out

    out = _build_baseline()

    # AND-compose with the user filter (when provided). The filter can only
    # narrow the role baseline — positions where filter_mask[k] is False
    # get dropped to None, but None positions stay None regardless of
    # filter_mask[k] (a True filter result cannot "add" echo). Strictly
    # narrowing.
    if filter_mask is not None:
        for k in range(expected_total_len):
            if not filter_mask[k]:
                out[k] = None

    return out


def apply_echo_filter(
    rollout: vf.RolloutOutput,
    filter_fn: Callable[..., list[list[bool]]],
) -> list[list[bool]]:
    """Invoke the user's echo filter and validate its return, returning the
    per-step masks for :func:`interleave_rollout`. Enforces the
    :class:`EchoFilterConfig` contract loudly: ``TypeError`` for a non-list /
    non-bool return, ``ValueError`` for an outer length ≠ trajectory steps or an
    inner length ≠ that step's ``prompt_ids + completion_ids``; the filter's own
    exceptions propagate.
    """
    trajectory = rollout["trajectory"]
    result = filter_fn(rollout)

    if not isinstance(result, list):
        raise TypeError(f"echo filter must return list[list[bool]], got {type(result).__name__}")
    if len(result) != len(trajectory):
        raise ValueError(
            f"echo filter returned {len(result)} per-step masks but the rollout has {len(trajectory)} trajectory steps"
        )

    for step_idx, (step, mask) in enumerate(zip(trajectory, result)):
        tokens = step["tokens"]
        prompt_len = len(tokens["prompt_ids"])
        completion_len = len(tokens["completion_ids"])
        expected = prompt_len + completion_len

        if not isinstance(mask, list):
            raise TypeError(f"echo filter step {step_idx}: mask must be a list, got {type(mask).__name__}")
        if len(mask) != expected:
            raise ValueError(
                f"echo filter step {step_idx}: mask length {len(mask)} "
                f"!= expected {expected} "
                f"(prompt_len={prompt_len}, completion_len={completion_len})"
            )
        for k, v in enumerate(mask):
            # Reject bool subclasses other than the literal bool type? No —
            # ``isinstance(v, bool)`` accepts True/False only (np.bool_ would
            # need its own handling; cast it at the filter boundary). We
            # intentionally do NOT accept int 0/1: numeric types passing as
            # bool is a common source of silent bugs.
            if type(v) is not bool:
                raise TypeError(
                    f"echo filter step {step_idx}: mask[{k}] must be a plain bool, got {type(v).__name__} ({v!r})"
                )

    return result


def interleave_rollout(
    output: vf.RolloutOutput,
    mm_token_type_ids_mapping: dict[int, int] | None = None,
    *,
    env_name: str = "",
    echo_config: EchoConfig | None = None,
    filter_masks: list[list[bool]] | None = None,
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

    For VLM models, each renderer-produced trajectory step carries its
    per-image processed tensors inline on ``multi_modal_data``; the last
    merged step's sidecar covers every image in the sample.

    When ``echo_config`` is provided, each sample carries a per-token
    ``echo_alpha`` array (see :func:`_step_echo_alpha`), extended across merged steps.

    Args:
        output: vf.RolloutOutput containing trajectory data
        mm_token_type_ids_mapping: Maps prompt-token ids to mm_token_type_ids
            (1 = image, 2 = video, 0 otherwise). Renderer-supplied.
        echo_config: Per-env echo config (None when echo is disabled for this
            env). Caller resolves it from ``train_envs.get(env_name).config.echo``.
        filter_masks: Optional per-step bool masks that narrow the role baseline
            (``None`` = no filter). See :func:`apply_echo_filter`.
    """
    logger = get_logger()

    trajectory = output["trajectory"]
    if filter_masks is not None and len(filter_masks) != len(trajectory):
        raise ValueError(f"filter_masks outer length {len(filter_masks)} != trajectory length {len(trajectory)}")
    if len(trajectory) == 0:
        error = output.get("error")
        stop = output.get("stop_condition")
        logger.warning(
            f"No trajectory steps for example {output['example_id']} (error={error}, stop={stop}). Skipping rollout."
        )
        return None

    has_error = output["error"] is not None
    # completion_temperatures is left empty; the train sink fills it per-env later.

    def prepare_step_tokens(step: vf.TrajectoryStep, step_idx: int) -> dict[str, Any] | None:
        tokens = step["tokens"]
        if tokens is not None:
            routed_experts_payload = tokens.get("routed_experts")
            routed_experts = None
            routed_experts_start = None
            if routed_experts_payload is not None:
                decoded_routed_experts = pybase64.b64decode_as_bytearray(routed_experts_payload["data"])
                routed_experts = np.frombuffer(decoded_routed_experts, dtype=np.uint8).reshape(
                    routed_experts_payload["shape"]
                )
                routed_experts_start = routed_experts_payload["start"]

            prompt_ids = list(tokens["prompt_ids"])
            completion_ids = list(tokens["completion_ids"])
            step_filter_mask = filter_masks[step_idx] if filter_masks is not None else None
            echo_alpha = _step_echo_alpha(
                prompt_attribution=tokens.get("prompt_attribution"),
                prompt_len=len(prompt_ids),
                completion_len=len(completion_ids),
                echo_config=echo_config,
                filter_mask=step_filter_mask,
            )
            return {
                "prompt_ids": prompt_ids,
                "prompt_mask": list(map(bool, tokens["prompt_mask"])),
                "completion_ids": completion_ids,
                "completion_mask": list(map(bool, tokens["completion_mask"])),
                "completion_logprobs": list(tokens["completion_logprobs"]),
                "routed_experts": routed_experts,
                "routed_experts_start": routed_experts_start,
                # Renderer-emitted multimodal sidecar (placeholders + per-item
                # processed tensors). Populated when the rollout went through
                # a multimodal-aware renderer (e.g. Qwen3VLRenderer); absent
                # for text-only rollouts.
                "multi_modal_data": tokens.get("multi_modal_data"),
                "echo_alpha": echo_alpha,
            }

        logger.warning(f"Missing rollout tokens for example {output['example_id']} step {step_idx}.")
        return None

    prepared_steps: list[dict[str, Any]] = []
    for step_idx, step in enumerate(trajectory):
        prepared = prepare_step_tokens(step, step_idx)
        if prepared is None:
            return None
        prepared_steps.append(prepared)

    # Deferred routed_experts state per sample: O(N) chunk list concatenated
    # once at finalize, replacing the prior O(N²) per-extension unpack/repack.
    sample_routed_state: dict[int, dict[str, Any]] = {}
    routed_prefix_states: dict[int, list[tuple[list[int], list[int], dict[str, Any]]]] = {}

    # Track (prefix_tokens, sample, step_indices) per active sample. step_indices
    # is the explicit list of prepared_steps positions merged into this sample —
    # non-contiguous when other agents' steps interleave.
    active_samples: list[tuple[list[int], TrainingSample, list[int]]] = []

    def make_sample(tokens: dict[str, Any], step_idx: int) -> TrainingSample:
        """Create a new TrainingSample from a trajectory step."""
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = list(tokens["completion_mask"])
        completion_ids = list(tokens["completion_ids"])

        prompt_ids = list(tokens["prompt_ids"])
        # ``echo_alpha`` was computed per-step against the env's EchoConfig.
        # When echo is disabled (echo_config is None or no role enabled) the
        # helper returns an all-None list; carry None on the sample in that
        # case to keep the transport payload lean.
        step_echo_alpha = tokens.get("echo_alpha")
        sample_echo_alpha = (
            list(step_echo_alpha) if step_echo_alpha and any(a is not None for a in step_echo_alpha) else None
        )
        sample = TrainingSample(
            prompt_ids=prompt_ids,
            prompt_mask=list(tokens["prompt_mask"]),
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=list(tokens["completion_logprobs"]),
            completion_temperatures=[],
            teacher_logprobs=None,
            advantage=None,
            env_name=env_name,
            mm_token_type_ids=None,
            routed_experts=None,  # deferred — finalized at end of interleave_rollout
            echo_alpha=sample_echo_alpha,
        )
        # Initialize routed-experts state for this sample. First chunk is the
        # raw step routed_experts (no pad, no copy). running_len is the
        # cumulative count across chunks; tracked so the boundary fix-up at
        # each extension is a no-op append rather than a destructive write.
        step_routed = tokens.get("routed_experts")
        if step_routed is not None:
            routed_start = tokens["routed_experts_start"]
            assert routed_start is not None, f"Missing routed_experts_start for step {step_idx}"
            chunks: list[np.ndarray] = []
            running_len = 0
            if routed_start > 0:
                source_len = routed_start + 1
                assert source_len in routed_prefix_states, (
                    f"Missing routed prefix state for step {step_idx}: "
                    f"routed_start={routed_start}, prompt_len={len(tokens['prompt_ids'])}"
                )
                source_state = None
                for prompt_ids, completion_ids, candidate_state in routed_prefix_states[source_len]:
                    prompt_len = len(prompt_ids)
                    if (
                        tokens["prompt_ids"][:prompt_len] == prompt_ids
                        and tokens["prompt_ids"][prompt_len:source_len] == completion_ids
                    ):
                        source_state = candidate_state
                        break
                assert source_state is not None, (
                    f"No matching routed prefix for step {step_idx}: "
                    f"routed_start={routed_start}, prompt_len={len(tokens['prompt_ids'])}"
                )
                assert source_state["running_len"] >= routed_start, (
                    f"Routed prefix too short for step {step_idx}: "
                    f"running_len={source_state['running_len']}, routed_start={routed_start}"
                )
                remaining = routed_start
                for chunk in source_state["chunks"]:
                    if remaining == 0:
                        break
                    take = min(remaining, int(chunk.shape[0]))
                    chunks.append(chunk[:take])
                    remaining -= take
                assert remaining == 0, (
                    f"Could not reconstruct routed prefix for step {step_idx}: "
                    f"remaining={remaining}, routed_start={routed_start}"
                )
                running_len = routed_start
            chunks.append(step_routed)
            running_len += int(step_routed.shape[0])
            sample_routed_state[id(sample)] = {
                "chunks": chunks,
                "running_len": running_len,
            }
        return sample

    def extend_sample(
        sample: TrainingSample,
        prefix_len: int,
        step_idx: int,
    ) -> None:
        """Extend an existing sample with a new trajectory step (extension property holds)."""
        tokens = prepared_steps[step_idx]

        # Extend with new prompt tokens (mask=False, no gradient)
        new_prompt_ids = tokens["prompt_ids"][prefix_len:]
        sample.completion_ids.extend(new_prompt_ids)
        sample.completion_mask.extend([False] * len(new_prompt_ids))
        sample.completion_logprobs.extend([0.0] * len(new_prompt_ids))

        # Extend with new completion tokens
        completion_ids = tokens["completion_ids"]
        sample.completion_ids.extend(completion_ids)
        if has_error:
            sample.completion_mask.extend([False] * len(tokens["completion_mask"]))
        else:
            sample.completion_mask.extend(tokens["completion_mask"])
        sample.completion_logprobs.extend(tokens["completion_logprobs"])

        # Extend echo_alpha in lockstep (prompt tail + completion), materializing
        # a previously-None array only once there's real echo signal to record.
        step_echo_alpha = tokens.get("echo_alpha")
        if step_echo_alpha is not None:
            step_prompt_len = len(tokens["prompt_ids"])
            new_prompt_echo = step_echo_alpha[prefix_len:step_prompt_len]
            new_completion_echo = step_echo_alpha[step_prompt_len:]
            extension = new_prompt_echo + new_completion_echo
            if any(a is not None for a in extension) or sample.echo_alpha is not None:
                if sample.echo_alpha is None:
                    existing_len = len(sample.prompt_ids) + len(sample.completion_ids) - len(extension)
                    sample.echo_alpha = [None] * existing_len
                sample.echo_alpha.extend(extension)

        step_routed = tokens.get("routed_experts")
        state = sample_routed_state.get(id(sample))
        if state is not None:
            assert step_routed is not None, f"Missing routed experts for routed sample extension at step {step_idx}"
        if step_routed is not None:
            assert state is not None, f"Unexpected routed experts for unrouted sample at step {step_idx}"
            assert tokens["routed_experts_start"] == prefix_len - 1, (
                f"Routed experts delta start mismatch at step {step_idx}: "
                f"start={tokens['routed_experts_start']}, expected={prefix_len - 1}, prefix_len={prefix_len}"
            )
            # Delta payloads start at prefix_len - 1. Row 0 fills the boundary
            # token missing from the previous request; the rest is the new suffix.
            if prefix_len > 0:
                boundary_chunk = step_routed[:1]
                state["chunks"].append(boundary_chunk)
                state["running_len"] += 1
                step_routed = step_routed[1:]
            new_chunk = step_routed
            state["chunks"].append(new_chunk)
            state["running_len"] += int(new_chunk.shape[0])

    first_tokens = prepared_steps[0]
    first_prefix = first_tokens["prompt_ids"] + first_tokens["completion_ids"]
    first_sample = make_sample(first_tokens, step_idx=0)
    active_samples.append((first_prefix, first_sample, [0]))
    first_routed_state = sample_routed_state.get(id(first_sample))
    if first_routed_state is not None:
        routed_prefix_states.setdefault(len(first_prefix), []).append(
            (first_tokens["prompt_ids"], first_tokens["completion_ids"], first_routed_state)
        )

    for step_idx, _step in enumerate(trajectory[1:], start=1):
        tokens = prepared_steps[step_idx]
        step_prompt_ids = tokens["prompt_ids"]

        # Pick the *longest* matching active prefix. With compaction/rollback,
        # one active sample's prefix can be a strict prefix of another (e.g. a
        # later sample re-generated tokens that overlap an earlier sample's
        # prefix). Both would satisfy the slice check; the shorter would
        # silently absorb the longer sample's generated tokens as user input.
        matched_idx = None
        matched_len = -1
        matching_prefix_lens: list[int] = []
        for idx, (prefix_tokens, _, _) in enumerate(active_samples):
            pl = len(prefix_tokens)
            if step_prompt_ids[:pl] == prefix_tokens:
                matching_prefix_lens.append(pl)
                if pl > matched_len:
                    matched_idx = idx
                    matched_len = pl

        if len(matching_prefix_lens) > 1:
            # Ambiguous extension: rare, but reachable via compaction/rollback
            # where a new sample's prefix happens to start with an older
            # sample's prefix. Longest-match is the correct choice; surface
            # the ambiguity so we can audit if it shows up in real rollouts.
            logger.warning(
                f"Ambiguous prefix match at step {step_idx} for example {output['example_id']}: "
                f"{len(matching_prefix_lens)} of {len(active_samples)} active prefixes match "
                f"(lens={sorted(matching_prefix_lens)}, step_prompt_len={len(step_prompt_ids)}). "
                f"Extending the longest (len={matched_len})."
            )

        if matched_idx is not None:
            # Extension holds - merge into matched sample
            prefix_tokens, sample, step_indices = active_samples[matched_idx]
            extend_sample(sample, len(prefix_tokens), step_idx=step_idx)
            new_prefix = tokens["prompt_ids"] + tokens["completion_ids"]
            active_samples[matched_idx] = (
                new_prefix,
                sample,
                step_indices + [step_idx],
            )
            routed_state = sample_routed_state.get(id(sample))
            if routed_state is not None:
                routed_prefix_states.setdefault(len(new_prefix), []).append(
                    (tokens["prompt_ids"], tokens["completion_ids"], routed_state)
                )
        else:
            # No prefix matches - start a new sample
            logger.debug(
                f"Extension property broke at step {step_idx + 1} for example {output['example_id']}. "
                f"Starting new sample (active_prefixes={len(active_samples)}, step_prompt_len={len(step_prompt_ids)})."
            )
            new_prefix = tokens["prompt_ids"] + tokens["completion_ids"]
            sample = make_sample(tokens, step_idx=step_idx)
            active_samples.append((new_prefix, sample, [step_idx]))
            routed_state = sample_routed_state.get(id(sample))
            if routed_state is not None:
                routed_prefix_states.setdefault(len(new_prefix), []).append(
                    (tokens["prompt_ids"], tokens["completion_ids"], routed_state)
                )

    # Finalize routed_experts for each sample. One concat per sample (O(N) byte
    # work) replaces the previous per-step unpack/concat/repack (O(N²)). The
    # boundary entries between steps were already inserted as one-entry chunks
    # during extend_sample, so a straight concat is correct.
    for _, sample, _ in active_samples:
        state = sample_routed_state.get(id(sample))
        if state is None:
            continue
        chunks = state["chunks"]
        if not chunks:
            continue
        combined = np.concatenate(chunks, axis=0) if len(chunks) > 1 else np.ascontiguousarray(chunks[0])
        expected_len = len(sample.prompt_ids) + len(sample.completion_ids)
        combined = align_routed_experts(combined, expected_len)
        combined = np.ascontiguousarray(combined)
        sample.routed_experts = RoutedExperts(
            data=combined.tobytes(),
            shape=list(combined.shape),
            dtype=str(combined.dtype),
        )

    # Attach images by concatenating mm_items across every step the
    # sample covers. verifiers' ``state_to_output`` ships per-step
    # *delta* mm_data (each step contains only items not present in the
    # prior step's cumulative set, with multiset-aware dedup), so
    # reading the last step alone would miss every earlier-turn image.
    # Concat in step order recovers the per-sample cumulative set;
    # deduping again here would drop legitimate duplicate placeholders.
    for _, sample, step_indices in active_samples:
        renderer_mm = _union_step_mm_data(prepared_steps, step_indices)
        if renderer_mm is not None:
            mm_kwargs = _pack_mm_kwargs_from_renderer(renderer_mm)
            if mm_kwargs is not None:
                sample.mm_kwargs = mm_kwargs
                # ``mm_token_type_ids``: 1 for image-placeholder tokens, 2
                # for video, 0 otherwise. Renderer-supplied via
                # ``mm_token_type_id_map`` (single source of truth).
                if mm_token_type_ids_mapping is not None:
                    sample.mm_token_type_ids = [
                        mm_token_type_ids_mapping.get(token_id, 0)
                        for token_id in sample.prompt_ids + sample.completion_ids
                    ]

    return [sample for _, sample, _ in active_samples]


def _union_step_mm_data(
    prepared_steps: list[dict[str, Any]],
    step_indices: list[int],
) -> "dict[str, Any] | None":
    """Concatenate renderer-emitted mm_items across this sample's owned steps.

    ``step_indices`` lists exactly the ``prepared_steps`` positions merged into
    the sample — explicit, not a range, so interleaved-agent trajectories skip
    steps owned by other agents.

    Verifiers ≥ c7731bbb ships per-step *delta* mm_data instead of
    cumulative — see ``verifiers/utils/save_utils.py::_delta_intermediate_mm_data``.
    The cross-step dedup is already done there with multiset semantics
    (preserving multiplicity for an image that appears in multiple
    placeholder runs in the token stream). We just concatenate in step
    order to recover the per-sample cumulative; deduping again here
    would drop legitimate duplicate placeholders.
    """
    union_items: dict[str, list] = {}
    union_hashes: dict[str, list] = {}
    has_any = False
    for i in step_indices:
        mm = prepared_steps[i].get("multi_modal_data")
        if mm is None:
            continue
        items = mm.mm_items if hasattr(mm, "mm_items") else (mm or {}).get("mm_items") or {}
        hashes = mm.mm_hashes if hasattr(mm, "mm_hashes") else (mm or {}).get("mm_hashes") or {}
        for modality, item_lst in items.items():
            hash_lst = hashes.get(modality, []) or []
            for j, item in enumerate(item_lst or []):
                h = hash_lst[j] if j < len(hash_lst) else None
                union_items.setdefault(modality, []).append(item)
                union_hashes.setdefault(modality, []).append(h)
                has_any = True
    if not has_any:
        return None
    return {"mm_items": union_items, "mm_hashes": union_hashes}


def _pack_mm_kwargs_from_renderer(mm_data: Any) -> "dict[str, Any] | None":
    """Batch the renderer's per-image ``mm_items`` into model-agnostic
    forward kwargs.

    ``mm_data`` may arrive as a ``MultiModalData`` instance (in-process
    for tests) or as a plain dict (after msgpack round-trip from the
    env-worker). Each item is a dict keyed by the names the model's
    ``forward`` expects (``pixel_values`` + ``image_grid_thw`` for
    Qwen3-VL, just ``pixel_values`` for Gemma3-VL, etc.). We batch by
    ``torch.cat(..., dim=0)`` per key — generic because every HF VLM
    processor emits a leading batch/patch dimension, and the renderer
    always processes one image per call.

    Returns a dict of ``EncodedTensor`` payloads keyed by kwarg name,
    or ``None`` when no multimodal data is present.
    """
    from verifiers.utils.serve_utils import decode_tensor_payload

    from prime_rl.transport.types import EncodedTensor

    mm_items = mm_data.mm_items if hasattr(mm_data, "mm_items") else (mm_data or {}).get("mm_items") or {}
    # Flatten across modalities into one kwarg dict — the model's
    # forward signature is the schema. ``mm_items`` is typically
    # ``{"image": [...], "video": [...]}`` but each modality's keys
    # don't collide for any HF VLM we ship today.
    per_kwarg: dict[str, list] = {}
    for _modality, items in mm_items.items():
        for item in items or []:
            for key, payload in item.items():
                per_kwarg.setdefault(key, []).append(decode_tensor_payload(payload))
    if not per_kwarg:
        return None
    out: dict[str, EncodedTensor] = {}
    for key, tensors in per_kwarg.items():
        cat = torch.cat(tensors, dim=0).contiguous()
        arr = cat.detach().cpu().numpy()
        out[key] = EncodedTensor(
            dtype=str(arr.dtype),
            shape=list(arr.shape),
            data=arr.tobytes(),
        )
    return out


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
