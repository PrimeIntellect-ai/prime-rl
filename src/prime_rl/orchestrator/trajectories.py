import base64
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pybase64
import torch
import verifiers as vf
from transformers.tokenization_utils import PreTrainedTokenizer

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


def pretokenize_rollout_trajectory(
    output: vf.RolloutOutput,
    tokenizer: PreTrainedTokenizer,
    renderer=None,
) -> bool:
    """Populate missing step tokens from prompt/completion messages.

    When a renderer is provided, uses it for tokenization (faster, deterministic).
    Otherwise falls back to the tokenizer + apply_chat_template path.
    """
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


def interleave_rollout(
    output: vf.RolloutOutput,
    mm_token_type_ids_mapping: dict[int, int] | None = None,
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

    def prepare_step_tokens(step: vf.TrajectoryStep, step_idx: int) -> dict[str, Any] | None:
        tokens = step["tokens"]
        if tokens is not None:
            routed_experts_payload = tokens.get("routed_experts")
            routed_experts = None
            if routed_experts_payload is not None:
                shape = [int(dim) for dim in routed_experts_payload["shape"]]
                decoded_routed_experts = pybase64.b64decode_as_bytearray(routed_experts_payload["data"])
                expected_size = int(np.prod(shape, dtype=np.int64))
                assert len(decoded_routed_experts) == expected_size, (
                    len(decoded_routed_experts),
                    expected_size,
                    shape,
                )
                routed_experts = np.frombuffer(decoded_routed_experts, dtype=np.uint8).reshape(shape)
                assert routed_experts.ndim == 3

            return {
                "prompt_ids": list(tokens["prompt_ids"]),
                "prompt_mask": [bool(i) for i in tokens["prompt_mask"]],
                "completion_ids": list(tokens["completion_ids"]),
                "completion_mask": [bool(i) for i in tokens["completion_mask"]],
                "completion_logprobs": list(tokens["completion_logprobs"]),
                "routed_experts": routed_experts,
                # Renderer-emitted multimodal sidecar (placeholders + per-item
                # processed tensors). Populated when the rollout went through
                # a multimodal-aware renderer (e.g. Qwen3VLRenderer); absent
                # for text-only rollouts.
                "multi_modal_data": tokens.get("multi_modal_data"),
            }

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
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]
        completion_ids = list(tokens["completion_ids"])

        routed_experts = align_routed_experts(
            tokens.get("routed_experts"),
            len(tokens["prompt_ids"]) + len(tokens["completion_ids"]),
        )
        prompt_ids = list(tokens["prompt_ids"])
        sample = TrainingSample(
            prompt_ids=prompt_ids,
            prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=list(tokens["completion_logprobs"]),
            completion_temperatures=[temperature] * len(completion_ids),
            teacher_logprobs=None,
            advantage=None,
            env_name=output["env_name"],
            mm_token_type_ids=None,
        )
        if routed_experts is not None:
            routed_experts = np.ascontiguousarray(routed_experts)
            sample.routed_experts = RoutedExperts(
                data=routed_experts.tobytes(),
                shape=list(routed_experts.shape),
                dtype=str(routed_experts.dtype),
            )
        return sample, routed_experts

    def extend_sample(
        sample: TrainingSample,
        sample_routed_experts: np.ndarray | None,
        prefix_len: int,
        step_idx: int,
    ) -> np.ndarray | None:
        """Extend an existing sample with a new trajectory step (extension property holds)."""
        tokens = prepared_steps[step_idx]

        # Extend with new prompt tokens (mask=False, no gradient)
        new_prompt_ids = tokens["prompt_ids"][prefix_len:]
        sample.completion_ids.extend(new_prompt_ids)
        sample.completion_mask.extend([False] * len(new_prompt_ids))
        sample.completion_logprobs.extend([0.0] * len(new_prompt_ids))
        sample.completion_temperatures.extend([temperature] * len(new_prompt_ids))

        # Extend with new completion tokens
        completion_ids = tokens["completion_ids"]
        sample.completion_ids.extend(completion_ids)
        if has_error:
            sample.completion_mask.extend([False] * len(tokens["completion_mask"]))
        else:
            sample.completion_mask.extend(bool(i) for i in tokens["completion_mask"])
        sample.completion_logprobs.extend(tokens["completion_logprobs"])
        sample.completion_temperatures.extend([temperature] * len(completion_ids))

        if tokens.get("routed_experts") is not None and sample_routed_experts is not None:
            step_routed = tokens["routed_experts"]
            # The previous step's last routing entry was zero-padded by align_routed_experts
            # (vLLM only captures num_tokens-1 routings per request). This step actually
            # processed that boundary token as part of its prompt, so replace the zero-fill
            # with the real routing decision before appending new entries.
            if prefix_len > 0 and prefix_len <= step_routed.shape[0]:
                sample_routed_experts[prefix_len - 1] = step_routed[prefix_len - 1]
            sample_routed_experts = np.concatenate((sample_routed_experts, step_routed[prefix_len:]), axis=0)
            expected_len = len(sample.prompt_ids) + len(sample.completion_ids)
            sample_routed_experts = align_routed_experts(sample_routed_experts, expected_len)
            sample_routed_experts = np.ascontiguousarray(sample_routed_experts)
            sample.routed_experts = RoutedExperts(
                data=sample_routed_experts.tobytes(),
                shape=list(sample_routed_experts.shape),
                dtype=str(sample_routed_experts.dtype),
            )
        return sample_routed_experts

    # Track (prefix_tokens, sample, step_indices) per active sample. step_indices
    # is the explicit list of prepared_steps positions merged into this sample —
    # non-contiguous when other agents' steps interleave.
    active_samples: list[tuple[list[int], TrainingSample, list[int], np.ndarray | None]] = []

    first_tokens = prepared_steps[0]
    first_prefix = first_tokens["prompt_ids"] + first_tokens["completion_ids"]
    first_sample, first_routed_experts = make_sample(first_tokens)
    active_samples.append((first_prefix, first_sample, [0], first_routed_experts))

    for step_idx, _step in enumerate(trajectory[1:], start=1):
        tokens = prepared_steps[step_idx]
        step_prompt_ids = tokens["prompt_ids"]

        # Check if this step extends ANY active prefix
        matched_idx = None
        for idx, (prefix_tokens, _, _, _) in enumerate(active_samples):
            if step_prompt_ids[: len(prefix_tokens)] == prefix_tokens:
                matched_idx = idx
                break

        if matched_idx is not None:
            # Extension holds - merge into matched sample
            prefix_tokens, sample, step_indices, sample_routed_experts = active_samples[matched_idx]
            sample_routed_experts = extend_sample(sample, sample_routed_experts, len(prefix_tokens), step_idx=step_idx)
            active_samples[matched_idx] = (
                tokens["prompt_ids"] + tokens["completion_ids"],
                sample,
                step_indices + [step_idx],
                sample_routed_experts,
            )
        else:
            # No prefix matches - start a new sample
            logger.debug(
                f"Extension property broke at step {step_idx + 1} for example {output['example_id']}. "
                f"Starting new sample (active_prefixes={len(active_samples)}, step_prompt_len={len(step_prompt_ids)})."
            )
            new_prefix = tokens["prompt_ids"] + tokens["completion_ids"]
            sample, routed_experts = make_sample(tokens)
            active_samples.append((new_prefix, sample, [step_idx], routed_experts))

    # Attach images by concatenating mm_items across every step the
    # sample covers. verifiers' ``state_to_output`` ships per-step
    # *delta* mm_data (each step contains only items not present in the
    # prior step's cumulative set, with multiset-aware dedup), so
    # reading the last step alone would miss every earlier-turn image.
    # Concat in step order recovers the per-sample cumulative set;
    # deduping again here would drop legitimate duplicate placeholders.
    for _, sample, step_indices, _ in active_samples:
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

    return [sample for _, sample, _, _ in active_samples]


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
