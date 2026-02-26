import base64
import json
import time
from io import BytesIO
from typing import Any

import verifiers as vf
from PIL import Image
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.transport import TrainingSample
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
    max_len = min(len(a), len(b))
    for idx in range(max_len):
        if a[idx] != b[idx]:
            return idx
    return max_len


def _normalize_messages(messages: Any, default_role: str) -> list[dict[str, Any]]:
    if messages is None:
        return []
    if isinstance(messages, str):
        return [{"role": default_role, "content": messages}]
    if isinstance(messages, dict):
        return [dict(messages)]
    if isinstance(messages, list):
        normalized: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, str):
                normalized.append({"role": default_role, "content": message})
            elif isinstance(message, dict):
                normalized.append(dict(message))
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")
        return normalized
    raise TypeError(f"Unsupported messages container type: {type(messages)}")


def _deserialize_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _deserialize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function", {})
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return {
            **tool_call,
            "function": {**function, "arguments": arguments},
        }

    return [
        {
            **message,
            "tool_calls": [_deserialize_tool_call(tc) for tc in (message.get("tool_calls") or [])],
        }
        for message in messages
    ]


def _strip_message_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _strip(message: dict[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        if isinstance(content, str):
            return {**message, "content": content.strip()}
        return message

    return [_strip(message) for message in messages]


def _should_add_generation_prompt(messages: list[dict[str, Any]], idx: int) -> bool:
    role = messages[idx].get("role")
    if role not in ("user", "tool"):
        return False
    if idx + 1 >= len(messages):
        return False
    return messages[idx + 1].get("role") == "assistant"


def _render_messages(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = False,
) -> list[int]:
    return list(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
        )
    )


def _tokenize_step_from_messages(
    step: vf.TrajectoryStep,
    tokenizer: PreTrainedTokenizer,
) -> dict[str, Any]:
    prompt = _normalize_messages(step.get("prompt"), default_role="user")
    completion = _normalize_messages(step.get("completion"), default_role="assistant")

    prompt = _strip_message_content(_deserialize_tool_calls(prompt))
    completion = _strip_message_content(_deserialize_tool_calls(completion))

    all_messages = prompt + completion
    prompt_has_assistant_completion = len(completion) > 0 and completion[0].get("role") == "assistant"
    prompt_ids = _render_messages(
        tokenizer,
        prompt,
        add_generation_prompt=prompt_has_assistant_completion,
    )
    full_ids = _render_messages(tokenizer, all_messages)

    full_mask: list[bool] = []
    prev_ids: list[int] = []
    for idx, message in enumerate(all_messages):
        cur_ids = _render_messages(
            tokenizer,
            all_messages[: idx + 1],
            add_generation_prompt=_should_add_generation_prompt(all_messages, idx),
        )
        lcp = _common_prefix_len(prev_ids, cur_ids)
        full_mask = full_mask[:lcp] + [message.get("role") == "assistant"] * (len(cur_ids) - lcp)
        prev_ids = cur_ids

    split_idx = _common_prefix_len(prompt_ids, full_ids)

    completion_ids = full_ids[split_idx:]
    completion_mask = full_mask[split_idx:]
    completion_logprobs = [0.0] * len(completion_ids)

    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": [False] * len(prompt_ids),
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "completion_logprobs": completion_logprobs,
        "routed_experts": None,
        "prompt_prefix_len": split_idx,
    }


def interleave_rollout(
    output: vf.RolloutOutput,
    tokenizer: PreTrainedTokenizer | None = None,
    vlm_cache: "VLMImageCache | None" = None,
    cache_key: int | None = None,
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
        tokenizer: Tokenizer used to reconstruct step tokens when rollout tokens are unavailable
        vlm_cache: Pre-computed VLM image cache for multimodal training
        cache_key: Cache key to use when retrieving images from the VLM cache
    """
    logger = get_logger()

    trajectory = output["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {output['example_id']}. Skipping rollout.")
        return None

    has_error = output["error"] is not None
    # this field should be guaranteed because we set temperature in get_sampling_args
    temperature = output["sampling_args"]["temperature"]

    def prepare_step_tokens(step: vf.TrajectoryStep, step_idx: int) -> dict[str, Any] | None:
        tokens = step["tokens"]
        if tokens is not None:
            return {
                "prompt_ids": list(tokens["prompt_ids"]),
                "prompt_mask": [bool(i) for i in tokens["prompt_mask"]],
                "completion_ids": list(tokens["completion_ids"]),
                "completion_mask": [bool(i) for i in tokens["completion_mask"]],
                "completion_logprobs": list(tokens["completion_logprobs"]),
                "routed_experts": tokens.get("routed_experts"),
            }

        if tokenizer is None:
            logger.warning(
                f"Missing rollout tokens for example {output['example_id']} step {step_idx}, but no tokenizer was provided."
            )
            return None

        reconstructed = _tokenize_step_from_messages(step, tokenizer)
        if reconstructed["prompt_prefix_len"] < len(reconstructed["prompt_ids"]):
            logger.debug(
                f"Prompt tokenization was non-prefix for example {output['example_id']} step {step_idx}. "
                f"Using longest common prefix length {reconstructed['prompt_prefix_len']}."
            )
        reconstructed.pop("prompt_prefix_len")
        return reconstructed

    prepared_steps: list[dict[str, Any]] = []
    for step_idx, step in enumerate(trajectory):
        prepared = prepare_step_tokens(step, step_idx)
        if prepared is None:
            return None
        prepared_steps.append(prepared)

    def get_images(step_idx: int) -> tuple[list | None, list | None]:
        if vlm_cache is None:
            return None, None
        key = output["example_id"] if cache_key is None else cache_key
        return vlm_cache.get_for_step(key, step_idx)

    def make_sample(tokens: dict[str, Any], step_idx: int) -> TrainingSample:
        """Create a new TrainingSample from a trajectory step."""
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]
        completion_ids = list(tokens["completion_ids"])
        pixel_values, image_grid_thw = get_images(step_idx)

        routed_experts = _align_routed_experts(
            tokens.get("routed_experts"),
            len(tokens["prompt_ids"]) + len(tokens["completion_ids"]),
        )

        return TrainingSample(
            prompt_ids=list(tokens["prompt_ids"]),
            prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=list(tokens["completion_logprobs"]),
            completion_temperatures=[temperature] * len(completion_ids),
            teacher_logprobs=None,
            advantage=None,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            routed_experts=routed_experts,
        )

    def extend_sample(sample: TrainingSample, prefix_len: int, step_idx: int) -> None:
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

        # Update cumulative images to include any new images from this step
        pixel_values, image_grid_thw = get_images(step_idx)
        sample.pixel_values = pixel_values
        sample.image_grid_thw = image_grid_thw

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

    # Track multiple active (prefix, sample) pairs to handle interleaved agents
    # Each entry is [prefix_tokens, sample] where prefix_tokens is the accumulated token sequence
    active_samples: list[list] = []

    first_tokens = prepared_steps[0]
    first_prefix = first_tokens["prompt_ids"] + first_tokens["completion_ids"]
    active_samples.append([first_prefix, make_sample(first_tokens, step_idx=0)])

    for step_idx, _step in enumerate(trajectory[1:], start=1):
        tokens = prepared_steps[step_idx]
        step_prompt_ids = tokens["prompt_ids"]

        # Check if this step extends ANY active prefix
        matched_idx = None
        for idx, (prefix_tokens, _) in enumerate(active_samples):
            if step_prompt_ids[: len(prefix_tokens)] == prefix_tokens:
                matched_idx = idx
                break

        if matched_idx is not None:
            # Extension holds - merge into matched sample
            prefix_tokens, sample = active_samples[matched_idx]
            extend_sample(sample, len(prefix_tokens), step_idx=step_idx)
            # Update prefix for this sample
            active_samples[matched_idx][0] = tokens["prompt_ids"] + tokens["completion_ids"]
        else:
            # No prefix matches - start a new sample
            logger.debug(
                f"Extension property broke at step {step_idx + 1} for example {output['example_id']}. "
                f"Starting new sample (active_prefixes={len(active_samples)}, step_prompt_len={len(step_prompt_ids)})."
            )
            new_prefix = tokens["prompt_ids"] + tokens["completion_ids"]
            active_samples.append([new_prefix, make_sample(tokens, step_idx=step_idx)])

    return [sample for _, sample in active_samples]


# =============================================================================
# VLM-specific functions
# =============================================================================


def _extract_images_from_messages(messages: list) -> list[tuple[Image.Image, str]]:
    """Extract (image, b64_key) pairs from OpenAI-style chat messages."""
    images = []
    if not messages or not isinstance(messages, list):
        return images

    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        b64_data = url.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_data)
                        img = Image.open(BytesIO(img_bytes))
                        images.append((img, b64_data))
    return images


def _extract_images_from_examples(
    examples: list[tuple[int, vf.RolloutOutput]],
) -> tuple[list[Image.Image], dict[int, list[list[int]]]]:
    """
    Extract images from all trajectory steps of each example.

    Parses OpenAI-style message content looking for image_url items with base64 data URLs.
    Images are deduplicated across the batch by their base64 content. Each step records
    the indices of its images into the deduplicated all_images list.

    Args:
        examples: List of (cache_key, output) tuples where output contains a "trajectory"
            list with steps that have "prompt" messages in OpenAI chat format.

    Returns:
        Tuple of (all_images, step_image_indices_per_example)
        - all_images: deduplicated flat list of decoded PIL images
        - step_image_indices_per_example: dict mapping cache_key to per-step lists of
          indices into all_images (e.g., [[0], [0, 1], [1]] for the decreasing-images case)
    """
    all_images: list[Image.Image] = []
    image_registry: dict[str, int] = {}  # b64_key -> index in all_images
    step_image_indices_per_example: dict[int, list[list[int]]] = {}

    for eid, output in examples:
        trajectory = output.get("trajectory", [])
        if not trajectory:
            step_image_indices_per_example[eid] = []
            continue

        step_image_indices = []
        for step in trajectory:
            prompt = step.get("prompt")
            step_image_pairs = _extract_images_from_messages(prompt)
            indices = []
            for img, key in step_image_pairs:
                if key not in image_registry:
                    image_registry[key] = len(all_images)
                    all_images.append(img)
                indices.append(image_registry[key])
            step_image_indices.append(indices)

        step_image_indices_per_example[eid] = step_image_indices

    return all_images, step_image_indices_per_example


def _preprocess_images_batched(
    images: list[Image.Image],
    step_image_indices_per_example: dict[int, list[list[int]]],
    processor,
) -> dict[int, list[tuple[list | None, list | None]]]:
    """
    Preprocess all images in a single batched call, then distribute results per step.

    Args:
        images: Deduplicated flat list of all PIL images
        step_image_indices_per_example: Dict mapping cache_key to per-step lists of
            indices into images
        processor: HuggingFace processor with image_processor attribute

    Returns:
        Dict mapping cache_key to list of (pixel_values, image_grid_thw) per step.
    """
    if not images or processor is None:
        return {
            eid: [(None, None)] * max(len(step_indices), 1)
            for eid, step_indices in step_image_indices_per_example.items()
        }

    image_sizes = [(img.width, img.height) for img in images]
    processed = processor.image_processor(images=images, return_tensors="pt")
    all_pixel_values = processed["pixel_values"]
    all_grid_thw = processed["image_grid_thw"]

    logger = get_logger()
    logger.debug(
        f"VLM image processing: {len(images)} images, sizes={image_sizes}, "
        f"pixel_values={all_pixel_values.shape}, grid_thw={all_grid_thw.tolist()}"
    )

    # Pre-compute patch start offset for each image
    patch_starts = [0]
    for g in all_grid_thw:
        patch_starts.append(patch_starts[-1] + int(g[0] * g[1] * g[2]))

    result = {}
    for eid, step_indices_list in step_image_indices_per_example.items():
        if not step_indices_list:
            result[eid] = [(None, None)]
            continue

        per_step = []
        for indices in step_indices_list:
            if not indices:
                per_step.append((None, None))
            else:
                grids = all_grid_thw[indices]
                patches = sum([all_pixel_values[patch_starts[i] : patch_starts[i + 1]].tolist() for i in indices], [])
                per_step.append((patches, grids.tolist()))

        result[eid] = per_step

    return result


class VLMImageCache:
    """Result of building VLM image cache with per-step image data."""

    def __init__(
        self,
        cache: dict[int, list[tuple[list | None, list | None]]],
        num_unique_examples: int,
        extract_time: float,
        preprocess_time: float,
    ):
        self.cache = cache
        self.num_unique_examples = num_unique_examples
        self.extract_time = extract_time
        self.preprocess_time = preprocess_time

    def get_for_step(self, cache_key: int, step_idx: int) -> tuple[list | None, list | None]:
        """Get cumulative images up to and including the given step."""
        steps = self.cache.get(cache_key, [])
        if not steps or step_idx >= len(steps):
            return (None, None)
        return steps[step_idx]

    def get_all(self, cache_key: int) -> tuple[list | None, list | None]:
        """Get all images for the cache key (last step's cumulative images)."""
        steps = self.cache.get(cache_key, [])
        if not steps:
            return (None, None)
        return steps[-1]


def build_vlm_image_cache(rollouts: list[vf.RolloutOutput], processor) -> VLMImageCache:
    """
    Build image cache for VLM training by extracting and preprocessing images.

    Caches per rollout to keep images aligned with divergent multi-turn trajectories.
    """
    examples = [(idx, rollout) for idx, rollout in enumerate(rollouts)]
    unique_example_ids = {rollout["example_id"] for rollout in rollouts}

    # Extract images
    extract_start = time.perf_counter()
    all_images, images_per_example = _extract_images_from_examples(examples)
    extract_time = time.perf_counter() - extract_start

    # Preprocess images
    preprocess_start = time.perf_counter()
    cache = _preprocess_images_batched(all_images, images_per_example, processor)
    preprocess_time = time.perf_counter() - preprocess_start

    return VLMImageCache(
        cache=cache,
        num_unique_examples=len(unique_example_ids),
        extract_time=extract_time,
        preprocess_time=preprocess_time,
    )
