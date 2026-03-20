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

from prime_rl.transport import TrainingSample
from prime_rl.utils.chat_template import (
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
    active_samples: list[list] = []

    first_tokens = prepared_steps[0]
    first_prefix = first_tokens["prompt_ids"] + first_tokens["completion_ids"]
    active_samples.append([first_prefix, make_sample(first_tokens), 0])

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
            active_samples[matched_idx][0] = tokens["prompt_ids"] + tokens["completion_ids"]
            active_samples[matched_idx][2] = step_idx
        else:
            # No prefix matches - start a new sample
            logger.debug(
                f"Extension property broke at step {step_idx + 1} for example {output['example_id']}. "
                f"Starting new sample (active_prefixes={len(active_samples)}, step_prompt_len={len(step_prompt_ids)})."
            )
            new_prefix = tokens["prompt_ids"] + tokens["completion_ids"]
            active_samples.append([new_prefix, make_sample(tokens), step_idx])

    # Attach images once per sample using only the last merged step
    if vlm_cache is not None:
        key = output["example_id"] if cache_key is None else cache_key
        for _, sample, last_step_idx in active_samples:
            pv, shape, grids = vlm_cache.get_for_step(key, last_step_idx)
            sample.pixel_values = pv
            sample.pixel_values_shape = shape
            sample.image_grid_thw = grids

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
