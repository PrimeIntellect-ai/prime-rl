import base64
import threading
import time
from io import BytesIO

import verifiers as vf
from PIL import Image

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger

# =============================================================================
# CONSIDERATIONS: Why we use list() instead of deepcopy() for TrainingSample fields
# =============================================================================
#
# Previously, this code used deepcopy() for all list fields (token IDs, logprobs,
# pixel_values, etc.) which caused ~97s overhead per batch (512 rollouts).
#
# We removed deepcopy() based on the following analysis:
#
# 1. FLAT LISTS (prompt_ids, completion_ids, completion_logprobs):
#    - These are lists of primitives (int/float)
#    - list() creates a shallow copy, which equals deep copy for flat lists
#    - Safe because primitives are immutable
#
# 2. NESTED LISTS (pixel_values, image_grid_thw):
#    - These are shared among 8 rollouts of the same example (GRPO-style)
#    - We don't copy them at all - multiple TrainingSamples reference the same list
#    - This is safe because:
#      a) Nothing in the orchestrator mutates pixel_values after creation
#      b) msgspec.msgpack serialization creates independent copies for the trainer
#      c) torch.tensor() in the trainer creates a new tensor without mutating the list
#
# VERIFICATION NEEDED if you change the data flow:
#    - Ensure nothing mutates pixel_values/image_grid_thw between creation and serialization
#    - Ensure serialization (msgspec.msgpack) still creates independent copies
#    - Test: mutation via shallow copy affects original (list()[0].append() propagates)
#
# If bugs appear related to corrupted pixel_values across rollouts, consider:
#    - Adding back deepcopy() for pixel_values/image_grid_thw only
#    - Or use: [list(inner) for inner in pixel_values] for one-level-deep copy
# =============================================================================

# Timing accumulators for profiling image preprocessing (thread-safe)
_IMAGE_TIMING_LOCK = threading.Lock()
_IMAGE_EXTRACT_TIME = 0.0
_IMAGE_PREPROCESS_TIME = 0.0
_IMAGE_COUNT = 0


def extract_images_from_prompt(prompt: list[dict]) -> list[Image.Image]:
    """
    Extract PIL images from OpenAI-format messages containing base64-encoded image_urls.

    Args:
        prompt: List of message dicts in OpenAI chat format

    Returns:
        List of PIL.Image.Image objects extracted from the prompt
    """
    global _IMAGE_EXTRACT_TIME, _IMAGE_COUNT
    start = time.perf_counter()
    images = []
    img_count = 0
    for msg in prompt:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        # Extract base64 data after the comma
                        b64_data = url.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_data)
                        img_count += 1
                        img = Image.open(BytesIO(img_bytes))
                        images.append(img)
    elapsed = time.perf_counter() - start
    with _IMAGE_TIMING_LOCK:
        _IMAGE_EXTRACT_TIME += elapsed
        _IMAGE_COUNT += img_count
    return images


def preprocess_images(images: list[Image.Image], processor) -> tuple[list | None, list | None]:
    """
    Preprocess images using HuggingFace processor's image_processor.

    Args:
        images: List of PIL images
        processor: HuggingFace processor with image_processor attribute

    Returns:
        Tuple of (pixel_values, image_grid_thw) as lists, or (None, None) if no images
    """
    global _IMAGE_PREPROCESS_TIME
    if not images or processor is None:
        return None, None
    start = time.perf_counter()

    processed = processor.image_processor(images=images, return_tensors="pt")
    pixel_values = processed["pixel_values"].tolist()
    image_grid_thw = processed["image_grid_thw"].tolist()
    elapsed = time.perf_counter() - start
    with _IMAGE_TIMING_LOCK:
        _IMAGE_PREPROCESS_TIME += elapsed
    return pixel_values, image_grid_thw


def get_image_timing_stats() -> dict:
    """Get accumulated image processing timing stats."""
    with _IMAGE_TIMING_LOCK:
        return {
            "extract_time": _IMAGE_EXTRACT_TIME,
            "preprocess_time": _IMAGE_PREPROCESS_TIME,
            "image_count": _IMAGE_COUNT,
            "avg_extract_ms": (_IMAGE_EXTRACT_TIME / _IMAGE_COUNT * 1000) if _IMAGE_COUNT > 0 else 0,
            "avg_preprocess_ms": (_IMAGE_PREPROCESS_TIME / _IMAGE_COUNT * 1000) if _IMAGE_COUNT > 0 else 0,
        }


def reset_image_timing_stats():
    """Reset timing accumulators."""
    global _IMAGE_EXTRACT_TIME, _IMAGE_PREPROCESS_TIME, _IMAGE_COUNT
    with _IMAGE_TIMING_LOCK:
        _IMAGE_EXTRACT_TIME = 0.0
        _IMAGE_PREPROCESS_TIME = 0.0
        _IMAGE_COUNT = 0


def preprocess_example_images(state: vf.State, processor) -> tuple[list | None, list | None]:
    """
    Preprocess images for a single example/rollout state.

    Args:
        state: vf.State containing trajectory data with prompt
        processor: HuggingFace processor with image_processor attribute

    Returns:
        Tuple of (pixel_values, image_grid_thw) as lists, or (None, None) if no images
    """
    if processor is None:
        return None, None

    trajectory = state.get("trajectory", [])
    if not trajectory:
        return None, None

    first_step = trajectory[0]
    prompt = first_step.get("prompt")
    if not prompt or not isinstance(prompt, list):
        return None, None

    images = extract_images_from_prompt(prompt)
    if not images:
        return None, None

    return preprocess_images(images, processor)


def extract_images_from_examples(
    examples: list[tuple[int, vf.State]],
) -> tuple[list[Image.Image], dict[int, int]]:
    """
    Extract all images from multiple examples.

    Args:
        examples: List of (example_id, state) tuples

    Returns:
        Tuple of (all_images, images_per_example)
        - all_images: flat list of all PIL images in order
        - images_per_example: dict mapping example_id to number of images
    """
    global _IMAGE_EXTRACT_TIME, _IMAGE_COUNT
    start = time.perf_counter()

    all_images = []
    images_per_example = {}

    for eid, state in examples:
        trajectory = state.get("trajectory", [])
        if not trajectory:
            images_per_example[eid] = 0
            continue

        first_step = trajectory[0]
        prompt = first_step.get("prompt")
        if not prompt or not isinstance(prompt, list):
            images_per_example[eid] = 0
            continue

        # Extract without timing (we'll time the whole batch)
        images = []
        for msg in prompt:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            b64_data = url.split(",", 1)[1]
                            img_bytes = base64.b64decode(b64_data)
                            img = Image.open(BytesIO(img_bytes))
                            images.append(img)

        images_per_example[eid] = len(images)
        all_images.extend(images)

    elapsed = time.perf_counter() - start
    with _IMAGE_TIMING_LOCK:
        _IMAGE_EXTRACT_TIME += elapsed
        _IMAGE_COUNT += len(all_images)

    return all_images, images_per_example


def preprocess_images_batched(
    images: list[Image.Image],
    images_per_example: dict[int, int],
    processor,
) -> dict[int, tuple[list | None, list | None]]:
    """
    Preprocess all images in a single batched call, then distribute results.

    Args:
        images: Flat list of all PIL images
        images_per_example: Dict mapping example_id to number of images for that example
        processor: HuggingFace processor with image_processor attribute

    Returns:
        Dict mapping example_id to (pixel_values, image_grid_thw)
    """
    global _IMAGE_PREPROCESS_TIME

    # Handle empty case
    if not images or processor is None:
        return {eid: (None, None) for eid in images_per_example}

    start = time.perf_counter()

    # Single batched call to processor
    processed = processor.image_processor(images=images, return_tensors="pt")
    all_pixel_values = processed["pixel_values"]  # (total_patches, patch_dim)
    all_grid_thw = processed["image_grid_thw"]  # (num_images, 3)

    elapsed = time.perf_counter() - start
    with _IMAGE_TIMING_LOCK:
        _IMAGE_PREPROCESS_TIME += elapsed

    # Distribute results back to examples
    result = {}
    img_idx = 0
    patch_idx = 0

    for eid, num_images in images_per_example.items():
        if num_images == 0:
            result[eid] = (None, None)
        else:
            # Get grid info for this example's images
            example_grids = all_grid_thw[img_idx : img_idx + num_images]

            # Calculate total patches for this example
            num_patches = sum(int(g[0] * g[1] * g[2]) for g in example_grids)

            # Extract pixel values for this example
            example_pixels = all_pixel_values[patch_idx : patch_idx + num_patches]

            result[eid] = (example_pixels.tolist(), example_grids.tolist())

            img_idx += num_images
            patch_idx += num_patches

    return result


def interleave_rollout(
    state: vf.State,
    processor=None,
    cached_pixel_values: list | None = None,
    cached_image_grid_thw: list | None = None,
) -> list[TrainingSample] | None:
    """
    Convert vf.State to a *single* trainable rollout by interleaving the trajectory.

    NOTE:
    - This requires that consecutive trajectory steps share token prefixes (incremental tokenization)
    - This approach is suceptible to introduce subtle difference due to re-tokenization in multi-turn environments.

    Args:
        state: vf.State containing trajectory data
        processor: Optional HuggingFace processor for VLM models (e.g., Qwen3VLProcessor).
            If provided and images are found in the prompt, they will be preprocessed
            to extract pixel_values and image_grid_thw for training.
        cached_pixel_values: Pre-computed pixel values (to avoid redundant preprocessing)
        cached_image_grid_thw: Pre-computed image grid thw (to avoid redundant preprocessing)
    """
    logger = get_logger()

    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    has_error = state["error"] is not None

    # Initialize the rollout with prompt and completion from first trajectory step
    first_step = trajectory[0]
    temperature = first_step["temperature"]
    if has_error:
        completion_mask = [False] * len(first_step["tokens"]["completion_mask"])
    else:
        completion_mask = [bool(i) for i in first_step["tokens"]["completion_mask"]]

    # Use cached image data if provided, otherwise extract and preprocess
    pixel_values = cached_pixel_values
    image_grid_thw = cached_image_grid_thw
    if pixel_values is None and processor is not None:
        prompt = first_step.get("prompt")
        if prompt and isinstance(prompt, list):
            images = extract_images_from_prompt(prompt)
            if images:
                pixel_values, image_grid_thw = preprocess_images(images, processor)

    # Use list() instead of deepcopy() for flat lists - much faster
    # pixel_values/image_grid_thw are from cache and not modified, so no copy needed
    completion_ids = list(first_step["tokens"]["completion_ids"])
    interleaved_rollout = TrainingSample(
        prompt_ids=list(first_step["tokens"]["prompt_ids"]),
        prompt_mask=[bool(i) for i in first_step["tokens"]["prompt_mask"]],
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_logprobs=list(first_step["tokens"]["completion_logprobs"]),
        completion_temperatures=[temperature] * len(completion_ids),  # Per-token temperatures
        teacher_logprobs=None,  # Populated at the end after full sequence length is known if teacher model is configured
        advantage=None,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )

    # Interleave all other trajectory steps into completion
    prefix_tokens = first_step["tokens"]["prompt_ids"] + first_step["tokens"]["completion_ids"]
    for step_idx, step in enumerate(trajectory[1:], start=2):
        tokens = step["tokens"]
        step_temperature = step["temperature"]
        assert tokens is not None
        prev_trajectory_and_new_prompt_ids = tokens["prompt_ids"]

        # Incremental tokenization assumption
        if not prefix_tokens == prev_trajectory_and_new_prompt_ids[: len(prefix_tokens)]:
            logger.warning(
                f"Found mismatch in prefix tokens for example {state['example_id']} at trajectory step {step_idx}"
            )

        # Extend the completion with the new prompt (use step's temperature for prompt tokens too)
        prompt_ids = list(prev_trajectory_and_new_prompt_ids[len(prefix_tokens) :])
        interleaved_rollout.completion_ids.extend(prompt_ids)
        interleaved_rollout.completion_mask.extend([False] * len(prompt_ids))
        interleaved_rollout.completion_logprobs.extend([0.0] * len(prompt_ids))
        interleaved_rollout.completion_temperatures.extend([step_temperature] * len(prompt_ids))

        # Extend the completion with the new completion tokens
        completion_ids = tokens["completion_ids"]
        completion_logprobs = tokens["completion_logprobs"]
        interleaved_rollout.completion_ids.extend(completion_ids)
        if has_error:
            interleaved_rollout.completion_mask.extend([False] * len(tokens["completion_mask"]))
        else:
            interleaved_rollout.completion_mask.extend([bool(i) for i in tokens["completion_mask"]])
        interleaved_rollout.completion_logprobs.extend(completion_logprobs)
        interleaved_rollout.completion_temperatures.extend([step_temperature] * len(completion_ids))

        # New prefix is the current prompt and completion ids concatenated
        prefix_tokens = tokens["prompt_ids"] + tokens["completion_ids"]

    return [interleaved_rollout]


def branch_rollout(
    state: vf.State,
    processor=None,
    cached_pixel_values: list | None = None,
    cached_image_grid_thw: list | None = None,
) -> list[TrainingSample] | None:
    """
    Convert vf.State to *multiple* trainable rollouts using branching trajectories strategy.

    Args:
        state: vf.State containing trajectory data
        processor: Optional HuggingFace processor for VLM models (e.g., Qwen3VLProcessor).
            If provided and images are found in the prompt, they will be preprocessed
            to extract pixel_values and image_grid_thw for training.
        cached_pixel_values: Pre-computed pixel values (to avoid redundant preprocessing)
        cached_image_grid_thw: Pre-computed image grid thw (to avoid redundant preprocessing)
    """
    logger = get_logger()

    rollouts = []
    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    has_error = state["error"] is not None
    for step in state["trajectory"]:
        assert "tokens" in step
        tokens = step["tokens"]
        temperature = step["temperature"]
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]

        # Use cached image data if provided, otherwise extract and preprocess
        pixel_values = cached_pixel_values
        image_grid_thw = cached_image_grid_thw
        if pixel_values is None and processor is not None:
            prompt = step.get("prompt")
            if prompt and isinstance(prompt, list):
                images = extract_images_from_prompt(prompt)
                if images:
                    pixel_values, image_grid_thw = preprocess_images(images, processor)

        # Use list() instead of deepcopy() for flat lists - much faster
        # pixel_values/image_grid_thw are from cache and not modified, so no copy needed
        completion_ids = list(tokens["completion_ids"])
        rollout = TrainingSample(
            prompt_ids=list(tokens["prompt_ids"]),
            prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=list(tokens["completion_logprobs"]),
            completion_temperatures=[temperature] * len(completion_ids),  # Per-token temperatures
            advantage=None,
            teacher_logprobs=None,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        rollouts.append(rollout)
    return rollouts
