import base64
import time
from io import BytesIO
from typing import TypedDict

import verifiers as vf
from PIL import Image

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


class TrajectoryStepWithTemp(TypedDict):
    """Trajectory step with temperature field added by prime-rl's extract_result."""

    tokens: vf.TrajectoryStepTokens
    temperature: float


# We use list() instead of deepcopy() for flat lists (token IDs, logprobs) - safe because
# primitives are immutable. pixel_values/image_grid_thw are not mutated after creation.


def interleave_rollout(
    state: vf.State,
    vlm_cache: "VLMImageCache | None" = None,
    cache_key: int | None = None,
) -> list[TrainingSample] | None:
    """
    Convert vf.State to trainable rollouts by interleaving trajectory steps
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
        state: vf.State containing trajectory data
        vlm_cache: Pre-computed VLM image cache for multimodal training
        cache_key: Cache key to use when retrieving images from the VLM cache
    """
    logger = get_logger()

    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    has_error = state["error"] is not None

    def get_images(step_idx: int) -> tuple[list | None, list | None]:
        if vlm_cache is None:
            return None, None
        key = state["example_id"] if cache_key is None else cache_key
        return vlm_cache.get_for_step(key, step_idx)

    def make_sample(step: TrajectoryStepWithTemp, step_idx: int) -> TrainingSample:
        """Create a new TrainingSample from a trajectory step."""
        tokens = step["tokens"]
        temperature = step["temperature"]
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]
        completion_ids = list(tokens["completion_ids"])
        pixel_values, image_grid_thw = get_images(step_idx)
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
        )

    def extend_sample(sample: TrainingSample, step: TrajectoryStepWithTemp, prefix_len: int, step_idx: int) -> None:
        """Extend an existing sample with a new trajectory step (extension property holds)."""
        tokens = step["tokens"]
        temperature = step["temperature"]

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

    # Track multiple active (prefix, sample) pairs to handle interleaved agents
    # Each entry is [prefix_tokens, sample] where prefix_tokens is the accumulated token sequence
    active_samples: list[list] = []

    first_tokens = trajectory[0]["tokens"]
    first_prefix = first_tokens["prompt_ids"] + first_tokens["completion_ids"]
    active_samples.append([first_prefix, make_sample(trajectory[0], step_idx=0)])

    for step_idx, step in enumerate(trajectory[1:], start=1):
        tokens = step["tokens"]
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
            extend_sample(sample, step, len(prefix_tokens), step_idx=step_idx)
            # Update prefix for this sample
            active_samples[matched_idx][0] = tokens["prompt_ids"] + tokens["completion_ids"]
        else:
            # No prefix matches - start a new sample
            logger.debug(
                f"Extension property broke at step {step_idx + 1} for example {state['example_id']}. "
                f"Starting new sample (active_prefixes={len(active_samples)}, step_prompt_len={len(step_prompt_ids)})."
            )
            new_prefix = tokens["prompt_ids"] + tokens["completion_ids"]
            active_samples.append([new_prefix, make_sample(step, step_idx=step_idx)])

    return [sample for _, sample in active_samples]


# =============================================================================
# VLM-specific functions
# =============================================================================


def _extract_images_from_messages(messages: list) -> list[Image.Image]:
    """Extract images from OpenAI-style chat messages."""
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
                        images.append(img)
    return images


def _extract_images_from_examples(
    examples: list[tuple[int, vf.State]],
) -> tuple[list[Image.Image], dict[int, list[int]]]:
    """
    Extract images from all trajectory steps of each example.

    Parses OpenAI-style message content looking for image_url items with base64 data URLs
    (e.g., "data:image/png;base64,..."). Each trajectory step's prompt is cumulative (contains
    full conversation history), so we extract only the NEW images introduced in each step.

    Args:
        examples: List of (cache_key, state) tuples where state contains a "trajectory"
            list with steps that have "prompt" messages in OpenAI chat format.

    Returns:
        Tuple of (all_images, images_per_step_per_example)
        - all_images: flat list of decoded PIL images, ordered by example then by step
        - images_per_step_per_example: dict mapping cache_key to list of cumulative image
          counts per step (e.g., [1, 2, 2] means 1 image after step 0, 2 after step 1, 2 after step 2)
    """
    all_images = []
    images_per_step_per_example = {}

    for eid, state in examples:
        trajectory = state.get("trajectory", [])
        if not trajectory:
            images_per_step_per_example[eid] = []
            continue

        example_images = []
        cumulative_counts = []

        for step in trajectory:
            prompt = step.get("prompt")
            # Extract all images from this step's prompt (which is cumulative)
            step_images = _extract_images_from_messages(prompt)
            # Only take images beyond what we already have (new images in this step)
            new_images = step_images[len(example_images) :]
            example_images.extend(new_images)
            cumulative_counts.append(len(example_images))

        images_per_step_per_example[eid] = cumulative_counts
        all_images.extend(example_images)

    return all_images, images_per_step_per_example


def _preprocess_images_batched(
    images: list[Image.Image],
    images_per_step_per_example: dict[int, list[int]],
    processor,
) -> dict[int, list[tuple[list | None, list | None]]]:
    """
    Preprocess all images in a single batched call, then distribute results per step.

    Args:
        images: Flat list of all PIL images
        images_per_step_per_example: Dict mapping cache_key to list of cumulative image
            counts per step
        processor: HuggingFace processor with image_processor attribute

    Returns:
        Dict mapping cache_key to list of (pixel_values, image_grid_thw) per step.
        Each step's entry contains cumulative images up to that step.
    """
    if not images or processor is None:
        return {
            eid: [(None, None)] * len(counts) if counts else [(None, None)]
            for eid, counts in images_per_step_per_example.items()
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

    result = {}
    img_idx = 0
    patch_idx = 0

    for eid, cumulative_counts in images_per_step_per_example.items():
        if not cumulative_counts or cumulative_counts[-1] == 0:
            result[eid] = [(None, None)] * max(len(cumulative_counts), 1)
            continue

        total_images = cumulative_counts[-1]
        example_grids = all_grid_thw[img_idx : img_idx + total_images]
        num_patches = sum(int(g[0] * g[1] * g[2]) for g in example_grids)
        example_pixels = all_pixel_values[patch_idx : patch_idx + num_patches]

        # Build per-step cumulative entries
        per_step = []
        for cum_count in cumulative_counts:
            if cum_count == 0:
                per_step.append((None, None))
            else:
                step_grids = example_grids[:cum_count]
                step_patches = sum(int(g[0] * g[1] * g[2]) for g in step_grids)
                per_step.append((example_pixels[:step_patches].tolist(), step_grids.tolist()))

        result[eid] = per_step
        img_idx += total_images
        patch_idx += num_patches

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


def build_vlm_image_cache(rollouts: list[vf.State], processor) -> VLMImageCache:
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
