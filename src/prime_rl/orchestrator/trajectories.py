import base64
import time
from io import BytesIO

import verifiers as vf
from PIL import Image

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger

# We use list() instead of deepcopy() for flat lists (token IDs, logprobs) - safe because
# primitives are immutable. pixel_values/image_grid_thw are shared across rollouts of the
# same example (not copied) which is safe since nothing mutates them after creation.


def interleave_rollout(
    state: vf.State,
    cached_pixel_values: list | None = None,
    cached_image_grid_thw: list | None = None,
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

    Args:
        state: vf.State containing trajectory data
        cached_pixel_values: Pre-computed pixel values for VLM training
        cached_image_grid_thw: Pre-computed image grid thw for VLM training
    """
    logger = get_logger()

    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    has_error = state["error"] is not None

    def make_sample(step: dict) -> TrainingSample:
        """Create a new TrainingSample from a trajectory step."""
        tokens = step["tokens"]
        temperature = step["temperature"]
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]
        completion_ids = list(tokens["completion_ids"])
        return TrainingSample(
            prompt_ids=list(tokens["prompt_ids"]),
            prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=list(tokens["completion_logprobs"]),
            completion_temperatures=[temperature] * len(completion_ids),
            teacher_logprobs=None,
            advantage=None,
            pixel_values=cached_pixel_values,
            image_grid_thw=cached_image_grid_thw,
        )

    def extend_sample(sample: TrainingSample, step: dict, prefix_len: int) -> None:
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

    # Track multiple active (prefix, sample) pairs to handle interleaved agents
    # Each entry is [prefix_tokens, sample] where prefix_tokens is the accumulated token sequence
    active_samples: list[list] = []

    first_tokens = trajectory[0]["tokens"]
    first_prefix = first_tokens["prompt_ids"] + first_tokens["completion_ids"]
    active_samples.append([first_prefix, make_sample(trajectory[0])])

    for step_idx, step in enumerate(trajectory[1:], start=2):
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
            extend_sample(sample, step, len(prefix_tokens))
            # Update prefix for this sample
            active_samples[matched_idx][0] = tokens["prompt_ids"] + tokens["completion_ids"]
        else:
            # No prefix matches - start a new sample
            logger.debug(
                f"Extension property broke at step {step_idx} for example {state['example_id']}. "
                f"Starting new sample (active_prefixes={len(active_samples)}, step_prompt_len={len(step_prompt_ids)})."
            )
            new_prefix = tokens["prompt_ids"] + tokens["completion_ids"]
            active_samples.append([new_prefix, make_sample(step)])

    # Return all samples
    return [sample for _, sample in active_samples]


def branch_rollout(
    state: vf.State,
    cached_pixel_values: list | None = None,
    cached_image_grid_thw: list | None = None,
) -> list[TrainingSample] | None:
    """
    Convert vf.State to *multiple* trainable rollouts using branching trajectories strategy.

    Args:
        state: vf.State containing trajectory data
        cached_pixel_values: Pre-computed pixel values for VLM training
        cached_image_grid_thw: Pre-computed image grid thw for VLM training
    """
    logger = get_logger()

    rollouts = []
    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    has_error = state["error"] is not None
    for step in trajectory:
        tokens = step["tokens"]
        temperature = step["temperature"]
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]

        completion_ids = list(tokens["completion_ids"])
        rollout = TrainingSample(
            prompt_ids=list(tokens["prompt_ids"]),
            prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=list(tokens["completion_logprobs"]),
            completion_temperatures=[temperature] * len(completion_ids),
            advantage=None,
            teacher_logprobs=None,
            pixel_values=cached_pixel_values,
            image_grid_thw=cached_image_grid_thw,
        )
        rollouts.append(rollout)
    return rollouts


# =============================================================================
# VLM-specific functions
# =============================================================================


def _extract_images_from_examples(
    examples: list[tuple[int, vf.State]],
) -> tuple[list[Image.Image], dict[int, int]]:
    """
    Extract all images from the first trajectory step of each example.

    Parses OpenAI-style message content looking for image_url items with base64 data URLs
    (e.g., "data:image/png;base64,..."). Only the first trajectory step's prompt is checked,
    as images are assumed to be provided in the initial prompt.

    Args:
        examples: List of (example_id, state) tuples where state contains a "trajectory"
            list with steps that have "prompt" messages in OpenAI chat format.

    Returns:
        Tuple of (all_images, images_per_example)
        - all_images: flat list of decoded PIL images, ordered by example then by appearance
        - images_per_example: dict mapping example_id to number of images for that example
    """
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

    return all_images, images_per_example


def _preprocess_images_batched(
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
    if not images or processor is None:
        return {eid: (None, None) for eid in images_per_example}

    processed = processor.image_processor(images=images, return_tensors="pt")
    all_pixel_values = processed["pixel_values"]
    all_grid_thw = processed["image_grid_thw"]

    result = {}
    img_idx = 0
    patch_idx = 0

    for eid, num_images in images_per_example.items():
        if num_images == 0:
            result[eid] = (None, None)
        else:
            example_grids = all_grid_thw[img_idx : img_idx + num_images]
            num_patches = sum(int(g[0] * g[1] * g[2]) for g in example_grids)
            example_pixels = all_pixel_values[patch_idx : patch_idx + num_patches]

            result[eid] = (example_pixels.tolist(), example_grids.tolist())

            img_idx += num_images
            patch_idx += num_patches

    return result


class VLMImageCache:
    """Result of building VLM image cache."""

    def __init__(
        self,
        cache: dict[int, tuple[list | None, list | None]],
        num_unique_examples: int,
        extract_time: float,
        preprocess_time: float,
    ):
        self.cache = cache
        self.num_unique_examples = num_unique_examples
        self.extract_time = extract_time
        self.preprocess_time = preprocess_time

    def get(self, example_id: int) -> tuple[list | None, list | None]:
        return self.cache.get(example_id, (None, None))


def build_vlm_image_cache(rollouts: list[vf.State], processor) -> VLMImageCache:
    """
    Build image cache for VLM training by extracting and preprocessing images.

    Groups rollouts by example_id to avoid redundant preprocessing (with rollouts_per_example=8,
    we only preprocess 1/8th of the images).
    """
    # Group rollouts by example_id
    example_id_to_rollout: dict[int, vf.State] = {}
    for rollout in rollouts:
        example_id = rollout["example_id"]
        if example_id not in example_id_to_rollout:
            example_id_to_rollout[example_id] = rollout

    unique_examples = [(eid, rollout) for eid, rollout in example_id_to_rollout.items()]

    # Extract images
    extract_start = time.perf_counter()
    all_images, images_per_example = _extract_images_from_examples(unique_examples)
    extract_time = time.perf_counter() - extract_start

    # Preprocess images
    preprocess_start = time.perf_counter()
    if all_images:
        cache = _preprocess_images_batched(all_images, images_per_example, processor)
    else:
        cache = {eid: (None, None) for eid in images_per_example}
    preprocess_time = time.perf_counter() - preprocess_start

    return VLMImageCache(
        cache=cache,
        num_unique_examples=len(unique_examples),
        extract_time=extract_time,
        preprocess_time=preprocess_time,
    )
