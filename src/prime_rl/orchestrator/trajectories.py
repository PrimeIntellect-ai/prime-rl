import base64
from copy import deepcopy
from io import BytesIO

import verifiers as vf
from PIL import Image

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


def extract_images_from_prompt(prompt) -> list[Image.Image]:
    """Extract PIL Images from multimodal prompt content."""
    images = []

    # Handle both string prompts and message list prompts
    if isinstance(prompt, str):
        return images  # Text-only prompt

    # Iterate through messages to find image content
    for message in prompt:
        content = message.get("content", "")

        # Handle list content (multimodal format)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")

                    # Extract base64 data from data URL
                    if image_url.startswith("data:image"):
                        # Format: "data:image/png;base64,<base64_data>"
                        base64_data = image_url.split(",", 1)[1]
                        image_bytes = base64.b64decode(base64_data)
                        img = Image.open(BytesIO(image_bytes))
                        images.append(img)

    return images


def interleave_rollout(state: vf.State, processor=None) -> list[TrainingSample] | None:
    """
    Convert vf.State to a *single* trainable rollout by interleaving the trajectory.

    NOTE:
    - This requires that consecutive trajectory steps share token prefixes (incremental tokenization)
    - This approach is suceptible to introduce subtle difference due to re-tokenization in multi-turn environments.
    """
    logger = get_logger()

    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    # NEW: Extract and preprocess images from the original prompt
    pixel_values = None
    if processor is not None and hasattr(processor, "image_processor"):
        # Extract images from the input prompt (stored in state["input"])
        input_prompt = state.get("input", {}).get("prompt", [])
        images = extract_images_from_prompt(input_prompt)

        if images:
            try:
                # Preprocess images using the processor's image_processor
                processed = processor.image_processor(images, return_tensors="pt")
                # Convert to nested list format for msgspec serialization
                # Shape: [num_images, channels, height, width]
                pixel_values = processed["pixel_values"].cpu().numpy().tolist()
                logger.debug(f"Preprocessed {len(images)} images for example {state['example_id']}")
            except Exception as e:
                logger.warning(f"Failed to preprocess images for example {state['example_id']}: {e}")

    has_error = state["error"] is not None

    # Initialize the rollout with prompt and completion from first trajectory step
    first_step = trajectory[0]
    if has_error:
        completion_mask = [False] * len(first_step["tokens"]["completion_mask"])
    else:
        completion_mask = [bool(i) for i in first_step["tokens"]["completion_mask"]]
    interleaved_rollout = TrainingSample(
        prompt_ids=deepcopy(first_step["tokens"]["prompt_ids"]),
        prompt_mask=[bool(i) for i in first_step["tokens"]["prompt_mask"]],
        completion_ids=deepcopy(first_step["tokens"]["completion_ids"]),
        completion_mask=completion_mask,
        completion_logprobs=deepcopy(first_step["tokens"]["completion_logprobs"]),
        advantage=None,
        pixel_values=pixel_values,  # NEW: Attach preprocessed images
    )

    # Interleave all other trajectory steps into completion
    prefix_tokens = first_step["tokens"]["prompt_ids"] + first_step["tokens"]["completion_ids"]
    for step_idx, step in enumerate(trajectory[1:], start=2):
        tokens = step["tokens"]
        assert tokens is not None
        prev_trajectory_and_new_prompt_ids = tokens["prompt_ids"]

        # Incremental tokenization assumption
        if not prefix_tokens == prev_trajectory_and_new_prompt_ids[: len(prefix_tokens)]:
            logger.warning(
                f"Found mismatch in prefix tokens for example {state['example_id']} at trajectory step {step_idx}"
            )

        # Extend the completion with the new prompt
        prompt_ids = deepcopy(prev_trajectory_and_new_prompt_ids[len(prefix_tokens) :])
        interleaved_rollout.completion_ids.extend(prompt_ids)
        interleaved_rollout.completion_mask.extend([False] * len(prompt_ids))
        interleaved_rollout.completion_logprobs.extend([0.0] * len(prompt_ids))

        # Extend the completion with the new completion tokens
        completion_ids = deepcopy(tokens["completion_ids"])
        completion_logprobs = deepcopy(tokens["completion_logprobs"])
        interleaved_rollout.completion_ids.extend(completion_ids)
        if has_error:
            interleaved_rollout.completion_mask.extend([False] * len(tokens["completion_mask"]))
        else:
            interleaved_rollout.completion_mask.extend([bool(i) for i in tokens["completion_mask"]])
        interleaved_rollout.completion_logprobs.extend(completion_logprobs)

        # New prefix is the current prompt and completion ids concatenated
        prefix_tokens = tokens["prompt_ids"] + tokens["completion_ids"]

    return [interleaved_rollout]


def branch_rollout(state: vf.State, processor=None) -> list[TrainingSample] | None:
    """Convert vf.State to *multiple* trainable rollouts using branching trajectories strategy."""
    logger = get_logger()

    rollouts = []
    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    # NEW: Extract and preprocess images from the original prompt
    pixel_values = None
    if processor is not None and hasattr(processor, "image_processor"):
        # Extract images from the input prompt (stored in state["input"])
        input_prompt = state.get("input", {}).get("prompt", [])
        images = extract_images_from_prompt(input_prompt)

        if images:
            try:
                # Preprocess images using the processor's image_processor
                processed = processor.image_processor(images, return_tensors="pt")
                # Convert to nested list format for msgspec serialization
                # Shape: [num_images, channels, height, width]
                pixel_values = processed["pixel_values"].cpu().numpy().tolist()
                logger.debug(f"Preprocessed {len(images)} images for example {state['example_id']}")
            except Exception as e:
                logger.warning(f"Failed to preprocess images for example {state['example_id']}: {e}")

    has_error = state["error"] is not None
    for step in state["trajectory"]:
        assert "tokens" in step
        tokens = step["tokens"]
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]
        rollout = TrainingSample(
            prompt_ids=deepcopy(tokens["prompt_ids"]),
            prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
            completion_ids=deepcopy(tokens["completion_ids"]),
            completion_mask=completion_mask,
            completion_logprobs=deepcopy(tokens["completion_logprobs"]),
            advantage=None,
            pixel_values=pixel_values,  # NEW: Attach preprocessed images
        )
        rollouts.append(rollout)
    return rollouts
