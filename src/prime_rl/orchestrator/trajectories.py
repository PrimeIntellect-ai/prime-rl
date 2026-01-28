import base64
from copy import deepcopy
from io import BytesIO

import verifiers as vf
from PIL import Image

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


def extract_images_from_prompt(prompt: list[dict]) -> list[Image.Image]:
    """
    Extract PIL images from OpenAI-format messages containing base64-encoded image_urls.

    Args:
        prompt: List of message dicts in OpenAI chat format

    Returns:
        List of PIL.Image.Image objects extracted from the prompt
    """
    images = []
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
                        img = Image.open(BytesIO(img_bytes))
                        images.append(img)
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
    if not images or processor is None:
        return None, None

    processed = processor.image_processor(images=images, return_tensors="pt")
    pixel_values = processed["pixel_values"].tolist()
    image_grid_thw = processed["image_grid_thw"].tolist()
    return pixel_values, image_grid_thw


def interleave_rollout(state: vf.State, processor=None) -> list[TrainingSample] | None:
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
    """
    logger = get_logger()

    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    has_error = state["error"] is not None

    # Initialize the rollout with prompt and completion from first trajectory step
    first_step = trajectory[0]
    if has_error:
        completion_mask = [False] * len(first_step["tokens"]["completion_mask"])
    else:
        completion_mask = [bool(i) for i in first_step["tokens"]["completion_mask"]]

    # Extract multimodal fields from prompt using processor (for VLM models)
    pixel_values = None
    image_grid_thw = None
    if processor is not None:
        prompt = first_step.get("prompt")
        if prompt and isinstance(prompt, list):
            images = extract_images_from_prompt(prompt)
            if images:
                pixel_values, image_grid_thw = preprocess_images(images, processor)

    interleaved_rollout = TrainingSample(
        prompt_ids=deepcopy(first_step["tokens"]["prompt_ids"]),
        prompt_mask=[bool(i) for i in first_step["tokens"]["prompt_mask"]],
        completion_ids=deepcopy(first_step["tokens"]["completion_ids"]),
        completion_mask=completion_mask,
        completion_logprobs=deepcopy(first_step["tokens"]["completion_logprobs"]),
        teacher_logprobs=None,  # Populated at the end after full sequence length is known if teacher model is configured
        advantage=None,
        pixel_values=deepcopy(pixel_values) if pixel_values is not None else None,
        image_grid_thw=deepcopy(image_grid_thw) if image_grid_thw is not None else None,
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
    """
    Convert vf.State to *multiple* trainable rollouts using branching trajectories strategy.

    Args:
        state: vf.State containing trajectory data
        processor: Optional HuggingFace processor for VLM models (e.g., Qwen3VLProcessor).
            If provided and images are found in the prompt, they will be preprocessed
            to extract pixel_values and image_grid_thw for training.
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
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]

        # Extract multimodal fields from prompt using processor (for VLM models)
        pixel_values = None
        image_grid_thw = None
        if processor is not None:
            prompt = step.get("prompt")
            if prompt and isinstance(prompt, list):
                images = extract_images_from_prompt(prompt)
                if images:
                    pixel_values, image_grid_thw = preprocess_images(images, processor)

        rollout = TrainingSample(
            prompt_ids=deepcopy(tokens["prompt_ids"]),
            prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
            completion_ids=deepcopy(tokens["completion_ids"]),
            completion_mask=completion_mask,
            completion_logprobs=deepcopy(tokens["completion_logprobs"]),
            advantage=None,
            teacher_logprobs=None,
            pixel_values=deepcopy(pixel_values) if pixel_values is not None else None,
            image_grid_thw=deepcopy(image_grid_thw) if image_grid_thw is not None else None,
        )
        rollouts.append(rollout)
    return rollouts
