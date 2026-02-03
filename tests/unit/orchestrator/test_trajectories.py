import base64
from io import BytesIO
from unittest.mock import MagicMock

import pytest
import verifiers as vf
from PIL import Image

from prime_rl.orchestrator.trajectories import (
    VLMImageCache,
    _extract_images_from_examples,
    _extract_images_from_messages,
    branch_rollout,
    build_vlm_image_cache,
    interleave_rollout,
)


@pytest.fixture
def single_step_trajectory_state():
    state = vf.State(
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
                temperature=1.0,
            )
        ],
        error=None,
    )
    return state


@pytest.fixture
def multi_step_trajectory_state():
    state = vf.State(
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
                temperature=1.0,
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                ],
                completion=[{"role": "assistant", "content": "A2"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3, 4, 5, 6],
                    prompt_mask=[0, 0, 0, 0, 0, 0],
                    completion_ids=[7, 8],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.3, -0.4],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
                temperature=1.0,
            ),
        ],
        error=None,
    )
    return state


@pytest.fixture
def multi_step_trajectory_with_tool_calls():
    state = vf.State(
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1 + TC1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
                temperature=1.0,
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1 + TC1"},
                    {"role": "tool", "tool_call_id": "TR1", "content": "TR1"},
                ],
                completion=[{"role": "assistant", "content": "A2 + TC2"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3, 4, 5, 6],
                    prompt_mask=[0, 0, 0, 0, 0, 0],
                    completion_ids=[7, 8],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.3, -0.4],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
                temperature=1.0,
            ),
        ],
        reward=1.0,
        advantage=None,
        stop_condition=None,
        metrics={"has_error": 0.0, "tool_calls": 1.0},
        error=None,
    )
    return state


def test_branching_rollout_single_step_trajectory(single_step_trajectory_state):
    rollouts = branch_rollout(single_step_trajectory_state)

    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2]
    assert rollout.completion_temperatures == [1.0, 1.0]


def test_branching_rollout_multi_step_trajectory(multi_step_trajectory_state):
    rollouts = branch_rollout(multi_step_trajectory_state)
    assert len(rollouts) == 2

    # first step
    rollout = rollouts[0]
    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2]
    assert rollout.completion_temperatures == [1.0, 1.0]

    # second step
    rollout = rollouts[1]
    assert rollout.prompt_ids == [1, 2, 3, 4, 5, 6]
    assert rollout.prompt_mask == [False, False, False, False, False, False]
    assert rollout.completion_ids == [7, 8]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.3, -0.4]
    assert rollout.completion_temperatures == [1.0, 1.0]


def test_branching_rollout_multi_step_trajectory_with_tool_calls(multi_step_trajectory_with_tool_calls):
    rollouts = branch_rollout(multi_step_trajectory_with_tool_calls)
    assert len(rollouts) == 2

    # first step
    rollout = rollouts[0]
    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2]
    assert rollout.completion_temperatures == [1.0, 1.0]

    # second step
    rollout = rollouts[1]
    assert rollout.prompt_ids == [1, 2, 3, 4, 5, 6]
    assert rollout.prompt_mask == [False, False, False, False, False, False]
    assert rollout.completion_ids == [7, 8]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.3, -0.4]
    assert rollout.completion_temperatures == [1.0, 1.0]


def test_interleave_rollout_single_step_trajectory(single_step_trajectory_state):
    rollouts = interleave_rollout(single_step_trajectory_state)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2]
    assert rollout.completion_temperatures == [1.0, 1.0]


def test_interleave_rollout_multi_step_trajectory(multi_step_trajectory_state):
    rollouts = interleave_rollout(multi_step_trajectory_state)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4, 5, 6, 7, 8]
    assert rollout.completion_mask == [True, True, False, False, True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2, 0, 0, -0.3, -0.4]
    # Temperatures: 2 completion tokens at temp 1.0, then 2 prompt tokens at temp 1.0, then 2 completion tokens at temp 1.0
    assert rollout.completion_temperatures == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_interleave_rollout_multi_step_trajectory_with_tool_calls(multi_step_trajectory_with_tool_calls):
    rollouts = interleave_rollout(multi_step_trajectory_with_tool_calls)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4, 5, 6, 7, 8]
    assert rollout.completion_mask == [True, True, False, False, True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2, 0, 0, -0.3, -0.4]
    # Temperatures: 2 completion tokens at temp 1.0, then 2 prompt tokens at temp 1.0, then 2 completion tokens at temp 1.0
    assert rollout.completion_temperatures == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


# =============================================================================
# VLM Multi-Turn Tests
# =============================================================================


def _create_test_image(color: str = "red") -> str:
    """Create a small test image and return its base64 data URL."""
    colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
    img = Image.new("RGB", (10, 10), colors.get(color, (255, 255, 255)))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _create_image_message(image_url: str, text: str = "What is this?") -> dict:
    """Create an OpenAI-style user message with an image."""
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": text},
        ],
    }


def test_extract_images_from_messages_no_images():
    messages = [{"role": "user", "content": "Hello"}]
    images = _extract_images_from_messages(messages)
    assert images == []


def test_extract_images_from_messages_single_image():
    image_url = _create_test_image("red")
    messages = [_create_image_message(image_url)]
    images = _extract_images_from_messages(messages)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)


def test_extract_images_from_messages_multiple_images():
    messages = [
        _create_image_message(_create_test_image("red")),
        {"role": "assistant", "content": "I see a red image"},
        _create_image_message(_create_test_image("green")),
    ]
    images = _extract_images_from_messages(messages)
    assert len(images) == 2


def test_extract_images_from_examples_single_turn():
    image_url = _create_test_image("red")
    state = vf.State(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[_create_image_message(image_url)],
                completion=[{"role": "assistant", "content": "A red square"}],
                response=MagicMock(),
                tokens=MagicMock(),
                temperature=1.0,
            )
        ],
        error=None,
    )

    all_images, images_per_step = _extract_images_from_examples([(1, state)])

    assert len(all_images) == 1
    assert images_per_step == {1: [1]}  # 1 image after step 0


def test_extract_images_from_examples_multi_turn_new_image_each_turn():
    """Test that new images in later turns are correctly extracted."""
    red_url = _create_test_image("red")
    green_url = _create_test_image("green")

    state = vf.State(
        example_id=1,
        trajectory=[
            # Turn 1: just the red image
            vf.TrajectoryStep(
                prompt=[_create_image_message(red_url, "What color is this?")],
                completion=[{"role": "assistant", "content": "Red"}],
                response=MagicMock(),
                tokens=MagicMock(),
                temperature=1.0,
            ),
            # Turn 2: cumulative prompt with red image + green image
            vf.TrajectoryStep(
                prompt=[
                    _create_image_message(red_url, "What color is this?"),
                    {"role": "assistant", "content": "Red"},
                    _create_image_message(green_url, "And this one?"),
                ],
                completion=[{"role": "assistant", "content": "Green"}],
                response=MagicMock(),
                tokens=MagicMock(),
                temperature=1.0,
            ),
        ],
        error=None,
    )

    all_images, images_per_step = _extract_images_from_examples([(1, state)])

    assert len(all_images) == 2  # 2 unique images total
    assert images_per_step == {1: [1, 2]}  # 1 after step 0, 2 after step 1


def test_extract_images_from_examples_multi_turn_no_new_images():
    """Test turns where no new images are added."""
    red_url = _create_test_image("red")

    state = vf.State(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[_create_image_message(red_url)],
                completion=[{"role": "assistant", "content": "Red"}],
                response=MagicMock(),
                tokens=MagicMock(),
                temperature=1.0,
            ),
            # Turn 2: same image, no new ones
            vf.TrajectoryStep(
                prompt=[
                    _create_image_message(red_url),
                    {"role": "assistant", "content": "Red"},
                    {"role": "user", "content": "Are you sure?"},  # text only
                ],
                completion=[{"role": "assistant", "content": "Yes"}],
                response=MagicMock(),
                tokens=MagicMock(),
                temperature=1.0,
            ),
        ],
        error=None,
    )

    all_images, images_per_step = _extract_images_from_examples([(1, state)])

    assert len(all_images) == 1  # Only 1 unique image
    assert images_per_step == {1: [1, 1]}  # 1 after step 0, still 1 after step 1


def test_vlm_image_cache_get_for_step():
    cache_data = {
        1: [
            ([[1.0, 2.0]], [[1, 2, 3]]),  # Step 0: 1 image
            ([[1.0, 2.0], [3.0, 4.0]], [[1, 2, 3], [1, 4, 4]]),  # Step 1: 2 images cumulative
        ],
    }
    cache = VLMImageCache(cache_data, num_unique_examples=1, extract_time=0.0, preprocess_time=0.0)

    # Step 0 should have 1 image
    pv, grid = cache.get_for_step(1, 0)
    assert pv == [[1.0, 2.0]]
    assert grid == [[1, 2, 3]]

    # Step 1 should have 2 images
    pv, grid = cache.get_for_step(1, 1)
    assert pv == [[1.0, 2.0], [3.0, 4.0]]
    assert grid == [[1, 2, 3], [1, 4, 4]]


def test_vlm_image_cache_get_all():
    cache_data = {
        1: [
            ([[1.0]], [[1, 2, 3]]),
            ([[1.0], [2.0]], [[1, 2, 3], [1, 4, 4]]),
        ],
    }
    cache = VLMImageCache(cache_data, num_unique_examples=1, extract_time=0.0, preprocess_time=0.0)

    # get_all should return the last step's data
    pv, grid = cache.get_all(1)
    assert pv == [[1.0], [2.0]]
    assert grid == [[1, 2, 3], [1, 4, 4]]


def test_vlm_image_cache_step_out_of_range():
    cache_data = {
        1: [
            ([[1.0]], [[1, 2, 3]]),
        ],
    }
    cache = VLMImageCache(cache_data, num_unique_examples=1, extract_time=0.0, preprocess_time=0.0)

    pv, grid = cache.get_for_step(1, 2)
    assert pv is None
    assert grid is None


def test_vlm_image_cache_missing_example():
    cache = VLMImageCache({}, num_unique_examples=0, extract_time=0.0, preprocess_time=0.0)

    pv, grid = cache.get_for_step(999, 0)
    assert pv is None
    assert grid is None

    pv, grid = cache.get_all(999)
    assert pv is None
    assert grid is None


def test_branch_rollout_with_vlm_cache():
    """Test that branch_rollout correctly uses per-step images from cache."""
    cache_data = {
        1: [
            ([[1.0]], [[1, 2, 3]]),  # Step 0
            ([[1.0], [2.0]], [[1, 2, 3], [1, 4, 4]]),  # Step 1
        ],
    }
    cache = VLMImageCache(cache_data, num_unique_examples=1, extract_time=0.0, preprocess_time=0.0)

    state = vf.State(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Turn 1"}],
                completion=[{"role": "assistant", "content": "Response 1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                temperature=1.0,
            ),
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Turn 2"}],
                completion=[{"role": "assistant", "content": "Response 2"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3, 4, 5],
                    prompt_mask=[0, 0, 0, 0, 0],
                    completion_ids=[6, 7],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.3, -0.4],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                temperature=1.0,
            ),
        ],
        error=None,
    )

    rollouts = branch_rollout(state, vlm_cache=cache)

    assert len(rollouts) == 2

    # First rollout should have step 0's images (1 image)
    assert rollouts[0].pixel_values == [[1.0]]
    assert rollouts[0].image_grid_thw == [[1, 2, 3]]

    # Second rollout should have step 1's cumulative images (2 images)
    assert rollouts[1].pixel_values == [[1.0], [2.0]]
    assert rollouts[1].image_grid_thw == [[1, 2, 3], [1, 4, 4]]


def test_branch_rollout_uses_cache_key_override():
    cache_data = {
        7: [
            ([[9.0]], [[1, 2, 3]]),
        ],
    }
    cache = VLMImageCache(cache_data, num_unique_examples=1, extract_time=0.0, preprocess_time=0.0)

    state = vf.State(
        example_id=123,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Turn 1"}],
                completion=[{"role": "assistant", "content": "Response 1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                temperature=1.0,
            ),
        ],
        error=None,
    )

    rollouts = branch_rollout(state, vlm_cache=cache, cache_key=7)

    assert len(rollouts) == 1
    assert rollouts[0].pixel_values == [[9.0]]
    assert rollouts[0].image_grid_thw == [[1, 2, 3]]


def test_build_vlm_image_cache_handles_divergent_rollouts():
    """Test that build_vlm_image_cache keys images per rollout when trajectories diverge."""
    import torch

    red_url = _create_test_image("red")
    blue_url = _create_test_image("blue")
    green_url = _create_test_image("green")

    rollout_a = vf.State(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[_create_image_message(red_url, "What color?")],
                completion=[{"role": "assistant", "content": "Red"}],
                response=MagicMock(),
                tokens=MagicMock(),
                temperature=1.0,
            ),
        ],
        error=None,
    )

    rollout_b = vf.State(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[_create_image_message(blue_url, "What color?")],
                completion=[{"role": "assistant", "content": "Blue"}],
                response=MagicMock(),
                tokens=MagicMock(),
                temperature=1.0,
            ),
            vf.TrajectoryStep(
                prompt=[
                    _create_image_message(blue_url, "What color?"),
                    {"role": "assistant", "content": "Blue"},
                    _create_image_message(green_url, "And this one?"),
                ],
                completion=[{"role": "assistant", "content": "Green"}],
                response=MagicMock(),
                tokens=MagicMock(),
                temperature=1.0,
            ),
        ],
        error=None,
    )

    # Mock processor that returns predictable tensors
    mock_processor = MagicMock()
    mock_processor.image_processor = MagicMock(
        side_effect=lambda images, return_tensors: {
            "pixel_values": torch.arange(len(images), dtype=torch.float32).view(-1, 1),
            "image_grid_thw": torch.tensor([[1, 1, 1]] * len(images)),
        }
    )

    rollouts = [rollout_a, rollout_b]
    cache = build_vlm_image_cache(rollouts, mock_processor)

    assert cache.num_unique_examples == 1

    pv, grid = cache.get_for_step(0, 0)
    assert pv == [[0.0]]
    assert grid == [[1, 1, 1]]

    pv, grid = cache.get_for_step(1, 0)
    assert pv == [[1.0]]
    assert grid == [[1, 1, 1]]

    pv, grid = cache.get_for_step(1, 1)
    assert pv == [[1.0], [2.0]]
    assert grid == [[1, 1, 1], [1, 1, 1]]


def test_build_vlm_image_cache_no_images():
    state = vf.State(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Hello"}],
                completion=[{"role": "assistant", "content": "Hi"}],
                response=MagicMock(),
                tokens=MagicMock(),
                temperature=1.0,
            )
        ],
        error=None,
    )

    cache = build_vlm_image_cache([state], MagicMock())

    pv, grid = cache.get_for_step(0, 0)
    assert pv is None
    assert grid is None
