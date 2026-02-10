import pytest

from prime_rl.inference.vllm.serving_chat_with_tokens import _collapse_image_placeholders

# Simulated Qwen3-VL special token IDs
VISION_START = 151652
IMAGE_PAD = 151655
VISION_END = 151653


def _make_vision_block(pad_count: int) -> list[int]:
    """Build a <|vision_start|> + N * <|image_pad|> + <|vision_end|> block."""
    return [VISION_START] + [IMAGE_PAD] * pad_count + [VISION_END]


# ---- Fast-path: no collapsing needed ----


def test_identical_tokens_returns_override():
    tokens = [1, 2, 3, 4, 5]
    assert _collapse_image_placeholders(tokens, tokens) is tokens


def test_empty_override_returns_empty():
    assert _collapse_image_placeholders([1, 2, 3], []) == []


def test_no_blocks_in_override_returns_unchanged():
    original = [1, 2, 3]
    override = [1, 2, 3, 4, 5]
    assert _collapse_image_placeholders(original, override) == override


def test_block_in_both_original_and_override_is_not_collapsed():
    """If a token appears in blocks in BOTH sequences, it's not a placeholder."""
    original = [1, 7, 7, 7, 2]
    override = [1, 7, 7, 7, 7, 7, 2]
    assert _collapse_image_placeholders(original, override) == override


# ---- VLM image placeholder collapsing ----


def test_single_expanded_block():
    """Turn 1: one image already expanded (64 pads), should collapse to 1."""
    original = [10, VISION_START, IMAGE_PAD, VISION_END, 20]
    override = [10, VISION_START] + [IMAGE_PAD] * 64 + [VISION_END, 20]
    result = _collapse_image_placeholders(original, override)
    assert result == original


def test_two_images_second_unexpanded():
    """Turn 1: image 1 expanded (64), image 2 unexpanded (1). Should collapse to 1 + 1."""
    original = [10, *_make_vision_block(1), 20, *_make_vision_block(1), 30]
    override = [10, *_make_vision_block(64), 20, *_make_vision_block(1), 30]
    result = _collapse_image_placeholders(original, override)
    assert result == original


def test_three_turn_compounding_inflation():
    """
    Reproduces the actual bug: 3-image conversation where _process_inputs
    re-expands already-expanded blocks, inflating 192 → 381 image_pad tokens.

    Turn 0 (message-based): 1 image → 64 pads (correct)
    Turn 1 (token-based): prev has 64 expanded + new has 1 unexpanded → 65 pads
      After buggy re-expansion: 64 + 64 + 63 = 191 pads
    Turn 2 (token-based): prev has 191 expanded + new has 1 unexpanded → 192 pads
      After buggy re-expansion: 381 pads

    The fix collapses expanded blocks before _process_inputs, so each image
    has exactly 1 placeholder token for correct expansion.
    """
    # Original engine tokens: 3 images, each with single placeholder
    original = [
        10,
        *_make_vision_block(1),  # image 1
        20,
        *_make_vision_block(1),  # image 2
        30,
        *_make_vision_block(1),  # image 3
        40,
    ]

    # Override tokens from get_prompt_ids at turn 2:
    # images 1+2 have been expanded in previous response (191 pads from the bug),
    # image 3 has 1 unexpanded pad from /tokenize
    override = [
        10,
        VISION_START,
        *([IMAGE_PAD] * 191),
        VISION_END,  # images 1+2 inflated
        30,
        *_make_vision_block(1),  # image 3 unexpanded
        40,
    ]

    result = _collapse_image_placeholders(original, override)

    # After collapsing, each block should have exactly 1 image_pad
    pad_count = sum(1 for t in result if t == IMAGE_PAD)
    assert pad_count == 2  # two vision blocks, one pad each


def test_realistic_turn1_token_sequence():
    """
    Realistic turn 1 scenario: prev turn response had 64 expanded pads,
    /tokenize returned 1 unexpanded pad for the new image.
    """
    # Surrounding text tokens
    system_tokens = [100, 101, 102]
    user1_text = [200, 201]
    assistant_response = [300, 301]
    user2_text = [400, 401]
    gen_prompt = [500]

    # Original: _preprocess_chat tokenizes full messages with single placeholders
    original = (
        system_tokens
        + user1_text
        + _make_vision_block(1)
        + assistant_response
        + user2_text
        + _make_vision_block(1)
        + gen_prompt
    )

    # Override: get_prompt_ids concatenated prev response (expanded) + tokenized new messages (unexpanded)
    override = (
        system_tokens
        + user1_text
        + _make_vision_block(64)  # expanded from prev response
        + assistant_response
        + user2_text
        + _make_vision_block(1)  # unexpanded from /tokenize
        + gen_prompt
    )

    result = _collapse_image_placeholders(original, override)

    # Both vision blocks should have exactly 1 pad token
    pad_count = sum(1 for t in result if t == IMAGE_PAD)
    assert pad_count == 2

    # Non-image tokens should be preserved
    non_pad = [t for t in result if t != IMAGE_PAD]
    non_pad_original = [t for t in original if t != IMAGE_PAD]
    assert non_pad == non_pad_original


@pytest.mark.parametrize("expanded_size", [4, 16, 64, 256])
def test_various_expansion_sizes(expanded_size):
    """Works for any expansion factor, not just 64."""
    original = _make_vision_block(1)
    override = _make_vision_block(expanded_size)
    result = _collapse_image_placeholders(original, override)
    assert result == original


def test_multiple_placeholder_types():
    """Handles models with both image_pad and video_pad tokens."""
    VIDEO_PAD = 151656

    original = [1, VISION_START, IMAGE_PAD, VISION_END, 2, VISION_START, VIDEO_PAD, VISION_END, 3]
    override = [
        1,
        VISION_START,
        *([IMAGE_PAD] * 64),
        VISION_END,
        2,
        VISION_START,
        *([VIDEO_PAD] * 32),
        VISION_END,
        3,
    ]

    result = _collapse_image_placeholders(original, override)
    assert sum(1 for t in result if t == IMAGE_PAD) == 1
    assert sum(1 for t in result if t == VIDEO_PAD) == 1


def test_text_only_no_placeholders():
    """Text-only models: no image tokens, function is a no-op."""
    original = [1, 2, 3, 4, 5]
    override = [1, 2, 3, 4, 5, 6, 7]
    assert _collapse_image_placeholders(original, override) == override
