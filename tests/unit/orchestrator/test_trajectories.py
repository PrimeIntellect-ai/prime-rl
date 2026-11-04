import numpy as np
import pytest

from prime_rl.orchestrator.trajectories import _build_mm_refs

VISION_START, VISION_END, IMAGE_PAD, VIDEO_PAD = 248053, 248054, 248056, 248057
TIMESTAMP = [27, 15, 13, 18, 6, 29]  # "<0.3 seconds>"


def video(frames: int, frame_tokens: int) -> list[int]:
    """A prompt's <|vision_start|><|video_pad|><|vision_end|> as vLLM expands it."""
    expanded = []
    for _ in range(frames):
        expanded += [*TIMESTAMP, VISION_START, *[VIDEO_PAD] * frame_tokens, VISION_END]
    return [VISION_START, *expanded, VISION_END]


def token_types(token_ids: list[int]) -> np.ndarray:
    ids = np.asarray(token_ids)
    types = np.zeros_like(ids)
    types[ids == IMAGE_PAD] = 1
    types[ids == VIDEO_PAD] = 2
    return types


def test_video_refs_cover_vllm_placeholder_ranges():
    image = [VISION_START, IMAGE_PAD, IMAGE_PAD, VISION_END]
    ids = [1, 2, *video(frames=3, frame_tokens=4), 3, *image, *video(frames=2, frame_tokens=1), 4, 5]
    refs = _build_mm_refs(["image"], ["first", "second"], np.asarray(ids), token_types(ids))

    first_start = 3
    first_length = 3 * (len(TIMESTAMP) + 4 + 2)
    second_start = 2 + len(video(3, 4)) + 1 + len(image) + 1
    assert [(ref.url, ref.offset, ref.length) for ref in refs.videos] == [
        ("first", first_start, first_length),
        ("second", second_start, 2 * (len(TIMESTAMP) + 1 + 2)),
    ]
    assert [(ref.offset, ref.length) for ref in refs.images] == [(2 + len(video(3, 4)) + 2, 2)]
    for ref in refs.videos:
        assert ids[ref.offset - 1] == VISION_START and ids[ref.offset + ref.length] == VISION_END


def test_video_refs_require_one_range_per_video():
    ids = [1, *video(frames=2, frame_tokens=2), 2]
    with pytest.raises(ValueError, match="videos=2, ranges=1"):
        _build_mm_refs([], ["first", "second"], np.asarray(ids), token_types(ids))
