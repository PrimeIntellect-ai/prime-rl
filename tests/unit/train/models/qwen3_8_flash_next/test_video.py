"""Qwen3.8 video inputs against vLLM's Qwen3-VL processing: frame sampling, timestamps, placeholder layout,
pixel values and MRoPE positions."""

import base64
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prime_rl.orchestrator.trajectories import _build_mm_refs
from prime_rl.trainer.models.qwen3_8_flash_next.rotary_embedding import build_mrope_position_ids
from prime_rl.trainer.models.qwen3_8_flash_next.vision import (
    materialize_qwen_vl_videos,
    qwen_vl_video_placeholder_ids,
    qwen_vl_video_timestamps,
)

MODEL = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
VISION_START, VISION_END, IMAGE_PAD, VIDEO_PAD = 248053, 248054, 248056, 248057
# A real clip to check as well (e.g. 4 s, 512x288, 10 fps), with the grid and placeholder length vLLM gave it.
VIDEO_ENV = "PRIME_RL_TEST_VIDEO"


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(char) for char in text]


def video_processor_defaults() -> SimpleNamespace:
    # Qwen3VLVideoProcessor class defaults
    return SimpleNamespace(fps=2, min_frames=4, max_frames=768, temporal_patch_size=2, merge_size=2)


def vllm_video_second_idx(metadata: dict) -> list[float]:
    from vllm.model_executor.models.qwen3_vl import Qwen3VLProcessingInfo

    info = SimpleNamespace(
        get_video_processor=video_processor_defaults,
        _calculate_timestamps=lambda *args: Qwen3VLProcessingInfo._calculate_timestamps(None, *args),
    )
    return Qwen3VLProcessingInfo._get_video_second_idx(info, metadata)


@pytest.mark.parametrize(
    "metadata",
    [
        # vLLM's qwen3_vl loader sampled 8 of 40 frames at 2 fps.
        {
            "total_num_frames": 40,
            "fps": 10.0,
            "frames_indices": [0, 6, 11, 17, 22, 28, 33, 39],
            "do_sample_frames": False,
        },
        # An odd frame count pads the last temporal patch with the last frame.
        {"total_num_frames": 50, "fps": 25.0, "frames_indices": [0, 12, 24, 37, 49], "do_sample_frames": False},
        # The loader kept every frame, so the HF processor samples them again.
        {"total_num_frames": 4, "fps": 2.0, "frames_indices": [0, 1, 2, 3], "do_sample_frames": True},
        {"total_num_frames": 6, "fps": 1.0, "frames_indices": list(range(6)), "do_sample_frames": True},
    ],
)
def test_video_timestamps_match_vllm(metadata):
    pytest.importorskip("vllm")
    assert qwen_vl_video_timestamps(metadata, video_processor_defaults()) == vllm_video_second_idx(metadata)


@pytest.mark.parametrize("timestamps", [[0.3], [0.3, 1.4, 2.5, 3.5999999999999996], [9.95, 10.05]])
def test_video_placeholder_ids_match_vllm(timestamps):
    pytest.importorskip("vllm")
    from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor

    grid_thw = torch.tensor([len(timestamps), 4, 6])
    expected = Qwen3VLMultiModalProcessor.get_video_repl(
        tokens_per_frame=[6] * len(timestamps),
        timestamps=timestamps,
        tokenizer=CharTokenizer(),
        vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END,
        video_token_id=VIDEO_PAD,
        select_token_id=True,
    ).full
    actual = qwen_vl_video_placeholder_ids(
        grid_thw=grid_thw,
        timestamps=timestamps,
        tokenizer=CharTokenizer(),
        merge_size=2,
        vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END,
        video_token_id=VIDEO_PAD,
    )
    assert actual == expected


def video_prompt(frame_tokens: int, timestamps: list[float]) -> list[int]:
    """Text, a video in the prompt's own vision markers as vLLM expands it, text, an image, text."""
    ids = [10, 11, VISION_START]
    for timestamp in timestamps:
        ids += CharTokenizer().encode(f"<{timestamp:.1f} seconds>", add_special_tokens=False)
        ids += [VISION_START, *[VIDEO_PAD] * frame_tokens, VISION_END]
    return ids + [VISION_END, 12, VISION_START, *[IMAGE_PAD] * 6, VISION_END, 13, 14]


def vllm_mrope_positions(input_ids: list[int], features: list) -> torch.Tensor:
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

    config = SimpleNamespace(
        video_token_id=VIDEO_PAD,
        vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    positions, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(input_ids, features, config)
    return positions


def feature(modality: str, grid_thw: list[int], offset: int) -> SimpleNamespace:
    return SimpleNamespace(
        modality=modality,
        data={f"{modality}_grid_thw": SimpleNamespace(data=torch.tensor(grid_thw))},
        mm_position=SimpleNamespace(offset=offset),
    )


def test_video_mrope_positions_match_vllm():
    pytest.importorskip("vllm")
    timestamps = [0.3, 1.4, 2.5]
    ids = video_prompt(frame_tokens=6, timestamps=timestamps)
    video_grid, image_grid = [3, 4, 6], [1, 4, 6]
    features = [feature("video", video_grid, 3), feature("image", image_grid, ids.index(IMAGE_PAD))]
    input_ids = torch.tensor([ids])
    token_types = torch.zeros_like(input_ids)
    token_types[input_ids == IMAGE_PAD] = 1
    token_types[input_ids == VIDEO_PAD] = 2

    positions = build_mrope_position_ids(
        input_ids=input_ids,
        mm_token_type_ids=token_types,
        image_grid_thw=torch.tensor([image_grid]),
        video_grid_thw=torch.tensor([video_grid]),
        spatial_merge_size=2,
        seq_lens=torch.tensor([len(ids)]),
    )
    torch.testing.assert_close(positions[:, 0], vllm_mrope_positions(ids, features).to(positions.dtype))

    # Packed with a copy of itself, each sequence restarts from position 0.
    packed = build_mrope_position_ids(
        input_ids=input_ids.repeat(1, 2),
        mm_token_type_ids=token_types.repeat(1, 2),
        image_grid_thw=torch.tensor([image_grid, image_grid]),
        video_grid_thw=torch.tensor([video_grid, video_grid]),
        spatial_merge_size=2,
        seq_lens=torch.tensor([len(ids), len(ids)]),
    )
    torch.testing.assert_close(packed[:, :, : len(ids)], positions)
    torch.testing.assert_close(packed[:, :, len(ids) :], positions)


def test_video_mrope_rejects_frame_length_mismatch():
    ids = video_prompt(frame_tokens=6, timestamps=[0.3])
    input_ids = torch.tensor([ids])
    token_types = torch.zeros_like(input_ids)
    token_types[input_ids == IMAGE_PAD] = 1
    token_types[input_ids == VIDEO_PAD] = 2
    with pytest.raises(ValueError, match="Video token group length does not match video_grid_thw"):
        build_mrope_position_ids(
            input_ids=input_ids,
            mm_token_type_ids=token_types,
            image_grid_thw=torch.tensor([[1, 4, 6]]),
            video_grid_thw=torch.tensor([[1, 6, 6]]),
            spatial_merge_size=2,
            seq_lens=torch.tensor([len(ids)]),
        )


def model_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(MODEL, "video_preprocessor_config.json", revision=REVISION), str)


def write_synthetic_video(path: Path) -> None:
    """2.5 s, 10 fps, 96x64: vLLM samples 5 frames, so the last temporal patch is padded."""
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
    generator = np.random.default_rng(0)
    for index in range(25):
        frame = generator.integers(0, 256, (64, 96, 3), dtype=np.uint8)
        frame[:, : 4 * index % 96] = (index * 10, 255 - index * 10, 128)
        writer.write(frame)
    writer.release()


def video_files(tmp_path: Path) -> list[tuple[Path, dict | None]]:
    synthetic = tmp_path / "synthetic.mp4"
    write_synthetic_video(synthetic)
    files = [(synthetic, None)]
    if os.environ.get(VIDEO_ENV):
        # The 4 s, 512x288, 10 fps sample clip: vLLM 0.32 gives it 4 temporal patches of 18x32 patches.
        files.append((Path(os.environ[VIDEO_ENV]), {"grid": [4, 18, 32], "pads": 576, "length": 608}))
    return files


def test_materialized_videos_match_vllm_processor(tmp_path):
    """The orchestrator's ref, the trainer's pixel values and grid, and the MRoPE positions, against what vLLM's own
    loader and Qwen3-VL multimodal processor produce for the same data URL."""
    pytest.importorskip("vllm")
    pytest.importorskip("cv2")
    if not model_cached():
        pytest.skip(f"{MODEL}: processor files not cached locally")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    for path, expected in video_files(tmp_path):
        out = tmp_path / f"{path.stem}.pt"
        script = Path(__file__).with_name("video_parity.py")
        command = ["processor", "--model", MODEL, "--revision", REVISION, "--video", path, "--out", out]
        subprocess.run([sys.executable, script, *command], check=True)
        reference = torch.load(out)
        ids = np.asarray(reference["prompt_ids"])
        token_types = np.zeros_like(ids)
        token_types[ids == VIDEO_PAD] = 2

        url = "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()
        refs = _build_mm_refs([], [url], ids, token_types)
        assert [(ref.offset, ref.length) for ref in refs.videos] == [reference["placeholder"]]

        kwargs = materialize_qwen_vl_videos(refs, processor)
        assert torch.equal(kwargs["video_grid_thw"], reference["video_grid_thw"])
        # vLLM normalizes uint8 patches on device in fp32 with one rounding (its Triton kernel); the HF float
        # pixels give the same bf16 model input.
        raw = reference["pixel_values_videos"]
        assert raw.dtype == torch.uint8
        weight = ((1 / 255) / torch.tensor(0.5)).double()  # vLLM's FusedMMInputNorm, from the 0.5 mean / std
        normalized = (raw.double() * weight - 1.0).float()
        assert torch.equal(kwargs["pixel_values_videos"].bfloat16(), normalized.bfloat16())

        offset, length = reference["placeholder"]
        placeholder_ids = qwen_vl_video_placeholder_ids(
            grid_thw=kwargs["video_grid_thw"][0],
            timestamps=reference["timestamps"][0],
            tokenizer=processor.tokenizer,
            merge_size=2,
            vision_start_token_id=VISION_START,
            vision_end_token_id=VISION_END,
            video_token_id=VIDEO_PAD,
        )
        assert placeholder_ids == ids[offset : offset + length].tolist()

        positions = build_mrope_position_ids(
            input_ids=torch.tensor([ids]),
            mm_token_type_ids=torch.tensor([token_types]),
            image_grid_thw=None,
            video_grid_thw=kwargs["video_grid_thw"],
            spatial_merge_size=2,
            seq_lens=torch.tensor([len(ids)]),
        )
        torch.testing.assert_close(positions[:, 0], reference["positions"].to(positions.dtype))

        if expected is not None:
            assert kwargs["video_grid_thw"].tolist() == [expected["grid"]]
            assert int((ids == VIDEO_PAD).sum()) == expected["pads"]
            assert length == expected["length"]
