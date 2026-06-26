import pytest
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTextRotaryEmbedding as HFQwen3_5MoeRotaryEmbedding,
)

from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeConfig
from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRotaryEmbedding
from prime_rl.trainer.models.qwen3_5_moe.mrope import build_qwen3_5_mrope_position_ids


def _tiny_config() -> Qwen3_5MoeConfig:
    return Qwen3_5MoeConfig(
        vocab_size=128,
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        num_hidden_layers=1,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
        },
    )


def test_qwen35_mrope_text_only_positions():
    input_ids = torch.tensor([[10, 11, 12, 13]])
    mm_token_type_ids = torch.zeros_like(input_ids)

    position_ids = build_qwen3_5_mrope_position_ids(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=None,
        spatial_merge_size=2,
    )

    expected = torch.arange(4).view(1, 1, -1).expand(3, 1, -1)
    torch.testing.assert_close(position_ids, expected)


def test_qwen35_mrope_single_image_with_surrounding_text():
    input_ids = torch.tensor([[10, 11, 99, 99, 99, 99, 12]])
    mm_token_type_ids = torch.tensor([[0, 0, 1, 1, 1, 1, 0]])
    image_grid_thw = torch.tensor([[1, 4, 4]])

    position_ids = build_qwen3_5_mrope_position_ids(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=2,
    )

    expected = torch.tensor(
        [
            [[0, 1, 2, 2, 2, 2, 4]],
            [[0, 1, 2, 2, 3, 3, 4]],
            [[0, 1, 2, 3, 2, 3, 4]],
        ]
    )
    torch.testing.assert_close(position_ids, expected)


def test_qwen35_mrope_adjacent_images_use_separate_grid_rows():
    input_ids = torch.tensor([[10, 99, 99, 99, 99, 99, 99, 99, 99, 11]])
    mm_token_type_ids = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
    image_grid_thw = torch.tensor([[1, 2, 2], [1, 2, 2]])

    position_ids = build_qwen3_5_mrope_position_ids(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=1,
    )

    expected = torch.tensor(
        [
            [[0, 1, 1, 1, 1, 3, 3, 3, 3, 5]],
            [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5]],
            [[0, 1, 2, 1, 2, 3, 4, 3, 4, 5]],
        ]
    )
    torch.testing.assert_close(position_ids, expected)


def test_qwen35_mrope_packed_segments_reset_independently():
    input_ids = torch.tensor([[10, 99, 99, 99, 99, 11, 99, 99, 99, 99]])
    mm_token_type_ids = torch.tensor([[0, 1, 1, 1, 1, 0, 1, 1, 1, 1]])
    image_grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]])
    seq_lens = torch.tensor([5, 5])

    position_ids = build_qwen3_5_mrope_position_ids(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=2,
        seq_lens=seq_lens,
    )

    expected_segment = torch.tensor(
        [
            [0, 1, 1, 1, 1],
            [0, 1, 1, 2, 2],
            [0, 1, 2, 1, 2],
        ]
    )
    torch.testing.assert_close(position_ids[:, 0, :5], expected_segment)
    torch.testing.assert_close(position_ids[:, 0, 5:], expected_segment)


def test_qwen35_mrope_packed_matches_standalone_segments():
    segment_input_ids = torch.tensor([[10, 99, 99, 99, 99, 11]])
    segment_token_types = torch.tensor([[0, 1, 1, 1, 1, 0]])
    image_grid_thw = torch.tensor([[1, 4, 4]])

    standalone = build_qwen3_5_mrope_position_ids(
        input_ids=segment_input_ids,
        mm_token_type_ids=segment_token_types,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=2,
    )
    packed = build_qwen3_5_mrope_position_ids(
        input_ids=torch.cat([segment_input_ids, segment_input_ids], dim=1),
        mm_token_type_ids=torch.cat([segment_token_types, segment_token_types], dim=1),
        image_grid_thw=torch.cat([image_grid_thw, image_grid_thw], dim=0),
        spatial_merge_size=2,
        seq_lens=torch.tensor([segment_input_ids.shape[1], segment_input_ids.shape[1]]),
    )

    torch.testing.assert_close(packed[:, :, : segment_input_ids.shape[1]], standalone)
    torch.testing.assert_close(packed[:, :, segment_input_ids.shape[1] :], standalone)


def test_qwen35_mrope_rejects_image_length_grid_mismatch():
    input_ids = torch.tensor([[10, 99, 99]])
    mm_token_type_ids = torch.tensor([[0, 1, 1]])
    image_grid_thw = torch.tensor([[1, 4, 4]])

    with pytest.raises(ValueError, match="Image token group length"):
        build_qwen3_5_mrope_position_ids(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=2,
        )


def test_qwen35_mrope_rejects_video_tokens():
    input_ids = torch.tensor([[10, 99]])
    mm_token_type_ids = torch.tensor([[0, 2]])

    with pytest.raises(ValueError, match="video MRoPE"):
        build_qwen3_5_mrope_position_ids(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=None,
            spatial_merge_size=2,
        )


def test_qwen35_rotary_matches_hf_for_2d_and_3d_positions():
    config = _tiny_config()
    prime_rotary = Qwen3_5MoeRotaryEmbedding(config)
    hf_rotary = HFQwen3_5MoeRotaryEmbedding(config)
    hidden_states = torch.randn(1, 6, config.hidden_size)

    text_positions = torch.arange(6).unsqueeze(0)
    prime_cos, prime_sin = prime_rotary(hidden_states, text_positions)
    hf_cos, hf_sin = hf_rotary(hidden_states, text_positions)
    torch.testing.assert_close(prime_cos, hf_cos)
    torch.testing.assert_close(prime_sin, hf_sin)

    mrope_positions = torch.stack(
        [
            torch.tensor([[0, 1, 2, 2, 2, 4]]),
            torch.tensor([[0, 1, 2, 2, 3, 4]]),
            torch.tensor([[0, 1, 2, 3, 2, 4]]),
        ]
    )
    prime_cos, prime_sin = prime_rotary(hidden_states, mrope_positions)
    hf_cos, hf_sin = hf_rotary(hidden_states, mrope_positions)
    torch.testing.assert_close(prime_cos, hf_cos)
    torch.testing.assert_close(prime_sin, hf_sin)
