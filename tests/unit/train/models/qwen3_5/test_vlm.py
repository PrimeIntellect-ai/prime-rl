import pytest
import torch

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import can_reinit_empty_buffers, resolve_auto_attn
from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_5 import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5VisionConfig,
)
from prime_rl.trainer.models.qwen3_5.rotary_embedding import build_qwen3_5_mrope_position_ids


def get_vlm_config():
    return Qwen3_5MoeConfig(
        text_config=Qwen3_5MoeTextConfig(
            vocab_size=256,
            hidden_size=256,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=512,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [3, 3, 2],
                "mrope_interleaved": True,
            },
        ),
        vision_config=Qwen3_5VisionConfig(
            depth=2,
            hidden_size=128,
            intermediate_size=256,
            num_heads=4,
            out_hidden_size=256,
        ),
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
        vision_end_token_id=253,
    )


def get_model(config, device="cuda"):
    runtime_config = ModelConfig()
    resolve_auto_attn(runtime_config)
    with torch.device(device):
        model = AutoModelForCausalLMPrimeRL.from_config(
            config, attn_implementation=runtime_config.attn, dtype=torch.bfloat16
        )
    inject_prime_lm_head(model)
    return model


def get_image_inputs(config, device="cuda", dtype=torch.bfloat16):
    """Create minimal image inputs matching the vision config."""
    vc = config.vision_config
    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    image_grid_thw = torch.tensor([[1, 2, 2]], device=device)
    num_patches = int(image_grid_thw.prod().item())
    pixel_values = torch.randn(num_patches, patch_dim, device=device, dtype=dtype)
    num_image_tokens = num_patches // (vc.spatial_merge_size**2)
    return pixel_values, image_grid_thw, num_image_tokens


def get_mm_token_type_ids(input_ids, image_token_id):
    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[input_ids == image_token_id] = 1
    return mm_token_type_ids


def get_seq_lens(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.tensor([input_ids.shape[1]], device=input_ids.device)


@pytest.mark.gpu
def test_vlm_forward():
    """Custom VLM produces logits for both text-only and multimodal inputs."""
    config = get_vlm_config()
    model = get_model(config)

    vocab = config.text_config.vocab_size

    # Text-only (avoid special token range 250-253)
    input_ids = torch.randint(0, 200, (1, 20), device="cuda")
    position_ids = torch.arange(1, 21, device="cuda").unsqueeze(0)
    out_text = model(input_ids=input_ids, position_ids=position_ids, seq_lens=get_seq_lens(input_ids))
    assert out_text["logits"].shape == (1, 20, vocab)

    # Multimodal
    pixel_values, image_grid_thw, n_img_tokens = get_image_inputs(config)
    text_part = torch.randint(0, 200, (1, 10), device="cuda")
    img_part = torch.full((1, n_img_tokens), config.image_token_id, device="cuda")
    input_ids_mm = torch.cat([text_part[:, :5], img_part, text_part[:, 5:]], dim=1)
    mm_token_type_ids = get_mm_token_type_ids(input_ids_mm, config.image_token_id)

    out_mm = model(
        input_ids=input_ids_mm,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        seq_lens=get_seq_lens(input_ids_mm),
    )
    assert out_mm["logits"].shape == (1, input_ids_mm.shape[1], vocab)


@pytest.mark.gpu
def test_vlm_backward():
    """Gradients flow through both vision scatter and text model."""
    config = get_vlm_config()
    model = get_model(config)

    pixel_values, image_grid_thw, n_img_tokens = get_image_inputs(config)
    text_part = torch.randint(0, 200, (1, 10), device="cuda")
    img_part = torch.full((1, n_img_tokens), config.image_token_id, device="cuda")
    input_ids = torch.cat([text_part[:, :5], img_part, text_part[:, 5:]], dim=1)
    mm_token_type_ids = get_mm_token_type_ids(input_ids, config.image_token_id)

    out = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        seq_lens=get_seq_lens(input_ids),
    )
    out["logits"].sum().backward()

    assert model.model.language_model.embed_tokens.weight.grad is not None
    assert model.model.visual.patch_embed.proj.weight.grad is not None


@pytest.mark.gpu
def test_vlm_router_replay():
    """routed_experts bypasses router computation in VLM multimodal forward."""
    config = get_vlm_config()
    model = get_model(config)

    vocab = config.text_config.vocab_size
    pixel_values, image_grid_thw, n_img_tokens = get_image_inputs(config)
    text_part = torch.randint(0, 200, (1, 10), device="cuda")
    img_part = torch.full((1, n_img_tokens), config.image_token_id, device="cuda")
    input_ids = torch.cat([text_part[:, :5], img_part, text_part[:, 5:]], dim=1)
    mm_token_type_ids = get_mm_token_type_ids(input_ids, config.image_token_id)
    seq_len = input_ids.shape[1]

    num_layers = config.text_config.num_hidden_layers
    topk = config.text_config.num_experts_per_tok
    routed_experts = torch.randint(0, config.text_config.num_experts, (1, seq_len, num_layers, topk), device="cuda")

    out = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        routed_experts=routed_experts,
        seq_lens=get_seq_lens(input_ids),
    )
    assert out["logits"].shape == (1, seq_len, vocab)

    out["logits"].sum().backward()
    assert model.model.language_model.embed_tokens.weight.grad is not None


@pytest.mark.gpu
def test_vlm_meta_device_and_buffer_reinit():
    """Model can be created on meta device and buffers reinitialized."""
    config = get_vlm_config()
    model = get_model(config, device="meta")

    assert can_reinit_empty_buffers(model)

    model.to_empty(device="cuda")
    model.init_buffers_post_meta()

    lm_inv = model.model.language_model.rotary_emb.inv_freq
    lm_original_inv = model.model.language_model.rotary_emb.original_inv_freq
    vis_inv = model.model.visual.rotary_pos_emb.inv_freq
    assert lm_inv.device.type == "cuda"
    assert lm_original_inv.device.type == "cuda"
    assert vis_inv.device.type == "cuda"
    assert lm_inv.abs().sum() > 0
    assert lm_original_inv.abs().sum() > 0
    assert vis_inv.abs().sum() > 0


def test_qwen35_mrope_text_only_positions():
    input_ids = torch.tensor([[10, 11, 12, 13]])
    mm_token_type_ids = torch.zeros_like(input_ids)

    position_ids = build_qwen3_5_mrope_position_ids(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=None,
        spatial_merge_size=2,
        seq_lens=torch.tensor([input_ids.shape[1]]),
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
        seq_lens=torch.tensor([input_ids.shape[1]]),
    )

    expected = torch.tensor(
        [
            [[0, 1, 2, 2, 2, 2, 4]],
            [[0, 1, 2, 2, 3, 3, 4]],
            [[0, 1, 2, 3, 2, 3, 4]],
        ]
    )
    torch.testing.assert_close(position_ids, expected)


def test_qwen35_mrope_adjacent_images_consume_multiple_grids():
    input_ids = torch.tensor([[10, 99, 99, 99, 99, 99, 99, 99, 99, 11]])
    mm_token_type_ids = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
    image_grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]])

    position_ids = build_qwen3_5_mrope_position_ids(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=2,
        seq_lens=torch.tensor([input_ids.shape[1]]),
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
        seq_lens=torch.tensor([segment_input_ids.shape[1]]),
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
            seq_lens=torch.tensor([input_ids.shape[1]]),
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
            seq_lens=torch.tensor([input_ids.shape[1]]),
        )
