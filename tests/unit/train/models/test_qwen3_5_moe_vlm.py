import pytest
import torch

from prime_rl.trainer.model import can_reinit_empty_buffers
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5VisionConfig,
)
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


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
        attn_implementation="flash_attention_2",
    )


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


def test_vlm_forward():
    """Custom VLM produces logits for both text-only and multimodal inputs."""
    config = get_vlm_config()
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = Qwen3_5ForCausalLM(config)
    inject_prime_lm_head(model)

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


def test_vlm_backward():
    """Gradients flow through both vision scatter and text model."""
    config = get_vlm_config()
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = Qwen3_5ForCausalLM(config)
    inject_prime_lm_head(model)

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


def test_vlm_router_replay():
    """routed_experts bypasses router computation in VLM multimodal forward."""
    config = get_vlm_config()
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = Qwen3_5ForCausalLM(config)
    inject_prime_lm_head(model)

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


def test_vlm_meta_device_and_buffer_reinit():
    """Model can be created on meta device and buffers reinitialized."""
    config = get_vlm_config()
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)

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
