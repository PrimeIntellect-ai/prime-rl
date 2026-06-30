import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig

from prime_rl.trainer.model import can_reinit_empty_buffers
from prime_rl.trainer.models import get_custom_vlm_cls
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_5 import Qwen3_5ForCausalLM
from prime_rl.utils.utils import default_dtype


def _tiny_vlm_config() -> Qwen3_5Config:
    text_config = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=512,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"],
        use_cache=False,
    )
    text_config._attn_implementation = "sdpa"
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        out_hidden_size=text_config.hidden_size,
    )
    vision_config._attn_implementation = "sdpa"
    config = Qwen3_5Config(text_config=text_config, vision_config=vision_config)
    config._attn_implementation = "sdpa"
    config.image_token_id = 250
    config.video_token_id = 251
    config.vision_start_token_id = 252
    config.vision_end_token_id = 253
    return config


def _make_image_inputs(config: Qwen3_5Config, device: str = "cpu"):
    vc = config.vision_config
    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    image_grid_thw = torch.tensor([[1, 2, 2]], device=device)
    num_patches = int(image_grid_thw.prod().item())
    pixel_values = torch.randn(num_patches, patch_dim, device=device)
    num_image_tokens = num_patches // (vc.spatial_merge_size**2)
    return pixel_values, image_grid_thw, num_image_tokens


def test_dense_qwen3_5_vlm_mapping():
    config = _tiny_vlm_config()

    assert config.model_type == "qwen3_5"
    assert get_custom_vlm_cls(config) is Qwen3_5ForCausalLM


def test_dense_qwen3_5_vlm_forward_text_and_image_cpu():
    config = _tiny_vlm_config()
    with default_dtype(torch.float32):
        model = Qwen3_5ForCausalLM(config)
    inject_prime_lm_head(model)

    input_ids = torch.randint(0, 200, (1, 12))
    out_text = model(input_ids=input_ids)
    assert out_text["logits"].shape == (1, input_ids.shape[1], config.text_config.vocab_size)

    pixel_values, image_grid_thw, n_img_tokens = _make_image_inputs(config)
    text_part = torch.randint(0, 200, (1, 10))
    img_part = torch.full((1, n_img_tokens), config.image_token_id)
    input_ids_mm = torch.cat([text_part[:, :5], img_part, text_part[:, 5:]], dim=1)
    mm_token_type_ids = torch.zeros_like(input_ids_mm)
    mm_token_type_ids[input_ids_mm == config.image_token_id] = 1

    out_mm = model(
        input_ids=input_ids_mm,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
    )
    assert out_mm["logits"].shape == (1, input_ids_mm.shape[1], config.text_config.vocab_size)


def test_dense_qwen3_5_vlm_meta_device_and_buffer_reinit():
    config = _tiny_vlm_config()
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM.from_config(config)

    assert can_reinit_empty_buffers(model)

    model.to_empty(device="cpu")
    model.init_buffers_post_meta()

    lm_inv = model.model.language_model.rotary_emb.inv_freq
    vis_inv = model.model.visual.rotary_pos_emb.inv_freq
    assert lm_inv.device.type == "cpu"
    assert vis_inv.device.type == "cpu"
    assert lm_inv.abs().sum() > 0
    assert vis_inv.abs().sum() > 0
