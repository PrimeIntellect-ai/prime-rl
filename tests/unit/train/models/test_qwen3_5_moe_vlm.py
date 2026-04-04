import pytest
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration as HFQwen3_5MoeVLM,
)

from prime_rl.trainer.model import can_reinit_empty_buffers
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def _tiny_vlm_config():
    """HF composite config shrunk for unit testing."""
    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True, attn_implementation="sdpa")
    config.use_cache = False
    tc = config.text_config
    tc.vocab_size = 256
    tc.hidden_size = 256
    tc.num_hidden_layers = 2
    tc.layer_types = ["linear_attention", "full_attention"]
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    tc.head_dim = 64
    tc.moe_intermediate_size = 128
    tc.shared_expert_intermediate_size = 128
    tc.num_experts = 4
    tc.num_experts_per_tok = 2
    tc.max_position_embeddings = 512
    tc.linear_key_head_dim = 32
    tc.linear_value_head_dim = 32
    tc.linear_num_key_heads = 4
    tc.linear_num_value_heads = 8
    tc.use_cache = False

    vc = config.vision_config
    vc.depth = 2
    vc.hidden_size = 128
    vc.intermediate_size = 256
    vc.num_heads = 4
    vc.out_hidden_size = tc.hidden_size

    # Special token IDs must fit within the tiny vocab
    config.image_token_id = 250
    config.video_token_id = 251
    config.vision_start_token_id = 252
    config.vision_end_token_id = 253
    return config


def _make_image_inputs(config, device="cuda", dtype=torch.float32):
    """Create minimal image inputs matching the vision config."""
    vc = config.vision_config
    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    image_grid_thw = torch.tensor([[1, 2, 2]], device=device)
    num_patches = int(image_grid_thw.prod().item())
    pixel_values = torch.randn(num_patches, patch_dim, device=device, dtype=dtype)
    num_image_tokens = num_patches // (vc.spatial_merge_size**2)
    return pixel_values, image_grid_thw, num_image_tokens


def test_vlm_forward():
    """Custom VLM produces logits for both text-only and multimodal inputs."""
    config = _tiny_vlm_config()
    with torch.device("cuda"), default_dtype(torch.float32):
        model = Qwen3_5MoeForCausalLM(config)
    inject_prime_lm_head(model)

    vocab = config.text_config.vocab_size

    # Text-only (avoid special token range 250-253)
    input_ids = torch.randint(0, 200, (1, 20), device="cuda")
    position_ids = torch.arange(1, 21, device="cuda").unsqueeze(0)
    out_text = model(input_ids=input_ids, position_ids=position_ids)
    assert out_text["logits"].shape == (1, 20, vocab)

    # Multimodal
    pixel_values, image_grid_thw, n_img_tokens = _make_image_inputs(config)
    text_part = torch.randint(0, 200, (1, 10), device="cuda")
    img_part = torch.full((1, n_img_tokens), config.image_token_id, device="cuda")
    input_ids_mm = torch.cat([text_part[:, :5], img_part, text_part[:, 5:]], dim=1)

    out_mm = model(input_ids=input_ids_mm, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    assert out_mm["logits"].shape == (1, input_ids_mm.shape[1], vocab)


def test_vlm_backward():
    """Gradients flow through both vision scatter and text model."""
    config = _tiny_vlm_config()
    with torch.device("cuda"), default_dtype(torch.float32):
        model = Qwen3_5MoeForCausalLM(config)
    inject_prime_lm_head(model)

    pixel_values, image_grid_thw, n_img_tokens = _make_image_inputs(config)
    text_part = torch.randint(0, 200, (1, 10), device="cuda")
    img_part = torch.full((1, n_img_tokens), config.image_token_id, device="cuda")
    input_ids = torch.cat([text_part[:, :5], img_part, text_part[:, 5:]], dim=1)

    out = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    out["logits"].sum().backward()

    assert model.model.language_model.embed_tokens.weight.grad is not None
    assert model.model.visual.patch_embed.proj.weight.grad is not None


def test_vlm_weight_load_from_hf():
    """Weights from HF VLM checkpoint load correctly into custom VLM after conversion.

    Text model numerical match is already validated by test_qwen3_5_moe.py::test_qwen3_5_moe.
    This test verifies that VLM weight conversion + loading produces a working model.
    """
    config = _tiny_vlm_config()
    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = HFQwen3_5MoeVLM._from_config(config)
        prime_model = Qwen3_5MoeForCausalLM(config)

    # Copy weights: HF -> PrimeRL (with MoE conversion)
    with torch.no_grad():
        hf_sd = hf_model.state_dict()
        prime_model.convert_to_prime(hf_sd)
        prime_model.load_state_dict(hf_sd)
    inject_prime_lm_head(prime_model)

    # Verify vision encoder weights match exactly (should be untouched by conversion)
    for name, param in hf_model.model.visual.named_parameters():
        prime_param = dict(prime_model.model.visual.named_parameters())[name]
        assert torch.equal(param, prime_param), f"Vision weight mismatch: {name}"

    # Verify model produces output after weight loading
    input_ids = torch.randint(0, 200, (1, 20), device="cuda")
    position_ids = torch.arange(1, 21, device="cuda").unsqueeze(0)
    out = prime_model(input_ids=input_ids, position_ids=position_ids)
    assert out["logits"].shape[2] == config.text_config.vocab_size
    assert not torch.isnan(out["logits"]).any()


def test_vlm_weight_roundtrip():
    """HF -> PrimeRL -> HF weight conversion is lossless (vision keys untouched, text keys converted)."""
    config = _tiny_vlm_config()
    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = HFQwen3_5MoeVLM._from_config(config)

    hf_sd = hf_model.state_dict()
    original_vision_key = "model.visual.blocks.0.mlp.linear_fc1.weight"
    original_vision_weight = hf_sd[original_vision_key].clone()

    # HF -> PrimeRL
    prime_sd = dict(hf_sd)
    Qwen3_5MoeForCausalLM.convert_to_prime(prime_sd)
    assert any("language_model" in k and "mlp.experts.w1" in k for k in prime_sd)
    assert original_vision_key in prime_sd

    # PrimeRL -> HF
    roundtripped = dict(prime_sd)
    Qwen3_5MoeForCausalLM.convert_to_hf(roundtripped)

    # Original HF also needs roundtrip for expert format normalization
    orig_rt = dict(hf_sd)
    Qwen3_5MoeForCausalLM.convert_to_prime(orig_rt)
    Qwen3_5MoeForCausalLM.convert_to_hf(orig_rt)

    for key in orig_rt:
        assert key in roundtripped, f"Missing key: {key}"
        assert torch.equal(orig_rt[key], roundtripped[key]), f"Mismatch at {key}"

    # Vision weights preserved through the whole roundtrip
    assert torch.equal(roundtripped[original_vision_key], original_vision_weight)


def test_vlm_router_replay():
    """routed_experts bypasses router computation in VLM multimodal forward."""
    config = _tiny_vlm_config()
    with torch.device("cuda"), default_dtype(torch.float32):
        model = Qwen3_5MoeForCausalLM(config)
    inject_prime_lm_head(model)

    vocab = config.text_config.vocab_size
    pixel_values, image_grid_thw, n_img_tokens = _make_image_inputs(config)
    text_part = torch.randint(0, 200, (1, 10), device="cuda")
    img_part = torch.full((1, n_img_tokens), config.image_token_id, device="cuda")
    input_ids = torch.cat([text_part[:, :5], img_part, text_part[:, 5:]], dim=1)
    seq_len = input_ids.shape[1]

    num_layers = config.text_config.num_hidden_layers
    topk = config.text_config.num_experts_per_tok
    routed_experts = torch.randint(0, config.text_config.num_experts, (1, seq_len, num_layers, topk), device="cuda")

    out = model(
        input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw, routed_experts=routed_experts
    )
    assert out["logits"].shape == (1, seq_len, vocab)

    out["logits"].sum().backward()
    assert model.model.language_model.embed_tokens.weight.grad is not None


def test_vlm_meta_device_and_buffer_reinit():
    """Model can be created on meta device and buffers reinitialized."""
    config = _tiny_vlm_config()
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM.from_config(config)

    assert can_reinit_empty_buffers(model)

    model.to_empty(device="cuda")
    model.init_buffers_post_meta()

    lm_inv = model.model.language_model.rotary_emb.inv_freq
    vis_inv = model.model.visual.rotary_pos_emb.inv_freq
    assert lm_inv.device.type == "cuda"
    assert vis_inv.device.type == "cuda"
    assert lm_inv.abs().sum() > 0
    assert vis_inv.abs().sum() > 0
