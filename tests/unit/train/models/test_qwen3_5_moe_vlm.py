import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration as HFQwen3_5MoeVLM,
)

from prime_rl.trainer.model import can_reinit_empty_buffers
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
from prime_rl.utils.cp import setup_cp_attention_params, shard_for_cp, shard_position_ids_for_cp
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def _tiny_vlm_config(attn_implementation="sdpa"):
    """HF composite config shrunk for unit testing."""
    config = AutoConfig.from_pretrained(
        "Qwen/Qwen3.5-35B-A3B",
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    config.use_cache = False
    config._attn_implementation = attn_implementation
    tc = config.text_config
    tc._attn_implementation = attn_implementation
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
    tc.rope_parameters["mrope_section"] = [3, 3, 2]

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


def _make_mm_token_type_ids(input_ids, image_token_id):
    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[input_ids == image_token_id] = 1
    return mm_token_type_ids


def _free_tcp_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _make_packed_two_image_batch(config, *, dtype=torch.bfloat16):
    device = "cuda"
    vc = config.vision_config
    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    image_grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]], device=device)
    num_patches = int(image_grid_thw.prod(dim=1).sum().item())
    pixel_values = torch.randn(num_patches, patch_dim, device=device, dtype=dtype)

    image_tokens = torch.full((1, 4), config.image_token_id, device=device)
    segment0 = torch.cat(
        [torch.tensor([[10, 11]], device=device), image_tokens, torch.tensor([[12, 13]], device=device)], dim=1
    )
    segment1 = torch.cat(
        [torch.tensor([[20, 21]], device=device), image_tokens, torch.tensor([[22, 23]], device=device)], dim=1
    )
    input_ids = torch.cat([segment0, segment1], dim=1)
    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": _make_mm_token_type_ids(input_ids, config.image_token_id),
        "seq_lens": torch.tensor([segment0.shape[1], segment1.shape[1]], device=device),
        "temperatures": torch.ones_like(input_ids, dtype=torch.float32),
    }


def _qwen35_vlm_ulysses_equivalence_worker(rank: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        torch.cuda.set_device(0)
        pytest.importorskip("flash_attn")
        from prime_rl.trainer.models.layers.ulysses_attn import substitute_ulysses_attn

        config = _tiny_vlm_config(attn_implementation="flash_attention_2")
        config.text_config.layer_types = ["full_attention"] * config.text_config.num_hidden_layers

        torch.manual_seed(2026)
        batch = _make_packed_two_image_batch(config)
        input_ids = batch["input_ids"]

        torch.manual_seed(1234)
        with torch.device("cuda"), default_dtype(torch.bfloat16):
            baseline = Qwen3_5MoeForCausalLM(config)
        inject_prime_lm_head(baseline)
        baseline.eval()

        with torch.no_grad():
            baseline_logits = baseline(
                input_ids=input_ids,
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                mm_token_type_ids=batch["mm_token_type_ids"],
                seq_lens=batch["seq_lens"],
            )["logits"].float()

        torch.manual_seed(1234)
        with torch.device("cuda"), default_dtype(torch.bfloat16):
            cp_model = Qwen3_5MoeForCausalLM(config)
        inject_prime_lm_head(cp_model)
        cp_model.eval()

        substitute_ulysses_attn(dist.group.WORLD, attn_impl="flash_attention_2")
        full_inputs_embeds, full_position_ids = cp_model.prepare_vlm_inputs_for_context_parallel(
            input_ids=input_ids,
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            seq_lens=batch["seq_lens"],
        )
        setup_cp_attention_params(
            full_position_ids,
            cp_group=dist.group.WORLD,
            cp_style="ulysses",
            seq_lens=batch["seq_lens"],
        )

        local_inputs_embeds = shard_for_cp(full_inputs_embeds, cp_rank=rank, cp_world_size=2)
        local_position_ids = shard_position_ids_for_cp(full_position_ids, cp_rank=rank, cp_world_size=2)

        with torch.no_grad():
            local_logits = cp_model(
                inputs_embeds=local_inputs_embeds,
                position_ids=local_position_ids,
                seq_lens=batch["seq_lens"],
                seq_lens_are_global=True,
            )["logits"].float()

        gathered = [torch.empty_like(local_logits) for _ in range(2)]
        dist.all_gather(gathered, local_logits)
        cp_logits = torch.cat(gathered, dim=1)
        torch.testing.assert_close(cp_logits, baseline_logits, rtol=5e-2, atol=5e-2)
    finally:
        dist.destroy_process_group()


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
    mm_token_type_ids = _make_mm_token_type_ids(input_ids_mm, config.image_token_id)

    out_mm = model(
        input_ids=input_ids_mm,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
    )
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
    mm_token_type_ids = _make_mm_token_type_ids(input_ids, config.image_token_id)

    out = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
    )
    out["logits"].sum().backward()

    assert model.model.language_model.embed_tokens.weight.grad is not None
    assert model.model.visual.patch_embed.proj.weight.grad is not None


def test_vlm_ulysses_cp_matches_unsharded_packed_multimodal_forward():
    pytest.importorskip("flash_attn")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Ulysses VLM CP equivalence")

    mp.start_processes(
        _qwen35_vlm_ulysses_equivalence_worker,
        args=(_free_tcp_port(),),
        nprocs=2,
        start_method="spawn",
    )


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


def test_vlm_forward_requires_mm_token_type_ids():
    """Image MRoPE needs renderer-supplied modality token types."""
    config = _tiny_vlm_config()
    with torch.device("cuda"), default_dtype(torch.float32):
        model = Qwen3_5MoeForCausalLM(config)
    inject_prime_lm_head(model)

    pixel_values, image_grid_thw, n_img_tokens = _make_image_inputs(config)
    input_ids = torch.full((1, n_img_tokens), config.image_token_id, device="cuda")

    with pytest.raises(ValueError, match="mm_token_type_ids"):
        model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)


def test_vlm_forward_rejects_2d_positions_with_images():
    """Trainer 1D/2D packed positions are not valid image MRoPE coordinates."""
    config = _tiny_vlm_config()
    with torch.device("cuda"), default_dtype(torch.float32):
        model = Qwen3_5MoeForCausalLM(config)
    inject_prime_lm_head(model)

    pixel_values, image_grid_thw, n_img_tokens = _make_image_inputs(config)
    input_ids = torch.full((1, n_img_tokens), config.image_token_id, device="cuda")
    mm_token_type_ids = _make_mm_token_type_ids(input_ids, config.image_token_id)
    position_ids = torch.arange(n_img_tokens, device="cuda").unsqueeze(0)

    with pytest.raises(ValueError, match="3D MRoPE position_ids"):
        model(
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )


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
    mm_token_type_ids = _make_mm_token_type_ids(input_ids, config.image_token_id)
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
