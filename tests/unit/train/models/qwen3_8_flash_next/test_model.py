import copy

import pytest
import torch

from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime, apply_prime_to_hf
from prime_rl.trainer.models.kernels.qwen3_8_hyper_connection import launch
from prime_rl.trainer.models.qwen3_8_flash_next import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextForCausalLM,
    Qwen3_8FlashNextTextConfig,
    Qwen3_8FlashNextVisionConfig,
)
from prime_rl.trainer.models.qwen3_8_flash_next.hyper_connection import HyperConnection
from prime_rl.trainer.perf import FlopCounter
from prime_rl.trainer.vlm import get_language_model, get_vision_encoder


@pytest.fixture(params=[False, True], ids=["text", "composite"])
def config(request, layer_type):
    text = Qwen3_8FlashNextTextConfig(
        vocab_size=32,
        bos_token_id=31,
        eos_token_id=31,
        hidden_size=16,
        num_hidden_layers=1,
        layer_types=[layer_type],
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        hc_count=2,
        hc_lowrank=4,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        ple_layer_ids=[1],
        ple_embed_dim=16,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=3,
    )
    if not request.param:
        return text
    vision = Qwen3_8FlashNextVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=16,
        num_heads=2,
        out_hidden_size=16,
        num_position_embeddings=16,
    )
    return Qwen3_8FlashNextConfig(text_config=text, vision_config=vision, image_token_id=30)


@pytest.fixture(params=["linear_attention", "full_attention"])
def layer_type(request):
    return request.param


def test_meta_flop_count_without_cuda(config, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = type(config).from_dict(config.to_dict(), attn_implementation="flash_attention_2")
    with torch.device("meta"):
        model = Qwen3_8FlashNextForCausalLM(config)
    assert all(parameter.is_meta for parameter in model.parameters())
    assert FlopCounter(model)([16, 32]) > 0


@pytest.mark.parametrize("shard_count", [3, 128])
def test_config_and_checkpoint_roundtrip(config, shard_count):
    getattr(config, "text_config", config).split_ngram_parts = shard_count
    restored = type(config).from_dict(config.to_dict(), attn_implementation="flash_attention_2")
    model = Qwen3_8FlashNextForCausalLM(restored)
    model.init_buffers_post_meta()
    assert len(get_language_model(model).layers) == 1
    assert model.supports_packed_multimodal_training == model.is_vlm
    assert (get_vision_encoder(model) is not None) == model.is_vlm
    if model.is_vlm:
        text_model = Qwen3_8FlashNextForCausalLM(restored.text_config)
        assert model.num_active_params() == text_model.num_active_params()
    original = {
        name: torch.randn_like(value) if value.is_floating_point() else value.clone()
        for name, value in model.state_dict().items()
    }
    converted = dict(original)
    operations = model.conversion_chain(restored)
    apply_prime_to_hf(converted, operations)
    assert model.is_hf_state_dict(converted)
    assert any(name.startswith("model.visual.") for name in converted) == model.is_vlm
    model_prefix = "model.language_model" if hasattr(restored, "text_config") else "model"
    hf_ple_prefix = f"{model_prefix}.layers.0.ple.ple_embedding"
    hf_table_keys = {f"{hf_ple_prefix}.ngram_embedding.shard_{index}.weight" for index in range(shard_count)}
    assert {key for key in converted if key.startswith(f"{hf_ple_prefix}.ngram_embedding.")} == hf_table_keys
    for buffer in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
        assert torch.equal(converted[f"{hf_ple_prefix}.{buffer}"], original[f"{model_prefix}.ngram.tables.0.{buffer}"])
    assert not any(".ngram.tables." in key for key in converted)
    apply_hf_to_prime(converted, operations)
    assert converted.keys() == original.keys()
    for name, value in original.items():
        torch.testing.assert_close(converted[name], value, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "stream_count, hidden_size, low_rank, tokens", [(4, 2560, 320, 300), (4, 1024, 128, 1000), (3, 200, 24, 777)]
)
@pytest.mark.parametrize("with_residual_injection", [True, False])
@pytest.mark.parametrize("pending_combine", [False, True])
def test_fused_hyper_connection_matches_eager(
    monkeypatch, stream_count, hidden_size, low_rank, tokens, with_residual_injection, pending_combine
):
    """Fused CUDA bf16 against eager CPU fp32, forward and backward: a mix, optionally combining a pending block output
    first, and its combine. The backward walks the tokens in several chunks, the last one partial."""
    monkeypatch.setattr(launch, "CHUNK_TOKENS", 256)
    torch.manual_seed(0)
    eager = HyperConnection(
        hidden_size=hidden_size,
        stream_count=stream_count,
        low_rank=low_rank,
        norm_eps=1e-6,
        with_residual_injection=with_residual_injection,
    )
    for parameter in eager.parameters():
        torch.nn.init.normal_(parameter, std=0.1 if parameter.dim() == 1 else parameter.shape[-1] ** -0.5)
    fused = copy.deepcopy(eager).to("cuda", torch.bfloat16)
    x = torch.randn(2, tokens, stream_count * hidden_size)
    pending = (torch.randn(2, tokens, hidden_size), 2 * torch.rand(2, tokens, stream_count))
    block = torch.randn(hidden_size)
    grad = torch.randn(2, tokens, stream_count * hidden_size if with_residual_injection else hidden_size)

    def run(module, x, pending_output, pending_injection, block, grad):
        leaves = [tensor.clone().requires_grad_() for tensor in (x, pending_output, pending_injection)]
        block_input, residual_state = module.mix(leaves[0], tuple(leaves[1:]) if pending_combine else None)
        out = module.combine(block_input * block, residual_state) if with_residual_injection else block_input
        out.backward(grad)
        grads = [leaf.grad for leaf in leaves[: 3 if pending_combine else 1]]
        return [out, *grads, *(parameter.grad for parameter in module.parameters())]

    expected = run(eager, x, *pending, block, grad)
    actual = run(fused, *(tensor.to("cuda", torch.bfloat16) for tensor in (x, *pending, block, grad)))
    for fused_value, eager_value in zip(actual, expected, strict=True):
        assert (fused_value.float().cpu() - eager_value).norm() / eager_value.norm() < 1e-2
