import torch

from prime_rl.trainer.models.layers.attn import fp8_kv_fake_quant


def test_fp8_kv_fake_quant_rounds_like_the_cache_with_a_straight_through_gradient():
    x = torch.tensor([0.1234, -3.3, 500.0, -1e4, 0.0], dtype=torch.bfloat16, requires_grad=True)
    y = fp8_kv_fake_quant(x, scale=1.0)
    # E4M3 rounding, saturating at +-448 like vLLM's cache store instead of overflowing to NaN
    expected = x.detach().float().clamp(-448, 448).to(torch.float8_e4m3fn).float().to(torch.bfloat16)
    assert torch.equal(y.detach(), expected)
    assert y[2].item() == 448 and y[3].item() == -448
    # Exact also where x + (rounded - x) would round in BF16
    big = torch.tensor([3.3e4, -1.7e5], dtype=torch.bfloat16)
    assert fp8_kv_fake_quant(big, scale=1.0).tolist() == [448.0, -448.0]
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    # A scale divides before rounding and multiplies back
    scaled = fp8_kv_fake_quant(x.detach(), scale=4.0)
    assert scaled[3].item() == -448 * 4


def test_subclasses_with_their_own_forward_round_too():
    # Qwen3.5 overrides forward; the rounding lives in _compute_attention, which every FlashAttention forward calls
    from prime_rl.trainer.models.qwen3_5.attention import Qwen3_5Attention
    from prime_rl.trainer.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(hidden_size=64, head_dim=16, num_attention_heads=4, num_key_value_heads=2)
    attention = Qwen3_5Attention(config, "flash_attention_2")
    attention.kv_precision, attention.kv_scale, attention.kv_fp8_query = "fp8", 1.0, False
    seen = {}

    def kernel(q, k, v, *args, **kwargs):
        seen.update(q=q, k=k, v=v)
        return q

    attention.func = kernel
    q, k, v = (torch.randn(5, 2, 16, dtype=torch.bfloat16) for _ in range(3))
    cu_seqlens = torch.tensor([0, 5], dtype=torch.int32)
    attention._compute_attention(q, k, v, cu_seqlens, 5)
    assert torch.equal(seen["k"], fp8_kv_fake_quant(k, 1.0)) and torch.equal(seen["v"], fp8_kv_fake_quant(v, 1.0))
    assert torch.equal(seen["q"], q)
