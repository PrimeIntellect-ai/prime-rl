import torch

from prime_rl.trainer.models.layers.attn import fp8_kv_fake_quant


def test_fp8_kv_fake_quant_rounds_like_the_cache_with_a_straight_through_gradient():
    x = torch.tensor([0.1234, -3.3, 500.0, -1e4, 0.0], dtype=torch.bfloat16, requires_grad=True)
    y = fp8_kv_fake_quant(x, scale=1.0)
    # E4M3 rounding, saturating at +-448 like vLLM's cache store instead of overflowing to NaN
    expected = x.detach().float().clamp(-448, 448).to(torch.float8_e4m3fn).float().to(torch.bfloat16)
    assert torch.equal(y.detach(), expected)
    assert y[2].item() == 448 and y[3].item() == -448
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    # A scale divides before rounding and multiplies back
    scaled = fp8_kv_fake_quant(x.detach(), scale=4.0)
    assert scaled[3].item() == -448 * 4
