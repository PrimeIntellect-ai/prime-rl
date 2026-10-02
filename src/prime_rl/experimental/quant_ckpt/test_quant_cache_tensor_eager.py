import torch

from prime_rl.experimental.quant_ckpt.fake_kernels import QuantLinear, _calls
from prime_rl.experimental.quant_ckpt.quant_cache_tensor import QuantCacheTensor


def test_cache_hit_avoids_recompute():
    x = QuantCacheTensor.from_tensor(torch.randn(4, 4))
    torch.ops.proto.fake_quantize(x, 2.0)
    torch.ops.proto.fake_quantize(x, 2.0)
    assert _calls["fake_quantize"] == 1


def test_different_keys_dont_collide():
    x = QuantCacheTensor.from_tensor(torch.randn(4, 4))
    torch.ops.proto.fake_quantize(x, 2.0)
    torch.ops.proto.fake_quantize(x, 3.0)
    assert _calls["fake_quantize"] == 2


def test_rewrap_preserves_cache_through_reshape_contiguous_detach():
    x = QuantCacheTensor.from_tensor(torch.randn(4, 4))

    chain_a = x.reshape(2, 8).contiguous().detach()
    torch.ops.proto.fake_quantize(chain_a, 2.0)

    chain_b = x.reshape(2, 8).contiguous().detach()
    torch.ops.proto.fake_quantize(chain_b, 2.0)

    assert _calls["fake_quantize"] == 1


def test_unregistered_op_strips_wrapper():
    x = QuantCacheTensor.from_tensor(torch.randn(4, 4))
    result = x + 1
    assert not isinstance(result, QuantCacheTensor)


def test_separate_instances_dont_share_cache():
    x1 = QuantCacheTensor.from_tensor(torch.randn(4, 4))
    x2 = QuantCacheTensor.from_tensor(torch.randn(4, 4))
    torch.ops.proto.fake_quantize(x1, 2.0)
    torch.ops.proto.fake_quantize(x2, 2.0)
    assert _calls["fake_quantize"] == 2


def test_fwd_and_wgrad_quantize_dedup_independently_across_sibling_calls():
    torch.manual_seed(0)
    raw_h = torch.randn(4, 3, requires_grad=True)
    h = QuantCacheTensor.from_tensor(raw_h)
    w1 = torch.randn(3, 5, requires_grad=True)
    w3 = torch.randn(3, 5, requires_grad=True)

    out_w1 = QuantLinear.apply(h, w1)
    out_w3 = QuantLinear.apply(h, w3)
    assert _calls["fake_quantize_fwd"] == 1  # w1's and w3's forward share one cast
    assert "fake_quantize_wgrad" not in _calls  # backward hasn't run yet

    (out_w1.sum() + out_w3.sum()).backward()

    assert _calls["fake_quantize_fwd"] == 1  # unchanged: no new forward calls
    assert _calls["fake_quantize_wgrad"] == 1  # w1's and w3's wgrad share one cast

    # Correctness, not just call count: the shared cache must hand back the right value.
    ref_h_wgrad_q = (raw_h * 2.0).to(torch.float16).float()
    ref_grad_w1 = ref_h_wgrad_q.T @ torch.ones_like(out_w1)
    ref_grad_w3 = ref_h_wgrad_q.T @ torch.ones_like(out_w3)
    assert torch.equal(w1.grad, ref_grad_w1)
    assert torch.equal(w3.grad, ref_grad_w3)
