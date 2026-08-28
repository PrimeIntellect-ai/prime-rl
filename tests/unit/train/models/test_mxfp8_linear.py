import torch
import torch.utils.checkpoint
import pytest

from prime_rl.trainer.models.layers.mxfp8_linear import (
    MXFP8Linear,
    _cache_mxfp8_dim0_weight_across_checkpoint_recompute,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
        reason="MXFP8 requires SM100 (Blackwell) or newer",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def enable_dim0_cache():
    _cache_mxfp8_dim0_weight_across_checkpoint_recompute()


def _make_layer(seed: int, in_features: int = 128, out_features: int = 128) -> MXFP8Linear:
    torch.manual_seed(seed)
    return MXFP8Linear(in_features, out_features, bias=False, device="cuda", dtype=torch.bfloat16)


def test_cached_forward_matches_fresh_quantization():
    """A second forward on an unmodified weight (cache hit) must be bit-identical to the
    first (cache miss, freshly quantized)."""
    layer = _make_layer(seed=0)
    x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)

    out_miss = layer(x)
    out_hit = layer(x)

    assert torch.equal(out_miss, out_hit)
    cached_version, _ = layer.weight._prime_rl_mxfp8_dim0_cache
    assert cached_version == layer.weight._version


def test_cache_invalidates_after_inplace_weight_update():
    """An in-place weight mutation bumps `_version`, so the cache must requantize
    rather than silently reuse the quantization of the old weight."""
    layer = _make_layer(seed=0)
    x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)

    out_before = layer(x)
    with torch.no_grad():
        layer.weight.add_(1.0)
    out_after = layer(x)

    assert not torch.equal(out_before, out_after)

    reference = _make_layer(seed=1)  # different init, then overwritten below
    with torch.no_grad():
        reference.weight.copy_(layer.weight)
    assert torch.equal(out_after, reference(x))


def test_distinct_weights_never_share_a_cache_entry():
    """Two different weight tensors, even allocated back-to-back, must never read
    each other's cached quantization."""
    layer_a = _make_layer(seed=0)
    layer_b = _make_layer(seed=1)
    assert not torch.equal(layer_a.weight, layer_b.weight)
    x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)

    out_a = layer_a(x)
    out_b = layer_b(x)
    out_a_again = layer_a(x)

    assert torch.equal(out_a, out_a_again)
    assert not torch.equal(out_a, out_b)


def test_activation_checkpoint_recompute_matches_uncheckpointed_grads():
    """Full activation checkpointing forces a second (recompute) forward call that hits
    the dim0 cache. Output and gradients must be bit-identical to an uncheckpointed run,
    since backward recomputes its own dim1 quantization independently of this cache."""
    layer = _make_layer(seed=0)
    x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)

    x_baseline = x.clone().requires_grad_()
    layer.weight.grad = None
    out_baseline = layer(x_baseline)
    out_baseline.float().pow(2).mean().backward()
    grad_w_baseline = layer.weight.grad.clone()
    grad_x_baseline = x_baseline.grad.clone()

    x_ckpt = x.clone().requires_grad_()
    layer.weight.grad = None
    out_ckpt = torch.utils.checkpoint.checkpoint(layer, x_ckpt, use_reentrant=False)
    out_ckpt.float().pow(2).mean().backward()
    grad_w_ckpt = layer.weight.grad.clone()
    grad_x_ckpt = x_ckpt.grad.clone()

    assert torch.equal(out_baseline, out_ckpt)
    assert torch.equal(grad_w_baseline, grad_w_ckpt)
    assert torch.equal(grad_x_baseline, grad_x_ckpt)


def test_mxfp8_forward_close_to_bf16_linear():
    """Sanity check the quantized path against a plain bf16 nn.Linear: MXFP8 is lossy,
    so we check relative error stays within microscaling's expected quantization noise
    rather than exact equality."""
    torch.manual_seed(0)
    mx_layer = MXFP8Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    ref_layer = torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        ref_layer.weight.copy_(mx_layer.weight)

    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    out_mx = mx_layer(x)
    out_ref = ref_layer(x)

    rel_error = (out_mx.float() - out_ref.float()).norm() / out_ref.float().norm()
    assert rel_error < 0.1
