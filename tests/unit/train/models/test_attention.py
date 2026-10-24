import pytest
import torch
import torch.nn.functional as F

from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention

HEADS, KV_HEADS, HEAD_DIM = 8, 2, 64
SEGMENTS = [37, 91, 12]


def build_attention(is_causal: bool) -> FlashAttention:
    config = AttentionConfig(
        hidden_size=HEADS * HEAD_DIM,
        head_dim=HEAD_DIM,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        is_causal=is_causal,
        attention_bias=False,
        use_qk_norm=False,
        rms_norm_eps=1e-6,
    )
    return FlashAttention(config)


def reference_attention(q, k, v, is_causal: bool) -> torch.Tensor:
    """SDPA over each packed segment on its own, in fp32."""
    outputs, start = [], 0
    for length in SEGMENTS:
        qs, ks, vs = (t[start : start + length].transpose(0, 1).float() for t in (q, k, v))
        ks, vs = (t.repeat_interleave(HEADS // KV_HEADS, dim=0) for t in (ks, vs))
        outputs.append(F.scaled_dot_product_attention(qs, ks, vs, is_causal=is_causal).transpose(0, 1))
        start += length
    return torch.cat(outputs)


@pytest.mark.gpu
@pytest.mark.parametrize("is_causal", [True, False])
def test_flash_attention_follows_is_causal(is_causal):
    torch.manual_seed(0)
    total = sum(SEGMENTS)
    q, k, v = (
        torch.randn(total, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for heads in (HEADS, KV_HEADS, KV_HEADS)
    )
    cu_seqlens = torch.tensor([0, *torch.tensor(SEGMENTS).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    attention = build_attention(is_causal).cuda()

    out = attention._compute_attention(q, k, v, cu_seqlens, max(SEGMENTS))
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    q_ref, k_ref, v_ref = (t.detach().clone().requires_grad_() for t in (q, k, v))
    expected = reference_attention(q_ref, k_ref, v_ref, is_causal)
    expected.backward(grad_out.float())

    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=0)
    for actual, reference in zip((q, k, v), (q_ref, k_ref, v_ref)):
        torch.testing.assert_close(actual.grad.float(), reference.grad.float(), atol=5e-2, rtol=0)

    # Packed segments stay apart: changing the second segment leaves the first's output as it was.
    k_changed = k.detach().clone()
    k_changed[SEGMENTS[0] : SEGMENTS[0] + SEGMENTS[1]] += 1.0
    out_changed = attention._compute_attention(q.detach(), k_changed, v.detach(), cu_seqlens, max(SEGMENTS))
    assert torch.equal(out_changed[: SEGMENTS[0]], out[: SEGMENTS[0]].detach())


def test_non_causal_sliding_window_raises():
    attention = build_attention(is_causal=False)
    attention.sliding_window = 16
    q = k = v = torch.zeros(4, HEADS, HEAD_DIM)
    with pytest.raises(NotImplementedError, match="causal attention only"):
        attention._compute_attention(q, k, v, torch.tensor([0, 4], dtype=torch.int32), 4)
