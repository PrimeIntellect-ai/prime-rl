from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from prime_rl.trainer.models.layers.gemma4_hybrid_attention import gemma4_hybrid_attention_forward

pytestmark = [pytest.mark.gpu]


def test_global_attention_isolates_packed_sequences():
    batch_size, num_heads, seq_len, head_dim = 1, 1, 6, 512
    query = torch.zeros(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.zeros_like(query)
    value = torch.ones_like(query)
    value[:, :, :3] = 10
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]], device="cuda")

    output, _ = gemma4_hybrid_attention_forward(
        SimpleNamespace(is_causal=True),
        query,
        key,
        value,
        attention_mask=None,
        scaling=1.0,
        position_ids=position_ids,
    )

    expected = torch.tensor([10, 10, 10, 1, 1, 1], device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(output[0, :, 0, 0], expected)


def test_packed_global_attention_matches_sdpa_forward_and_backward():
    torch.manual_seed(0)
    batch, query_heads, key_value_heads, seq_len, head_dim = 1, 16, 2, 2048, 512
    segment_len = seq_len // 2
    query = torch.randn(batch, query_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    key_value = torch.randn(
        batch, key_value_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16
    )
    # Match layer 5's learned Q/K RMSNorm scales from the Gemma 4 26B base checkpoint.
    query = (
        query.float()
        * torch.rsqrt(query.float().square().mean(-1, keepdim=True))
        * 1.0078125
    ).bfloat16()
    key_value = (
        key_value.float()
        * torch.rsqrt(key_value.float().square().mean(-1, keepdim=True))
        * 0.062255859375
    ).bfloat16()
    query_flex = query.detach().clone().requires_grad_()
    key_value_flex = key_value.detach().clone().requires_grad_()
    query_sdpa = query_flex.detach().clone().requires_grad_()
    key_value_sdpa = key_value_flex.detach().clone().requires_grad_()
    position_ids = torch.arange(segment_len, device="cuda").repeat(batch, 2)
    segment_ids = torch.tensor(
        [[0] * segment_len + [1] * segment_len], device="cuda", dtype=torch.long
    )
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda").tril()
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    attention_mask = causal[None, None] & same_segment[:, None]
    output_gradient = torch.randn_like(query)

    output_flex, _ = gemma4_hybrid_attention_forward(
        SimpleNamespace(is_causal=True),
        query_flex,
        key_value_flex,
        key_value_flex,
        attention_mask=None,
        scaling=1.0,
        position_ids=position_ids,
    )
    output_flex = output_flex.transpose(1, 2)
    repeated_key_value = key_value_sdpa.repeat_interleave(query_heads // key_value_heads, dim=1)
    with sdpa_kernel(SDPBackend.MATH):
        output_sdpa = F.scaled_dot_product_attention(
            query_sdpa,
            repeated_key_value,
            repeated_key_value,
            attn_mask=attention_mask,
            scale=1.0,
        )

    output_flex.backward(output_gradient)
    output_sdpa.backward(output_gradient)

    torch.testing.assert_close(output_flex, output_sdpa, atol=1e-3, rtol=2e-2)
    for actual, expected in (
        (query_flex.grad, query_sdpa.grad),
        (key_value_flex.grad, key_value_sdpa.grad),
    ):
        actual_flat = actual.float().flatten()
        expected_flat = expected.float().flatten()
        assert F.cosine_similarity(actual_flat, expected_flat, dim=0) > 0.9999
        torch.testing.assert_close(actual_flat.norm(), expected_flat.norm(), rtol=5e-4, atol=0)
