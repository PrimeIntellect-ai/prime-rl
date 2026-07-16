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
    shape = (1, 2, 6, 512)
    query_flex = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key_flex = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value_flex = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    query_sdpa = query_flex.detach().clone().requires_grad_()
    key_sdpa = key_flex.detach().clone().requires_grad_()
    value_sdpa = value_flex.detach().clone().requires_grad_()
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]], device="cuda")
    segment_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], device="cuda")
    causal = torch.ones(6, 6, dtype=torch.bool, device="cuda").tril()
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    attention_mask = causal[None, None] & same_segment[:, None]
    output_gradient = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    output_flex, _ = gemma4_hybrid_attention_forward(
        SimpleNamespace(is_causal=True),
        query_flex,
        key_flex,
        value_flex,
        attention_mask=None,
        scaling=0.1,
        position_ids=position_ids,
    )
    output_flex = output_flex.transpose(1, 2)
    with sdpa_kernel(SDPBackend.MATH):
        output_sdpa = F.scaled_dot_product_attention(
            query_sdpa,
            key_sdpa,
            value_sdpa,
            attn_mask=attention_mask,
            scale=0.1,
        )

    output_flex.backward(output_gradient)
    output_sdpa.backward(output_gradient)

    torch.testing.assert_close(output_flex, output_sdpa, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(query_flex.grad, query_sdpa.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(key_flex.grad, key_sdpa.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(value_flex.grad, value_sdpa.grad, atol=2e-2, rtol=2e-2)
