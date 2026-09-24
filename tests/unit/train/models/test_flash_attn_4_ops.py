import pytest
import torch

pytestmark = pytest.mark.gpu

flash_attn_cute = pytest.importorskip("flash_attn.cute")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="FA4 kernels need datacenter Blackwell (SM100)",
)
def test_flash_attn_4_varlen_fullgraph_matches_fa4():
    from prime_rl.trainer.models.layers.flash_attn_4_ops import flash_attn_4_varlen

    torch.manual_seed(0)
    num_heads, num_kv_heads, head_dim = 16, 4, 256
    cu_seqlens = torch.tensor([0, 1000, 4000, 8192], dtype=torch.int32, device="cuda")
    total_tokens = int(cu_seqlens[-1])

    def make(heads):
        return torch.randn(total_tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    q, k, v = make(num_heads), make(num_kv_heads), make(num_kv_heads)
    reference_output, _ = flash_attn_cute.flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens.clone(), causal=True
    )
    grad_output = torch.randn_like(reference_output)
    reference_output.backward(grad_output)
    reference_grads = [tensor.grad.clone() for tensor in (q, k, v)]
    for tensor in (q, k, v):
        tensor.grad = None

    @torch.compile(fullgraph=True)
    def compiled(q, k, v, cu_seqlens):
        return flash_attn_4_varlen(q, k, v, cu_seqlens, cu_seqlens.clone(), causal=True)

    actual_output = compiled(q, k, v, cu_seqlens)
    actual_output.backward(grad_output)

    torch.testing.assert_close(actual_output, reference_output, atol=0, rtol=0)
    for tensor, reference_grad in zip((q, k, v), reference_grads):
        torch.testing.assert_close(tensor.grad, reference_grad, atol=0, rtol=0)
