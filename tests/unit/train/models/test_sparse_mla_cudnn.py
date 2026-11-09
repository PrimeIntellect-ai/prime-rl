import importlib.util

import pytest
import torch

from prime_rl.trainer.models.kernels.sparse_mla_bwd import (
    cudnn_backward_arch,
    cudnn_sparse_mla_backward,
    tilelang_sparse_mla_backward,
)
from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla

# GLM-5's sparse MLA: 64 heads over a 512-wide latent plus a 64-wide rope tail, 2048 picks per query.
HEADS, D_V, D_QK, TOPK = 64, 512, 576, 2048
SM_SCALE = D_QK**-0.5
# Two bf16 backwards that accumulate in different orders. On H200, dq deviates by 1.2e-3 and dkv by up to
# 4e-4. dkv's atomic adds make it vary run to run, and one bf16 step on its largest entry can reach 2**-7.
RTOL = 1e-2

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        importlib.util.find_spec("cudnn") is None
        or not torch.cuda.is_available()
        or (
            torch.cuda.get_device_capability()[0] != 9 and torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
        ),
        reason="the cuDNN DSA backward needs the cuDNN frontend and an SM90, SM100 or SM103 GPU",
    ),
]


def _causal_indices(seq_len: int) -> torch.Tensor:
    """`(1, seq_len, 1, TOPK)` distinct picks from each query's causal prefix, padded with the sentinel row `seq_len`."""
    scores = (torch.rand(seq_len, seq_len, device="cuda") + 1).tril()
    values, picks = scores.topk(TOPK, dim=-1)
    picks = torch.where(values > 0, picks, seq_len)
    return picks.to(torch.int32).view(1, seq_len, 1, TOPK).contiguous()


def test_cudnn_backward_matches_tilelang():
    """The `cudnn_flashmla` backward GLM-5 trains with, against TileLang's on the same forward."""
    torch.manual_seed(0)
    seq_len = 4096
    with torch.device("cuda"):
        q = torch.randn(1, seq_len, HEADS, D_QK, dtype=torch.bfloat16)
        kv = torch.randn(1, seq_len + 1, 1, D_QK, dtype=torch.bfloat16)
        kv[:, -1] = 0
        grad_out = torch.randn(1, seq_len, HEADS, D_V, dtype=torch.bfloat16)
    indices = _causal_indices(seq_len)
    out, lse = sparse_mla(q, kv, indices, "tilelang", SM_SCALE, D_V)

    reference = tilelang_sparse_mla_backward(q, kv, out, grad_out, indices, lse, SM_SCALE)
    actual = cudnn_sparse_mla_backward(q, kv, out, grad_out, indices, lse, SM_SCALE, cudnn_backward_arch(q, kv))

    for label, grad, reference_grad in zip(("dq", "dkv"), actual, reference):
        assert grad.shape == reference_grad.shape and grad.dtype == reference_grad.dtype, label
        deviation = (grad.float() - reference_grad.float()).abs().max()
        scale = reference_grad.float().abs().max()
        assert scale > 0, f"{label}: the reference gradient is zero"
        assert deviation <= RTOL * scale, f"{label}: max deviation {deviation} exceeds {RTOL} * scale {scale}"
