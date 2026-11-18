"""CPU check of how prime-rl calls the cuDNN frontend DSA backward.

Since cuDNN frontend 1.32 the backward never allocates: `dq`, `dkv`, `d_sink` (and on SM90 the workspace) are
caller-provided, and SM100 raises when one is missing. The kernels need a Hopper/Blackwell GPU, so the interfaces are
stubbed here: each call is bound against the real signature and its output buffers are checked.
"""

import inspect

import cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90 as interface_sm90
import cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 as interface_sm100
import pytest
import torch

import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn as dsv4_sparse_attn
from prime_rl.trainer.models.kernels.sparse_mla_bwd import cudnn_sparse_mla_backward

ARCH_INTERFACES = {"sm90": (interface_sm90, "flash_attn_bwd_sm90"), "sm100": (interface_sm100, "flash_attn_bwd_sm100")}


@pytest.fixture
def stub_backward(monkeypatch):
    calls = []

    def install(arch: str):
        module, name = ARCH_INTERFACES[arch]
        signature = inspect.signature(getattr(module, name))

        def stub(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            calls.append(bound.arguments)
            return bound.arguments["dq"], bound.arguments["dkv"], bound.arguments["d_sink"]

        monkeypatch.setattr(module, name, stub)
        return calls

    return install


def check_outputs(arguments: dict) -> None:
    q, kv, attn_sink = arguments["q"], arguments["kv"], arguments["attn_sink"]
    for name, like in (("dq", q), ("dkv", kv), ("d_sink", attn_sink)):
        buffer = arguments[name]
        assert buffer is not None, f"{name} must be preallocated"
        assert buffer.shape == like.shape and buffer.dtype == like.dtype and buffer.is_contiguous(), name
        assert buffer.data_ptr() != like.data_ptr(), f"{name} aliases an input"
    assert attn_sink.dtype == torch.float32 and torch.isneginf(attn_sink).all()
    assert arguments["workspace"] is not None and arguments["workspace"].dtype == torch.uint8


def make_inputs(batch: int, seq_len: int, heads: int, dim: int, slots: int, dim_v: int):
    q = torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16)
    kv = torch.randn(batch, seq_len, 1, dim, dtype=torch.bfloat16)
    out = torch.randn(batch, seq_len, heads, dim_v, dtype=torch.bfloat16)
    grad_out = torch.randn_like(out)
    indices = torch.randint(0, seq_len, (batch, seq_len, 1, slots), dtype=torch.int32)
    lse = torch.randn(batch, seq_len, heads, dtype=torch.float32)
    return q, kv, out, grad_out, indices, lse


@pytest.mark.parametrize("arch", ["sm90", "sm100"])
def test_glm_sparse_mla_backward_preallocates_outputs(arch, stub_backward):
    calls = stub_backward(arch)
    q, kv, out, grad_out, indices, lse = make_inputs(batch=2, seq_len=16, heads=16, dim=576, slots=8, dim_v=512)
    dq, dkv = cudnn_sparse_mla_backward(q, kv, out, grad_out, indices, lse, 576**-0.5, arch)
    assert len(calls) == 1
    check_outputs(calls[0])
    assert dq.shape == q.shape and dkv.shape == kv.shape


@pytest.mark.parametrize("major", [9, 10])
def test_dsv4_cudnn_flashmla_backward_preallocates_outputs(major, stub_backward, monkeypatch):
    calls = stub_backward("sm90" if major == 9 else "sm100")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (major, 0))
    # The TileLang delta kernel needs a GPU; its output is not under test.
    monkeypatch.setattr(dsv4_sparse_attn, "preprocess", lambda heads, dim: lambda out, grad_out: None)
    q, kv, out, grad_out, indices, lse = make_inputs(batch=2, seq_len=16, heads=64, dim=512, slots=8, dim_v=512)
    dq, dkv, _ = dsv4_sparse_attn.cudnn_flashmla_backward(q, kv, out, grad_out, indices, lse, 512**-0.5, max_slots=8)
    assert len(calls) == 1
    check_outputs(calls[0])
    assert calls[0]["topk_length"] is not None
    assert dq.shape == q.shape and dkv.shape == kv.shape
