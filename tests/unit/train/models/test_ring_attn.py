import sys
import types

import pytest
import torch

import prime_rl._compat  # noqa: F401
from prime_rl.trainer.models.layers import ring_attn


def _make_tensors() -> tuple[torch.Tensor, ...]:
    q = torch.empty((1, 1, 64), dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)
    out = torch.empty_like(q)
    lse = torch.empty((1, 1), dtype=torch.float32)
    return q, k, v, cu_seqlens, out, lse


def test_fa3_varlen_uses_legacy_causal_and_window_size_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_forward(
        q,
        k,
        v,
        k_new=None,
        v_new=None,
        qv=None,
        out=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        cu_seqlens_k_new=None,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        page_table=None,
        kv_batch_idx=None,
        leftpad_k=None,
        rotary_cos=None,
        rotary_sin=None,
        seqlens_rotary=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=True,
        scheduler_metadata=None,
        num_splits=1,
        pack_gqa=None,
        sm_margin=0,
    ):
        calls["forward"] = {"causal": causal, "window_size": window_size}
        return q, torch.empty((1, 1), dtype=torch.float32), None, None

    def fake_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        sequed_q=None,
        sequed_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        dq=None,
        dk=None,
        dv=None,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        deterministic=False,
        sm_margin=0,
    ):
        calls["backward"] = {"causal": causal, "window_size": window_size}

    monkeypatch.setitem(
        sys.modules,
        "flash_attn_interface",
        types.SimpleNamespace(_flash_attn_forward=fake_forward, _flash_attn_backward=fake_backward),
    )

    q, k, v, cu_seqlens, out, lse = _make_tensors()
    ring_attn._fa3_varlen_forward(q, k, v, cu_seqlens, cu_seqlens, 1, 1, 1.0, True, window_size=(3, 0))
    ring_attn._fa3_varlen_backward(
        out, q, k, v, out, lse, cu_seqlens, cu_seqlens, 1, 1, q, k, v, 1.0, True, window_size=(3, 0)
    )

    assert calls == {
        "forward": {"causal": True, "window_size": (3, 0)},
        "backward": {"causal": True, "window_size": (3, 0)},
    }


def test_fa3_varlen_uses_split_causal_and_window_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_forward(
        q,
        k,
        v,
        k_new=None,
        v_new=None,
        qv=None,
        out=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        cu_seqlens_k_new=None,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        page_table=None,
        kv_batch_idx=None,
        leftpad_k=None,
        rotary_cos=None,
        rotary_sin=None,
        seqlens_rotary=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=None,
        is_causal=False,
        window_size_left=-1,
        window_size_right=-1,
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=True,
        scheduler_metadata=None,
        num_splits=1,
        pack_gqa=None,
        sm_margin=0,
    ):
        calls["forward"] = {
            "is_causal": is_causal,
            "window_size_left": window_size_left,
            "window_size_right": window_size_right,
        }
        return q, torch.empty((1, 1), dtype=torch.float32), None, None

    def fake_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        sequed_q=None,
        sequed_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        dq=None,
        dk=None,
        dv=None,
        softmax_scale=None,
        is_causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        deterministic=False,
        sm_margin=0,
    ):
        calls["backward"] = {
            "is_causal": is_causal,
            "window_size_left": window_size_left,
            "window_size_right": window_size_right,
        }

    monkeypatch.setitem(
        sys.modules,
        "flash_attn_interface",
        types.SimpleNamespace(_flash_attn_forward=fake_forward, _flash_attn_backward=fake_backward),
    )

    q, k, v, cu_seqlens, out, lse = _make_tensors()
    ring_attn._fa3_varlen_forward(q, k, v, cu_seqlens, cu_seqlens, 1, 1, 1.0, True, window_size=(3, 0))
    ring_attn._fa3_varlen_backward(
        out, q, k, v, out, lse, cu_seqlens, cu_seqlens, 1, 1, q, k, v, 1.0, True, window_size=(3, 0)
    )

    assert calls == {
        "forward": {"is_causal": True, "window_size_left": 3, "window_size_right": 0},
        "backward": {"is_causal": True, "window_size_left": 3, "window_size_right": 0},
    }
