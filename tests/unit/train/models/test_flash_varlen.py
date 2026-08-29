"""Sink-aware varlen attention against an fp32 reference with an explicit sink column."""

import importlib.util

import pytest
import torch

from prime_rl.trainer.models.layers.flash_varlen import sink_flash_attn_varlen_func

pytestmark = [pytest.mark.gpu]

NHEADS = 4
NHEADS_K = 2
HEAD_DIM = 64
SEQLENS = (48, 80)

# Measured below: every bf16 quantity lands at 1.7e-3..3.4e-3 against fp32, which is bf16's own
# 3.9e-3 resolution. A backstop at ~3x that; the sharper check is the ratio against what
# sink-free flash attention already costs, which needs no constant.
BF16_TOL = 1e-2
# Gradients come out at 1.00x..1.03x the sink-free path's own fp32 error: the sink costs them
# nothing. The forward sits at 1.40x, which is rounding bookkeeping rather than lost precision:
# rescaling by the sink rounds the output to bf16 a second time, and shrinks it, so the same
# absolute error is divided by a smaller reference norm. One extra rounding caps that at 2x.
SINK_COST_RATIO = 2.0

_HAS_FLASH_ATTN = importlib.util.find_spec("flash_attn") is not None


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.float()
    return ((actual.float() - expected).norm() / expected.norm().clamp_min(1e-12)).item()


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    err = _rel_err(actual, expected)
    print(f"{label}: rel_err={err:.3e} tol={BF16_TOL:.1e} margin={BF16_TOL / max(err, 1e-12):.1f}x")
    assert err < BF16_TOL, f"{label}: relative error {err:.3e} exceeds {BF16_TOL:.1e}"


def _fp32_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sink: torch.Tensor | None) -> torch.Tensor:
    """Causal varlen attention in fp32, with the sink as a literal extra softmax column.

    This is the definition the kernel path is supposed to implement: append one column of logits
    equal to `sink[h]`, softmax over the widened row, then drop that column. Its value vector is
    zero, so dropping it after the softmax is the whole of its effect.
    """
    starts = (0, *torch.tensor(SEQLENS).cumsum(0).tolist())
    outs = []
    for start, stop in zip(starts, starts[1:]):
        q_d = q[start:stop].transpose(0, 1)
        repeats = q.shape[1] // k.shape[1]
        k_d = k[start:stop].transpose(0, 1).repeat_interleave(repeats, dim=0)
        v_d = v[start:stop].transpose(0, 1).repeat_interleave(repeats, dim=0)

        seqlen = stop - start
        logits = (q_d @ k_d.transpose(-1, -2)) * HEAD_DIM**-0.5
        causal = torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool).tril()
        logits = logits.masked_fill(~causal, float("-inf"))

        if sink is not None:
            sink_column = sink[:, None, None].expand(q.shape[1], seqlen, 1)
            probs = torch.cat([logits, sink_column], dim=-1).softmax(-1)[..., :-1]
        else:
            probs = logits.softmax(-1)
        outs.append((probs @ v_d).transpose(0, 1))
    return torch.cat(outs, dim=0)


@pytest.fixture
def inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    total = sum(SEQLENS)
    randn = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16)  # noqa: E731
    q, k, v = randn(total, NHEADS, HEAD_DIM), randn(total, NHEADS_K, HEAD_DIM), randn(total, NHEADS_K, HEAD_DIM)
    sink = randn(NHEADS)
    dout = randn(total, NHEADS, HEAD_DIM)
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQLENS).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    return q, k, v, sink, dout, cu_seqlens


def _backward(out: torch.Tensor, dout: torch.Tensor, leaves: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    out.backward(dout)
    return [leaf.grad for leaf in leaves]


@pytest.mark.skipif(not _HAS_FLASH_ATTN, reason="flash_attn not installed")
def test_sink_attention_matches_fp32_reference(inputs):
    """The sink path must reproduce the explicit-sink-column definition, gradients included.

    A tolerance alone would only say "close enough". The ratio against sink-free flash attention's
    own fp32 error says the sink costs no accuracy beyond what bf16 attention already costs, which
    is the claim that matters and needs no magic number.
    """
    from flash_attn import flash_attn_varlen_func

    q, k, v, sink, dout, cu_seqlens = inputs
    max_seqlen = max(SEQLENS)

    leaves = tuple(t.clone().requires_grad_() for t in (q, k, v, sink))
    out = sink_flash_attn_varlen_func(
        *leaves, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True, flash_attn_version=2
    )
    grads = _backward(out, dout, leaves)

    truth_leaves = tuple(t.float().clone().requires_grad_() for t in (q, k, v, sink))
    truth = _fp32_attention(*truth_leaves)
    truth_grads = _backward(truth, dout.float(), truth_leaves)

    # Sink-free flash attention against its own fp32 truth: the accuracy floor bf16 already costs.
    free_leaves = tuple(t.clone().requires_grad_() for t in (q, k, v))
    free_out = flash_attn_varlen_func(*free_leaves, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True)
    free_grads = _backward(free_out, dout, free_leaves)
    free_truth_leaves = tuple(t.float().clone().requires_grad_() for t in (q, k, v))
    free_truth = _fp32_attention(*free_truth_leaves, sink=None)
    free_truth_grads = _backward(free_truth, dout.float(), free_truth_leaves)

    assert _rel_err(out, free_out) > 0.1, "the sink barely moved the output; it is being ignored"

    for name, got, want, free, free_want in zip(
        ("out", "dq", "dk", "dv"),
        (out, *grads[:3]),
        (truth, *truth_grads[:3]),
        (free_out, *free_grads),
        (free_truth, *free_truth_grads),
    ):
        _assert_close(got, want, name)
        floor, err = _rel_err(free, free_want), _rel_err(got, want)
        print(f"{name}: sink-free floor={floor:.3e} sink={err:.3e} ratio={err / max(floor, 1e-12):.2f}x")
        assert err <= SINK_COST_RATIO * floor, (
            f"{name}: sink path errs {err:.3e} against fp32, over {SINK_COST_RATIO}x the "
            f"{floor:.3e} that sink-free flash attention already costs"
        )

    assert grads[3] is not None, "the sink received no gradient"
    _assert_close(grads[3], truth_grads[3], "dsink")
