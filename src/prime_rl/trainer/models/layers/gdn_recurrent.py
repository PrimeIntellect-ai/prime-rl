"""Recurrent GatedDeltaNet forward matching vLLM's decode kernel, with a chunked backward.

The trainer normally computes GDN with FLA's chunked kernel, while vLLM generates
tokens with a fused recurrent kernel — two different algorithms whose outputs
diverge in the low bits and compound through the recurrent state. This module
removes that mismatch on the trainer side: the forward pass calls vLLM's own
``fused_sigmoid_gating_delta_rule_update`` kernel, and the backward pass
recomputes with FLA's chunked kernel, whose gradients are well tested.

Importing vLLM's kernel (rather than vendoring a copy) is deliberate: the point
of this path is bitwise parity with the generator, and a copy only stays
bitwise identical until vLLM next touches the kernel. The import is lazy so the
trainer doesn't pay vLLM's import cost unless the path is actually used.

vLLM's kernel is inference-shaped but handles the training case natively:
``initial_state=None`` starts each sequence from a zero state and
``ssm_state_indices=None`` disables the paged-cache indirection, so a full
packed batch recomputes exactly the way training needs.
"""

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def _vllm_recurrent_update():
    from vllm.third_party.flash_linear_attention.ops import fused_sigmoid_gating_delta_rule_update

    return fused_sigmoid_gating_delta_rule_update


def gdn_recurrent_fwd(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recurrent GDN forward over full sequences from a zero initial state, via vLLM's kernel."""
    o, _ = _vllm_recurrent_update()(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=None,
        inplace_final_state=False,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=None,
        use_qk_l2norm_in_kernel=True,
    )
    return o


def gdn_gate(a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """Log-space GDN decay from raw gate inputs: ``-exp(A_log) * softplus(a + dt_bias)``."""
    return -A_log.float().exp() * F.softplus(a.float() + dt_bias)


class _GDNRecurrentFwdChunkedBwd(torch.autograd.Function):
    """Recurrent forward (bit-matching vLLM decode), chunked backward.

    FLA implements no backward for the recurrent kernel (computing ``dg``
    without materializing every intermediate state is an open problem), so the
    backward recomputes the forward with FLA's chunked kernel and
    differentiates through it. Gradients are therefore identical to the
    chunked training path; only the forward activations change.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        o = gdn_recurrent_fwd(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v, cu_seqlens=cu_seqlens)
        ctx.save_for_backward(q, k, v, a, b, A_log, dt_bias)
        ctx.cu_seqlens = cu_seqlens
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, a, b, A_log, dt_bias = (t.detach().requires_grad_() for t in ctx.saved_tensors)
        with torch.enable_grad():
            g = gdn_gate(a, A_log, dt_bias)
            beta = b.sigmoid()
            o, _ = chunk_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=ctx.cu_seqlens,
            )
        grads = torch.autograd.grad(o, (q, k, v, a, b, A_log, dt_bias), do)
        return *grads, None


@torch.compiler.disable
def gdn_recurrent_fwd_chunked_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """GDN with a recurrent forward (raw ``a``/``b`` gates fused in-kernel) and chunked backward."""
    return _GDNRecurrentFwdChunkedBwd.apply(q, k, v, a, b, A_log, dt_bias, cu_seqlens)
