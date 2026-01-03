from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _online_logsumexp_update(m: Tensor, s: Tensor, chunk_logits: Tensor) -> tuple[Tensor, Tensor]:
    """
    Online logsumexp accumulator.

    Maintains:
      m: running max
      s: running sum(exp(x - m))
    """
    # chunk_logits: [N, C] float32
    chunk_m = torch.amax(chunk_logits, dim=-1)  # [N]
    m_new = torch.maximum(m, chunk_m)  # [N]
    # When m == -inf initially, exp(-inf - m_new) becomes 0, which is fine.
    exp_old = torch.exp(m - m_new)
    s_new = s * exp_old + torch.exp(chunk_logits - m_new.unsqueeze(-1)).sum(dim=-1)
    return m_new, s_new


def _online_logsumexp_and_weighted_update(
    m: Tensor, s: Tensor, t: Tensor, chunk_logits: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Online logsumexp + weighted-sum accumulator for entropy.

    Maintains:
      m: running max
      s: running sum(exp(x - m))
      t: running sum(exp(x - m) * x)
    """
    chunk_m = torch.amax(chunk_logits, dim=-1)  # [N]
    m_new = torch.maximum(m, chunk_m)  # [N]
    exp_old = torch.exp(m - m_new)

    chunk_exp = torch.exp(chunk_logits - m_new.unsqueeze(-1))  # [N, C]
    s_new = s * exp_old + chunk_exp.sum(dim=-1)
    t_new = t * exp_old + (chunk_exp * chunk_logits).sum(dim=-1)
    return m_new, s_new, t_new


class _ChunkedLogProbFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: Tensor,  # [N, H]
        weight: Tensor,  # [V, H]
        labels: Tensor,  # [N]
        inv_temperature: float,
        chunk_size: int,
    ) -> Tensor:
        """
        Returns per-token logprobs (log softmax gathered at `labels`) without materializing [N, V].

        Notes:
        - Forward is executed under autograd.Function (no graph for internals); backward recomputes per-chunk logits.
        - Output dtype is float32 for numerical stability (matches current trainer behavior which upcasts logits).
        """
        assert hidden.dim() == 2, f"expected hidden [N,H], got {tuple(hidden.shape)}"
        assert weight.dim() == 2, f"expected weight [V,H], got {tuple(weight.shape)}"
        assert labels.dim() == 1, f"expected labels [N], got {tuple(labels.shape)}"
        assert hidden.shape[0] == labels.shape[0], "hidden/labels N mismatch"
        assert hidden.shape[1] == weight.shape[1], "hidden/weight H mismatch"
        assert chunk_size > 0

        device = hidden.device
        n = hidden.shape[0]
        vocab = weight.shape[0]

        # Running logsumexp stats in fp32.
        m = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)
        s = torch.zeros((n,), device=device, dtype=torch.float32)
        target_logits = torch.empty((n,), device=device, dtype=torch.float32)

        # Iterate over vocab in chunks; only allocate [N, C] at a time.
        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            w_chunk = weight[start:end]  # [C, H]
            # logits in model dtype; reduce in fp32 (mirrors baseline behavior: bf16 logits upcast to fp32)
            logits = hidden @ w_chunk.t()  # [N, C]
            logits_f = logits.to(torch.float32).mul_(inv_temperature)  # [N, C] fp32

            # Update running logsumexp
            m, s = _online_logsumexp_update(m, s, logits_f)

            # Fill target logits for labels that fall in this chunk.
            mask = (labels >= start) & (labels < end)
            if torch.any(mask):
                idx = (labels[mask] - start).to(torch.long)
                target_logits[mask] = logits_f[mask, idx]

        logz = m + torch.log(s)
        out = target_logits - logz

        # Save for backward (recompute logits per chunk)
        ctx.save_for_backward(hidden, weight, labels, logz)
        ctx.inv_temperature = inv_temperature
        ctx.chunk_size = chunk_size
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        hidden, weight, labels, logz = ctx.saved_tensors
        inv_temperature: float = ctx.inv_temperature
        chunk_size: int = ctx.chunk_size

        # Shapes:
        # hidden: [N, H], weight: [V, H], labels: [N], logz: [N], grad_out: [N]
        n, h = hidden.shape
        vocab = weight.shape[0]

        # We'll accumulate in model dtypes to keep memory low (no full-size fp32 grad buffers).
        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)

        # Flatten for easier chunk math.
        # (Already flat, but keep naming consistent)
        hidden_flat = hidden
        labels_flat = labels
        logz_flat = logz
        g = grad_out.to(torch.float32)  # [N] fp32 for stable scaling

        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            w_chunk = weight[start:end]  # [C, H]

            logits = hidden_flat @ w_chunk.t()  # [N, C] (model dtype)
            logits_f = logits.to(torch.float32).mul_(inv_temperature)  # [N, C] fp32

            # p = softmax(logits_f) chunk = exp(logits_f - logz)
            p = torch.exp(logits_f - logz_flat.unsqueeze(-1))  # [N, C] fp32

            # dL/dlogits = g * (1_{label} - p)
            grad_logits = (-g).unsqueeze(-1) * p  # [N, C] fp32
            mask = (labels_flat >= start) & (labels_flat < end)
            if torch.any(mask):
                idx = (labels_flat[mask] - start).to(torch.long)
                grad_logits[mask, idx] += g[mask]

            # Apply chain rule through temperature scaling: logits_f = logits * inv_temperature
            grad_logits.mul_(inv_temperature)

            # grad_hidden += grad_logits @ W_chunk
            # Use bf16/bf16 matmul for performance/memory, with fp32 grad_logits cast down.
            grad_hidden.add_(grad_logits.to(hidden_flat.dtype) @ w_chunk)

            # grad_weight_chunk += grad_logits^T @ hidden
            # Accumulate chunk-wise to avoid full-size fp32 grad buffer.
            # Keep memory low: use model dtype matmul (baseline training keeps lm_head grads in bf16/bf16 too).
            grad_w_chunk = grad_logits.to(weight.dtype).t() @ hidden_flat  # [C, H] (model dtype)
            grad_weight[start:end].add_(grad_w_chunk)

        return grad_hidden, grad_weight, None, None, None


@dataclass(frozen=True)
class ChunkedLogProbsConfig:
    chunk_size: int = 8192


def chunked_selective_log_softmax(
    hidden: Tensor,  # [B, S_pred, H] or [N, H]
    weight: Tensor,  # [V, H]
    labels: Tensor,  # [B, S_pred] or [N]
    *,
    temperature: float = 1.0,
    chunk_size: int = 8192,
) -> Tensor:
    """
    Compute per-token logprobs for `labels` under softmax(hidden @ weight^T / temperature),
    without materializing full logits [*, V].

    Returns float32 logprobs with shape labels.shape.
    """
    if hidden.dim() == 3:
        b, s, h = hidden.shape
        hidden_2d = hidden.reshape(b * s, h).contiguous()
        labels_1d = labels.reshape(b * s).contiguous()
        out = _ChunkedLogProbFn.apply(hidden_2d, weight, labels_1d, 1.0 / float(temperature), int(chunk_size))
        return out.reshape(b, s)
    elif hidden.dim() == 2:
        out = _ChunkedLogProbFn.apply(
            hidden.contiguous(), weight, labels.contiguous(), 1.0 / float(temperature), int(chunk_size)
        )
        return out
    else:
        raise ValueError(f"expected hidden dim 2 or 3, got {hidden.dim()}")


@torch.no_grad()
def chunked_entropy_from_hidden(
    hidden: Tensor,  # [B, S_pred, H] or [N, H]
    weight: Tensor,  # [V, H]
    *,
    temperature: float = 1.0,
    chunk_size: int = 8192,
) -> Tensor:
    """
    Entropy of softmax(hidden @ weight^T / temperature) per position, without materializing logits.
    Output dtype is float32.
    """
    inv_temperature = 1.0 / float(temperature)
    if hidden.dim() == 3:
        b, s, h = hidden.shape
        hidden_2d = hidden.reshape(b * s, h).contiguous()
        ent = chunked_entropy_from_hidden(hidden_2d, weight, temperature=temperature, chunk_size=chunk_size)
        return ent.reshape(b, s)
    if hidden.dim() != 2:
        raise ValueError(f"expected hidden dim 2 or 3, got {hidden.dim()}")

    n = hidden.shape[0]
    vocab = weight.shape[0]
    device = hidden.device

    m = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)
    s = torch.zeros((n,), device=device, dtype=torch.float32)
    t = torch.zeros((n,), device=device, dtype=torch.float32)

    for start in range(0, vocab, int(chunk_size)):
        end = min(start + int(chunk_size), vocab)
        w_chunk = weight[start:end]
        logits = hidden @ w_chunk.t()
        logits_f = logits.to(torch.float32).mul_(inv_temperature)
        m, s, t = _online_logsumexp_and_weighted_update(m, s, t, logits_f)

    logz = m + torch.log(s)
    expected_logits = t / s
    entropy = logz - expected_logits
    return entropy


class ChunkedLogprobsLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, chunk_size: int, *, temperature: float = 1.0):
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size
        self.temperature = float(temperature)

    def forward(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return chunked_selective_log_softmax(
            hidden,
            self.weight,
            labels,
            temperature=self.temperature,
            chunk_size=self.chunk_size,
        )


class BaselineLogprobsLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, *, temperature: float = 1.0):
        super().__init__(in_features, out_features, bias=False)
        self.temperature = float(temperature)

    def forward(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # hidden: [B, S, H], labels: [B, S]
        logits = super().forward(hidden)  # [B, S, V] in model dtype (bf16/fp16)
        logits_f = logits.to(torch.float32).mul_(1.0 / float(self.temperature))
        lp = torch.log_softmax(logits_f, dim=-1)
        return lp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
