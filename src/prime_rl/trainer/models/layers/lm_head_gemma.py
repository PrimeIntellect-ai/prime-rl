from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from prime_rl.trainer.models.layers.lm_head import (
    PrimeLmOutput,
    _online_logsumexp_and_weighted_update,
    _patch_model_forward,
)
from prime_rl.utils.logger import get_logger


class GemmaFusedOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, chunk_size: int, softcap: float):
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size
        self.softcap = softcap

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
    ) -> PrimeLmOutput:
        assert labels is not None, "GemmaFusedOutputLinear requires labels for chunked logprob computation"
        assert temperature is not None, "GemmaFusedOutputLinear requires per-token temperatures"

        b, s, h = hidden_states.shape
        hidden_states = hidden_states.reshape(b * s, h).contiguous()
        labels = labels.reshape(b * s).contiguous()
        inv_t = 1.0 / temperature.reshape(b * s).contiguous()  # [N]

        logprobs, entropy = _GemmaChunkedLogProbEntropyFn.apply(
            hidden_states, self.weight, labels, inv_t, self.chunk_size, self.softcap
        )

        logprobs = logprobs.reshape(b, s)
        entropy = entropy.reshape(b, s)
        return PrimeLmOutput(logprobs=logprobs, entropy=entropy)


class GemmaVanillaOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, softcap: float):
        super().__init__(in_features, out_features, bias=False)
        self.softcap = softcap

    def forward(
        self, hidden_states: torch.Tensor, labels: torch.Tensor | None = None, temperature: Tensor | None = None
    ) -> PrimeLmOutput:
        logits = super().forward(hidden_states)
        logits = self.softcap * torch.tanh(logits / self.softcap)
        return PrimeLmOutput(logits=logits)


class _GemmaChunkedLogProbEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        labels: torch.Tensor,  # [N]
        inv_temperature: torch.Tensor,  # [N]
        chunk_size: int,
        softcap: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (per-token logprobs, per-token entropy) without materializing [N, V].

        Important: entropy is computed from the *same* per-chunk logits used for the softmax
        normalization (no extra W @ hidden matmul).
        """
        assert hidden.dim() == 2, f"expected hidden [N,H], got {tuple(hidden.shape)}"
        assert weight.dim() == 2, f"expected weight [V,H], got {tuple(weight.shape)}"
        assert labels.dim() == 1, f"expected labels [N], got {tuple(labels.shape)}"
        assert inv_temperature.dim() == 1, f"expected inv_temperature [N], got {tuple(inv_temperature.shape)}"
        assert hidden.shape[0] == labels.shape[0], "hidden/labels N mismatch"
        assert hidden.shape[1] == weight.shape[1], "hidden/weight H mismatch"
        assert hidden.shape[0] == inv_temperature.shape[0], "hidden/inv_temperature N mismatch"
        assert chunk_size > 0

        device = hidden.device
        n = hidden.shape[0]
        vocab = weight.shape[0]

        # Running stats in fp32.
        m = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)
        s = torch.zeros((n,), device=device, dtype=torch.float32)
        t = torch.zeros((n,), device=device, dtype=torch.float32)
        target_logits = torch.zeros((n,), device=device, dtype=torch.float32)

        inv_t_broadcast = inv_temperature.unsqueeze(-1)  # [N, 1]

        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            w_chunk = weight[start:end]  # [C, H]
            logits = hidden @ w_chunk.t()  # [N, C] (model dtype)
            logits_f = logits.to(torch.float32)  # [N, C] fp32

            # Apply final logit softcapping (Gemma2/3) before temperature
            logits_f = softcap * torch.tanh(logits_f / softcap)
            logits_f = logits_f * inv_t_broadcast  # [N, C] fp32

            # Shared intermediates for logZ and entropy stats.
            m, s, t = _online_logsumexp_and_weighted_update(m, s, t, logits_f)

            # Fill target logits for labels that fall in this chunk.
            mask = (labels >= start) & (labels < end)
            if torch.any(mask):
                idx = (labels[mask] - start).to(torch.long)
                target_logits[mask] = logits_f[mask, idx]

        logz = m + torch.log(s)
        logprobs = target_logits - logz
        entropy = logz - (t / s)

        # Save for backward (recompute logits per chunk for grad)
        ctx.save_for_backward(hidden, weight, labels, logz)
        ctx.inv_temperature = inv_temperature
        ctx.chunk_size = chunk_size
        ctx.softcap = softcap

        # Return fp32 for numerical stability (matching baseline behavior).
        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor | None):
        assert grad_entropy is None or torch.all(grad_entropy == 0.0), (
            "Backward through entropy is not implemented in GemmaFusedOutputLinear"
        )

        hidden, weight, labels, logz = ctx.saved_tensors
        inv_temperature: torch.Tensor = ctx.inv_temperature  # [N]
        chunk_size: int = ctx.chunk_size
        softcap: float = ctx.softcap

        n, h = hidden.shape
        vocab = weight.shape[0]

        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)

        g = grad_logprobs.to(torch.float32)  # [N] fp32 for stable scaling

        inv_t_broadcast = inv_temperature.unsqueeze(-1)  # [N, 1]

        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            w_chunk = weight[start:end]  # [C, H]

            logits = hidden @ w_chunk.t()  # [N, C] (model dtype)
            logits_f = logits.to(torch.float32)  # [N, C] fp32

            # Apply final logit softcapping (Gemma2/3) before temperature
            tanh_val = torch.tanh(logits_f / softcap)
            logits_f = softcap * tanh_val
            logits_f = logits_f * inv_t_broadcast  # [N, C] fp32

            # p = softmax(logits_f) chunk = exp(logits_f - logz)
            p = torch.exp(logits_f - logz.unsqueeze(-1))  # [N, C] fp32

            # dL/dlogits = g * (1_{label} - p)
            grad_logits = (-g).unsqueeze(-1) * p  # [N, C] fp32
            mask = (labels >= start) & (labels < end)
            if torch.any(mask):
                idx = (labels[mask] - start).to(torch.long)
                grad_logits[mask, idx] += g[mask]

            # Chain through temperature scaling
            grad_logits = grad_logits * inv_t_broadcast

            # Chain through softcapping: d/dx[c*tanh(x/c)] = 1 - tanh^2(x/c)
            grad_logits = grad_logits * (1 - tanh_val**2)

            grad_hidden.add_(grad_logits.to(hidden.dtype) @ w_chunk)
            grad_w_chunk = grad_logits.to(weight.dtype).t() @ hidden  # [C, H]
            grad_weight[start:end].add_(grad_w_chunk)

        return grad_hidden, grad_weight, None, None, None, None


def inject_gemma_lm_head(model: nn.Module, chunk_size: int | None, softcap: float) -> None:
    logger = get_logger()
    logger.info(f"Injecting Gemma LM head with chunk size {chunk_size}, softcap={softcap}")

    old_lm_head = model.lm_head
    if chunk_size is not None:
        model.lm_head = GemmaFusedOutputLinear(
            in_features=old_lm_head.in_features,
            out_features=old_lm_head.out_features,
            chunk_size=chunk_size,
            softcap=softcap,
        )
    else:
        model.lm_head = GemmaVanillaOutputLinear(
            in_features=old_lm_head.in_features,
            out_features=old_lm_head.out_features,
            softcap=softcap,
        )
    model.lm_head.weight = old_lm_head.weight
    del old_lm_head

    _patch_model_forward(model)
