from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from prime_rl.trainer.models.layers.lm_head import (
    PrimeLmOutput,
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
        Returns per-token logprobs and entropy by chunking over flattened sequence tokens.
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
        logprobs = torch.empty((n,), device=device, dtype=torch.float32)
        entropy = torch.empty((n,), device=device, dtype=torch.float32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)

            logits = hidden_chunk @ weight.t()
            scaled_logits = logits.to(torch.float32)
            scaled_logits = softcap * torch.tanh(scaled_logits / softcap)
            scaled_logits = scaled_logits * inv_t_chunk

            logprobs_chunk = scaled_logits.log_softmax(dim=-1)
            token_idx = torch.arange(end - start, device=device)
            logprobs[start:end] = logprobs_chunk[token_idx, labels_chunk]

            probs = torch.softmax(scaled_logits, dim=-1)
            entropy[start:end] = torch.logsumexp(scaled_logits, dim=-1) - torch.sum(probs * scaled_logits, dim=-1)

        ctx.save_for_backward(hidden, weight, labels, inv_temperature)
        ctx.chunk_size = chunk_size
        ctx.softcap = softcap

        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor | None):
        assert grad_entropy is None or torch.all(grad_entropy == 0.0), (
            "Backward through entropy is not implemented in GemmaFusedOutputLinear"
        )

        hidden, weight, labels, inv_temperature = ctx.saved_tensors
        chunk_size: int = ctx.chunk_size
        softcap: float = ctx.softcap

        n, _ = hidden.shape

        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            grad_chunk = grad_logprobs[start:end].to(torch.float32)
            inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)

            logits = hidden_chunk @ weight.t()
            logits_f = logits.to(torch.float32)
            tanh_val = torch.tanh(logits_f / softcap)
            scaled_logits = softcap * tanh_val
            scaled_logits = scaled_logits * inv_t_chunk
            probs = torch.softmax(scaled_logits, dim=-1)

            grad_logits = (-grad_chunk).unsqueeze(-1) * probs
            token_idx = torch.arange(end - start, device=hidden.device)
            grad_logits[token_idx, labels_chunk] += grad_chunk
            grad_logits = grad_logits * inv_t_chunk
            grad_logits = grad_logits * (1 - tanh_val**2)

            grad_hidden[start:end].add_(grad_logits.to(hidden.dtype) @ weight)
            grad_weight.add_(grad_logits.to(weight.dtype).t() @ hidden_chunk)

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
