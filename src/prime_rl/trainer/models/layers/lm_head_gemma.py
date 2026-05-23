from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from prime_rl.trainer.models.layers.lm_head import (
    PrimeLmOutput,
    _online_logsumexp_and_weighted_update,
    _online_logsumexp_update,
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
        top_ks: Tensor | None = None,
    ) -> PrimeLmOutput:
        assert labels is not None, "GemmaFusedOutputLinear requires labels for chunked logprob computation"
        assert temperature is not None, "GemmaFusedOutputLinear requires per-token temperatures"

        b, s, h = hidden_states.shape
        hidden_states = hidden_states.reshape(b * s, h).contiguous()
        labels = labels.reshape(b * s).contiguous()
        inv_t = 1.0 / temperature.reshape(b * s).contiguous()  # [N]
        if top_ks is None:
            top_ks = torch.full_like(labels, -1)
        else:
            top_ks = top_ks.reshape(b * s).contiguous()

        logprobs, entropy = _GemmaChunkedLogProbEntropyFn.apply(
            hidden_states, self.weight, labels, inv_t, top_ks, self.chunk_size, self.softcap
        )

        logprobs = logprobs.reshape(b, s)
        entropy = entropy.reshape(b, s)
        return PrimeLmOutput(logprobs=logprobs, entropy=entropy)


class GemmaVanillaOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, softcap: float):
        super().__init__(in_features, out_features, bias=False)
        self.softcap = softcap

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
        top_ks: Tensor | None = None,
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
        top_ks: torch.Tensor,  # [N]
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
        assert top_ks.dim() == 1, f"expected top_ks [N], got {tuple(top_ks.shape)}"
        assert hidden.shape[0] == labels.shape[0], "hidden/labels N mismatch"
        assert hidden.shape[1] == weight.shape[1], "hidden/weight H mismatch"
        assert hidden.shape[0] == inv_temperature.shape[0], "hidden/inv_temperature N mismatch"
        assert hidden.shape[0] == top_ks.shape[0], "hidden/top_ks N mismatch"
        assert chunk_size > 0

        device = hidden.device
        n = hidden.shape[0]
        vocab = weight.shape[0]
        vocab_chunk_size = min(vocab, 8192)
        logprobs = torch.empty((n,), device=device, dtype=torch.float32)
        entropy = torch.empty((n,), device=device, dtype=torch.float32)
        logz = torch.empty((n,), device=device, dtype=torch.float32)
        top_k_thresholds = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)
            top_ks_chunk = top_ks[start:end]
            token_count = end - start
            top_k_mask = (top_ks_chunk > 0) & (top_ks_chunk < vocab)
            max_top_k = int(top_ks_chunk[top_k_mask].max().item()) if torch.any(top_k_mask) else 0

            m = torch.full((token_count,), float("-inf"), device=device, dtype=torch.float32)
            s = torch.zeros((token_count,), device=device, dtype=torch.float32)
            t = torch.zeros((token_count,), device=device, dtype=torch.float32)
            target_logits = torch.zeros((token_count,), device=device, dtype=torch.float32)
            top_values = (
                torch.full((token_count, max_top_k), float("-inf"), device=device, dtype=torch.float32)
                if max_top_k > 0
                else None
            )

            for vocab_start in range(0, vocab, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                weight_chunk = weight[vocab_start:vocab_end]
                logits_chunk = hidden_chunk @ weight_chunk.t()
                scaled_logits = logits_chunk.to(torch.float32)
                scaled_logits = softcap * torch.tanh(scaled_logits / softcap)
                scaled_logits = scaled_logits * inv_t_chunk

                m, s, t = _online_logsumexp_and_weighted_update(m, s, t, scaled_logits)
                if top_values is not None:
                    top_values = torch.cat((top_values, scaled_logits), dim=-1).topk(max_top_k, dim=-1).values

                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(mask):
                    idx = (labels_chunk[mask] - vocab_start).to(torch.long)
                    target_logits[mask] = scaled_logits[mask, idx]

            logz_chunk = m + torch.log(s)
            logprob_logz_chunk = logz_chunk
            if top_values is not None:
                thresholds = torch.full((token_count,), float("-inf"), device=device, dtype=torch.float32)
                thresholds[top_k_mask] = (
                    top_values[top_k_mask].gather(-1, (top_ks_chunk[top_k_mask] - 1).unsqueeze(-1)).squeeze(-1)
                )
                top_k_thresholds[start:end] = thresholds

                topk_m = torch.full((token_count,), float("-inf"), device=device, dtype=torch.float32)
                topk_s = torch.zeros((token_count,), device=device, dtype=torch.float32)
                for vocab_start in range(0, vocab, vocab_chunk_size):
                    vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                    weight_chunk = weight[vocab_start:vocab_end]
                    logits_chunk = hidden_chunk @ weight_chunk.t()
                    scaled_logits = logits_chunk.to(torch.float32)
                    scaled_logits = softcap * torch.tanh(scaled_logits / softcap)
                    scaled_logits = scaled_logits * inv_t_chunk
                    support = scaled_logits >= thresholds.unsqueeze(-1)
                    label_mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                    if torch.any(label_mask):
                        idx = (labels_chunk[label_mask] - vocab_start).to(torch.long)
                        support[label_mask, idx] = True
                    masked_logits = scaled_logits.masked_fill(~support, float("-inf"))
                    topk_m, topk_s = _online_logsumexp_update(topk_m, topk_s, masked_logits)
                logprob_logz_chunk = topk_m + torch.log(topk_s)

            logz[start:end] = logprob_logz_chunk
            logprobs[start:end] = target_logits - logprob_logz_chunk
            entropy[start:end] = logz_chunk - (t / s)

        ctx.save_for_backward(hidden, weight, labels, inv_temperature, logz, top_k_thresholds)
        ctx.chunk_size = chunk_size
        ctx.softcap = softcap

        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor | None):
        assert grad_entropy is None or torch.all(grad_entropy == 0.0), (
            "Backward through entropy is not implemented in GemmaFusedOutputLinear"
        )

        hidden, weight, labels, inv_temperature, logz, top_k_thresholds = ctx.saved_tensors
        chunk_size: int = ctx.chunk_size
        softcap: float = ctx.softcap

        n, _ = hidden.shape
        vocab = weight.shape[0]
        vocab_chunk_size = min(vocab, 8192)

        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            grad_chunk = grad_logprobs[start:end].to(torch.float32)
            inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)
            logz_chunk = logz[start:end]
            thresholds = top_k_thresholds[start:end]

            for vocab_start in range(0, vocab, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                weight_chunk = weight[vocab_start:vocab_end]
                logits_chunk = hidden_chunk @ weight_chunk.t()
                logits_f = logits_chunk.to(torch.float32)
                tanh_val = torch.tanh(logits_f / softcap)
                scaled_logits = softcap * tanh_val
                scaled_logits = scaled_logits * inv_t_chunk
                support = scaled_logits >= thresholds.unsqueeze(-1)
                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(mask):
                    idx = (labels_chunk[mask] - vocab_start).to(torch.long)
                    support[mask, idx] = True
                probs = torch.exp(scaled_logits - logz_chunk.unsqueeze(-1)).masked_fill(~support, 0.0)

                grad_logits = (-grad_chunk).unsqueeze(-1) * probs
                if torch.any(mask):
                    grad_logits[mask, idx] += grad_chunk[mask]
                grad_logits = grad_logits * inv_t_chunk
                grad_logits = grad_logits * (1 - tanh_val**2)

                grad_hidden[start:end].add_(grad_logits.to(hidden.dtype) @ weight_chunk)
                grad_weight[vocab_start:vocab_end].add_(grad_logits.to(weight.dtype).t() @ hidden_chunk)

        return grad_hidden, grad_weight, None, None, None, None, None


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
