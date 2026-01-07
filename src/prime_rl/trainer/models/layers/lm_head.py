from __future__ import annotations

import types
from typing import TypedDict

import torch
import torch.nn as nn
from torch import Tensor

from prime_rl.utils.logger import get_logger


class PrimeLmOutput(TypedDict, total=False):
    """Output from LM head - a TypedDict so pytree can find tensors for FSDP2 hooks."""

    logits: Tensor | None
    logprobs: Tensor | None
    entropy: Tensor | None


def cast_float_and_contiguous(output: PrimeLmOutput) -> PrimeLmOutput:
    """Convert tensors in PrimeLmOutput to float and make contiguous."""

    def _float_and_contiguous(tensor: Tensor | None) -> Tensor | None:
        return tensor.float().contiguous() if tensor is not None else None

    return PrimeLmOutput(
        logits=_float_and_contiguous(output.get("logits")),
        logprobs=_float_and_contiguous(output.get("logprobs")),
        entropy=_float_and_contiguous(output.get("entropy")),
    )


class FusedOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, chunk_size: int):
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> PrimeLmOutput:
        assert labels is not None, "FusedOutputLinear requires labels for chunked logprob computation"

        inv_t = 1.0 / float(temperature)
        b, s, h = hidden_states.shape
        hidden_states = hidden_states.reshape(b * s, h).contiguous()
        labels = labels.reshape(b * s).contiguous()

        logprobs, entropy = _ChunkedLogProbEntropyFn.apply(hidden_states, self.weight, labels, inv_t, self.chunk_size)

        logprobs = logprobs.reshape(b, s)
        entropy = entropy.reshape(b, s)
        return PrimeLmOutput(logprobs=logprobs, entropy=entropy)


class VanillaOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, labels: torch.Tensor | None = None, temperature: float = 1.0
    ) -> PrimeLmOutput:
        return PrimeLmOutput(logits=super().forward(hidden_states))


class _ChunkedLogProbEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        labels: torch.Tensor,  # [N]
        inv_temperature: float,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (per-token logprobs, per-token entropy) without materializing [N, V].

        Important: entropy is computed from the *same* per-chunk logits used for the softmax
        normalization (no extra W @ hidden matmul).
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

        # Running stats in fp32.
        m = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)
        s = torch.zeros((n,), device=device, dtype=torch.float32)
        t = torch.zeros((n,), device=device, dtype=torch.float32)
        target_logits = torch.zeros((n,), device=device, dtype=torch.float32)

        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            w_chunk = weight[start:end]  # [C, H]
            logits = hidden @ w_chunk.t()  # [N, C] (model dtype)
            logits_f = logits.to(torch.float32).mul_(inv_temperature)  # [N, C] fp32

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

        # Return fp32 for numerical stability (matching baseline behavior).
        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor | None):
        assert grad_entropy is None or torch.all(grad_entropy == 0.0), (
            "Backward through entropy is not implemented in FusedOutputLinear"
        )

        hidden, weight, labels, logz = ctx.saved_tensors
        inv_temperature: float = ctx.inv_temperature
        chunk_size: int = ctx.chunk_size

        n, h = hidden.shape
        vocab = weight.shape[0]

        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)

        g = grad_logprobs.to(torch.float32)  # [N] fp32 for stable scaling

        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            w_chunk = weight[start:end]  # [C, H]

            logits = hidden @ w_chunk.t()  # [N, C] (model dtype)
            logits_f = logits.to(torch.float32).mul_(inv_temperature)  # [N, C] fp32

            # p = softmax(logits_f) chunk = exp(logits_f - logz)
            p = torch.exp(logits_f - logz.unsqueeze(-1))  # [N, C] fp32

            # dL/dlogits = g * (1_{label} - p)
            grad_logits = (-g).unsqueeze(-1) * p  # [N, C] fp32
            mask = (labels >= start) & (labels < end)
            if torch.any(mask):
                idx = (labels[mask] - start).to(torch.long)
                grad_logits[mask, idx] += g[mask]

            # Chain through temperature scaling: logits_f = logits * inv_temperature
            grad_logits.mul_(inv_temperature)

            grad_hidden.add_(grad_logits.to(hidden.dtype) @ w_chunk)
            grad_w_chunk = grad_logits.to(weight.dtype).t() @ hidden  # [C, H]
            grad_weight[start:end].add_(grad_w_chunk)

        return grad_hidden, grad_weight, None, None, None


def _online_logsumexp_and_weighted_update(
    m: torch.Tensor, s: torch.Tensor, t: torch.Tensor, chunk_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def inject_prime_lm_head(model: nn.Module, chunk_size: int | None = None) -> None:
    """
    Inject a PrimeRL LM head (FusedOutputLinear or VanillaOutputLinear) into a model.

    This replaces the model's lm_head and overrides the forward method to use labels
    and temperature for chunked loss computation.

    Args:
        model: The model to wrap.
        chunk_size: If provided, use FusedOutputLinear with chunked logprob/entropy computation.
                    If None, use VanillaOutputLinear which just returns logits.
    """
    # Guards so we have nicer error messages when a non-standard model is used
    assert hasattr(model, "model"), f"model doesnt have backbone in model.model:\n{model}"
    assert isinstance(model.model, nn.Module), f"model.model is not a nn.Module: {type(model.model)}\n{model}"
    assert hasattr(model, "lm_head"), f"model doesnt have lm_head in model.lm_head:\n{model}"
    assert isinstance(model.lm_head, nn.Linear), f"model.lm_head is not a nn.Linear: {type(model.lm_head)}\n{model}"
    assert not hasattr(model.lm_head, "bias") or model.lm_head.bias is None, (
        f"model.lm_head.bias is not supported: {model.lm_head}\n{model}"
    )

    logger = get_logger()
    logger.info(f"Injecting Prime LM head with chunk size {chunk_size}")

    # Replace the lm_head with the appropriate wrapper
    old_lm_head = model.lm_head
    if chunk_size is not None:
        model.lm_head = FusedOutputLinear(
            in_features=old_lm_head.in_features, out_features=old_lm_head.out_features, chunk_size=chunk_size
        )
    else:
        model.lm_head = VanillaOutputLinear(in_features=old_lm_head.in_features, out_features=old_lm_head.out_features)
    model.lm_head.weight = old_lm_head.weight
    del old_lm_head

    # Patch the forward method to use the new lm_head with labels and temperature
    def new_forward(
        self: nn.Module,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.Tensor | None = None,
        logits_to_keep: int = 0,
        temperature: float = 1.0,
        **kwargs: object,
    ) -> PrimeLmOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        # Slice hidden states for logits_to_keep
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        )

        # Pass through the wrapped lm_head
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    # Bind the new forward to the model
    model.forward = types.MethodType(new_forward, model)
