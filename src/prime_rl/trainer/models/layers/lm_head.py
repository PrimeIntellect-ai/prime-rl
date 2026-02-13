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
    # TRPL: sparse current policy data from fused forward
    current_top_k_indices: Tensor | None  # [b, s, K] int64
    current_top_k_values: Tensor | None  # [b, s, K] fp32, with grad
    old_indices_current_lp: Tensor | None  # [b, s, K_old] fp32, with grad


def cast_float_and_contiguous(output: PrimeLmOutput) -> PrimeLmOutput:
    """Convert tensors in PrimeLmOutput to float and make contiguous."""

    def _float_and_contiguous(tensor: Tensor | None) -> Tensor | None:
        return tensor.float().contiguous() if tensor is not None else None

    result = PrimeLmOutput(
        logits=_float_and_contiguous(output.get("logits")),
        logprobs=_float_and_contiguous(output.get("logprobs")),
        entropy=_float_and_contiguous(output.get("entropy")),
    )

    # Pass through TRPL fields (preserve types and gradients)
    for key in ("current_top_k_indices", "current_top_k_values", "old_indices_current_lp"):
        val = output.get(key)
        if val is not None:
            if val.is_floating_point():
                result[key] = _float_and_contiguous(val)
            else:
                result[key] = val.contiguous()

    return result


class FusedOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, chunk_size: int):
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
        trpl_top_k: int | None = None,
        old_top_indices: torch.Tensor | None = None,
    ) -> PrimeLmOutput:
        assert labels is not None, "FusedOutputLinear requires labels for chunked logprob computation"
        assert temperature is not None, "FusedOutputLinear requires per-token temperatures"

        b, s, h = hidden_states.shape
        hidden_states = hidden_states.reshape(b * s, h).contiguous()
        labels = labels.reshape(b * s).contiguous()
        inv_t = 1.0 / temperature.reshape(b * s).contiguous()  # [N]

        old_top_flat = old_top_indices.reshape(b * s, -1).contiguous() if old_top_indices is not None else None

        results = _ChunkedLogProbEntropyFn.apply(
            hidden_states, self.weight, labels, inv_t, self.chunk_size, trpl_top_k, old_top_flat
        )
        logprobs, entropy, topk_idxs, topk_lp, old_idx_lp = results

        out = PrimeLmOutput(
            logprobs=logprobs.reshape(b, s),
            entropy=entropy.reshape(b, s),
        )

        if topk_idxs is not None:
            out["current_top_k_indices"] = topk_idxs.reshape(b, s, -1)
            out["current_top_k_values"] = topk_lp.reshape(b, s, -1)
            out["old_indices_current_lp"] = old_idx_lp.reshape(b, s, -1)

        return out


class VanillaOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
        **kwargs: object,
    ) -> PrimeLmOutput:
        # VanillaOutputLinear just returns logits - temperature scaling is done externally in train.py
        return PrimeLmOutput(logits=super().forward(hidden_states))


class _ChunkedLogProbEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        labels: torch.Tensor,  # [N]
        inv_temperature: torch.Tensor,  # [N]
        chunk_size: int,
        trpl_top_k: int | None = None,
        old_top_indices: torch.Tensor | None = None,  # [N, K_old]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Returns (per-token logprobs, per-token entropy, topk_indices, topk_lp, old_idx_lp)
        without materializing [N, V].

        When trpl_top_k is None: topk_indices/topk_lp/old_idx_lp are None.
        When trpl_top_k > 0: also returns current policy top-K and log-probs at old positions.
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

        trpl_enabled = trpl_top_k is not None and trpl_top_k > 0

        # Running stats in fp32.
        m = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)
        s = torch.zeros((n,), device=device, dtype=torch.float32)
        t = torch.zeros((n,), device=device, dtype=torch.float32)
        target_logits = torch.zeros((n,), device=device, dtype=torch.float32)

        # TRPL: running top-K and old-index logit accumulators
        if trpl_enabled:
            topk_vals = torch.full((n, trpl_top_k), float("-inf"), device=device, dtype=torch.float32)
            topk_idxs = torch.zeros((n, trpl_top_k), device=device, dtype=torch.long)
            k_old = old_top_indices.shape[1]
            old_idx_logits = torch.full((n, k_old), float("-inf"), device=device, dtype=torch.float32)

        inv_t_broadcast = inv_temperature.unsqueeze(-1)  # [N, 1]

        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            w_chunk = weight[start:end]  # [C, H]
            logits = hidden @ w_chunk.t()  # [N, C] (model dtype)
            logits_f = logits.to(torch.float32) * inv_t_broadcast  # [N, C] fp32

            # Shared intermediates for logZ and entropy stats.
            m, s, t = _online_logsumexp_and_weighted_update(m, s, t, logits_f)

            # Fill target logits for labels that fall in this chunk.
            mask = (labels >= start) & (labels < end)
            if torch.any(mask):
                idx = (labels[mask] - start).to(torch.long)
                target_logits[mask] = logits_f[mask, idx]

            if trpl_enabled:
                c = end - start
                # Running top-K: merge current chunk with running top-K
                chunk_indices = torch.arange(start, end, device=device, dtype=torch.long)  # [C]
                combined_vals = torch.cat([topk_vals, logits_f], dim=1)  # [N, K+C]
                combined_idxs = torch.cat([topk_idxs, chunk_indices.unsqueeze(0).expand(n, -1)], dim=1)  # [N, K+C]
                _, sel = combined_vals.topk(trpl_top_k, dim=1, largest=True, sorted=False)  # [N, K]
                topk_vals = combined_vals.gather(1, sel)  # [N, K]
                topk_idxs = combined_idxs.gather(1, sel)  # [N, K]

                # Gather temperature-scaled logits at old_top_indices that fall in this chunk
                old_in_chunk = (old_top_indices >= start) & (old_top_indices < end)  # [N, K_old]
                if old_in_chunk.any():
                    local_idx = (old_top_indices - start).clamp(0, c - 1)  # [N, K_old]
                    gathered = logits_f.gather(1, local_idx)  # [N, K_old]
                    old_idx_logits = torch.where(old_in_chunk, gathered, old_idx_logits)

        logz = m + torch.log(s)
        logprobs = target_logits - logz
        entropy = logz - (t / s)

        if trpl_enabled:
            topk_lp = topk_vals - logz.unsqueeze(1)  # [N, K]
            old_idx_lp = old_idx_logits - logz.unsqueeze(1)  # [N, K_old]
        else:
            topk_idxs = None
            topk_lp = None
            old_idx_lp = None

        # Save for backward (recompute logits per chunk for grad)
        ctx.save_for_backward(hidden, weight, labels, logz)
        ctx.inv_temperature = inv_temperature  # float or Tensor[N]
        ctx.chunk_size = chunk_size
        ctx.topk_indices = topk_idxs
        ctx.old_top_indices = old_top_indices

        # Return fp32 for numerical stability (matching baseline behavior).
        return logprobs, entropy, topk_idxs, topk_lp, old_idx_lp

    @staticmethod
    def backward(
        ctx,
        grad_logprobs: torch.Tensor,
        grad_entropy: torch.Tensor | None,
        grad_topk_indices,
        grad_topk_lp: torch.Tensor | None,
        grad_old_idx_lp: torch.Tensor | None,
    ):
        assert grad_entropy is None or torch.all(grad_entropy == 0.0), (
            "Backward through entropy is not implemented in FusedOutputLinear"
        )

        hidden, weight, labels, logz = ctx.saved_tensors
        inv_temperature: torch.Tensor = ctx.inv_temperature  # [N]
        chunk_size: int = ctx.chunk_size
        topk_indices = ctx.topk_indices  # [N, K] or None
        old_top_indices = ctx.old_top_indices  # [N, K_old] or None

        n, h = hidden.shape
        vocab = weight.shape[0]

        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)

        g = grad_logprobs.to(torch.float32)  # [N] fp32 for stable scaling

        # Total gradient flowing through logZ normalization.
        # logprobs = target - logZ => d(loss)/d(logZ) from logprobs = -g
        # topk_lp = topk_val - logZ => d(loss)/d(logZ) from topk_lp = -sum(grad_topk_lp)
        # old_idx_lp = old_val - logZ => d(loss)/d(logZ) from old_idx_lp = -sum(grad_old_idx_lp)
        # Net softmax contribution: -(g + sum(grad_topk_lp) + sum(grad_old_idx_lp)) * p
        G = g.clone()
        if grad_topk_lp is not None:
            G = G + grad_topk_lp.to(torch.float32).sum(dim=1)
        if grad_old_idx_lp is not None:
            G = G + grad_old_idx_lp.to(torch.float32).sum(dim=1)

        inv_t_broadcast = inv_temperature.unsqueeze(-1)  # [N, 1]

        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            w_chunk = weight[start:end]  # [C, H]

            logits = hidden @ w_chunk.t()  # [N, C] (model dtype)
            logits_f = logits.to(torch.float32) * inv_t_broadcast  # [N, C] fp32

            # p = softmax(logits_f) chunk = exp(logits_f - logz)
            p = torch.exp(logits_f - logz.unsqueeze(-1))  # [N, C] fp32

            # Softmax normalization gradient (from ALL outputs that go through logZ)
            grad_logits = (-G).unsqueeze(-1) * p  # [N, C] fp32

            # Delta at label positions (from logprobs = target_logit - logZ)
            mask = (labels >= start) & (labels < end)
            if torch.any(mask):
                idx = (labels[mask] - start).to(torch.long)
                grad_logits[mask, idx] += g[mask]

            # Delta at top-K positions (from topk_lp = topk_val - logZ)
            if grad_topk_lp is not None and topk_indices is not None:
                g_topk = grad_topk_lp.to(torch.float32)  # [N, K]
                in_chunk = (topk_indices >= start) & (topk_indices < end)  # [N, K]
                if in_chunk.any():
                    local_idx = (topk_indices - start).clamp(0, end - start - 1)  # [N, K]
                    masked_g = torch.where(in_chunk, g_topk, torch.zeros_like(g_topk))  # [N, K]
                    grad_logits.scatter_add_(1, local_idx, masked_g)

            # Delta at old_top_indices positions (from old_idx_lp = old_val - logZ)
            if grad_old_idx_lp is not None and old_top_indices is not None:
                g_old = grad_old_idx_lp.to(torch.float32)  # [N, K_old]
                in_chunk = (old_top_indices >= start) & (old_top_indices < end)  # [N, K_old]
                if in_chunk.any():
                    local_idx = (old_top_indices - start).clamp(0, end - start - 1)  # [N, K_old]
                    masked_g = torch.where(in_chunk, g_old, torch.zeros_like(g_old))  # [N, K_old]
                    grad_logits.scatter_add_(1, local_idx, masked_g)

            # Chain through temperature scaling: logits_f = logits * inv_temperature
            grad_logits = grad_logits * inv_t_broadcast

            grad_hidden.add_(grad_logits.to(hidden.dtype) @ w_chunk)
            grad_w_chunk = grad_logits.to(weight.dtype).t() @ hidden  # [C, H]
            grad_weight[start:end].add_(grad_w_chunk)

        # 7 inputs: hidden, weight, labels, inv_temperature, chunk_size, trpl_top_k, old_top_indices
        return grad_hidden, grad_weight, None, None, None, None, None


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
        chunk_size: If int, use FusedOutputLinear with chunked logprob/entropy computation with the given chunk size.
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

    # Check for Gemma-style softcapping - dispatch to specialized implementation
    final_logit_softcapping = getattr(model.config, "final_logit_softcapping", None)
    if final_logit_softcapping:
        from prime_rl.trainer.models.layers.lm_head_gemma import inject_gemma_lm_head

        inject_gemma_lm_head(model, chunk_size, final_logit_softcapping)
        return

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

    _patch_model_forward(model)


def _patch_model_forward(model: nn.Module) -> None:
    # Patch the forward method to use the new lm_head with labels and temperature
    def new_forward(
        self: nn.Module,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        logits_to_keep: int = 0,
        temperature: torch.Tensor | None = None,
        trpl_top_k: int | None = None,
        old_top_indices: torch.Tensor | None = None,
        **kwargs: object,
    ) -> PrimeLmOutput:
        # For VLM with images, don't create position_ids - let model compute MRoPE internally
        is_multimodal = kwargs.get("pixel_values") is not None
        if position_ids is None and not is_multimodal:
            reference_tensor = input_ids if input_ids is not None else inputs_embeds
            position_ids = torch.arange(1, reference_tensor.shape[1] + 1, device=reference_tensor.device).unsqueeze(0)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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
            temperature=temperature[:, slice_indices] if temperature is not None else None,
            trpl_top_k=trpl_top_k,
            old_top_indices=old_top_indices[:, slice_indices, :] if old_top_indices is not None else None,
        )

    # Bind the new forward to the model
    model.forward = types.MethodType(new_forward, model)
