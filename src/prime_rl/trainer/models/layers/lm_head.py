from __future__ import annotations

import types
from typing import TypedDict

import torch
import torch.nn as nn
from torch import Tensor

from prime_rl.utils.logger import get_logger
from prime_rl.utils.vlm import get_final_logit_softcapping


class PrimeLmOutput(TypedDict, total=False):
    """Output from LM head - a TypedDict so pytree can find tensors for FSDP2 hooks."""

    logits: Tensor | None
    logprobs: Tensor | None
    entropy: Tensor | None
    loss: Tensor | None


def cast_float_and_contiguous(output: PrimeLmOutput) -> PrimeLmOutput:
    """Convert tensors in PrimeLmOutput to float and make contiguous."""

    def _float_and_contiguous(tensor: Tensor | None) -> Tensor | None:
        return tensor.float().contiguous() if tensor is not None else None

    return PrimeLmOutput(
        logits=_float_and_contiguous(output.get("logits")),
        logprobs=_float_and_contiguous(output.get("logprobs")),
        entropy=_float_and_contiguous(output.get("entropy")),
        loss=output.get("loss"),
    )


class FusedOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, chunk_size: int):
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
        sampling_mask: Tensor | None = None,
    ) -> PrimeLmOutput:
        assert labels is not None, "FusedOutputLinear requires labels for chunked logprob computation"
        assert temperature is not None, "FusedOutputLinear requires per-token temperatures"

        b, s, h = hidden_states.shape
        hidden_states = hidden_states.reshape(b * s, h).contiguous()
        labels = labels.reshape(b * s).contiguous()
        inv_t = 1.0 / temperature.reshape(b * s).contiguous()  # [N]
        if sampling_mask is not None:
            sampling_mask = sampling_mask.reshape(b * s, sampling_mask.shape[-1]).contiguous()

        logprobs, entropy = _SequenceChunkedLogProbEntropyFn.apply(
            hidden_states, self.weight, labels, inv_t, self.chunk_size, sampling_mask
        )

        logprobs = logprobs.reshape(b, s)
        entropy = entropy.reshape(b, s)
        return PrimeLmOutput(logprobs=logprobs, entropy=entropy)


class VanillaOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
        sampling_mask: Tensor | None = None,
    ) -> PrimeLmOutput:
        # VanillaOutputLinear just returns logits. train.py applies temperature
        # scaling and sampling-mask replay.
        return PrimeLmOutput(logits=super().forward(hidden_states))


def _online_logsumexp_and_weighted_update(
    m: torch.Tensor, s: torch.Tensor, t: torch.Tensor, chunk_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_m = torch.amax(chunk_logits, dim=-1)
    m_new = torch.maximum(m, chunk_m)
    exp_old = torch.exp(m - m_new)

    chunk_exp = torch.exp(chunk_logits - m_new.unsqueeze(-1))
    s_new = s * exp_old + chunk_exp.sum(dim=-1)
    t_new = t * exp_old + (chunk_exp * chunk_logits).sum(dim=-1)
    return m_new, s_new, t_new


def sampling_replay_mask(sampling_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Positions replayable against their sampling mask: non-empty (-1 is padding)
    and containing the label (a label outside its mask means misaligned data —
    fall back to full-vocab rather than emit a corrupt logprob)."""
    return (sampling_mask >= 0).any(dim=-1) & (sampling_mask == labels.unsqueeze(-1)).any(dim=-1)


def _sampling_mask_local_indices(
    mask_chunk: torch.Tensor, vocab_start: int, vocab_end: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampling-mask ids mapped into a vocab chunk as clamped local indices
    plus the in-chunk validity mask (-1 entries are padding)."""
    in_range = (mask_chunk >= vocab_start) & (mask_chunk < vocab_end)
    local = (mask_chunk - vocab_start).clamp(0, vocab_end - vocab_start - 1)
    return local, in_range


class _SequenceChunkedLogProbEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        labels: torch.Tensor,  # [N]
        inv_temperature: torch.Tensor,  # [N]
        chunk_size: int,
        sampling_mask: torch.Tensor | None = None,  # [N, K] int32, -1-padded
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns per-token logprobs and entropy by chunking over flattened sequence tokens.

        Positions with a usable ``sampling_mask`` (see ``sampling_replay_mask``) are
        renormalized over that mask: ``logprob = scaled_logits[label] -
        logsumexp(scaled_logits[mask])``. Other positions — and entropy, a
        full-distribution diagnostic — keep full-vocab normalization.
        """
        assert hidden.dim() == 2, f"expected hidden [N,H], got {tuple(hidden.shape)}"
        assert weight.dim() == 2, f"expected weight [V,H], got {tuple(weight.shape)}"
        assert labels.dim() == 1, f"expected labels [N], got {tuple(labels.shape)}"
        assert inv_temperature.dim() == 1, f"expected inv_temperature [N], got {tuple(inv_temperature.shape)}"
        assert hidden.shape[0] == labels.shape[0], "hidden/labels N mismatch"
        assert hidden.shape[1] == weight.shape[1], "hidden/weight H mismatch"
        assert hidden.shape[0] == inv_temperature.shape[0], "hidden/inv_temperature N mismatch"
        assert chunk_size > 0
        if sampling_mask is not None:
            assert sampling_mask.dim() == 2 and sampling_mask.shape[0] == hidden.shape[0], (
                f"expected sampling_mask [N,K], got {tuple(sampling_mask.shape)}"
            )

        device = hidden.device
        n = hidden.shape[0]
        vocab = weight.shape[0]
        vocab_chunk_size = min(vocab, 8192)
        logprobs = torch.empty((n,), device=device, dtype=torch.float32)
        entropy = torch.empty((n,), device=device, dtype=torch.float32)
        logz = torch.empty((n,), device=device, dtype=torch.float32)
        replay = torch.zeros((n,), device=device, dtype=torch.bool) if sampling_mask is not None else None

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)
            token_count = end - start

            m = torch.full((token_count,), float("-inf"), device=device, dtype=torch.float32)
            s = torch.zeros((token_count,), device=device, dtype=torch.float32)
            t = torch.zeros((token_count,), device=device, dtype=torch.float32)
            target_logits = torch.zeros((token_count,), device=device, dtype=torch.float32)

            mask_chunk = sampling_mask[start:end].to(torch.long) if sampling_mask is not None else None
            if mask_chunk is not None:
                replay_chunk = sampling_replay_mask(mask_chunk, labels_chunk)
                # Each mask id lives in exactly one vocab chunk; collect its logit
                # into a [tokens, K] buffer and logsumexp once after the loop.
                mask_logits = torch.full_like(mask_chunk, float("-inf"), dtype=torch.float32)

            for vocab_start in range(0, vocab, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                weight_chunk = weight[vocab_start:vocab_end]
                logits_chunk = hidden_chunk @ weight_chunk.t()
                scaled_logits = logits_chunk.to(torch.float32) * inv_t_chunk

                m, s, t = _online_logsumexp_and_weighted_update(m, s, t, scaled_logits)

                if mask_chunk is not None:
                    mask_local, mask_in_range = _sampling_mask_local_indices(mask_chunk, vocab_start, vocab_end)
                    mask_logits = torch.where(mask_in_range, scaled_logits.gather(1, mask_local), mask_logits)

                # Branchless target extraction - we don't want to stall the GPU here (because: if torch.any()) calls bool(Tensor) calls Tensor.items() <- sync here we don't want
                in_range = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                local_idx = (labels_chunk - vocab_start).clamp(0, vocab_end - vocab_start - 1).to(torch.int64)
                chunk_target = scaled_logits.gather(1, local_idx.unsqueeze(1)).squeeze(1)
                target_logits = torch.where(in_range, chunk_target, target_logits)

            logz_full = m + torch.log(s)
            if mask_chunk is not None:
                logz_chunk = torch.where(replay_chunk, torch.logsumexp(mask_logits, dim=-1), logz_full)
                replay[start:end] = replay_chunk
            else:
                logz_chunk = logz_full
            logz[start:end] = logz_chunk
            logprobs[start:end] = target_logits - logz_chunk
            entropy[start:end] = logz_full - (t / s)

        ctx.set_materialize_grads(
            False
        )  # Without materialized grads unused outputs get grad None instead of zeros and backward can reject them without a sync
        ctx.save_for_backward(hidden, weight, labels, inv_temperature, logz, sampling_mask, replay)
        ctx.chunk_size = chunk_size

        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor | None):
        # Grads are not materialized (see forward above) so an unused entropy output arrives becomes None, and as we don't compare values we don't have any sync
        assert grad_entropy is None, "Backward through entropy is not implemented in FusedOutputLinear"
        assert grad_logprobs is not None, "FusedOutputLinear backward requires logprobs gradients"

        hidden, weight, labels, inv_temperature, logz, sampling_mask, replay = ctx.saved_tensors
        chunk_size: int = ctx.chunk_size

        n, _ = hidden.shape
        vocab = weight.shape[0]
        vocab_chunk_size = min(vocab, 8192)

        needs_hidden, needs_weight = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_hidden = torch.zeros_like(hidden) if needs_hidden else None
        grad_weight = torch.zeros_like(weight) if needs_weight else None

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            grad_chunk = grad_logprobs[start:end].to(torch.float32)
            inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)
            logz_chunk = logz[start:end]
            mask_chunk = sampling_mask[start:end].to(torch.long) if sampling_mask is not None else None
            replay_chunk = replay[start:end] if replay is not None else None

            for vocab_start in range(0, vocab, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                weight_chunk = weight[vocab_start:vocab_end]
                logits_chunk = hidden_chunk @ weight_chunk.t()
                scaled_logits = logits_chunk.to(torch.float32) * inv_t_chunk

                if mask_chunk is not None:
                    # Replayed rows get softmax gradient only on mask ids. Set masked-out
                    # logits to -inf before exp: a masked-out logit above the sampling-mask
                    # logZ would overflow exp() to inf (and inf * 0 = NaN in the grads).
                    local, in_range = _sampling_mask_local_indices(mask_chunk, vocab_start, vocab_end)
                    mask_indicator = torch.zeros(scaled_logits.shape, dtype=torch.int8, device=scaled_logits.device)
                    mask_indicator.scatter_add_(1, local, in_range.to(torch.int8))
                    masked_out = replay_chunk.unsqueeze(-1) & (mask_indicator == 0)
                    scaled_logits.masked_fill_(masked_out, float("-inf"))
                probs = torch.exp(scaled_logits - logz_chunk.unsqueeze(-1))

                grad_logits = (-grad_chunk).unsqueeze(-1) * probs
                # Branchless grad scatter like we did in the sync-free forward
                in_range = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                local_idx = (labels_chunk - vocab_start).clamp(0, vocab_end - vocab_start - 1).to(torch.int64)
                grad_logits.scatter_add_(1, local_idx.unsqueeze(1), (grad_chunk * in_range).unsqueeze(1))
                grad_logits = grad_logits * inv_t_chunk

                if needs_hidden:
                    grad_hidden[start:end].add_(grad_logits.to(hidden.dtype) @ weight_chunk)
                if needs_weight:
                    grad_weight[vocab_start:vocab_end].add_(grad_logits.to(weight.dtype).t() @ hidden_chunk)

        return grad_hidden, grad_weight, None, None, None, None


def inject_prime_lm_head(
    model: nn.Module,
    chunk_size: int | None = None,
) -> None:
    """
    Inject a PrimeRL LM head into a model.

    This replaces the model's lm_head and overrides the forward method to use labels
    and temperature for chunked loss computation.

    Args:
        model: The model to wrap.
        chunk_size: When set to an int, uses FusedOutputLinear with sequence-token chunked
            logprob/entropy computation.
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

    # Check for Gemma-style softcapping - dispatch to specialized implementation.
    final_logit_softcapping = get_final_logit_softcapping(model.config)
    if final_logit_softcapping:
        from prime_rl.trainer.models.layers.lm_head_gemma import inject_gemma_lm_head

        inject_gemma_lm_head(model, chunk_size, final_logit_softcapping)
        return

    # Replace the lm_head with the appropriate wrapper
    old_lm_head = model.lm_head
    if isinstance(chunk_size, int):
        logger.info(f"Injecting chunked LM head with chunk size {chunk_size}")
        model.lm_head = FusedOutputLinear(
            in_features=old_lm_head.in_features, out_features=old_lm_head.out_features, chunk_size=chunk_size
        )
    else:
        logger.info("Injecting vanilla LM head")
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
        sampling_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> PrimeLmOutput:
        # For VLM with images, don't create position_ids - let model compute MRoPE internally
        is_multimodal = kwargs.get("pixel_values") is not None
        if position_ids is None and not is_multimodal:
            reference_tensor = input_ids if input_ids is not None else inputs_embeds
            position_ids = torch.arange(reference_tensor.shape[1], device=reference_tensor.device).unsqueeze(0)
        model_kwargs = {"input_ids": input_ids, "position_ids": position_ids, **kwargs}
        if inputs_embeds is not None:
            model_kwargs["inputs_embeds"] = inputs_embeds
        outputs = self.model(**model_kwargs)
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
            sampling_mask=sampling_mask[:, slice_indices] if sampling_mask is not None else None,
        )

    # Bind the new forward to the model
    model.forward = types.MethodType(new_forward, model)
