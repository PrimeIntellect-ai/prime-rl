from __future__ import annotations

import types
from typing import TypedDict

import torch
import torch.nn as nn
from torch import Tensor

from prime_rl.utils.logger import get_logger

FUSED_CE_IGNORE_INDEX = -100


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
    ) -> PrimeLmOutput:
        assert labels is not None, "FusedOutputLinear requires labels for chunked logprob computation"
        assert temperature is not None, "FusedOutputLinear requires per-token temperatures"

        b, s, h = hidden_states.shape
        hidden_states = hidden_states.reshape(b * s, h).contiguous()
        labels = labels.reshape(b * s).contiguous()
        inv_t = 1.0 / temperature.reshape(b * s).contiguous()  # [N]

        logprobs, entropy = _SequenceChunkedLogProbEntropyFn.apply(
            hidden_states, self.weight, labels, inv_t, self.chunk_size
        )

        logprobs = logprobs.reshape(b, s)
        entropy = entropy.reshape(b, s)
        return PrimeLmOutput(logprobs=logprobs, entropy=entropy)


class VanillaOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, labels: torch.Tensor | None = None, temperature: Tensor | None = None
    ) -> PrimeLmOutput:
        # VanillaOutputLinear just returns logits - temperature scaling is done externally in train.py
        return PrimeLmOutput(logits=super().forward(hidden_states))


class FusedCrossEntropyOutputLinear(torch.nn.Linear):
    """Fused lm_head + cross-entropy loss using Liger kernel.

    Avoids materializing the full [N, V] logits tensor by fusing the linear
    projection with the cross-entropy loss computation.
    """

    IGNORE_INDEX = FUSED_CE_IGNORE_INDEX

    def __init__(self, in_features: int, out_features: int, softcap: float | None = None):
        super().__init__(in_features, out_features, bias=False)
        from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

        self.fused_ce = LigerFusedLinearCrossEntropyLoss(
            ignore_index=self.IGNORE_INDEX, reduction="mean", softcap=softcap
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
    ) -> PrimeLmOutput:
        if labels is None:
            return PrimeLmOutput(logits=super().forward(hidden_states))

        b, s, h = hidden_states.shape
        hidden_flat = hidden_states.reshape(b * s, h).contiguous()
        labels_flat = labels.reshape(b * s).contiguous()
        loss = self.fused_ce(self.weight, hidden_flat, labels_flat)
        return PrimeLmOutput(loss=loss)


class QuackFusedCrossEntropyOutputLinear(torch.nn.Linear):
    """Fused lm_head + cross-entropy loss using quack-kernels.

    Chunks the linear projection and cross-entropy computation to avoid
    materializing the full [N, V] logits tensor, using quack's optimized
    CuTe DSL kernels for CE and GEMM.
    """

    IGNORE_INDEX = FUSED_CE_IGNORE_INDEX

    def __init__(self, in_features: int, out_features: int, chunk_size: int = 4096):
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
    ) -> PrimeLmOutput:
        if labels is None:
            return PrimeLmOutput(logits=super().forward(hidden_states))

        from quack.linear_cross_entropy import chunked_linear_cross_entropy

        b, s, h = hidden_states.shape
        hidden_flat = hidden_states.reshape(b * s, h).contiguous()
        labels_flat = labels.reshape(b * s).contiguous()
        loss = chunked_linear_cross_entropy(
            hidden_flat,
            self.weight,
            labels_flat,
            chunk_size=self.chunk_size,
            ignore_index=self.IGNORE_INDEX,
            reduction="mean",
        )
        return PrimeLmOutput(loss=loss)


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


class _SequenceChunkedLogProbEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        labels: torch.Tensor,  # [N]
        inv_temperature: torch.Tensor,  # [N]
        chunk_size: int,
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
        vocab = weight.shape[0]
        vocab_chunk_size = min(vocab, 8192)
        logprobs = torch.empty((n,), device=device, dtype=torch.float32)
        entropy = torch.empty((n,), device=device, dtype=torch.float32)
        logz = torch.empty((n,), device=device, dtype=torch.float32)

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

            for vocab_start in range(0, vocab, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                weight_chunk = weight[vocab_start:vocab_end]
                logits_chunk = hidden_chunk @ weight_chunk.t()
                scaled_logits = logits_chunk.to(torch.float32) * inv_t_chunk

                m, s, t = _online_logsumexp_and_weighted_update(m, s, t, scaled_logits)

                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(mask):
                    idx = (labels_chunk[mask] - vocab_start).to(torch.long)
                    target_logits[mask] = scaled_logits[mask, idx]

            logz_chunk = m + torch.log(s)
            logz[start:end] = logz_chunk
            logprobs[start:end] = target_logits - logz_chunk
            entropy[start:end] = logz_chunk - (t / s)

        ctx.save_for_backward(hidden, weight, labels, inv_temperature, logz)
        ctx.chunk_size = chunk_size

        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor | None):
        assert grad_entropy is None or torch.all(grad_entropy == 0.0), (
            "Backward through entropy is not implemented in FusedOutputLinear"
        )

        hidden, weight, labels, inv_temperature, logz = ctx.saved_tensors
        chunk_size: int = ctx.chunk_size

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

            for vocab_start in range(0, vocab, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                weight_chunk = weight[vocab_start:vocab_end]
                logits_chunk = hidden_chunk @ weight_chunk.t()
                scaled_logits = logits_chunk.to(torch.float32) * inv_t_chunk
                probs = torch.exp(scaled_logits - logz_chunk.unsqueeze(-1))

                grad_logits = (-grad_chunk).unsqueeze(-1) * probs
                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(mask):
                    idx = (labels_chunk[mask] - vocab_start).to(torch.long)
                    grad_logits[mask, idx] += grad_chunk[mask]
                grad_logits = grad_logits * inv_t_chunk

                grad_hidden[start:end].add_(grad_logits.to(hidden.dtype) @ weight_chunk)
                grad_weight[vocab_start:vocab_end].add_(grad_logits.to(weight.dtype).t() @ hidden_chunk)

        return grad_hidden, grad_weight, None, None, None


def inject_prime_lm_head(
    model: nn.Module,
    chunk_size: int | None = None,
    fused_cross_entropy: bool | str = False,
) -> None:
    """
    Inject a PrimeRL LM head into a model.

    This replaces the model's lm_head and overrides the forward method to use labels
    and temperature for chunked loss computation.

    Args:
        model: The model to wrap.
        chunk_size: When set to an int, uses FusedOutputLinear with sequence-token chunked
            logprob/entropy computation (for RL).
        fused_cross_entropy: Controls fused lm_head + CE loss. Accepts:
            - False: no fusion
            - True or "liger": Liger kernel fusion
            - "quack": quack-kernels fusion (chunked linear + CE with CuTe DSL kernels)
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
        if fused_cross_entropy == "quack":
            raise ValueError(
                "quack_fused does not support Gemma logit softcapping. "
                "Use loss_impl='liger_fused' or loss_impl='torch' instead."
            )
        if not fused_cross_entropy:
            from prime_rl.trainer.models.layers.lm_head_gemma import inject_gemma_lm_head

            inject_gemma_lm_head(model, chunk_size, final_logit_softcapping)
            return

    # Replace the lm_head with the appropriate wrapper
    old_lm_head = model.lm_head
    if fused_cross_entropy == "quack":
        logger.info("Injecting fused cross-entropy LM head (quack-kernels)")
        model.lm_head = QuackFusedCrossEntropyOutputLinear(
            in_features=old_lm_head.in_features,
            out_features=old_lm_head.out_features,
        )
    elif fused_cross_entropy:
        logger.info("Injecting fused cross-entropy LM head (Liger kernel)")
        model.lm_head = FusedCrossEntropyOutputLinear(
            in_features=old_lm_head.in_features,
            out_features=old_lm_head.out_features,
            softcap=final_logit_softcapping,
        )
    elif isinstance(chunk_size, int):
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
        )

    # Bind the new forward to the model
    model.forward = types.MethodType(new_forward, model)
