from __future__ import annotations

import types
from dataclasses import dataclass, replace
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from prime_rl.trainer.models.kernels.segment_reduce import csr_matmul, segment_logsumexp, segment_sum
from prime_rl.utils.logger import get_logger

# Same as torch's cross entropy loss
IGNORE_INDEX = -100


class PrimeLmOutput(TypedDict, total=False):
    """Output from LM head - a TypedDict so pytree can find tensors for FSDP2 hooks."""

    logits: Tensor | None
    logprobs: Tensor | None
    entropy: Tensor | None
    support_logprobs: Tensor | None
    loss: Tensor | None


@dataclass(frozen=True)
class TokenSupport:
    """Per-token sampler support as CSR rows: token ``i`` owns entries ``offsets[i]:offsets[i + 1]``. ``offsets``
    stays on the host, so slicing rows never waits on the device."""

    ids: Tensor
    offsets: Tensor
    replayed: Tensor
    logprobs: Tensor | None = None
    sparse_backward: bool = False

    @property
    def num_rows(self) -> int:
        return self.offsets.numel() - 1

    def rows(self, start: int, end: int) -> TokenSupport:
        first, last = int(self.offsets[start]), int(self.offsets[end])
        return replace(
            self,
            ids=self.ids[first:last],
            offsets=self.offsets[start : end + 1] - first,
            replayed=self.replayed[start:end],
            logprobs=self.logprobs[first:last] if self.logprobs is not None else None,
        )

    def shift_left(self) -> TokenSupport:
        """Align with labels: drop the first token's row and append an empty one."""
        shifted = self.rows(1, self.num_rows)
        return replace(
            shifted,
            offsets=torch.cat([shifted.offsets, shifted.offsets[-1:]]),
            replayed=torch.cat([shifted.replayed, shifted.replayed.new_zeros(1)]),
        )

    def to(self, device: torch.device | str) -> TokenSupport:
        return replace(
            self,
            ids=self.ids.to(device),
            replayed=self.replayed.to(device),
            logprobs=self.logprobs.to(device) if self.logprobs is not None else None,
        )

    def row_index(self) -> Tensor:
        """The token row of every entry, int32 on the device of ``ids``."""
        counts = self.offsets.diff().to(self.ids.device)
        rows = torch.arange(self.num_rows, device=self.ids.device, dtype=torch.int32)
        return torch.repeat_interleave(rows, counts, output_size=self.ids.numel())


def support_replay_mask(support: TokenSupport, labels: Tensor, rows: Tensor) -> Tensor:
    """Tokens replayable against their sampling mask: the row is a replayed mask and contains the label (a label
    outside its mask means misaligned data, so those tokens fall back to full-vocab normalization)."""
    hits = (support.ids == labels[rows]).to(torch.int32)
    return support.replayed & (hits.new_zeros(support.num_rows).index_add_(0, rows, hits) > 0)


def cast_float_and_contiguous(output: PrimeLmOutput) -> PrimeLmOutput:
    """Convert tensors in PrimeLmOutput to float and make contiguous."""

    def _float_and_contiguous(tensor: Tensor | None) -> Tensor | None:
        return tensor.float().contiguous() if tensor is not None else None

    return PrimeLmOutput(
        logits=_float_and_contiguous(output.get("logits")),
        logprobs=_float_and_contiguous(output.get("logprobs")),
        entropy=_float_and_contiguous(output.get("entropy")),
        support_logprobs=_float_and_contiguous(output.get("support_logprobs")),
        loss=output.get("loss"),
    )


class FusedOutputLinear(torch.nn.Linear):
    """Chunked LM head that never materializes the full [N, V] logits.

    With ``labels`` and no ``temperature`` it returns the summed cross-entropy over labels != IGNORE_INDEX
    as ``loss``, computing the gradients chunk by chunk in the forward pass (see ``_ChunkedCrossEntropySumFn``).
    With ``temperature`` it returns per-token ``logprobs`` and ``entropy``, and with a ``support`` (label-aligned)
    also ``support_logprobs`` per support entry (see ``_SequenceChunkedLogProbEntropyFn``). With
    ``support.sparse_backward`` the backward skips the dense softmax term: the caller guarantees each token's logit
    gradient lies on its label and support entries (replayed tokens, or a loss whose weights on ``logprobs`` and
    ``support_logprobs`` cancel per token).
    With ``softcap`` the logits are capped to ``softcap * tanh(logits / softcap)`` before anything else.
    """

    def __init__(self, in_features: int, out_features: int, chunk_size: int, softcap: float | None = None):
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size
        self.softcap = softcap

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
        support: TokenSupport | None = None,
    ) -> PrimeLmOutput:
        assert labels is not None, "FusedOutputLinear requires labels for chunked logprob computation"

        b, s, h = hidden_states.shape
        hidden_states = hidden_states.reshape(b * s, h).contiguous()
        labels = labels.reshape(b * s).contiguous()

        if temperature is None:
            assert support is None, "Sampling evidence requires per-token temperatures"
            loss = _ChunkedCrossEntropySumFn.apply(hidden_states, self.weight, labels, self.chunk_size, self.softcap)
            return PrimeLmOutput(loss=loss)

        inv_t = 1.0 / temperature.reshape(b * s).contiguous()  # [N]
        support_ids = support_rows = replay = offsets = None
        if support is not None:
            support_ids = support.ids
            support_rows = support.row_index()
            replay = support_replay_mask(support, labels, support_rows)
            offsets = support.offsets
        logprobs, entropy, support_logprobs = _SequenceChunkedLogProbEntropyFn.apply(
            hidden_states,
            self.weight,
            labels,
            inv_t,
            self.chunk_size,
            support_ids,
            support_rows,
            offsets,
            replay,
            self.softcap,
            support is not None and support.sparse_backward,
        )
        return PrimeLmOutput(
            logprobs=logprobs.reshape(b, s), entropy=entropy.reshape(b, s), support_logprobs=support_logprobs
        )


class VanillaOutputLinear(torch.nn.Linear):
    """LM head that returns the full logits, or with ``labels`` and no ``temperature`` the summed fp32
    cross-entropy over labels != IGNORE_INDEX as ``loss``. With ``softcap`` the logits are capped first."""

    def __init__(self, in_features: int, out_features: int, softcap: float | None = None):
        super().__init__(in_features, out_features, bias=False)
        self.softcap = softcap

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
        support: TokenSupport | None = None,
    ) -> PrimeLmOutput:
        logits = super().forward(hidden_states)
        if self.softcap is not None:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        if labels is not None and temperature is None:
            return PrimeLmOutput(loss=cross_entropy_sum(logits, labels))
        # train.py applies temperature scaling and support replay to the logits.
        return PrimeLmOutput(logits=logits)


def cross_entropy_sum(logits: Tensor, labels: Tensor) -> Tensor:
    """Summed fp32 cross-entropy over labels != IGNORE_INDEX."""
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1]).float(), labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum"
    )


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


def _support_flat_index(
    ids: torch.Tensor, rows: torch.Tensor, vocab_start: int, vocab_end: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Support entries as flat indices into a [tokens, vocab_end - vocab_start] logits chunk (``rows`` are
    chunk-local), plus whether each entry falls into that vocab chunk."""
    in_range = (ids >= vocab_start) & (ids < vocab_end)
    flat = rows.long() * (vocab_end - vocab_start) + (ids.long() - vocab_start)
    return torch.where(in_range, flat, 0), in_range


def _sparse_logit_backward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    inv_temperature: torch.Tensor,
    logz: torch.Tensor,
    support_ids: torch.Tensor,
    support_rows: torch.Tensor,
    offsets: torch.Tensor,
    replay: torch.Tensor,
    label_logits: torch.Tensor,
    entry_logits: torch.Tensor,
    grad_logprobs: torch.Tensor | None,
    grad_support: torch.Tensor | None,
    softcap: float | None,
    needs_grad: tuple[bool, bool],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Hidden and weight gradients from a logit gradient that lies on each token's label and support entries.

    Per token the logit gradient is ``g_label * e_label + sum_e g_e * e_id - p * net`` with ``net = g_label +
    sum_e g_e``. Replayed tokens normalize ``p`` over their support, so the last term stays on it; for the others the
    caller guarantees ``net = 0``. The gradient is one [N, V] CSR matrix (each token's label, then its entries),
    multiplied into ``weight`` and, sorted by vocab id, into ``hidden``."""
    n, device = hidden.shape[0], hidden.device
    rows = support_rows.long()
    label_grad = grad_logprobs.float() if grad_logprobs is not None else torch.zeros_like(logz)
    entry_grad = grad_support.float() if grad_support is not None else torch.zeros_like(entry_logits)
    net = label_grad + segment_sum(entry_grad, offsets.to(device))
    entry_grad = entry_grad - torch.where(replay[rows], torch.exp(entry_logits - logz[rows]) * net[rows], 0.0)

    label_grad = label_grad * inv_temperature
    entry_grad = entry_grad * inv_temperature[rows]
    if softcap is not None:
        label_grad = label_grad * (1 - (label_logits / (softcap * inv_temperature)).square())
        entry_grad = entry_grad * (1 - (entry_logits / (softcap * inv_temperature[rows])).square())

    row_offsets = offsets.to(device) + torch.arange(n + 1, device=device)
    label_pos = row_offsets[:-1]
    entry_pos = torch.arange(rows.numel(), device=device) + rows + 1
    nnz = n + rows.numel()
    cols = torch.empty(nnz, device=device, dtype=torch.int32)
    vals = torch.empty(nnz, device=device, dtype=torch.float32)
    cols[label_pos] = labels.int()
    cols[entry_pos] = support_ids
    vals[label_pos] = label_grad
    vals[entry_pos] = entry_grad

    grad_hidden = grad_weight = None
    if needs_grad[0]:
        grad_hidden = csr_matmul(row_offsets, cols, vals, weight.contiguous(), torch.empty_like(hidden))
    if needs_grad[1]:
        sorted_cols, order = torch.sort(cols, stable=True)
        vocab_offsets = torch.searchsorted(
            sorted_cols, torch.arange(weight.shape[0] + 1, device=device, dtype=torch.int32)
        )
        token = torch.repeat_interleave(
            torch.arange(n, device=device, dtype=torch.int32), row_offsets.diff(), output_size=nnz
        )
        grad_weight = csr_matmul(vocab_offsets, token[order], vals[order], hidden, torch.empty_like(weight))
    return grad_hidden, grad_weight


class _SequenceChunkedLogProbEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        labels: torch.Tensor,  # [N]
        inv_temperature: torch.Tensor,  # [N]
        chunk_size: int,
        support_ids: torch.Tensor | None = None,
        support_rows: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        replay: torch.Tensor | None = None,  # [N] bool
        softcap: float | None = None,
        sparse_backward: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns per-token logprobs and entropy by chunking over flattened sequence tokens.

        ``replay`` rows are renormalized over their support (the sampling mask):
        ``logprob = scaled_logits[label] - logsumexp(scaled_logits[support])``. Other
        rows — and entropy, a full-distribution diagnostic — keep full-vocab
        normalization. With a support it also returns the logprob of every support
        entry under the same normalization as its token's label.
        """
        assert hidden.dim() == 2, f"expected hidden [N,H], got {tuple(hidden.shape)}"
        assert weight.dim() == 2, f"expected weight [V,H], got {tuple(weight.shape)}"
        assert labels.dim() == 1, f"expected labels [N], got {tuple(labels.shape)}"
        assert inv_temperature.dim() == 1, f"expected inv_temperature [N], got {tuple(inv_temperature.shape)}"
        assert hidden.shape[0] == labels.shape[0], "hidden/labels N mismatch"
        assert hidden.shape[1] == weight.shape[1], "hidden/weight H mismatch"
        assert hidden.shape[0] == inv_temperature.shape[0], "hidden/inv_temperature N mismatch"
        assert chunk_size > 0
        if support_ids is not None:
            assert offsets.shape == (hidden.shape[0] + 1,), f"expected offsets [N+1], got {tuple(offsets.shape)}"

        device = hidden.device
        n = hidden.shape[0]
        vocab = weight.shape[0]
        vocab_chunk_size = min(vocab, 8192)
        logprobs = torch.empty((n,), device=device, dtype=torch.float32)
        entropy = torch.empty((n,), device=device, dtype=torch.float32)
        logz = torch.empty((n,), device=device, dtype=torch.float32)
        support_logprobs = (
            torch.empty(support_ids.shape, device=device, dtype=torch.float32) if support_ids is not None else None
        )
        if sparse_backward:
            label_logits = torch.empty((n,), device=device, dtype=torch.float32)
            entry_logits = torch.empty(support_ids.shape, device=device, dtype=torch.float32)

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

            if support_ids is not None:
                first, last = int(offsets[start]), int(offsets[end])
                ids_chunk, rows_chunk = support_ids[first:last], support_rows[first:last] - start
                support_logits = torch.full((last - first,), float("-inf"), device=device, dtype=torch.float32)

            for vocab_start in range(0, vocab, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                weight_chunk = weight[vocab_start:vocab_end]
                logits_chunk = hidden_chunk @ weight_chunk.t()
                scaled_logits = logits_chunk.to(torch.float32)
                if softcap is not None:
                    scaled_logits = softcap * torch.tanh(scaled_logits / softcap)
                scaled_logits = scaled_logits * inv_t_chunk

                m, s, t = _online_logsumexp_and_weighted_update(m, s, t, scaled_logits)

                if support_ids is not None:
                    flat, in_range = _support_flat_index(ids_chunk, rows_chunk, vocab_start, vocab_end)
                    support_logits = torch.where(in_range, scaled_logits.view(-1)[flat], support_logits)

                # Branchless target extraction - we don't want to stall the GPU here (because: if torch.any()) calls bool(Tensor) calls Tensor.items() <- sync here we don't want
                in_range = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                local_idx = (labels_chunk - vocab_start).clamp(0, vocab_end - vocab_start - 1).to(torch.int64)
                chunk_target = scaled_logits.gather(1, local_idx.unsqueeze(1)).squeeze(1)
                target_logits = torch.where(in_range, chunk_target, target_logits)

            logz_full = m + torch.log(s)
            logz_chunk = logz_full
            if support_ids is not None:
                support_logz = segment_logsumexp(support_logits, (offsets[start : end + 1] - first).to(device))
                logz_chunk = torch.where(replay[start:end], support_logz, logz_full)
                support_logprobs[first:last] = support_logits - logz_chunk[rows_chunk]
                if sparse_backward:
                    entry_logits[first:last] = support_logits
            if sparse_backward:
                label_logits[start:end] = target_logits
            logz[start:end] = logz_chunk
            logprobs[start:end] = target_logits - logz_chunk
            entropy[start:end] = logz_full - (t / s)

        ctx.set_materialize_grads(
            False
        )  # Without materialized grads unused outputs get grad None instead of zeros and backward can reject them without a sync
        ctx.save_for_backward(hidden, weight, labels, inv_temperature, logz, support_ids, support_rows, replay)
        if sparse_backward:
            ctx.sparse_logits = (label_logits, entry_logits)
        ctx.sparse_backward = sparse_backward
        ctx.offsets = offsets
        ctx.chunk_size = chunk_size
        ctx.softcap = softcap

        return logprobs, entropy, support_logprobs

    @staticmethod
    def backward(
        ctx, grad_logprobs: torch.Tensor | None, grad_entropy: torch.Tensor | None, grad_support: torch.Tensor | None
    ):
        # Grads are not materialized (see forward above) so an unused entropy output arrives becomes None, and as we don't compare values we don't have any sync
        assert grad_entropy is None, "Backward through entropy is not implemented in FusedOutputLinear"
        assert grad_logprobs is not None or grad_support is not None, "FusedOutputLinear requires a logprob gradient"

        hidden, weight, labels, inv_temperature, logz, support_ids, support_rows, replay = ctx.saved_tensors
        offsets: torch.Tensor | None = ctx.offsets
        chunk_size: int = ctx.chunk_size
        softcap: float | None = ctx.softcap
        if ctx.sparse_backward:
            grad_hidden, grad_weight = _sparse_logit_backward(
                hidden,
                weight,
                labels,
                inv_temperature,
                logz,
                support_ids,
                support_rows,
                offsets,
                replay,
                *ctx.sparse_logits,
                grad_logprobs,
                grad_support,
                softcap,
                ctx.needs_input_grad[:2],
            )
            return grad_hidden, grad_weight, None, None, None, None, None, None, None, None, None

        n, _ = hidden.shape
        device = hidden.device
        vocab = weight.shape[0]
        vocab_chunk_size = min(vocab, 8192)

        needs_hidden, needs_weight = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_hidden = torch.zeros_like(hidden) if needs_hidden else None
        grad_weight = torch.zeros_like(weight) if needs_weight else None

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            grad_chunk = (
                grad_logprobs[start:end].float() if grad_logprobs is not None else torch.zeros_like(logz[start:end])
            )
            inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)
            logz_chunk = logz[start:end]
            grad_scale = -grad_chunk
            grad_head = None
            if support_ids is not None:
                first, last = int(offsets[start]), int(offsets[end])
                ids_chunk, rows_chunk = support_ids[first:last], support_rows[first:last] - start
                replay_chunk = replay[start:end]
                if grad_support is not None:
                    grad_head = grad_support[first:last].float()
                    grad_scale = grad_scale - segment_sum(grad_head, (offsets[start : end + 1] - first).to(device))

            for vocab_start in range(0, vocab, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                weight_chunk = weight[vocab_start:vocab_end]
                logits_chunk = hidden_chunk @ weight_chunk.t()
                scaled_logits = logits_chunk.to(torch.float32)
                if softcap is not None:
                    tanh_logits = torch.tanh(scaled_logits / softcap)
                    scaled_logits = softcap * tanh_logits
                scaled_logits = scaled_logits * inv_t_chunk

                if support_ids is not None:
                    flat, in_range = _support_flat_index(ids_chunk, rows_chunk, vocab_start, vocab_end)
                    # Replayed rows get softmax gradient only on support ids. Set masked-out
                    # logits to -inf before exp: a masked-out logit above the sampling-mask
                    # logZ would overflow exp() to inf (and inf * 0 = NaN in the grads).
                    in_support = torch.zeros(scaled_logits.numel() + 1, dtype=torch.bool, device=device)
                    in_support[torch.where(in_range, flat, scaled_logits.numel())] = True
                    masked_out = replay_chunk.unsqueeze(-1) & ~in_support[:-1].view(scaled_logits.shape)
                    scaled_logits.masked_fill_(masked_out, float("-inf"))
                probs = torch.exp(scaled_logits - logz_chunk.unsqueeze(-1))

                grad_logits = grad_scale.unsqueeze(-1) * probs
                # Branchless grad scatter like we did in the sync-free forward
                in_range_label = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                local_idx = (labels_chunk - vocab_start).clamp(0, vocab_end - vocab_start - 1).to(torch.int64)
                grad_logits.scatter_add_(1, local_idx.unsqueeze(1), (grad_chunk * in_range_label).unsqueeze(1))
                if grad_head is not None:
                    grad_logits.view(-1).index_add_(0, flat, grad_head * in_range)
                grad_logits = grad_logits * inv_t_chunk
                if softcap is not None:
                    grad_logits = grad_logits * (1 - tanh_logits.square())

                if needs_hidden:
                    grad_hidden[start:end].add_(grad_logits.to(hidden.dtype) @ weight_chunk)
                if needs_weight:
                    grad_weight[vocab_start:vocab_end].add_(grad_logits.to(weight.dtype).t() @ hidden_chunk)

        return grad_hidden, grad_weight, None, None, None, None, None, None, None, None, None


class _ChunkedCrossEntropySumFn(torch.autograd.Function):
    """Summed cross-entropy over labels != IGNORE_INDEX, with the gradients computed during forward.

    Per chunk of ``chunk_size`` tokens: one full-vocab logits GEMM, the loss, and the gradient
    (softmax minus one-hot) folded straight into ``grad_hidden`` and ``grad_weight``. Only one chunk's
    logits are alive at a time and backward does no recompute: it scales the stored gradients by the
    incoming scalar. That drops the head's matmul passes from four to three (forward logits, dX, dW).
    Same approach as torchtitan's ChunkedLossWrapper.

    Backward only accepts a scalar upstream gradient (e.g. ``loss_sum / k``). Per-token weighted losses
    must pass a temperature and go through the logprob path instead.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        labels: torch.Tensor,  # [N], IGNORE_INDEX where the token has no loss
        chunk_size: int,
        softcap: float | None = None,
    ) -> torch.Tensor:
        assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() == 1
        assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[1]
        assert chunk_size > 0

        needs_hidden, needs_weight = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_hidden = torch.empty_like(hidden) if needs_hidden else None
        grad_weight = torch.zeros_like(weight) if needs_weight else None
        loss = torch.zeros((), device=hidden.device, dtype=torch.float32)

        for start in range(0, hidden.shape[0], chunk_size):
            end = min(start + chunk_size, hidden.shape[0])
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            valid = labels_chunk != IGNORE_INDEX
            target = labels_chunk.clamp(min=0).unsqueeze(-1)

            logits = (hidden_chunk @ weight.t()).float()
            if softcap is not None:
                tanh_logits = torch.tanh(logits / softcap)
                logits = softcap * tanh_logits
            logz = torch.logsumexp(logits, dim=-1)
            target_logits = logits.gather(1, target).squeeze(1)
            loss += torch.where(valid, logz - target_logits, 0.0).sum()

            if needs_hidden or needs_weight:
                # dx(loss)/dx(logits) is softmax minus one-hot on valid rows and zero elsewhere so it reuses the logits buffer
                grad_logits = logits.sub_(logz.unsqueeze(-1)).exp_()
                grad_logits.scatter_add_(1, target, torch.full_like(target, -1, dtype=grad_logits.dtype))
                grad_logits = grad_logits.mul_(valid.unsqueeze(-1))
                if softcap is not None:
                    grad_logits = grad_logits.mul_(1 - tanh_logits.square())
                grad_logits = grad_logits.to(hidden.dtype)
                if needs_hidden:
                    torch.mm(grad_logits, weight, out=grad_hidden[start:end])
                if needs_weight:
                    grad_weight.addmm_(grad_logits.t(), hidden_chunk)

        ctx.save_for_backward(grad_hidden, grad_weight)
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        grad_hidden, grad_weight = ctx.saved_tensors
        if grad_hidden is not None:
            grad_hidden = grad_hidden * grad_loss.to(grad_hidden.dtype)
        if grad_weight is not None:
            grad_weight = grad_weight * grad_loss.to(grad_weight.dtype)
        return grad_hidden, grad_weight, None, None, None


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
            loss/logprob/entropy computation.
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

    # Replace the lm_head with the appropriate wrapper, keeping a final-logit softcap the model's own head applies
    old_lm_head = model.lm_head
    softcap = getattr(old_lm_head, "softcap", None)
    if isinstance(chunk_size, int):
        logger.info(f"Injecting chunked LM head with chunk size {chunk_size}")
        model.lm_head = FusedOutputLinear(
            in_features=old_lm_head.in_features,
            out_features=old_lm_head.out_features,
            chunk_size=chunk_size,
            softcap=softcap,
        )
    else:
        logger.info("Injecting vanilla LM head")
        model.lm_head = VanillaOutputLinear(
            in_features=old_lm_head.in_features, out_features=old_lm_head.out_features, softcap=softcap
        )
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
        support: TokenSupport | None = None,
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
            support=support,
        )

    # Bind the new forward to the model
    model.forward = types.MethodType(new_forward, model)
