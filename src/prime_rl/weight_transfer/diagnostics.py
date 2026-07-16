"""Exact, dtype-stable fingerprints for weight-transfer diagnostics."""

from __future__ import annotations

import torch

from prime_rl.weight_transfer.wire import TensorFingerprint

_SAMPLE_COUNT = 16


@torch.no_grad()
def fingerprint_tensor(name: str, tensor: torch.Tensor) -> TensorFingerprint:
    """Fingerprint BF16 payload bits without reduction-order ambiguity.

    Integer moments are invariant to layout permutations; the sampled words
    additionally expose ordering mistakes.  Casting here mirrors the serving
    buffer dtype used by the NIXL broadcaster.
    """

    values = tensor.detach().to(torch.bfloat16).contiguous().view(torch.uint16).reshape(-1)
    words = values.to(torch.int64)
    numel = words.numel()
    stride = max(numel // _SAMPLE_COUNT, 1)
    samples = tuple(int(value) for value in words[::stride][:_SAMPLE_COUNT].cpu().tolist())
    return TensorFingerprint(
        name=name,
        numel=numel,
        word_sum=int(words.sum().item()),
        word_square_sum=int(words.square().sum().item()),
        samples=samples,
    )
