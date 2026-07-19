"""Compose PrimeRL trainer conversion with vLLM's own weight loaders."""

from __future__ import annotations

import torch

from prime_rl.weight_transfer.lazy import LazyWeight, WeightLoadRecorder
from prime_rl.weight_transfer.wire import TrainerTable


def make_hf_lazy_weights(
    table: TrainerTable,
    *,
    device: torch.device,
    recorder: WeightLoadRecorder,
    hf_config,
) -> list[tuple[str, LazyWeight]]:
    """Create HF-named graph values rooted in trainer wire tensors.

    The returned values retain their trainer root name and accumulated view
    chain. Passing them to ``vLLM.model.load_weights`` composes the second half
    of the graph without any handwritten vLLM kernel conversion.
    """
    state: dict[str, LazyWeight] = {
        tensor.name: LazyWeight(
            tensor.name,
            torch.Size(tensor.shape),
            getattr(torch, tensor.wire_dtype),
            device,
            recorder,
        )
        for group in table.groups
        for tensor in group.tensors
    }

    # TODO(matej): Figure out how to avoid depending on trainer code here.
    from prime_rl.trainer.models import get_custom_causal_lm_cls

    model_cls = get_custom_causal_lm_cls(hf_config)
    model_cls.convert_to_hf(state)

    # AutoWeightsLoader groups adjacent names by module prefix. Stable sorting
    # matches normal checkpoint iterators and keeps every expert group intact.
    return sorted(state.items(), key=lambda item: item[0])
