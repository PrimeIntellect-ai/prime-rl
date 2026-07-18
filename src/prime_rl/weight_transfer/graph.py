"""Compose PrimeRL trainer conversion with vLLM's own weight loaders."""

from __future__ import annotations

import torch

from prime_rl.weight_transfer.lazy import BakeRecorder, LazyWeight
from prime_rl.weight_transfer.wire import TrainerTable


def make_hf_lazy_weights(
    table: TrainerTable,
    *,
    device: torch.device,
    recorder: BakeRecorder,
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
            getattr(torch, tensor.dtype),
            device,
            recorder,
        )
        for tensor in table.tensors
    }

    model_type = getattr(hf_config, "model_type", None)
    if model_type == "qwen3_moe":
        from prime_rl.trainer.models.conversion_ops import apply_tt_to_hf
        from prime_rl.trainer.models.qwen3_moe.converting_qwen3_moe import conversion_chain

        apply_tt_to_hf(state, conversion_chain(hf_config))
    elif model_type == "glm_moe_dsa":
        from prime_rl.trainer.models.conversion_ops import apply_tt_to_hf
        from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import conversion_chain

        apply_tt_to_hf(state, conversion_chain(hf_config))
    else:
        raise NotImplementedError(
            f"NIXL composed weight transfer does not yet declare a trainer-to-HF graph for {model_type!r}"
        )

    # AutoWeightsLoader groups adjacent names by module prefix. Stable sorting
    # matches normal checkpoint iterators and keeps every expert group intact.
    return sorted(state.items(), key=lambda item: item[0])
