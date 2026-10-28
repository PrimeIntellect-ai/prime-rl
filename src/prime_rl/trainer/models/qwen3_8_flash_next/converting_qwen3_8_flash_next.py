from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from prime_rl.trainer.models.conversion_ops import ChunkedConvOp, ConvOp, Drop, Rename, SplitConcat
from prime_rl.trainer.models.layers.ngram import LAYOUT_BUFFERS


@dataclass
class NGramEmbeddingWeights(ChunkedConvOp):
    checkpoint_prefix: str
    runtime_weight: str
    shard_count: int

    def hf_to_prime(self, state_dict: dict[str, Tensor]) -> None:
        first_shard = f"{self.checkpoint_prefix}.shard_0.weight"
        if first_shard not in state_dict:
            return

        state_dict[self.runtime_weight] = torch.cat(
            [
                state_dict.pop(f"{self.checkpoint_prefix}.shard_{shard_index}.weight")
                for shard_index in range(self.shard_count)
            ],
            dim=0,
        )

    def iter_prime_to_hf(
        self, weight: Tensor, resolve_rows: Callable[[Tensor, int, int], Tensor | None]
    ) -> Iterator[dict[str, Tensor] | None]:
        shard_size = (weight.shape[0] + self.shard_count - 1) // self.shard_count
        for index in range(self.shard_count):
            start = min(index * shard_size, weight.shape[0])
            stop = min(start + shard_size, weight.shape[0])
            chunk = resolve_rows(weight, start, stop)
            yield {f"{self.checkpoint_prefix}.shard_{index}.weight": chunk} if chunk is not None else None
            del chunk


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(
        name.endswith("mlp.experts.gate_up_proj") or ".ngram_embedding.shard_0.weight" in name for name in state_dict
    )


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(name.endswith("mlp.experts.gate_proj") or ".ngram.tables." in name for name in state_dict)


def conversion_chain(config) -> list[ConvOp]:
    text_config = getattr(config, "text_config", config)
    model_prefix = "model.language_model" if hasattr(config, "text_config") else "model"
    operations: list[ConvOp] = [Drop("mtp.", is_prefix=True)]

    for layer_index in range(text_config.num_hidden_layers):
        prefix = f"{model_prefix}.layers.{layer_index}.mlp"
        operations.extend(
            [
                Rename(f"{prefix}.gate.weight", f"{prefix}.router.gate.weight"),
                Rename(
                    f"{prefix}.shared_expert_gate.weight",
                    f"{prefix}.shared_expert.output_gate.weight",
                ),
                SplitConcat(
                    combined=f"{prefix}.experts.gate_up_proj",
                    parts=[
                        (f"{prefix}.experts.gate_proj", None),
                        (f"{prefix}.experts.up_proj", None),
                    ],
                    dim=1,
                ),
            ]
        )

    for layer_index in (layer_id - 1 for layer_id in sorted(text_config.ple_layer_ids)):
        checkpoint_prefix = f"{model_prefix}.layers.{layer_index}.ple.ple_embedding"
        runtime_prefix = f"{model_prefix}.ngram.tables.{layer_index}"
        operations.append(
            NGramEmbeddingWeights(
                checkpoint_prefix=f"{checkpoint_prefix}.ngram_embedding",
                runtime_weight=f"{runtime_prefix}.weight",
                shard_count=text_config.split_ngram_parts,
            )
        )
        operations.extend(
            Rename(f"{checkpoint_prefix}.{buffer}", f"{runtime_prefix}.{buffer}") for buffer in LAYOUT_BUFFERS
        )
    return operations


__all__ = ["conversion_chain", "is_hf_state_dict", "is_prime_state_dict"]
