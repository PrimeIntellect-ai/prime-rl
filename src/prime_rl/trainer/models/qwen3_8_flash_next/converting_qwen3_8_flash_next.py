from dataclasses import dataclass

import torch
from torch import Tensor

from prime_rl.trainer.models.conversion_ops import ConvOp, Drop, Rename, SplitConcat


@dataclass
class NGramEmbeddingWeights(ConvOp):
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

    def prime_to_hf(self, state_dict: dict[str, Tensor]) -> None:
        if self.runtime_weight not in state_dict:
            return

        shards = state_dict.pop(self.runtime_weight).chunk(self.shard_count, dim=0)
        for index, shard in enumerate(shards):
            state_dict[f"{self.checkpoint_prefix}.shard_{index}.weight"] = shard


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(
        name.endswith("mlp.experts.gate_up_proj") or ".ngram_embedding.shard_0.weight" in name for name in state_dict
    )


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(
        name.endswith("mlp.experts.gate_proj") or name.endswith("ple_embedding.ngram_embedding.weight")
        for name in state_dict
    )


def conversion_chain(config) -> list[ConvOp]:
    text_config = getattr(config, "text_config", config)
    model_prefix = "model.language_model"
    operations: list[ConvOp] = [
        Drop("model.visual.", is_prefix=True),
        Drop("mtp.", is_prefix=True),
    ]

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
        prefix = f"{model_prefix}.layers.{layer_index}.ple.ple_embedding.ngram_embedding"
        operations.append(
            NGramEmbeddingWeights(
                checkpoint_prefix=prefix,
                runtime_weight=f"{prefix}.weight",
                shard_count=text_config.split_ngram_parts,
            )
        )
    return operations


__all__ = ["conversion_chain", "is_hf_state_dict", "is_prime_state_dict"]
