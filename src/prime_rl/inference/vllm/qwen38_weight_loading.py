import re
from collections.abc import Iterable
from functools import wraps

import torch


def patch_qwen38_weight_loading() -> None:
    """Preserve CPU PLE storage and skip nonlocal checkpoint pieces during reload.

    Remove when vLLM preserves CPU PLE placement and checks TP overlap before
    calling the embedding parameter's weight_loader.
    """
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
    from vllm.models.qwen4_exp.common.ple import compute_ple_shard_overlap
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding

    original_load_weights = Qwen4ExpNGramEmbedding.load_weights
    if getattr(original_load_weights, "_prime_rl_filters_ple_shards", False):
        return

    @wraps(original_load_weights)
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        embedding = self.ngram_embedding
        info = get_layerwise_info(embedding)
        if info.kernel_tensors is not None:
            original_parameters, _ = info.kernel_tensors
            original_weight = original_parameters["weight"]
            # vLLM records the default CUDA device even for CPU-offloaded PLE weights.
            if original_weight.device.type == "cpu":
                info.restore_device = original_weight.device
        shard_size = (embedding.org_vocab_size + self.split_ngram_parts - 1) // self.split_ngram_parts
        skipped: set[str] = set()

        def local_weights():
            for name, weight in weights:
                match = re.fullmatch(r"ngram_embedding\.shard_(\d+)\.weight", name)
                if match is not None:
                    shard_index = int(match[1])
                    checkpoint_start = shard_index * shard_size
                    expected_rows = max(0, min(shard_size, embedding.org_vocab_size - checkpoint_start))
                    # Leave malformed shards to the original loader's validation.
                    if shard_index < self.split_ngram_parts and weight.shape == (
                        expected_rows,
                        embedding.embedding_dim,
                    ):
                        overlap = compute_ple_shard_overlap(
                            checkpoint_start=checkpoint_start,
                            checkpoint_rows=expected_rows,
                            tp_start=embedding.shard_indices.org_vocab_start_index,
                            tp_end=embedding.shard_indices.org_vocab_end_index,
                        )
                        if overlap is None:
                            skipped.add("ngram_embedding.weight")
                            continue
                yield name, weight

        loaded = original_load_weights(self, local_weights())
        return loaded | skipped

    load_weights._prime_rl_filters_ple_shards = True
    Qwen4ExpNGramEmbedding.load_weights = load_weights
