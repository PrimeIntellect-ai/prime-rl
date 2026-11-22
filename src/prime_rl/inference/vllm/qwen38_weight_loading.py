import re
from collections.abc import Iterable
from functools import wraps

import torch


def patch_qwen38_weight_loading() -> None:
    """Preserve CPU PLE storage and skip nonlocal checkpoint pieces during reload.

    During a reload of a CPU-offloaded unquantized PLE table, each shard's local rows are
    copied from the received tensor straight into the live pinned table. vLLM's layerwise
    reload would otherwise hold every shard (each in a fresh pinned copy) until the whole
    table arrived, then rebuild the table and copy it over.

    Remove when vLLM preserves CPU PLE placement and checks TP overlap before
    calling the embedding parameter's weight_loader.
    """
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding

    original_load_weights = Qwen4ExpNGramEmbedding.load_weights
    if getattr(original_load_weights, "_prime_rl_filters_ple_shards", False):
        return

    @wraps(original_load_weights)
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        embedding = self.ngram_embedding
        cpu_offloaded = embedding.weight.device.type == "cpu"
        info = get_layerwise_info(embedding)
        live_weight = None
        if info.kernel_tensors is not None:
            original_parameters, _ = info.kernel_tensors
            original_weight = original_parameters["weight"]
            cpu_offloaded = original_weight.device.type == "cpu"
            # vLLM records the default CUDA device even for CPU-offloaded PLE weights.
            if original_weight.device.type == "cpu":
                info.restore_device = original_weight.device
            # Quantized tables also load scale shards through the layerwise loader, whose
            # processing would overwrite rows written in place, so they keep the buffered path.
            if cpu_offloaded and set(original_parameters) == {"weight"}:
                live_weight = original_weight
        shard_size = (embedding.org_vocab_size + self.split_ngram_parts - 1) // self.split_ngram_parts
        skipped: set[str] = set()
        wrote_in_place = False

        def local_weights():
            nonlocal wrote_in_place
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
                        overlap = embedding._compute_shard_overlap(
                            checkpoint_start=checkpoint_start, checkpoint_rows=expected_rows
                        )
                        if overlap is None:
                            skipped.add("ngram_embedding.weight")
                            continue
                        # Preserve symbolic tensor subclasses used to trace NIXL transfers.
                        if live_weight is not None and type(weight) is torch.Tensor:
                            rows = weight.narrow(0, overlap.source_start, overlap.row_count)
                            live_weight.narrow(0, overlap.destination_start, overlap.row_count).copy_(rows)
                            skipped.add("ngram_embedding.weight")
                            wrote_in_place = True
                            continue
                    if cpu_offloaded and type(weight) is torch.Tensor and weight.is_cuda:
                        cpu_weight = torch.empty_like(weight, device="cpu", pin_memory=True)
                        cpu_weight.copy_(weight, non_blocking=False)
                        weight = cpu_weight
                yield name, weight

        loaded = original_load_weights(self, local_weights())
        if wrote_in_place:
            # Every local row already sits in the live table: finalize puts it back as is.
            info.load_numel_total = 0
        return loaded | skipped

    load_weights._prime_rl_filters_ple_shards = True
    Qwen4ExpNGramEmbedding.load_weights = load_weights
