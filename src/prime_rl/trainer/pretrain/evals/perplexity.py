import math
from dataclasses import dataclass
from itertools import islice

import torch
from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.pretrain import ValSourceConfig
from prime_rl.trainer.model import forward
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.pretrain.data import StreamPacker, load_eval_documents, tokenize_document
from prime_rl.trainer.pretrain.evals.common import all_reduce_sum, local_batches
from prime_rl.trainer.sft.data import Sample


@dataclass
class PerplexitySet:
    """A fixed, pre-packed held-out token set."""

    name: str
    rows: list[Sample]


def build_perplexity_set(
    name: str, documents, tokenizer: PreTrainedTokenizer, seq_len: int, max_tokens: int, text_column: str = "text"
) -> PerplexitySet:
    """Pack documents (in order) into full rows until ``max_tokens`` is reached. Rows use
    document masking, so every document is scored from its own start."""
    token_stream = (tokenize_document(tokenizer, example[text_column]) for example in documents)
    rows = list(islice(StreamPacker(token_stream, seq_len, document_masking=True), max(1, max_tokens // seq_len)))
    assert rows, f"Eval source {name!r} has fewer than {seq_len + 1} tokens"
    return PerplexitySet(name=name, rows=rows)


def load_perplexity_set(source: ValSourceConfig, tokenizer: PreTrainedTokenizer, seq_len: int) -> PerplexitySet:
    documents = load_eval_documents(source.dataset, source.subset, source.split, source.revision, source.text_column)
    return build_perplexity_set(source.name, documents, tokenizer, seq_len, source.max_tokens, source.text_column)


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module, sets: list[PerplexitySet], byte_lengths: torch.Tensor, batch_size: int
) -> dict[str, float]:
    """Loss (nats per token), perplexity and bits per byte for every set, ``batch_size`` rows per forward."""
    byte_lengths = byte_lengths.to("cuda")
    metrics = {}
    for eval_set in sets:
        totals = torch.zeros(3, dtype=torch.float64, device="cuda")  # loss, tokens, bytes
        for batch in local_batches(eval_set.rows, batch_size):
            loss_mask = batch["loss_mask"]
            labels = batch["target_ids"].masked_fill(~loss_mask, IGNORE_INDEX)
            out = forward(model, batch["input_ids"], batch["position_ids"], seq_lens=batch["seq_lens"], labels=labels)
            totals[0] += out["loss"].double()
            totals[1] += loss_mask.sum()
            totals[2] += byte_lengths[batch["target_ids"][loss_mask]].sum()
        loss_sum, num_tokens, num_bytes = all_reduce_sum(totals).tolist()
        loss = loss_sum / num_tokens
        metrics[f"val/{eval_set.name}/loss"] = loss
        metrics[f"val/{eval_set.name}/ppl"] = math.exp(min(loss, 20))
        metrics[f"val/{eval_set.name}/bpb"] = loss_sum / (math.log(2) * num_bytes)
    return metrics
