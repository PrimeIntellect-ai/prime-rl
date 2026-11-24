"""Validation sources (``[[val.source]]``), scored on the training model. Every source is tokenized once at startup
into requests (`prepare_source`). Every validation scores the requests of all sources together (`score`, with the
model's LM head swapped for `scoring_head`) and summarizes each source (`ValSource.summarize`)."""

import json
import math
import time
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from renderers import RendererConfig
from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.val import ChatValSourceConfig, DocumentsValSourceConfig, ValConfig, ValSourceConfig
from prime_rl.trainer.model import forward, is_tt_moe_model, reshard_module, take_moe_routing_stats
from prime_rl.trainer.models.layers.fp32_linear import fp32_output_mm
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.pretrain.data import TokenShards, special_token_ids
from prime_rl.trainer.sft.data import HFSource, source_renderers
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger

Request = tuple[np.ndarray, np.ndarray]
"""Input token ids and their labels: the next token where it is scored, else ``IGNORE_INDEX``."""


def loss_and_bpb(nll: np.ndarray, tokens: np.ndarray, num_bytes: np.ndarray | None) -> dict[str, float]:
    """Mean loss per scored token, and bits per UTF-8 byte of the scored tokens when their bytes are known."""
    summary = {"loss": nll.sum() / tokens.sum()}
    if num_bytes is not None:
        summary["bpb"] = nll.sum() / (num_bytes.sum() * math.log(2))
    return summary


@dataclass
class ValSource:
    """A source's requests and the item (document, row) every request belongs to."""

    name: str
    seq_len: int
    requests: list[Request]
    items: np.ndarray
    num_items: int

    def summarize(
        self, logprobs: np.ndarray, hits: np.ndarray, tokens: np.ndarray, num_bytes: np.ndarray | None
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        """Summary metrics and per-item arrays, from every request's summed log-prob, greedy hits, scored tokens and
        their UTF-8 bytes (None if the tokenizer is not byte-level BPE)."""

        def per_item(values: np.ndarray) -> np.ndarray:
            return np.bincount(self.items, weights=values, minlength=self.num_items)

        arrays = {
            "nll": per_item(-logprobs),
            "tokens": per_item(tokens).astype(np.int64),
            "greedy_tokens": per_item(hits).astype(np.int64),
        }
        if num_bytes is not None:
            arrays["bytes"] = per_item(num_bytes).astype(np.int64)
        return loss_and_bpb(arrays["nll"], arrays["tokens"], arrays.get("bytes")), arrays


Window = tuple[int, int, int]
"""File index and the token range ``[start, end)`` of a window in the file."""


def document_windows(shards: TokenShards, seq_len: int, num_documents: int | None = None) -> list[list[Window]]:
    """Every document of ``shards`` as stored (or the first ``num_documents``), each cut into non-overlapping windows
    of ``seq_len`` scored tokens. A window holds up to ``seq_len + 1`` tokens and scores all but its first. The next
    window starts at the last token of the one before and sees none of the tokens before it."""
    documents = []
    for file_index in range(len(shards.files)):
        begin = 0
        for end in shards.index(file_index).tolist():
            if len(documents) == num_documents:
                return documents
            documents.append(
                [(file_index, window, min(window + seq_len + 1, end)) for window in range(begin, end - 1, seq_len)]
            )
            begin = end
    return documents


def documents_source(config: DocumentsValSourceConfig, num_samples: int | None) -> ValSource:
    """Every document (or the first ``num_samples``) in windows of ``seq_len`` scored tokens, as stored."""
    shards = TokenShards([config.path])
    documents = document_windows(shards, config.seq_len, num_samples)
    requests, items = [], []
    for item, windows in enumerate(documents):
        for file_index, start, end in windows:
            ids = shards.tokens(file_index)[start:end].astype(np.int32)
            requests.append((ids[:-1], ids[1:]))
            items.append(item)
    return ValSource(config.name, config.seq_len, requests, np.array(items, dtype=np.int64), len(documents))


def chat_source(
    config: ChatValSourceConfig, num_samples: int | None, tokenizer: PreTrainedTokenizer, renderer: RendererConfig
) -> ValSource:
    """The rows of the split (or the first ``num_samples``), rendered as SFT renders them, with the tokens SFT trains
    on scored and every row cut after ``seq_len`` input tokens. A row with no scored token in that window scores
    nothing, as SFT skips it."""
    source = HFSource(
        load_dataset(config.dataset, config.subset, split=config.split, revision=config.revision),
        source_renderers(tokenizer, renderer, config),
        name=config.name,
        shuffle=False,
        seq_len=config.seq_len,
        loss_mask_config=config.loss_mask,
        max_examples=num_samples,
        columns=config.columns,
    )
    requests, items = [], []
    for item, example in enumerate(source.dataset):
        sample = source.process(example)
        if sample is None:
            continue
        inputs = np.array(sample["input_ids"][: config.seq_len], dtype=np.int32)
        labels = np.where(sample["loss_mask"], sample["target_ids"], IGNORE_INDEX)[: config.seq_len].astype(np.int32)
        requests.append((inputs, labels))
        items.append(item)
    return ValSource(config.name, config.seq_len, requests, np.array(items, dtype=np.int64), source.size)


def prepare_source(
    config: ValSourceConfig, num_samples: int | None, tokenizer: PreTrainedTokenizer, renderer: RendererConfig
) -> ValSource:
    if config.type == "documents":
        return documents_source(config, num_samples)
    return chat_source(config, num_samples, tokenizer, renderer)


# The 256 characters byte-level BPE tokenizers (GPT-2, Llama 3, Qwen, DeepSeek) write the bytes as.
BYTE_CHARS = {chr(c) for c in [*range(33, 127), *range(161, 173), *range(174, 256), *range(256, 324)]}


def token_byte_lengths(tokenizer: PreTrainedTokenizer) -> np.ndarray | None:
    """The UTF-8 bytes of every token id, special tokens 0. None if the tokenizer is not byte-level BPE."""
    special = special_token_ids(tokenizer)
    lengths = np.zeros(len(tokenizer), dtype=np.int64)
    for token_id, token in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))):
        if token_id in special:
            continue
        if token_id in tokenizer.added_tokens_decoder:
            # Added tokens hold their text as is, not in the byte alphabet.
            lengths[token_id] = len(token.encode())
            continue
        if not set(token) <= BYTE_CHARS:
            return None
        lengths[token_id] = len(token)
    return lengths


def head_logprobs(hidden, weight, targets, chunk: int = 2048, softcap: float | None = None):
    """Log-prob and greedy hit of ``targets`` under fp32 logits ``hidden @ weight.T``, in chunks over tokens, so the
    logits stay at ``chunk x vocab``. With ``softcap`` (the head's final-logit soft-cap, as in training) the logits
    are ``softcap * tanh(logits / softcap)`` first."""
    logprobs, hits = [], []
    # At least one chunk: a CP shard without scored tokens returns empty tensors.
    for start in range(0, max(len(targets), 1), chunk):
        logits = fp32_output_mm(hidden[start : start + chunk], weight)
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        target = targets[start : start + chunk]
        logprobs.append(logits.gather(-1, target[:, None])[:, 0] - torch.logsumexp(logits, -1))
        hits.append(logits.argmax(-1) == target)
    return torch.cat(logprobs), torch.cat(hits)


def scoring_head(self, hidden_states, labels, temperature=None, support=None):
    """Replaces the forward of prime-rl's LM head: log-prob and greedy hit of every label that is not
    IGNORE_INDEX, from fp32 logits (an fp32 model's without TF32), returned in the ``logprobs`` / ``entropy`` slots
    of its output dict."""
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    labels = labels.reshape(-1)
    keep = labels != IGNORE_INDEX
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        logprobs, hits = head_logprobs(
            hidden[keep], self.projection_weight(), labels[keep], softcap=getattr(self, "softcap", None)
        )
    finally:
        torch.set_float32_matmul_precision(precision)
    return {"logprobs": logprobs, "entropy": hits}


@contextmanager
def scoring_head_active(model: nn.Module) -> Iterator[None]:
    """The model's LM head returns per-token log-probs and greedy hits (`scoring_head`) inside the block."""
    model.lm_head.forward = types.MethodType(scoring_head, model.lm_head)
    try:
        yield
    finally:
        del model.lm_head.forward


def pack_rows(lengths: list[int], micro_batch_tokens: int) -> list[list[int]]:
    """Request indices grouped into rows of at most ``micro_batch_tokens`` tokens, longest first."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    rows, row, total = [], [], 0
    for index in order:
        if row and total + lengths[index] > micro_batch_tokens:
            rows.append(row)
            row, total = [], 0
        row.append(index)
        total += lengths[index]
    if row:
        rows.append(row)
    return rows


def rank_rows(
    requests: list[Request], micro_batch_tokens: int, data_rank: int, data_world_size: int, cp_rank: int, cp_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray | None]]:
    """The rows one rank runs. The requests are packed into document-masked rows of ``micro_batch_tokens`` tokens
    (positions restart per request, the segment lengths mark them), padded with unscored tokens, so every forward has
    the same shape. The rows are dealt round-robin over the data ranks, which all run the same number of rows (each
    forward is an FSDP collective): a rank short of a row runs the first one again, unscored. With CP, a rank holds a
    contiguous shard of the row, as in training.

    Yields the shard's input ids, position ids and labels, the whole row's segment lengths, and the request of every
    scored label in the shard (None for a row that is not scored)."""
    rows = pack_rows([len(inputs) for inputs, _ in requests], micro_batch_tokens)
    mine = rows[data_rank::data_world_size]
    for i in range(-(-len(rows) // data_world_size)):
        row = mine[i] if i < len(mine) else rows[0]
        lengths = [len(requests[index][0]) for index in row]
        inputs = np.concatenate([requests[index][0] for index in row])
        labels = np.concatenate([requests[index][1] for index in row])
        owners = np.where(labels != IGNORE_INDEX, np.repeat(row, lengths), -1)
        padding = micro_batch_tokens - len(inputs)
        if padding:
            # The row's own tokens, as in training: identical pad tokens would all route to the same MoE experts.
            inputs = np.concatenate([inputs, np.resize(inputs, padding)])
            labels = np.concatenate([labels, np.full(padding, IGNORE_INDEX)])
            owners = np.concatenate([owners, np.full(padding, -1)])
            lengths.append(padding)
        positions = np.concatenate([np.arange(length) for length in lengths])
        inputs, positions, labels, owners = (np.split(x, cp_size)[cp_rank] for x in (inputs, positions, labels, owners))
        yield inputs, positions, labels, lengths, owners[owners >= 0] if i < len(mine) else None


def score(
    model: nn.Module, requests: list[Request], micro_batch_tokens: int, parallel_dims: ParallelDims
) -> tuple[np.ndarray, np.ndarray] | None:
    """Summed log-prob and greedy hits of every request's scored tokens, on rank 0 (None on the others). Every rank
    scores its rows (`rank_rows`), and the sums are reduced onto rank 0."""
    world = get_world()
    dp = parallel_dims.get_mesh("dp")
    cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if parallel_dims.cp_enabled else 0
    rows = rank_rows(requests, micro_batch_tokens, dp.get_local_rank(), dp.size(), cp_rank, parallel_dims.cp)
    # Every request's summed log-prob and greedy hits on this rank (hits are counts, exact in float64).
    sums = np.zeros((2, len(requests)))
    for inputs, positions, labels, lengths, owners in rows:
        out = forward(
            model,
            torch.from_numpy(inputs.astype(np.int64))[None].cuda(),
            torch.from_numpy(positions)[None].cuda(),
            seq_lens=torch.tensor(lengths, device="cuda"),
            labels=torch.from_numpy(labels.astype(np.int64))[None].cuda(),
            seq_lens_are_pre_shard=parallel_dims.cp_enabled,
        )
        if owners is None:
            continue
        logprobs, hits = out["logprobs"].double().cpu().numpy(), out["entropy"].cpu().numpy()
        # The scored tokens of a request are contiguous in the output: one run per request.
        starts = np.flatnonzero(np.diff(owners, prepend=-1))
        for start, end in zip(starts, [*starts[1:], len(owners)]):
            sums[0, owners[start]] += logprobs[start:end].sum()
            sums[1, owners[start]] += hits[start:end].sum()
    total = torch.from_numpy(sums).cuda()
    dist.reduce(total, dst=0)
    if not world.is_master:
        return None
    logprobs, hits = total.cpu().numpy()
    return logprobs, hits.astype(np.int64)


def length_label(seq_len: int) -> str:
    return f"{seq_len // 1024}k" if seq_len % 1024 == 0 else str(seq_len)


class Validation:
    """The ``[[val.source]]``, tokenized once at startup and scored on the training model at every validation."""

    def __init__(
        self,
        config: ValConfig,
        tokenizer: PreTrainedTokenizer,
        renderer: RendererConfig,
        parallel_dims: ParallelDims,
        run_dir: Path,
    ):
        start_time = time.perf_counter()
        self.config = config
        self.parallel_dims = parallel_dims
        self.run_dir = run_dir
        self.sources = [
            prepare_source(val_source, config.num_samples, tokenizer, renderer) for val_source in config.source
        ]
        for val_source in self.sources:
            if not val_source.requests:
                raise ValueError(f"Validation source {val_source.name} has nothing to score")
        self.requests = [request for val_source in self.sources for request in val_source.requests]
        byte_lengths = token_byte_lengths(tokenizer)
        if byte_lengths is None:
            get_logger().warning("The tokenizer is not byte-level BPE: validation logs no bits per byte")
        tokens, num_bytes = [], []
        for _, labels in self.requests:
            scored = labels[labels != IGNORE_INDEX]
            tokens.append(len(scored))
            if byte_lengths is not None:
                num_bytes.append(byte_lengths[scored].sum())
        self.tokens = np.array(tokens)
        self.bytes = np.array(num_bytes) if byte_lengths is not None else None
        get_logger().info(
            f"Prepared {len(self.sources)} validation sources ({len(self.requests)} requests, "
            f"{self.tokens.sum()} scored tokens) in {time.perf_counter() - start_time:.0f}s"
        )

    @torch.no_grad()
    def run(self, model: nn.Module, step: int) -> dict[str, float]:
        """Score every source at step ``step``. The metrics are on rank 0 only."""
        # No train/eval switch: no dropout in these models, and toggling would trigger torch.compile recompilation
        with scoring_head_active(model):
            scored = score(model, self.requests, self.config.micro_batch_tokens, self.parallel_dims)
        if is_tt_moe_model(model):
            # Keep validation routing out of the next training step's statistics.
            take_moe_routing_stats(model)
        # The no-grad forward leaves the [lm_head, norm] FSDP2 group unsharded (it opts out of reshard_after_forward),
        # so a gradient clip would skip its grad-less unsharded parameters.
        reshard_module(model)
        if scored is None:
            return {}
        logprobs, hits = scored
        metrics, offset = {}, 0
        for val_source in self.sources:
            window = slice(offset, offset + len(val_source.requests))
            offset = window.stop
            num_bytes = self.bytes[window] if self.bytes is not None else None
            summary, arrays = val_source.summarize(logprobs[window], hits[window], self.tokens[window], num_bytes)
            length = length_label(val_source.seq_len)
            label = f"{val_source.name}@{length}"
            metrics |= {f"val/{val_source.name}/{key}@{length}": float(value) for key, value in summary.items()}
            get_logger().success(
                f"Validation | Step {step} | {label} | "
                + " | ".join(f"{key} {value:.4f}" for key, value in summary.items())
            )
            if self.config.save_arrays:
                out = self.run_dir / "val" / f"step_{step}"
                out.mkdir(parents=True, exist_ok=True)
                np.savez(out / f"{label}.npz", **arrays)
                (out / f"{label}.json").write_text(json.dumps(summary))
        return metrics
