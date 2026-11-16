"""Validation sets (``[[val.sets]]``), scored on the training model. Every set is tokenized once at startup into
requests (`prepare_set`). Every validation scores the requests of all sets together (`score`, with the model's LM
head swapped for `scoring_head`) and summarizes each set (`ValSet.summarize`)."""

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

from prime_rl.configs.supervised import ChatValSetConfig, DocumentsValSetConfig, ValConfig, ValSetConfig
from prime_rl.trainer.model import forward
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.pretrain.data import TokenShards, document_windows, pack_windows, special_token_ids
from prime_rl.trainer.sft.data import RendererResolver, SFTDataset
from prime_rl.trainer.world import get_world
from prime_rl.utils.cp import setup_cp_params, shard_for_cp
from prime_rl.utils.logger import get_logger

Request = tuple[np.ndarray, np.ndarray]
"""Input token ids and their labels: the next token where it is scored, else ``IGNORE_INDEX``."""


def perplexity(nll: np.ndarray, tokens: np.ndarray, num_bytes: np.ndarray | None) -> dict[str, float]:
    """Mean loss per scored token, and bits per UTF-8 byte of the scored tokens when their bytes are known."""
    summary = {"loss": nll.sum() / tokens.sum()}
    if num_bytes is not None:
        summary["bpb"] = nll.sum() / (num_bytes.sum() * math.log(2))
    return summary


@dataclass
class ValSet:
    """A set's requests and the item (document, row) every request belongs to."""

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

        arrays = {"nll": per_item(-logprobs), "tokens": per_item(tokens).astype(np.int64)}
        if num_bytes is not None:
            arrays["bytes"] = per_item(num_bytes).astype(np.int64)
        return perplexity(arrays["nll"], arrays["tokens"], arrays.get("bytes")), arrays


def documents_set(config: DocumentsValSetConfig, num_samples: int | None) -> ValSet:
    """Every document (or the first ``num_samples``) in windows of ``seq_len`` scored tokens, as stored."""
    shards = TokenShards([config.path])
    documents = document_windows(shards, config.seq_len, num_samples)
    requests, items = [], []
    for item, windows in enumerate(documents):
        for file_index, start, end in windows:
            ids = shards.open(file_index)[0][start:end].astype(np.int64)
            requests.append((ids[:-1], ids[1:]))
            items.append(item)
    return ValSet(config.name, config.seq_len, requests, np.array(items), len(documents))


def chat_set(
    config: ChatValSetConfig, num_samples: int | None, tokenizer: PreTrainedTokenizer, renderer: RendererConfig
) -> ValSet:
    """The rows of the split (or the first ``num_samples``), rendered as SFT renders them, with the tokens SFT trains
    on scored and every row cut after ``seq_len`` input tokens. Rows with no scored token in that window are left out,
    as in SFT."""
    dataset = SFTDataset(
        load_dataset(config.path, config.subset, split=config.split, revision=config.revision),
        RendererResolver(tokenizer, renderer, columns=config.columns.renderer),
        shuffle=False,
        seq_len=config.seq_len,
        loss_mask_config=config.loss_mask,
        max_examples=num_samples,
        columns=config.columns,
    )
    requests = []
    for example in dataset.dataset:
        sample = dataset.process(example)
        if sample is None:
            continue
        labels = np.where(sample["loss_mask"], sample["target_ids"], IGNORE_INDEX)
        requests.append((np.array(sample["input_ids"][: config.seq_len]), labels[: config.seq_len]))
    return ValSet(config.name, config.seq_len, requests, np.arange(len(requests)), len(requests))


def prepare_set(
    config: ValSetConfig, num_samples: int | None, tokenizer: PreTrainedTokenizer, renderer: RendererConfig
) -> ValSet:
    if config.type == "documents":
        return documents_set(config, num_samples)
    return chat_set(config, num_samples, tokenizer, renderer)


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
    """Log-prob and greedy hit of ``targets`` under ``hidden @ weight.T``, in fp32 chunks over tokens, so the
    fp32 logits stay at ``chunk x vocab``. With ``softcap`` (the head's final-logit soft-cap, as in training) the
    logits are ``softcap * tanh(logits / softcap)`` first."""
    weight = weight.float()
    logprobs, hits = [], []
    # At least one chunk: a CP shard without scored tokens returns empty tensors.
    for start in range(0, max(len(targets), 1), chunk):
        logits = hidden[start : start + chunk].float() @ weight.T
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        target = targets[start : start + chunk]
        logprobs.append(logits.gather(-1, target[:, None])[:, 0] - torch.logsumexp(logits, -1))
        hits.append(logits.argmax(-1) == target)
    return torch.cat(logprobs), torch.cat(hits)


def scoring_head(self, hidden_states, labels, temperature=None, support=None):
    """Replaces the forward of prime-rl's LM head: log-prob and greedy hit of every label that is not
    IGNORE_INDEX, in fp32 without TF32, returned in the ``logprobs`` / ``entropy`` slots of its output dict."""
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    labels = labels.reshape(-1)
    keep = labels != IGNORE_INDEX
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        logprobs, hits = head_logprobs(hidden[keep], self.weight, labels[keep], softcap=getattr(self, "softcap", None))
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


def score(
    model: nn.Module, requests: list[Request], micro_batch_tokens: int, parallel_dims: ParallelDims
) -> tuple[np.ndarray, np.ndarray] | None:
    """Summed log-prob and greedy hits of every request's scored tokens, on rank 0 (None on the others). Requests are
    packed longest first into document-masked rows of ``micro_batch_tokens`` tokens (positions restart per request,
    ``seq_lens`` marks them), padded with unscored tokens, so every forward has the same shape. The rows are dealt
    round-robin over the data-parallel ranks, which run the same number of forwards (each is an FSDP collective);
    with CP, every rank scores its contiguous shard of the row. The sums are gathered on rank 0."""
    world = get_world()
    dp = parallel_dims.get_mesh("dp")
    data_rank, data_world_size = dp.get_local_rank(), dp.size()
    cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if parallel_dims.cp_enabled else 0
    cp_size = parallel_dims.cp
    rows = pack_windows([len(inputs) for inputs, _ in requests], micro_batch_tokens)
    mine = rows[data_rank::data_world_size]
    indices, logprob_sums, hit_sums = [], [], []
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
        input_ids = torch.from_numpy(inputs)[None].cuda()
        position_ids = torch.cat([torch.arange(length) for length in lengths])[None].cuda()
        label_ids = torch.from_numpy(labels)[None].cuda()
        if cp_size > 1:
            input_ids, position_ids = setup_cp_params(input_ids, position_ids, cp_rank, cp_size)
            label_ids = shard_for_cp(label_ids, cp_rank=cp_rank, cp_world_size=cp_size)
            owners = np.split(owners, cp_size)[cp_rank]
        out = forward(
            model,
            input_ids,
            position_ids,
            seq_lens=torch.tensor(lengths, device="cuda"),
            labels=label_ids,
            seq_lens_are_pre_shard=cp_size > 1,
        )
        if i >= len(mine):
            continue
        logprobs, hits = out["logprobs"].double().cpu().numpy(), out["entropy"].cpu().numpy()
        # The scored tokens of a request are contiguous in the output: one run per owner.
        owners = owners[owners >= 0]
        starts = np.flatnonzero(np.diff(owners, prepend=-1))
        for start, end in zip(starts, [*starts[1:], len(owners)]):
            indices.append(owners[start])
            logprob_sums.append(logprobs[start:end].sum())
            hit_sums.append(int(hits[start:end].sum()))
    gathered = [None] * world.world_size if world.is_master else None
    sums = (np.array(indices, dtype=np.int64), np.array(logprob_sums), np.array(hit_sums, dtype=np.int64))
    dist.gather_object(sums, gathered, dst=0)
    if not world.is_master:
        return None
    logprobs, hits = np.zeros(len(requests)), np.zeros(len(requests), dtype=np.int64)
    for rank_indices, rank_logprobs, rank_hits in gathered:
        np.add.at(logprobs, rank_indices, rank_logprobs)
        np.add.at(hits, rank_indices, rank_hits)
    return logprobs, hits


def length_label(seq_len: int) -> str:
    return f"{seq_len // 1024}k" if seq_len % 1024 == 0 else str(seq_len)


class Validation:
    """The ``[[val.sets]]``, tokenized once at startup and scored on the training model at every validation."""

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
        self.sets = [prepare_set(val_set, config.num_samples, tokenizer, renderer) for val_set in config.sets]
        for val_set in self.sets:
            if not val_set.requests:
                raise ValueError(f"Validation set {val_set.name} has nothing to score")
        self.requests = [request for val_set in self.sets for request in val_set.requests]
        scored = [labels[labels != IGNORE_INDEX] for _, labels in self.requests]
        self.tokens = np.array([len(labels) for labels in scored])
        byte_lengths = token_byte_lengths(tokenizer)
        if byte_lengths is None:
            get_logger().warning("The tokenizer is not byte-level BPE: validation logs no bits per byte")
            self.bytes = None
        else:
            self.bytes = np.array([byte_lengths[labels].sum() for labels in scored])
        get_logger().info(
            f"Prepared {len(self.sets)} validation sets ({len(self.requests)} requests, {self.tokens.sum()} scored "
            f"tokens) in {time.perf_counter() - start_time:.0f}s"
        )

    @torch.no_grad()
    def run(self, model: nn.Module, step: int) -> dict[str, float]:
        """Score every set at step ``step``. The metrics are on rank 0 only."""
        with scoring_head_active(model):
            scored = score(model, self.requests, self.config.micro_batch_tokens, self.parallel_dims)
        if scored is None:
            return {}
        logprobs, hits = scored
        metrics, offset = {}, 0
        for val_set in self.sets:
            window = slice(offset, offset + len(val_set.requests))
            offset = window.stop
            num_bytes = self.bytes[window] if self.bytes is not None else None
            summary, arrays = val_set.summarize(logprobs[window], hits[window], self.tokens[window], num_bytes)
            length = length_label(val_set.seq_len)
            label = f"{val_set.name}@{length}"
            metrics |= {f"val/{val_set.name}/{key}@{length}": float(value) for key, value in summary.items()}
            get_logger().success(
                f"Validation | Step {step} | {label} | "
                + " | ".join(f"{key} {value:.4f}" for key, value in summary.items())
            )
            if self.config.save_arrays:
                out = self.run_dir / "val" / f"step_{step}"
                out.mkdir(parents=True, exist_ok=True)
                np.savez(out / f"{label}.npz", **arrays)
                (out / f"{label}.json").write_text(json.dumps(summary, default=float))
        return metrics
