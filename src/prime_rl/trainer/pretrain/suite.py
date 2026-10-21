"""The pretraining eval suite: log-likelihood scoring of text eval sets, as in the offline studies
(`pretrain-evals` offline suite). A set lives in ``<data_dir>/<name>/`` as ``test.parquet`` plus ``meta.json``
(``kind``, ``parquet_sha256``). Kinds:

Requests start with BOS only if the training documents do (``bos``), so evals see the training format.

- ``ppl``: column ``text``. Each document is [BOS +] text + EOS, cut into non-overlapping windows of ``seq_len``
  scored tokens (later windows start cold). Every token after the first is scored; without BOS, bpb leaves out
  the bytes of the first text token.
- ``ctxcont``: ``context``/``continuation`` (+ ``choices``/``gold``, optional ``choice_contexts``). The
  continuation is scored given [BOS +] context; context and continuation are tokenized jointly and the
  continuation span is found by character offsets.
- ``pairs``: ``good``/``bad`` sentences, each scored as [BOS +] sentence from its second token.

A set is tokenized once (`prepare_set`); every round scores its requests (`score_requests`, the model's LM head
replaced by `scoring_head`) and summarizes them (`summarize`)."""

import json
import math
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.world import get_world

POSITION_BINS = (0, 128, 512, 2048, 4096, 16384)
"""ppl loss is also reported per bin of the target's position in its window."""

Request = tuple[list[int], int]
"""Token ids and the index of the first scored token: ``ids[start:]`` is scored, each token given everything before it."""


@dataclass
class EvalSet:
    name: str
    kind: str
    table: Any
    requests: list[Request]
    context: dict
    sha256: str


def head_logprobs(hidden, weight, targets, chunk: int = 2048):
    """Log-prob and greedy hit of ``targets`` under ``hidden @ weight.T``, in fp32 chunks over tokens, so the
    fp32 logits stay at ``chunk x vocab``."""
    weight = weight.float()
    logprobs, hits = [], []
    for start in range(0, len(targets), chunk):
        logits = hidden[start : start + chunk].float() @ weight.T
        target = targets[start : start + chunk]
        logprobs.append(logits.gather(-1, target[:, None])[:, 0] - torch.logsumexp(logits, -1))
        hits.append(logits.argmax(-1) == target)
    return torch.cat(logprobs), torch.cat(hits)


def scoring_head(self, hidden_states, labels, temperature=None, sampling_mask=None, topk_ids=None):
    """Replaces the forward of prime-rl's LM head: log-prob and greedy hit of every label that is not
    IGNORE_INDEX, in fp32 without TF32, returned in the ``logprobs`` / ``entropy`` slots of its output dict."""
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    labels = labels.reshape(-1)
    keep = labels != IGNORE_INDEX
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    logprobs, hits = head_logprobs(hidden[keep], self.weight, labels[keep])
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


def pack_requests(requests: list[Request], token_budget: int) -> list[list[int]]:
    """Request indices grouped into rows of at most ``token_budget`` tokens, longest first."""
    order = sorted(range(len(requests)), key=lambda i: -len(requests[i][0]))
    batches, batch, total = [], [], 0
    for index in order:
        length = len(requests[index][0]) - 1
        if batch and total + length > token_budget:
            batches.append(batch)
            batch, total = [], 0
        batch.append(index)
        total += length
    if batch:
        batches.append(batch)
    return batches


@torch.no_grad()
def score_requests(
    model: nn.Module, requests: list[Request], token_budget: int, group: dist.ProcessGroup
) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    """Per-token log-probs and greedy hits of every request, on rank 0 (None on the others). Requests are
    packed into document-masked rows (positions restart per request, ``seq_lens`` marks them); the rows are
    dealt round-robin over the ranks, which run the same number of forwards (each is an FSDP collective), and
    the results are gathered on rank 0 over ``group``."""
    world = get_world()
    batches = pack_requests(requests, token_budget)
    mine = batches[world.rank :: world.world_size]
    results = []
    for i in range(-(-len(batches) // world.world_size)):
        batch = mine[i] if i < len(mine) else batches[0]
        inputs, labels, positions, lengths = [], [], [], []
        for index in batch:
            ids, start = requests[index]
            inputs += ids[:-1]
            labels += [IGNORE_INDEX] * (start - 1) + ids[start:]
            positions += range(len(ids) - 1)
            lengths.append(len(ids) - 1)
        out = model(
            input_ids=torch.tensor([inputs], device="cuda"),
            position_ids=torch.tensor([positions], device="cuda"),
            labels=torch.tensor([labels], device="cuda"),
            seq_lens=torch.tensor(lengths, device="cuda"),
        )
        if i < len(mine):
            results.append((batch, out["logprobs"].cpu(), out["entropy"].cpu()))
    gathered = [None] * world.world_size if world.is_master else None
    dist.gather_object(results, gathered, dst=0, group=group)
    if not world.is_master:
        return None
    logprobs: list[np.ndarray] = [None] * len(requests)
    hits: list[np.ndarray] = [None] * len(requests)
    for batch, flat_logprobs, flat_hits in (result for rank_results in gathered for result in rank_results):
        offset = 0
        for index in batch:
            n = len(requests[index][0]) - requests[index][1]
            logprobs[index] = flat_logprobs[offset : offset + n].double().numpy()
            hits[index] = flat_hits[offset : offset + n].numpy()
            offset += n
    return logprobs, hits


def score_sets(
    model: nn.Module, eval_sets: list["MultiLengthSet"], token_budget: int, group: dist.ProcessGroup
) -> list[tuple[list[np.ndarray], list[np.ndarray]]] | None:
    """`score_requests` over the requests of all sets at once, so the rows of every set are packed and dealt
    together (fewer padding forwards on many ranks), split back per set on rank 0 (None on the others)."""
    scored = score_requests(model, [request for s in eval_sets for request in s.requests], token_budget, group)
    if scored is None:
        return None
    logprobs, hits = scored
    per_set, offset = [], 0
    for eval_set in eval_sets:
        end = offset + len(eval_set.requests)
        per_set.append((logprobs[offset:end], hits[offset:end]))
        offset = end
    return per_set


def continuation_ids(tokenizer: PreTrainedTokenizer, context: str, continuation: str, prefix: list[int]) -> Request:
    """``prefix`` + jointly tokenized context + continuation, and the index of the first continuation token (the
    first token that ends after the context; a token straddling the boundary counts as continuation)."""
    encoding = tokenizer(context + continuation, add_special_tokens=False, return_offsets_mapping=True)
    first = next(i for i, (_, end) in enumerate(encoding["offset_mapping"]) if end > len(context))
    return [*prefix, *encoding["input_ids"]], first + len(prefix)


def truncate_left(ids: list[int], start: int, seq_len: int) -> Request:
    cut = max(0, len(ids) - seq_len - 1)
    assert start - cut >= 1, "continuation longer than the window"
    return ids[cut:], start - cut


def requests_of(row: dict) -> list[tuple[str, str]]:
    """(context, continuation) per option: alternative continuations (``choices``), alternative contexts with a
    shared continuation (``choice_contexts``, WinoGrande), or the single gold one."""
    if row.get("choice_contexts") is not None:
        return [(context, row["continuation"]) for context in row["choice_contexts"]]
    if row.get("choices") is not None:
        return [(row["context"], choice) for choice in row["choices"]]
    return [(row["context"], row["continuation"])]


def prepare_set(name: str, data_dir: Path, tokenizer: PreTrainedTokenizer, seq_len: int, bos: bool) -> EvalSet:
    """Load and tokenize one eval set; requests start with BOS if ``bos``."""
    meta = json.loads((data_dir / name / "meta.json").read_text())
    table = pq.read_table(data_dir / name / "test.parquet")
    kind, requests, context = meta["kind"], [], {}
    prefix = [tokenizer.bos_token_id] if bos else []
    if kind == "ppl":
        texts = table.column("text").to_pylist()
        encodings = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
        owner, num_bytes = [], []
        for doc, (text, text_ids) in enumerate(zip(texts, encodings["input_ids"])):
            ids = [*prefix, *text_ids, tokenizer.eos_token_id]
            for begin in range(0, len(ids) - 1, seq_len):
                requests.append((ids[begin : begin + seq_len + 1], 1))
                owner.append(doc)
            # Without BOS the first text token is only context.
            scored_text = text if bos else text[encodings["offset_mapping"][doc][0][1] :]
            num_bytes.append(len(scored_text.encode()))
        context = {"owner": owner, "bytes": np.array(num_bytes)}
    elif kind == "ctxcont":
        rows = table.to_pylist()
        options = [requests_of(row) for row in rows]
        requests = [
            truncate_left(*continuation_ids(tokenizer, ctx, cont, prefix), seq_len)
            for item in options
            for ctx, cont in item
        ]
        context = {"rows": rows, "options": options}
    elif kind == "pairs":
        good, bad = table.column("good").to_pylist(), table.column("bad").to_pylist()
        encoded = tokenizer(good + bad, add_special_tokens=False)["input_ids"]
        # Cut to the window like the other kinds, so every request fits in a row of ``token_budget`` tokens.
        requests = [([*prefix, *ids][: seq_len + 1], 1) for ids in encoded]
        context = {"num_good": len(good)}
    else:
        raise ValueError(f"Unknown eval set kind {kind!r} ({name})")
    return EvalSet(name, kind, table, requests, context, meta["parquet_sha256"])


@dataclass
class MultiLengthSet:
    """One eval set at several context lengths (``ppl`` window / ``ctxcont`` context limit). A request that is the
    same at two lengths (e.g. a document shorter than both windows) is scored once."""

    name: str
    versions: dict[int, EvalSet]
    requests: list[Request]
    """The distinct requests over all lengths."""
    index: dict[int, list[int]]
    """Per length: the position in ``requests`` of each of that version's requests."""


def prepare_multi_length_set(
    name: str, data_dir: Path, tokenizer: PreTrainedTokenizer, seq_lens: list[int], bos: bool
) -> MultiLengthSet:
    versions = {seq_len: prepare_set(name, data_dir, tokenizer, seq_len, bos) for seq_len in seq_lens}
    distinct: dict[tuple[tuple[int, ...], int], int] = {}
    index = {
        seq_len: [distinct.setdefault((tuple(ids), start), len(distinct)) for ids, start in version.requests]
        for seq_len, version in versions.items()
    }
    return MultiLengthSet(name, versions, [(list(ids), start) for ids, start in distinct], index)


def set_metrics(name: str, seq_len: int, summary: dict) -> dict[str, float]:
    """A set's summary as metrics ``eval/<set>/<metric>@<length>``, as Python floats (the summaries hold NumPy
    scalars)."""
    return {f"eval/{name}/{key}@{length_label(seq_len)}": float(value) for key, value in summary.items()}


def length_label(seq_len: int) -> str:
    """``2048`` -> ``2k``, the suffix of the metric names."""
    return f"{seq_len // 1024}k" if seq_len % 1024 == 0 else str(seq_len)


def summarize(eval_set: EvalSet, logprobs: list[np.ndarray], hits: list[np.ndarray]) -> tuple[dict, dict]:
    """Summary metrics and per-document / per-item arrays of one scored set."""
    if eval_set.kind == "ppl":
        return summarize_ppl(eval_set.context["bytes"], eval_set.context["owner"], logprobs)
    if eval_set.kind == "ctxcont":
        return summarize_ctxcont(eval_set.context["rows"], eval_set.context["options"], logprobs, hits)
    return summarize_pairs(eval_set.context["num_good"], logprobs)


def summarize_ppl(num_bytes: np.ndarray, owner: list[int], logprobs: list[np.ndarray]) -> tuple[dict, dict]:
    """``num_bytes``: the bytes of each document's scored text."""
    bins = list(zip(POSITION_BINS[:-1], POSITION_BINS[1:]))
    nll = np.zeros(len(num_bytes))
    tokens = np.zeros(len(num_bytes), dtype=np.int64)
    nll_bins = np.zeros((len(num_bytes), len(bins)))
    for doc, window in zip(owner, logprobs):
        nll[doc] -= window.sum()
        tokens[doc] += len(window)
        for b, (lo, hi) in enumerate(bins):
            nll_bins[doc, b] -= window[lo:hi].sum()
    summary = {
        "loss": nll.sum() / tokens.sum(),
        "bpb": nll.sum() / (num_bytes.sum() * math.log(2)),
        "docs": len(num_bytes),
        "tokens": int(tokens.sum()),
    }
    for b, (lo, hi) in enumerate(bins):
        bin_tokens = sum(max(0, min(hi, len(w)) - lo) for w in logprobs)
        if bin_tokens:
            summary[f"loss_pos{lo}-{hi}"] = nll_bins[:, b].sum() / bin_tokens
    return summary, {"nll": nll, "bytes": num_bytes, "tokens": tokens, "nll_bins": nll_bins}


def summarize_ctxcont(rows, options, logprobs, hits) -> tuple[dict, dict]:
    choice_logprob, choice_hits, choice_tokens = [], [], []
    offset = 0
    for item in options:
        item_logprobs = logprobs[offset : offset + len(item)]
        choice_logprob.append(np.array([lp.sum() for lp in item_logprobs]))
        choice_tokens.append([len(lp) for lp in item_logprobs])
        choice_hits.append(hits[offset : offset + len(item)])
        offset += len(item)
    gold = np.array([row["gold"] if len(item) > 1 else 0 for row, item in zip(rows, options)])
    nll = np.array([-choice_logprob[i][g] for i, g in enumerate(gold)])
    tokens = np.array([choice_tokens[i][g] for i, g in enumerate(gold)])
    greedy = np.array([choice_hits[i][g].all() for i, g in enumerate(gold)])
    greedy_tokens = np.array([choice_hits[i][g].sum() for i, g in enumerate(gold)])
    num_bytes = np.array([len(row["continuation"].encode()) for row in rows])
    summary = {
        "bpb": nll.sum() / (num_bytes.sum() * math.log(2)),
        "loss": nll.sum() / tokens.sum(),
        "greedy_em": greedy.mean(),
        "greedy_token_acc": greedy_tokens.sum() / tokens.sum(),
        "items": len(rows),
    }
    arrays = {"nll": nll, "bytes": num_bytes, "tokens": tokens, "greedy": greedy, "greedy_tokens": greedy_tokens}
    mc = np.array([len(item) > 1 for item in options])
    if mc.any():
        acc = np.array([lp.argmax() == g for lp, g in zip(choice_logprob, gold)]) & mc
        # length normalization only where the options differ in their continuation
        has_norm = mc & np.array([row.get("choice_contexts") is None for row in rows])
        acc_norm = np.zeros(len(rows), dtype=bool)
        for i in np.flatnonzero(has_norm):
            per_byte = choice_logprob[i] / [len(c.encode()) for _, c in options[i]]
            acc_norm[i] = per_byte.argmax() == gold[i]
        summary["acc"] = acc[mc].mean()
        if has_norm.any():
            summary["acc_norm"] = acc_norm[has_norm].mean()
        arrays |= {"acc": acc, "acc_norm": acc_norm, "mc": mc}
    return summary, arrays


def summarize_pairs(num_good: int, logprobs) -> tuple[dict, dict]:
    sums = np.array([lp.sum() for lp in logprobs])
    good_logprob, bad_logprob = sums[:num_good], sums[num_good:]
    correct = good_logprob > bad_logprob
    summary = {"acc": correct.mean(), "items": num_good}
    return summary, {"good_logprob": good_logprob, "bad_logprob": bad_logprob, "acc": correct}
