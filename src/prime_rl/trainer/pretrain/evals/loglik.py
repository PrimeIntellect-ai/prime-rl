import math
from dataclasses import dataclass

import torch
from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.pretrain import TaskName
from prime_rl.trainer.model import forward
from prime_rl.trainer.pretrain.evals.common import all_reduce_sum, local_rows, to_device
from prime_rl.trainer.pretrain.evals.tasks import TaskItem, load_task
from prime_rl.trainer.sft.data import Sample


@dataclass
class LoglikRow:
    """Requests packed into one row. ``spans`` are ``(request, start, end)``: the target
    positions of each request's continuation."""

    sample: Sample
    spans: list[tuple[int, int, int]]


@dataclass
class LoglikTask:
    name: str
    items: list[TaskItem]
    rows: list[LoglikRow]


def pack_requests(requests: list[tuple[list[int], list[int]]], seq_len: int) -> list[LoglikRow]:
    """Pack ``(context_ids, continuation_ids)`` requests into rows of ``seq_len`` tokens, one
    document-masked segment per request. Contexts are cut from the left to fit."""
    rows: list[LoglikRow] = []
    inputs: list[int] = []
    targets: list[int] = []
    seq_lens: list[int] = []
    spans: list[tuple[int, int, int]] = []

    def flush():
        pad = seq_len - len(inputs)
        sample: Sample = {
            "input_ids": inputs + [0] * pad,
            "target_ids": targets + [0] * pad,
            "loss_mask": [False] * seq_len,
            "position_ids": [p for length in seq_lens for p in range(length)] + list(range(pad)),
            "seq_lens": seq_lens + ([pad] if pad else []),
            "mm_kwargs": None,
            "mm_token_type_ids": None,
        }
        for _, start, end in spans:
            sample["loss_mask"][start:end] = [True] * (end - start)
        rows.append(LoglikRow(sample, list(spans)))
        inputs.clear()
        targets.clear()
        seq_lens.clear()
        spans.clear()

    for index, (context_ids, continuation_ids) in enumerate(requests):
        assert 0 < len(continuation_ids) <= seq_len
        tokens = (context_ids + continuation_ids)[-(seq_len + 1) :]
        length = len(tokens) - 1
        if len(inputs) + length > seq_len:
            flush()
        offset = len(inputs)
        inputs.extend(tokens[:-1])
        targets.extend(tokens[1:])
        seq_lens.append(length)
        spans.append((index, offset + length - len(continuation_ids), offset + length))
    if inputs:
        flush()
    return rows


def load_loglik_task(name: TaskName, tokenizer: PreTrainedTokenizer, seq_len: int, limit: int | None) -> LoglikTask:
    items = load_task(name, limit)
    requests = [
        (tokenizer(context)["input_ids"], tokenizer(continuation, add_special_tokens=False)["input_ids"])
        for item in items
        for context, continuation in item.requests
    ]
    return LoglikTask(name=name, items=items, rows=pack_requests(requests, seq_len))


def score_task(task: LoglikTask, logprobs: list[float]) -> dict[str, float]:
    """Accuracy (multiple-choice tasks), length-normalized accuracy (choices that differ in their
    continuation) and bits per byte of the gold continuation."""
    metrics: dict[str, float] = {}
    correct = correct_norm = gold_nll = gold_bytes = 0.0
    offset = 0
    for item in task.items:
        scores = logprobs[offset : offset + len(item.requests)]
        continuations = [continuation for _, continuation in item.requests]
        offset += len(item.requests)
        correct += max(range(len(scores)), key=scores.__getitem__) == item.gold
        # Normalized by the characters of the answer itself, without the leading delimiter space.
        normalized = [score / len(continuation.removeprefix(" ")) for score, continuation in zip(scores, continuations)]
        correct_norm += max(range(len(scores)), key=normalized.__getitem__) == item.gold
        gold_nll -= scores[item.gold]
        gold_bytes += len(continuations[item.gold].encode())
    num_choices = len(task.items[0].requests)
    if num_choices > 1:
        metrics[f"evals/{task.name}/acc"] = correct / len(task.items)
        if len({continuation for _, continuation in task.items[0].requests}) > 1:
            metrics[f"evals/{task.name}/acc_norm"] = correct_norm / len(task.items)
    metrics[f"evals/{task.name}/bpb"] = gold_nll / (math.log(2) * gold_bytes)
    return metrics


@torch.no_grad()
def evaluate_loglik(model: nn.Module, tasks: list[LoglikTask]) -> dict[str, float]:
    metrics = {}
    for task in tasks:
        num_requests = sum(len(item.requests) for item in task.items)
        sums = torch.zeros(num_requests, dtype=torch.float64, device="cuda")
        for row in local_rows(task.rows):
            batch = to_device((row or task.rows[0]).sample)
            temperature = torch.ones_like(batch["target_ids"], dtype=torch.float32)
            out = forward(
                model,
                batch["input_ids"],
                batch["position_ids"],
                seq_lens=batch["seq_lens"],
                labels=batch["target_ids"],
                temperature=temperature,
            )
            if out.get("logprobs") is not None:
                logprobs = out["logprobs"][0]
            else:
                logprobs = torch.log_softmax(out["logits"][0], dim=-1).gather(-1, batch["target_ids"][0, :, None])[:, 0]
            if row is None:
                continue
            for index, start, end in row.spans:
                sums[index] += logprobs[start:end].sum()
        metrics |= score_task(task, all_reduce_sum(sums).tolist())
    return metrics
