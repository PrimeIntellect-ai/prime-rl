"""In-trainer base-model evaluations of nanochat (karpathy/nanochat @ ff46300): validation bits per byte
(``nanochat/loss_eval.py:evaluate_bpb``) and the CORE metric (``nanochat/core_eval.py``, ``scripts/base_eval.py``).

Hosted pretraining runs the trainer only, so both run inside it. CORE packs the example sequences of a rank into
fixed-size varlen rows (documents never attend to each other, positions restart at every document), which gives
the same per-sequence losses and argmax predictions as nanochat's one-example-at-a-time padded batches.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.tensor import DTensor


# ----------------------------------------------------------------------------------------------------------------------
# bits per byte


def build_token_bytes(tokenizer, vocab_size: int) -> Tensor:
    """Bytes of every token id as nanochat's ``tok_train.py`` counts them: ``len(decode([id]).encode())``, 0 for
    special tokens. Decoded without whitespace cleanup, token by token."""
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    backend = getattr(tokenizer, "backend_tokenizer", None)
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64)
    for token_id in range(vocab_size):
        if token_id in special:
            continue
        if backend is not None:
            text = backend.decode([token_id], skip_special_tokens=False)
        else:
            text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        token_bytes[token_id] = len(text.encode("utf-8"))
    return token_bytes


def bpb_from_sums(total_nats: float, total_bytes: int) -> float:
    return float("inf") if total_bytes == 0 else total_nats / (math.log(2) * total_bytes)


# ----------------------------------------------------------------------------------------------------------------------
# CORE: prompt rendering (verbatim templates of nanochat/core_eval.py)


def _template(source: str):
    from jinja2 import Template

    return Template(source)


MC_TEMPLATE = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()

SCHEMA_TEMPLATE = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()

LM_TEMPLATE = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()


def render_prompts(task_type: str, item: dict, delimiter: str, fewshot: list[dict]) -> list[str]:
    context = {"fewshot_examples": fewshot, "continuation_delimiter": delimiter, "item": item}
    if task_type == "multiple_choice":
        template = _template(MC_TEMPLATE)
        return [template.render(choice=choice, **context) for choice in item["choices"]]
    if task_type == "schema":
        template = _template(SCHEMA_TEMPLATE)
        return [template.render(context=option, **context) for option in item["context_options"]]
    if task_type == "language_modeling":
        template = _template(LM_TEMPLATE)
        without = template.render(include_continuation=False, **context).strip()
        with_ = template.render(include_continuation=True, **context)
        return [without, with_]
    raise ValueError(f"Unsupported task type: {task_type}")


def find_common_length(token_sequences: list[list[int]], direction: str = "left") -> int:
    min_len = min(len(seq) for seq in token_sequences)
    indices = {"left": range(min_len), "right": range(-1, -min_len - 1, -1)}[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def example_sequences(task_type: str, prompts: list[str], encode: Callable[[str], list[int]], bos: int):
    """Token sequences (BOS prepended) and the [start, end) span of the scored tokens of each, as nanochat's
    ``batch_sequences_*``."""
    tokens = [[bos] + encode(prompt) for prompt in prompts]
    if task_type == "multiple_choice":
        start = find_common_length(tokens, "left")
        return tokens, [start] * len(tokens), [len(t) for t in tokens]
    if task_type == "schema":
        suffix = find_common_length(tokens, "right")
        ends = [len(t) for t in tokens]
        return tokens, [e - suffix for e in ends], ends
    without, with_ = tokens
    start, end = len(without), len(with_)
    assert start < end and without == with_[:start], "prompt without continuation must prefix the full prompt"
    return [with_], [start], [end]


def score_example(task_type: str, gold: int | None, losses: list[Tensor], preds: list[Tensor], tokens, starts, ends):
    """Same decision rule as nanochat ``evaluate_example``. ``losses[i][t]``/``preds[i][t]`` are the loss of and the
    argmax prediction for token ``t + 1`` of sequence ``i``."""
    if task_type == "language_modeling":
        s, e = starts[0], ends[0]
        predicted = preds[0][s - 1 : e - 1]
        actual = torch.tensor(tokens[0][s:e], device=predicted.device)
        return bool(torch.all(predicted == actual).item())
    mean_losses = [losses[i][s - 1 : e - 1].mean().item() for i, (s, e) in enumerate(zip(starts, ends))]
    return mean_losses.index(min(mean_losses)) == gold


def load_core_tasks(bundle_dir: str, max_per_task: int) -> tuple[list[dict], dict[str, float]]:
    import yaml

    if not os.path.exists(os.path.join(bundle_dir, "core.yaml")) and os.path.exists(
        os.path.join(bundle_dir, "eval_bundle", "core.yaml")
    ):
        bundle_dir = os.path.join(bundle_dir, "eval_bundle")
    with open(os.path.join(bundle_dir, "core.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    baselines = {}
    with open(os.path.join(bundle_dir, "eval_meta_data.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            baselines[row["Eval Task"]] = float(row["Random baseline"])
    tasks = []
    for task in config["icl_tasks"]:
        with open(os.path.join(bundle_dir, "eval_data", task["dataset_uri"]), encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]
        random.Random(1337).shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]
        tasks.append(
            dict(
                label=task["label"],
                task_type=task["icl_task_type"],
                num_fewshot=task["num_fewshot"][0],
                delimiter=task.get("continuation_delimiter", " "),
                data=data,
            )
        )
    return tasks, baselines


def build_requests(tasks: list[dict], rank: int, world_size: int, encode, bos: int) -> list[dict]:
    """The examples of this rank (strided, as nanochat), rendered and tokenized."""
    requests = []
    for task_index, task in enumerate(tasks):
        data = task["data"]
        for idx in range(rank, len(data), world_size):
            item = data[idx]
            fewshot = []
            if task["num_fewshot"] > 0:
                rng = random.Random(1234 + idx)
                available = [i for i in range(len(data)) if i != idx]
                fewshot = [data[i] for i in rng.sample(available, task["num_fewshot"])]
            prompts = render_prompts(task["task_type"], item, task["delimiter"], fewshot)
            tokens, starts, ends = example_sequences(task["task_type"], prompts, encode, bos)
            requests.append(
                dict(task=task_index, idx=idx, task_type=task["task_type"], gold=item.get("gold"), tokens=tokens,
                     starts=starts, ends=ends)
            )
    return requests


def pack_requests(requests: list[dict], pack_tokens: int) -> list[list[tuple[int, int]]]:
    """Greedy packing of (request, sequence) pairs into rows of at most ``pack_tokens`` tokens; a request's
    sequences stay in one row. Longer requests get a row of their own (padded up to a multiple of pack_tokens)."""
    packs, current, used = [], [], 0
    for r, request in enumerate(requests):
        size = sum(len(t) for t in request["tokens"])
        if current and used + size > pack_tokens:
            packs.append(current)
            current, used = [], 0
        current.append(r)
        used += size
    if current:
        packs.append(current)
    return packs


def run_core(
    forward_logits: Callable[[Tensor, Tensor, Tensor], Tensor],
    tokenizer,
    bundle_dir: str,
    max_per_task: int,
    device: torch.device,
    pack_tokens: int = 16384,
    log: Callable[[str], None] = print,
) -> dict:
    """CORE metric. ``forward_logits(input_ids [1, T], position_ids [1, T], seq_lens [n])`` returns float logits
    [T, V] (softcapped). Every rank runs the same number of forwards."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    start_time = time.time()
    tasks, baselines = load_core_tasks(bundle_dir, max_per_task)
    bos = tokenizer.bos_token_id

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    requests = build_requests(tasks, rank, world_size, encode, bos)
    packs = pack_requests(requests, pack_tokens)
    num_packs = torch.tensor(len(packs), device=device)
    if world_size > 1:
        dist.all_reduce(num_packs, op=dist.ReduceOp.MAX)
    log(f"CORE: {len(requests)} examples in {len(packs)} packs on rank {rank} (prep {time.time() - start_time:.1f}s)")
    correct = [torch.zeros(len(task["data"]), dtype=torch.float32, device=device) for task in tasks]
    for p in range(int(num_packs.item())):
        members = packs[p] if p < len(packs) else []
        sequences = [t for r in members for t in requests[r]["tokens"]]
        total = sum(len(t) for t in sequences)
        row = max(pack_tokens, -(-total // pack_tokens) * pack_tokens)
        if total < row:
            sequences_padded = sequences + [[bos] * (row - total)]
        else:
            sequences_padded = sequences
        input_ids = torch.tensor([t for s in sequences_padded for t in s], dtype=torch.long, device=device)[None]
        positions = torch.tensor([i for s in sequences_padded for i in range(len(s))], device=device)[None]
        seq_lens = torch.tensor([len(s) for s in sequences_padded], dtype=torch.int32, device=device)
        with torch.no_grad():
            logits = forward_logits(input_ids, positions, seq_lens)
            targets = torch.roll(input_ids[0], -1)
            losses = F.cross_entropy(logits, targets, reduction="none")
            preds = logits.argmax(dim=-1)
        offset = 0
        for r in members:
            request = requests[r]
            seq_losses, seq_preds = [], []
            for t in request["tokens"]:
                seq_losses.append(losses[offset : offset + len(t)])
                seq_preds.append(preds[offset : offset + len(t)])
                offset += len(t)
            ok = score_example(request["task_type"], request["gold"], seq_losses, seq_preds, request["tokens"],
                               request["starts"], request["ends"])
            correct[request["task"]][request["idx"]] = float(ok)
    results, centered = {}, {}
    for task, task_correct in zip(tasks, correct):
        if world_size > 1:
            dist.all_reduce(task_correct, op=dist.ReduceOp.SUM)
        accuracy = task_correct.mean().item()
        baseline = baselines[task["label"]]
        results[task["label"]] = accuracy
        centered[task["label"]] = (accuracy - 0.01 * baseline) / (1.0 - 0.01 * baseline)
    core = sum(centered.values()) / len(centered)
    log(f"CORE {core:.4f} in {time.time() - start_time:.1f}s")
    return {"results": results, "centered_results": centered, "core_metric": core}


def local_tensor(tensor: Tensor) -> Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
