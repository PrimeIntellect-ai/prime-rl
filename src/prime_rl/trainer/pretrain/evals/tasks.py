"""Zero-shot log-likelihood tasks, formatted as in lm-evaluation-harness (EleutherAI, main).

Each choice is a (context, continuation) pair; the continuation carries the harness's default
target delimiter (a leading space)."""

import re
from collections.abc import Callable
from dataclasses import dataclass

from datasets import load_dataset

from prime_rl.configs.pretrain import TaskName


@dataclass(frozen=True)
class TaskItem:
    requests: list[tuple[str, str]]
    gold: int


def _cloze(context: str, choices: list[str], gold: int) -> TaskItem:
    return TaskItem(requests=[(context, " " + choice) for choice in choices], gold=gold)


def _hellaswag_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def _hellaswag(doc: dict) -> TaskItem:
    context = _hellaswag_preprocess(doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"].capitalize())
    return _cloze(context, [_hellaswag_preprocess(ending) for ending in doc["endings"]], int(doc["label"]))


def _arc(doc: dict) -> TaskItem:
    choices = doc["choices"]
    return _cloze(f"Question: {doc['question']}\nAnswer:", choices["text"], choices["label"].index(doc["answerKey"]))


def _piqa(doc: dict) -> TaskItem:
    return _cloze(f"Question: {doc['goal']}\nAnswer:", [doc["sol1"], doc["sol2"]], doc["label"])


def _winogrande(doc: dict) -> TaskItem:
    # Partial evaluation: the context differs per option, the scored continuation is shared.
    sentence = doc["sentence"]
    idx = sentence.index("_")
    continuation = " " + sentence[idx + 1 :].strip()
    requests = [(sentence[:idx] + option, continuation) for option in (doc["option1"], doc["option2"])]
    return TaskItem(requests=requests, gold=int(doc["answer"]) - 1)


def _openbookqa(doc: dict) -> TaskItem:
    choices = doc["choices"]
    return _cloze(doc["question_stem"], choices["text"], choices["label"].index(doc["answerKey"].lstrip()))


def _sciq(doc: dict) -> TaskItem:
    context = f"{doc['support'].lstrip()}\nQuestion: {doc['question']}\nAnswer:"
    return _cloze(context, [doc["distractor1"], doc["distractor2"], doc["distractor3"], doc["correct_answer"]], 3)


def _lambada(doc: dict) -> TaskItem:
    context, word = doc["text"].rsplit(" ", 1)
    return TaskItem(requests=[(context, " " + word)], gold=0)


# name -> (dataset path, config, eval split, adapter)
_TASKS: dict[str, tuple[str, str | None, str, Callable[[dict], TaskItem]]] = {
    "hellaswag": ("Rowan/hellaswag", None, "validation", _hellaswag),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test", _arc),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", _arc),
    "piqa": ("baber/piqa", None, "validation", _piqa),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation", _winogrande),
    "openbookqa": ("allenai/openbookqa", "main", "test", _openbookqa),
    "sciq": ("allenai/sciq", None, "test", _sciq),
    "lambada": ("EleutherAI/lambada_openai", "default", "test", _lambada),
}


def load_task(name: TaskName, limit: int | None = None) -> list[TaskItem]:
    """The first `limit` items (dataset order) of the task's eval split."""
    path, config, split, adapter = _TASKS[name]
    if limit is not None:
        split = f"{split}[:{limit}]"
    return [adapter(doc) for doc in load_dataset(path, config, split=split)]
