"""Check a decision model against vLLM and a labelled dataset. docs/development.md, "Adding a Decision Model", has the
steps and the limits.

    # System One records of a public dataset, as train, validation and test files that SFT reads
    uv run python scripts/decision_parity.py data --dataset typed-decisions --out data/typed-decisions

    # The trainer's answers: a release, or the export of a run. More processes for a model that one GPU does not hold:
    # each answers every record and rank 0 writes the file
    uv run torchrun --nproc-per-node 1 scripts/decision_parity.py trainer --model MODEL \\
        --records data/typed-decisions/test.jsonl --out trainer.jsonl

    # vLLM's answers for the same checkpoint, from the engine's port. serve.toml has enable_prefix_caching = false under
    # [vllm]
    uv run inference @ serve.toml --vllm.model MODEL
    uv run python scripts/decision_parity.py served --url http://localhost:8100 \\
        --records data/typed-decisions/test.jsonl --out served.jsonl

    uv run python scripts/decision_parity.py compare trainer.jsonl served.jsonl
    uv run python scripts/decision_parity.py compare trainer-release.jsonl trainer-export.jsonl --changed
    uv run python scripts/decision_parity.py score trainer.jsonl --records data/typed-decisions/test.jsonl

An answers file has a line per record: its prompt's tokens and, per question, its type and each option's probability,
or ``"answers": null`` for a record that is skipped or refused.
"""

import argparse
import base64
import json
import math
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import fmean

MAX_KL = 5e-4
"""Mean KL(vLLM || trainer) per question type, the direction of the trainer's mismatch_kl. 3.7x the largest clean value
measured."""
MIN_AGREEMENT = 0.998
"""Share of the decided questions, of every type, whose top option is the same."""
DECIDED = 0.04
"""A question is decided if both sides' top option leads by more than this: kernels alone can swap a closer pair."""
MIN_CHANGE = 10 * MAX_KL
"""Mean KL between two sets of weights that must differ."""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


# data


def typed_decisions(split: str) -> list[dict]:
    """LocalLLaMA/typed-decisions (Apache-2.0): five questions about a JSON state, labelled with probabilities."""
    from datasets import load_dataset

    revision = "93e940ed20f3671d74b4bdc15e49050355fa397a"
    records = []
    for row in load_dataset("LocalLLaMA/typed-decisions", "all", split=split, revision=revision):
        soft_label = {key: gold["probabilities"] for key, gold in json.loads(row["gold"]).items()}
        records.append({key: row[key] for key in ("id", "state", "questions")} | {"soft_label": json.dumps(soft_label)})
    return records


def clevr4(split: str) -> list[dict]:
    """The CLEVR-4 part of IamBusy/OpenJev-Vision-Research-v0.1 (CC-BY-4.0): five questions about an image."""
    from datasets import Image, load_dataset

    revision = "43e4918445d82ae780e65b0ad08b9d4345547004"
    rows = load_dataset("IamBusy/OpenJev-Vision-Research-v0.1", "clevr4", split=split, revision=revision)
    records = []
    for row in rows.cast_column("image", Image(decode=False)):
        questions, label = {}, {}
        for question in json.loads(row["questions_json"]):
            criteria = dict.fromkeys(question["criteria"]) if question["type"] == "choice" else None
            questions[question["id"]] = {"type": question["type"], "instructions": question["instructions"]}
            if criteria:
                questions[question["id"]]["criteria"] = criteria
            label[question["id"]] = bool(question["target"]) if question["type"] == "noul" else question["target"]
        url = f"data:image/png;base64,{base64.b64encode(row['image']['bytes']).decode()}"
        records.append(
            {
                "id": row["id"],
                "state": json.dumps("See the image."),
                "questions": json.dumps(questions),
                "label": json.dumps(label),
                "images": json.dumps([url]),
            }
        )
    return records


DATASETS = {"typed-decisions": typed_decisions, "clevr4": clevr4}


def reversed_keys(value):
    if isinstance(value, dict):
        return {key: reversed_keys(child) for key, child in reversed(value.items())}
    return [reversed_keys(child) for child in value] if isinstance(value, list) else value


def reordered(record: dict) -> dict:
    """The record without its labels, the keys of its state and questions reversed. The datasets' keys are sorted, which
    hides a side that sorts them."""
    texts = {key: json.dumps(reversed_keys(json.loads(record[key]))) for key in ("state", "questions")}
    images = {"images": record["images"]} if "images" in record else {}
    return {"id": f"{record['id']}-reordered"} | texts | images


def data(args: argparse.Namespace) -> None:
    """Every tenth training record is for validation, and the test records get reordered copies of every fourth."""
    train, test = DATASETS[args.dataset]("train"), DATASETS[args.dataset]("test")
    splits = {
        "train": [record for i, record in enumerate(train) if i % 10],
        "validation": train[::10],
        "test": test + [reordered(record) for record in test[::4]],
    }
    for split, records in splits.items():
        write_jsonl(args.out / f"{split}.jsonl", records)
        print(f"{split}: {len(records)} records")


# answers


def trainer(args: argparse.Namespace) -> None:
    """Each record's answers as the trainer's forward gives them, alone in a row."""
    import torch

    from prime_rl.configs.sft import SystemOneColumnsConfig
    from prime_rl.configs.trainer import ModelConfig, TokenizerConfig
    from prime_rl.trainer.ckpt import setup_model_files
    from prime_rl.trainer.model import setup_model, setup_tokenizer
    from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
    from prime_rl.trainer.sft.system_one.data import render_sample
    from prime_rl.trainer.sft.system_one.forward import decision_forward
    from prime_rl.trainer.sft.system_one.loss import system_one_collate
    from prime_rl.trainer.sft.system_one.record import parse_record
    from prime_rl.trainer.sft.system_one.strategies import get_system_one_strategy_cls
    from prime_rl.trainer.utils import setup_torch_distributed
    from prime_rl.trainer.vlm import setup_processor
    from prime_rl.trainer.world import get_world
    from prime_rl.utils.logger import setup_logger

    setup_logger("info")
    setup_torch_distributed()
    torch.set_float32_matmul_precision("high")
    config = ModelConfig(name=args.model, **({"vlm": {}} if args.vlm else {}))
    resolve_ep(config)
    tokenizer = setup_tokenizer(TokenizerConfig(name=args.model), args.model)
    processor = setup_processor(config) if args.vlm else None
    model_files = setup_model_files(config, tokenizer, processor)
    strategy_cls = get_system_one_strategy_cls(model_files.config, model_files.selection)
    model = setup_model(config, get_parallel_dims(config))
    strategy = strategy_cls(tokenizer, model.config, *([processor] if args.vlm else []))
    logits = []
    read = strategy.option_logits
    strategy.option_logits = lambda *inputs: logits.append(read(*inputs)) or logits[-1]

    results = []
    for row in read_jsonl(args.records):
        request, _ = parse_record(row, SystemOneColumnsConfig())
        sample = render_sample(strategy, request, args.seq_len)
        if sample is None:
            results.append({"id": request.id, "answers": None})
            continue
        num_tokens = len(sample["input_ids"])
        row = sample | {"sample_lens": [num_tokens], "sample_data": [sample["sample_data"]]}
        batch = system_one_collate([row])
        with torch.no_grad():
            decision_forward(model, batch, strategy=strategy, brier_weight=0.0, device="cuda")
        read_questions = batch["decision_loss_inputs"].questions
        split = logits.pop().float().split([len(question.options) for question in read_questions])
        answers = {}
        for question, definition, options in zip(read_questions, request.questions, split, strict=True):
            probabilities = options.double().softmax(-1).tolist()
            answers[definition.id] = {"type": definition.type, "probs": dict(zip(question.options, probabilities))}
        results.append({"id": request.id, "tokens": num_tokens, "answers": answers})
    if get_world().rank == 0:
        write_jsonl(args.out, results)
    torch.distributed.destroy_process_group()


def served(args: argparse.Namespace) -> None:
    """Each record's answers from vLLM's ``/v1/systemone``, one request at a time. The engine's port, not the router's:
    a router that reorders a request's keys changes the prompt."""
    results = []
    for row in read_jsonl(args.records):
        questions = json.loads(row["questions"])
        body = {"state": json.loads(row["state"]), "questions": questions}
        if row.get("images"):
            body["images"] = json.loads(row["images"])
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                response = json.load(response)
        except urllib.error.HTTPError as error:
            # 400 refuses the record; anything else is a broken server
            if error.code != 400:
                raise
            print(f"{row['id']}: {error.read().decode()}", file=sys.stderr)
            results.append({"id": row["id"], "answers": None})
            continue
        answers = {}
        for key, definition in questions.items():
            probs = response["answers"][key]["probabilities"]
            if definition["type"] == "noul":
                probs = {"false": probs["no"], "true": probs["yes"]}
            answers[key] = {"type": definition["type"], "probs": probs}
        results.append({"id": row["id"], "tokens": response["usage"]["input_tokens"], "answers": answers})
    write_jsonl(args.out, results)


# compare and score


def kl(p: dict[str, float], q: dict[str, float]) -> float:
    return sum(value * math.log(value / max(q[option], 1e-12)) for option, value in p.items() if value > 0)


def lead(p: dict[str, float]) -> float:
    top = sorted(p.values(), reverse=True)
    return top[0] - top[1] if len(top) > 1 else 1.0


def top(p: dict[str, float]) -> str:
    return max(p, key=p.get)


def compare_answers(a: list[dict], b: list[dict], changed: bool = False) -> tuple[list[str], list[str]]:
    """The comparison as a Markdown table's lines, and what fails. With ``changed``, two checkpoints on one side that
    must differ."""
    a, b = {row["id"]: row for row in a}, {row["id"]: row for row in b}
    failures = []
    if a.keys() != b.keys():
        failures.append(f"{len(a.keys() ^ b.keys())} records are in one file only")
    pairs: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for key in a.keys() & b.keys():
        left, right = a[key], b[key]
        if (left["answers"] is None) != (right["answers"] is None):
            failures.append(f"record {key!r} is answered on one side only")
            continue
        if left["answers"] is None:
            continue
        if left["tokens"] != right["tokens"]:
            failures.append(f"record {key!r} has {left['tokens']} prompt tokens against {right['tokens']}")
        if left["answers"].keys() != right["answers"].keys():
            failures.append(f"record {key!r} has other questions")
            continue
        for question, answer in left["answers"].items():
            p, q = answer["probs"], right["answers"][question]["probs"]
            if p.keys() != q.keys() or not all(math.isfinite(value) for value in (*p.values(), *q.values())):
                failures.append(f"record {key!r}, question {question!r}: other options, or not finite")
                continue
            pairs[answer["type"]].append((p, q))
    if not pairs:
        return [], [*failures, "no question is answered on both sides"]

    lines = [
        "| Questions | Count | Mean KL | Largest difference | Decided | Same top option, of the decided |",
        "|---|---|---|---|---|---|",
    ]
    for name in [*sorted(pairs), "all"]:
        rows = [pair for group in pairs.values() for pair in group] if name == "all" else pairs[name]
        mean_kl = fmean(kl(q, p) for p, q in rows)
        largest = max(abs(p[option] - q[option]) for p, q in rows for option in p)
        decided = [(p, q) for p, q in rows if min(lead(p), lead(q)) > DECIDED]
        agreement = fmean(top(p) == top(q) for p, q in decided) if decided else 0.0
        share = len(decided) / len(rows)
        lines.append(f"| {name} | {len(rows)} | {mean_kl:.1e} | {largest:.3f} | {share:.1%} | {agreement:.2%} |")
        if changed:
            if name == "all" and mean_kl < MIN_CHANGE:
                failures.append(f"mean KL {mean_kl:.1e} is under {MIN_CHANGE:.0e}: the weights barely differ")
            continue
        if mean_kl > MAX_KL:
            failures.append(f"{name}: mean KL {mean_kl:.1e} is over {MAX_KL:.0e}")
        # A model that decides nothing agrees with anything
        if len(decided) < len(rows) / 2:
            failures.append(f"{name}: only {len(decided)} of {len(rows)} questions are decided")
        elif name == "all" and agreement < MIN_AGREEMENT:
            failures.append(f"{agreement:.2%} of the decided questions agree, under {MIN_AGREEMENT:.1%}")
    return lines, failures


def compare(args: argparse.Namespace) -> None:
    lines, failures = compare_answers(read_jsonl(args.a), read_jsonl(args.b), args.changed)
    print("\n".join(lines))
    for failure in failures[:20]:
        print(f"FAIL {failure}")
    if failures:
        sys.exit(f"{len(failures)} failures")
    print("PASS")


def score(args: argparse.Namespace) -> None:
    """Accuracy (any top option of a tied label counts), KL from the labels and Brier score, per labelled question. One
    that is not answered counts as uniform."""
    answers = {row["id"]: row["answers"] for row in read_jsonl(args.answers)}
    scores, missing = [], 0
    for row in read_jsonl(args.records):
        questions = json.loads(row["questions"])
        soft = json.loads(row.get("soft_label") or "{}")
        hard = json.loads(row.get("label") or "{}")
        for key, definition in questions.items():
            if definition["type"] == "noul":
                options = ["false", "true"]
            elif definition["type"] == "choice":
                options = list(definition["criteria"])
            else:
                options = [str(level) for level in range(len(definition["criteria"]))]
            if key in soft:
                label = {option: soft[key].get(option, 0.0) for option in options}
            elif key in hard:
                value = str(hard[key]).lower() if definition["type"] == "noul" else str(hard[key])
                label = {option: float(option == value) for option in options}
            else:
                continue
            given = answers.get(row["id"])
            missing += given is None
            probs = given[key]["probs"] if given else dict.fromkeys(options, 1 / len(options))
            brier = sum((probs[option] - label[option]) ** 2 for option in options)
            correct = label[top(probs)] >= max(label.values()) - 1e-9
            scores.append((correct, kl(label, probs), brier))
    accuracy, mean_kl, brier = (fmean(values) for values in zip(*scores))
    print("| Questions | Not answered | Accuracy | KL from the labels | Brier |\n|---|---|---|---|---|")
    print(f"| {len(scores)} | {missing} | {accuracy:.3f} | {mean_kl:.3f} | {brier:.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)
    command = commands.add_parser("data")
    command.add_argument("--dataset", choices=DATASETS, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(run=data)
    command = commands.add_parser("trainer")
    command.add_argument("--model", required=True)
    command.add_argument("--records", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--seq-len", type=int, default=16384)
    command.add_argument("--vlm", action="store_true", help="read the records' images")
    command.set_defaults(run=trainer)
    command = commands.add_parser("served")
    command.add_argument("--url", default="http://localhost:8100", help="the engine's port, not the router's")
    command.add_argument("--records", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(run=served)
    command = commands.add_parser("compare")
    command.add_argument("a", type=Path)
    command.add_argument("b", type=Path)
    command.add_argument("--changed", action="store_true", help="two checkpoints that must differ")
    command.set_defaults(run=compare)
    command = commands.add_parser("score")
    command.add_argument("answers", type=Path)
    command.add_argument("--records", type=Path, required=True)
    command.set_defaults(run=score)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
