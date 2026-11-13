"""Convert Typed Decisions Synth (n4ze3m/typed-decisions-synth; MIT) into System One records.

    uv run python examples/decision_models/data/convert_typed_decisions_synth.py DATASET OUT

DATASET is the downloaded dataset directory. Writes `OUT/default/<split>.parquet` for the splits train, validation and
train_hard, and the dataset card that names them, so prime-rl loads them with `[data] name = OUT,
subsets = ["default"], splits = [<split>]`.

Each case (a state and its 1 to 8 questions, written and labelled by DeepSeek V4.1 Flash) becomes one record: `gold`
(an option, a level, or true/false per question) becomes `label`, and `teacher` (the mean of three answers'
probabilities; a noul gives P(true)) becomes `soft_label`. `train_hard` is the training records without their soft
labels, to train on the hard labels alone. Every record is parsed by prime-rl, as training will read it, and every
question must have both labels.
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, Features, Value

from prime_rl.configs.sft import SystemOneColumnsConfig
from prime_rl.trainer.sft.system_one.record import parse_record

SUBSET = "default"
# Every split has the same string columns: train_hard's soft_label is null in every row.
FEATURES = Features({column: Value("string") for column in ("id", "state", "questions", "label", "soft_label")})


def soft_label(teacher: dict) -> dict[str, float]:
    """A teacher answer as option -> probability, renormalized (the means are rounded)."""
    if "probabilities" in teacher:
        probabilities = teacher["probabilities"]
    else:
        probabilities = {"false": 1 - teacher["noul"], "true": teacher["noul"]}
    total = sum(probabilities.values())
    return {option: p / total for option, p in probabilities.items()}


def to_record(row: dict) -> dict:
    teacher = json.loads(row["teacher"])
    record = {
        "id": row["state_id"],
        "state": row["state"] if row["state_is_json"] else json.dumps(row["state"]),
        "questions": row["questions"],
        "label": row["gold"],
        "soft_label": json.dumps({question_id: soft_label(answer) for question_id, answer in teacher.items()}),
    }
    _, labels = parse_record(record, SystemOneColumnsConfig())
    if any(given.hard is None or given.soft is None for given in labels):
        raise ValueError(f"record {record['id']!r}: a question has no gold or no teacher answer")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path, help="the typed-decisions-synth dataset directory")
    parser.add_argument("out", type=Path, help="the dataset directory to write")
    args = parser.parse_args()

    splits = {}
    for split in ("train", "validation"):
        with (args.dataset / "data" / f"{split}.jsonl").open() as file:
            splits[split] = [to_record(json.loads(line)) for line in file]
    splits["train_hard"] = [{**record, "soft_label": None} for record in splits["train"]]

    (args.out / SUBSET).mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        Dataset.from_list(records, features=FEATURES).to_parquet(str(args.out / SUBSET / f"{split}.parquet"))
        questions = sum(len(json.loads(record["questions"])) for record in records)
        print(f"{split}: {len(records)} records, {questions} questions")
    # The card names the splits: without it, train_hard.parquet is read as a part of train.
    data_files = "".join(f"  - split: {split}\n    path: {SUBSET}/{split}.parquet\n" for split in splits)
    (args.out / "README.md").write_text(
        f"---\nconfigs:\n- config_name: {SUBSET}\n  data_files:\n{data_files}license: mit\n---\n"
        "System One records converted from Typed Decisions Synth (n4ze3m/typed-decisions-synth; MIT). `train_hard` is "
        "`train` without the soft labels.\n"
    )


if __name__ == "__main__":
    main()
