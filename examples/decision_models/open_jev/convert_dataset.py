"""Convert a subset of Open-Jev's dataset (one compiled question per row) into System One records.

    PYTHONPATH=examples uv run python examples/decision_models/open_jev/convert_dataset.py OPEN_JEV OUT

OPEN_JEV is the downloaded dataset directory. Writes `OUT/<subset>/<split>.parquet` for every split and the dataset
card that names them, so prime-rl loads them with `[data] name = OUT, subsets = [<subset>], splits = [<split>]`.

Each row becomes a record with one question: Open-Jev's `group_id` groups questions for splitting, not requests
(questions of a group can have different states and repeat ids). Every record is parsed by prime-rl and checked to
compile back to the question, options and target of its row, so training on the records reads Open-Jev's prompts.
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from decision_models.open_jev.encoder import compile_question

from prime_rl.configs.sft import SystemOneColumnsConfig
from prime_rl.trainer.sft.system_one.record import Question, parse_record

SPLITS = ("train", "validation", "calibration", "test", "ood")


def criteria(kind: str, options: list[str]) -> dict | list | None:
    """Open-Jev's compiled options -> System One criteria. A choice option is "name: description" or "name"."""
    if kind == "noul":
        return None  # any true/false descriptions are already part of the question text
    if kind == "score":
        return list(options)
    return dict(option.split(": ", 1) if ": " in option else (option, None) for option in options)


def to_record(row: dict) -> dict:
    question_id = json.loads(row["metadata_json"]).get("question_id") or row["id"]
    question = Question(question_id, row["kind"], row["question"], criteria(row["kind"], row["options"]))
    definition = {"type": question.type, "instructions": question.instructions}
    if question.criteria is not None:
        definition["criteria"] = question.criteria
    record = {
        "id": row["id"],
        "state": row["state_json"],
        "questions": json.dumps({question.id: definition}, ensure_ascii=False),
        # Open-Jev's target follows its compiled options, which is the order of the question's options (noul: no, yes).
        "soft_label": json.dumps({question.id: dict(zip(question.options, row["target"], strict=True))}),
        "source": row["source"],
        "group_id": row["group_id"],
        "metadata": row["metadata_json"],
    }
    parsed, (labels,) = parse_record(record, SystemOneColumnsConfig())
    compiled = (*compile_question(parsed.questions[0]), labels.soft)
    if compiled != (row["question"], list(row["options"]), list(row["target"])):
        raise ValueError(f"row {row['id']!r} does not compile back to itself")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("open_jev", type=Path, help="the Open-Jev dataset directory")
    parser.add_argument("out", type=Path, help="the dataset directory to write")
    parser.add_argument("--subset", default="release-v2-redistributable")
    args = parser.parse_args()

    (args.out / args.subset).mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (source,) = sorted((args.open_jev / "data" / args.subset).glob(f"{split}-*.parquet"))
        records = [to_record(row) for row in pq.read_table(source).to_pylist()]
        Dataset.from_list(records).to_parquet(str(args.out / args.subset / f"{split}.parquet"))
        print(f"{split}: {len(records)} records")
    data_files = "".join(f"  - split: {split}\n    path: {args.subset}/{split}.parquet\n" for split in SPLITS)
    (args.out / "README.md").write_text(
        f"---\nconfigs:\n- config_name: {args.subset}\n  data_files:\n{data_files}license: other\n---\n"
        f"System One records (one question and its soft label per row) converted from Open-Jev `{args.subset}`. "
        "Licenses: those of the Open-Jev dataset.\n"
    )


if __name__ == "__main__":
    main()
