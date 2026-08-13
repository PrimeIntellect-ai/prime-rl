import json
from pathlib import Path

LABELS = [
    "SITUATIONAL GROUNDING",
    "CONCRETE TASK PROGRESS",
    "EVIDENCE DISCIPLINE",
    "NO OBVIOUS ISSUES",
    "NEEDS REVIEW",
    "PASS",
    "FAIL",
    "TRUE",
    "FALSE",
    "SUPPORTED",
    "UNSUPPORTED",
    "INCONCLUSIVE",
]
TEMPLATES = [
    "Audit label: {label}",
    "Final judgment: {label}",
    "Selected category: {label}",
    "Rubric result: {label}",
    "The applicable label is {label}.",
    "My determination is {label}.",
    "Recorded outcome: {label}",
    "Classification: {label}",
]


def rows(offset: int, count: int):
    for index in range(offset, offset + count):
        label = LABELS[index % len(LABELS)]
        response = TEMPLATES[(index // len(LABELS)) % len(TEMPLATES)].format(label=label)
        yield {
            "prompt": [{"role": "user", "content": "State the selected audit label exactly."}],
            "chosen": [{"role": "assistant", "content": response}],
            "rejected": [{"role": "assistant", "content": response.lower()}],
        }


root = Path(__file__).parent
for name, offset, count in (("train", 0, 96), ("validation", 96, 48)):
    with (root / f"{name}.jsonl").open("w") as file:
        for row in rows(offset, count):
            file.write(json.dumps(row) + "\n")
