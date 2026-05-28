from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            if is_dataclass(row):
                payload = asdict(row)
            else:
                payload = row
            file.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
            count += 1
    return count


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=True)
        file.write("\n")
