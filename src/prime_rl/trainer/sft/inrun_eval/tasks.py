"""The eval's task list: which task belongs to which collection run (source) and to which data split.

The task table (``tasks.tsv`` in the eval bundle; private task names never go into this repository) has one line per
task: ``<name>\\t<source>\\t<split>``, source one of ``tb-eng``, ``tb-eng-extra9``, ``tb-sci`` and split one of
``val`` (the dataset's held-out tasks: ``data-prep/tb4-val-tasks-v2.txt``, the 44 frozen ones plus 6 zero-pass
tasks, 50 = 10%), ``train`` (tasks with training rows) or ``other``.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

SOURCES = ("tb-eng", "tb-eng-extra9", "tb-sci")
SPLITS = ("val", "train", "other")


@dataclass(frozen=True)
class TaskRow:
    name: str
    source: str
    split: str


def read_names(path: Path) -> list[str]:
    """One task name per line; blank lines and ``#`` comments are skipped."""
    names = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def parse_task_table(text: str) -> list[TaskRow]:
    rows: list[TaskRow] = []
    seen: set[str] = set()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            raise ValueError(f"tasks.tsv line {number}: expected name<TAB>source<TAB>split, got {line!r}")
        name, source, split = (part.strip() for part in parts)
        if source not in SOURCES:
            raise ValueError(f"tasks.tsv line {number}: unknown source {source!r} (one of {SOURCES})")
        if split not in SPLITS:
            raise ValueError(f"tasks.tsv line {number}: unknown split {split!r} (one of {SPLITS})")
        if name in seen:
            raise ValueError(f"tasks.tsv line {number}: {name!r} listed twice")
        seen.add(name)
        rows.append(TaskRow(name, source, split))
    return rows


def select_tasks(table: list[TaskRow], sources: list[str], restrict: list[str] | None = None) -> dict[str, list[str]]:
    """The tasks each source evaluates: the table's tasks of that source, or only ``restrict``'s (every restricted
    name must be in the table). Sources left without tasks are dropped."""
    by_name = {row.name: row for row in table}
    if restrict is not None:
        unknown = sorted(set(restrict) - set(by_name))
        if unknown:
            raise ValueError(f"tasks not in the task table: {unknown[:5]}")
        wanted = set(restrict)
    else:
        wanted = set(by_name)
    out: dict[str, list[str]] = {}
    for source in sources:
        names = sorted(row.name for row in table if row.source == source and row.name in wanted)
        if names:
            out[source] = names
    return out


def slug(name: str) -> str:
    """The name part of a task-QC job id (prime_flow ``qc.solving.job_id`` at ed23e23)."""
    return re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")[:48]


JOB_ID = re.compile(r"^(?P<slug>.*?)-?(?P<key>[0-9a-f]{12})-(?P<what>[^-]+)$")


def job_slug(job_id: str) -> str | None:
    """The name slug of a job id ``<slug>-<key[:12]>-<draw>`` (e.g. ``...-697db0896cda-qc``), None if malformed."""
    match = JOB_ID.match(job_id)
    return match.group("slug") if match else None


def slug_index(names: list[str]) -> tuple[dict[str, str], set[str]]:
    """slug -> name for the names whose slug is unique, and the ambiguous slugs (their jobs need the trace)."""
    seen: dict[str, list[str]] = {}
    for name in names:
        seen.setdefault(slug(name), []).append(name)
    unique = {key: found[0] for key, found in seen.items() if len(found) == 1}
    return unique, {key for key, found in seen.items() if len(found) > 1}


def harbor_cache_name(dataset: str, repo: str | None, registry_path: str | None) -> str:
    """The directory name verifiers' Harbor taskset caches a dataset under (``~/.cache/harbor/<name>``), as
    ``verifiers/v1/tasksets/harbor/taskset.py:cache_dir`` computes it at 24107a4ea for a repo-hosted registry."""
    parts = [dataset]
    if repo is not None:
        parts += ["repo", repo]
    if registry_path is not None:
        if repo is None:
            raise ValueError("only repo-relative registry paths are supported (a local one resolves against disk)")
        parts += ["registry_path", registry_path]
    name = dataset.replace("/", "_").replace("@", "_")
    if len(parts) > 1:
        name = f"{name}_{hashlib.sha256(chr(0).join(parts).encode()).hexdigest()[:12]}"
    return name
