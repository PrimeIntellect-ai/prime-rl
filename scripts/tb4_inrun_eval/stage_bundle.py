#!/usr/bin/env python3
"""Assemble the in-run TB4 eval bundle (``[inrun_eval.harness] bundle_dir``) in a local directory, check it, and
write its ``manifest.json``. Uploading it to the volume (``datasets/tb4-eval`` on nemotron-sft-tb4-v2) is a separate
step. Nothing private is in this script: every input comes from the command line (see PREREQS.md for where each one
lives).

    python scripts/tb4_inrun_eval/stage_bundle.py --out <dir> \\
        --sft-gen <sft-gen @ bca802f checkout> \\
        --flow-config tb-eng=<.../tb-engineering-glm-measure-20261023/configs/attempt_2/flow.toml> \\
        --flow-config tb-eng-extra9=<.../tb-engineering-extra9-glm-measure-20261023/configs/attempt_2/flow.toml> \\
        --flow-config tb-sci=<.../tb-science-glm-measure-20261023/configs/attempt_2/flow.toml> \\
        --tasks-tsv <tasks.tsv> | (--collection-traces <traces.jsonl> --val-tasks <tb4-val-tasks.txt>
                                   --train-dataset <dataset dir> | --train-tasks <file>  [--extra-task name=source]) \\
        --harbor-cache <~/.cache/harbor/<name>> [...] \\
        --kernel <landlock-6.12.113-deb13.tgz> \\
        --requirements <requirements.txt> [--wheels <dir> [--wheels-complete]] \\
        [--python-tarball <python-build-standalone .tar.gz>] \\
        [--frpc <dir with frpc and .frpc_version>] [--smoke-tasks <file>] [--tb-sci-dataset tb-science-vendor@71124a84e6f3]

``--collection-traces`` is a JSON-lines extract of the collection's solver traces with ``run`` (the collection run
name) and ``task.name`` per line (scratchpad replica-inv/collection-config/traces.jsonl has that shape).
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prime_rl.trainer.sft.inrun_eval import flowcfg  # noqa: E402
from prime_rl.trainer.sft.inrun_eval.tasks import (  # noqa: E402
    SOURCES,
    harbor_cache_name,
    parse_task_table,
    read_names,
)

PINNED = {
    "tb-eng": "d0e5b6c7bd031fcffaf6438210dbab60f78dcdd26ac26f887a32df719858ae5a",
    "tb-eng-extra9": "86a3fba822506c754a192edb4cb14926767ed33fff47a277cb51b8c2227e6512",
    "tb-sci": "06d60eb7f1eb1dbc8def8124e9d9d0097c8d6a4d82c0b3a492aa4d99b2084a92",
}
RUN_TO_SOURCE = {"tb-engineering": "tb-eng", "tb-engineering-extra9": "tb-eng-extra9", "tb-science": "tb-sci"}
EXPECTED = {"tb-eng": 200, "tb-eng-extra9": 9, "tb-sci": 291}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def train_tasks_from_dataset(directory: Path) -> set[str]:
    import pyarrow.parquet as pq

    names: set[str] = set()
    files = sorted(directory.rglob("train-*.parquet"))
    if not files:
        raise SystemExit(f"no train-*.parquet under {directory}")
    for path in files:
        names.update(pq.read_table(path, columns=["task"]).column("task").to_pylist())
    return names


def build_task_table(args) -> str:
    source_of: dict[str, str] = {}
    with open(args.collection_traces) as f:
        for line in f:
            row = json.loads(line)
            name = (row.get("task") or {}).get("name")
            source = RUN_TO_SOURCE.get(row.get("run"))
            if not name or source is None:
                continue
            if source_of.setdefault(name, source) != source:
                raise SystemExit(f"{name} is in two collection runs")
    for spec in args.extra_task:
        name, source = spec.split("=", 1)
        if source not in SOURCES:
            raise SystemExit(f"--extra-task {spec}: unknown source")
        source_of[name] = source
    val = set(read_names(args.val_tasks))
    train = train_tasks_from_dataset(args.train_dataset) if args.train_dataset else set(read_names(args.train_tasks))
    missing = sorted((val | train) - set(source_of))
    if missing:
        raise SystemExit(f"val/train tasks without a collection run: {missing[:5]}")
    if val & train:
        raise SystemExit(f"tasks both val and train: {sorted(val & train)[:5]}")
    lines = ["# name\tsource\tsplit (val: frozen held-out tasks; train: tasks with training rows; other: neither)"]
    for name in sorted(source_of):
        split = "val" if name in val else "train" if name in train else "other"
        lines.append(f"{name}\t{source_of[name]}\t{split}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sft-gen", type=Path, required=True)
    p.add_argument("--flow-config", action="append", default=[], metavar="SOURCE=PATH")
    p.add_argument("--tasks-tsv", type=Path)
    p.add_argument("--collection-traces", type=Path)
    p.add_argument("--val-tasks", type=Path)
    p.add_argument("--train-dataset", type=Path)
    p.add_argument("--train-tasks", type=Path)
    p.add_argument("--extra-task", action="append", default=[], metavar="NAME=SOURCE")
    p.add_argument("--harbor-cache", action="append", default=[], type=Path)
    p.add_argument("--kernel", type=Path, required=True)
    p.add_argument("--requirements", type=Path, required=True)
    p.add_argument("--wheels", type=Path, help="wheels to install from (at least the private prime-flow)")
    p.add_argument("--wheels-complete", action="store_true", help="every requirement has a wheel: install offline")
    p.add_argument("--python-tarball", type=Path)
    p.add_argument("--frpc", type=Path)
    p.add_argument("--smoke-tasks", type=Path)
    p.add_argument("--tb-sci-dataset", default="tb-science-vendor@71124a84e6f3")
    args = p.parse_args()

    out: Path = args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)

    # Flow configs: exactly the pinned collection configs.
    configs = dict(spec.split("=", 1) for spec in args.flow_config)
    if set(configs) != set(SOURCES):
        raise SystemExit(f"--flow-config needs {SOURCES}, got {sorted(configs)}")
    (out / "flow-configs").mkdir()
    caches = set()
    for source, path in configs.items():
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != PINNED[source]:
            raise SystemExit(f"{path}: not the pinned attempt_2 config of {source}")
        (out / "flow-configs" / f"{source}.toml").write_bytes(data)
        fields = flowcfg.taskset_fields(data)
        if source == "tb-sci" and args.tb_sci_dataset:
            fields["dataset"] = args.tb_sci_dataset
        caches.add(harbor_cache_name(fields["dataset"], fields["repo"], fields["registry_path"]))
        for bundle in flowcfg.kernel_bundles(data):
            if Path(bundle).name != args.kernel.name:
                raise SystemExit(f"{source} names kernel bundle {bundle}, --kernel is {args.kernel.name}")

    # Task table.
    if args.tasks_tsv:
        table_text = args.tasks_tsv.read_text()
    else:
        if not (args.collection_traces and args.val_tasks and (args.train_dataset or args.train_tasks)):
            raise SystemExit("--tasks-tsv, or --collection-traces + --val-tasks + --train-dataset/--train-tasks")
        table_text = build_task_table(args)
    table = parse_task_table(table_text)
    counts = {source: sum(row.source == source for row in table) for source in SOURCES}
    splits = {split: sum(row.split == split for row in table) for split in ("val", "train", "other")}
    print(f"tasks: {counts} {splits}")
    if counts != EXPECTED:
        print(f"WARNING: task counts {counts} differ from the eval's {EXPECTED} (expected_tasks = 500)")
    (out / "tasks.tsv").write_text(table_text)
    if args.smoke_tasks:
        names = read_names(args.smoke_tasks)
        unknown = sorted(set(names) - {row.name for row in table})
        if unknown:
            raise SystemExit(f"smoke tasks not in the task table: {unknown}")
        shutil.copyfile(args.smoke_tasks, out / "smoke-tasks.txt")

    # Harness: sft-gen tree, requirements, optional wheels / Python / frpc.
    ignore = shutil.ignore_patterns(".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache", "outputs")
    shutil.copytree(args.sft_gen, out / "sft-gen", ignore=ignore, symlinks=False)
    for rel in ("plugins/sft_gen_task_qc.py", "plugins/sft_gen_kernel_harbor.py", "sft_gen/task_qc.py"):
        if not (out / "sft-gen" / rel).is_file():
            raise SystemExit(f"--sft-gen lacks {rel}")
    (out / "harness").mkdir()
    shutil.copyfile(args.requirements, out / "harness" / "requirements.txt")
    if args.wheels:
        shutil.copytree(args.wheels, out / "harness" / "wheels")
        if args.wheels_complete:
            (out / "harness" / "wheels" / ".complete").write_text("")
    if args.python_tarball:
        (out / "harness" / "python").mkdir()
        shutil.copyfile(args.python_tarball, out / "harness" / "python" / args.python_tarball.name)
    if args.frpc:
        shutil.copytree(args.frpc, out / "harness" / "frpc")

    # Harbor caches (named as verifiers names them) and the kernel bundle.
    given = {path.name: path for path in args.harbor_cache}
    if set(given) != caches:
        raise SystemExit(f"--harbor-cache must be exactly {sorted(caches)}, got {sorted(given)}")
    for name, path in given.items():
        shutil.copytree(path, out / "harbor-cache" / name, symlinks=False)
    (out / "kernels").mkdir()
    shutil.copyfile(args.kernel, out / "kernels" / args.kernel.name)

    files = {str(path.relative_to(out)): sha256(path) for path in sorted(out.rglob("*")) if path.is_file()}
    (out / "manifest.json").write_text(json.dumps({"version": 1, "files": files}, indent=1) + "\n")
    print(f"bundle at {out}: {len(files)} files, caches {sorted(caches)}")


if __name__ == "__main__":
    main()
