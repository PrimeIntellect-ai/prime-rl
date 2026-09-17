"""Select a frozen SWE sample and split completed PRL avg@8 episodes by difficulty."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

BUCKETS = ("easy", "medium", "hard", "extra-hard")


def bucket_for(solves: int) -> str:
    if not 0 <= solves <= 8:
        raise ValueError(f"Expected 0–8 solves, got {solves}")
    if solves >= 6:
        return "easy"
    if solves >= 3:
        return "medium"
    return "hard" if solves else "extra-hard"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def select(parquet: Path, output: Path, revision: str, seed: int, count: int) -> None:
    import pyarrow.parquet as pq

    ids = pq.read_table(parquet, columns=["instance_id"])["instance_id"].to_pylist()
    if len(ids) != len(set(ids)):
        raise ValueError("Source dataset contains duplicate instance IDs")
    chosen = random.Random(seed).sample(sorted(ids), count)
    write_json(
        output,
        {
            "dataset": "PrimeIntellect/SWE-rebench-V2-Filtered-Verified",
            "revision": revision,
            "filename": "data/train-00000-of-00001.parquet",
            "population": len(ids),
            "seed": seed,
            "task_ids": chosen,
        },
    )


def records_from_run(run_dir: Path):
    import pyzstd

    monitors = run_dir / "monitors"
    directories = sorted(monitors.glob("file.attempt_*")) + [monitors / "file"]
    for directory in directories:
        stream = directory / "traces" / "stream"
        numbers = sorted({p.name.split(".")[0] for p in stream.glob("*.jsonl*")})
        for number in numbers:
            plain = stream / f"{number}.jsonl"
            compressed = stream / f"{number}.jsonl.zst"
            opener = plain.open("rb") if plain.exists() else pyzstd.open(compressed, "rb")
            with opener as lines:
                for line in lines:
                    yield json.loads(line)


def count_solves(records, task_ids: list[str]) -> dict[str, int]:
    selected = set(task_ids)
    outcomes = defaultdict(dict)
    seen = {}
    for record in records:
        if (record.get("env") or {}).get("name") != "swerebench-1k":
            continue
        if not record.get("ok"):
            continue
        task_id = record["task"]["data"]["name"]
        if task_id not in selected:
            raise ValueError(f"Unexpected task in profile: {task_id}")
        traces = record["traces"]
        if len(traces) != 1:
            raise ValueError(f"Expected one SWE trace: {record['id']}")
        reward = traces[0]["rewards"]["solved"]
        score = reward["score"]
        if score not in (0, 1) or reward.get("weight", 1) != 1:
            raise ValueError(f"Nonbinary solved reward: {reward}")
        identity = (task_id, record["task"]["hash"], score)
        if record["id"] in seen and seen[record["id"]] != identity:
            raise ValueError(f"Conflicting repeated episode: {record['id']}")
        seen[record["id"]] = identity
        outcomes[task_id][record["id"]] = (record["task"]["hash"], int(score))
    incomplete = {task: len(outcomes[task]) for task in task_ids if len(outcomes[task]) != 8}
    if incomplete:
        raise ValueError(f"Need exactly 8 valid episodes per task; resume eval first: {incomplete}")
    for task, samples in outcomes.items():
        if len({item[0] for item in samples.values()}) != 1:
            raise ValueError(f"Task content changed during eval: {task}")
    return {task: sum(item[1] for item in outcomes[task].values()) for task in task_ids}


def source_for(bucket: str, manifest: Path) -> dict:
    return {
        "name": f"swerebench-1k-{bucket}",
        "num_examples": -1,
        "group_size": 1,
        "env": {
            "taskset": {"id": "ngu-swe", "manifest": str(manifest.resolve())},
            "agent": {
                "harness": {"id": "bash"},
                "timeout": {"rollout": 3600},
                "runtime": {"type": "prime"},
            },
        },
    }


def split(manifest_path: Path, run_dir: Path, output: Path) -> None:
    import tomli_w

    manifest = json.loads(manifest_path.read_text())
    solves = count_solves(records_from_run(run_dir), manifest["task_ids"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen splits: {output}")
    output.mkdir(parents=True)
    sources = []
    sizes = {}
    for bucket in BUCKETS:
        ids = [task for task in manifest["task_ids"] if bucket_for(solves[task]) == bucket]
        path = output / f"{bucket}.json"
        write_json(path, {**manifest, "bucket": bucket, "task_ids": ids, "profile_run": str(run_dir.resolve())})
        sizes[bucket] = len(ids)
        if ids:
            sources.append(source_for(bucket, path))
    write_json(
        output / "results.json",
        {
            "avg_at_8": sum(solves.values()) / (8 * len(solves)),
            "bucket_sizes": sizes,
            "solves": solves,
        },
    )
    # Source lists replace on composition; include the existing held-out SWE eval.
    verified = {
        "name": "swebench-verified",
        "env": {
            "taskset": {"id": "swebench-verified"},
            "agent": {
                "harness": {"id": "bash"},
                "runtime": {"type": "prime"},
                "timeout": {"rollout": 3600},
            },
        },
    }
    (output / "online-eval.toml").write_text(
        tomli_w.dumps(
            {
                "orchestrator": {
                    "eval": {
                        "interval": 20,
                        "source": [verified, *sources],
                    }
                },
            }
        )
    )
    for bucket in BUCKETS:
        (output / f"{bucket}.toml").write_text(
            tomli_w.dumps(
                {
                    "id": "ngu-swe",
                    "manifest": str((output / f"{bucket}.json").resolve()),
                }
            )
        )
    print(json.dumps({"bucket_sizes": sizes, "output": str(output)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("select")
    choose.add_argument("parquet", type=Path)
    choose.add_argument("output", type=Path)
    choose.add_argument("revision")
    choose.add_argument("--seed", type=int, default=42)
    choose.add_argument("--count", type=int, default=1000)
    classify = commands.add_parser("split")
    classify.add_argument("manifest", type=Path)
    classify.add_argument("run_dir", type=Path)
    classify.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "select":
        select(args.parquet, args.output, args.revision, args.seed, args.count)
    else:
        split(args.manifest, args.run_dir, args.output)


if __name__ == "__main__":
    main()
