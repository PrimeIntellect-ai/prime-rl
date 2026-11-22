#!/usr/bin/env python3
"""Pinned, resumable Reasoning Gym transfer evaluation for GlyphBench."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL_PATH = Path(
    os.environ.get("REASONING_GYM_PROTOCOL_PATH", HERE / "protocol.json")
).resolve()
DEFAULT_CACHE = Path.home() / ".cache" / "glyphbench-reasoning-gym-eval"
DEFAULT_RUNS = REPO / "runs" / "reasoning-gym-transfer"


class EvalError(RuntimeError):
    """A reproducibility or evaluation invariant failed."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def protocol() -> dict[str, Any]:
    return read_json(PROTOCOL_PATH)


def protocol_hash(*, smoke: bool = False) -> str:
    digest = hashlib.sha256()
    for path in (PROTOCOL_PATH, Path(__file__).resolve()):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(b"\0smoke" if smoke else b"\0full")
    return digest.hexdigest()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return cleaned or "run"


def command_output(command: Sequence[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def checkout_path(cache_root: Path) -> Path:
    return cache_root / "upstreams" / "reasoning-gym"


def env_python(cache_root: Path) -> Path:
    return cache_root / "env" / "bin" / "python"


def ensure_checkout(cache_root: Path) -> None:
    source = protocol()["source"]
    target = checkout_path(cache_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not (target / ".git").exists():
        if target.exists():
            raise EvalError(f"Refusing to replace non-Git path: {target}")
        subprocess.run(["git", "clone", "--filter=blob:none", source["repo"], str(target)], check=True)
    origin = command_output(["git", "remote", "get-url", "origin"], cwd=target)
    if origin.rstrip("/") != source["repo"].rstrip("/"):
        raise EvalError(f"Unexpected Reasoning Gym origin: {origin}")
    subprocess.run(
        ["git", "fetch", "--depth=1", "origin", source["revision"]],
        cwd=target,
        check=True,
    )
    subprocess.run(["git", "checkout", "--detach", source["revision"]], cwd=target, check=True)


def ensure_environment(cache_root: Path) -> None:
    python = env_python(cache_root)
    root = python.parents[1]
    installs = [
        str(checkout_path(cache_root)),
        "httpx==0.28.1",
        "wandb==0.26.1",
        "huggingface-hub==1.13.0",
    ]
    marker_hash = sha256_bytes(
        (protocol()["source"]["revision"] + "\n" + "\n".join(installs)).encode()
    )[:16]
    marker = root / f".prepared-{marker_hash}"
    if python.exists() and marker.exists():
        return
    uv = shutil.which("uv")
    if not uv:
        raise EvalError("uv is required to prepare the Reasoning Gym environment")
    if not python.exists():
        subprocess.run([uv, "venv", "--python", sys.executable, str(root)], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(python), *installs], check=True)
    freeze = command_output([uv, "pip", "freeze", "--python", str(python)])
    (root / "requirements.freeze.txt").write_text(freeze + "\n", encoding="utf-8")
    marker.write_text(utc_now() + "\n", encoding="utf-8")


def canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): canonicalize(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, set):
        return sorted((canonicalize(item) for item in value), key=repr)
    if hasattr(value, "tolist"):
        return canonicalize(value.tolist())
    return repr(value)


def configured_tasks(spec: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    spec = spec or protocol()
    return [
        (category, dataset)
        for category, datasets in spec["categories"].items()
        for dataset in datasets
    ]


def validate_public_configs(cache_root: Path) -> None:
    import yaml

    spec = protocol()
    settings = spec["evaluation"]
    source = checkout_path(cache_root)
    root = source / spec["source"]["protocol_config_root"]
    for category, expected_tasks in spec["categories"].items():
        config = yaml.safe_load((root / f"{category}.yaml").read_text(encoding="utf-8"))
        expected_globals = {
            "max_tokens": settings.get("public_max_tokens", settings["max_tokens"]),
            "top_p": settings["top_p"],
            "temperature": settings["temperature"],
            "developer_prompt": settings["system_prompt_id"],
            "developer_role": "system",
            "eval_repeats": settings["completions_per_prompt"],
        }
        for key, expected in expected_globals.items():
            if config.get(key) != expected:
                raise EvalError(f"Official {category} config changed {key}: {config.get(key)!r}")
        categories = config.get("categories", [])
        if len(categories) != 1 or categories[0].get("category") != category:
            raise EvalError(f"Unexpected official category structure for {category}")
        rows = categories[0].get("datasets", [])
        actual_tasks = [row.get("dataset") for row in rows]
        if actual_tasks != expected_tasks:
            raise EvalError(f"Official {category} task list differs: {actual_tasks}")
        for row in rows:
            if row.get("size") != settings["size_per_dataset"] or row.get("seed") != settings["dataset_seed"]:
                raise EvalError(f"Official {category}/{row.get('dataset')} size or seed changed")
            if row.get("params"):
                raise EvalError(f"Unexpected task overrides for {category}/{row.get('dataset')}")
    evaluator = (source / "training" / "evaluations" / "evaluate_model.py").read_text(encoding="utf-8")
    if "score = 0.0 if score < 1 else score" not in evaluator:
        raise EvalError("Official exact-success threshold is absent from the pinned evaluator")


def materialize(cache_root: Path) -> dict[str, Any]:
    import reasoning_gym
    from reasoning_gym.utils import SYSTEM_PROMPTS

    spec = protocol()
    settings = spec["evaluation"]
    expected_hash_seed = str(settings["python_hash_seed"])
    if os.getenv("PYTHONHASHSEED") != expected_hash_seed:
        raise EvalError(
            "Materialization must start with "
            f"PYTHONHASHSEED={expected_hash_seed}; setting it inside Python is too late"
        )
    rows: list[dict[str, Any]] = []
    task_hashes: dict[str, str] = {}
    for category, dataset_name in configured_tasks(spec):
        dataset = reasoning_gym.create_dataset(
            dataset_name,
            size=settings["size_per_dataset"],
            seed=settings["dataset_seed"],
        )
        task_rows = []
        for index, entry in enumerate(dataset):
            record = {
                "key": f"{category}/{dataset_name}/{index:03d}",
                "category": category,
                "dataset": dataset_name,
                "index": index,
                "question": entry["question"],
                "expected_answer": str(entry["answer"]),
                "entry_sha256": sha256_bytes(
                    json.dumps(canonicalize(entry), sort_keys=True, ensure_ascii=False).encode()
                ),
            }
            rows.append(record)
            task_rows.append(record)
        task_hashes[f"{category}/{dataset_name}"] = sha256_bytes(
            json.dumps(task_rows, sort_keys=True, ensure_ascii=False).encode()
        )
    data_path = cache_root / "datasets" / "reasoning_gym_games_transfer.jsonl"
    write_jsonl(data_path, rows)
    manifest = {
        "schema_version": 1,
        "source_revision": spec["source"]["revision"],
        "protocol_file_sha256": sha256_file(PROTOCOL_PATH),
        "examples": len(rows),
        "tasks": len(configured_tasks(spec)),
        "size_per_dataset": settings["size_per_dataset"],
        "dataset_seed": settings["dataset_seed"],
        "python_hash_seed": settings["python_hash_seed"],
        "system_prompt_id": settings["system_prompt_id"],
        "system_prompt": SYSTEM_PROMPTS[settings["system_prompt_id"]],
        "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPTS[settings["system_prompt_id"]].encode()),
        "task_sha256": task_hashes,
        "dataset_sha256": sha256_file(data_path),
        "materialized_at": utc_now(),
    }
    write_json(data_path.with_suffix(".manifest.json"), manifest)
    return manifest


def prepare(args: argparse.Namespace) -> None:
    cache_root = args.cache_root.resolve()
    ensure_checkout(cache_root)
    ensure_environment(cache_root)
    validate_public_configs(cache_root)
    child_env = os.environ.copy()
    child_env["PYTHONHASHSEED"] = str(protocol()["evaluation"]["python_hash_seed"])
    subprocess.run(
        [str(env_python(cache_root)), str(Path(__file__).resolve()), "materialize", "--cache-root", str(cache_root)],
        check=True,
        env=child_env,
    )
    verify(argparse.Namespace(cache_root=cache_root))


def verify(args: argparse.Namespace) -> None:
    cache_root = args.cache_root.resolve()
    spec = protocol()
    failures: list[str] = []
    checkout = checkout_path(cache_root)
    try:
        head = command_output(["git", "rev-parse", "HEAD"], cwd=checkout)
        if head != spec["source"]["revision"]:
            failures.append(f"Reasoning Gym checkout is {head}, expected {spec['source']['revision']}")
    except Exception as exc:
        failures.append(f"Reasoning Gym checkout unavailable: {exc}")
    python = env_python(cache_root)
    if not python.exists():
        failures.append("Reasoning Gym virtual environment is missing")
    else:
        probe = subprocess.run(
            [str(python), "-c", "import httpx, reasoning_gym, wandb; print('ok')"],
            capture_output=True,
            text=True,
        )
        if probe.returncode:
            failures.append(f"Reasoning Gym import probe failed: {probe.stderr[-500:]}")
    data_path = cache_root / "datasets" / "reasoning_gym_games_transfer.jsonl"
    manifest_path = data_path.with_suffix(".manifest.json")
    if not data_path.exists() or not manifest_path.exists():
        failures.append("Materialized dataset is missing")
    else:
        manifest = read_json(manifest_path)
        expected_examples = len(configured_tasks(spec)) * spec["evaluation"]["size_per_dataset"]
        if manifest.get("source_revision") != spec["source"]["revision"]:
            failures.append("Materialized dataset source revision differs")
        if manifest.get("examples") != expected_examples:
            failures.append(f"Materialized dataset has {manifest.get('examples')}/{expected_examples} examples")
        if manifest.get("python_hash_seed") != spec["evaluation"]["python_hash_seed"]:
            failures.append("Materialized dataset used another Python hash seed")
        if manifest.get("dataset_sha256") != sha256_file(data_path):
            failures.append("Materialized dataset checksum differs")
    try:
        validate_public_configs(cache_root)
    except Exception as exc:
        failures.append(str(exc))
    if failures:
        raise EvalError("Preparation verification failed:\n- " + "\n- ".join(failures))
    print(f">>> verified Reasoning Gym cache at {cache_root}")


def fingerprint_model(model: str) -> dict[str, Any]:
    spec = protocol()
    path = Path(model).expanduser()
    if not path.exists():
        baseline = spec["models"]["baseline"]
        if model != baseline["model"]:
            raise EvalError("Remote evaluation is restricted to the pinned baseline")
        fingerprint = sha256_bytes(f"{model}@{baseline['revision']}".encode())
        return {
            "kind": "huggingface",
            "source": model,
            "revision": baseline["revision"],
            "fingerprint": fingerprint,
        }
    path = path.resolve()
    files = []
    candidates: list[Path] = []
    for pattern in (
        "*.safetensors",
        "*.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer*.json",
        "*.model",
    ):
        candidates.extend(path.glob(pattern))
    for candidate in sorted(set(candidates)):
        if candidate.is_file():
            files.append(
                {
                    "path": candidate.relative_to(path).as_posix(),
                    "bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    if not any(item["path"].endswith(".safetensors") for item in files):
        raise EvalError(f"Local model has no safetensor weights: {path}")
    return {
        "kind": "local",
        "source": str(path),
        "fingerprint": sha256_bytes(json.dumps(files, sort_keys=True).encode()),
        "files": files,
    }


def effective_run_dir(
    *, label: str, model_info: dict[str, Any], smoke: bool, runs_root: Path
) -> Path:
    mode = "smoke" if smoke else "full"
    name = (
        f"{safe_name(label)}--{model_info['fingerprint'][:12]}--"
        f"{protocol_hash(smoke=smoke)[:8]}-{mode}"
    )
    return runs_root / name


@contextlib.contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvalError(f"Another process is writing {run_dir}") from exc
        yield


def initialize_manifest(
    run_dir: Path,
    *,
    label: str,
    model_info: dict[str, Any],
    smoke: bool,
    cache_root: Path,
    base_url: str,
) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    expected_hash = protocol_hash(smoke=smoke)
    dataset_manifest = read_json(
        cache_root / "datasets" / "reasoning_gym_games_transfer.manifest.json"
    )
    if path.exists():
        value = read_json(path)
        if value.get("protocol_hash") != expected_hash:
            raise EvalError("Existing run has a different protocol hash")
        if value.get("model", {}).get("fingerprint") != model_info["fingerprint"]:
            raise EvalError("Existing run belongs to another model")
        if value.get("dataset_sha256") != dataset_manifest["dataset_sha256"]:
            raise EvalError("Existing run used another materialized dataset")
        return value
    value = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "label": label,
        "mode": "smoke" if smoke else "full",
        "protocol_name": protocol()["name"],
        "protocol_hash": expected_hash,
        "protocol_file_sha256": sha256_file(PROTOCOL_PATH),
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "dataset_manifest": dataset_manifest,
        "model": model_info,
        "base_url": base_url,
        "started_at": utc_now(),
        "status": "running",
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_nodelist": os.getenv("SLURM_JOB_NODELIST"),
    }
    write_json(path, value)
    implementation = run_dir / "implementation"
    implementation.mkdir(exist_ok=True)
    shutil.copy2(PROTOCOL_PATH, implementation / "protocol.json")
    shutil.copy2(Path(__file__).resolve(), implementation / "reasoning_gym_eval.py")
    return value


def load_completion_cache(path: Path) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    for row in jsonl_rows(path):
        key = row.get("completion_key")
        if not isinstance(key, str):
            raise EvalError(f"Completion cache row has no key: {row}")
        if key in cached and cached[key] != row:
            raise EvalError(f"Completion cache contains conflicting rows for {key}")
        cached[key] = row
    return cached


async def request_completion(
    client: Any,
    semaphore: asyncio.Semaphore,
    *,
    base_url: str,
    system_prompt: str,
    example: dict[str, Any],
    repeat: int,
    settings: dict[str, Any],
    served_name: str,
) -> dict[str, Any]:
    completion_key = f"{example['key']}/r{repeat}"
    body = {
        "model": served_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
        ],
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "max_tokens": settings["max_tokens"],
        "n": 1,
    }
    last_error: Exception | None = None
    for attempt in range(1, settings["request_attempts"] + 1):
        try:
            async with semaphore:
                response = await client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers={"authorization": "Bearer EMPTY"},
                    json=body,
                )
            response.raise_for_status()
            payload = response.json()
            choices = payload.get("choices") or []
            if len(choices) != 1:
                raise EvalError(f"{completion_key}: API returned {len(choices)} choices")
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            if not isinstance(content, str) or not isinstance(reasoning, str):
                raise EvalError(f"{completion_key}: API returned non-string content")
            full_response = content
            if reasoning and "<answer" not in content:
                full_response = f"<think>{reasoning}</think>\n{content}"
            return {
                "completion_key": completion_key,
                "example_key": example["key"],
                "category": example["category"],
                "dataset": example["dataset"],
                "index": example["index"],
                "repeat": repeat,
                "full_response": full_response,
                "content": content,
                "reasoning_content": reasoning or None,
                "finish_reason": choices[0].get("finish_reason"),
                "response_id": payload.get("id"),
                "usage": payload.get("usage"),
                "completed_at": utc_now(),
            }
        except EvalError:
            raise
        except Exception as exc:  # network and transient server failures
            last_error = exc
            if attempt == settings["request_attempts"]:
                break
            await asyncio.sleep(min(30.0, 1.5 * (2 ** (attempt - 1))))
    raise EvalError(f"{completion_key}: API failed after retries: {last_error}")


async def generate_completions(
    *,
    examples: list[dict[str, Any]],
    output_path: Path,
    base_url: str,
    system_prompt: str,
    settings: dict[str, Any],
    served_name: str,
    repeats: int,
) -> None:
    import httpx

    cached = load_completion_cache(output_path)
    jobs = [
        (example, repeat)
        for example in examples
        for repeat in range(repeats)
        if f"{example['key']}/r{repeat}" not in cached
    ]
    expected = len(examples) * repeats
    if not jobs:
        print(f">>> completion cache is complete ({expected}/{expected})")
        return
    print(f">>> generating {len(jobs)} missing completions; preserving {len(cached)}/{expected}")
    timeout = httpx.Timeout(settings["request_timeout_seconds"])
    limits = httpx.Limits(
        max_connections=settings["max_concurrent"],
        max_keepalive_connections=settings["max_concurrent"],
    )
    semaphore = asyncio.Semaphore(settings["max_concurrent"])
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = [
            asyncio.create_task(
                request_completion(
                    client,
                    semaphore,
                    base_url=base_url,
                    system_prompt=system_prompt,
                    example=example,
                    repeat=repeat,
                    settings=settings,
                    served_name=served_name,
                )
            )
            for example, repeat in jobs
        ]
        completed = len(cached)
        try:
            for future in asyncio.as_completed(tasks):
                row = await future
                append_jsonl(output_path, row)
                completed += 1
                if completed % 100 == 0 or completed == expected:
                    print(f">>> completions {completed}/{expected}", flush=True)
        except Exception:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


def select_examples(cache_root: Path, smoke: bool) -> list[dict[str, Any]]:
    rows = jsonl_rows(cache_root / "datasets" / "reasoning_gym_games_transfer.jsonl")
    if not smoke:
        return rows
    selected_tasks = set((row["category"], row["dataset"]) for row in rows[:1])
    for row in rows:
        key = (row["category"], row["dataset"])
        if key not in selected_tasks and len(selected_tasks) < 2:
            selected_tasks.add(key)
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    selected = []
    for row in rows:
        key = (row["category"], row["dataset"])
        if key in selected_tasks and counts[key] < 2:
            selected.append(row)
            counts[key] += 1
    return selected


def score_completions(
    *,
    cache_root: Path,
    examples: list[dict[str, Any]],
    completions_path: Path,
    scored_path: Path,
    repeats: int,
) -> list[dict[str, Any]]:
    import reasoning_gym
    from reasoning_gym.utils import extract_answer

    spec = protocol()
    threshold = spec["evaluation"]["exact_success_threshold"]
    examples_by_key = {row["key"]: row for row in examples}
    completions = load_completion_cache(completions_path)
    expected_keys = {
        f"{example['key']}/r{repeat}" for example in examples for repeat in range(repeats)
    }
    if set(completions) != expected_keys:
        missing = sorted(expected_keys - set(completions))[:10]
        extra = sorted(set(completions) - expected_keys)[:10]
        raise EvalError(f"Completion keys differ: missing={missing}, extra={extra}")
    datasets: dict[tuple[str, str], Any] = {}
    entries: dict[str, dict[str, Any]] = {}
    settings = spec["evaluation"]
    for category, dataset_name in sorted({(row["category"], row["dataset"]) for row in examples}):
        dataset = reasoning_gym.create_dataset(
            dataset_name,
            size=settings["size_per_dataset"],
            seed=settings["dataset_seed"],
        )
        datasets[(category, dataset_name)] = dataset
    for key, example in examples_by_key.items():
        entry = datasets[(example["category"], example["dataset"])][example["index"]]
        current_hash = sha256_bytes(
            json.dumps(canonicalize(entry), sort_keys=True, ensure_ascii=False).encode()
        )
        if current_hash != example["entry_sha256"]:
            raise EvalError(f"Pinned generator no longer recreates {key}")
        entries[key] = entry
    scored = []
    for completion_key in sorted(completions):
        row = completions[completion_key]
        entry = entries[row["example_key"]]
        answer = None
        raw_score = 0.0
        score_error = None
        try:
            answer = extract_answer(
                row["full_response"], tag_name=spec["evaluation"]["answer_tag"]
            )
            if answer is not None:
                raw_score = float(
                    datasets[(row["category"], row["dataset"])].score_answer(
                        answer=answer,
                        entry=entry,
                    )
                )
        except Exception as exc:
            # The pinned public evaluator treats any per-completion extraction
            # or verifier exception as an incorrect answer and continues.
            score_error = repr(exc)
        exact = raw_score >= threshold
        scored.append(
            {
                "completion_key": completion_key,
                "example_key": row["example_key"],
                "category": row["category"],
                "dataset": row["dataset"],
                "index": row["index"],
                "repeat": row["repeat"],
                "answer": answer,
                "parseable": answer is not None,
                "raw_verifier_score": raw_score,
                "exact": exact,
                "official_thresholded_score": float(exact),
                "finish_reason": row.get("finish_reason"),
                "score_error": score_error,
            }
        )
    write_jsonl(scored_path, scored)
    return scored


def aggregate_records(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = len(rows)
    parsed = sum(bool(row["parseable"]) for row in rows)
    exact = sum(bool(row["exact"]) for row in rows)
    return {
        "name": name,
        "attempted_completions": attempted,
        "parseable_completions": parsed,
        "exact_completions": exact,
        "strict_accuracy": exact / attempted if attempted else 0.0,
        "published_valid_only_accuracy": exact / parsed if parsed else 0.0,
        "answer_format_rate": parsed / attempted if attempted else 0.0,
    }


def group_task_names(spec: dict[str, Any], group_name: str) -> set[str]:
    members = spec["groups"][group_name]
    selected: set[str] = set()
    for member in members:
        if "/" in member:
            selected.add(member)
        else:
            selected.update(f"{member}/{dataset}" for dataset in spec["categories"][member])
    return selected


def build_summary(
    *, run_dir: Path, scored: list[dict[str, Any]], examples: list[dict[str, Any]], repeats: int
) -> dict[str, Any]:
    spec = protocol()
    by_task: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_task[f"{row['category']}/{row['dataset']}"].append(row)
        by_category[row["category"]].append(row)
    tasks = {name: aggregate_records(name, rows) for name, rows in sorted(by_task.items())}
    categories = {
        name: aggregate_records(name, rows) for name, rows in sorted(by_category.items())
    }
    groups = {}
    for name in spec["groups"]:
        selected = group_task_names(spec, name)
        rows = [row for task in selected for row in by_task.get(task, [])]
        groups[name] = aggregate_records(name, rows)
        groups[name]["tasks"] = sorted(selected)
    all_metrics = aggregate_records("all", scored)
    manifest = read_json(run_dir / "manifest.json")
    summary = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "label": manifest["label"],
        "mode": manifest["mode"],
        "model": manifest["model"],
        "protocol_hash": manifest["protocol_hash"],
        "dataset_sha256": manifest["dataset_sha256"],
        "problems": len(examples),
        "completions_per_problem": repeats,
        "all": all_metrics,
        "groups": groups,
        "categories": categories,
        "tasks": tasks,
        "collected_at": utc_now(),
    }
    write_json(run_dir / "summary.json", summary)
    rows = []
    for level, values in (
        ("all", {"all": all_metrics}),
        ("group", groups),
        ("category", categories),
        ("task", tasks),
    ):
        for metrics in values.values():
            rows.append({"level": level, **metrics})
    columns = [
        "level",
        "name",
        "strict_accuracy",
        "published_valid_only_accuracy",
        "answer_format_rate",
        "exact_completions",
        "parseable_completions",
        "attempted_completions",
    ]
    with (run_dir / "paper_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary


def publish_run(run_dir: Path, summary: dict[str, Any]) -> str:
    import wandb

    reporting = protocol()["reporting"]
    run_id = f"reasoning-gym-{safe_name(summary['label'])}-{summary['protocol_hash'][:8]}"
    run = wandb.init(
        entity=reporting.get("wandb_entity") or os.getenv("WANDB_ENTITY"),
        project=reporting["wandb_project"],
        group=reporting["wandb_group"],
        id=run_id,
        resume="allow",
        job_type="reasoning-gym-eval",
        name=f"Reasoning Gym: {summary['label']}",
        config={
            "protocol": protocol(),
            "protocol_hash": summary["protocol_hash"],
            "dataset_sha256": summary["dataset_sha256"],
            "model": summary["model"],
        },
    )
    metrics = {}
    for level in ("groups", "categories", "tasks"):
        for name, value in summary[level].items():
            prefix = f"reasoning_gym/{level[:-1]}/{name}"
            metrics[f"{prefix}/strict_accuracy"] = value["strict_accuracy"]
            metrics[f"{prefix}/published_valid_only_accuracy"] = value[
                "published_valid_only_accuracy"
            ]
            metrics[f"{prefix}/answer_format_rate"] = value["answer_format_rate"]
    run.log(metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    artifact = wandb.Artifact(
        name=run_id,
        type="reasoning-gym-eval",
        metadata={
            "protocol_hash": summary["protocol_hash"],
            "dataset_sha256": summary["dataset_sha256"],
            "model_fingerprint": summary["model"]["fingerprint"],
        },
    )
    for name in (
        "manifest.json",
        "summary.json",
        "paper_table.csv",
        "completions.jsonl",
        "scored.jsonl",
    ):
        artifact.add_file(str(run_dir / name), name=name)
    artifact.add_dir(str(run_dir / "implementation"), name="implementation")
    run.log_artifact(artifact)
    url = run.url
    run.finish()
    return str(url)


def run_eval(args: argparse.Namespace) -> None:
    cache_root = args.cache_root.resolve()
    runs_root = args.runs_root.resolve()
    expected_hash_seed = str(protocol()["evaluation"]["python_hash_seed"])
    if os.getenv("PYTHONHASHSEED") != expected_hash_seed:
        raise EvalError(
            f"Evaluation must start with PYTHONHASHSEED={expected_hash_seed} so scorer reconstruction is stable"
        )
    verify(argparse.Namespace(cache_root=cache_root))
    model_info = fingerprint_model(args.model)
    run_dir = effective_run_dir(
        label=args.label,
        model_info=model_info,
        smoke=args.smoke,
        runs_root=runs_root,
    )
    print(f">>> result directory: {run_dir}")
    with run_lock(run_dir):
        manifest = initialize_manifest(
            run_dir,
            label=args.label,
            model_info=model_info,
            smoke=args.smoke,
            cache_root=cache_root,
            base_url=args.base_url,
        )
        if manifest.get("status") == "complete" and (run_dir / "summary.json").exists():
            print(">>> run is already complete")
            return
        examples = select_examples(cache_root, args.smoke)
        dataset_manifest = read_json(
            cache_root / "datasets" / "reasoning_gym_games_transfer.manifest.json"
        )
        settings = protocol()["evaluation"]
        repeats = 1 if args.smoke else settings["completions_per_prompt"]
        try:
            asyncio.run(
                generate_completions(
                    examples=examples,
                    output_path=run_dir / "completions.jsonl",
                    base_url=args.base_url,
                    system_prompt=dataset_manifest["system_prompt"],
                    settings=settings,
                    served_name=protocol()["serving"]["served_name"],
                    repeats=repeats,
                )
            )
            scored = score_completions(
                cache_root=cache_root,
                examples=examples,
                completions_path=run_dir / "completions.jsonl",
                scored_path=run_dir / "scored.jsonl",
                repeats=repeats,
            )
            summary = build_summary(
                run_dir=run_dir,
                scored=scored,
                examples=examples,
                repeats=repeats,
            )
            manifest = read_json(run_dir / "manifest.json")
            manifest.update({"status": "complete", "completed_at": utc_now()})
            write_json(run_dir / "manifest.json", manifest)
            if not args.smoke and args.publish_wandb:
                url = publish_run(run_dir, summary)
                manifest = read_json(run_dir / "manifest.json")
                manifest["wandb_url"] = url
                write_json(run_dir / "manifest.json", manifest)
            print(json.dumps(summary["groups"], indent=2))
        except Exception as exc:
            manifest = read_json(run_dir / "manifest.json")
            manifest.update({"status": "failed", "failed_at": utc_now(), "error": repr(exc)})
            write_json(run_dir / "manifest.json", manifest)
            raise


def result_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        key = row["completion_key"]
        if key in result:
            raise EvalError(f"Duplicate scored completion: {key}")
        result[key] = row
    return result


def selected_task_names(spec: dict[str, Any], level: str, name: str) -> set[str]:
    if level == "all":
        return {f"{category}/{dataset}" for category, dataset in configured_tasks(spec)}
    if level == "task":
        return {name}
    if level == "category":
        return {f"{name}/{dataset}" for dataset in spec["categories"][name]}
    if level == "group":
        return group_task_names(spec, name)
    raise EvalError(f"Unknown comparison level: {level}")


def paired_problem_differences(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    tasks: set[str],
) -> dict[str, list[float]]:
    values: defaultdict[tuple[str, int], list[float]] = defaultdict(list)
    for key in sorted(set(before) & set(after)):
        row = before[key]
        task = f"{row['category']}/{row['dataset']}"
        if task not in tasks:
            continue
        if after[key]["example_key"] != row["example_key"]:
            raise EvalError(f"Paired completion mismatch: {key}")
        values[(task, int(row["index"]))].append(float(after[key]["exact"]) - float(row["exact"]))
    by_task: defaultdict[str, list[float]] = defaultdict(list)
    for (task, _), differences in values.items():
        by_task[task].append(sum(differences) / len(differences))
    return dict(by_task)


def stratified_paired_bootstrap(
    differences: dict[str, list[float]], *, samples: int, seed: int
) -> tuple[float, float]:
    if not differences or any(not values for values in differences.values()):
        raise EvalError("Paired bootstrap received no problem differences")
    rng = random.Random(seed)
    task_names = sorted(differences)
    draws = []
    for _ in range(samples):
        task_means = []
        for task in task_names:
            values = differences[task]
            task_means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
        draws.append(sum(task_means) / len(task_means))
    draws.sort()
    return draws[int(0.025 * samples)], draws[int(0.975 * samples)]


def comparison_row(
    *,
    level: str,
    name: str,
    baseline_summary: dict[str, Any],
    checkpoint_summary: dict[str, Any],
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if level == "all":
        baseline = baseline_summary["all"]
        checkpoint = checkpoint_summary["all"]
    else:
        plural = {"group": "groups", "category": "categories", "task": "tasks"}[level]
        baseline = baseline_summary[plural][name]
        checkpoint = checkpoint_summary[plural][name]
    spec = protocol()
    differences = paired_problem_differences(
        before, after, selected_task_names(spec, level, name)
    )
    reporting = spec["reporting"]
    low, high = stratified_paired_bootstrap(
        differences,
        samples=reporting["bootstrap_samples"],
        seed=reporting["bootstrap_seed"],
    )
    delta = checkpoint["strict_accuracy"] - baseline["strict_accuracy"]
    return {
        "level": level,
        "name": name,
        "baseline_strict_accuracy": baseline["strict_accuracy"],
        "checkpoint_strict_accuracy": checkpoint["strict_accuracy"],
        "strict_delta_pp": 100 * delta,
        "paired_bootstrap_95_low_pp": 100 * low,
        "paired_bootstrap_95_high_pp": 100 * high,
        "baseline_published_valid_only_accuracy": baseline["published_valid_only_accuracy"],
        "checkpoint_published_valid_only_accuracy": checkpoint[
            "published_valid_only_accuracy"
        ],
        "published_valid_only_delta_pp": 100
        * (
            checkpoint["published_valid_only_accuracy"]
            - baseline["published_valid_only_accuracy"]
        ),
        "baseline_answer_format_rate": baseline["answer_format_rate"],
        "checkpoint_answer_format_rate": checkpoint["answer_format_rate"],
        "paired_problems": sum(len(values) for values in differences.values()),
        "tasks": len(differences),
    }


def publish_comparison(output: Path, report: dict[str, Any]) -> str:
    import wandb

    reporting = protocol()["reporting"]
    run_id = f"reasoning-gym-comparison-{report['protocol_hash'][:8]}"
    run = wandb.init(
        entity=reporting.get("wandb_entity") or os.getenv("WANDB_ENTITY"),
        project=reporting["wandb_project"],
        group=reporting["wandb_group"],
        id=run_id,
        resume="allow",
        job_type="reasoning-gym-comparison",
        name=(
            "Reasoning Gym: base vs GlyphBench step 450 "
            f"({protocol()['evaluation']['max_tokens']} tokens)"
        ),
        config={
            "protocol": protocol(),
            "protocol_hash": report["protocol_hash"],
            "baseline_run": report["baseline_run"],
            "checkpoint_run": report["checkpoint_run"],
        },
    )
    metrics = {}
    for row in report["comparisons"]:
        prefix = f"reasoning_gym/comparison/{row['level']}/{row['name']}"
        metrics[f"{prefix}/baseline_strict_accuracy"] = row["baseline_strict_accuracy"]
        metrics[f"{prefix}/checkpoint_strict_accuracy"] = row["checkpoint_strict_accuracy"]
        metrics[f"{prefix}/strict_delta_pp"] = row["strict_delta_pp"]
        metrics[f"{prefix}/ci_low_pp"] = row["paired_bootstrap_95_low_pp"]
        metrics[f"{prefix}/ci_high_pp"] = row["paired_bootstrap_95_high_pp"]
    run.log(metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    artifact = wandb.Artifact(run_id, type="reasoning-gym-comparison")
    for name in ("full_analysis.json", "full_analysis.csv", "full_analysis.md"):
        artifact.add_file(str(output / name), name=name)
    run.log_artifact(artifact)
    url = run.url
    run.finish()
    return str(url)


def compare(args: argparse.Namespace) -> None:
    baseline_dir = args.baseline.resolve()
    checkpoint_dir = args.checkpoint.resolve()
    baseline = read_json(baseline_dir / "summary.json")
    checkpoint = read_json(checkpoint_dir / "summary.json")
    if baseline["mode"] != "full" or checkpoint["mode"] != "full":
        raise EvalError("Smoke runs cannot be used for the paper comparison")
    if baseline["protocol_hash"] != checkpoint["protocol_hash"]:
        raise EvalError("Protocol hashes differ")
    if baseline["dataset_sha256"] != checkpoint["dataset_sha256"]:
        raise EvalError("Materialized datasets differ")
    before = result_index(jsonl_rows(baseline_dir / "scored.jsonl"))
    after = result_index(jsonl_rows(checkpoint_dir / "scored.jsonl"))
    if set(before) != set(after):
        raise EvalError("Scored completion keys differ between models")
    rows = []
    for level, names in (
        ("all", ["all"]),
        ("group", sorted(baseline["groups"])),
        ("category", sorted(baseline["categories"])),
        ("task", sorted(baseline["tasks"])),
    ):
        for name in names:
            rows.append(
                comparison_row(
                    level=level,
                    name=name,
                    baseline_summary=baseline,
                    checkpoint_summary=checkpoint,
                    before=before,
                    after=after,
                )
            )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "protocol_hash": baseline["protocol_hash"],
        "dataset_sha256": baseline["dataset_sha256"],
        "baseline_run": baseline["run_id"],
        "checkpoint_run": checkpoint["run_id"],
        "bootstrap_samples": protocol()["reporting"]["bootstrap_samples"],
        "bootstrap_seed": protocol()["reporting"]["bootstrap_seed"],
        "comparisons": rows,
        "created_at": utc_now(),
    }
    write_json(output / "full_analysis.json", report)
    with (output / "full_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    max_tokens = protocol()["evaluation"]["max_tokens"]
    lines = [
        f"# Qwen3.5-4B vs GlyphBench step 450: Reasoning Gym transfer ({max_tokens:,}-token cap)",
        "",
        f"Protocol hash: `{baseline['protocol_hash']}`  ",
        f"Materialized dataset hash: `{baseline['dataset_sha256']}`",
        "",
        "The primary metric is strict exact accuracy over all three attempted completions per problem. "
        "The paired confidence interval bootstraps problem-level mean success, stratified by task. "
        "The published-compatible metric excludes completions without a parseable `<answer>` tag.",
        "",
        "## Aggregate and category results",
        "",
        "| Scope | Base strict | Checkpoint strict | Delta (pp) | Paired 95% CI (pp) | Base published | Checkpoint published |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["level"] not in {"all", "group", "category"}:
            continue
        lines.append(
            f"| {row['level']}: {row['name']} | {100 * row['baseline_strict_accuracy']:.2f} | "
            f"{100 * row['checkpoint_strict_accuracy']:.2f} | {row['strict_delta_pp']:+.2f} | "
            f"[{row['paired_bootstrap_95_low_pp']:+.2f}, {row['paired_bootstrap_95_high_pp']:+.2f}] | "
            f"{100 * row['baseline_published_valid_only_accuracy']:.2f} | "
            f"{100 * row['checkpoint_published_valid_only_accuracy']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Per-task results",
            "",
            "| Task | Base strict | Checkpoint strict | Delta (pp) | Paired 95% CI (pp) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["level"] != "task":
            continue
        lines.append(
            f"| {row['name']} | {100 * row['baseline_strict_accuracy']:.2f} | "
            f"{100 * row['checkpoint_strict_accuracy']:.2f} | {row['strict_delta_pp']:+.2f} | "
            f"[{row['paired_bootstrap_95_low_pp']:+.2f}, {row['paired_bootstrap_95_high_pp']:+.2f}] |"
        )
    (output / "full_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.publish_wandb:
        report["wandb_url"] = publish_comparison(output, report)
        write_json(output / "full_analysis.json", report)
    print(
        json.dumps(
            [row for row in rows if row["level"] in {"all", "group", "category"}],
            indent=2,
        )
    )


def resolve_run(args: argparse.Namespace) -> None:
    model_info = fingerprint_model(args.model)
    print(
        effective_run_dir(
            label=args.label,
            model_info=model_info,
            smoke=args.smoke,
            runs_root=args.runs_root.resolve(),
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    prepare_parser.set_defaults(func=prepare)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    materialize_parser.set_defaults(
        func=lambda args: print(json.dumps(materialize(args.cache_root.resolve()), indent=2))
    )

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    verify_parser.set_defaults(func=verify)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8004/v1")
    run_parser.add_argument("--smoke", action="store_true")
    run_parser.add_argument(
        "--publish-wandb",
        action="store_true",
        help="publish results to the W&B project configured in the protocol",
    )
    run_parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    run_parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    run_parser.set_defaults(func=run_eval)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--checkpoint", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument(
        "--publish-wandb",
        action="store_true",
        help="publish the comparison to the W&B project configured in the protocol",
    )
    compare_parser.set_defaults(func=compare)

    resolve_parser = subparsers.add_parser("resolve-run")
    resolve_parser.add_argument("--model", required=True)
    resolve_parser.add_argument("--label", required=True)
    resolve_parser.add_argument("--smoke", action="store_true")
    resolve_parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    resolve_parser.set_defaults(func=resolve_run)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        args.func(args)
    except (EvalError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
