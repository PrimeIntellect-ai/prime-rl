"""Local dashboard for prime-rl runs: logs, metrics, rollout traces, and reports.

This package is fully AI-generated and maintained by agents - it is not meant to be read or edited by humans. Change it by asking an agent, and verify through the browser smoke tests.

Reads everything from run output directories (the file monitor's metrics and trace
streams, logs/attempt_N) — no wandb or network required. Usage:

    uv run dashboard [output_dir ...] [--port 7788] [--host 127.0.0.1]

Multiple output directories can be tracked at once.
"""

import argparse
import asyncio
import hashlib
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import orjson

from prime_rl.entrypoints.dashboard import DAEMON_FILE, DIRS_FILE, STATE_DIR, registry_lock
from prime_rl.utils.config import default_output_dir
from prime_rl.utils.pathing import get_file_monitor_dir
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.trace_updates import branch_node_paths, fold_trace_updates

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
except ModuleNotFoundError as error:  # the dashboard ships as an extra
    raise SystemExit("the dashboard needs the 'dashboard' extra - install with `uv sync --extra dashboard`") from error

STATIC_DIR = Path(__file__).parent / "static"
MASTER_LOGS = {"trainer.log", "orchestrator.log", "inference.log", "evals.log"}
MAX_LOG_CHUNK = 2_000_000

app = FastAPI()
output_dirs: list[Path] = [default_output_dir()]

# run id -> run dir, rebuilt on every /api/runs poll; ids are the run name,
# qualified with the output dir's basename when two dirs hold the same name
_run_registry: dict[str, Path] = {}

_lock = threading.Lock()
MAX_CACHED_FILES = 64
"""Per-cache LRU bound: paging through thousands of steps must not grow the
server without limit (each entry holds one traces file's offsets/summaries)."""


def _lru_get(cache: OrderedDict, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _lru_put(cache: OrderedDict, key, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > MAX_CACHED_FILES:
        cache.popitem(last=False)


# Append-only file caches keyed by absolute path: line-start offsets and per-episode summaries.
_offsets_cache: OrderedDict[Path, tuple[int, bytes, list[int]]] = OrderedDict()
_summaries_cache: OrderedDict[Path, tuple[int, list[dict]]] = OrderedDict()
_annotations_cache: OrderedDict[Path, tuple[tuple[int, ...], dict[str, list[dict]]]] = OrderedDict()
_tokenizer_cache: dict[str, object] = {}
_piece_cache: dict[tuple[str, int], str] = {}
_json_cache: dict[Path, tuple[tuple[int, float], dict]] = {}


def is_run_dir(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith("."):
        return False
    return any((path / marker).exists() for marker in ("configs", "logs", "monitors", "traces.jsonl"))


_dirs_file_state: tuple[float, list[Path]] = (0.0, [])


def registered_dirs() -> list[Path]:
    """Output dirs registered in dirs.json (by launchers and by every dashboard's
    own CLI dirs), re-read when the file changes so a run in a new output dir
    appears on the one live dashboard without a restart."""
    global _dirs_file_state
    try:
        mtime = DIRS_FILE.stat().st_mtime
    except OSError:
        return []
    if mtime != _dirs_file_state[0]:
        try:
            dirs = [Path(d) for d in orjson.loads(DIRS_FILE.read_bytes())]
        except (OSError, ValueError):
            dirs = []
        _dirs_file_state = (mtime, dirs)
    return _dirs_file_state[1]


def register_dirs(dirs: list[Path]) -> None:
    """Add dirs to the shared registry (idempotent, atomic)."""
    with registry_lock():
        try:
            known = list(orjson.loads(DIRS_FILE.read_bytes()))
        except (OSError, ValueError):
            known = []
        fresh = [str(d.resolve()) for d in dirs if str(d.resolve()) not in known]
        if not fresh:
            return
        tmp = DIRS_FILE.with_suffix(".tmp")
        tmp.write_bytes(orjson.dumps(known + fresh))
        tmp.replace(DIRS_FILE)


isolated = False


def tracked_dirs() -> list[Path]:
    """One dashboard per user serves everything: the dirs it was started with
    plus every dir any launcher (or other dashboard start) has registered.
    --isolated opts out: only the CLI dirs, no registry."""
    dirs = list(output_dirs)
    if isolated:
        return dirs
    known = {d.resolve() for d in dirs}
    dirs.extend(d for d in registered_dirs() if d.resolve() not in known and d.is_dir())
    return dirs


def scan_runs() -> dict[str, Path]:
    global _run_registry
    by_name: dict[str, list[Path]] = {}
    for base in tracked_dirs():
        if not base.is_dir():
            continue
        for run_dir in sorted(base.iterdir()):
            if is_run_dir(run_dir):
                by_name.setdefault(run_dir.name, []).append(run_dir)
    registry: dict[str, Path] = {}
    for name, dirs in by_name.items():
        if len(dirs) == 1:
            registry[name] = dirs[0]
        else:
            for run_dir in dirs:
                registry[f"{run_dir.parent.name}:{name}"] = run_dir  # ":" survives a URL path segment, "/" does not
    _run_registry = registry
    return registry


def get_run_dir(run: str) -> Path:
    run_dir = _run_registry.get(run) or scan_runs().get(run)
    if run_dir is None or not run_dir.is_dir():
        raise HTTPException(404, f"run {run} not found")
    return run_dir


def safe_child(base: Path, rel: str, *, suffix: str | None = None) -> Path:
    path = (base / rel).resolve()
    if not path.is_relative_to(base.resolve()) or (suffix and path.suffix != suffix) or not path.is_file():
        raise HTTPException(404, f"{rel} not found")
    return path


def read_json(path: Path) -> dict:
    """Config reads are cached by (size, mtime) — they sit on every /api/runs poll."""
    try:
        stat = path.stat()
    except OSError:
        return {}
    key = (stat.st_size, stat.st_mtime)
    with _lock:
        cached = _json_cache.get(path)
        if cached and cached[0] == key:
            return cached[1]
    try:
        data = orjson.loads(path.read_bytes())
    except (OSError, ValueError):
        data = {}
    with _lock:
        _json_cache[path] = (key, data)
    return data


def numbered_dirs(parent: Path, prefix: str) -> list[tuple[int, Path]]:
    return sorted(
        (int(p.name.removeprefix(prefix)), p)
        for p in parent.glob(f"{prefix}*")
        if p.name.removeprefix(prefix).isdigit()
    )


def model_name(config: dict) -> str | None:
    """`model` is a string on eval configs, a {name} object on rl/sft ones."""
    model = config.get("model")
    return model if isinstance(model, str) else (model or {}).get("name")


def config_attempt_numbers(run_dir: Path) -> list[int]:
    return [n for n, _ in numbered_dirs(run_dir / "configs", "attempt_")]


def config_attempt_dir(run_dir: Path, attempt: str = "latest") -> tuple[Path, int | None]:
    """Return one config attempt root, with a fallback for legacy run layouts."""
    configs = run_dir / "configs"
    attempts = config_attempt_numbers(run_dir)
    if not attempts:
        return configs, None
    if attempt == "latest":
        latest = (configs / "latest").resolve()
        attempt_num = int(latest.name.removeprefix("attempt_")) if latest.is_dir() else attempts[-1]
    else:
        attempt_num = int(attempt)
    return configs / f"attempt_{attempt_num}", attempt_num


def resolved_config_dir(run_dir: Path, attempt: str = "latest") -> Path:
    """Return resolved JSON dumps for one attempt or a legacy run."""
    configs, _ = config_attempt_dir(run_dir, attempt)
    resolved = configs / "resolved"
    return resolved if resolved.is_dir() else configs


def main_config(run_dir: Path) -> tuple[str, dict]:
    configs = resolved_config_dir(run_dir)
    if (configs / "sft.json").exists():
        return "sft", read_json(configs / "sft.json")
    if (configs / "orchestrator.json").exists() or (configs / "trainer.json").exists():
        return "rl", read_json(configs / "orchestrator.json") or read_json(configs / "trainer.json")
    if (configs / "eval.json").exists():  # verifiers `uv run eval` run dir
        return "eval", read_json(configs / "eval.json")
    return "other", {}


def attempt_numbers(run_dir: Path) -> list[int]:
    return [n for n, _ in numbered_dirs(run_dir / "logs", "attempt_")]


def metrics_file(run_dir: Path) -> Path:
    """The run's metric stream, as the file monitor dumps it."""
    return get_file_monitor_dir(run_dir) / "metrics.jsonl"


def step_numbers(run_dir: Path) -> list[int]:
    """The steps a cohort shipped at — what a run's progress is measured in."""
    return sorted({step for _, step in effective_steps(run_dir)})


def run_meta(run_dir: Path) -> dict:
    configs = run_dir / "configs"
    run_type, config = main_config(run_dir)
    resolved = resolved_config_dir(run_dir)

    def envs(split: str) -> list[str]:
        return sorted(p.stem for p in (resolved / "envs" / split).glob("*.json"))

    steps = step_numbers(run_dir)
    metrics_path = metrics_file(run_dir)
    started = updated = None
    if metrics_path.is_file():
        updated = metrics_path.stat().st_mtime
        # read fresh every time: a relaunch replaces the file, and a cached
        # first-row time from the previous run makes the duration absurd
        with metrics_path.open("rb") as f:
            try:
                started = orjson.loads(f.readline()).get("time")
            except orjson.JSONDecodeError:
                started = None
    stream = traces_file(run_dir)
    if updated is None and stream is not None:  # eval runs have no metrics
        updated = stream.stat().st_mtime
        started = configs.stat().st_mtime if configs.is_dir() else None
    return {
        "name": run_dir.name,
        "type": run_type,
        "model": model_name(config),
        "dataset": (config.get("data") or {}).get("name"),
        "has_validation": run_type == "sft" and config.get("val") is not None,
        "env": ((config.get("env") or {}).get("taskset") or {}).get("id"),
        "total_episodes": (config.get("num_tasks") or 0) * (config.get("num_rollouts") or 0) or None,
        "max_steps": config.get("max_steps"),
        "train_envs": envs("train"),
        "eval_envs": envs("eval"),
        "has_metrics": metrics_path.exists(),
        "last_step": max(steps, default=None),
        "started": started,
        "updated": updated,
        "created": configs.stat().st_mtime if configs.is_dir() else run_dir.stat().st_mtime,
        "mtime": run_dir.stat().st_mtime,
    }


@app.get("/api/runs")
def list_runs() -> dict:
    runs = []
    for run_id, run_dir in scan_runs().items():
        meta = run_meta(run_dir)
        meta["name"] = run_id
        runs.append(meta)
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return {"output_dir": ", ".join(str(d.resolve()) for d in output_dirs), "runs": runs}


@app.get("/api/runs/{run}")
def get_run(run: str) -> dict:
    return run_meta(get_run_dir(run))


# ---------------------------------------------------------------------------- logs


def log_component(rel: Path) -> tuple[str, str]:
    """Map a path relative to the attempt dir to (component, label)."""
    parts = rel.parts
    if len(parts) == 1:
        return {
            "trainer.log": ("trainer", "trainer"),
            "orchestrator.log": ("orch", "orchestrator"),
            "inference.log": ("infer", "inference"),
            "evals.log": ("evals", "evals"),
            "eval.log": ("evals", "eval"),
        }.get(parts[0], ("other", parts[0]))
    if parts[0] == "trainer":
        if parts[1] == "torchrun":  # trainer/torchrun/<rdzv>/attempt_0/<rank>/std{out,err}.log
            return "trainer", f"rank{parts[-2]}/{rel.stem}"
        return "trainer", rel.stem
    if parts[0] == "inference":
        return "infer", rel.stem
    if parts[0] == "envs" and len(parts) == 3:  # envs/<split>/<env>.log
        return f"env:{rel.stem}", f"{rel.stem} ({parts[1]})"
    return "other", str(rel.with_suffix(""))


@app.get("/api/runs/{run}/logfiles")
def list_logfiles(run: str, attempt: str = "latest") -> dict:
    run_dir = get_run_dir(run)
    attempts = attempt_numbers(run_dir)
    if attempt == "latest":
        latest = (run_dir / "logs" / "latest").resolve()
        attempt_num = (
            int(latest.name.removeprefix("attempt_")) if latest.is_dir() else (attempts[-1] if attempts else 0)
        )
    else:
        attempt_num = int(attempt)
    attempt_dir = run_dir / "logs" / f"attempt_{attempt_num}"
    files = []

    def add(path: Path, component: str, label: str) -> None:
        real = path.resolve()  # multi-node masters are symlinks to node_0 logs
        if not real.is_file():
            return
        files.append(
            {
                "id": str(path.relative_to(run_dir)),
                "component": component,
                "label": label,
                "size": real.stat().st_size,
                "master": path.name in MASTER_LOGS and path.parent == attempt_dir,
            }
        )

    if attempt_dir.is_dir():
        for path in sorted(attempt_dir.rglob("*.log")):
            component, label = log_component(path.relative_to(attempt_dir))
            add(path, component, label)
    # Include unscoped env logs from runs created before attempt-scoped eval logging.
    for path in sorted((run_dir / "logs" / "envs").rglob("*.log")):
        add(path, f"env:{path.stem}", f"{path.stem} (evals)")
    return {"attempt": attempt_num, "attempts": attempts, "files": files}


@app.get("/api/runs/{run}/log")
def read_log(run: str, file: str, start: int | None = None, end: int | None = None, tail: int | None = None) -> dict:
    path = safe_child(get_run_dir(run), file)
    size = path.stat().st_size
    if tail is not None:
        start = max(0, size - tail)
    start = min(start or 0, size)
    with path.open("rb") as f:
        f.seek(start)
        data = f.read(min(MAX_LOG_CHUNK, (end if end is not None else size) - start))
    if tail is not None and start > 0:  # snap the head to a line boundary
        cut = data.find(b"\n")
        if cut != -1:
            start += cut + 1
            data = data[cut + 1 :]
    chunk_end = start + len(data)
    if end is None and chunk_end == size:
        # Read to EOF: drop a partially-written trailing line so follow-mode gets whole lines.
        last_nl = data.rfind(b"\n")
        if last_nl != -1 and last_nl + 1 < len(data):
            data = data[: last_nl + 1]
            chunk_end = start + len(data)
    return {"text": data.decode("utf-8", errors="replace"), "start": start, "end": chunk_end}


# ------------------------------------------------------------------------- configs

CONFIG_ORDER = ["rl", "sft", "eval", "evals", "orchestrator", "trainer", "inference"]


def config_rank(name: str) -> tuple[int, str]:
    stem = name.split("/")[0].removesuffix(".toml")
    return (CONFIG_ORDER.index(stem) if stem in CONFIG_ORDER else len(CONFIG_ORDER), name)


@app.get("/api/runs/{run}/configs")
def list_configs(run: str, attempt: str = "latest") -> dict:
    """Two views of a run's config: each launch TOML (verbatim, as the run was
    started) and one "resolved" document concatenating every resolved JSON dump."""
    run_dir = get_run_dir(run)
    configs_dir, attempt_num = config_attempt_dir(run_dir, attempt)
    files = sorted((p.name for p in configs_dir.glob("*.toml")), key=config_rank) if configs_dir.is_dir() else []
    if any(resolved_config_dir(run_dir, attempt).rglob("*.json")):
        files.append("resolved")
    if (configs_dir / "command.txt").is_file():
        files.append("command.txt")
    return {"attempt": attempt_num, "attempts": config_attempt_numbers(run_dir), "files": files}


@app.get("/api/runs/{run}/config")
def read_config(run: str, file: str, attempt: str = "latest") -> dict:
    run_dir = get_run_dir(run)
    if file == "resolved":
        base = resolved_config_dir(run_dir, attempt)
        names = sorted((str(p.relative_to(base).with_suffix("")) for p in base.rglob("*.json")), key=config_rank)
        doc = {name: read_json(base / f"{name}.json") for name in names}
        return {"file": file, "content": orjson.dumps(doc).decode()}
    configs_dir, _ = config_attempt_dir(run_dir, attempt)
    path = safe_child(configs_dir, file, suffix=".txt" if file == "command.txt" else ".toml")
    return {"file": file, "content": path.read_text()}


# ------------------------------------------------------------------------- reports


def report_title(path: Path) -> str | None:
    """`title:` from the frontmatter block, scanning only the first bytes of the file."""
    try:
        with path.open("r", errors="replace") as f:
            head = f.read(4096)
    except OSError:
        return None
    if not head.startswith("---"):
        return None
    for line in head.splitlines()[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        if key.strip() == "title":
            return value.strip().strip("\"'") or None
    return None


@app.get("/api/runs/{run}/reports")
def list_reports(run: str) -> dict:
    """Markdown reports under <run>/reports/, newest first."""
    reports_dir = get_run_dir(run) / "reports"
    rows = []
    for path in reports_dir.glob("*.md") if reports_dir.is_dir() else []:
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append({"file": path.name, "title": report_title(path), "mtime": stat.st_mtime, "size": stat.st_size})
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return {"reports": rows}


@app.get("/api/runs/{run}/report")
def read_report(run: str, file: str) -> dict:
    path = safe_child(get_run_dir(run) / "reports", file, suffix=".md")
    stat = path.stat()
    return {"file": file, "text": path.read_text(errors="replace"), "mtime": stat.st_mtime, "size": stat.st_size}


# ------------------------------------------------------------------------- metrics


MAX_METRICS_CHUNK = 4 * 1024 * 1024
"""Per-response cap on /metrics: huge runs stream in chunks the client loops over,
so the first charts paint long before a 100MB metrics.jsonl finishes loading."""


@app.get("/api/runs/{run}/metrics")
def read_metrics(run: str, offset: int = 0) -> dict:
    path = metrics_file(get_run_dir(run))
    if not path.is_file():
        return {"rows": [], "offset": 0, "size": 0}
    size = path.stat().st_size
    if offset > size:  # file was truncated/replaced
        offset = 0
    rows = []
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read(MAX_METRICS_CHUNK)
        if data and b"\n" not in data:  # a single line larger than the chunk
            data += f.readline()
    consumed = data.rfind(b"\n") + 1  # leave a partially-written last line for the next poll
    for line in data[:consumed].splitlines():
        try:
            rows.append(orjson.loads(line))
        except orjson.JSONDecodeError:
            continue
    return {"rows": rows, "offset": offset + consumed, "size": size}


# ------------------------------------------------------------------------ rollouts


def traces_file(run_dir: Path) -> Path | None:
    """The run's episode stream. prime-rl dumps it through the file monitor; a
    verifiers ``uv run eval`` run writes the same file at its root."""
    for path in (get_file_monitor_dir(run_dir) / "traces.jsonl", run_dir / "traces.jsonl"):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def rollout_steps(run_dir: Path) -> list[dict]:
    """The steps a cohort can be addressed by, per kind.

    Only ``effective`` ties to a step — the orchestrator step whose batch shipped the
    trace, or for eval the policy version it measured — so the steps come from the
    annotations. ``all`` is the whole arrival-ordered stream and ignores the step; it
    is offered on every listed step so the subset toggle always works."""
    if traces_file(run_dir) is None:
        return []
    by_step: dict[int, dict[str, bool]] = {}
    for kind, step in effective_steps(run_dir):
        by_step.setdefault(step, {})[f"{kind}/effective"] = True
    if not by_step:
        by_step[0] = {}
    for available in by_step.values():
        for kind in episode_kinds(run_dir):
            available[f"{kind}/all"] = True
    return [{"step": step, "available": available} for step, available in sorted(by_step.items())]


def line_offsets(path: Path) -> list[int]:
    size = path.stat().st_size
    with _lock:
        cached_size, checkpoint, offsets = _lru_get(_offsets_cache, path) or (0, b"", [])
        if cached_size > size or (cached_size and file_checkpoint(path, cached_size) != checkpoint):
            # The bytes immediately before the old EOF changed: this is a rewrite,
            # not an append. Invalidate offsets and their derived summaries together.
            cached_size, checkpoint, offsets = 0, b"", []
            _summaries_cache.pop(path, None)
            _sidecar_written.pop(path, None)
        if cached_size == size:
            return offsets
    # re-scan from the last recorded line (it may have been partial) into a local
    # list, then merge under the lock: the shared list only ever grows by strictly
    # increasing appends, so concurrent readers' indices stay valid and concurrent
    # growers can't double-append the same line start
    scan_from = offsets[-1] if offsets else 0
    found = []
    with path.open("rb") as f:
        f.seek(scan_from)
        pos = scan_from
        for line in f:
            if line.strip():
                found.append(pos)
            pos += len(line)
        scanned_size = f.tell()
    with _lock:
        cached_size, checkpoint, current = _lru_get(_offsets_cache, path) or (0, b"", offsets)
        current_matches = not cached_size or file_checkpoint(path, cached_size) == checkpoint
        if cached_size > scanned_size and current_matches:
            return current  # a concurrent reader already scanned farther
        if not current_matches:
            cached_size, current = 0, []
            _summaries_cache.pop(path, None)
            _sidecar_written.pop(path, None)
        for offset in found:
            if not current or offset > current[-1]:
                current.append(offset)
        cached_size = max(scanned_size, cached_size)
        _lru_put(_offsets_cache, path, (cached_size, file_checkpoint(path, cached_size), current))
    return current


def walk_timing(obj: dict, prefix: str, out: dict[str, float]) -> None:
    """Flatten a trace timing tree to phase -> seconds (same walk as the viewer)."""
    if isinstance(obj.get("duration"), (int, float)):
        out[prefix] = out.get(prefix, 0.0) + obj["duration"]
    elif isinstance(obj.get("start"), (int, float)) and isinstance(obj.get("end"), (int, float)):
        out[prefix] = out.get(prefix, 0.0) + (obj["end"] - obj["start"])
    for key, value in obj.items():
        if isinstance(value, dict):
            walk_timing(value, f"{prefix}/{key}" if prefix else key, out)


def episode_kind(rec: dict) -> str:
    """The kind of work an episode did. The file monitor stamps it as the episode
    lands; a stream written by another producer (a verifiers ``uv run eval`` run) is
    read off its run info instead."""
    for trace in rec.get("traces") or []:
        if (kind := (trace.get("info") or {}).get("kind")) in ("train", "eval"):
            return kind
    run = rec.get("run") or {}
    return (run.get("work") or {}).get("type") or run.get("type") or "eval"


def summarize_episode(line: int, rec: dict) -> dict:
    rewards, advantages = [], []
    input_tokens = output_tokens = turns = branches = 0
    stop_condition = None
    reward_parts: dict[str, list[float]] = {}
    metric_parts: dict[str, list[float]] = {}
    timing: dict[str, float] = {}
    costs: list[float] = []
    for trace in rec.get("traces") or []:
        costs.extend(
            usage["cost"]
            for call in trace.get("calls") or []
            if isinstance(usage := call.get("usage") or {}, dict) and isinstance(usage.get("cost"), (int, float))
        )
        nodes = trace.get("nodes") or []
        parents = {node.get("parent") for node in nodes if "parent" in node}
        branches += max(0, len(nodes) - len(parents))
        if trace.get("rewards"):  # skip reward-less seats (e.g. a judge) in the episode mean
            rewards.append(
                sum(
                    (r.get("score") or 0) * (r.get("weight") if r.get("weight") is not None else 1)
                    for r in trace["rewards"].values()
                    if isinstance(r, dict)
                )
            )
        for name, r in (trace.get("rewards") or {}).items():
            if isinstance(r, dict) and isinstance(r.get("score"), (int, float)):
                reward_parts.setdefault(name, []).append(r["score"])
        for name, value in (trace.get("metrics") or {}).items():
            if isinstance(value, (int, float)):
                metric_parts.setdefault(name, []).append(value)
        if isinstance(trace.get("timing"), dict):
            walk_timing(trace["timing"], "", timing)
        advantage = (trace.get("info") or {}).get("advantage")
        if advantage is not None:
            advantages.append(advantage)
        for node in nodes:
            n_tokens = len(node.get("token_ids") or [])
            if node.get("sampled"):
                output_tokens += n_tokens
            else:
                input_tokens += n_tokens
            if (node.get("message") or {}).get("role") == "assistant":
                turns += 1
        stop_condition = trace.get("stop_condition", stop_condition)
        if input_tokens == 0 and output_tokens == 0:  # some eval traces carry no token arrays
            for call in trace.get("calls") or []:
                usage = call.get("usage") or {}
                input_tokens += usage.get("prompt_tokens") or 0
                output_tokens += usage.get("completion_tokens") or 0
    first_info = ((rec.get("traces") or [{}])[0].get("info")) or {}
    return {
        "rewards": {name: sum(v) / len(v) for name, v in reward_parts.items()},
        "metrics": {name: sum(v) / len(v) for name, v in metric_parts.items()},
        "timing": timing,
        "cost": sum(costs) if costs else None,
        "line": line,
        "id": rec.get("id"),
        "kind": episode_kind(rec),
        "dispatch_step": (first_info.get("dispatch") or {}).get("step")
        or ((rec.get("run") or {}).get("work") or {}).get("step"),
        "trace_ids": [trace_id for trace in rec.get("traces") or [] if (trace_id := trace.get("id"))],
        "env": (rec.get("env") or {}).get("id") or (rec.get("env") or {}).get("name"),
        "group": (rec.get("group") or {}).get("id"),
        "ok": rec.get("ok"),
        "num_errors": len(rec.get("errors") or []),
        "reward": sum(rewards) / len(rewards) if rewards else None,
        "advantage": sum(advantages) / len(advantages) if advantages else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "turns": turns,
        "branches": branches,
        "stop_condition": stop_condition,
        # when the episode landed and how long it was alive: the stream's x axis,
        # and the one duration worth sorting a stream by
        "arrival": (first_info.get("arrival") or {}).get("time"),
        "duration": _episode_duration(first_info, timing),
    }


def _episode_duration(info: dict, timing: dict[str, float]) -> float | None:
    """Wall-clock seconds from dispatch to arrival, or the summed phases for a record
    that predates those stamps."""
    dispatched = (info.get("dispatch") or {}).get("time")
    arrived = (info.get("arrival") or {}).get("time")
    if isinstance(dispatched, (int, float)) and isinstance(arrived, (int, float)):
        return max(0.0, arrived - dispatched)
    phases = [value for key, value in timing.items() if "/" not in key]
    return sum(phases) if phases else None


def episode_summaries(path: Path) -> list[dict]:
    offsets = line_offsets(path)
    with _lock:
        cached_count, summaries = _lru_get(_summaries_cache, path) or (0, [])
        if cached_count > len(offsets):
            cached_count, summaries = 0, []
    if cached_count == 0:
        loaded = load_sidecar(path)
        if loaded is not None:
            cached_count, summaries = len(loaded), loaded
            cached_count = min(cached_count, len(offsets))
    if cached_count == len(offsets):
        with _lock:
            _lru_put(_summaries_cache, path, (cached_count, summaries))
        return summaries[:cached_count] if len(summaries) != cached_count else summaries
    summaries = list(summaries[:cached_count])
    with path.open("rb") as f:
        f.seek(offsets[cached_count])
        for line_no in range(cached_count, len(offsets)):
            raw = f.readline()
            try:
                summaries.append(summarize_episode(line_no, orjson.loads(raw)))
            except orjson.JSONDecodeError:
                summaries.append({"line": line_no, "id": None, "error": "unparseable"})
    with _lock:
        _lru_put(_summaries_cache, path, (len(offsets), summaries))
    write_sidecar(path, summaries)
    return summaries


# ------------------------------------------------------- summary sidecars
# Parsing a step's traces file for the table can mean reading gigabytes; the
# result is persisted outside the run dir (the dashboard never writes there),
# so a revisit — or a dashboard restart — skips the parse entirely.
SIDECAR_DIR = STATE_DIR
SIDECAR_FORMAT = 4
SIDECAR_WRITE_INTERVAL_S = 20.0
_sidecar_written: dict[Path, tuple[float, int]] = {}  # path -> (last write time, count)


def sidecar_path(path: Path) -> Path:
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]
    return SIDECAR_DIR / f"{digest}.json"


def file_checkpoint(path: Path, end: int) -> bytes:
    """Bytes immediately before a previously observed EOF.

    Appends preserve this window; rewrites and truncate-then-regrow resumes do
    not. Checking 64 bytes stays constant-time even for multi-gigabyte traces.
    """
    with path.open("rb") as f:
        start = max(0, end - 64)
        f.seek(start)
        return f.read(end - start)


def load_sidecar(path: Path) -> list[dict] | None:
    data = read_json(sidecar_path(path))
    try:
        if (
            data.get("format") != SIDECAR_FORMAT
            or data.get("path") != str(path.resolve())
            or path.stat().st_size < data["size"]
            or data.get("checkpoint") != file_checkpoint(path, data["size"]).hex()
        ):
            return None
        return data["summaries"]
    except (KeyError, OSError):
        return None


def write_sidecar(path: Path, summaries: list[dict]) -> None:
    now = time.monotonic()
    last_time, last_count = _sidecar_written.get(path, (0.0, -1))
    if len(summaries) == last_count or now - last_time < SIDECAR_WRITE_INTERVAL_S:
        return
    _sidecar_written[path] = (now, len(summaries))
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    size = path.stat().st_size
    payload = {
        "format": SIDECAR_FORMAT,
        "path": str(path.resolve()),
        "size": size,
        "checkpoint": file_checkpoint(path, size).hex(),
        "summaries": summaries,
    }
    target = sidecar_path(path)
    tmp = target.with_suffix(".tmp")
    tmp.write_bytes(orjson.dumps(payload))
    tmp.replace(target)


def traces_path(run: str, step: int, kind: str, subset: str) -> Path:
    """The run's episode stream. ``step``/``kind``/``subset`` never select a file —
    every episode is on disk exactly once — they select the read-time filter."""
    if kind not in ("train", "eval") or subset not in ("all", "effective"):
        raise HTTPException(400, "kind must be train|eval, subset all|effective")
    path = traces_file(get_run_dir(run))
    if path is None:
        raise HTTPException(404, "no traces for this run")
    return path


def read_episode_record(path: Path, line: int) -> dict:
    offsets = line_offsets(path)
    if not 0 <= line < len(offsets):
        raise HTTPException(404, "episode line out of range")
    with path.open("rb") as f:
        f.seek(offsets[line])
        return orjson.loads(f.readline())


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return "" if content is None else str(content)


def timeline_status(trace: dict) -> str:
    if not trace.get("is_completed"):
        return "running"
    if not trace.get("ok") and (trace.get("errors") or trace.get("stop_condition") == "error"):
        return "failed"
    return "completed"


def timeline_reward(trace: dict) -> float | None:
    rewards = [reward for reward in (trace.get("rewards") or {}).values() if isinstance(reward, dict)]
    if not rewards:
        return None
    return sum(
        (reward.get("score") or 0) * (reward.get("weight") if reward.get("weight") is not None else 1)
        for reward in rewards
    )


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def annotations_files(run_dir: Path) -> list[Path]:
    annotations_dir = get_file_monitor_dir(run_dir) / "annotations"
    return sorted(annotations_dir.glob("*.jsonl")) if annotations_dir.is_dir() else []


def run_annotations(run_dir: Path) -> dict[str, list[dict]]:
    """Every trace update for the run, by trace id, in producer order. Cached by the
    annotation files' sizes — they are append-only, so unchanged sizes mean an
    unchanged index."""
    files = annotations_files(run_dir)
    key = tuple((path.name, _file_size(path)) for path in files)
    cache_key = get_file_monitor_dir(run_dir) / "annotations"
    with _lock:
        cached = _lru_get(_annotations_cache, cache_key)
        if cached and cached[0] == key:
            return cached[1]
    index: dict[str, list[dict]] = {}
    for path in files:
        try:
            lines = path.read_bytes().splitlines()
        except OSError:
            continue
        for raw in lines:
            if not raw:
                continue
            try:
                update = orjson.loads(raw)
            except orjson.JSONDecodeError:
                continue  # torn tail line of a live append
            if trace_id := update.get("trace_id"):
                index.setdefault(trace_id, []).append(update)
    with _lock:
        _lru_put(_annotations_cache, cache_key, (key, index))
    return index


def trace_ship_steps(run_dir: Path) -> dict[str, int]:
    """Trace id -> the step its cohort shipped at, for traces marked effective."""
    steps: dict[str, int] = {}
    for trace_id, updates in run_annotations(run_dir).items():
        for update in updates:
            info = update.get("info") or {}
            step = (info.get("ship") or {}).get("step")
            if info.get("effective") and isinstance(step, int):
                steps[trace_id] = step
    return steps


def ipo_eps(run_dir: Path) -> float:
    """The run's IPO stable-mask threshold, for the frontend's stable-mask overlay."""
    loss = read_json(resolved_config_dir(run_dir) / "trainer.json").get("loss") or {}
    eps = loss.get("eps") if loss.get("type") == "ipo" else None
    return eps if isinstance(eps, (int, float)) else 0.1


def token_usage(usage: dict) -> tuple[int | None, int | None, int | None]:
    prompt_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if "cached_input_tokens" in usage:
        return prompt_tokens, usage.get("cached_input_tokens"), output_tokens
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    input_tokens = (
        max(0, prompt_tokens - cached_tokens) if prompt_tokens is not None and cached_tokens else prompt_tokens
    )
    return input_tokens, cached_tokens, output_tokens


def activity_spans(
    trace: dict,
    node_indexes: set[int],
    *,
    include_unlinked: bool = False,
    shared_node_indexes: set[int] | None = None,
) -> list[dict]:
    nodes = trace.get("nodes") or []
    spans = []
    for call_index, call in enumerate(trace.get("calls") or []):
        node_index = call.get("node")
        if node_index is None:
            if not include_unlinked:
                continue
            node = {}
        elif node_index not in node_indexes:
            continue
        else:
            node = nodes[node_index]
        call_time = call.get("time") or {}
        started = call_time.get("start")
        ended = call_time.get("end")
        usage = call.get("usage") or {}
        input_tokens, cached_tokens, output_tokens = token_usage(usage)
        reasoning_tokens = usage.get("reasoning_tokens")
        if reasoning_tokens is None:
            reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        text = " ".join(message_text(node.get("message") or {}).split())
        spans.append(
            {
                "kind": "model_call",
                "label": f"turn {len(spans) + 1}",
                "track": "activity",
                "call_index": call_index,
                "node_index": node_index,
                "shared": node_index in shared_node_indexes if shared_node_indexes is not None else False,
                "started_at": started,
                "ended_at": ended,
                "status": "completed" if ended is not None or trace.get("is_completed") else "running",
                "snippet": text[:240],
                "input_tokens": input_tokens,
                "cached_tokens": cached_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "cost": usage.get("cost"),
            }
        )
    return sorted(spans, key=lambda span: span["started_at"] if span["started_at"] is not None else float("inf"))


def lifecycle_spans(trace: dict) -> list[dict]:
    timing = trace.get("timing") or {}
    spans = []
    for kind, label in (
        ("boot", "boot"),
        ("setup", "setup"),
        ("agent", "agent"),
        ("finalize", "finalize"),
        ("scoring", "scoring"),
    ):
        span = timing.get(kind) or {}
        if not span.get("start"):
            continue
        spans.append(
            {
                "kind": kind,
                "label": label,
                "track": "lifecycle",
                "started_at": span["start"],
                "ended_at": span.get("end") or None,
                "status": "completed" if span.get("end") or trace.get("is_completed") else "running",
            }
        )
    return spans


def timeline_lane(
    trace: dict,
    trace_index: int,
    *,
    label: str,
    depth: int,
    branch: bool,
    lifecycle: list[dict],
    activities: list[dict],
    started_at: float | None = None,
) -> dict:
    spans = lifecycle + activities
    starts = [span["started_at"] for span in spans if span.get("started_at") is not None]
    ends = [span["ended_at"] for span in spans if span.get("ended_at") is not None]
    started = started_at if started_at is not None else min(starts or ends, default=None)
    status = (
        ("completed" if all(span["status"] == "completed" for span in lifecycle + activities) else "running")
        if branch
        else timeline_status(trace)
    )
    ended = max(ends, default=None) if status != "running" else None
    agent = trace.get("agent") or {}
    config = agent.get("config") or {}
    client = config.get("client") or {}

    def total(field: str) -> int | float | None:
        values = [span[field] for span in activities if isinstance(span.get(field), (int, float))]
        return sum(values) if values else None

    input_tokens = total("input_tokens")
    cached_tokens = total("cached_tokens")
    output_tokens = total("output_tokens")
    reasoning_tokens = total("reasoning_tokens")
    total_input_tokens = input_tokens + (cached_tokens or 0) if input_tokens is not None else None
    total_tokens = (
        total_input_tokens + output_tokens if total_input_tokens is not None and output_tokens is not None else None
    )
    context_lengths = [
        (span.get("input_tokens") or 0) + (span.get("cached_tokens") or 0)
        for span in activities
        if span.get("input_tokens") is not None
    ]
    return {
        "trace_index": trace_index,
        "label": label,
        "model": config.get("model") or client.get("renderer_model_name") or "",
        "depth": depth,
        "started_at": started,
        "ended_at": ended,
        "status": status,
        "outcome": status if branch else (trace.get("stop_condition") or status),
        "reward": timeline_reward(trace),
        "usage": {
            "model_calls": len(activities),
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "total_input_tokens": total_input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "max_context_tokens": max(context_lengths, default=None),
            "total_tokens": total_tokens,
            "cost": total("cost"),
        },
        "spans": spans,
    }


def project_episode_timeline(episode: dict) -> dict:
    lane_groups = []
    for trace_index, trace in enumerate(episode.get("traces") or []):
        nodes = trace.get("nodes") or []
        branch_paths = branch_node_paths(nodes)
        node_branch_counts: dict[int, int] = {}
        for path in branch_paths:
            for node_index in path:
                node_branch_counts[node_index] = node_branch_counts.get(node_index, 0) + 1
        role = (trace.get("agent") or {}).get("name") or "agent"
        parent = timeline_lane(
            trace,
            trace_index,
            label=role,
            depth=0,
            branch=False,
            lifecycle=lifecycle_spans(trace),
            activities=activity_spans(trace, set(), include_unlinked=True),
            started_at=(trace.get("timing") or {}).get("start"),
        )
        branches = []
        for branch_index, path in enumerate(branch_paths):
            node_indexes = set(path)
            shared_node_indexes = {node_index for node_index in path if node_branch_counts.get(node_index, 0) > 1}
            unique_node_indexes = node_indexes - shared_node_indexes
            activities = activity_spans(
                trace,
                node_indexes,
                shared_node_indexes=shared_node_indexes,
            )
            path_timestamps = [
                nodes[node_index].get("timestamp")
                for node_index in path
                if nodes[node_index].get("timestamp") is not None
            ]
            unique_timestamps = [
                nodes[node_index].get("timestamp")
                for node_index in unique_node_indexes
                if nodes[node_index].get("timestamp") is not None
            ]
            activity_starts = [span["started_at"] for span in activities if span["started_at"] is not None]
            unique_activity_starts = [
                span["started_at"] for span in activities if not span["shared"] and span["started_at"] is not None
            ]
            branch_start = min(activity_starts or path_timestamps, default=None)
            sort_start = min(
                unique_activity_starts or unique_timestamps or activity_starts or path_timestamps, default=None
            )
            branch_end = max(
                [span["ended_at"] for span in activities if span.get("ended_at") is not None] + path_timestamps,
                default=None,
            )
            branch_completed = bool(trace.get("is_completed"))
            label = f"branch {branch_index}"
            branch_lifecycle = (
                [
                    {
                        "kind": "agent",
                        "label": label,
                        "track": "lifecycle",
                        "started_at": branch_start,
                        "ended_at": branch_end if branch_completed else None,
                        "status": "completed" if branch_completed else "running",
                        "node_index": path[-1],
                    }
                ]
                if branch_start is not None
                else []
            )
            branches.append(
                (
                    sort_start,
                    branch_index,
                    timeline_lane(
                        trace,
                        trace_index,
                        label=label,
                        depth=1,
                        branch=True,
                        lifecycle=branch_lifecycle,
                        activities=activities,
                    ),
                )
            )
        branches.sort(key=lambda item: (item[0] if item[0] is not None else float("inf"), item[1]))
        lane_groups.append((parent, [lane for _, _, lane in branches]))
    lane_groups.sort(key=lambda group: group[0]["started_at"] if group[0]["started_at"] is not None else float("inf"))
    lanes = [lane for parent, children in lane_groups for lane in (parent, *children)]
    return {"lanes": lanes}


NICE_BINS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400]


def nice_bin(ideal: float) -> float:
    """The smallest round interval at least this wide, so bars land on seconds,
    minutes or hours a reader recognises rather than 93.4s."""
    return next((step for step in NICE_BINS if step >= ideal), NICE_BINS[-1])


SORT_KEYS = {"arrival", "duration", "reward", "output_tokens", "turns", "group"}


def episode_rows(run_dir: Path) -> list[dict]:
    """The run's episode index: one compact row per line of the stream, in arrival
    order, with the cohort step its annotations give it. Parsed incrementally and
    persisted, so a poll only reads the lines that arrived since the last one."""
    path = traces_file(run_dir)
    if path is None:
        return []
    ship_steps = trace_ship_steps(run_dir)
    rows = []
    for summary in episode_summaries(path):
        steps = {step for trace_id in summary.get("trace_ids") or [] if (step := ship_steps.get(trace_id)) is not None}
        rows.append({**summary, "step": min(steps) if steps else None})
    return rows


def filter_rows(
    rows: list[dict],
    *,
    step: int | None = None,
    kind: str | None = None,
    env: str | None = None,
    errors_only: bool = False,
    start: float | None = None,
    end: float | None = None,
) -> list[dict]:
    """Narrow the index. A step selects the cohort shipped at it — for train work the
    orchestrator step, for eval the policy version — which is the only sense in which
    an episode belongs to a step."""
    if step is not None:
        rows = [row for row in rows if row.get("step") == step]
    if kind:
        rows = [row for row in rows if row.get("kind") == kind]
    if env:
        rows = [row for row in rows if row.get("env") == env]
    if errors_only:
        rows = [row for row in rows if row.get("num_errors") or not row.get("ok")]
    if start is not None:
        rows = [row for row in rows if (row.get("arrival") or 0) >= start]
    if end is not None:
        rows = [row for row in rows if (row.get("arrival") or 0) < end]
    return rows


@app.get("/api/runs/{run}/episodes")
def list_stream_episodes(
    run: str,
    step: int | None = None,
    kind: str | None = None,
    env: str | None = None,
    episode: str | None = None,
    errors_only: bool = False,
    sort: str = "arrival",
    order: str = "desc",
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=128, ge=1, le=512),
    start: float | None = None,
    end: float | None = None,
    etag: str | None = None,
) -> dict:
    """One page of the stream. The client scrolls by asking for the next offset, so a
    run of any length costs the same to browse."""
    run_dir = get_run_dir(run)
    path = traces_file(run_dir)
    if path is None:
        raise HTTPException(404, "no traces for this run")
    current_etag = subset_etag(run_dir, "effective", path)
    if etag is not None and etag == current_etag and offset == 0:
        return {"unchanged": True, "etag": current_etag}
    rows = episode_rows(run_dir)
    envs = sorted({row["env"] for row in rows if row.get("env")})
    kinds = [name for name in ("train", "eval") if any(row.get("kind") == name for row in rows)]
    rows = filter_rows(rows, step=step, kind=kind, env=env, errors_only=errors_only, start=start, end=end)
    if episode is not None:
        rows = [row for row in rows if row.get("id") == episode]
    if sort in SORT_KEYS:
        blank = "" if sort == "group" else 0
        rows = sorted(rows, key=lambda row: (row.get(sort) is None, row.get(sort) or blank), reverse=order == "desc")
    return {
        "etag": current_etag,
        "total": len(rows),
        "offset": offset,
        "envs": envs,
        "kinds": kinds,
        "episodes": rows[offset : offset + limit],
    }


@app.get("/api/runs/{run}/episodes/histogram")
def episode_histogram(
    run: str,
    step: int | None = None,
    kind: str | None = None,
    env: str | None = None,
    errors_only: bool = False,
    bars: int = Query(default=80, ge=8, le=500),
) -> dict:
    """Episodes finishing per time bin, over the same filters as the table — the
    stream's shape, and the thing you click to narrow it to a moment. The range fits
    the episodes and the bin is the roundest interval that keeps the bar count sane,
    so a run of any length reads the same."""
    run_dir = get_run_dir(run)
    rows = filter_rows(episode_rows(run_dir), step=step, kind=kind, env=env, errors_only=errors_only)
    arrivals = sorted(row["arrival"] for row in rows if isinstance(row.get("arrival"), (int, float)))
    if not arrivals:
        return {"bins": [], "bin": 60, "start": None, "end": None, "total": 0}
    bin = nice_bin(max(arrivals[-1] - arrivals[0], 1e-9) / bars)
    # square the edges to bin boundaries so the bars carry round timestamps
    start = math.floor(arrivals[0] / bin) * bin
    end = math.floor(arrivals[-1] / bin) * bin + bin
    counts: dict[int, int] = {}
    for arrival in arrivals:
        index = int((arrival - start) // bin)
        counts[index] = counts.get(index, 0) + 1
    span = max(1, int(round((end - start) / bin)))
    return {
        "bins": [[start + index * bin, counts.get(index, 0)] for index in range(span)],
        "bin": bin,
        "start": start,
        "end": end,
        "total": sum(counts.values()),
    }


@app.get("/api/runs/{run}/rollouts")
def list_rollouts(run: str) -> dict:
    return {"steps": rollout_steps(get_run_dir(run))}


def subset_etag(run_dir: Path, subset: str, path: Path) -> str:
    """The effective listing also moves when annotations grow, so its etag covers
    their total size too."""
    tag = str(path.stat().st_size)
    if subset == "effective":
        tag += f"-{sum(_file_size(file) for file in annotations_files(run_dir))}"
    return tag


def episode_kinds(run_dir: Path) -> list[str]:
    """The kinds of work the stream holds, for the kind toggle."""
    path = traces_file(run_dir)
    kinds = {summary.get("kind") for summary in episode_summaries(path)} if path is not None else set()
    return [kind for kind in ("train", "eval") if kind in kinds]


def effective_steps(run_dir: Path) -> set[tuple[str, int]]:
    """The (kind, step) pairs an effective cohort can be addressed by."""
    path = traces_file(run_dir)
    if path is None:
        return set()
    ship_steps = trace_ship_steps(run_dir)
    return {
        (summary.get("kind"), step)
        for summary in episode_summaries(path)
        for trace_id in summary.get("trace_ids") or []
        if (step := ship_steps.get(trace_id)) is not None
    }


def get_tokenizer(model: str):
    with _lock:
        if model in _tokenizer_cache:
            return _tokenizer_cache[model]
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_pretrained(model)
    except Exception:
        tokenizer = None
    with _lock:
        _tokenizer_cache[model] = tokenizer
    return tokenizer


def decode_pieces(model: str, ids: list[int]) -> list[str] | None:
    tokenizer = get_tokenizer(model)
    if tokenizer is None:
        return None
    pieces = []
    for token_id in ids:
        piece = _piece_cache.get((model, token_id))
        if piece is None:
            piece = tokenizer.decode([token_id], skip_special_tokens=False)
            _piece_cache[(model, token_id)] = piece
        pieces.append(piece)
    return pieces


def trace_node_paths(trace: dict) -> list[list[int]]:
    """Root-to-leaf node indexes in the same order as the trace viewer."""
    nodes = trace.get("nodes") or []
    has_child = {node.get("parent") for node in nodes if isinstance(node, dict) and isinstance(node.get("parent"), int)}
    paths = []
    for leaf in (index for index in range(len(nodes)) if index not in has_child):
        path = []
        seen = set()
        index = leaf
        while isinstance(index, int) and 0 <= index < len(nodes) and index not in seen:
            seen.add(index)
            path.append(index)
            parent = nodes[index].get("parent") if isinstance(nodes[index], dict) else None
            index = parent if isinstance(parent, int) else None
        paths.append(list(reversed(path)))
    return paths


def rendered_token_text(trace: dict, model: str | None) -> dict:
    """Decode recorded post-renderer IDs as full branch sequences."""
    nodes = trace.get("nodes") or []
    paths = trace_node_paths(trace)
    if not any(isinstance(node, dict) and node.get("token_ids") for node in nodes):
        return {"status": "missing_token_ids", "model": model, "paths": []}
    if not model:
        return {"status": "missing_model", "model": None, "paths": []}
    tokenizer = get_tokenizer(model)
    if tokenizer is None:
        return {"status": "tokenizer_unavailable", "model": model, "paths": []}

    def decode_path(path: list[int]) -> dict:
        ids = [token_id for index in path for token_id in (nodes[index].get("token_ids") or [])]
        try:
            text = tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            text = None
        return {"nodes": path, "token_count": len(ids), "text": text}

    rendered_paths = [decode_path(path) for path in paths]
    all_nodes = list(range(len(nodes)))
    all_nodes_rendered = decode_path(all_nodes)
    status = "ok" if all(path["text"] is not None for path in rendered_paths + [all_nodes_rendered]) else "decode_error"
    return {
        "status": status,
        "model": model,
        "paths": rendered_paths,
        "all_nodes": all_nodes_rendered,
    }


@app.get("/api/runs/{run}/episodes/series")
def episode_series(run: str, kind: str | None = None, etag: str | None = None, after: int = 0) -> dict:
    """Per-episode series over the stream (x = arrival order): reward, shape, and the
    nested rewards/metrics/timing keys — the metrics view for eval runs. `after` returns
    only episodes past that index, so a growing stream ships increments, not the world."""
    run_dir = get_run_dir(run)
    path = traces_file(run_dir)
    if path is None:
        raise HTTPException(404, "no traces for this run")
    current_etag = subset_etag(run_dir, "effective", path)
    if etag is not None and etag == current_etag:
        return {"unchanged": True, "etag": current_etag}
    summaries = [s for s in episode_summaries(path) if not kind or s.get("kind") == kind]
    keys: set[str] = set()
    for s in summaries:
        keys.update(
            k
            for k in ("reward", "advantage", "cost", "turns", "branches", "input_tokens", "output_tokens")
            if s.get(k) is not None
        )
        for group in ("rewards", "metrics", "timing"):
            keys.update(f"{group}/{name}" for name in s.get(group) or {})

    def value(s: dict, key: str):
        group, _, name = key.partition("/")
        return (s.get(group) or {}).get(name) if name else s.get(key)

    tail = summaries[max(0, after) :]
    series = {key: [value(s, key) for s in tail] for key in sorted(keys)}
    return {"etag": current_etag, "count": len(summaries), "after": max(0, after), "series": series}


def read_episode_at(path: Path, line: int) -> dict:
    offsets = line_offsets(path)
    if not 0 <= line < len(offsets):
        raise HTTPException(404, "episode line out of range")
    with path.open("rb") as f:
        f.seek(offsets[line])
        return orjson.loads(f.readline())


@app.get("/api/runs/{run}/episodes/{line}")
def get_episode(
    run: str,
    line: int,
    tokens: bool = False,
    rendered: bool = False,
) -> dict:
    """One episode, by its line in the stream, with every annotation folded on."""
    path = traces_file(get_run_dir(run))
    if path is None:
        raise HTTPException(404, "no traces for this run")
    rec = read_episode_at(path, line)
    run_dir = get_run_dir(run)
    annotations = run_annotations(run_dir)
    if annotations:
        for trace in rec.get("traces") or []:
            updates = annotations.get(trace.get("id") or "")
            if not updates:
                continue
            stamped = fold_trace_updates(trace, updates)
            if stamped:
                ship_step = ((trace.get("info") or {}).get("ship") or {}).get("step")
                trace["train_annotations"] = {"step": ship_step, "eps": ipo_eps(run_dir), "nodes": stamped}
    if not tokens and not rendered:
        return rec
    fallback_model = model_name(main_config(run_dir)[1])
    for trace in rec.get("traces") or []:
        client = ((trace.get("agent") or {}).get("config") or {}).get("client") or {}
        model = client.get("renderer_model_name") or fallback_model
        if tokens and model:
            for node in trace.get("nodes") or []:
                if node.get("token_ids"):
                    node["token_strs"] = decode_pieces(model, node["token_ids"])
        if rendered:
            trace["rendered_tokens"] = rendered_token_text(trace, model)
    return rec


# --------------------------------------------------------------------- view command
#
# The agent-facing control plane: a local agent that already knows the on-disk
# address of what it wants shown POSTs that address here, and every connected
# browser tab navigates there. One-way fan-out over SSE — no WebSocket dependency,
# and the browser's EventSource reconnects on its own. The dashboard stays a
# filesystem reader: commands carry addresses and short highlight anchors, never
# report or episode payloads.

VIEW_TABS = {"metrics", "config", "traces", "logs", "report"}
VIEW_KEYS = {"run", "tab", "step", "kind", "subset", "episode", "line", "trace", "branch", "report", "highlight"}
HIGHLIGHT_KEYS = {"node", "quote", "prefix", "suffix", "reason", "field"}

_view_command: dict | None = None
_view_seq = 0
_view_clients: set["asyncio.Queue"] = set()
_view_url = ""  # actual serve url, stamped by main() for the 409 body
MAX_VIEW_QUEUE = 32


def normalize_view_int(obj: dict, key: str, minimum: int) -> int | None:
    value = obj.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise HTTPException(400, f"{key} must be an integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{key} must be an integer") from None
    if isinstance(value, float) and not value.is_integer():
        raise HTTPException(400, f"{key} must be an integer")
    if normalized < minimum:
        raise HTTPException(400, f"{key} must be >= {minimum}")
    obj[key] = normalized
    return normalized


def validate_view_command(cmd: dict) -> dict:
    """Check the command against the filesystem so the agent cannot point at nothing.
    Returns the command with an episode id resolved to its line number."""
    unknown = set(cmd) - VIEW_KEYS
    if unknown:
        raise HTTPException(400, f"unknown fields: {sorted(unknown)} (known: {sorted(VIEW_KEYS)})")
    run = cmd.get("run")
    if not isinstance(run, str) or not run:
        raise HTTPException(400, "run must be a non-empty string")
    run_dir = get_run_dir(run)
    tab = cmd.get("tab")
    if tab is not None and tab not in VIEW_TABS:
        raise HTTPException(400, f"tab must be one of {sorted(VIEW_TABS)}")
    report = cmd.get("report")
    if report is not None:
        if not isinstance(report, str) or not report:
            raise HTTPException(400, "report must be a non-empty string")
        name = report if report.endswith(".md") else f"{report}.md"
        safe_child(run_dir / "reports", name, suffix=".md")
        cmd["report"] = name

    step = normalize_view_int(cmd, "step", 0)
    line = normalize_view_int(cmd, "line", 0)
    trace_index = normalize_view_int(cmd, "trace", 0)
    branch_index = normalize_view_int(cmd, "branch", -1)
    episode = cmd.get("episode")
    if episode is not None and (not isinstance(episode, str) or not episode):
        raise HTTPException(400, "episode must be a non-empty string")

    kind, subset = cmd.get("kind"), cmd.get("subset")
    address = (step, kind, subset)
    if any(value is not None for value in address) and not all(value is not None for value in address):
        raise HTTPException(400, "trace addressing needs step, kind, and subset together")
    path = traces_path(run, step, kind, subset) if all(value is not None for value in address) else None

    needs_episode = episode is not None or line is not None
    if needs_episode and path is None:
        raise HTTPException(400, "episode addressing needs step, kind, and subset")
    if any(value is not None for value in (trace_index, branch_index)) or cmd.get("highlight") is not None:
        if not needs_episode:
            raise HTTPException(400, "trace, branch, and highlight need an episode or line")

    rec = None
    if episode is not None:
        matches = [s["line"] for s in episode_summaries(path) if s.get("id") == episode]
        if not matches:
            raise HTTPException(404, f"episode id {episode!r} not found in {kind}/{subset} at step {step}")
        if len(matches) > 1:
            raise HTTPException(409, f"episode id {episode!r} is not unique in {kind}/{subset} at step {step}")
        line = cmd["line"] = matches[0]
    if line is not None:
        rec = read_episode_at(path, line)

    selected_trace = None
    if rec is not None and (trace_index is not None or branch_index is not None or cmd.get("highlight") is not None):
        traces = rec.get("traces") or []
        trace_index = trace_index or 0
        if not 0 <= trace_index < len(traces):
            raise HTTPException(404, f"trace {trace_index} not found in episode")
        selected_trace = traces[trace_index]
    if branch_index is not None:
        nodes = selected_trace.get("nodes") or []
        parents = {node.get("parent") for node in nodes if "parent" in node}
        branches = [i for i in range(len(nodes)) if i not in parents]
        if branch_index != -1 and not 0 <= branch_index < len(branches):
            raise HTTPException(404, f"branch {branch_index} not found in trace {trace_index}")

    highlight = cmd.get("highlight")
    if highlight is not None:
        if not isinstance(highlight, list) or not all(isinstance(h, dict) for h in highlight):
            raise HTTPException(400, "highlight must be a list of objects")
        if len(highlight) > 50:
            raise HTTPException(400, "highlight may contain at most 50 entries")
        nodes = selected_trace.get("nodes") or []
        for h in highlight:
            unknown = set(h) - HIGHLIGHT_KEYS
            if unknown:
                raise HTTPException(
                    400, f"unknown highlight fields: {sorted(unknown)} (known: {sorted(HIGHLIGHT_KEYS)})"
                )
            node = normalize_view_int(h, "node", 0)
            if node is None or node >= len(nodes):
                raise HTTPException(404, f"highlight node {node} not found in trace {trace_index}")
            quote = h.get("quote")
            if not isinstance(quote, str) or not quote.strip():
                raise HTTPException(400, "highlight quote must be a non-empty string")
            for key in ("prefix", "suffix", "reason"):
                if key in h and not isinstance(h[key], str):
                    raise HTTPException(400, f"highlight {key} must be a string")
            if h.get("field") not in (None, "content", "reasoning"):
                raise HTTPException(400, "highlight field must be content|reasoning")
    return cmd


def _view_event(cmd: dict) -> str:
    return f"data: {orjson.dumps(cmd).decode()}\n\n"


@app.post("/api/view")
async def post_view(cmd: dict) -> dict:
    global _view_command, _view_seq
    # validation walks the filesystem (and may summarize a traces file on first
    # touch) — keep it off the event loop so open SSE streams never stall
    cmd = await asyncio.to_thread(validate_view_command, cmd)
    _view_seq += 1
    cmd["seq"] = _view_seq
    cmd["ts"] = time.time()
    _view_command = cmd
    for queue in _view_clients:
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(cmd)
    if not _view_clients:
        # stored for late-joining tabs, but the agent must not pretend it pointed
        # at something a human saw
        raise HTTPException(
            409, {"error": "no dashboard tab is connected", "url": _view_url, "stored": True, "seq": _view_seq}
        )
    return {"ok": True, "seq": _view_seq, "clients": len(_view_clients), "line": cmd.get("line")}


@app.get("/api/view/events")
async def view_events() -> "StreamingResponse":
    queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_VIEW_QUEUE)

    async def stream():
        _view_clients.add(queue)
        try:
            yield "retry: 2000\n\n"
            if _view_command is not None:
                # catch-up for a late tab: flagged so the client can decide whether
                # a stored command is still worth replaying
                yield _view_event({**_view_command, "replay": True})
            while True:
                try:
                    yield _view_event(await asyncio.wait_for(queue.get(), timeout=15))
                except asyncio.TimeoutError:
                    yield ": ping\n\n"  # flushes dead connections out of the client count
        finally:
            _view_clients.discard(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


@app.get("/api/runs/{run}/episodes/{line}/timeline")
def get_episode_timeline(run: str, line: int) -> dict:
    path = traces_file(get_run_dir(run))
    if path is None:
        raise HTTPException(404, "no traces for this run")
    return project_episode_timeline(read_episode_at(path, line))


# -------------------------------------------------------------------------- static

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def render_status(url: str):
    """The startup console: the URL plus each tracked dir with its discovered runs."""
    from rich.text import Text

    text = Text()
    text.append("\n  prime-rl dashboard", style="bold white")
    text.append(" · ", style="dim")
    text.append(url + "\n", style="underline #B6FF3C")
    runs = scan_runs()
    for base in tracked_dirs():
        text.append(f"\n  {base.resolve()}/\n", style="dim")
        names = sorted(name for name, run_dir in runs.items() if run_dir.parent == base) or ["(no runs yet)"]
        for i, name in enumerate(names):
            connector = "└── " if i == len(names) - 1 else "├── "
            text.append(f"  {connector}{name}\n", style="dim")
    return text


def free_port(host: str, start: int) -> int:
    """The first free port at or above `start`, so several dashboards on one node
    (e.g. a cluster head node) never collide."""
    import socket

    for port in range(start, start + 100):
        with socket.socket() as sock:
            # match uvicorn's own listener: without SO_REUSEADDR a just-killed
            # dashboard's TIME_WAIT sockets push restarts onto the next port
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise SystemExit(f"no free port in [{start}, {start + 100})")


def claim_daemon(url: str) -> bool:
    """Record this process as THE dashboard daemon (pid + actual url, port
    spillover included) so launchers can find it instead of starting another."""
    with registry_lock():
        existing = read_json(DAEMON_FILE)
        if existing.get("pid") and existing.get("pid") != os.getpid():
            try:
                os.kill(existing["pid"], 0)
                print(f"another dashboard daemon is already running at {existing.get('url')}", file=sys.stderr)
                return False
            except OSError:
                pass  # stale file from a dead daemon
        tmp = DAEMON_FILE.with_suffix(".tmp")
        tmp.write_bytes(orjson.dumps({"pid": os.getpid(), "url": url, "started": time.time()}))
        tmp.replace(DAEMON_FILE)
        return True


def release_daemon() -> None:
    with registry_lock():
        if read_json(DAEMON_FILE).get("pid") == os.getpid():
            DAEMON_FILE.unlink(missing_ok=True)


def main() -> None:
    global output_dirs, isolated, _view_url
    parser = argparse.ArgumentParser(description="prime-rl run dashboard")
    parser.add_argument("output_dirs", nargs="*", default=[default_output_dir()], type=Path, metavar="output_dir")
    parser.add_argument("--port", type=int, default=7788)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="serve only the given dirs: don't join the per-user registry, don't remember "
        "these dirs, and don't claim discovery (launchers will ignore this instance)",
    )
    args = parser.parse_args()
    isolated = args.isolated
    output_dirs = [d for d in args.output_dirs if d.is_dir()]
    for missing in set(args.output_dirs) - set(output_dirs):
        print(f"warning: output dir {missing} does not exist", file=sys.stderr)
    if not isolated:
        # this instance's dirs join the per-user registry, and it serves the union -
        # one dashboard per host per user covers every run
        register_dirs(output_dirs)
    if not tracked_dirs():
        raise SystemExit("no existing output dir given" + ("" if isolated else " (and none registered)"))
    set_proc_title("Dashboard")
    port = free_port(args.host, args.port)
    if port != args.port:
        print(f"port {args.port} is taken - serving on {port}", file=sys.stderr)
    args.port = port
    url = f"http://localhost:{args.port}"
    _view_url = url
    # first non-isolated instance owns discovery; extras and --isolated still serve
    claimed = False if isolated else claim_daemon(url)
    live = None
    stop = threading.Event()
    if sys.stdout.isatty():
        # live console: re-scan every few seconds so new runs show up as tracked
        from rich.console import Console
        from rich.live import Live

        live = Live(render_status(url), console=Console(), refresh_per_second=1)
        live.start()

        def watch() -> None:
            while not stop.wait(3):
                live.update(render_status(url))

        threading.Thread(target=watch, daemon=True).start()
    else:
        dirs = ", ".join(str(d.resolve()) for d in output_dirs)
        print(f"\n  prime-rl dashboard · {dirs}\n  {url}\n", flush=True)
    try:
        # short graceful-shutdown window: a Ctrl-C must not hang on the browser's
        # open keep-alive connections
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=2)
    finally:
        # always unwind the Live display - it hides the terminal cursor, and an
        # interrupt that skips its exit path would leave the shell cursorless
        stop.set()
        if live is not None:
            live.stop()
        if claimed:
            release_daemon()


if __name__ == "__main__":
    main()
