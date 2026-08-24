"""Local dashboard for prime-rl runs: logs, metrics, and rollout traces.

This package is fully AI-generated and maintained by agents - it is not meant to be read or edited by humans. Change it by asking an agent, and verify through the browser smoke tests.

Reads everything from run output directories (metrics.jsonl, logs/attempt_N,
rollouts/step_N) — no wandb or network required. Usage:

    uv run dashboard [output_dir ...] [--port 7788] [--host 127.0.0.1]

Multiple output directories can be tracked at once.
"""

import argparse
import hashlib
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import orjson

from prime_rl.entrypoints.dashboard import DAEMON_FILE, DIRS_FILE, STATE_DIR, registry_lock
from prime_rl.utils.config import default_output_dir
from prime_rl.utils.process import set_proc_title

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse
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
_offsets_cache: OrderedDict[Path, tuple[int, list[int]]] = OrderedDict()
_summaries_cache: OrderedDict[Path, tuple[int, list[dict]]] = OrderedDict()
_tokenizer_cache: dict[str, object] = {}
_piece_cache: dict[tuple[str, int], str] = {}
_json_cache: dict[Path, tuple[tuple[int, float], dict]] = {}
_steps_cache: dict[Path, tuple[float, list[int]]] = {}


def is_run_dir(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith("."):
        return False
    return any((path / marker).exists() for marker in ("configs", "logs", "metrics.jsonl", "rollouts"))


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


def size_etag(path: Path, etag: str | None) -> tuple[str, dict | None]:
    """Etag = file size: the trace files are append-only, so an unchanged size means
    an unchanged response. Returns (current_etag, short-circuit response or None)."""
    current = str(path.stat().st_size)
    if etag is not None and etag == current:
        return current, {"unchanged": True, "etag": current}
    return current, None


def model_name(config: dict) -> str | None:
    """`model` is a string on eval configs, a {name} object on rl/sft ones."""
    model = config.get("model")
    return model if isinstance(model, str) else (model or {}).get("name")


def resolved_config_dir(run_dir: Path) -> Path:
    """Where the resolved JSON dumps live: `configs/resolved/`, or `configs/` on
    runs from before the launch-TOML split."""
    configs = run_dir / "configs"
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


def step_numbers(run_dir: Path) -> list[int]:
    """Rollout step numbers, cached by the rollouts dir mtime (which changes when a
    step dir is created) — polled per run on every /api/runs tick."""
    rollouts = run_dir / "rollouts"
    try:
        mtime = rollouts.stat().st_mtime
    except OSError:
        return []
    with _lock:
        cached = _steps_cache.get(rollouts)
        if cached and cached[0] == mtime:
            return cached[1]
    steps = [n for n, _ in numbered_dirs(rollouts, "step_")]
    with _lock:
        _steps_cache[rollouts] = (mtime, steps)
    return steps


def run_meta(run_dir: Path) -> dict:
    configs = run_dir / "configs"
    run_type, config = main_config(run_dir)
    resolved = resolved_config_dir(run_dir)

    def envs(split: str) -> list[str]:
        return sorted(p.stem for p in (resolved / "envs" / split).glob("*.json"))

    steps = step_numbers(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
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
    root_traces = run_dir / "traces.jsonl"
    if updated is None and root_traces.is_file():  # eval runs have no metrics.jsonl
        updated = root_traces.stat().st_mtime
        started = configs.stat().st_mtime if configs.is_dir() else None
    return {
        "name": run_dir.name,
        "type": run_type,
        "model": model_name(config),
        "dataset": (config.get("data") or {}).get("name"),
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
    # The evals process writes its env-server logs outside attempt_N (logs/envs/eval/*.log).
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
def list_configs(run: str) -> dict:
    """Two views of a run's config: each launch TOML (verbatim, as the run was
    started) and one "resolved" document concatenating every resolved JSON dump."""
    run_dir = get_run_dir(run)
    configs_dir = run_dir / "configs"
    files = sorted((p.name for p in configs_dir.glob("*.toml")), key=config_rank) if configs_dir.is_dir() else []
    if any(resolved_config_dir(run_dir).rglob("*.json")):
        files.append("resolved")
    return {"files": files}


@app.get("/api/runs/{run}/config")
def read_config(run: str, file: str) -> dict:
    run_dir = get_run_dir(run)
    if file == "resolved":
        base = resolved_config_dir(run_dir)
        names = sorted((str(p.relative_to(base).with_suffix("")) for p in base.rglob("*.json")), key=config_rank)
        doc = {name: read_json(base / f"{name}.json") for name in names}
        return {"file": file, "content": orjson.dumps(doc).decode()}
    path = safe_child(run_dir / "configs", file, suffix=".toml")
    return {"file": file, "content": path.read_text()}


# ------------------------------------------------------------------------- metrics


MAX_METRICS_CHUNK = 4 * 1024 * 1024
"""Per-response cap on /metrics: huge runs stream in chunks the client loops over,
so the first charts paint long before a 100MB metrics.jsonl finishes loading."""


@app.get("/api/runs/{run}/metrics")
def read_metrics(run: str, offset: int = 0) -> dict:
    path = get_run_dir(run) / "metrics.jsonl"
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


RECENT_STEPS = 64
_avail_cache: dict[Path, dict[str, bool]] = {}  # step dir -> known-present subsets
_avail_scan_counter = 0


EVAL_ROOT_STEP = (0, "eval", "all")
"""The virtual address of a stepless `uv run eval` run's root traces.jsonl —
the one convention shared by the step listing and traces_path."""


def eval_root_traces(run_dir: Path) -> Path | None:
    root = run_dir / "traces.jsonl"
    return root if root.is_file() and root.stat().st_size > 0 else None


def rollout_steps(run_dir: Path) -> list[dict]:
    """Presence only — never reads trace files. Presence is monotonic (files only
    appear), so known-present subsets are cached forever; absent ones re-stat every
    poll only for the newest steps — older gaps (an eval landing late at its trigger
    step) are picked up by a full rescan every 10th call."""
    global _avail_scan_counter
    _avail_scan_counter += 1
    full_rescan = _avail_scan_counter % 10 == 1
    numbered = numbered_dirs(run_dir / "rollouts", "step_")
    recent_cutoff = numbered[-1][0] - RECENT_STEPS if numbered else 0
    steps = []
    for number, step_dir in numbered:
        available = _avail_cache.setdefault(step_dir, {})
        if len(available) < 4 and (full_rescan or number >= recent_cutoff):
            for kind in ("train", "eval"):
                for subset in ("all", "effective"):
                    key = f"{kind}/{subset}"
                    if key in available:
                        continue
                    path = step_dir / kind / subset / "traces.jsonl"
                    if path.is_file() and path.stat().st_size > 0:
                        available[key] = True
        if available:
            steps.append({"step": number, "available": available})
    root = eval_root_traces(run_dir)
    if not steps and root is not None:
        steps.append({"step": EVAL_ROOT_STEP[0], "available": {f"{EVAL_ROOT_STEP[1]}/{EVAL_ROOT_STEP[2]}": True}})
    return steps


def line_offsets(path: Path) -> list[int]:
    size = path.stat().st_size
    with _lock:
        cached_size, offsets = _lru_get(_offsets_cache, path) or (0, [])
        if cached_size > size:  # rewritten (e.g. resume cleanup): rebuild
            cached_size, offsets = 0, []
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
    with _lock:
        cached_size, current = _lru_get(_offsets_cache, path) or (0, offsets)
        if cached_size > size:
            current = []
        for offset in found:
            if not current or offset > current[-1]:
                current.append(offset)
        _lru_put(_offsets_cache, path, (max(size, cached_size), current))
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
    return {
        "rewards": {name: sum(v) / len(v) for name, v in reward_parts.items()},
        "metrics": {name: sum(v) / len(v) for name, v in metric_parts.items()},
        "timing": timing,
        "cost": sum(costs) if costs else None,
        "line": line,
        "id": rec.get("id"),
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
    }


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
    write_sidecar(path, offsets, summaries)
    return summaries


# ------------------------------------------------------- summary sidecars
# Parsing a step's traces file for the table can mean reading gigabytes; the
# result is persisted outside the run dir (the dashboard never writes there),
# so a revisit — or a dashboard restart — skips the parse entirely.
SIDECAR_DIR = STATE_DIR
SIDECAR_WRITE_INTERVAL_S = 20.0
_sidecar_written: dict[Path, tuple[float, int]] = {}  # path -> (last write time, count)


def sidecar_path(path: Path) -> Path:
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]
    return SIDECAR_DIR / f"{digest}.json"


def load_sidecar(path: Path) -> list[dict] | None:
    data = read_json(sidecar_path(path))
    try:
        if data.get("path") != str(path.resolve()) or path.stat().st_size < data["size"]:
            return None  # different file, or rewritten smaller (relaunch)
        return data["summaries"]
    except (KeyError, OSError):
        return None


def write_sidecar(path: Path, offsets: list[int], summaries: list[dict]) -> None:
    now = time.monotonic()
    last_time, last_count = _sidecar_written.get(path, (0.0, -1))
    if len(summaries) == last_count or now - last_time < SIDECAR_WRITE_INTERVAL_S:
        return
    _sidecar_written[path] = (now, len(summaries))
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    consumed = offsets[len(summaries) - 1] if summaries else 0
    payload = {"path": str(path.resolve()), "size": consumed, "summaries": summaries}
    target = sidecar_path(path)
    tmp = target.with_suffix(".tmp")
    tmp.write_bytes(orjson.dumps(payload))
    tmp.replace(target)


def traces_path(run: str, step: int, kind: str, subset: str) -> Path:
    if kind not in ("train", "eval") or subset not in ("all", "effective"):
        raise HTTPException(400, "kind must be train|eval, subset all|effective")
    run_dir = get_run_dir(run)
    path = run_dir / "rollouts" / f"step_{step}" / kind / subset / "traces.jsonl"
    if path.is_file():
        return path
    if (step, kind, subset) == EVAL_ROOT_STEP:
        root = eval_root_traces(run_dir)
        if root is not None:
            return root
    raise HTTPException(404, "no traces for this step/kind/subset")


@app.get("/api/runs/{run}/rollouts")
def list_rollouts(run: str) -> dict:
    return {"steps": rollout_steps(get_run_dir(run))}


@app.get("/api/runs/{run}/rollouts/{step}/{kind}/{subset}")
def list_episodes(
    run: str,
    step: int,
    kind: str,
    subset: str,
    limit: int = Query(default=5000, le=5000),
    env: str | None = None,
    errors_only: bool = False,
    sort: str = "line",
    order: str = "asc",
    etag: str | None = None,
) -> dict:
    path = traces_path(run, step, kind, subset)
    current_etag, unchanged = size_etag(path, etag)
    if unchanged:
        return unchanged
    summaries = episode_summaries(path)
    envs = sorted({s["env"] for s in summaries if s.get("env")})
    if env:
        summaries = [s for s in summaries if s.get("env") == env]
    if errors_only:
        summaries = [s for s in summaries if s.get("num_errors") or not s.get("ok")]
    if sort in ("reward", "advantage", "output_tokens", "turns", "group"):
        summaries = sorted(summaries, key=lambda s: (s.get(sort) is None, s.get(sort) or 0), reverse=(order == "desc"))
    return {"total": len(summaries), "etag": current_etag, "envs": envs, "episodes": summaries[:limit]}


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


@app.get("/api/runs/{run}/rollouts/{step}/{kind}/{subset}/series")
def episode_series(run: str, step: int, kind: str, subset: str, etag: str | None = None, after: int = 0) -> dict:
    """Per-episode series over a traces file (x = episode order): reward, shape, and the
    nested rewards/metrics/timing keys — the metrics view for eval runs. `after` returns
    only episodes past that index, so a growing file ships increments, not the world."""
    path = traces_path(run, step, kind, subset)
    current_etag, unchanged = size_etag(path, etag)
    if unchanged:
        return unchanged
    summaries = episode_summaries(path)
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


@app.get("/api/runs/{run}/rollouts/{step}/{kind}/{subset}/{line}")
def get_episode(
    run: str,
    step: int,
    kind: str,
    subset: str,
    line: int,
    tokens: bool = False,
    rendered: bool = False,
) -> dict:
    path = traces_path(run, step, kind, subset)
    offsets = line_offsets(path)
    if not 0 <= line < len(offsets):
        raise HTTPException(404, "episode line out of range")
    with path.open("rb") as f:
        f.seek(offsets[line])
        rec = orjson.loads(f.readline())
    if not tokens and not rendered:
        return rec
    fallback_model = model_name(main_config(get_run_dir(run))[1])
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
    global output_dirs, isolated
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
