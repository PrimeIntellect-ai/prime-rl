#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi>=0.115",
#   "uvicorn>=0.30",
#   "orjson>=3.10",
#   "tokenizers>=0.20",
#   "huggingface_hub>=0.25",
#   "rich>=14.0",
# ]
# ///
"""Local dashboard for prime-rl runs: logs, metrics, and rollout traces.

This package is fully AI-generated and maintained by agents - it is not meant to be read or edited by humans. Change it by asking an agent, and verify through the browser smoke tests.

Reads everything from run output directories (metrics.jsonl, logs/attempt_N,
rollouts/step_N) — no wandb or network required. Usage:

    uv run dashboard [output_dir ...] [--port 7788] [--host 127.0.0.1]

Multiple output directories can be tracked at once.
"""

import argparse
import sys
import threading
from pathlib import Path

import orjson

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
except ModuleNotFoundError as error:  # the dashboard ships as an extra
    raise SystemExit(
        "the dashboard needs the 'dashboard' extra - install with `uv sync --extra dashboard` "
        "(or run the standalone script: `uv run --script src/prime_rl/dashboard/server.py`)"
    ) from error

STATIC_DIR = Path(__file__).parent / "static"
MASTER_LOGS = {"trainer.log", "orchestrator.log", "inference.log", "evals.log"}
MAX_LOG_CHUNK = 2_000_000

app = FastAPI()
output_dirs: list[Path] = [Path("outputs")]

# run id -> run dir, rebuilt on every /api/runs poll; ids are the run name,
# qualified with the output dir's basename when two dirs hold the same name
_run_registry: dict[str, Path] = {}

_lock = threading.Lock()
# Append-only file caches keyed by absolute path: line-start offsets and per-episode summaries.
_offsets_cache: dict[Path, tuple[int, list[int]]] = {}
_summaries_cache: dict[Path, tuple[int, list[dict]]] = {}
_tokenizer_cache: dict[str, object] = {}
_piece_cache: dict[tuple[str, int], str] = {}
_json_cache: dict[Path, tuple[tuple[int, float], dict]] = {}
_steps_cache: dict[Path, tuple[float, list[int]]] = {}


def is_run_dir(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith("."):
        return False
    return any((path / marker).exists() for marker in ("configs", "logs", "metrics.jsonl", "rollouts"))


def scan_runs() -> dict[str, Path]:
    global _run_registry
    by_name: dict[str, list[Path]] = {}
    for base in output_dirs:
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
                registry[f"{run_dir.parent.name}/{name}"] = run_dir
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


def rollout_steps(run_dir: Path) -> list[dict]:
    """Presence only — never reads trace files, so it stays cheap over thousands of steps."""
    steps = []
    for number, step_dir in numbered_dirs(run_dir / "rollouts", "step_"):
        available = {}
        for kind in ("train", "eval"):
            for subset in ("all", "effective"):
                path = step_dir / kind / subset / "traces.jsonl"
                if path.is_file() and path.stat().st_size > 0:
                    available[f"{kind}/{subset}"] = True
        if available:
            steps.append({"step": number, "available": available})
    root_traces = run_dir / "traces.jsonl"  # `uv run eval` writes one stepless file
    if not steps and root_traces.is_file() and root_traces.stat().st_size > 0:
        steps.append({"step": 0, "available": {"eval/all": True}})
    return steps


def line_offsets(path: Path) -> list[int]:
    size = path.stat().st_size
    with _lock:
        cached_size, offsets = _offsets_cache.get(path, (0, []))
        if cached_size > size:  # rewritten (e.g. resume cleanup): rebuild
            cached_size, offsets = 0, []
        if cached_size == size:
            return offsets
    # re-scan from the last recorded line (it may have been partial) and grow the
    # cached list strictly by appending — a concurrent reader's indices stay valid
    scan_from = offsets[-1] if offsets else 0
    with path.open("rb") as f:
        f.seek(scan_from)
        pos = scan_from
        for line in f:
            if line.strip() and (not offsets or pos > offsets[-1]):
                offsets.append(pos)
            pos += len(line)
    with _lock:
        _offsets_cache[path] = (size, offsets)
    return offsets


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
        cached_count, summaries = _summaries_cache.get(path, (0, []))
        if cached_count > len(offsets):
            cached_count, summaries = 0, []
    if cached_count == len(offsets):
        return summaries
    summaries = list(summaries)
    with path.open("rb") as f:
        f.seek(offsets[cached_count])
        for line_no in range(cached_count, len(offsets)):
            raw = f.readline()
            try:
                summaries.append(summarize_episode(line_no, orjson.loads(raw)))
            except orjson.JSONDecodeError:
                summaries.append({"line": line_no, "id": None, "error": "unparseable"})
    with _lock:
        _summaries_cache[path] = (len(offsets), summaries)
    return summaries


def traces_path(run: str, step: int, kind: str, subset: str) -> Path:
    if kind not in ("train", "eval") or subset not in ("all", "effective"):
        raise HTTPException(400, "kind must be train|eval, subset all|effective")
    run_dir = get_run_dir(run)
    path = run_dir / "rollouts" / f"step_{step}" / kind / subset / "traces.jsonl"
    if path.is_file():
        return path
    root = run_dir / "traces.jsonl"
    if (step, kind, subset) == (0, "eval", "all") and root.is_file():
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
def get_episode(run: str, step: int, kind: str, subset: str, line: int, tokens: bool = False) -> dict:
    path = traces_path(run, step, kind, subset)
    offsets = line_offsets(path)
    if not 0 <= line < len(offsets):
        raise HTTPException(404, "episode line out of range")
    with path.open("rb") as f:
        f.seek(offsets[line])
        rec = orjson.loads(f.readline())
    if not tokens:
        return rec
    fallback_model = model_name(main_config(get_run_dir(run))[1])
    for trace in rec.get("traces") or []:
        client = ((trace.get("agent") or {}).get("config") or {}).get("client") or {}
        model = client.get("renderer_model_name") or fallback_model
        if not model:
            continue
        for node in trace.get("nodes") or []:
            if node.get("token_ids"):
                node["token_strs"] = decode_pieces(model, node["token_ids"])
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
    for base in output_dirs:
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


def main() -> None:
    global output_dirs
    parser = argparse.ArgumentParser(description="prime-rl run dashboard")
    parser.add_argument("output_dirs", nargs="*", default=[Path("outputs")], type=Path, metavar="output_dir")
    parser.add_argument("--port", type=int, default=7788)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    output_dirs = [d for d in args.output_dirs if d.is_dir()]
    for missing in set(args.output_dirs) - set(output_dirs):
        print(f"warning: output dir {missing} does not exist", file=sys.stderr)
    if not output_dirs:
        raise SystemExit("no existing output dir given")
    port = free_port(args.host, args.port)
    if port != args.port:
        print(f"port {args.port} is taken - serving on {port}", file=sys.stderr)
    args.port = port
    url = f"http://localhost:{args.port}"
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


if __name__ == "__main__":
    main()
