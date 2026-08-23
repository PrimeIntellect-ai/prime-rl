#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi>=0.115",
#   "uvicorn>=0.30",
#   "orjson>=3.10",
#   "tokenizers>=0.20",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Local dashboard for prime-rl runs: logs, metrics, and rollout traces.

Reads everything from a run's output directory (metrics.jsonl, logs/attempt_N,
rollouts/step_N) — no wandb or network required. Usage:

    ./dashboard/dashboard.py [output_dir] [--port 7788] [--host 127.0.0.1]

View from another machine via an SSH tunnel (the startup banner prints the command).
"""

import argparse
import json
import sys
import threading
from pathlib import Path

import orjson
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"
MASTER_LOGS = {"trainer.log", "orchestrator.log", "inference.log", "evals.log"}

app = FastAPI()
output_dir = Path("outputs")

_lock = threading.Lock()
# Append-only file caches keyed by absolute path: line-start offsets and per-episode summaries.
_offsets_cache: dict[Path, tuple[int, list[int]]] = {}
_summaries_cache: dict[Path, tuple[int, list[dict]]] = {}
_tokenizer_cache: dict[str, object] = {}
_piece_cache: dict[tuple[str, int], str] = {}


def get_run_dir(run: str) -> Path:
    if "/" in run or run.startswith("."):
        raise HTTPException(400, "invalid run name")
    run_dir = output_dir / run
    if not run_dir.is_dir():
        raise HTTPException(404, f"run {run} not found")
    return run_dir


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def run_meta(run_dir: Path) -> dict:
    configs = run_dir / "configs"
    if (configs / "sft.json").exists():
        run_type, main_config = "sft", read_json(configs / "sft.json")
    elif (configs / "orchestrator.json").exists() or (configs / "trainer.json").exists():
        run_type = "rl"
        main_config = read_json(configs / "orchestrator.json") or read_json(configs / "trainer.json")
    else:
        run_type, main_config = "other", {}
    envs = lambda split: sorted(p.stem for p in (configs / "envs" / split).glob("*.json"))  # noqa: E731
    attempts = sorted(
        int(p.name.removeprefix("attempt_"))
        for p in (run_dir / "logs").glob("attempt_*")
        if p.name.removeprefix("attempt_").isdigit()
    )
    steps = [
        int(p.name.removeprefix("step_"))
        for p in (run_dir / "rollouts").glob("step_*")
        if p.name.removeprefix("step_").isdigit()
    ]
    metrics_path = run_dir / "metrics.jsonl"
    started = updated = None
    if metrics_path.is_file():
        updated = metrics_path.stat().st_mtime
        with metrics_path.open("rb") as f:
            try:
                started = orjson.loads(f.readline()).get("time")
            except orjson.JSONDecodeError:
                pass
    return {
        "name": run_dir.name,
        "type": run_type,
        "model": (main_config.get("model") or {}).get("name"),
        "max_steps": main_config.get("max_steps"),
        "train_envs": envs("train"),
        "eval_envs": envs("eval"),
        "attempts": attempts,
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
    for run_dir in sorted(output_dir.iterdir()) if output_dir.is_dir() else []:
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        if not any((run_dir / marker).exists() for marker in ("configs", "logs", "metrics.jsonl", "rollouts")):
            continue
        runs.append(run_meta(run_dir))
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return {"output_dir": str(output_dir.resolve()), "runs": runs}


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
    meta = run_meta(run_dir)
    attempts = meta["attempts"]
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
def read_log(
    run: str,
    file: str,
    start: int | None = None,
    end: int | None = None,
    tail: int | None = None,
    max_bytes: int = Query(default=2_000_000, le=8_000_000),
) -> dict:
    run_dir = get_run_dir(run)
    path = (run_dir / file).resolve()
    if not path.is_relative_to(run_dir.resolve()) or not path.is_file():
        raise HTTPException(404, "log file not found")
    size = path.stat().st_size
    if tail is not None:
        start = max(0, size - tail)
    start = min(start or 0, size)
    with path.open("rb") as f:
        f.seek(start)
        data = f.read(min(max_bytes, (end if end is not None else size) - start))
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
    return {"text": data.decode("utf-8", errors="replace"), "start": start, "end": chunk_end, "size": size}


# ------------------------------------------------------------------------- configs

CONFIG_ORDER = ["rl.json", "sft.json", "orchestrator.json", "trainer.json", "inference.json", "evals.json"]


@app.get("/api/runs/{run}/configs")
def list_configs(run: str) -> dict:
    configs_dir = get_run_dir(run) / "configs"
    files = [str(p.relative_to(configs_dir)) for p in configs_dir.rglob("*.json")] if configs_dir.is_dir() else []
    rank = {name: i for i, name in enumerate(CONFIG_ORDER)}
    files.sort(key=lambda f: (rank.get(f, len(rank)), f))
    return {"files": files}


@app.get("/api/runs/{run}/config")
def read_config(run: str, file: str) -> dict:
    configs_dir = (get_run_dir(run) / "configs").resolve()
    path = (configs_dir / file).resolve()
    if not path.is_relative_to(configs_dir) or path.suffix != ".json" or not path.is_file():
        raise HTTPException(404, "config file not found")
    return {"file": file, "content": path.read_text()}


# ------------------------------------------------------------------------- metrics


@app.get("/api/runs/{run}/metrics")
def read_metrics(run: str, offset: int = 0) -> dict:
    path = get_run_dir(run) / "metrics.jsonl"
    if not path.is_file():
        return {"rows": [], "offset": 0}
    size = path.stat().st_size
    if offset > size:  # file was truncated/replaced
        offset = 0
    rows = []
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
    consumed = data.rfind(b"\n") + 1  # leave a partially-written last line for the next poll
    for line in data[:consumed].splitlines():
        try:
            rows.append(orjson.loads(line))
        except orjson.JSONDecodeError:
            continue
    return {"rows": rows, "offset": offset + consumed}


# ------------------------------------------------------------------------ rollouts


def rollout_steps(run_dir: Path) -> list[dict]:
    steps = []
    for step_dir in (run_dir / "rollouts").glob("step_*"):
        if not step_dir.name.removeprefix("step_").isdigit():
            continue
        counts = {}
        for kind in ("train", "eval"):
            for subset in ("all", "effective"):
                path = step_dir / kind / subset / "traces.jsonl"
                if path.is_file():
                    counts[f"{kind}/{subset}"] = len(line_offsets(path))
        if counts:
            steps.append({"step": int(step_dir.name.removeprefix("step_")), "counts": counts})
    return sorted(steps, key=lambda s: s["step"])


def line_offsets(path: Path) -> list[int]:
    size = path.stat().st_size
    with _lock:
        cached_size, offsets = _offsets_cache.get(path, (0, []))
        if cached_size > size:  # rewritten (e.g. resume cleanup): rebuild
            cached_size, offsets = 0, []
        if cached_size == size:
            return offsets
    scan_from = offsets[-1] if offsets else 0
    new_offsets = offsets[: len(offsets) - 1] if offsets else []
    with path.open("rb") as f:
        f.seek(scan_from)
        pos = scan_from
        for line in f:
            if line.strip():
                new_offsets.append(pos)
            pos += len(line)
    with _lock:
        _offsets_cache[path] = (size, new_offsets)
    return new_offsets


def summarize_episode(line: int, rec: dict) -> dict:
    rewards, advantages = [], []
    input_tokens = output_tokens = turns = branches = 0
    stop_condition = completed = None
    for trace in rec.get("traces") or []:
        nodes = trace.get("nodes") or []
        parents = {node.get("parent") for node in nodes if "parent" in node}
        branches += max(0, len(nodes) - len(parents)) if nodes else 0
        total = sum(
            (r.get("score") or 0) * (r.get("weight") if r.get("weight") is not None else 1)
            for r in (trace.get("rewards") or {}).values()
            if isinstance(r, dict)
        )
        rewards.append(total)
        advantage = (trace.get("info") or {}).get("advantage")
        if advantage is not None:
            advantages.append(advantage)
        for node in trace.get("nodes") or []:
            n_tokens = len(node.get("token_ids") or [])
            if node.get("sampled"):
                output_tokens += n_tokens
            else:
                input_tokens += n_tokens
            if (node.get("message") or {}).get("role") == "assistant":
                turns += 1
        stop_condition = trace.get("stop_condition", stop_condition)
        completed = trace.get("is_completed", completed)
        if input_tokens == 0 and output_tokens == 0:  # some eval traces carry no token arrays
            for call in trace.get("calls") or []:
                usage = call.get("usage") or {}
                input_tokens += usage.get("prompt_tokens") or 0
                output_tokens += usage.get("completion_tokens") or 0
    run = rec.get("run") or {}
    dispatch_step = ((run.get("work") or {}).get("step")) or ((run.get("metadata") or {}).get("step"))
    return {
        "line": line,
        "id": rec.get("id"),
        "env": (rec.get("env") or {}).get("id") or (rec.get("env") or {}).get("name"),
        "group": (rec.get("group") or {}).get("id"),
        "ok": rec.get("ok"),
        "num_errors": len(rec.get("errors") or []),
        "num_traces": len(rec.get("traces") or []),
        "reward": sum(rewards) / len(rewards) if rewards else None,
        "advantage": sum(advantages) / len(advantages) if advantages else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "turns": turns,
        "branches": branches,
        "stop_condition": stop_condition,
        "is_completed": completed,
        "dispatch_step": dispatch_step,
    }


def episode_summaries(path: Path) -> list[dict]:
    offsets = line_offsets(path)
    with _lock:
        cached_count, summaries = _summaries_cache.get(path, (0, []))
        if cached_count > len(offsets):
            cached_count, summaries = 0, []
        summaries = list(summaries)
    if cached_count < len(offsets):
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
    path = get_run_dir(run) / "rollouts" / f"step_{step}" / kind / subset / "traces.jsonl"
    if not path.is_file():
        raise HTTPException(404, "no traces for this step/kind/subset")
    return path


@app.get("/api/runs/{run}/rollouts")
def list_rollouts(run: str) -> dict:
    return {"steps": rollout_steps(get_run_dir(run))}


@app.get("/api/runs/{run}/rollouts/{step}/{kind}/{subset}")
def list_episodes(
    run: str,
    step: int,
    kind: str,
    subset: str,
    page: int = 0,
    limit: int = Query(default=50, le=5000),
    env: str | None = None,
    errors_only: bool = False,
    sort: str = "line",
    order: str = "asc",
) -> dict:
    summaries = episode_summaries(traces_path(run, step, kind, subset))
    envs = sorted({s["env"] for s in summaries if s.get("env")})
    if env:
        summaries = [s for s in summaries if s.get("env") == env]
    if errors_only:
        summaries = [s for s in summaries if s.get("num_errors") or not s.get("ok")]
    if sort in ("reward", "advantage", "output_tokens", "turns", "group"):
        summaries = sorted(summaries, key=lambda s: (s.get(sort) is None, s.get(sort) or 0), reverse=(order == "desc"))
    total = len(summaries)
    return {"total": total, "envs": envs, "episodes": summaries[page * limit : (page + 1) * limit]}


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


@app.get("/api/runs/{run}/rollouts/{step}/{kind}/{subset}/{line}")
def get_episode(run: str, step: int, kind: str, subset: str, line: int, tokens: bool = False) -> dict:
    path = traces_path(run, step, kind, subset)
    offsets = line_offsets(path)
    if not 0 <= line < len(offsets):
        raise HTTPException(404, "episode line out of range")
    with path.open("rb") as f:
        f.seek(offsets[line])
        rec = orjson.loads(f.readline())
    if tokens:
        fallback_model = run_meta(get_run_dir(run)).get("model")
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


def main() -> None:
    global output_dir
    parser = argparse.ArgumentParser(description="prime-rl run dashboard")
    parser.add_argument("output_dir", nargs="?", default="outputs", type=Path)
    parser.add_argument("--port", type=int, default=7788)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir.is_dir():
        raise SystemExit(f"output dir {output_dir} does not exist")
    url = f"http://localhost:{args.port}"
    sep = "·"
    if sys.stdout.isatty():
        url = f"\033[4;38;2;182;255;60m{url}\033[0m"  # accent green, underlined
        sep = "\033[2m·\033[0m"
    print(f"\n  prime-rl dashboard {sep} {output_dir.resolve()}\n  {url}\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
