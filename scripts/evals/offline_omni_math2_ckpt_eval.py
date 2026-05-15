from __future__ import annotations

import argparse
import asyncio
import copy
import json
import multiprocessing
import os
import queue
import re
import time
import traceback
from collections.abc import Iterable
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import httpx

from prime_rl.baselines.benchmark import artifact_complete, expected_rollouts
from prime_rl.baselines.config import BaselineConfig, load_config
from prime_rl.baselines.metrics import summarize_records
from prime_rl.baselines.provision import (
    Endpoint,
    InferenceProvisioner,
    _find_vllm_router,
    _router_health_url,
    wait_for_endpoint,
)
from prime_rl.baselines.runner import _eval_examples, _example_id, prepare_import_paths, run_baseline
from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.client import setup_admin_clients, update_weights
from prime_rl.utils.logger import ProgressTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/baselines/omni_math2_hybridmath_local.toml"
DEFAULT_MODEL = "allenai/Olmo-3-7B-Instruct-DPO"
DEFAULT_JUDGE_CACHE = "outputs/judge-cache/omni_math2_rlvr_gpt54mini.sqlite3"
DEFAULT_KS = (1, 2, 3, 4, 5, 6, 8)
SHARD_STATUS_POLL_INTERVAL_S = 5.0
SHARD_JOIN_TIMEOUT_S = 20.0
STEP_RE = re.compile(r"step_(\d+)$")


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected ARM=RUN_ROOT, e.g. default=outputs/.../run")
    arm, run_root = value.split("=", 1)
    arm = arm.strip()
    if not arm:
        raise argparse.ArgumentTypeError("ARM must be non-empty")
    return arm, Path(run_root)


def _step_from_path(path: Path) -> int | None:
    match = STEP_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _discover_weight_steps(
    weights_root: Path,
    *,
    steps: set[int] | None,
    step_interval: int | None,
    min_step: int | None,
    max_step: int | None,
) -> list[tuple[int, Path]]:
    if not weights_root.exists():
        raise FileNotFoundError(f"Missing weights root: {weights_root}")

    discovered: list[tuple[int, Path]] = []
    for path in weights_root.glob("step_*"):
        step = _step_from_path(path)
        if step is None or not path.is_dir():
            continue
        if not (path / "STABLE").exists():
            continue
        if not (path / "model.safetensors.index.json").exists() and not list(path.glob("*.safetensors")):
            continue
        if steps is not None and step not in steps:
            continue
        if step_interval is not None and step % step_interval != 0:
            continue
        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        discovered.append((step, path))

    return sorted(discovered)


def _range_filters_for_explicit_steps(
    step_filter: set[int] | None,
    *,
    step_interval: int | None,
    min_step: int | None,
    max_step: int | None,
) -> tuple[int | None, int | None, int | None]:
    if step_filter is not None:
        return None, None, None
    return step_interval, min_step, max_step


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w") as f:
        for row in rows:
            json.dump(row, f)
            f.write("\n")
    tmp_path.replace(path)


def _merge_completed_shards(output_dir: Path, expected_rollout_count: int) -> bool:
    shard_root = output_dir / "shards"
    if not shard_root.exists():
        return False

    shard_dirs = sorted(path for path in shard_root.glob("shard_*") if path.is_dir())
    if not shard_dirs:
        return False

    records: list[dict[str, Any]] = []
    total_rollouts = 0
    for shard_dir in shard_dirs:
        summary_path = shard_dir / "summary.json"
        records_path = shard_dir / "records.jsonl"
        raw_path = shard_dir / "raw_rollouts.jsonl"
        if not summary_path.exists() or not records_path.exists() or not raw_path.exists():
            return False
        try:
            summary = json.loads(summary_path.read_text())
            shard_rollouts = int(summary["num_rollouts"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if not artifact_complete(shard_dir, shard_rollouts):
            return False
        total_rollouts += shard_rollouts
        records.extend(_read_jsonl(records_path))

    if total_rollouts != expected_rollout_count:
        return False

    records.sort(key=lambda r: (str(r["example_id"]), int(r["trial_index"])))
    _write_jsonl(output_dir / "records.jsonl", records)

    raw_output = output_dir / "raw_rollouts.jsonl"
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    tmp_raw = raw_output.with_name(f".{raw_output.name}.{os.getpid()}.tmp")
    with tmp_raw.open("w") as out:
        for shard_dir in shard_dirs:
            with (shard_dir / "raw_rollouts.jsonl").open() as source:
                for line in source:
                    if line.strip():
                        out.write(line)
    tmp_raw.replace(raw_output)
    return True


def _completed_shard_rollout_count(output_dir: Path) -> int | None:
    shard_root = output_dir / "shards"
    if not shard_root.exists():
        return None

    total = 0
    shard_dirs = sorted(path for path in shard_root.glob("shard_*") if path.is_dir())
    if not shard_dirs:
        return None

    for shard_dir in shard_dirs:
        summary_path = shard_dir / "summary.json"
        if not summary_path.exists():
            return None
        try:
            summary = json.loads(summary_path.read_text())
            shard_rollouts = int(summary["num_rollouts"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not artifact_complete(shard_dir, shard_rollouts):
            return None
        total += shard_rollouts
    return total


def _annotate_and_summarize(
    output_dir: Path,
    *,
    arm: str,
    step: int,
    weight_dir: Path | None,
    ks: tuple[int, ...],
) -> dict[str, Any]:
    records_path = output_dir / "records.jsonl"
    records = _read_jsonl(records_path)
    for row in records:
        row["arm"] = arm
        row["step"] = step
        row["sample_idx"] = int(row.get("trial_index", 0))
        if weight_dir is not None:
            row["checkpoint_weight_dir"] = str(weight_dir)
        row["cache_key"] = f"{arm}:step_{step}:{row['example_id']}:{row['sample_idx']}"
    records.sort(key=lambda r: (str(r["arm"]), int(r["step"]), str(r["example_id"]), int(r["sample_idx"])))
    _write_jsonl(records_path, records)

    previous_summary: dict[str, Any] = {}
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        previous_summary = json.loads(summary_path.read_text())

    summary = summarize_records(records, ks)
    for key in ("run_id", "elapsed_s", "env_id", "protocol", "dataset", "model", "config", "judge_cache"):
        if key in previous_summary:
            summary[key] = previous_summary[key]
    summary.update(
        {
            "arm": arm,
            "step": step,
            "checkpoint_weight_dir": str(weight_dir) if weight_dir is not None else None,
            "cache_key_fields": ["arm", "step", "example_id", "sample_idx"],
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def _normal_generation_url(url: str) -> str:
    stripped = url.rstrip("/")
    return stripped if stripped.endswith("/v1") else f"{stripped}/v1"


def _configure_eval(
    base: BaselineConfig,
    *,
    endpoint: Endpoint,
    output_dir: Path,
    request_model: str,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrency: int,
    score_max_concurrency: int,
    max_retries: int,
    resume_partial: bool,
    ks: tuple[int, ...],
    judge_cache_path: str,
    fail_on_error: bool,
) -> BaselineConfig:
    config = copy.deepcopy(base)
    config.model = request_model
    config.output_dir = output_dir
    config.run_id = output_dir.name
    config.protocol = "maj_at_n"
    config.dataset = "omni_math2_stratified_600"
    config.seed = 42
    config.num_examples = num_examples
    config.rollouts_per_example = rollouts_per_example
    config.max_concurrency = max_concurrency
    config.score_max_concurrency = score_max_concurrency
    config.max_retries = max_retries
    config.resume_partial = resume_partial
    config.fail_on_error = fail_on_error
    config.ks = ks
    config.base_url = endpoint.base_url
    config.api_key_var = endpoint.api_key_var
    config.api_profile = "vllm_permissive"
    config.progress = "none"
    config.launch.mode = "external"
    config.launch.wait_timeout_s = 900.0
    config.sampling_args = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_completion_tokens": 15360,
    }
    config.env_args = dict(config.env_args)
    config.env_args.update(
        {
            "data_files": "benchmarks/datasets/omni_math2/omni_math2_stratified_600_seed42.jsonl",
            "prompts_ref": "configs/baselines/omni_math2_prompt_pack.yaml",
            "use_judge_fallback": True,
            "judge_model": "gpt-5.4-mini",
            "judge_persistent_cache_path": judge_cache_path,
            "math_verify_max_workers": 50,
            "math_verify_timeout_seconds": 10,
        }
    )
    return config


def _run_baseline_process(
    index: int,
    config: BaselineConfig,
    log_path: Path,
    status_queue: multiprocessing.Queue,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    config.progress = "none"

    status: dict[str, Any]
    exit_code = 1
    with log_path.open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:

            def report_progress(phase: str, n: int) -> None:
                status_queue.put({"type": "progress", "index": index, "phase": phase, "n": n})

            summary = run_baseline(config, progress_callback=report_progress)
            status = {
                "type": "done",
                "index": index,
                "ok": True,
                "num_rollouts": summary.get("num_rollouts"),
                "output_dir": str(config.output_dir),
                "log_path": str(log_path),
            }
            exit_code = 0
        except BaseException:
            status = {
                "type": "done",
                "index": index,
                "ok": False,
                "traceback": traceback.format_exc(),
                "output_dir": str(config.output_dir),
                "log_path": str(log_path),
            }
            traceback.print_exc()
        finally:
            status_queue.put(status)
            status_queue.close()
            status_queue.join_thread()
            log.flush()

    os._exit(exit_code)


def _wait_for_shard_processes(
    *,
    processes: list[tuple[int, multiprocessing.Process]],
    status_queue: multiprocessing.Queue,
    shard_dirs: list[Path],
    expected_rollouts_by_index: dict[int, int],
) -> None:
    process_by_index = dict(processes)
    pending = set(process_by_index)
    errors: list[str] = []
    total_rollouts = sum(expected_rollouts_by_index.values())
    generation_progress = ProgressTracker(total=total_rollouts, desc="Offline generations", position=0)
    scoring_progress = ProgressTracker(total=total_rollouts, desc="Offline scoring", position=1)

    try:
        while pending:
            try:
                status = status_queue.get(timeout=SHARD_STATUS_POLL_INTERVAL_S)
            except queue.Empty:
                for index in list(pending):
                    process = process_by_index[index]
                    if process.exitcode is not None:
                        process.join()
                        errors.append(
                            f"shard {index:02d} exited with status {process.exitcode} before reporting status"
                        )
                        pending.remove(index)
                continue

            if status.get("type") == "progress":
                phase = status.get("phase")
                try:
                    n = int(status.get("n", 0))
                except (TypeError, ValueError):
                    continue
                if phase == "generated":
                    generation_progress.update(n)
                elif phase == "scored":
                    scoring_progress.update(n)
                elif phase == "rollout":
                    generation_progress.update(n)
                    scoring_progress.update(n)
                continue

            try:
                index = int(status["index"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"received malformed shard status: {status!r}")
                continue
            if index not in pending:
                continue

            process = process_by_index[index]
            process.join(timeout=SHARD_JOIN_TIMEOUT_S)
            if process.is_alive():
                print(f"shard {index:02d} reported completion but did not exit; terminating leftover worker process")
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=10)

            if not status.get("ok"):
                errors.append(f"shard {index:02d} failed; see {status.get('log_path')}\n{status.get('traceback')}")
            else:
                expected = expected_rollouts_by_index[index]
                if not artifact_complete(shard_dirs[index], expected):
                    errors.append(
                        f"shard {index:02d} reported success but artifacts are incomplete; see {status.get('log_path')}"
                    )
            pending.remove(index)
    finally:
        generation_progress.close()
        scoring_progress.close()
        status_queue.close()
        for _, process in processes:
            process.join(timeout=1)

    if errors:
        raise RuntimeError("\n".join(errors))


def _eval_record_ids(config: BaselineConfig) -> list[str]:
    prepare_import_paths(config)
    import verifiers as vf

    env = vf.load_environment(config.env_id, **config.env_args)
    return [_example_id(example) for example in _eval_examples(config, env)]


def _configure_server(
    base: BaselineConfig,
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> BaselineConfig:
    config = copy.deepcopy(base)
    config.model = args.base_model
    config.output_dir = output_dir / "_inference_server"
    config.run_id = "offline-omni-math2-server"
    config.launch.gpu_memory_utilization = args.launch_gpu_memory_utilization
    config.launch.max_model_len = args.launch_max_model_len
    config.launch.enforce_eager = False
    config.launch.vllm_extra = {
        **config.launch.vllm_extra,
        "language_model_only": True,
        "skip_mm_profiling": True,
        "mm_processor_cache_gb": 0.0,
        "max_num_seqs": args.launch_max_num_seqs,
        "max_num_batched_tokens": args.launch_max_num_batched_tokens,
        "performance_mode": "throughput",
        "generation_config": "vllm",
    }
    if args.launch_mode is not None:
        config.launch.mode = args.launch_mode
    if args.launch_port is not None:
        config.launch.port = args.launch_port
    if args.launch_host is not None:
        config.launch.host = args.launch_host
    if args.launch_gpus is not None:
        config.launch.gpus = args.launch_gpus
    if args.launch_tp is not None:
        config.launch.tp = args.launch_tp
    if args.launch_dp is not None:
        config.launch.dp = args.launch_dp
    if args.launch_api_server_count is not None:
        config.launch.api_server_count = args.launch_api_server_count
    if args.launch_nodes is not None:
        config.launch.nodes = args.launch_nodes
    if args.launch_gpus_per_node is not None:
        config.launch.gpus_per_node = args.launch_gpus_per_node
    if args.launch_srun_job_id is not None:
        config.launch.srun_job_id = args.launch_srun_job_id
    if args.launch_router_port is not None:
        config.launch.router_port = args.launch_router_port
    if args.launch_backend_port is not None:
        config.launch.backend_port = args.launch_backend_port
    if args.launch_data_parallel_size_local is not None:
        config.launch.data_parallel_size_local = args.launch_data_parallel_size_local
    if args.launch_srun_network is not None:
        config.launch.srun_network = args.launch_srun_network
    if args.launch_srun_cpus_per_task is not None:
        config.launch.srun_cpus_per_task = args.launch_srun_cpus_per_task
    config.launch.wait_timeout_s = args.launch_wait_timeout_s
    config.launch.server_start_retries = args.launch_server_start_retries
    return config


def _derive_admin_urls(config: BaselineConfig, endpoint: Endpoint) -> list[str]:
    if config.launch.mode != "srun_multinode":
        return [endpoint.base_url]

    from prime_rl.baselines.provision import _multinode_hostnames

    hostnames = _multinode_hostnames(config.launch.srun_job_id)
    backend_port = config.launch.backend_port or (config.launch.port + 100)
    return [f"http://{host}:{backend_port}" for host in hostnames[: config.launch.nodes]]


def _derive_generation_urls(
    config: BaselineConfig | None,
    endpoint: Endpoint,
    admin_urls: list[str],
) -> list[str]:
    disable_router = os.environ.get("PRIME_RL_DISABLE_VLLM_ROUTER") == "1"
    if (
        config is not None
        and config.launch.mode == "srun_multinode"
        and (disable_router or _find_vllm_router() is None)
    ):
        return [_normal_generation_url(url) for url in admin_urls]
    return [_normal_generation_url(endpoint.base_url)]


def _wait_for_all_endpoints(
    *,
    urls: Iterable[str],
    api_key_var: str,
    timeout_s: float,
    router_urls: Iterable[str] = (),
) -> None:
    router_set = {_normal_generation_url(url) for url in router_urls}
    headers = {"Authorization": f"Bearer {os.environ.get(api_key_var, 'EMPTY')}"}
    for url in sorted(set(urls)):
        print(f"waiting for inference endpoint {url}")
        if _normal_generation_url(url) not in router_set:
            wait_for_endpoint(url, api_key_var, timeout_s)
            continue

        deadline = time.time() + timeout_s
        last_error: Exception | None = None
        with httpx.Client(timeout=10.0, headers=headers) as client:
            while time.time() < deadline:
                try:
                    response = client.get(_router_health_url(url))
                    response.raise_for_status()
                    break
                except Exception as exc:
                    last_error = exc
                time.sleep(1.0)
            else:
                raise RuntimeError(f"Inference router {url} did not become ready. Last error: {last_error}")


async def _update_endpoint_weights(
    *,
    base_url: str,
    admin_urls: list[str],
    api_key_var: str,
    weight_dir: Path,
) -> None:
    client_config = ClientConfig(
        base_url=[base_url],
        admin_base_url=admin_urls,
        api_key_var=api_key_var,
        api_profile="vllm_permissive",
        skip_model_check=True,
    )
    admin_clients = setup_admin_clients(client_config)
    try:
        await update_weights(admin_clients, weight_dir)
    finally:
        await asyncio.gather(*(client.aclose() for client in admin_clients))


def _run_one_checkpoint(
    *,
    base_config: BaselineConfig,
    endpoint: Endpoint,
    admin_urls: list[str],
    generation_urls: list[str],
    arm: str,
    step: int,
    weight_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    ks: tuple[int, ...],
) -> dict[str, Any]:
    expected = expected_rollouts(args.num_examples, [], args.rollouts_per_example)
    if args.skip_existing and not args.force and artifact_complete(output_dir, expected):
        return _annotate_and_summarize(output_dir, arm=arm, step=step, weight_dir=weight_dir, ks=ks)
    if (
        args.skip_existing
        and not args.force
        and len(generation_urls) > 1
        and _merge_completed_shards(output_dir, expected)
    ):
        print(f"merged completed shards for {arm} step {step}: {output_dir}")
        return _annotate_and_summarize(output_dir, arm=arm, step=step, weight_dir=weight_dir, ks=ks)

    t0 = time.perf_counter()
    asyncio.run(
        _update_endpoint_weights(
            base_url=endpoint.base_url,
            admin_urls=admin_urls,
            api_key_var=endpoint.api_key_var,
            weight_dir=weight_dir,
        )
    )
    update_elapsed_s = time.perf_counter() - t0

    eval_config = _configure_eval(
        base_config,
        endpoint=endpoint,
        output_dir=output_dir,
        request_model=args.request_model or args.base_model,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        max_concurrency=args.max_concurrency,
        score_max_concurrency=args.score_max_concurrency,
        max_retries=args.max_retries,
        resume_partial=not args.force,
        ks=ks,
        judge_cache_path=args.judge_cache_path,
        fail_on_error=not args.allow_errors,
    )

    if len(generation_urls) == 1:
        eval_config.base_url = generation_urls[0]
        if generation_urls[0] == endpoint.base_url and len(admin_urls) > 1:
            eval_config.launch.external_health_check = "router_health"
        run_baseline(eval_config)
    else:
        record_ids = _eval_record_ids(eval_config)
        shards = [record_ids[index :: len(generation_urls)] for index in range(len(generation_urls))]
        shard_dirs = [output_dir / "shards" / f"shard_{index:02d}" for index in range(len(generation_urls))]

        shard_configs: list[tuple[int, BaselineConfig]] = []
        expected_rollouts_by_index: dict[int, int] = {}
        for index in range(len(generation_urls)):
            if not shards[index]:
                continue
            shard_config = copy.deepcopy(eval_config)
            shard_config.base_url = generation_urls[index]
            shard_config.record_ids = shards[index]
            shard_config.num_examples = len(shards[index])
            shard_config.output_dir = shard_dirs[index]
            shard_config.run_id = f"{output_dir.name}-shard-{index:02d}"
            shard_configs.append((index, shard_config))
            expected_rollouts_by_index[index] = len(shards[index]) * args.rollouts_per_example

        process_context = multiprocessing.get_context("spawn")
        status_queue = process_context.Queue()
        processes: list[tuple[int, multiprocessing.Process]] = []
        for index, shard_config in shard_configs:
            log_path = shard_dirs[index] / "worker.log"
            process = process_context.Process(
                target=_run_baseline_process,
                args=(index, shard_config, log_path, status_queue),
            )
            process.start()
            processes.append((index, process))
        try:
            _wait_for_shard_processes(
                processes=processes,
                status_queue=status_queue,
                shard_dirs=shard_dirs,
                expected_rollouts_by_index=expected_rollouts_by_index,
            )
        except BaseException:
            for _, process in processes:
                if process.is_alive():
                    process.terminate()
            for _, process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
            status_queue.close()
            raise

        records: list[dict[str, Any]] = []
        raw_rollouts: list[dict[str, Any]] = []
        for shard_dir in shard_dirs:
            records_path = shard_dir / "records.jsonl"
            if records_path.exists():
                records.extend(_read_jsonl(records_path))
            raw_path = shard_dir / "raw_rollouts.jsonl"
            if raw_path.exists():
                raw_rollouts.extend(_read_jsonl(raw_path))
        records.sort(key=lambda r: (str(r["example_id"]), int(r["trial_index"])))
        raw_rollouts.sort(key=lambda r: (str(r.get("example_id")), int(r.get("trial_index", 0))))
        _write_jsonl(output_dir / "records.jsonl", records)
        _write_jsonl(output_dir / "raw_rollouts.jsonl", raw_rollouts)

    summary = _annotate_and_summarize(output_dir, arm=arm, step=step, weight_dir=weight_dir, ks=ks)
    summary["weight_update_elapsed_s"] = update_elapsed_s
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _recompute_existing(output_root: Path, *, ks: tuple[int, ...]) -> None:
    for arm_dir in sorted(path for path in output_root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        arm = arm_dir.name
        for step_dir in sorted(arm_dir.glob("step_*")):
            step = _step_from_path(step_dir)
            if step is None:
                continue

            shard_rollouts = _completed_shard_rollout_count(step_dir)
            if shard_rollouts is not None and not artifact_complete(step_dir, shard_rollouts):
                if _merge_completed_shards(step_dir, shard_rollouts):
                    print(f"merged completed shards for {arm} step {step}: {step_dir}")

            if not (step_dir / "records.jsonl").exists():
                continue
            _annotate_and_summarize(step_dir, arm=arm, step=step, weight_dir=None, ks=ks)
            print(f"recomputed {arm} step {step}: {step_dir}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Omni-MATH-2 eval for PrimeRL HF weight checkpoints.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", action="append", type=_parse_run, default=[])
    parser.add_argument("--arm", help="Single-arm name. Use with --run-root.")
    parser.add_argument("--run-root", type=Path, help="Single-arm PrimeRL output dir.")
    parser.add_argument("--weights-root", type=Path, help="Override weights root for single-arm mode.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--request-model")
    parser.add_argument(
        "--base-url",
        action="append",
        default=[],
        help="Existing OpenAI-compatible endpoint. Repeat to shard generation across endpoints. If omitted, launch one.",
    )
    parser.add_argument("--admin-url", action="append", default=[])
    parser.add_argument("--api-key-var", default="VLLM_API_KEY")
    parser.add_argument("--steps", type=_parse_csv_ints)
    parser.add_argument("--step-interval", type=int, default=50)
    parser.add_argument("--min-step", type=int)
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--num-examples", type=int, default=600)
    parser.add_argument("--rollouts-per-example", type=int, default=8)
    parser.add_argument("--max-concurrency", type=int, default=256)
    parser.add_argument("--score-max-concurrency", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--ks", type=_parse_csv_ints, default=DEFAULT_KS)
    parser.add_argument("--judge-cache-path", default=DEFAULT_JUDGE_CACHE)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--recompute-only", action="store_true")
    parser.add_argument("--launch-mode", choices=["local", "srun", "srun_multinode"])
    parser.add_argument("--launch-host")
    parser.add_argument("--launch-port", type=int)
    parser.add_argument("--launch-backend-port", type=int)
    parser.add_argument("--launch-router-port", type=int)
    parser.add_argument("--launch-gpus")
    parser.add_argument("--launch-tp", type=int)
    parser.add_argument("--launch-dp", type=int)
    parser.add_argument("--launch-api-server-count", type=int)
    parser.add_argument("--launch-nodes", type=int)
    parser.add_argument("--launch-gpus-per-node", type=int)
    parser.add_argument("--launch-data-parallel-size-local", type=int)
    parser.add_argument("--launch-srun-job-id")
    parser.add_argument("--launch-srun-network")
    parser.add_argument("--launch-srun-cpus-per-task", type=int)
    parser.add_argument("--launch-wait-timeout-s", type=float, default=1800.0)
    parser.add_argument("--launch-server-start-retries", type=int, default=1)
    parser.add_argument("--launch-gpu-memory-utilization", type=float, default=0.93)
    parser.add_argument("--launch-max-model-len", type=int, default=16384)
    parser.add_argument("--launch-max-num-seqs", type=int, default=192)
    parser.add_argument("--launch-max-num-batched-tokens", type=int, default=65536)
    return parser


def main() -> None:
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

    parser = _build_parser()
    args = parser.parse_args()
    ks = tuple(args.ks)

    if args.recompute_only:
        _recompute_existing(args.output_dir, ks=ks)
        return

    runs = list(args.run)
    if args.run_root is not None or args.arm is not None:
        if args.run_root is None or args.arm is None:
            parser.error("--arm and --run-root must be provided together")
        runs.append((args.arm, args.run_root))
    if not runs:
        parser.error("Provide at least one --run ARM=RUN_ROOT or --arm/--run-root pair")
    if args.weights_root is not None and len(runs) != 1:
        parser.error("--weights-root is only valid in single-arm mode")

    base_config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step_filter = set(args.steps) if args.steps is not None else None

    if args.base_url:
        generation_urls = [_normal_generation_url(url) for url in args.base_url]
        endpoint_context = nullcontext(Endpoint(base_url=generation_urls[0], api_key_var=args.api_key_var))
        server_config = None
    else:
        server_config = _configure_server(base_config, args, output_dir=args.output_dir)
        endpoint_context = InferenceProvisioner(server_config)

    with endpoint_context as endpoint:
        admin_urls = args.admin_url
        if not admin_urls:
            admin_urls = (
                _derive_admin_urls(server_config, endpoint) if server_config is not None else list(generation_urls)
            )
        generation_urls = (
            generation_urls if args.base_url else _derive_generation_urls(server_config, endpoint, admin_urls)
        )
        if args.base_url and len(admin_urls) != len(generation_urls):
            parser.error(
                "external endpoint mode requires either no --admin-url or exactly one --admin-url per --base-url"
            )

        print(f"generation endpoint: {endpoint.base_url}")
        print(f"generation shards: {', '.join(generation_urls)}")
        print(f"admin endpoints: {', '.join(admin_urls)}")
        _wait_for_all_endpoints(
            urls=[*generation_urls, *admin_urls],
            api_key_var=endpoint.api_key_var,
            timeout_s=args.launch_wait_timeout_s,
            router_urls=[] if args.base_url else [endpoint.base_url],
        )

        for arm, run_root in runs:
            weights_root = args.weights_root or (run_root / "weights")
            step_interval, min_step, max_step = _range_filters_for_explicit_steps(
                step_filter,
                step_interval=args.step_interval,
                min_step=args.min_step,
                max_step=args.max_step,
            )
            steps = _discover_weight_steps(
                weights_root,
                steps=step_filter,
                step_interval=step_interval,
                min_step=min_step,
                max_step=max_step,
            )
            if not steps:
                raise RuntimeError(f"No matching stable weight checkpoints found for {arm} in {weights_root}")

            for step, weight_dir in steps:
                step_output = args.output_dir / arm / f"step_{step:06d}"
                print(f"evaluating {arm} step {step} from {weight_dir}")
                summary = _run_one_checkpoint(
                    base_config=base_config,
                    endpoint=endpoint,
                    admin_urls=admin_urls,
                    generation_urls=generation_urls,
                    arm=arm,
                    step=step,
                    weight_dir=weight_dir,
                    output_dir=step_output,
                    args=args,
                    ks=ks,
                )
                pass8 = (summary.get("pass") or {}).get("8", {}).get("pass_at_k")
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "step": step,
                            "num_rollouts": summary.get("num_rollouts"),
                            "single_shot_accuracy": summary.get("single_shot_accuracy"),
                            "pass_at_8": pass8,
                            "output_dir": str(step_output),
                        }
                    )
                )


if __name__ == "__main__":
    main()
