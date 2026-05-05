from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from prime_rl.baselines.benchmark import artifact_complete, expected_rollouts, slug
from prime_rl.baselines.config import BaselineConfig, load_config
from prime_rl.baselines.provision import InferenceProvisioner
from prime_rl.baselines.runner import run_baseline

REPO_ROOT = Path(__file__).resolve().parents[2]
MCQ_CONFIG = REPO_ROOT / "configs/baselines/gpqa_mcq_gemma4_local.toml"
OE_CONFIG = REPO_ROOT / "configs/baselines/gpqa_openended_gemma4_local.toml"
TASK_CONFIGS = (MCQ_CONFIG, OE_CONFIG)


def _task_output_dir(config: BaselineConfig, args: argparse.Namespace) -> Path:
    return args.output_root / f"{args.run_prefix}-{config.dataset}-k{args.rollouts_per_example}"


def _load_task_config(path: Path, args: argparse.Namespace, endpoint_url: str) -> BaselineConfig:
    config = load_config(path)
    config.model = args.model
    config.output_dir = _task_output_dir(config, args)
    config.num_examples = args.num_examples
    config.rollouts_per_example = args.rollouts_per_example
    config.max_concurrency = args.max_concurrency
    config.score_max_concurrency = args.score_max_concurrency
    config.max_retries = args.max_retries
    config.fail_on_error = False
    config.base_url = endpoint_url
    config.client_type = args.client_type
    config.api_profile = "vllm_permissive"
    config.launch.mode = "external"
    config.launch.wait_timeout_s = args.wait_timeout_s
    config.sampling_args.update(args.sampling_extra)
    return config


def _server_config(args: argparse.Namespace) -> BaselineConfig:
    config = load_config(MCQ_CONFIG)
    config.model = args.model
    config.output_dir = args.output_root / f"{args.run_prefix}-server"
    config.launch.mode = "local"
    config.launch.port = args.port
    config.launch.gpus = args.gpus
    config.launch.tp = args.tp
    config.launch.dp = args.dp
    config.launch.api_server_count = args.api_server_count
    config.launch.max_model_len = args.max_model_len
    config.launch.gpu_memory_utilization = args.gpu_memory_utilization
    config.launch.wait_timeout_s = args.wait_timeout_s
    config.launch.vllm_extra = args.vllm_extra
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPQA MCQ + open-ended baselines for one model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-prefix")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/baselines/matrix-20260501"))
    parser.add_argument("--num-examples", type=int, default=198)
    parser.add_argument("--rollouts-per-example", type=int, default=16)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--score-max-concurrency", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--client-type", default="openai_chat_completions")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--api-server-count", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--wait-timeout-s", type=float, default=900.0)
    parser.add_argument("--vllm-extra-json", default="{}")
    parser.add_argument("--sampling-extra-json", default="{}")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.vllm_extra = json.loads(args.vllm_extra_json)
    args.sampling_extra = json.loads(args.sampling_extra_json)
    args.run_prefix = args.run_prefix or slug(args.model)
    expected_rollout_count = expected_rollouts(args.num_examples, [], args.rollouts_per_example)

    planned_outputs = [_task_output_dir(load_config(path), args) for path in TASK_CONFIGS]
    if not args.force and all(artifact_complete(path, expected_rollout_count) for path in planned_outputs):
        print(
            json.dumps(
                {"model": args.model, "skipped": True, "outputs": [str(path) for path in planned_outputs]},
                indent=2,
            )
        )
        return

    summaries: dict[str, Any] = {}
    with InferenceProvisioner(_server_config(args)) as endpoint:
        for path in TASK_CONFIGS:
            config = _load_task_config(path, args, endpoint.base_url)
            if not args.force and artifact_complete(config.output_dir, expected_rollout_count):
                summaries[config.dataset or path.stem] = {"skipped": True, "output_dir": str(config.output_dir)}
                continue
            summary = run_baseline(copy.deepcopy(config))
            summaries[config.dataset or path.stem] = {"summary": summary, "output_dir": str(config.output_dir)}

    print(json.dumps({"model": args.model, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
