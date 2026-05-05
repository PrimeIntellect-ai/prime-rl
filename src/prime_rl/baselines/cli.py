from __future__ import annotations

import argparse
import json
from pathlib import Path

from prime_rl.baselines.config import BaselineConfig, load_config
from prime_rl.baselines.runner import run_baseline


def _parse_k_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run verifier-backed model baselines.")
    parser.add_argument("--config", type=Path, help="TOML config file.")
    parser.add_argument("--env-id")
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocol")
    parser.add_argument("--dataset")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-var")
    parser.add_argument("--client-type")
    parser.add_argument("--api-profile")
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--record-ids", nargs="+")
    parser.add_argument("--rollouts-per-example", type=int)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--score-max-concurrency", type=int)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--no-fail-on-error", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ks", type=_parse_k_list)
    parser.add_argument("--env-args-json", default=None)
    parser.add_argument("--sampling-json", default=None)
    parser.add_argument("--vllm-extra-json", default=None)
    parser.add_argument("--verifiers-path", type=Path)
    parser.add_argument("--env-path", action="append", type=Path, default=[])
    parser.add_argument("--launch-mode", choices=["external", "local", "srun", "srun_multinode"])
    parser.add_argument("--port", type=int)
    parser.add_argument("--gpus")
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument("--srun-job-id")
    parser.add_argument("--router-port", type=int)
    parser.add_argument("--backend-port", type=int)
    parser.add_argument("--data-parallel-size-local", type=int)
    parser.add_argument("--data-parallel-rpc-port", type=int)
    parser.add_argument("--enable-expert-parallel", action="store_true", default=None)
    parser.add_argument("--tp", type=int)
    parser.add_argument("--dp", type=int)
    parser.add_argument("--api-server-count", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--wait-timeout-s", type=float)
    parser.add_argument("--enforce-eager", action="store_true", default=None)
    parser.add_argument("--chat-template")
    return parser


def _apply_overrides(config: BaselineConfig, args: argparse.Namespace) -> BaselineConfig:
    for field in (
        "env_id",
        "model",
        "output_dir",
        "protocol",
        "dataset",
        "base_url",
        "api_key_var",
        "client_type",
        "api_profile",
        "num_examples",
        "record_ids",
        "rollouts_per_example",
        "max_concurrency",
        "score_max_concurrency",
        "max_retries",
        "seed",
        "ks",
        "verifiers_path",
    ):
        value = getattr(args, field, None)
        if value is not None:
            setattr(config, field, value)

    if args.env_path:
        config.env_paths.extend(args.env_path)
    if args.no_fail_on_error:
        config.fail_on_error = False
    if args.env_args_json:
        config.env_args.update(json.loads(args.env_args_json))
    if args.sampling_json:
        config.sampling_args.update(json.loads(args.sampling_json))
    if args.vllm_extra_json:
        config.launch.vllm_extra.update(json.loads(args.vllm_extra_json))
    if args.launch_mode:
        config.launch.mode = args.launch_mode
    for arg_name, field in (
        ("port", "port"),
        ("gpus", "gpus"),
        ("nodes", "nodes"),
        ("gpus_per_node", "gpus_per_node"),
        ("srun_job_id", "srun_job_id"),
        ("router_port", "router_port"),
        ("backend_port", "backend_port"),
        ("data_parallel_size_local", "data_parallel_size_local"),
        ("data_parallel_rpc_port", "data_parallel_rpc_port"),
        ("enable_expert_parallel", "enable_expert_parallel"),
        ("tp", "tp"),
        ("dp", "dp"),
        ("api_server_count", "api_server_count"),
        ("max_model_len", "max_model_len"),
        ("gpu_memory_utilization", "gpu_memory_utilization"),
        ("wait_timeout_s", "wait_timeout_s"),
        ("enforce_eager", "enforce_eager"),
        ("chat_template", "chat_template"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config.launch, field, value)
    return config


def _config_from_args(args: argparse.Namespace) -> BaselineConfig:
    if args.config is not None:
        config = load_config(args.config)
    else:
        missing = [name for name in ("env_id", "model", "output_dir") if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"Missing required args without --config: {', '.join(missing)}")
        config = BaselineConfig(env_id=args.env_id, model=args.model, output_dir=args.output_dir)
    return _apply_overrides(config, args)


def main() -> None:
    args = build_parser().parse_args()
    config = _config_from_args(args)
    summary = run_baseline(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
