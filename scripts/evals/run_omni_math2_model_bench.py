from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from prime_rl.baselines.benchmark import (
    ModelSpec,
    artifact_complete,
    expected_rollouts,
    filter_blocked_specs,
    select_specs,
)
from prime_rl.baselines.config import BaselineConfig, load_config
from prime_rl.baselines.provision import InferenceProvisioner
from prime_rl.baselines.runner import run_baseline

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "configs/baselines/omni_math2_hybridmath_local.toml"
QWEN_PROMPT_PACK = "configs/baselines/omni_math2_prompt_pack_qwen_nothink.yaml"
QWEN_HF_NONTHINKING_TEXT_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "max_completion_tokens": 16_384,
    "presence_penalty": 2.0,
    "repetition_penalty": 1.0,
}
GEMMA_HF_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_completion_tokens": 16_384,
}
TRINITY_MINI_SAMPLING = {
    "temperature": 0.15,
    "top_p": 0.75,
    "top_k": 50,
    "min_p": 0.06,
}
TRINITY_MINI_NONTHINKING_SAMPLING = {
    **TRINITY_MINI_SAMPLING,
    "bad_words": ["<think>", "</think>"],
    "max_completion_tokens": 16_384,
}
TRINITY_LARGE_PREVIEW_SAMPLING = {
    "temperature": 0.8,
    "top_p": 0.8,
    "top_k": 50,
    "min_p": 0.0,
    "max_completion_tokens": 32_768,
    "bad_words": ["<think>", "</think>"],
}
MARIN_HF_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 50,
}
RNJ_HF_SAMPLING = {
    "temperature": 0.2,
    "top_p": 0.95,
}
OLMO3_HF_SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
}
NO_RAW_THINK_REQUEST_EXTRAS = {
    "bad_words": ["<think>", "</think>"],
}
QWEN_NONTHINKING_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
QWEN_NONTHINKING_REQUEST_EXTRAS = {
    "include_reasoning": False,
    "chat_template_kwargs": QWEN_NONTHINKING_CHAT_TEMPLATE_KWARGS,
    **NO_RAW_THINK_REQUEST_EXTRAS,
}
TRINITY_PROMPT_PACK = "configs/baselines/omni_math2_prompt_pack_trinity_nothink.yaml"
DEFAULT_VLLM_EXTRA = {
    "enable_chunked_prefill": True,
    "generation_config": "vllm",
    "max_num_batched_tokens": 16_384,
    "max_num_seqs": 16,
}
QWEN_VLLM_EXTRA = {
    "enable_prefix_caching": False,
    "reasoning_parser": "qwen3",
    "default_chat_template_kwargs": QWEN_NONTHINKING_CHAT_TEMPLATE_KWARGS,
}
GEMMA_VLLM_EXTRA = {"enable_prefix_caching": True}
TRINITY_VLLM_EXTRA = {
    "trust_remote_code": True,
    "tool_call_parser": "hermes",
}


REQUESTED_MODELS = (
    ModelSpec("Qwen/Qwen3.5-4B", "qwen35-4b", "qwen", tp=1, dp=4, max_concurrency=64),
    ModelSpec("Qwen/Qwen3.5-9B", "qwen35-9b", "qwen", tp=1, dp=4, max_concurrency=48),
    ModelSpec("Qwen/Qwen3.5-27B", "qwen35-27b", "qwen", tp=2, dp=2, max_concurrency=32),
    ModelSpec("Qwen/Qwen3.5-35B-A3B", "qwen35-35b-a3b", "qwen", tp=2, dp=2, max_concurrency=32),
    ModelSpec("google/gemma-4-E4B-it", "gemma4-e4b-it", "gemma", tp=1, dp=4, max_concurrency=64),
    ModelSpec(
        "google/gemma-4-26B-A4B-it",
        "gemma4-26b-a4b-it",
        "gemma",
        tp=2,
        dp=2,
        max_concurrency=32,
    ),
    ModelSpec("google/gemma-4-31B-it", "gemma4-31b-it", "gemma", tp=2, dp=2, max_concurrency=24),
    ModelSpec(
        "arcee-ai/Trinity-Mini",
        "trinity-mini",
        "trinity",
        tp=2,
        dp=2,
        max_concurrency=32,
        sampling=TRINITY_MINI_NONTHINKING_SAMPLING,
        vllm_extra=TRINITY_VLLM_EXTRA,
        prompt_pack=TRINITY_PROMPT_PACK,
    ),
    ModelSpec(
        "arcee-ai/Trinity-Large-Preview",
        "trinity-large-preview-bf16",
        "trinity",
        tp=1,
        dp=8,
        max_concurrency=8,
        sampling=TRINITY_LARGE_PREVIEW_SAMPLING,
        vllm_extra=TRINITY_VLLM_EXTRA,
        prompt_pack=TRINITY_PROMPT_PACK,
        max_model_len=36_864,
        requires_multinode=True,
        launch_vllm_extra={
            "dtype": "bfloat16",
            "moe_backend": "triton",
        },
        launch_enforce_eager=True,
        launch_srun_network="disable_rdzv_get",
        launch_srun_cpus_per_task=72,
        blocked_reason=(
            "Known OOM with the current DP=8/EP/eager BF16 GH200 launch; "
            "needs a real sharded BF16 config before default eval."
        ),
    ),
    ModelSpec(
        "arcee-ai/Trinity-Large-Preview-W4A16",
        "trinity-large-preview-w4a16",
        "trinity",
        tp=4,
        dp=1,
        max_concurrency=16,
        sampling=TRINITY_LARGE_PREVIEW_SAMPLING,
        vllm_extra=TRINITY_VLLM_EXTRA,
        max_model_len=36_864,
        blocked_reason=(
            "Known cudaErrorUnsupportedPtxVersion in gptq_marlin_moe_repack "
            "on the current GH200/CUDA/vLLM stack."
        ),
    ),
    ModelSpec(
        "arcee-ai/Trinity-Large-Preview-FP8-Block",
        "trinity-large-preview-fp8-block",
        "trinity",
        tp=4,
        dp=1,
        max_concurrency=8,
        sampling=TRINITY_LARGE_PREVIEW_SAMPLING,
        vllm_extra=TRINITY_VLLM_EXTRA,
        prompt_pack=TRINITY_PROMPT_PACK,
        max_model_len=36_864,
        requires_multinode=True,
        multinode_strategy="tp",
        launch_vllm_extra={
            "pipeline_parallel_size": 2,
            "moe_backend": "triton",
        },
        launch_srun_network="disable_rdzv_get",
        launch_srun_cpus_per_task=72,
    ),
    ModelSpec(
        "marin-community/marin-8b-instruct",
        "marin-8b-instruct",
        "marin",
        tp=1,
        dp=4,
        max_concurrency=64,
        sampling=MARIN_HF_SAMPLING,
        max_model_len=4096,
        max_completion_tokens=3840,
    ),
    ModelSpec(
        "allenai/Olmo-3-7B-Instruct-SFT",
        "olmo3-7b-instruct-sft",
        "olmo",
        tp=1,
        dp=4,
        max_concurrency=64,
        sampling=OLMO3_HF_SAMPLING,
        sampling_drop=("top_k", "min_p"),
    ),
    ModelSpec(
        "allenai/Olmo-3-7B-Instruct-DPO",
        "olmo3-7b-instruct-dpo",
        "olmo",
        tp=1,
        dp=4,
        max_concurrency=64,
        sampling=OLMO3_HF_SAMPLING,
        sampling_drop=("top_k", "min_p"),
    ),
    ModelSpec(
        "allenai/Olmo-3-7B-Think-SFT",
        "olmo3-7b-think-sft",
        "olmo",
        tp=1,
        dp=4,
        max_concurrency=64,
        sampling=OLMO3_HF_SAMPLING,
        sampling_drop=("top_k", "min_p"),
    ),
    ModelSpec(
        "allenai/Olmo-3-7B-Think-DPO",
        "olmo3-7b-think-dpo",
        "olmo",
        tp=1,
        dp=4,
        max_concurrency=64,
        sampling=OLMO3_HF_SAMPLING,
        sampling_drop=("top_k", "min_p"),
    ),
    ModelSpec(
        "EssentialAI/rnj-1-instruct",
        "rnj-1-instruct",
        "rnj",
        tp=1,
        dp=4,
        max_concurrency=64,
        sampling=RNJ_HF_SAMPLING,
        sampling_drop=("top_k", "min_p"),
    ),
)

BASE_MODELS = (
    ModelSpec("Qwen/Qwen3.5-4B-Base", "qwen35-4b-base", "qwen", tp=1, dp=4, max_concurrency=64),
    ModelSpec("Qwen/Qwen3.5-9B-Base", "qwen35-9b-base", "qwen", tp=1, dp=4, max_concurrency=48),
    ModelSpec(
        "Qwen/Qwen3.5-35B-A3B-Base",
        "qwen35-35b-a3b-base",
        "qwen",
        tp=2,
        dp=2,
        max_concurrency=32,
    ),
    ModelSpec("google/gemma-4-E4B", "gemma4-e4b-base", "gemma", tp=1, dp=4, max_concurrency=64),
    ModelSpec("google/gemma-4-26B-A4B", "gemma4-26b-a4b-base", "gemma", tp=2, dp=2, max_concurrency=32),
    ModelSpec("google/gemma-4-31B", "gemma4-31b-base", "gemma", tp=2, dp=2, max_concurrency=24),
)


def _model_specs(model_set: str) -> list[ModelSpec]:
    if model_set == "requested":
        return list(REQUESTED_MODELS)
    if model_set == "all":
        return [*REQUESTED_MODELS, *BASE_MODELS]
    if model_set == "qwen":
        return [spec for spec in REQUESTED_MODELS if spec.family == "qwen"]
    if model_set == "gemma":
        return [spec for spec in REQUESTED_MODELS if spec.family == "gemma"]
    if model_set == "trinity":
        return [spec for spec in REQUESTED_MODELS if spec.family == "trinity"]
    raise ValueError(f"unknown model_set={model_set!r}")


def _apply_common_overrides(config: BaselineConfig, args: argparse.Namespace) -> None:
    config.num_examples = args.num_examples
    config.record_ids = [str(record_id) for record_id in args.record_ids]
    config.rollouts_per_example = args.rollouts_per_example
    config.score_max_concurrency = args.score_max_concurrency
    config.max_retries = args.max_retries
    config.fail_on_error = False
    config.launch.gpus = args.gpus
    if args.server_start_retries is not None:
        config.launch.server_start_retries = args.server_start_retries
    if args.max_model_len is not None:
        config.launch.max_model_len = args.max_model_len
    config.launch.gpu_memory_utilization = args.gpu_memory_utilization
    config.launch.wait_timeout_s = args.wait_timeout_s


def _effective_max_model_len(base: BaselineConfig, spec: ModelSpec, args: argparse.Namespace) -> int:
    if args.max_model_len is not None:
        max_model_len = args.max_model_len
        if spec.max_model_len is not None:
            return min(max_model_len, spec.max_model_len)
        return max_model_len
    if spec.max_model_len is not None:
        return spec.max_model_len
    return base.launch.max_model_len


def _server_vllm_extra(spec: ModelSpec, args: argparse.Namespace) -> dict[str, Any]:
    extra = copy.deepcopy(DEFAULT_VLLM_EXTRA)
    if spec.family == "qwen":
        extra.update(copy.deepcopy(QWEN_VLLM_EXTRA))
    if spec.family == "gemma":
        extra.update(copy.deepcopy(GEMMA_VLLM_EXTRA))
    if spec.vllm_extra:
        extra.update(copy.deepcopy(spec.vllm_extra))
    if spec.launch_vllm_extra:
        extra.update(copy.deepcopy(spec.launch_vllm_extra))
    extra.update(args.vllm_extra)
    return extra


def _family_sampling_preset(spec: ModelSpec) -> dict[str, Any]:
    if spec.sampling is not None:
        return dict(spec.sampling)
    if spec.family == "gemma":
        return dict(GEMMA_HF_SAMPLING)
    if spec.family == "qwen":
        return copy.deepcopy({
            **QWEN_HF_NONTHINKING_TEXT_SAMPLING,
            **QWEN_NONTHINKING_REQUEST_EXTRAS,
        })
    raise ValueError(f"no sampling preset for family={spec.family!r}")


def _sampling_args(base: BaselineConfig, spec: ModelSpec, args: argparse.Namespace) -> dict[str, Any]:
    sampling = dict(base.sampling_args)
    for key in spec.sampling_drop:
        sampling.pop(key, None)
    sampling.update(_family_sampling_preset(spec))
    sampling.update(args.sampling_extra)
    if spec.max_completion_tokens is not None:
        current = int(sampling.get("max_completion_tokens") or spec.max_completion_tokens)
        sampling["max_completion_tokens"] = min(current, spec.max_completion_tokens)
    return sampling


def _task_output_dir(spec: ModelSpec, args: argparse.Namespace) -> Path:
    return args.output_root / f"{args.run_prefix}-{spec.short_name}-omni_math2-k{args.rollouts_per_example}"


def _server_config(base: BaselineConfig, spec: ModelSpec, args: argparse.Namespace) -> BaselineConfig:
    config = copy.deepcopy(base)
    _apply_common_overrides(config, args)
    use_multinode = spec.requires_multinode and args.multinode_nodes > 1
    use_multinode_tp = use_multinode and spec.multinode_strategy == "tp"
    config.model = spec.model
    config.output_dir = args.output_root / f"{args.run_prefix}-{spec.short_name}-server"
    config.launch.mode = "srun_multinode" if use_multinode else args.server_launch_mode
    config.launch.port = args.port
    config.launch.tp = spec.tp
    config.launch.dp = spec.dp
    config.launch.api_server_count = 1 if use_multinode_tp else (args.gpus_per_node if use_multinode else config.launch.dp)
    config.launch.max_model_len = _effective_max_model_len(base, spec, args)
    if spec.launch_enforce_eager is not None:
        config.launch.enforce_eager = spec.launch_enforce_eager
    config.launch.use_deep_gemm = spec.launch_use_deep_gemm
    config.launch.nodes = args.multinode_nodes
    config.launch.gpus_per_node = args.gpus_per_node
    config.launch.srun_job_id = args.srun_job_id
    config.launch.data_parallel_size_local = None if use_multinode_tp else (args.gpus_per_node if use_multinode else None)
    config.launch.enable_expert_parallel = use_multinode and spec.multinode_strategy == "dp_ep"
    config.launch.srun_network = spec.launch_srun_network
    config.launch.srun_cpus_per_task = spec.launch_srun_cpus_per_task
    config.launch.vllm_extra.update(_server_vllm_extra(spec, args))
    return config


def _task_config(
    base: BaselineConfig,
    spec: ModelSpec,
    args: argparse.Namespace,
    endpoint_url: str,
) -> BaselineConfig:
    config = copy.deepcopy(base)
    _apply_common_overrides(config, args)
    config.model = spec.model
    config.output_dir = _task_output_dir(spec, args)
    config.max_concurrency = min(args.max_concurrency, spec.max_concurrency)
    config.base_url = endpoint_url
    config.client_type = args.client_type
    config.api_profile = "vllm_permissive"
    config.launch.mode = "external"
    config.launch.max_model_len = _effective_max_model_len(base, spec, args)
    if spec.prompt_pack is not None:
        config.env_args["prompts_ref"] = spec.prompt_pack
    elif spec.family == "qwen":
        config.env_args["prompts_ref"] = QWEN_PROMPT_PACK
    config.sampling_args = _sampling_args(config, spec, args)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Omni-MATH-2 HybridMath baseline matrix.")
    parser.add_argument("--config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/baselines/omni-math2-matrix-20260501"))
    parser.add_argument("--run-prefix", default="hybridmath")
    parser.add_argument("--model-set", choices=["requested", "all", "qwen", "gemma", "trinity"], default="requested")
    parser.add_argument("--models", nargs="*", help="Optional explicit HF model IDs or short names.")
    parser.add_argument("--num-examples", type=int, default=600)
    parser.add_argument("--record-ids", nargs="*", default=[])
    parser.add_argument("--rollouts-per-example", type=int, default=16)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--score-max-concurrency", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--client-type", default="openai_chat_completions")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--multinode-nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--srun-job-id")
    parser.add_argument("--server-launch-mode", choices=["local", "srun"], default="local")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--wait-timeout-s", type=float, default=1200.0)
    parser.add_argument("--vllm-extra-json", default="{}")
    parser.add_argument("--sampling-extra-json", default="{}")
    parser.add_argument("--server-start-retries", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include-blocked",
        action="store_true",
        help="Include model specs with known fatal launch blockers.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.vllm_extra = json.loads(args.vllm_extra_json)
    args.sampling_extra = json.loads(args.sampling_extra_json)
    requested = set(args.models or [])
    try:
        specs = select_specs(_model_specs(args.model_set), requested)
        specs, blocked_summaries = filter_blocked_specs(
            specs,
            explicitly_requested=bool(requested),
            include_blocked=args.include_blocked,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    missing_multinode_specs = [
        spec for spec in specs if spec.requires_multinode and args.multinode_nodes <= 1
    ]
    if missing_multinode_specs:
        details = "\n".join(
            f"- {spec.short_name}: requires --multinode-nodes > 1"
            for spec in missing_multinode_specs
        )
        if requested or args.include_blocked:
            raise SystemExit(
                "Explicitly selected multi-node model(s) without "
                f"--multinode-nodes > 1:\n{details}"
            )
        for spec in missing_multinode_specs:
            blocked_summaries[spec.short_name] = {
                "skipped": True,
                "blocked": True,
                "blocked_reason": "requires --multinode-nodes > 1",
            }
        missing_multinode_names = {spec.short_name for spec in missing_multinode_specs}
        specs = [spec for spec in specs if spec.short_name not in missing_multinode_names]
    if not specs:
        if blocked_summaries:
            print(json.dumps({"summaries": blocked_summaries}, indent=2))
            return
        raise SystemExit("No models selected.")

    base = load_config(args.config)
    expected_rollout_count = expected_rollouts(args.num_examples, args.record_ids, args.rollouts_per_example)
    if args.dry_run:
        plan = []
        for spec in specs:
            server_config = _server_config(base, spec, args)
            task_config = _task_config(base, spec, args, f"http://127.0.0.1:{args.port}/v1")
            plan.append(
                {
                    "model": spec.model,
                    "short_name": spec.short_name,
                    "tp": server_config.launch.tp,
                    "dp": server_config.launch.dp,
                    "api_server_count": server_config.launch.api_server_count,
                    "server_launch_mode": server_config.launch.mode,
                    "nodes": server_config.launch.nodes,
                    "gpus_per_node": server_config.launch.gpus_per_node,
                    "multinode_strategy": spec.multinode_strategy if spec.requires_multinode else None,
                    "data_parallel_size_local": server_config.launch.data_parallel_size_local,
                    "enable_expert_parallel": server_config.launch.enable_expert_parallel,
                    "use_deep_gemm": server_config.launch.use_deep_gemm,
                    "srun_network": server_config.launch.srun_network,
                    "srun_cpus_per_task": server_config.launch.srun_cpus_per_task,
                    "gpu_memory_utilization": server_config.launch.gpu_memory_utilization,
                    "max_model_len": server_config.launch.max_model_len,
                    "server_vllm_extra": server_config.launch.vllm_extra,
                    "max_concurrency": task_config.max_concurrency,
                    "record_ids": task_config.record_ids,
                    "sampling": task_config.sampling_args,
                    "prompts_ref": task_config.env_args.get("prompts_ref"),
                    "output_dir": str(task_config.output_dir),
                    "blocked_reason": spec.blocked_reason,
                }
            )
        print(
            json.dumps(
                {
                    "expected_rollouts_per_model": expected_rollout_count,
                    "blocked_skipped": blocked_summaries,
                    "plan": plan,
                },
                indent=2,
            )
        )
        return

    summaries: dict[str, Any] = dict(blocked_summaries)
    for spec in specs:
        planned_output = _task_output_dir(spec, args)
        if not args.force and artifact_complete(
            planned_output,
            expected_rollout_count,
        ):
            summaries[spec.short_name] = {"skipped": True, "output_dir": str(planned_output)}
            continue
        with InferenceProvisioner(_server_config(base, spec, args)) as endpoint:
            config = _task_config(base, spec, args, endpoint.base_url)
            summary = run_baseline(config)
            summaries[spec.short_name] = {"summary": summary, "output_dir": str(config.output_dir)}

    print(json.dumps({"summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
