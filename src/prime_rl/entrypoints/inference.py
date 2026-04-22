import os
import subprocess
import sys
from pathlib import Path

import tomli_w

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import format_log_message, get_config_dir, get_log_dir
from prime_rl.utils.process import set_proc_title

INFERENCE_TOML = "inference.toml"
INFERENCE_SBATCH = "inference.sbatch"


def _log_level() -> str:
    return os.environ.get("PRIME_LOG_LEVEL", "info")


def _router_bind_host(host: str | None) -> str:
    return host or "0.0.0.0"


def _router_worker_host(host: str | None) -> str:
    if host in (None, "0.0.0.0", "::"):
        return "127.0.0.1"
    return host


def build_single_node_router_cmd(config: InferenceConfig) -> list[str]:
    assert config.deployment.type == "single_node"

    dp_per_node = config.data_parallel_size_local or config.parallel.dp
    worker_url = f"http://{_router_worker_host(config.server.host)}:{config.deployment.backend_port}"

    return [
        "vllm-router",
        "--policy",
        config.deployment.router.policy,
        "--worker-urls",
        worker_url,
        "--host",
        _router_bind_host(config.server.host),
        "--port",
        str(config.deployment.router.port),
        "--intra-node-data-parallel-size",
        str(dp_per_node),
        "--worker-startup-timeout-secs",
        "4200",
        "--log-level",
        _log_level(),
    ]


def build_single_node_backend_config(config: InferenceConfig) -> InferenceConfig:
    assert config.deployment.type == "single_node"

    backend_config = config.model_copy(deep=True)
    backend_config.server.port = config.deployment.backend_port
    return backend_config


def write_config(config: InferenceConfig, output_dir: Path, exclude: set[str] | None = None) -> Path:
    """Write resolved config to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / INFERENCE_TOML
    with open(config_path, "wb") as f:
        tomli_w.dump(config.model_dump(exclude=exclude, exclude_none=True, mode="json"), f)
    return config_path


def write_slurm_script(config: InferenceConfig, config_path: Path, script_path: Path) -> None:
    """Write the SLURM script to disk."""
    from jinja2 import Environment, FileSystemLoader

    assert config.slurm is not None
    assert config.slurm.template_path is not None

    env = Environment(loader=FileSystemLoader(config.slurm.template_path.parent), keep_trailing_newline=True)
    template = env.get_template(config.slurm.template_path.name)

    is_disaggregated = config.deployment.type == "disaggregated"
    dp_per_node = config.deployment.gpus_per_node // config.parallel.tp

    template_vars = dict(
        **config.slurm.template_vars,
        config_path=config_path,
        output_dir=config.output_dir,
        gpus_per_node=config.deployment.gpus_per_node,
        dp_per_node=dp_per_node,
        num_nodes=getattr(config.deployment, "num_nodes", 1),
        port=config.server.port,
        disaggregated=is_disaggregated,
        prime_log_level=_log_level(),
    )

    is_multi_node = config.deployment.type == "multi_node"

    if is_disaggregated:
        template_vars.update(
            num_prefill_nodes=config.deployment.num_prefill_nodes,
            num_decode_nodes=config.deployment.num_decode_nodes,
            num_prefill_replicas=config.deployment.num_prefill_replicas,
            num_decode_replicas=config.deployment.num_decode_replicas,
            prefill_port=config.deployment.prefill_port,
            decode_port=config.deployment.decode_port,
            router_port=config.deployment.router.port,
            router_policy=config.deployment.router.policy,
            data_parallel_rpc_port=config.data_parallel_rpc_port,
            use_deep_gemm=config.use_deep_gemm,
            prefill_env_overrides=config.deployment.prefill_env_overrides,
            decode_env_overrides=config.deployment.decode_env_overrides,
            kv_offload=config.deployment.kv_cache_offload is not None,
            kv_offload_cpu_bytes=int(config.deployment.kv_cache_offload.cpu_bytes)
            if config.deployment.kv_cache_offload
            else 0,
        )
    elif is_multi_node:
        template_vars.update(
            router_port=config.deployment.router.port,
            backend_port=config.deployment.backend_port,
            router_policy=config.deployment.router.policy,
        )

    script = template.render(**template_vars)

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)


def inference_slurm(config: InferenceConfig):
    """Run inference via SLURM."""
    assert config.slurm is not None

    logger = setup_logger(_log_level())

    config_dir = get_config_dir(config.output_dir)
    exclude = (
        {"deployment", "slurm", "dry_run"}
        if config.deployment.type in ("multi_node", "disaggregated")
        else {"slurm", "dry_run"}
    )
    config_path = write_config(config, config_dir, exclude=exclude)
    logger.info(f"Wrote config to {config_path}")

    script_path = config.output_dir / INFERENCE_SBATCH
    write_slurm_script(config, config_path, script_path)
    logger.info(f"Wrote SLURM script to {script_path}")

    log_dir = get_log_dir(config.output_dir)
    num_nodes = getattr(config.deployment, "num_nodes", 1)
    log_message = format_log_message(log_dir=log_dir, inference=True, job_log=True, num_infer_nodes=num_nodes)

    if config.dry_run:
        logger.success(f"Dry run complete. To submit manually:\n\n  sbatch {script_path}\n\n{log_message}")
        return

    logger.info(f"Submitting: sbatch {script_path}")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr.strip()}")
        sys.exit(1)

    logger.success(f"{result.stdout.strip()}\n\n{log_message}")


def inference_local(config: InferenceConfig):
    """Run inference locally."""
    from prime_rl.inference.server import setup_vllm_env

    logger = setup_logger(_log_level())

    if config.dry_run:
        logger.success("Dry run complete. To start inference locally, remove --dry-run from your command.")
        return

    setup_vllm_env(config)

    from prime_rl.inference.vllm.server import server  # pyright: ignore

    if config.deployment.type != "single_node":
        host = config.server.host or "0.0.0.0"
        port = config.server.port
        logger.info(f"Starting inference on http://{host}:{port}/v1\n")
        server(config, vllm_extra=config.vllm_extra)
        return

    router_cmd = build_single_node_router_cmd(config)
    backend_config = build_single_node_backend_config(config)
    public_host = config.server.host or "0.0.0.0"
    logger.info(
        f"Starting inference router on http://{public_host}:{config.deployment.router.port}/v1 "
        f"with backend on http://{_router_worker_host(config.server.host)}:{config.deployment.backend_port}/v1\n"
    )

    router_process = subprocess.Popen(router_cmd)
    try:
        server(backend_config, vllm_extra=config.vllm_extra)
    finally:
        if router_process.poll() is None:
            router_process.terminate()
            try:
                router_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                router_process.kill()
                router_process.wait(timeout=5)


def inference(config: InferenceConfig):
    if config.slurm is not None:
        inference_slurm(config)
    else:
        inference_local(config)


def main():
    set_proc_title("Inference")
    inference(cli(InferenceConfig))


if __name__ == "__main__":
    main()
