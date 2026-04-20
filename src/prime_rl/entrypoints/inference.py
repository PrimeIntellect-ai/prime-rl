import subprocess
import sys
import tempfile
from pathlib import Path

import tomli_w

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import format_log_message, get_config_dir, get_log_dir
from prime_rl.utils.process import set_proc_title

INFERENCE_TOML = "inference.toml"
INFERENCE_SBATCH = "inference.sbatch"


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
    is_multi_node = config.deployment.type == "multi_node"
    is_single_node_router = config.deployment.type == "single_node" and config.deployment.use_router
    dp_per_node = (
        config.data_parallel_size_local or config.parallel.dp
        if is_single_node_router
        else config.deployment.gpus_per_node // config.parallel.tp
    )

    template_vars = dict(
        **config.slurm.template_vars,
        config_path=config_path,
        output_dir=config.output_dir,
        gpus_per_node=config.deployment.gpus_per_node,
        dp_per_node=dp_per_node,
        num_nodes=getattr(config.deployment, "num_nodes", 1),
        port=config.server.port,
        disaggregated=is_disaggregated,
        single_node_router=is_single_node_router,
    )

    if is_disaggregated:
        template_vars.update(
            num_prefill_nodes=config.deployment.num_prefill_nodes,
            num_decode_nodes=config.deployment.num_decode_nodes,
            num_prefill_replicas=config.deployment.num_prefill_replicas,
            num_decode_replicas=config.deployment.num_decode_replicas,
            prefill_port=config.deployment.prefill_port,
            decode_port=config.deployment.decode_port,
            router_port=config.deployment.router_port,
            router_policy=config.deployment.router_policy,
            data_parallel_rpc_port=config.data_parallel_rpc_port,
            use_deep_gemm=config.use_deep_gemm,
            prefill_env_overrides=config.deployment.prefill_env_overrides,
            decode_env_overrides=config.deployment.decode_env_overrides,
            kv_offload=config.deployment.kv_cache_offload is not None,
            kv_offload_cpu_bytes=int(config.deployment.kv_cache_offload.cpu_bytes)
            if config.deployment.kv_cache_offload
            else 0,
        )
    elif is_multi_node or is_single_node_router:
        template_vars.update(
            router_port=config.deployment.router_port,
            backend_port=config.deployment.backend_port,
            router_policy=config.deployment.router_policy,
        )
    script = template.render(**template_vars)

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)


def inference_slurm(config: InferenceConfig):
    """Run inference via SLURM."""
    assert config.slurm is not None

    logger = setup_logger("info")

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


def run_single_node_with_router(config: InferenceConfig):
    """Run a single-node backend behind a local vllm-router."""
    from prime_rl.inference.server import setup_vllm_env

    logger = setup_logger("info")

    host = config.server.host or "0.0.0.0"
    router_port = config.deployment.router_port
    backend_port = config.deployment.backend_port
    router_dp_size = config.data_parallel_size_local or config.parallel.dp

    logger.info(f"Starting single-node backend on http://{host}:{backend_port}/v1")
    logger.info(f"Starting single-node router on http://{host}:{router_port}/v1\n")

    setup_vllm_env(config)

    with tempfile.TemporaryDirectory(prefix="prime-rl-inference-") as tmpdir:
        config_path = write_config(config, Path(tmpdir))

        backend_cmd = [
            sys.executable,
            "-m",
            "prime_rl.entrypoints.inference",
            "@",
            str(config_path),
            "--server.host",
            host,
            "--server.port",
            str(backend_port),
            "--deployment.use_router",
            "false",
        ]
        router_cmd = [
            "vllm-router",
            "--policy",
            config.deployment.router_policy,
            "--worker-urls",
            f"http://127.0.0.1:{backend_port}",
            "--host",
            "0.0.0.0",
            "--port",
            str(router_port),
            "--intra-node-data-parallel-size",
            str(router_dp_size),
            "--worker-startup-timeout-secs",
            "4200",
            "--log-level",
            "debug",
        ]

        backend_proc = subprocess.Popen(backend_cmd)
        try:
            result = subprocess.run(router_cmd)
        finally:
            if backend_proc.poll() is None:
                backend_proc.terminate()
                try:
                    backend_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    backend_proc.kill()
                    backend_proc.wait()

        if result.returncode != 0:
            sys.exit(result.returncode)


def inference_local(config: InferenceConfig):
    """Run inference locally."""
    from prime_rl.inference.server import setup_vllm_env

    logger = setup_logger("info")

    if config.dry_run:
        logger.success("Dry run complete. To start inference locally, remove --dry-run from your command.")
        return

    if config.deployment.type == "single_node" and config.deployment.use_router:
        run_single_node_with_router(config)
        return

    host = config.server.host or "0.0.0.0"
    port = config.server.port
    logger.info(f"Starting inference on http://{host}:{port}/v1\n")

    setup_vllm_env(config)

    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_extra=config.vllm_extra)


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
