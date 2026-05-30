import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread

import pynvml
import tomli_w

from prime_rl.configs.rl import GpuLayoutDeploymentConfig, RLConfig
from prime_rl.utils.config import cli, find_package_resource
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import (
    clean_future_steps,
    format_log_message,
    get_ckpt_dir,
    get_log_dir,
    resolve_latest_ckpt_step,
    validate_output_dir,
)
from prime_rl.utils.process import cleanup_processes, cleanup_threads, monitor_process, set_proc_title

RL_TOML = "rl.toml"
RL_SBATCH = "rl.sbatch"

TRAINER_TOML = "trainer.toml"
ORCHESTRATOR_TOML = "orchestrator.toml"
INFERENCE_TOML = "inference.toml"
GPU_LAYOUT_SCRIPT = "gpu_layout_rl.sh"
MULTI_NODE_SCRIPT = "multi_node_rl.sh"
MULTI_NODE_TEMPLATE = "multi_node_rl.sbatch.j2"
_DEFAULT_PORT_BASE = 29500
_GPU_LAYOUT_CPU_MARGIN = 16
_GPU_LAYOUT_MEM_MARGIN_MB = 16_384


def get_physical_gpu_ids() -> list[int]:
    """Return physical GPU IDs visible to the launcher."""
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_visible is None:
        pynvml.nvmlInit()
        return list(range(pynvml.nvmlDeviceGetCount()))
    return [int(token.strip()) for token in raw_visible.split(",") if token.strip()]


def write_config(config: RLConfig, output_dir: Path, exclude: set[str] | None = None) -> None:
    """Write resolved config to disk, excluding launcher-only fields."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = config.model_dump(exclude=exclude, exclude_none=True, mode="json")
    with open(output_dir / RL_TOML, "wb") as f:
        tomli_w.dump(config_dict, f)


def write_subconfigs(config: RLConfig, output_dir: Path) -> None:
    """Write resolved subconfigs to disk as TOML files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / TRAINER_TOML, "wb") as f:
        tomli_w.dump(config.trainer.model_dump(exclude_none=True, mode="json"), f)

    with open(output_dir / ORCHESTRATOR_TOML, "wb") as f:
        tomli_w.dump(config.orchestrator.model_dump(exclude_none=True, mode="json"), f)

    if config.inference is not None:
        # Exclude launcher-only fields that are not needed by the vLLM server
        exclude_inference = {"deployment", "slurm", "output_dir", "dry_run"}
        with open(output_dir / INFERENCE_TOML, "wb") as f:
            tomli_w.dump(config.inference.model_dump(exclude=exclude_inference, exclude_none=True, mode="json"), f)


def _gpu_layout_inference_servers(deployment: GpuLayoutDeploymentConfig) -> list[dict[str, int]]:
    servers: list[dict[str, int]] = []
    port = deployment.inference_port_start
    for node_idx, node in enumerate(deployment.nodes):
        for gpu_id in node.inference:
            servers.append(
                {
                    "server_idx": len(servers),
                    "node_idx": node_idx,
                    "gpu_id": gpu_id,
                    "port": port,
                }
            )
            port += 1
    return servers


def _gpu_layout_template_vars(config: RLConfig, config_dir: Path) -> dict:
    assert config.deployment.type == "gpu_layout"
    deployment = config.deployment
    trainer_node_idx = deployment.trainer_node_indices[0]
    trainer_gpu_ids = deployment.nodes[trainer_node_idx].trainer

    return {
        "project_dir": (config.slurm.project_dir if config.slurm is not None else Path.cwd()).resolve(),
        "config_dir": config_dir,
        "output_dir": config.output_dir,
        "orchestrator_output_dir": config.orchestrator.output_dir,
        "gpu_layout_script_name": GPU_LAYOUT_SCRIPT,
        "gpu_layout_cpu_margin": _GPU_LAYOUT_CPU_MARGIN,
        "gpu_layout_mem_margin_mb": _GPU_LAYOUT_MEM_MARGIN_MB,
        "num_nodes": deployment.num_nodes,
        "gpus_per_node": deployment.gpus_per_node,
        "inference_servers": _gpu_layout_inference_servers(deployment),
        "trainer_node_idx": trainer_node_idx,
        "trainer_gpu_ids": ",".join(map(str, trainer_gpu_ids)),
        "trainer_gpu_count": len(trainer_gpu_ids),
        "master_port": int(os.environ.get("MASTER_PORT", "29500")),
        "use_nccl_broadcast": config.weight_broadcast is not None and config.weight_broadcast.type == "nccl",
        "ranks_filter": ",".join(map(str, config.trainer.log.ranks_filter)),
    }


def _multi_node_template_vars(config: RLConfig, config_dir: Path) -> dict:
    """Topology/placement vars for the multi_node template, shared by the sbatch and
    in-allocation launch paths. The launch axis (sbatch vs srun) is decoupled from the
    topology axis (these vars): both paths render the same template; only the per-node
    `LANE_*` env (hosts/ports/tag) differs, and the template reads those from the
    environment with fallbacks (see lane-contract.md)."""
    assert config.deployment.type == "multi_node"
    deployment = config.deployment

    if config.inference is not None and config.inference.deployment.type == "disaggregated":
        infer_deploy = config.inference.deployment
        return {
            "is_disaggregated": True,
            "config_dir": config_dir,
            "output_dir": config.output_dir,
            "orchestrator_output_dir": config.orchestrator.output_dir,
            "num_train_nodes": deployment.num_train_nodes,
            "num_infer_nodes": infer_deploy.num_nodes * deployment.num_infer_replicas,
            "nodes_per_infer_replica": infer_deploy.num_nodes,
            "num_infer_replicas": deployment.num_infer_replicas,
            "num_prefill_nodes": infer_deploy.num_prefill_nodes,
            "num_decode_nodes": infer_deploy.num_decode_nodes,
            "num_prefill_replicas": infer_deploy.num_prefill_replicas,
            "num_decode_replicas": infer_deploy.num_decode_replicas,
            "gpus_per_node": deployment.gpus_per_node,
            "router_port": infer_deploy.router_port,
            "prefill_port": infer_deploy.prefill_port,
            "decode_port": infer_deploy.decode_port,
            "inference_tp": config.inference.parallel.tp,
            "inference_data_parallel_rpc_port": config.inference.data_parallel_rpc_port,
            "use_deep_gemm": config.inference.use_deep_gemm,
            "prefill_env_overrides": infer_deploy.prefill_env_overrides,
            "decode_env_overrides": infer_deploy.decode_env_overrides,
            "dp_per_node": deployment.gpus_per_node // config.inference.parallel.tp,
            "kv_offload": config.inference.kv_cache_offload is not None,
            "kv_offload_cpu_bytes": int(config.inference.kv_cache_offload.cpu_bytes)
            if config.inference.kv_cache_offload
            else 0,
            "use_nccl_broadcast": config.weight_broadcast is not None and config.weight_broadcast.type == "nccl",
            "ranks_filter": ",".join(map(str, config.trainer.log.ranks_filter)),
        }

    return {
        "is_disaggregated": False,
        "config_dir": config_dir,  # TODO: should prob have each subconfig path separately
        "output_dir": config.output_dir,
        "orchestrator_output_dir": config.orchestrator.output_dir,
        "num_train_nodes": deployment.num_train_nodes,
        "num_infer_nodes": deployment.total_infer_nodes,
        "nodes_per_infer_replica": deployment.num_infer_nodes,
        "num_infer_replicas": deployment.num_infer_replicas,
        "gpus_per_node": deployment.gpus_per_node,
        "router_port": getattr(config.inference.deployment, "router_port", 8000) if config.inference else 8000,
        "backend_port": getattr(config.inference.deployment, "backend_port", 8100) if config.inference else 8100,
        "inference_tp": config.inference.parallel.tp if config.inference else 1,
        "inference_enable_expert_parallel": config.inference.enable_expert_parallel if config.inference else False,
        "inference_data_parallel_rpc_port": config.inference.data_parallel_rpc_port if config.inference else 29600,
        "dp_per_node": (config.deployment.gpus_per_node // config.inference.parallel.tp) if config.inference else 1,
        "kv_offload": config.inference is not None and config.inference.kv_cache_offload is not None,
        "use_nccl_broadcast": config.weight_broadcast is not None and config.weight_broadcast.type == "nccl",
        "ranks_filter": ",".join(map(str, config.trainer.log.ranks_filter)),
    }


def write_gpu_layout_script(config: RLConfig, config_dir: Path, script_path: Path) -> None:
    """Write the executable launcher script for gpu_layout deployments."""
    from jinja2 import Environment, FileSystemLoader

    templates_dir = find_package_resource("templates")
    if templates_dir is None:
        raise RuntimeError("prime_rl templates resource not found; cannot render gpu_layout launcher.")
    env = Environment(loader=FileSystemLoader(templates_dir), keep_trailing_newline=True)
    template = env.get_template("gpu_layout_rl.sh.j2")
    script = template.render(**_gpu_layout_template_vars(config, config_dir))
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    script_path.chmod(0o755)


def rl_local(config: RLConfig):
    assert config.deployment.type == "single_node"

    logger = setup_logger(
        config.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"),
        json_logging=config.log.json_logging,
    )

    config_dir = config.output_dir / "configs"
    write_subconfigs(config, config_dir)
    logger.info(f"Wrote subconfigs to {config_dir}")

    if config.dry_run:
        logger.success("Dry run complete. To start an RL run locally, remove --dry-run from your command.")
        return

    # Derive launcher-local GPU IDs from deployment config
    gpu_offset = 0
    num_infer_gpus = config.deployment.num_infer_gpus if config.inference is not None else 0
    infer_local_gpu_ids = list(range(gpu_offset, gpu_offset + num_infer_gpus))
    gpu_offset += num_infer_gpus
    trainer_local_gpu_ids = list(range(gpu_offset, gpu_offset + config.deployment.num_train_gpus))

    total_requested_gpus = num_infer_gpus + config.deployment.num_train_gpus
    physical_gpu_ids = get_physical_gpu_ids()
    if total_requested_gpus > len(physical_gpu_ids):
        raise ValueError(
            f"Requested {total_requested_gpus} GPUs via deployment settings, but only "
            f"{len(physical_gpu_ids)} physical GPU(s) are available: {physical_gpu_ids}"
        )
    physical_gpu_mapping = {local_id: physical_gpu_ids[local_id] for local_id in range(total_requested_gpus)}
    logger.info(f"Using local->physical GPU mapping: {physical_gpu_mapping}")

    infer_gpu_ids = [physical_gpu_mapping[local_gpu_id] for local_gpu_id in infer_local_gpu_ids]
    trainer_gpu_ids = [physical_gpu_mapping[local_gpu_id] for local_gpu_id in trainer_local_gpu_ids]

    start_command = sys.argv
    logger.info("Starting RL run")
    logger.debug(f"RL start command: {' '.join(start_command)}")

    # Build shared W&B env vars for subprocesses. Shared mode is always on for
    # the rl entrypoint — trainer and orchestrator log to a single W&B run.
    # The monitor short-circuits when WANDB_MODE=disabled/offline is also set.
    wandb_shared_env: dict[str, str] = {
        "WANDB_SHARED_MODE": "1",
        "WANDB_SHARED_RUN_ID": os.environ.get("WANDB_SHARED_RUN_ID", uuid.uuid4().hex),
    }

    # Validate client port matches inference server port
    if config.inference is not None and not config.orchestrator.student.client.is_elastic:
        from urllib.parse import urlparse

        base_url = config.orchestrator.student.client.base_url[0]
        parsed = urlparse(base_url)
        client_port = parsed.port
        expected_port = config.inference.server.port
        if client_port != expected_port:
            raise ValueError(
                f"orchestrator.student.client.base_url port ({client_port}) does not match "
                f"inference.server.port ({expected_port}). "
                f"Update the base_url to use port {expected_port} to match the inference server."
            )

    # Prepare paths to communicate with the trainer
    log_dir = get_log_dir(config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Start processes
    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    stop_events: dict[str, Event] = {}

    def sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM, terminating all processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        sys.exit(1)

    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        # Optionally, start inference process
        if config.inference:
            inference_cmd = ["inference", "@", (config_dir / INFERENCE_TOML).as_posix()]
            logger.info(f"Starting inference on GPU(s) {' '.join(map(str, infer_gpu_ids))}")
            logger.debug(f"Inference start command: {' '.join(inference_cmd)}")
            # If we don't log stdout, the server hangs
            with open(log_dir / "inference.log", "w") as log_file:
                inference_process = Popen(
                    inference_cmd,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": ",".join(map(str, infer_gpu_ids)),
                    },
                    stdout=log_file,
                    stderr=log_file,
                )
            processes.append(inference_process)

            # Start monitoring thread
            stop_event = Event()
            stop_events["inference"] = stop_event
            monitor_thread = Thread(
                target=monitor_process,
                args=(inference_process, stop_event, error_queue, "inference"),
                daemon=True,
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)
        else:
            logger.warning(
                "No [inference] block configured - the student inference server will not be started here. "
                "All training modes (rl/opd/sft) require a student inference pool for evals + weight sync; "
                "make sure one is running at orchestrator.student.client.base_url "
                f"({', '.join(config.orchestrator.student.client.base_url)}), otherwise the orchestrator "
                "will hang waiting for it."
            )
            if config.trainer.model.lora is not None:
                logger.warning(
                    "LoRA training is enabled with an external student inference pool. Start that pool with "
                    f"--enable-lora and --max-lora-rank {config.trainer.model.lora.rank}, or adapter loads "
                    "will fail at weight sync."
                )

        if config.orchestrator.teacher:
            logger.info(
                "orchestrator.teacher is configured - the rl entrypoint does not start teacher inference "
                "servers. Make sure your teacher endpoint at "
                f"{', '.join(config.orchestrator.teacher.client.base_url)} is running before the "
                "orchestrator starts, otherwise rollouts will hang."
            )

        # Start orchestrator process
        orchestrator_cmd = [
            "orchestrator",
            "@",
            (config_dir / ORCHESTRATOR_TOML).as_posix(),
        ]
        logger.info("Starting orchestrator process")
        logger.debug(f"Orchestrator start command: {' '.join(orchestrator_cmd)}")
        with open(log_dir / "orchestrator.log", "w") as log_file:
            orchestrator_process = Popen(
                orchestrator_cmd,
                stdout=log_file,
                stderr=log_file,
                env={
                    **os.environ,
                    **wandb_shared_env,
                    "WANDB_SHARED_LABEL": "orchestrator",
                    "LOGURU_FORCE_COLORS": "1",
                    "WANDB_PROGRAM": "uv run rl",
                    "WANDB_ARGS": json.dumps(start_command),
                },
            )
        processes.append(orchestrator_process)

        # Start monitoring thread
        stop_event = Event()
        stop_events["orchestrator"] = stop_event
        monitor_thread = Thread(
            target=monitor_process,
            args=(orchestrator_process, stop_event, error_queue, "orchestrator"),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        # Start training process
        from prime_rl.utils.utils import get_free_port

        trainer_cmd = [
            "torchrun",
            "--role=trainer",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            f"--rdzv-id={uuid.uuid4().hex}",
            # Pipe all logs to file, and only master rank logs to stdout
            f"--log-dir={log_dir / 'trainer' / 'torchrun'}",
            f"--local-ranks-filter={','.join(map(str, config.trainer.log.ranks_filter))}",
            "--redirect=3",
            "--tee=3",
            f"--nproc-per-node={len(trainer_gpu_ids)}",
            "-m",
            "prime_rl.trainer.rl.train",
            "@",
            (config_dir / TRAINER_TOML).as_posix(),
        ]
        logger.info(f"Starting trainer on GPU(s) {' '.join(map(str, trainer_gpu_ids))}")
        logger.debug(f"Training start command: {' '.join(trainer_cmd)}")
        with open(log_dir / "trainer.log", "w") as log_file:
            trainer_process = Popen(
                trainer_cmd,
                env={
                    **os.environ,
                    **wandb_shared_env,
                    "WANDB_SHARED_LABEL": "trainer",
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, trainer_gpu_ids)),
                    "PYTHONUNBUFFERED": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "LOGURU_FORCE_COLORS": "1",
                    "WANDB_PROGRAM": "uv run rl",
                    "WANDB_ARGS": json.dumps(start_command),
                },
                stdout=log_file,
                stderr=log_file,
            )
        processes.append(trainer_process)

        # Start monitoring thread
        stop_event = Event()
        stop_events["trainer"] = stop_event
        monitor_thread = Thread(
            target=monitor_process, args=(trainer_process, stop_event, error_queue, "trainer"), daemon=True
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        # Monitor all processes for failures
        logger.success("Startup complete. Showing orchestrator logs...")

        tail_process = Popen(
            f"tail -F '{log_dir / 'orchestrator.log'}'",
            shell=True,
        )
        processes.append(tail_process)

        # Check for errors from monitor threads
        while not (stop_events["orchestrator"].is_set() and stop_events["trainer"].is_set()):
            if error_queue:
                error = error_queue[0]
                logger.error(f"Error: {error}")
                logger.error("Terminating all processes...")
                cleanup_threads(monitor_threads)
                cleanup_processes(processes)
                sys.exit(1)

            # Small delay to avoid busy waiting
            time.sleep(1)

        # Check if any critical process failed
        if orchestrator_process.returncode != 0:
            logger.error(f"Orchestrator failed with exit code {orchestrator_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        if trainer_process.returncode != 0:
            logger.error(f"Trainer failed with exit code {trainer_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        logger.success("RL training finished!")

        # Cleanup threads and processes
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)

    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, terminating all processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        raise


def write_slurm_script(config: RLConfig, config_dir: Path, script_path: Path) -> None:
    """Write the SLURM script to disk."""
    from jinja2 import Environment, FileSystemLoader

    assert config.slurm is not None
    assert config.slurm.template_path is not None

    env = Environment(loader=FileSystemLoader(config.slurm.template_path.parent), keep_trailing_newline=True)
    template = env.get_template(config.slurm.template_path.name)

    if config.deployment.type == "gpu_layout":
        write_gpu_layout_script(config, config_dir, config.output_dir / GPU_LAYOUT_SCRIPT)
        nodelist = config.slurm.nodelist
        if nodelist is None and config.deployment.hosts is not None:
            nodelist = ",".join(config.deployment.hosts)

        template_vars = {
            **config.slurm.template_vars,
            **_gpu_layout_template_vars(config, config_dir),
            "nodelist": nodelist,
        }
        script = template.render(**template_vars)
    elif config.deployment.type == "single_node":
        script = template.render(
            **config.slurm.template_vars,
            config_path=config_dir / RL_TOML,
            output_dir=config.output_dir,
            gpus_per_node=config.deployment.gpus_per_node,
        )
    else:
        # multi_node: topology vars come from the shared helper so the sbatch and
        # in-alloc paths can never drift. `template_vars` supplies sbatch-only header
        # context (job_name, partition, nodelist, ...).
        script = template.render(
            **config.slurm.template_vars,
            **_multi_node_template_vars(config, config_dir),
        )

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)


def rl_slurm(config: RLConfig):
    assert config.slurm is not None

    logger = setup_logger(
        config.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"), json_logging=config.log.json_logging
    )

    config_dir = config.output_dir / "configs"
    log_dir = get_log_dir(config.output_dir)

    if config.deployment.type == "single_node":
        write_config(config, config_dir, exclude={"slurm", "dry_run", "clean_output_dir"})
        logger.info(f"Wrote config to {config_dir / RL_TOML}")

        train_env_names = [env.resolved_name for env in config.orchestrator.train.env]
        eval_env_names = [env.resolved_name for env in config.orchestrator.eval.env] if config.orchestrator.eval else []

        log_message = format_log_message(
            log_dir=log_dir,
            trainer=True,
            orchestrator=True,
            inference=True,
            train_env_names=train_env_names,
            eval_env_names=eval_env_names,
        )
    else:
        write_subconfigs(config, config_dir)
        logger.info(f"Wrote subconfigs to {config_dir}")

        train_env_names = [env.resolved_name for env in config.orchestrator.train.env]
        eval_env_names = [env.resolved_name for env in config.orchestrator.eval.env] if config.orchestrator.eval else []

        if config.deployment.type == "gpu_layout":
            has_infer = True
            num_train_nodes = 1
            num_infer_nodes = 1
        else:
            has_infer = config.deployment.num_infer_nodes > 0
            num_train_nodes = config.deployment.num_train_nodes
            num_infer_nodes = config.deployment.total_infer_nodes if has_infer else 0

        log_message = format_log_message(
            log_dir=log_dir,
            trainer=True,
            orchestrator=has_infer,
            inference=has_infer,
            train_env_names=train_env_names,
            eval_env_names=eval_env_names,
            num_train_nodes=num_train_nodes,
            num_infer_nodes=num_infer_nodes,
        )
        if config.deployment.type == "gpu_layout":
            log_message += f"\n   All servers:     tail -F {log_dir}/inference/server_*.log"

    script_path = config.output_dir / RL_SBATCH
    write_slurm_script(config, config_dir, script_path)
    logger.info(f"Wrote SLURM script to {script_path}")

    if config.dry_run:
        logger.success(f"Dry run complete. To submit manually:\n\n  sbatch {script_path}\n\n{log_message}")
        return

    logger.info(f"Submitting: sbatch {script_path}")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr.strip()}")
        sys.exit(1)

    logger.success(f"{result.stdout.strip()}\n\n{log_message}")


def _select_gpu_layout_hosts(deployment: GpuLayoutDeploymentConfig) -> list[str]:
    """Select hosts for gpu_layout: explicit hosts if configured, else the exact Slurm allocation."""
    if deployment.hosts is not None:
        return deployment.hosts

    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not nodelist:
        raise RuntimeError("gpu_layout without [slurm] must run inside a Slurm allocation with SLURM_JOB_NODELIST set.")

    result = subprocess.run(["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to resolve Slurm hosts from {nodelist}: {result.stderr.strip()}")

    hosts = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(hosts) < deployment.num_nodes:
        raise RuntimeError(
            f"gpu_layout needs {deployment.num_nodes} hosts, but allocation exposes only {len(hosts)}: {hosts}."
        )
    if len(hosts) > deployment.num_nodes:
        raise RuntimeError(
            f"gpu_layout needs {deployment.num_nodes} hosts, but allocation exposes {len(hosts)}: {hosts}. "
            "Set deployment.hosts explicitly to select the intended lane."
        )
    return hosts


def _gpu_layout_step_resources(host: str) -> tuple[int, int]:
    """Derive per-step CPU and memory requests from physical node resources minus a margin."""
    result = subprocess.run(["scontrol", "show", "node", host, "-o"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to query node resources for {host}: {result.stderr.strip()}")

    cpu_tot: int | None = None
    real_mem_mb: int | None = None
    for token in result.stdout.split():
        if token.startswith("CPUTot="):
            cpu_tot = int(token.split("=", 1)[1])
        elif token.startswith("RealMemory="):
            real_mem_mb = int(token.split("=", 1)[1])
    if cpu_tot is None or real_mem_mb is None:
        raise RuntimeError(f"Could not parse CPUTot/RealMemory from scontrol for {host}: {result.stdout.strip()}")

    cpus_per_task = cpu_tot - _GPU_LAYOUT_CPU_MARGIN
    mem_mb = real_mem_mb - _GPU_LAYOUT_MEM_MARGIN_MB
    if cpus_per_task < 1 or mem_mb < 1024:
        raise RuntimeError(f"Node {host} too small for gpu_layout: CPUTot={cpu_tot} RealMemory={real_mem_mb}MB")
    return cpus_per_task, mem_mb


def rl_gpu_layout_allocation(config: RLConfig):
    """Launch gpu_layout RL inside an existing Slurm allocation via a single srun step."""
    assert config.deployment.type == "gpu_layout"

    logger = setup_logger(
        config.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"),
        json_logging=config.log.json_logging,
    )

    config_dir = config.output_dir / "configs"
    write_subconfigs(config, config_dir)
    logger.info(f"Wrote subconfigs to {config_dir}")

    script_path = config.output_dir / GPU_LAYOUT_SCRIPT
    write_gpu_layout_script(config, config_dir, script_path)

    hosts = _select_gpu_layout_hosts(config.deployment)
    slurm_job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if not slurm_job_id:
        raise RuntimeError("gpu_layout allocation launcher requires SLURM_JOB_ID/SLURM_JOBID for srun --jobid.")

    cpus_per_task, mem_mb = _gpu_layout_step_resources(hosts[0])
    srun_cmd = [
        "srun",
        f"--jobid={slurm_job_id}",
        "--overlap",
        "--exact",
        f"--nodes={config.deployment.num_nodes}",
        "--ntasks-per-node=1",
        f"--cpus-per-task={cpus_per_task}",
        f"--mem={mem_mb}M",
        f"--gres=gpu:{config.deployment.gpus_per_node}",
        "--kill-on-bad-exit=1",
        f"--nodelist={','.join(hosts)}",
        "bash",
        script_path.as_posix(),
    ]

    log_dir = get_log_dir(config.output_dir)
    logger.success(
        f"gpu_layout srun:\n\n  {' '.join(srun_cmd)}\n\nhosts={hosts}"
        f"\n\n  tail -F {log_dir}/trainer.log"
        f"\n  tail -F {log_dir}/inference/server_*.log"
    )
    if config.dry_run:
        return

    env = {**os.environ, "GPU_LAYOUT_HOSTS": " ".join(hosts)}
    result = subprocess.run(srun_cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"gpu_layout srun failed with exit code {result.returncode}")


def write_multi_node_script(config: RLConfig, config_dir: Path, script_path: Path) -> None:
    """Render the multi_node orchestrator script for the in-allocation path.

    Same template as the sbatch path (``multi_node_rl.sbatch.j2``); the ``#SBATCH``
    header lines are inert comments when the script is run directly with ``bash``. The
    per-lane ``LANE_*`` parameters are NOT baked in here — the template reads them from
    the environment with fallbacks, and the launcher exports them (see lane-contract.md)."""
    from jinja2 import Environment, FileSystemLoader

    assert config.deployment.type == "multi_node"
    templates_dir = find_package_resource("templates")
    if templates_dir is None:
        raise RuntimeError("prime_rl templates resource not found; cannot render multi_node launcher.")
    env = Environment(loader=FileSystemLoader(templates_dir), keep_trailing_newline=True)
    template = env.get_template(MULTI_NODE_TEMPLATE)
    # No [slurm] block in-alloc: `project_dir` comes from the cwd (mirrors gpu_layout).
    # Other header vars (job_name, partition, ...) are only referenced by the inert
    # #SBATCH comments, so they render empty harmlessly.
    script = template.render(project_dir=Path.cwd().resolve(), **_multi_node_template_vars(config, config_dir))
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    script_path.chmod(0o755)


def _select_multi_node_hosts(deployment, num_hosts: int) -> list[str]:
    """Resolve this lane's hosts: explicit ``deployment.hosts`` if set (the slice),
    else every host of the whole Slurm allocation (the whole alloc = one lane)."""
    if deployment.hosts is not None:
        hosts = deployment.hosts
    else:
        nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
        if not nodelist:
            raise RuntimeError(
                "multi_node without [slurm] must run inside a Slurm allocation with SLURM_JOB_NODELIST set."
            )
        result = subprocess.run(["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to resolve Slurm hosts from {nodelist}: {result.stderr.strip()}")
        hosts = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    if len(hosts) != num_hosts:
        raise RuntimeError(
            f"multi_node lane needs {num_hosts} hosts (= num_train_nodes + total_infer_nodes), "
            f"but resolved {len(hosts)}: {hosts}. "
            "Set deployment.hosts explicitly to carve the intended lane out of the allocation."
        )
    return hosts


def rl_multinode_allocation(config: RLConfig):
    """Launch a multi_node RL lane inside an existing Slurm allocation.

    The rendered ``multi_node_rl.sbatch.j2`` is a head-node orchestrator that fans out
    via its OWN internal ``srun --exact -w <slice>`` calls. So we run it DIRECTLY with
    ``bash`` on the current node (NO outer srun — nesting would deadlock step creation
    and re-run the run-once preamble per node), mirroring how the sbatch path runs it as
    the batch step. We export the lane parameters; the template's internal sruns are
    lane-aware and consume them (``LANE_CPUS_PER_TASK``/``--gpus-per-node`` ride the WORK
    srun, not any launcher srun). Disjoint slices of the same allocation coexist as
    independent lanes (see lane-contract.md)."""
    assert config.deployment.type == "multi_node"

    logger = setup_logger(
        config.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"),
        json_logging=config.log.json_logging,
    )

    slurm_job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if not slurm_job_id:
        raise RuntimeError(
            "multi_node in-allocation launcher requires SLURM_JOB_ID/SLURM_JOBID. "
            "Run this inside a held Slurm allocation, or add a [slurm] block to submit via sbatch."
        )

    deployment = config.deployment
    lane_nnodes = deployment.num_train_nodes + deployment.total_infer_nodes
    hosts = _select_multi_node_hosts(deployment, lane_nnodes)

    port_base = deployment.port_base if deployment.port_base is not None else _DEFAULT_PORT_BASE
    lane_tag = deployment.lane_tag if deployment.lane_tag is not None else slurm_job_id

    # Namespace the output dir per lane so concurrent lanes don't clobber each other's
    # script / subconfigs / logs / rollouts / checkpoints / weight broadcasts. Only when an
    # explicit lane_tag distinguishes the lane; the bare-$SLURM_JOB_ID fallback (single lane
    # = whole alloc) keeps the flat layout.
    if deployment.lane_tag is not None:
        config.output_dir = config.output_dir / lane_tag
        # propagate_shared_fields set the sub-config output_dirs at construction (trainer =
        # output_dir, orchestrator = output_dir / "run_default") BEFORE this suffix runs, so
        # re-home them under the lane dir too — otherwise the trainer (rollouts/checkpoints/
        # broadcasts via multi_run_manager) and orchestrator operate on the un-namespaced
        # base and concurrent lanes race on it. Preserves trainer == orchestrator.parent.
        config.trainer.output_dir = config.output_dir
        config.orchestrator.output_dir = config.output_dir / config.orchestrator.output_dir.name

    config_dir = config.output_dir / "configs"
    write_subconfigs(config, config_dir)
    logger.info(f"Wrote subconfigs to {config_dir}")

    script_path = config.output_dir / MULTI_NODE_SCRIPT
    write_multi_node_script(config, config_dir, script_path)

    # Per-task resources for the template's internal WORK sruns: --exact otherwise
    # clamps each task to 1 CPU and starves vLLM/torch. Derive from the node (whole
    # node minus a margin) via the same scontrol probe gpu_layout uses.
    cpus_per_task, mem_mb = _gpu_layout_step_resources(hosts[0])

    log_dir = get_log_dir(config.output_dir)
    logger.success(
        f"multi_node lane (direct bash, no outer srun):\n\n  bash {script_path.as_posix()}\n\n"
        f"hosts={hosts} LANE_NNODES={lane_nnodes} PORT_BASE={port_base} LANE_TAG={lane_tag} "
        f"LANE_CPUS_PER_TASK={cpus_per_task} LANE_MEM_MB={mem_mb}M"
        f"\n\n  tail -F {log_dir}/trainer.log"
        f"\n  tail -F {log_dir}/inference.log"
    )
    if config.dry_run:
        return

    # LANE_* delivery = env-export (pinned decision #1). The template reads these with
    # fallbacks and threads them onto its internal sruns (--export=ALL propagates to the
    # compute nodes), so the SAME template also serves the legacy full-allocation sbatch
    # path when LANE_* are unset.
    env = {
        **os.environ,
        "LANE_HOSTS": " ".join(hosts),
        "PORT_BASE": str(port_base),
        "LANE_TAG": str(lane_tag),
        "LANE_CPUS_PER_TASK": str(cpus_per_task),
        "LANE_MEM_MB": f"{mem_mb}M",
    }
    result = subprocess.run(["bash", script_path.as_posix()], env=env)
    if result.returncode != 0:
        raise RuntimeError(f"multi_node lane failed with exit code {result.returncode}")


def rl(config: RLConfig):
    resuming = config.ckpt is not None and config.ckpt.resume_step is not None
    clean = config.clean_output_dir and not os.environ.get("NEVER_CLEAN_OUTPUT_DIR")
    ckpt_output_dir = config.ckpt.output_dir if config.ckpt else None
    validate_output_dir(config.output_dir, resuming=resuming, clean=clean, ckpt_output_dir=ckpt_output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_output_dir is not None:
        ckpt_output_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale rollouts and broadcasts. When resuming, anything past the resume
    # step is stale. When training from scratch, every existing step directory is
    # stale — without this, a fresh run in a dirty output_dir would pick up rollouts
    # from a previous run and the orchestrator would see a negative async level.
    resume_step: int | None = None
    if resuming:
        resume_step = config.ckpt.resume_step
        if resume_step == -1:
            ckpt_base = ckpt_output_dir if ckpt_output_dir is not None else config.output_dir
            resume_step = resolve_latest_ckpt_step(get_ckpt_dir(ckpt_base))

    if resume_step is not None:
        get_logger().info(f"Resuming from step {resume_step}, cleaning future rollouts and broadcasts")
        clean_future_steps(config.output_dir, resume_step)
    else:
        get_logger().info("Training from scratch, cleaning any stale rollouts and broadcasts")
        clean_future_steps(config.output_dir, -1)

    if not config.dry_run:
        from prime_rl.trainer.model import pre_download_model

        pre_download_model(config.trainer.model.name)

    # Launch axis (HOW we start) is decoupled from the topology axis (deployment.type,
    # i.e. WHAT we place). Presence of a [slurm] block = "submit via sbatch" (kept
    # option). Absence + an in-allocation context (SLURM_JOB_ID set) = "run in THIS
    # held allocation": gpu_layout and multi_node each get their own in-alloc srun path;
    # everything else runs locally.
    in_allocation = bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID"))
    if config.slurm is not None:
        rl_slurm(config)
    elif config.deployment.type == "gpu_layout":
        rl_gpu_layout_allocation(config)
    elif config.deployment.type == "multi_node" and in_allocation:
        rl_multinode_allocation(config)
    else:
        rl_local(config)


def main():
    set_proc_title("Launcher")
    rl(cli(RLConfig))


if __name__ == "__main__":
    main()
