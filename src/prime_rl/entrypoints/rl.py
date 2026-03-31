import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread

import pynvml
import tomli_w

from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import validate_output_dir
from prime_rl.utils.process import cleanup_processes, cleanup_threads, monitor_process
from prime_rl.utils.utils import (
    get_free_port,
    get_log_dir,
)

RL_TOML = "rl.toml"
RL_SBATCH = "rl.sbatch"

TRAINER_TOML = "trainer.toml"
ORCHESTRATOR_TOML = "orchestrator.toml"
INFERENCE_TOML = "inference.toml"
TEACHER_INFERENCE_TOML = "teacher_inference.toml"


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

    teacher_inference = getattr(config, "teacher_inference", None)
    if teacher_inference is not None:
        with open(output_dir / TEACHER_INFERENCE_TOML, "wb") as f:
            tomli_w.dump(teacher_inference.model_dump(exclude_none=True, mode="json"), f)


def check_gpus_available(gpu_ids: list[int]) -> None:
    """Raise error if there are existing processes on the specified GPUs."""
    pynvml.nvmlInit()

    occupied = []
    for gpu_id in gpu_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if processes:
            pids = [p.pid for p in processes]
            occupied.append((gpu_id, pids))

    if occupied:
        msg = "Existing processes found on GPUs:\n"
        for gpu_id, pids in occupied:
            msg += f"  GPU {gpu_id}: PIDs {pids}\n"
        msg += "Kill these processes or use different GPUs."
        raise RuntimeError(msg)


def rl_local(config: RLConfig):
    assert config.deployment.type == "single_node"

    logger = setup_logger(
        config.log.level or "info",
        log_file=config.output_dir / "logs" / "rl.log" if config.log.file else None,
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
    gpu_offset += config.deployment.num_train_gpus
    num_teacher_gpus = config.deployment.num_teacher_gpus or 0
    teacher_local_gpu_ids = list(range(gpu_offset, gpu_offset + num_teacher_gpus)) if num_teacher_gpus > 0 else []

    total_requested_gpus = num_infer_gpus + config.deployment.num_train_gpus + num_teacher_gpus
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
    teacher_gpu_ids = [physical_gpu_mapping[local_gpu_id] for local_gpu_id in teacher_local_gpu_ids]

    start_command = sys.argv
    logger.info("Starting RL run")
    logger.debug(f"RL start command: {' '.join(start_command)}")

    # Build shared W&B env vars for subprocesses
    wandb_shared_env: dict[str, str] = {}
    if config.wandb and config.wandb.shared:
        wandb_shared_env["WANDB_SHARED_MODE"] = "1"
        wandb_shared_env["WANDB_SHARED_RUN_ID"] = uuid.uuid4().hex

    # Check for existing processes on GPUs
    all_gpu_ids = list(set(infer_gpu_ids + trainer_gpu_ids + teacher_gpu_ids))
    check_gpus_available(all_gpu_ids)

    # Validate client port matches inference server port
    if config.inference is not None and not config.orchestrator.client.is_elastic:
        from urllib.parse import urlparse

        base_url = config.orchestrator.client.base_url[0]
        parsed = urlparse(base_url)
        client_port = parsed.port
        expected_port = config.inference.server.port
        if client_port != expected_port:
            raise ValueError(
                f"orchestrator.client.base_url port ({client_port}) does not match "
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

    try:
        # Optionally, start inference process
        if config.inference:
            inference_cmd = ["uv", "run", "inference", "@", (config_dir / INFERENCE_TOML).as_posix()]
            logger.info(f"Starting inference on GPU(s) {' '.join(map(str, infer_gpu_ids))}")
            logger.debug(f"Inference start command: {' '.join(inference_cmd)}")
            # If we don't log stdout, the server hangs
            with open(log_dir / "inference.stdout", "w") as log_file:
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
            if config.orchestrator.teacher_rollout_model is None:
                logger.warning(
                    "No inference config specified, skipping starting inference server. Make sure your inference server is running."
                )
            else:
                logger.info(
                    "No inference config specified, using orchestrator.teacher_rollout_model for rollout generation."
                )

        # Optionally, start teacher inference process
        if config.teacher_inference:
            if not teacher_gpu_ids:
                raise ValueError(
                    "teacher_inference is configured but deployment.num_teacher_gpus is not set. "
                    "Either set deployment.num_teacher_gpus to start a teacher inference server, "
                    "or omit teacher_inference and configure orchestrator.teacher_model to use an existing server."
                )

            teacher_inference_cmd = ["uv", "run", "inference", "@", (config_dir / TEACHER_INFERENCE_TOML).as_posix()]
            logger.info(f"Starting teacher inference process on GPU(s) {' '.join(map(str, teacher_gpu_ids))}")
            logger.debug(f"Teacher inference start command: {' '.join(teacher_inference_cmd)}")
            with open(log_dir / "teacher_inference.stdout", "w") as log_file:
                teacher_inference_process = Popen(
                    teacher_inference_cmd,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": ",".join(map(str, teacher_gpu_ids)),
                    },
                    stdout=log_file,
                    stderr=log_file,
                )
            processes.append(teacher_inference_process)

            # Start monitoring thread
            stop_event = Event()
            stop_events["teacher_inference"] = stop_event
            monitor_thread = Thread(
                target=monitor_process,
                args=(teacher_inference_process, stop_event, error_queue, "teacher_inference"),
                daemon=True,
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)
        elif (
            config.trainer.loss.type == "default" and config.trainer.loss.teacher_tau > 0
        ) or config.orchestrator.teacher_model:
            logger.warning(
                "No teacher_inference config specified, skipping starting teacher inference server. "
                "Is your teacher inference server running? Make sure orchestrator.teacher_model is configured."
            )

        # Start orchestrator process
        orchestrator_cmd = [
            "uv",
            "run",
            "orchestrator",
            "@",
            (config_dir / ORCHESTRATOR_TOML).as_posix(),
        ]
        logger.info("Starting orchestrator process")
        logger.debug(f"Orchestrator start command: {' '.join(orchestrator_cmd)}")
        with open(log_dir / "orchestrator.stdout", "w") as log_file:
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
        trainer_cmd = [
            "uv",
            "run",
            "env",
            "PYTHONUNBUFFERED=1",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "torchrun",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            f"--rdzv-id={uuid.uuid4().hex}",
            # Pipe all logs to file, and only master rank logs to stdout
            f"--log-dir={config.output_dir / 'torchrun'}",
            "--local-ranks-filter=0",
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
        with open(log_dir / "trainer.stdout", "w") as log_file:
            trainer_process = Popen(
                trainer_cmd,
                env={
                    **os.environ,
                    **wandb_shared_env,
                    "WANDB_SHARED_LABEL": "trainer",
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, trainer_gpu_ids)),
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
        logger.success("Startup complete. Showing trainer logs...")

        tail_process = Popen(["tail", "-F", log_dir / "trainer.stdout"])
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

    if config.deployment.type == "single_node":
        script = template.render(
            **config.slurm.template_vars,
            config_path=config_dir / RL_TOML,
            output_dir=config.output_dir,
            gpus_per_node=config.deployment.gpus_per_node,
        )
    elif config.inference is not None and config.inference.deployment.type == "disaggregated":
        infer_deploy = config.inference.deployment

        script = template.render(
            **config.slurm.template_vars,
            is_disaggregated=True,
            config_dir=config_dir,
            output_dir=config.output_dir,
            orchestrator_output_dir=config.orchestrator.output_dir,
            num_train_nodes=config.deployment.num_train_nodes,
            num_infer_nodes=infer_deploy.num_nodes * config.deployment.num_infer_replicas,
            nodes_per_infer_replica=infer_deploy.num_nodes,
            num_infer_replicas=config.deployment.num_infer_replicas,
            num_prefill_nodes=infer_deploy.num_prefill_nodes,
            num_decode_nodes=infer_deploy.num_decode_nodes,
            gpus_per_node=config.deployment.gpus_per_node,
            router_port=infer_deploy.router_port,
            prefill_port=infer_deploy.prefill_port,
            decode_port=infer_deploy.decode_port,
            inference_tp=config.inference.parallel.tp,
            inference_data_parallel_rpc_port=config.inference.data_parallel_rpc_port,
            use_deep_gemm=config.inference.use_deep_gemm,
            use_nccl_broadcast=config.weight_broadcast is not None and config.weight_broadcast.type == "nccl",
            wandb_shared=config.wandb is not None and config.wandb.shared,
        )
    else:
        script = template.render(
            **config.slurm.template_vars,
            is_disaggregated=False,
            config_dir=config_dir,  # TODO: should prob have each subconfig path separately
            output_dir=config.output_dir,
            orchestrator_output_dir=config.orchestrator.output_dir,
            num_train_nodes=config.deployment.num_train_nodes,
            num_infer_nodes=config.deployment.total_infer_nodes,
            nodes_per_infer_replica=config.deployment.num_infer_nodes,
            num_infer_replicas=config.deployment.num_infer_replicas,
            num_teacher_nodes=config.deployment.num_teacher_nodes,
            gpus_per_node=config.deployment.gpus_per_node,
            router_port=getattr(config.inference.deployment, "router_port", 8000) if config.inference else 8000,
            backend_port=getattr(config.inference.deployment, "backend_port", 8100) if config.inference else 8100,
            inference_tp=config.inference.parallel.tp if config.inference else 1,
            inference_enable_expert_parallel=config.inference.enable_expert_parallel if config.inference else False,
            inference_data_parallel_rpc_port=config.inference.data_parallel_rpc_port if config.inference else 29600,
            use_nccl_broadcast=config.weight_broadcast is not None and config.weight_broadcast.type == "nccl",
            wandb_shared=config.wandb is not None and config.wandb.shared,
        )

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)


def format_log_message(
    trainer_log: str,
    orchestrator_log: str | None,
    inference_log: str | None,
    env_log_dir: Path,
    train_env_names: list[str],
    eval_env_names: list[str],
) -> str:
    col = 18
    i1 = " " * 2
    i2 = " " * 3
    i3 = " " * 4
    max_name = col - 4

    log_lines = [f"{i1}{'Trainer:':<{col}}tail -F {trainer_log}"]
    if orchestrator_log:
        log_lines.append(f"{i1}{'Orchestrator:':<{col}}tail -F {orchestrator_log}")
    if inference_log:
        log_lines.append(f"{i1}{'Inference:':<{col}}tail -F {inference_log}")
    log_lines.append(f"{i1}{'Envs:':<{col}}tail -F {env_log_dir}/*/*/*.log")
    log_lines.append(f"{i2}{'Train:':<{col - 1}}tail -F {env_log_dir}/train/*/*.log")
    for name in train_env_names:
        short = name if len(name) <= max_name else name[: max_name - 3] + "..."
        log_lines.append(f"{i3}{f'{short}:':<{col - 2}}tail -F {env_log_dir}/train/{name}/*.log")
    if eval_env_names:
        log_lines.append(f"{i2}{'Eval:':<{col - 1}}tail -F {env_log_dir}/eval/*/*.log")
        for name in eval_env_names:
            short = name if len(name) <= max_name else name[: max_name - 3] + "..."
            log_lines.append(f"{i3}{f'{short}:':<{col - 2}}tail -F {env_log_dir}/eval/{name}/*.log")
    return "Logs:\n" + "\n".join(log_lines)


def rl_slurm(config: RLConfig):
    assert config.slurm is not None

    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    config_dir = config.output_dir / "configs"
    if config.deployment.type == "single_node":
        write_config(config, config_dir, exclude={"slurm", "dry_run", "clean_output_dir"})
        logger.info(f"Wrote config to {config_dir / RL_TOML}")

        log_dir = get_log_dir(config.output_dir)
        env_log_dir = get_log_dir(config.output_dir) / "envs"
        train_env_names = [env.resolved_name for env in config.orchestrator.env]
        eval_env_names = [env.resolved_name for env in config.orchestrator.eval.env] if config.orchestrator.eval else []

        log_message = format_log_message(
            trainer_log=f"{log_dir}/trainer.stdout",
            orchestrator_log=f"{log_dir}/orchestrator.stdout",
            inference_log=f"{log_dir}/inference.stdout",
            env_log_dir=env_log_dir,
            train_env_names=train_env_names,
            eval_env_names=eval_env_names,
        )
    else:
        write_subconfigs(config, config_dir)
        logger.info(f"Wrote subconfigs to {config_dir}")

        slurm_log_dir = config.output_dir / "slurm"
        env_log_dir = get_log_dir(config.output_dir) / "envs"
        train_env_names = [env.resolved_name for env in config.orchestrator.env]
        eval_env_names = [env.resolved_name for env in config.orchestrator.eval.env] if config.orchestrator.eval else []

        has_infer = config.deployment.num_infer_nodes > 0
        log_message = format_log_message(
            trainer_log=f"{slurm_log_dir}/latest_train_node_rank_0.log",
            orchestrator_log=f"{slurm_log_dir}/latest_orchestrator.log" if has_infer else None,
            inference_log=f"{slurm_log_dir}/latest_infer_node_rank_0.log" if has_infer else None,
            env_log_dir=env_log_dir,
            train_env_names=train_env_names,
            eval_env_names=eval_env_names,
        )

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


def rl(config: RLConfig):
    resuming = config.ckpt is not None and config.ckpt.resume_step is not None
    clean = config.clean_output_dir and not os.environ.get("NEVER_CLEAN_OUTPUT_DIR")
    ckpt_output_dir = config.ckpt.output_dir if config.ckpt else None
    validate_output_dir(config.output_dir, resuming=resuming, clean=clean, ckpt_output_dir=ckpt_output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_output_dir is not None:
        ckpt_output_dir.mkdir(parents=True, exist_ok=True)

    if config.slurm is not None:
        rl_slurm(config)
    else:
        rl_local(config)


def main():
    rl(cli(RLConfig))


if __name__ == "__main__":
    main()
