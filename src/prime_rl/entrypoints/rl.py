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

import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before transitive import
from prime_rl.configs.rl import RLConfig
from prime_rl.trainer.model import pre_download_model
from prime_rl.utils.config import cli
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import (
    clean_future_steps,
    format_log_message,
    get_ckpt_dir,
    resolve_latest_ckpt_step,
    validate_output_dir,
)
from prime_rl.utils.process import cleanup_processes, cleanup_threads, monitor_process, set_proc_title
from prime_rl.utils.utils import (
    get_free_port,
    get_log_dir,
)

RL_TOML = "rl.toml"
RL_SBATCH = "rl.sbatch"

TRAINER_TOML = "trainer.toml"
ORCHESTRATOR_TOML = "orchestrator.toml"


def inference_toml_name(tag: str) -> str:
    return f"inference_{tag}.toml"


def inference_log_name(tag: str) -> str:
    return f"inference_{tag}.log"


# Launcher-only fields stripped from the per-deployment vLLM TOML — the
# inference entrypoint validates against InferenceConfig but doesn't know about
# multi-deployment placement, SLURM, or RL-only output_dir overrides.
_INFERENCE_LAUNCHER_FIELDS = {"deployment", "slurm", "output_dir", "dry_run", "tag", "gpu_ids"}


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

    for entry in config.inference:
        with open(output_dir / inference_toml_name(entry.tag), "wb") as f:
            tomli_w.dump(entry.model_dump(exclude=_INFERENCE_LAUNCHER_FIELDS, exclude_none=True, mode="json"), f)


def _resolve_inference_gpu_placement(config: RLConfig) -> dict[str, list[int]]:
    """Compute local GPU ids for each [[inference]] entry and the trainer.

    Entries with explicit ``gpu_ids`` use them verbatim (allowing overlapped
    placement when multiple tags share an id). Entries without ``gpu_ids`` fall
    back to sequential allocation from ``deployment.num_infer_gpus``, preserving
    the legacy single-deployment behavior. The trainer takes the first
    ``num_train_gpus`` ids not used by any inference entry.

    Returns a dict keyed by tag (plus ``"trainer"``) mapping to local GPU ids.
    """
    assert config.deployment.type == "single_node"

    explicit: dict[str, list[int]] = {}
    implicit_tags: list[str] = []
    used: set[int] = set()
    for entry in config.inference:
        if entry.gpu_ids is not None:
            explicit[entry.tag] = list(entry.gpu_ids)
            used.update(entry.gpu_ids)
        else:
            implicit_tags.append(entry.tag)

    # Sequential allocation from the legacy num_infer_gpus pool, starting at 0
    # and skipping ids already claimed by explicit entries. Splitting the pool
    # across multiple implicit entries isn't supported (ambiguous); each
    # implicit entry would need to declare its own GPU budget. Today this only
    # fires for legacy single-block configs (one implicit entry), so the
    # behavior is the same as before #2554.
    if implicit_tags:
        if len(implicit_tags) > 1:
            raise ValueError(
                "Multiple [[inference]] entries left gpu_ids unset; the legacy sequential "
                "allocation only handles one such entry. Pin gpu_ids on the others."
            )
        tag = implicit_tags[0]
        pool: list[int] = []
        cursor = 0
        while len(pool) < config.deployment.num_infer_gpus:
            if cursor not in used:
                pool.append(cursor)
            cursor += 1
        explicit[tag] = pool

    trainer_ids = [g for g in range(config.deployment.gpus_per_node) if g not in set().union(*explicit.values())]
    trainer_ids = trainer_ids[: config.deployment.num_train_gpus]

    return {**explicit, "trainer": trainer_ids}


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

    placement = _resolve_inference_gpu_placement(config)
    trainer_local_gpu_ids = placement["trainer"]
    inference_local_gpu_ids: dict[str, list[int]] = {tag: ids for tag, ids in placement.items() if tag != "trainer"}

    requested_locals = set(trainer_local_gpu_ids)
    for ids in inference_local_gpu_ids.values():
        requested_locals.update(ids)
    physical_gpu_ids = get_physical_gpu_ids()
    max_requested = max(requested_locals, default=-1)
    if max_requested >= len(physical_gpu_ids):
        raise ValueError(
            f"Requested local GPU id {max_requested}, but only {len(physical_gpu_ids)} physical "
            f"GPU(s) are available: {physical_gpu_ids}"
        )
    physical_gpu_mapping = {local_id: physical_gpu_ids[local_id] for local_id in requested_locals}
    logger.info(f"Using local->physical GPU mapping: {physical_gpu_mapping}")

    inference_gpu_ids: dict[str, list[int]] = {
        tag: [physical_gpu_mapping[i] for i in ids] for tag, ids in inference_local_gpu_ids.items()
    }
    trainer_gpu_ids = [physical_gpu_mapping[i] for i in trainer_local_gpu_ids]

    start_command = sys.argv
    logger.info("Starting RL run")
    logger.debug(f"RL start command: {' '.join(start_command)}")

    # Build shared W&B env vars for subprocesses
    wandb_shared_env: dict[str, str] = {}
    if config.wandb and config.wandb.shared:
        wandb_shared_env["WANDB_SHARED_MODE"] = "1"
        wandb_shared_env["WANDB_SHARED_RUN_ID"] = os.environ.get("WANDB_SHARED_RUN_ID", uuid.uuid4().hex)

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
        if not config.inference:
            logger.warning(
                "No [[inference]] entries configured - no inference server will be started here. "
                "All training modes (rl/opd/sft) require a student inference pool for evals + weight sync; "
                "make sure one is running at orchestrator.student.client.base_url "
                f"({', '.join(config.orchestrator.student.client.base_url)}), otherwise the orchestrator "
                "will hang waiting for it."
            )

        # Start one inference subprocess per [[inference]] entry. Each one gets
        # its own log file, monitor thread, and CUDA_VISIBLE_DEVICES; multiple
        # entries can share GPU ids (overlapped layout) as long as their
        # gpu_memory_utilization budgets fit together.
        for entry in config.inference:
            tag = entry.tag
            gpu_ids = inference_gpu_ids[tag]
            label = f"inference[{tag}]"
            inference_cmd = ["inference", "@", (config_dir / inference_toml_name(tag)).as_posix()]
            logger.info(f"Starting {label} on GPU(s) {' '.join(map(str, gpu_ids))}")
            logger.debug(f"{label} start command: {' '.join(inference_cmd)}")
            with open(log_dir / inference_log_name(tag), "w") as log_file:
                inference_process = Popen(
                    inference_cmd,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)),
                    },
                    stdout=log_file,
                    stderr=log_file,
                )
            processes.append(inference_process)

            stop_event = Event()
            stop_key = f"inference_{tag}"
            stop_events[stop_key] = stop_event
            monitor_thread = Thread(
                target=monitor_process,
                args=(inference_process, stop_event, error_queue, label),
                daemon=True,
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)

        if config.orchestrator.teacher and config.teacher_inference is None:
            logger.warning(
                'orchestrator.teacher is configured but no [[inference]] entry tagged "teacher" was found. '
                "Make sure an external teacher server is running at "
                f"{', '.join(config.orchestrator.teacher.client.base_url)}; otherwise the orchestrator will hang."
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
    """Write the SLURM script to disk.

    The multi-node template iterates over ``inference_deployments`` (one entry per
    tagged [[inference]] block) when rendering srun steps. Today only the
    student deployment lands on multi-node — additional tags are gated by
    RLConfig.validate_no_teacher_in_multinode — so the list collapses to a
    single entry when present. The data is exposed as a list anyway so the
    template doesn't need to special-case future per-tag fan-out.
    """
    from jinja2 import Environment, FileSystemLoader

    assert config.slurm is not None
    assert config.slurm.template_path is not None

    env = Environment(loader=FileSystemLoader(config.slurm.template_path.parent), keep_trailing_newline=True)
    template = env.get_template(config.slurm.template_path.name)

    student_inf = config.student_inference
    inference_deployments = [
        {
            "tag": entry.tag,
            "server_port": entry.server.port,
            "tp": entry.parallel.tp,
            "dp": entry.parallel.dp,
            "data_parallel_size_local": entry.data_parallel_size_local,
            "data_parallel_rpc_port": entry.data_parallel_rpc_port,
            "api_server_count": entry.api_server_count,
            "config_path": (config_dir / inference_toml_name(entry.tag)).as_posix(),
            "log_name": inference_log_name(entry.tag),
        }
        for entry in config.inference
    ]

    if config.deployment.type == "single_node":
        script = template.render(
            **config.slurm.template_vars,
            config_path=config_dir / RL_TOML,
            output_dir=config.output_dir,
            gpus_per_node=config.deployment.gpus_per_node,
            inference_deployments=inference_deployments,
        )
    elif student_inf is not None and student_inf.deployment.type == "disaggregated":
        infer_deploy = student_inf.deployment

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
            num_prefill_replicas=infer_deploy.num_prefill_replicas,
            num_decode_replicas=infer_deploy.num_decode_replicas,
            gpus_per_node=config.deployment.gpus_per_node,
            router_port=infer_deploy.router_port,
            prefill_port=infer_deploy.prefill_port,
            decode_port=infer_deploy.decode_port,
            inference_tp=student_inf.parallel.tp,
            inference_data_parallel_rpc_port=student_inf.data_parallel_rpc_port,
            use_deep_gemm=student_inf.use_deep_gemm,
            prefill_env_overrides=infer_deploy.prefill_env_overrides,
            decode_env_overrides=infer_deploy.decode_env_overrides,
            dp_per_node=config.deployment.gpus_per_node // student_inf.parallel.tp,
            kv_offload=student_inf.kv_cache_offload is not None,
            kv_offload_cpu_bytes=int(student_inf.kv_cache_offload.cpu_bytes) if student_inf.kv_cache_offload else 0,
            use_nccl_broadcast=config.weight_broadcast is not None and config.weight_broadcast.type == "nccl",
            wandb_shared=config.wandb is not None and config.wandb.shared,
            ranks_filter=",".join(map(str, config.trainer.log.ranks_filter)),
            inference_deployments=inference_deployments,
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
            gpus_per_node=config.deployment.gpus_per_node,
            router_port=getattr(student_inf.deployment, "router_port", 8000) if student_inf else 8000,
            backend_port=getattr(student_inf.deployment, "backend_port", 8100) if student_inf else 8100,
            inference_tp=student_inf.parallel.tp if student_inf else 1,
            inference_enable_expert_parallel=student_inf.enable_expert_parallel if student_inf else False,
            inference_data_parallel_rpc_port=student_inf.data_parallel_rpc_port if student_inf else 29600,
            dp_per_node=(config.deployment.gpus_per_node // student_inf.parallel.tp) if student_inf else 1,
            kv_offload=student_inf is not None and student_inf.kv_cache_offload is not None,
            use_nccl_broadcast=config.weight_broadcast is not None and config.weight_broadcast.type == "nccl",
            wandb_shared=config.wandb is not None and config.wandb.shared,
            ranks_filter=",".join(map(str, config.trainer.log.ranks_filter)),
            inference_deployments=inference_deployments,
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

        has_infer = config.deployment.num_infer_nodes > 0
        log_message = format_log_message(
            log_dir=log_dir,
            trainer=True,
            orchestrator=has_infer,
            inference=has_infer,
            train_env_names=train_env_names,
            eval_env_names=eval_env_names,
            num_train_nodes=config.deployment.num_train_nodes,
            num_infer_nodes=config.deployment.total_infer_nodes if has_infer else 0,
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
        pre_download_model(config.trainer.model.name)

    if config.slurm is not None:
        rl_slurm(config)
    else:
        rl_local(config)


def main():
    set_proc_title("Launcher")
    rl(cli(RLConfig))


if __name__ == "__main__":
    main()
