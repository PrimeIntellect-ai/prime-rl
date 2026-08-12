import json
import os
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread

import tomli_w

from prime_rl.configs.evaluator import EvaluatorConfig
from prime_rl.configs.orchestrator import EvalSourceConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.shared import BaseModelConfig, LogConfig
from prime_rl.utils.config import cli, to_toml_dict
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import (
    format_log_message,
    get_all_ckpt_steps,
    get_ckpt_dir,
    get_config_dir,
    get_log_dir,
    get_step_path,
    get_weights_dir,
    resolve_latest_ckpt_step,
    validate_output_dir,
)
from prime_rl.utils.process import (
    DEFAULT_COMMON_ENV_VARS,
    DEFAULT_INFERENCE_ENV_VARS,
    DEFAULT_TRAINER_ENV_VARS,
    cleanup_processes,
    cleanup_threads,
    get_physical_gpu_ids,
    monitor_process,
    set_proc_title,
)

SFT_TOML = "sft.toml"
SFT_SBATCH = "sft.sbatch"

INFERENCE_TOML = "inference.toml"
EVALUATOR_TOML = "evaluator.toml"

ENVS_DIR = "envs"


def eval_env_servers(config: SFTConfig) -> list[tuple[EvalSourceConfig, str]]:
    """``(source, address)`` for every launcher-managed eval source. A source with
    ``serve.address`` set is externally managed — the launcher neither writes its
    TOML nor spawns a server for it."""
    if config.eval is None:
        return []
    addresses = config.eval.env_addresses
    return [
        (source, addresses[("eval", source.resolved_name)])
        for source in config.eval.source
        if source.serve.address is None
    ]


def resolve_resume_step(config: SFTConfig) -> int | None:
    if config.ckpt is None or config.ckpt.resume_step is None:
        return None
    if config.ckpt.resume_step == -1:
        ckpt_base = config.ckpt.output_dir or config.output_dir
        return resolve_latest_ckpt_step(get_ckpt_dir(ckpt_base))
    return config.ckpt.resume_step


def build_evaluator_config(config: SFTConfig) -> EvaluatorConfig:
    """Derive the evaluator subconfig from the resolved SFT config."""
    assert config.eval is not None
    ckpt_base = (config.ckpt.output_dir if config.ckpt else None) or config.output_dir
    return EvaluatorConfig(
        model=BaseModelConfig(name=config.model.name),
        eval=config.eval,
        weights_dir=get_weights_dir(ckpt_base),
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        resume_step=resolve_resume_step(config),
        log=LogConfig(level=config.log.level, json_logging=config.log.json_logging),
        wandb=config.wandb,
        file_monitor=config.file_monitor,
    )


def write_config(config: SFTConfig, config_path: Path, exclude: set[str] | None = None) -> None:
    """Write resolved config to disk, excluding launcher-only fields."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "wb") as f:
        tomli_w.dump(to_toml_dict(config, exclude=exclude), f)


def write_eval_subconfigs(config: SFTConfig, config_dir: Path) -> None:
    """Write the inference, evaluator, and env-server TOMLs for online evals."""
    config_dir.mkdir(parents=True, exist_ok=True)

    if config.inference is not None:
        # Exclude launcher-only fields that are not needed by the vLLM server
        exclude_inference = {"deployment", "slurm", "output_dir", "dry_run"}
        with open(config_dir / INFERENCE_TOML, "wb") as f:
            tomli_w.dump(to_toml_dict(config.inference, exclude=exclude_inference), f)

    with open(config_dir / EVALUATOR_TOML, "wb") as f:
        tomli_w.dump(to_toml_dict(build_evaluator_config(config)), f)

    # One EnvServerConfig TOML per launcher-managed eval source: `env-server @ <path>`
    # binds at the source's deterministic address, where the evaluator connects.
    for source, address in eval_env_servers(config):
        env_dir = config_dir / ENVS_DIR / "eval"
        env_dir.mkdir(parents=True, exist_ok=True)
        source_dict = to_toml_dict(source)
        env_server_dict = {
            "env": source_dict["env"],
            "serve": {**source_dict.get("serve", {}), "address": address},
            "log": {"level": config.log.vf_level, "json_logging": config.log.json_logging},
        }
        with open(env_dir / f"{source.resolved_name}.toml", "wb") as f:
            tomli_w.dump(env_server_dict, f)


def write_slurm_script(config: SFTConfig, config_path: Path, script_path: Path) -> None:
    """Write the SLURM script to disk."""
    from jinja2 import Environment, FileSystemLoader

    assert config.slurm is not None
    assert config.slurm.template_path is not None

    env = Environment(loader=FileSystemLoader(config.slurm.template_path.parent), keep_trailing_newline=True)
    template = env.get_template(config.slurm.template_path.name)

    trainer_env_vars = {
        **DEFAULT_COMMON_ENV_VARS,
        **DEFAULT_TRAINER_ENV_VARS,
        **config.env_vars,
    }

    if config.deployment.type == "single_node":
        script = template.render(
            **config.slurm.template_vars,
            config_path=config_path,
            output_dir=config.output_dir,
            gpus_per_node=config.deployment.gpus_per_node,
        )
    else:
        script = template.render(
            **config.slurm.template_vars,
            config_path=config_path,
            output_dir=config.output_dir,
            trainer_env_vars=trainer_env_vars,
            num_nodes=config.deployment.num_nodes,
            gpus_per_node=config.deployment.gpus_per_node,
            ranks_filter=",".join(map(str, config.log.ranks_filter)),
        )

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)


def sft_slurm(config: SFTConfig):
    """Run SFT training via SLURM."""
    assert config.slurm is not None

    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    config_dir = get_config_dir(config.output_dir)
    config_path = config_dir / SFT_TOML
    exclude = (
        {"deployment", "slurm", "dry_run", "clean_output_dir"}
        if config.deployment.type == "multi_node"
        else {"slurm", "dry_run", "clean_output_dir"}
    )
    write_config(config, config_path, exclude=exclude)
    logger.info(f"Wrote config to {config_path}")

    script_path = config.output_dir / SFT_SBATCH
    write_slurm_script(config, config_path, script_path)
    logger.info(f"Wrote SLURM script to {script_path}")

    log_dir = get_log_dir(config.output_dir)
    num_nodes = config.deployment.num_nodes if config.deployment.type == "multi_node" else 1
    log_message = format_log_message(log_dir=log_dir, trainer=True, num_train_nodes=num_nodes)

    if config.dry_run:
        logger.success(f"Dry run complete. To submit manually:\n\n  sbatch {script_path}\n\n{log_message}")
        return

    logger.info(f"Submitting: sbatch {script_path}")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr.strip()}")
        sys.exit(1)

    logger.success(f"{result.stdout.strip()}\n\n{log_message}")


def sft_local(config: SFTConfig):
    """Run SFT training locally with process monitoring and cleanup."""
    assert config.deployment.type == "single_node"

    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    config_dir = get_config_dir(config.output_dir)
    config_path = config_dir / SFT_TOML
    write_config(config, config_path)
    logger.info(f"Wrote config to {config_path}")

    if config.eval is not None:
        write_eval_subconfigs(config, config_dir)
        logger.info(f"Wrote eval subconfigs to {config_dir}")

    if config.dry_run:
        logger.success("Dry run complete. To start an SFT run locally, remove --dry-run from your command.")
        return

    log_dir = get_log_dir(config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Derive launcher-local GPU IDs (inference first, then the trainer) only when the
    # launcher must partition GPUs between processes; plain SFT leaves them to torchrun.
    infer_gpu_ids: list[int] = []
    trainer_gpu_ids: list[int] = []
    if config.inference is not None:
        num_infer_gpus = config.deployment.num_infer_gpus
        total_requested_gpus = num_infer_gpus + config.deployment.num_gpus
        physical_gpu_ids = get_physical_gpu_ids()
        if total_requested_gpus > len(physical_gpu_ids):
            raise ValueError(
                f"Requested {total_requested_gpus} GPUs via deployment settings, but only "
                f"{len(physical_gpu_ids)} physical GPU(s) are available: {physical_gpu_ids}"
            )
        infer_gpu_ids = physical_gpu_ids[:num_infer_gpus]
        trainer_gpu_ids = physical_gpu_ids[num_infer_gpus:total_requested_gpus]

    # Trainer and evaluator log to a single shared W&B run, one label per process.
    wandb_shared_env: dict[str, str] = {}
    if config.eval is not None:
        wandb_shared_env = {
            "WANDB_SHARED_MODE": "1",
            "WANDB_SHARED_RUN_ID": os.environ.get("WANDB_SHARED_RUN_ID", uuid.uuid4().hex),
            "WANDB_SHARED_PRIMARY": "evaluator",
            "WANDB_PROGRAM": "uv run sft",
            "WANDB_ARGS": json.dumps(sys.argv),
        }

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

    def start_process(name: str, cmd: list[str], env: dict[str, str], log_path: Path) -> Popen:
        logger.debug(f"{name.capitalize()} command: {' '.join(cmd)}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log_file:
            process = Popen(cmd, env=env, stdout=log_file, stderr=log_file)
        processes.append(process)
        stop_event = Event()
        stop_events[name] = stop_event
        monitor_thread = Thread(target=monitor_process, args=(process, stop_event, error_queue, name), daemon=True)
        monitor_thread.start()
        monitor_threads.append(monitor_thread)
        return process

    try:
        # Optionally, start the inference server for online evals
        if config.inference is not None:
            logger.info(f"Starting inference on GPU(s) {' '.join(map(str, infer_gpu_ids))}")
            start_process(
                "inference",
                ["inference", "@", (config_dir / INFERENCE_TOML).as_posix()],
                env={
                    **os.environ,
                    **DEFAULT_COMMON_ENV_VARS,
                    **DEFAULT_INFERENCE_ENV_VARS,
                    **config.env_vars,
                    **config.inference.env_vars,
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, infer_gpu_ids)),
                },
                log_path=log_dir / "inference.log",
            )

        # Start one env server per eval source. The evaluator connects to each source's
        # deterministic address, polling until the server is up.
        for source, address in eval_env_servers(config):
            name = source.resolved_name
            logger.info(f"Starting eval env server {name} at {address}")
            start_process(
                f"env/eval/{name}",
                ["env-server", "@", (config_dir / ENVS_DIR / "eval" / f"{name}.toml").as_posix()],
                env={**os.environ, **DEFAULT_COMMON_ENV_VARS, **config.env_vars},
                log_path=log_dir / ENVS_DIR / "eval" / f"{name}.log",
            )

        if config.eval is not None:
            logger.info("Starting evaluator process")
            start_process(
                "evaluator",
                ["evaluator", "@", (config_dir / EVALUATOR_TOML).as_posix()],
                env={
                    **os.environ,
                    **DEFAULT_COMMON_ENV_VARS,
                    "LOGURU_FORCE_COLORS": "1",
                    **config.env_vars,
                    **wandb_shared_env,
                    "WANDB_SHARED_LABEL": "evaluator",
                },
                log_path=log_dir / "evaluator.log",
            )

        from prime_rl.utils.utils import get_free_port

        trainer_cmd = [
            "torchrun",
            "--role=trainer",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            f"--rdzv-id={uuid.uuid4().hex}",
            f"--log-dir={log_dir / 'trainer' / 'torchrun'}",
            f"--local-ranks-filter={','.join(map(str, config.log.ranks_filter))}",
            "--redirect=3",
            "--tee=3",
            f"--nproc-per-node={config.deployment.num_gpus}",
            "-m",
            "prime_rl.trainer.sft.train",
            "@",
            config_path.as_posix(),
        ]
        gpus_suffix = f" on GPU(s) {' '.join(map(str, trainer_gpu_ids))}" if trainer_gpu_ids else ""
        logger.info(f"Starting SFT trainer with {config.deployment.num_gpus} GPU(s){gpus_suffix}")
        trainer_env = {
            **os.environ,
            **DEFAULT_COMMON_ENV_VARS,
            **DEFAULT_TRAINER_ENV_VARS,
            **config.env_vars,
            **wandb_shared_env,
        }
        if config.eval is not None:
            trainer_env["LOGURU_FORCE_COLORS"] = "1"
            trainer_env["WANDB_SHARED_LABEL"] = "trainer"
        if trainer_gpu_ids:
            trainer_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, trainer_gpu_ids))
        trainer_process = start_process("trainer", trainer_cmd, env=trainer_env, log_path=log_dir / "trainer.log")

        logger.success("Startup complete. Showing trainer logs...")
        tail_process = Popen(
            f"tail -F '{log_dir / 'trainer.log'}' | sed -u 's/^\\[[a-zA-Z]*[0-9]*\\]://'",
            shell=True,
        )
        processes.append(tail_process)

        # Wait for the trainer (and the evaluator, which drains its final evals after
        # the trainer's last checkpoint) while surfacing any process failure.
        terminal_events = [stop_events["trainer"]]
        if "evaluator" in stop_events:
            terminal_events.append(stop_events["evaluator"])
        while True:
            pending = [event for event in terminal_events if not event.is_set()]
            if error_queue:
                logger.error(f"Error: {error_queue[0]}")
                logger.error("Terminating all processes...")
                cleanup_threads(monitor_threads)
                cleanup_processes(processes)
                sys.exit(1)
            if not pending:
                break
            pending[0].wait(timeout=1)

        if trainer_process.returncode != 0:
            logger.error(f"Trainer failed with exit code {trainer_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        logger.success("SFT training finished!")
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


def clean_stale_weights(config: SFTConfig) -> None:
    """Remove weight checkpoints a previous run left behind: everything on a fresh
    start, steps past the resume step on resume. Without this the evaluator would
    replay stale checkpoints (and then skip the re-trained ones at the same steps)."""
    ckpt_base = (config.ckpt.output_dir if config.ckpt else None) or config.output_dir
    weights_dir = get_weights_dir(ckpt_base)
    resume_step = resolve_resume_step(config)
    stale_steps = [step for step in get_all_ckpt_steps(weights_dir) if resume_step is None or step > resume_step]
    if not stale_steps:
        return
    setup_logger(config.log.level or "info").info(
        f"Deleting {len(stale_steps)} stale weight checkpoint(s) in {weights_dir} ({','.join(map(str, stale_steps))})"
    )
    for step in stale_steps:
        shutil.rmtree(get_step_path(weights_dir, step), ignore_errors=True)


def sft(config: SFTConfig):
    resuming = config.ckpt is not None and config.ckpt.resume_step is not None
    clean = config.clean_output_dir and not os.environ.get("NEVER_CLEAN_OUTPUT_DIR")
    validate_output_dir(config.output_dir, resuming=resuming, clean=clean)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.eval is not None and not config.dry_run:
        clean_stale_weights(config)

    if not config.dry_run:
        from prime_rl.trainer.model import pre_download_model

        pre_download_model(config.model.name, skip_weights=config.model.debug.random_init)

    if config.slurm is not None:
        sft_slurm(config)
    else:
        sft_local(config)


def main():
    set_proc_title("SFT")
    sft(cli(SFTConfig))


if __name__ == "__main__":
    main()
