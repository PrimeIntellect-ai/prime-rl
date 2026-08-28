import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread

from prime_rl.configs.evals import EvalsConfig, OnlineConfig
from prime_rl.configs.orchestrator import EvalSourceConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.shared import LogConfig
from prime_rl.entrypoints.dashboard import ensure_dashboard, log_dashboard_url
from prime_rl.utils.config import cli, dump_resolved_config, find_package_resource
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import (
    clean_future_steps,
    format_log_message,
    get_broadcast_dir,
    get_ckpt_dir,
    get_launcher_dir,
    get_launcher_log_dir,
    prepare_attempt_dirs,
    resolve_latest_ckpt_step,
    validate_run_dir,
    write_launch_artifacts,
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

SFT_CONFIG = "sft.json"
SFT_SBATCH = "sft.sbatch"

INFERENCE_CONFIG = "inference.json"
EVALS_CONFIG = "evals.json"

ENVS_DIR = "envs"


def eval_env_servers(config: SFTConfig) -> list[tuple[EvalSourceConfig, str]]:
    """``(source, address)`` for every launcher-managed eval source. A source with
    ``serve.address`` set is externally managed — the launcher neither writes its
    config nor spawns a server for it."""
    if config.eval is None:
        return []
    addresses = config.eval.env_addresses
    return [
        (source, addresses[("eval", source.resolved_name)])
        for source in config.eval.source
        if source.serve.address is None
    ]


def get_ckpt_base(config: SFTConfig) -> Path:
    """Where checkpoints live: ``ckpt.output_dir`` when set, else the run dir."""
    return (config.ckpt.output_dir if config.ckpt else None) or config.run_dir


def resolve_resume_step(config: SFTConfig) -> int | None:
    if config.resume is None:
        return None
    if config.resume.dir is not None:
        return config.resume.dir_step
    if config.resume.step is not None:
        return config.resume.step
    return resolve_latest_ckpt_step(get_ckpt_dir(get_ckpt_base(config)))


def build_evals_config(config: SFTConfig) -> EvalsConfig:
    """Derive the evals subconfig from the resolved SFT config. The launcher
    spawns the env servers itself, so each source's derived address is stamped in,
    marking it externally managed for the evals process."""
    assert config.eval is not None
    eval_config = config.eval.model_copy(deep=True)
    addresses = config.eval.env_addresses
    for source in eval_config.source:
        source.serve.address = addresses[("eval", source.resolved_name)]
    return EvalsConfig(
        model=config.model.name,
        eval=eval_config,
        weight_broadcast=config.weight_broadcast,
        online=OnlineConfig(
            broadcasts_dir=get_broadcast_dir(config.run_dir),
            max_steps=config.max_steps,
            resume_step=resolve_resume_step(config),
        ),
        output_dir=config.run_dir,
        log=LogConfig(level=config.log.level, json_logging=config.log.json_logging),
        monitors=config.monitors,
    )


def write_config(config: SFTConfig, config_path: Path, exclude: set[str] | None = None) -> None:
    """Write resolved config to disk, excluding launcher-only fields."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(dump_resolved_config(config, exclude=exclude), f, indent=2)


def write_eval_subconfigs(config: SFTConfig, config_dir: Path, strip_router: bool = False) -> None:
    """Write the inference, evals, and env-server configs for online evals."""
    config_dir.mkdir(parents=True, exist_ok=True)

    if config.inference is not None:
        # Exclude launcher-only fields that are not needed by the vLLM server
        exclude_inference = {"deployment", "slurm", "output_dir", "dry_run"}
        inference_dict = dump_resolved_config(config.inference, exclude=exclude_inference)
        if strip_router:
            # Per-rank processes run bare engines; the sbatch starts the single global router.
            inference_dict["router"] = None
        with open(config_dir / INFERENCE_CONFIG, "w") as f:
            json.dump(inference_dict, f, indent=2)

    with open(config_dir / EVALS_CONFIG, "w") as f:
        json.dump(dump_resolved_config(build_evals_config(config)), f, indent=2)

    # One EnvServerConfig per launcher-managed eval source: `env-server @ <path>`
    # binds at the source's deterministic address, where the evals process connects.
    for source, address in eval_env_servers(config):
        env_dir = config_dir / ENVS_DIR / "eval"
        env_dir.mkdir(parents=True, exist_ok=True)
        source_dict = dump_resolved_config(source)
        env_server_dict = {
            "env": source_dict["env"],
            "serve": {**(source_dict.get("serve") or {}), "address": address},
            "log": {"level": config.log.vf_level, "json_logging": config.log.json_logging},
        }
        with open(env_dir / f"{source.resolved_name}.json", "w") as f:
            json.dump(env_server_dict, f, indent=2)


def write_slurm_script(
    config: SFTConfig, config_path: Path, log_dir: Path, script_path: Path, prl_run_id: str | None = None
) -> None:
    """Write the SLURM script to disk."""
    from jinja2 import Environment, FileSystemLoader

    assert config.slurm is not None
    assert config.slurm.template_path is not None

    template_dirs = [config.slurm.template_path.parent]
    bundled_templates = find_package_resource("templates")
    if bundled_templates is not None:
        template_dirs.append(bundled_templates)
    env = Environment(loader=FileSystemLoader(template_dirs), keep_trailing_newline=True)
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
            config_dir=config_path.parent,
            log_dir=log_dir,
            output_dir=config.run_dir,
            launcher_dir=get_launcher_dir(config.run_dir),
            launcher_log_dir=get_launcher_log_dir(config.run_dir),
            gpus_per_node=config.deployment.gpus_per_node,
        )
    else:
        online_eval = config.eval is not None
        eval_vars = {}
        if online_eval:
            assert config.inference is not None
            inference_env_vars = {
                **DEFAULT_COMMON_ENV_VARS,
                **DEFAULT_INFERENCE_ENV_VARS,
                **config.env_vars,
                **config.inference.env_vars,
            }
            eval_vars = {
                "router": config.inference.router,
                "router_port": config.inference.server.port,
                "backend_port": config.inference.backend_port,
                "data_parallel_rpc_port": config.inference.vllm.data_parallel_rpc_port,
                "dp_per_node": config.deployment.gpus_per_node // config.inference.vllm.tensor_parallel_size,
                "enable_expert_parallel": config.inference.vllm.enable_expert_parallel,
                "inference_env_vars": inference_env_vars,
                "evals_env_vars": {
                    **DEFAULT_COMMON_ENV_VARS,
                    "LOGURU_FORCE_COLORS": "1",
                    **config.env_vars,
                },
                "eval_env_names": [source.resolved_name for source, _ in eval_env_servers(config)],
            }
        script = template.render(
            **config.slurm.template_vars,
            config_path=config_path,
            config_dir=config_path.parent,
            log_dir=log_dir,
            output_dir=config.run_dir,
            launcher_dir=get_launcher_dir(config.run_dir),
            launcher_log_dir=get_launcher_log_dir(config.run_dir),
            trainer_env_vars=trainer_env_vars,
            num_nodes=config.deployment.num_train_nodes,
            num_train_nodes=config.deployment.num_train_nodes,
            num_infer_nodes=config.deployment.num_infer_nodes if online_eval else 0,
            gpus_per_node=config.deployment.gpus_per_node,
            ranks_filter=",".join(map(str, config.log.ranks_filter)),
            prl_run_id=prl_run_id,
            run_name=config.run.name,
            online_eval=online_eval,
            use_nccl_broadcast=(
                config.eval is not None
                and config.weight_broadcast is not None
                and config.weight_broadcast.type == "nccl"
            ),
            **eval_vars,
        )

    script_path.parent.mkdir(parents=True, exist_ok=True)
    get_launcher_log_dir(config.run_dir).mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)


def sft_slurm(config: SFTConfig):
    """Run SFT training and its online-eval deployment in one SLURM allocation."""
    assert config.slurm is not None

    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    online_eval = config.deployment.type == "multi_node" and config.eval is not None

    config_dir, log_dir = prepare_attempt_dirs(config.run_dir)
    write_launch_artifacts(config_dir, "sft")
    config_path = config_dir / SFT_CONFIG
    exclude = (
        {"deployment", "slurm", "dry_run", "clean"}
        if config.deployment.type == "multi_node"
        else {"slurm", "dry_run", "clean"}
    )
    if online_eval:
        # The trainer job only needs [eval] for the weight-broadcast cadence; the
        # inference pool runs on its dedicated nodes in the same allocation.
        exclude = exclude | {"inference"}
    write_config(config, config_path, exclude=exclude)
    logger.info(f"Wrote config to {config_path}")

    # Trainer and evals processes log to a single shared W&B run.
    prl_run_id: str | None = None
    if online_eval and config.monitors.wandb is not None:
        prl_run_id = os.environ["PRL_RUN_ID"]

    launcher_dir = get_launcher_dir(config.run_dir)
    if online_eval:
        write_eval_subconfigs(config, config_dir, strip_router=True)
        logger.info(f"Wrote eval subconfigs to {config_dir}")
    script_path = launcher_dir / SFT_SBATCH
    write_slurm_script(config, config_path, log_dir, script_path, prl_run_id)
    logger.info(f"Wrote SLURM script to {script_path}")

    num_nodes = config.deployment.num_train_nodes if config.deployment.type == "multi_node" else 1
    log_message = format_log_message(
        log_dir=log_dir,
        trainer=True,
        num_train_nodes=num_nodes,
        evals=online_eval,
        inference=online_eval,
        eval_env_names=[source.resolved_name for source, _ in eval_env_servers(config)] if online_eval else None,
        num_infer_nodes=config.deployment.num_infer_nodes if online_eval else 0,
    )

    if config.dry_run:
        logger.success(f"Dry run complete. To submit manually:\n\n  sbatch {script_path}\n\n{log_message}")
        return

    dashboard_url = ensure_dashboard(config.output_dir, logger) if config.dashboard else None

    logger.info(f"Submitting: sbatch {script_path}")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr.strip()}")
        sys.exit(1)

    logger.success(f"{result.stdout.strip()}\n\n{log_message}")
    log_dashboard_url(logger, dashboard_url)


def sft_local(config: SFTConfig):
    """Run SFT training locally with process monitoring and cleanup."""
    assert config.deployment.type == "single_node"

    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    config_dir, log_dir = prepare_attempt_dirs(config.run_dir)
    write_launch_artifacts(config_dir, "sft")
    config_path = config_dir / SFT_CONFIG
    write_config(config, config_path)
    logger.info(f"Wrote config to {config_path}")

    if config.eval is not None:
        write_eval_subconfigs(config, config_dir)
        logger.info(f"Wrote eval subconfigs to {config_dir}")

    if config.dry_run:
        logger.success("Dry run complete. To start an SFT run locally, remove --dry-run from your command.")
        return

    dashboard_url = ensure_dashboard(config.output_dir, logger) if config.dashboard else None

    # Derive launcher-local GPU IDs (inference first, then the trainer) only when the
    # launcher must partition GPUs between processes; plain SFT leaves them to torchrun.
    infer_gpu_ids: list[int] = []
    trainer_gpu_ids: list[int] = []
    if config.inference is not None:
        num_infer_gpus = config.deployment.num_infer_gpus
        total_requested_gpus = num_infer_gpus + config.deployment.num_train_gpus
        physical_gpu_ids = get_physical_gpu_ids()
        if total_requested_gpus > len(physical_gpu_ids):
            raise ValueError(
                f"Requested {total_requested_gpus} GPUs via deployment settings, but only "
                f"{len(physical_gpu_ids)} physical GPU(s) are available: {physical_gpu_ids}"
            )
        infer_gpu_ids = physical_gpu_ids[:num_infer_gpus]
        trainer_gpu_ids = physical_gpu_ids[num_infer_gpus:total_requested_gpus]

    # Trainer and evals log to a single shared W&B run whose id ($WANDB_RUN_ID)
    # equals $PRL_RUN_ID, one label per process.
    wandb_shared_env: dict[str, str] = {}
    if config.eval is not None:
        # The trainer creates the run; the evals process (which drains its final evals
        # after the trainer exits) finalizes it.
        wandb_shared_env = {
            "WANDB_SHARED_MODE": "1",
            "WANDB_RUN_ID": os.environ["PRL_RUN_ID"],
            "WANDB_SHARED_PRIMARY": "trainer",
            "WANDB_SHARED_FINISHER": "evals",
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
                ["inference", "@", (config_dir / INFERENCE_CONFIG).as_posix()],
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

        # Start one env server per eval source. The evals process connects to each source's
        # deterministic address, polling until the server is up.
        for source, address in eval_env_servers(config):
            name = source.resolved_name
            logger.info(f"Starting {name} server")
            start_process(
                f"env/eval/{name}",
                ["env-server", "@", (config_dir / ENVS_DIR / "eval" / f"{name}.json").as_posix()],
                env={**os.environ, **DEFAULT_COMMON_ENV_VARS, **config.env_vars},
                log_path=log_dir / ENVS_DIR / "eval" / f"{name}.log",
            )

        if config.eval is not None:
            logger.info("Starting evals")
            start_process(
                "evals",
                ["evals", "@", (config_dir / EVALS_CONFIG).as_posix()],
                env={
                    **os.environ,
                    **DEFAULT_COMMON_ENV_VARS,
                    "LOGURU_FORCE_COLORS": "1",
                    **config.env_vars,
                    "PRL_ATTEMPT_CONFIG_DIR": str(config_dir),
                    "PRL_ATTEMPT_LOG_DIR": str(log_dir),
                    "PRL_LOG_DIR": str(log_dir),
                    **wandb_shared_env,
                    "WANDB_SHARED_LABEL": "evals",
                },
                log_path=log_dir / "evals.log",
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
            f"--nproc-per-node={config.deployment.num_train_gpus}",
            "-m",
            "prime_rl.trainer.sft.train",
            "@",
            config_path.as_posix(),
        ]
        gpus_suffix = f" on GPU(s) {' '.join(map(str, trainer_gpu_ids))}" if trainer_gpu_ids else ""
        logger.info(f"Starting SFT trainer with {config.deployment.num_train_gpus} GPU(s){gpus_suffix}")
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

        logger.success("Launcher complete")
        log_dashboard_url(logger, dashboard_url)

        # Wait for the trainer (and the evals process, which drains its final evals after
        # the trainer's last checkpoint) while surfacing any process failure.
        terminal_events = [stop_events["trainer"]]
        if "evals" in stop_events:
            terminal_events.append(stop_events["evals"])
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


def clean_stale_eval_artifacts(config: SFTConfig) -> None:
    """Remove eval artifacts a previous run left behind: weight broadcasts and rollout
    trace dirs — everything on a fresh start, steps past the resume step on resume.
    Without this the evals process would replay stale broadcasts (and then skip the
    re-trained ones at the same steps), and the append-only trace files would mix two
    policies' rollouts under one step."""
    logger = setup_logger(config.log.level or "info")
    if os.environ.get("NEVER_CLEAN"):
        logger.warning("NEVER_CLEAN is set - keeping stale weight broadcasts; the evals process may replay them")
        return
    resume_step = resolve_resume_step(config)
    clean_future_steps(config.run_dir, resume_step if resume_step is not None else -1)


def sft(config: SFTConfig):
    # Launcher-only check: the trainer re-parses a sub-config with [inference] and
    # [deployment] stripped, so the model validator cannot enforce this.
    if config.weight_broadcast is not None and config.weight_broadcast.type == "nccl" and config.inference is None:
        raise ValueError(
            "NCCL weight broadcast requires launcher-managed inference. "
            "Add an [inference] block or set weight_broadcast.type = 'filesystem'."
        )

    # The run identity is runtime-only, never sub-config: $PRL_RUN_ID / $PRL_RUN_NAME are
    # the vehicle for runtime info between processes, and every spawned process inherits
    # them. Components launched standalone have no run identity.
    os.environ.setdefault("PRL_RUN_ID", uuid.uuid4().hex)
    assert config.run.name is not None  # resolved at construction
    os.environ["PRL_RUN_NAME"] = config.run.name

    resuming = config.resume is not None
    clean = config.clean and not os.environ.get("NEVER_CLEAN")
    ckpt_output_dir = config.ckpt.output_dir if config.ckpt else None
    validate_run_dir(
        config.run_dir, output_dir=config.output_dir, resuming=resuming, clean=clean, ckpt_output_dir=ckpt_output_dir
    )
    config.run_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_output_dir is not None:
        ckpt_output_dir.mkdir(parents=True, exist_ok=True)

    if config.eval is not None and not config.dry_run:
        clean_stale_eval_artifacts(config)

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
