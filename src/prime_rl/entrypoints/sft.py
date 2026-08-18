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

from prime_rl.configs.evaluator import EvaluatorConfig
from prime_rl.configs.orchestrator import EvalSourceConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.shared import LogConfig
from prime_rl.utils.config import cli, dump_resolved_config, find_package_resource
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import (
    clean_future_steps,
    create_attempt_log_dir,
    format_log_message,
    get_all_ckpt_steps,
    get_ckpt_dir,
    get_config_dir,
    get_log_dir,
    get_step_path,
    get_weights_dir,
    latest_log_dir,
    resolve_latest_ckpt_step,
    validate_run_dir,
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
EVAL_SBATCH = "eval.sbatch"
EVAL_TEMPLATE = "multi_node_sft_eval.sbatch.j2"

INFERENCE_CONFIG = "inference.json"
EVALUATOR_CONFIG = "evaluator.json"

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
    """Where checkpoints and weights live: ``ckpt.output_dir`` when set, else the run dir."""
    return (config.ckpt.output_dir if config.ckpt else None) or config.run_dir


def resolve_resume_step(config: SFTConfig) -> int | None:
    if config.resume is None:
        return None
    if config.resume.dir is not None:
        return config.resume.dir_step
    if config.resume.step is not None:
        return config.resume.step
    return resolve_latest_ckpt_step(get_ckpt_dir(get_ckpt_base(config)))


def build_evaluator_config(config: SFTConfig) -> EvaluatorConfig:
    """Derive the evaluator subconfig from the resolved SFT config."""
    assert config.eval is not None
    return EvaluatorConfig(
        model=config.model.name,
        eval=config.eval,
        weights_dir=get_weights_dir(get_ckpt_base(config)),
        output_dir=config.run_dir,
        max_steps=config.max_steps,
        resume_step=resolve_resume_step(config),
        log=LogConfig(level=config.log.level, json_logging=config.log.json_logging),
        monitors=config.monitors,
    )


def write_config(config: SFTConfig, config_path: Path, exclude: set[str] | None = None) -> None:
    """Write resolved config to disk, excluding launcher-only fields."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(dump_resolved_config(config, exclude=exclude), f, indent=2)


def write_eval_subconfigs(config: SFTConfig, config_dir: Path, strip_router: bool = False) -> None:
    """Write the inference, evaluator, and env-server configs for online evals."""
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

    with open(config_dir / EVALUATOR_CONFIG, "w") as f:
        json.dump(dump_resolved_config(build_evaluator_config(config)), f, indent=2)

    # One EnvServerConfig per launcher-managed eval source: `env-server @ <path>`
    # binds at the source's deterministic address, where the evaluator connects.
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


def write_slurm_script(config: SFTConfig, config_path: Path, script_path: Path, prl_run_id: str | None = None) -> None:
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
            output_dir=config.run_dir,
            gpus_per_node=config.deployment.gpus_per_node,
        )
    else:
        script = template.render(
            **config.slurm.template_vars,
            config_path=config_path,
            output_dir=config.run_dir,
            trainer_env_vars=trainer_env_vars,
            num_nodes=config.deployment.num_train_nodes,
            gpus_per_node=config.deployment.gpus_per_node,
            ranks_filter=",".join(map(str, config.log.ranks_filter)),
            prl_run_id=prl_run_id,
            run_name=config.run.name,
        )

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)


def write_eval_slurm_script(config: SFTConfig, config_dir: Path, script_path: Path, prl_run_id: str | None) -> None:
    """Write the SLURM script for the decoupled online-eval job (inference pool +
    env servers + evaluator) to disk."""
    from jinja2 import Environment, FileSystemLoader

    assert config.slurm is not None
    assert config.slurm.template_path is not None
    assert config.deployment.type == "multi_node"
    assert config.inference is not None

    # The eval template is bundled next to the trainer templates; a custom
    # slurm.template_path directory takes precedence so both can be overridden.
    template_dirs = [config.slurm.template_path.parent]
    bundled_templates = find_package_resource("templates")
    if bundled_templates is not None:
        template_dirs.append(bundled_templates)
    env = Environment(loader=FileSystemLoader(template_dirs), keep_trailing_newline=True)
    template = env.get_template(EVAL_TEMPLATE)

    inference_env_vars = {
        **DEFAULT_COMMON_ENV_VARS,
        **DEFAULT_INFERENCE_ENV_VARS,
        **config.env_vars,
        **config.inference.env_vars,
    }
    evaluator_env_vars = {**DEFAULT_COMMON_ENV_VARS, "LOGURU_FORCE_COLORS": "1", **config.env_vars}

    script = template.render(
        **config.slurm.template_vars,
        config_dir=config_dir,
        output_dir=config.run_dir,
        num_infer_nodes=config.deployment.num_infer_nodes,
        gpus_per_node=config.deployment.gpus_per_node,
        router=config.inference.router,
        router_port=config.inference.server.port,
        backend_port=config.inference.backend_port,
        data_parallel_rpc_port=config.inference.vllm.data_parallel_rpc_port,
        dp_per_node=config.deployment.gpus_per_node // config.inference.vllm.tensor_parallel_size,
        enable_expert_parallel=config.inference.vllm.enable_expert_parallel,
        inference_env_vars=inference_env_vars,
        evaluator_env_vars=evaluator_env_vars,
        eval_env_names=[source.resolved_name for source, _ in eval_env_servers(config)],
        prl_run_id=prl_run_id,
        run_name=config.run.name,
    )

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)


def sft_slurm(config: SFTConfig):
    """Run SFT training via SLURM. With online evals on a multi-node deployment, the
    trainer and the eval deployment (inference pool + evaluator) are two independent
    SLURM jobs: the handoff is weight checkpoints on the shared filesystem, so the
    trainer job releases its allocation when training finishes while the eval job
    keeps draining evals and exits after the final checkpoint."""
    assert config.slurm is not None

    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    decoupled_eval = config.deployment.type == "multi_node" and config.eval is not None

    config_dir = get_config_dir(config.run_dir)
    config_path = config_dir / SFT_CONFIG
    exclude = (
        {"deployment", "slurm", "dry_run", "clean"}
        if config.deployment.type == "multi_node"
        else {"slurm", "dry_run", "clean"}
    )
    if decoupled_eval:
        # The trainer job only needs [eval] for the weight-checkpoint cadence; the
        # inference pool lives in the eval job.
        exclude = exclude | {"inference"}
    write_config(config, config_path, exclude=exclude)
    logger.info(f"Wrote config to {config_path}")

    # Trainer and evaluator processes log to a single shared W&B run across both jobs,
    # keyed by the launcher's run id.
    prl_run_id: str | None = None
    if decoupled_eval and config.monitors.wandb is not None:
        prl_run_id = os.environ["PRL_RUN_ID"]

    script_path = config.run_dir / SFT_SBATCH
    write_slurm_script(config, config_path, script_path, prl_run_id)
    logger.info(f"Wrote SLURM script to {script_path}")

    # The trainer job is submitted first and the eval job depends on it having started:
    # the trainer creates the shared W&B run, and the evaluator's joiner init only
    # retries for a bounded window — starting the evaluator while the trainer job pends
    # would crash it at wandb init.
    script_paths = [script_path]
    if decoupled_eval:
        write_eval_subconfigs(config, config_dir, strip_router=True)
        logger.info(f"Wrote eval subconfigs to {config_dir}")
        eval_script_path = config.run_dir / EVAL_SBATCH
        write_eval_slurm_script(config, config_dir, eval_script_path, prl_run_id)
        logger.info(f"Wrote eval SLURM script to {eval_script_path}")
        script_paths = [script_path, eval_script_path]

    num_nodes = config.deployment.num_train_nodes if config.deployment.type == "multi_node" else 1
    log_message = format_log_message(
        log_dir=latest_log_dir(config.run_dir),
        trainer=True,
        num_train_nodes=num_nodes,
    )
    if decoupled_eval:
        # The eval job logs at stable (non-attempt) paths under the run dir.
        log_message += "\n" + format_log_message(
            log_dir=get_log_dir(config.run_dir),
            evaluator=True,
            inference=True,
            eval_env_names=[source.resolved_name for source, _ in eval_env_servers(config)],
            num_infer_nodes=config.deployment.num_infer_nodes,
        ).removeprefix("Logs:\n")

    if config.dry_run:
        submit = "\n".join(f"  sbatch {path}" for path in script_paths)
        note = (
            "\n\nSubmit the trainer job first — the evaluator joins the W&B run the trainer creates."
            if decoupled_eval
            else ""
        )
        logger.success(f"Dry run complete. To submit manually:\n\n{submit}{note}\n\n{log_message}")
        return

    submitted_job_ids: list[str] = []
    for path in script_paths:
        # --parsable prints ``<job_id>[;<cluster>]`` — the human-readable format varies
        # (e.g. multi-cluster sbatch appends "on cluster <name>").
        cmd = ["sbatch", "--parsable"]
        if submitted_job_ids:
            # Hold the eval job until the trainer job has started (not finished).
            cmd.append(f"--dependency=after:{submitted_job_ids[-1]}")
        cmd.append(str(path))
        logger.info(f"Submitting: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"sbatch failed: {result.stderr.strip()}")
            for job_id in submitted_job_ids:
                logger.warning(f"Cancelling already-submitted job {job_id}")
                subprocess.run(["scancel", job_id], capture_output=True, text=True)
            sys.exit(1)
        job_id = result.stdout.strip().split(";")[0]
        submitted_job_ids.append(job_id)
        logger.success(f"Submitted batch job {job_id}")

    logger.success(log_message)


def sft_local(config: SFTConfig):
    """Run SFT training locally with process monitoring and cleanup."""
    assert config.deployment.type == "single_node"

    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    config_dir = get_config_dir(config.run_dir)
    config_path = config_dir / SFT_CONFIG
    write_config(config, config_path)
    logger.info(f"Wrote config to {config_path}")

    if config.eval is not None:
        write_eval_subconfigs(config, config_dir)
        logger.info(f"Wrote eval subconfigs to {config_dir}")

    if config.dry_run:
        logger.success("Dry run complete. To start an SFT run locally, remove --dry-run from your command.")
        return

    log_dir = create_attempt_log_dir(config.run_dir)

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

    # Trainer and evaluator log to a single shared W&B run whose id ($WANDB_RUN_ID)
    # equals $PRL_RUN_ID, one label per process.
    wandb_shared_env: dict[str, str] = {}
    if config.eval is not None:
        # The trainer creates the run; the evaluator (which drains its final evals
        # after the trainer exits) finalizes it.
        wandb_shared_env = {
            "WANDB_SHARED_MODE": "1",
            "WANDB_RUN_ID": os.environ["PRL_RUN_ID"],
            "WANDB_SHARED_PRIMARY": "trainer",
            "WANDB_SHARED_FINISHER": "evaluator",
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

        # Start one env server per eval source. The evaluator connects to each source's
        # deterministic address, polling until the server is up.
        for source, address in eval_env_servers(config):
            name = source.resolved_name
            logger.info(f"Starting eval env server {name} at {address}")
            start_process(
                f"env/eval/{name}",
                ["env-server", "@", (config_dir / ENVS_DIR / "eval" / f"{name}.json").as_posix()],
                env={**os.environ, **DEFAULT_COMMON_ENV_VARS, **config.env_vars},
                log_path=log_dir / ENVS_DIR / "eval" / f"{name}.log",
            )

        if config.eval is not None:
            logger.info("Starting evaluator process")
            start_process(
                "evaluator",
                ["evaluator", "@", (config_dir / EVALUATOR_CONFIG).as_posix()],
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


def clean_stale_eval_artifacts(config: SFTConfig) -> None:
    """Remove eval artifacts a previous run left behind: weight checkpoints and rollout
    trace dirs — everything on a fresh start, steps past the resume step on resume.
    Without this the evaluator would replay stale checkpoints (and then skip the
    re-trained ones at the same steps), and the append-only trace files would mix two
    policies' rollouts under one step."""
    logger = setup_logger(config.log.level or "info")
    if os.environ.get("NEVER_CLEAN"):
        logger.warning("NEVER_CLEAN is set - keeping stale weight checkpoints; the evaluator may replay them")
        return
    resume_step = resolve_resume_step(config)
    weights_dir = get_weights_dir(get_ckpt_base(config))
    stale_steps = [step for step in get_all_ckpt_steps(weights_dir) if resume_step is None or step > resume_step]
    if stale_steps:
        logger.info(
            f"Deleting {len(stale_steps)} stale weight checkpoint(s) in {weights_dir} "
            f"({','.join(map(str, stale_steps))})"
        )
        for step in stale_steps:
            shutil.rmtree(get_step_path(weights_dir, step), ignore_errors=True)
    clean_future_steps(config.run_dir, resume_step if resume_step is not None else -1)


def sft(config: SFTConfig):
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
