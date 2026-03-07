import copy
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread

import pynvml
import tomli_w

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import validate_output_dir
from prime_rl.utils.process import cleanup_processes, cleanup_threads, monitor_process
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import (
    get_free_port,
    get_log_dir,
    install_env,
    strip_env_version,
)


def write_subconfigs(config: RLConfig, output_dir: Path) -> None:
    """Write resolved subconfigs to disk as TOML files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "trainer.toml", "wb") as f:
        tomli_w.dump(config.trainer.model_dump(exclude_none=True, mode="json"), f)

    with open(output_dir / "orchestrator.toml", "wb") as f:
        tomli_w.dump(config.orchestrator.model_dump(exclude_none=True, mode="json"), f)

    if config.inference is not None:
        with open(output_dir / "inference.toml", "wb") as f:
            tomli_w.dump(config.inference.model_dump(exclude_none=True, mode="json"), f)

    teacher_inference = getattr(config, "teacher_inference", None)
    if teacher_inference is not None:
        with open(output_dir / "teacher_inference.toml", "wb") as f:
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


def detect_training_mode(config: RLConfig) -> tuple[str, dict[str, str]]:
    """Detect training mode from env actors and config.

    Detection axes:
      - Env: number of actors, whether they use different models
      - Config: whether [trainer.model.lora] is present

    Modes:
      - "single": 1 actor, or multiple actors with same model and no LoRA
      - "lora": multiple actors, same model, LoRA configured
      - "multi_model": multiple actors, different models (LoRA handled within)

    Returns:
        (mode, actor_models) where actor_models maps actor_id -> model_name
        (populated for multi_model and lora modes).
    """
    import verifiers as vf

    for env_config in config.orchestrator.env:
        install_env(strip_env_version(env_config.id))

    env_id = strip_env_version(config.orchestrator.env[0].id)
    env = vf.load_environment(env_id, **config.orchestrator.env[0].args)

    if not hasattr(env, "actors") or len(env.actors) <= 1:
        return "single", {}

    default_model = config.orchestrator.model.name
    actor_models = {}
    for actor_id in env.actors:
        agent = env.get_actor(actor_id)
        actor_models[actor_id] = agent.model or default_model

    unique_models = set(actor_models.values())
    has_lora = config.trainer.model.lora is not None

    if len(unique_models) > 1:
        return "multi_model", actor_models

    if has_lora:
        return "lora", actor_models

    return "single", actor_models


@dataclass
class ModelGroup:
    """A group of actors sharing the same base model."""

    index: int
    model_name: str
    actors: list[str]
    port: int
    group_dir: Path
    run_dir: Path


def setup_multi_model(
    config: RLConfig,
    actor_models: dict[str, str],
    logger,
) -> list[ModelGroup]:
    """Set up directory structure and config fields for multi-model training.

    Creates per-model-group directories, assigns inference ports,
    and populates actor_clients and actor_run_paths on the orchestrator config.

    Returns the list of model groups for process spawning.
    """
    # Group actors by model
    model_to_actors: dict[str, list[str]] = {}
    for actor_id, model in actor_models.items():
        model_to_actors.setdefault(model, []).append(actor_id)

    n_models = len(model_to_actors)
    logger.info(f"[MULTI-MODEL] {n_models} unique models: {list(model_to_actors.keys())}")

    has_lora = config.trainer.model.lora is not None
    actor_clients: dict[str, ClientConfig] = {}
    actor_run_paths: dict[str, str] = {}
    actor_lora_names: dict[str, str] = {}
    model_groups: list[ModelGroup] = []

    for i, (model_name, actors) in enumerate(model_to_actors.items()):
        group_dir = config.output_dir / f"model_{i}"
        port = get_free_port()

        if has_lora and len(actors) > 1:
            # Multi-LoRA within this group: per-actor run dirs
            for actor_id in actors:
                run_dir = group_dir / f"run_{actor_id}"
                control_dir = run_dir / "control"
                control_dir.mkdir(parents=True, exist_ok=True)
                actor_orch_config = {
                    "model": {"lora": {"name": f"run_{actor_id}"}},
                    "optim": {"lr": config.orchestrator.optim.lr},
                }
                with open(control_dir / "orch.toml", "wb") as f:
                    tomli_w.dump(actor_orch_config, f)
                actor_run_paths[actor_id] = str(run_dir)
                actor_lora_names[actor_id] = f"run_{actor_id}"

            group = ModelGroup(
                index=i,
                model_name=model_name,
                actors=actors,
                port=port,
                group_dir=group_dir,
                run_dir=group_dir / f"run_{actors[0]}",
            )
        else:
            # Single run for this group (1 actor per group, or no LoRA)
            run_dir = group_dir / "run_default"
            control_dir = run_dir / "control"
            control_dir.mkdir(parents=True, exist_ok=True)
            model_dict = {"name": model_name}
            if has_lora:
                model_dict["lora"] = {"name": "run_default"}
            actor_orch_config = {
                "model": model_dict,
                "optim": {"lr": config.orchestrator.optim.lr},
            }
            with open(control_dir / "orch.toml", "wb") as f:
                tomli_w.dump(actor_orch_config, f)

            for actor_id in actors:
                actor_run_paths[actor_id] = str(run_dir)
                if has_lora:
                    actor_lora_names[actor_id] = "run_default"

            group = ModelGroup(
                index=i,
                model_name=model_name,
                actors=actors,
                port=port,
                group_dir=group_dir,
                run_dir=run_dir,
            )

        model_groups.append(group)

        for actor_id in actors:
            actor_clients[actor_id] = ClientConfig(base_url=[f"http://localhost:{port}/v1"])

        logger.info(
            f"[MULTI-MODEL] Group {i}: model={model_name}, actors={actors}, "
            f"port={port}, group_dir={group_dir}"
        )

    config.orchestrator.actor_clients = actor_clients
    config.orchestrator.actor_run_paths = actor_run_paths
    config.orchestrator.actor_model_names = actor_models
    config.orchestrator.actor_lora_names = actor_lora_names if actor_lora_names else None

    # Inject per-actor vLLM endpoints into env args so env servers create per-actor clients
    actor_endpoints = {
        actor_id: client_cfg.base_url[0]
        for actor_id, client_cfg in actor_clients.items()
    }
    for env_config in config.orchestrator.env:
        env_config.args["actor_endpoints"] = actor_endpoints

    return model_groups


def _spawn_inference(
    config_path: Path,
    gpu_ids: list[int],
    log_file_path: Path,
    name: str,
    processes: list[Popen],
    monitor_threads: list[Thread],
    error_queue: list[Exception],
    stop_events: dict[str, Event],
    logger,
) -> None:
    """Spawn an inference server process and its monitor thread."""
    cmd = ["uv", "run", "inference", "@", config_path.as_posix()]
    logger.info(f"Starting {name} inference on GPU(s) {' '.join(map(str, gpu_ids))}")
    logger.debug(f"{name} inference command: {' '.join(cmd)}")
    with open(log_file_path, "w") as log_file:
        proc = Popen(
            cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids))},
            stdout=log_file,
            stderr=log_file,
        )
    processes.append(proc)

    stop_event = Event()
    stop_events[name] = stop_event
    thread = Thread(target=monitor_process, args=(proc, stop_event, error_queue, name), daemon=True)
    thread.start()
    monitor_threads.append(thread)


def _spawn_trainer(
    config_path: Path,
    gpu_ids: list[int],
    output_dir: Path,
    log_file_path: Path,
    name: str,
    start_command: list[str],
    processes: list[Popen],
    monitor_threads: list[Thread],
    error_queue: list[Exception],
    stop_events: dict[str, Event],
    logger,
) -> None:
    """Spawn a trainer process (torchrun) and its monitor thread."""
    cmd = [
        "uv", "run", "env",
        "PYTHONUNBUFFERED=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "torchrun",
        f"--rdzv-endpoint=localhost:{get_free_port()}",
        f"--rdzv-id={uuid.uuid4().hex}",
        f"--log-dir={output_dir / 'torchrun'}",
        "--local-ranks-filter=0",
        "--redirect=3",
        "--tee=3",
        f"--nproc-per-node={len(gpu_ids)}",
        "-m", "prime_rl.trainer.rl.train",
        "@", config_path.as_posix(),
    ]
    logger.info(f"Starting {name} trainer on GPU(s) {' '.join(map(str, gpu_ids))}")
    logger.debug(f"{name} trainer command: {' '.join(cmd)}")
    with open(log_file_path, "w") as log_file:
        proc = Popen(
            cmd,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "LOGURU_FORCE_COLORS": "1",
                "WANDB_PROGRAM": "uv run rl",
                "WANDB_ARGS": json.dumps(start_command),
            },
            stdout=log_file,
            stderr=log_file,
        )
    processes.append(proc)

    stop_event = Event()
    stop_events[name] = stop_event
    thread = Thread(target=monitor_process, args=(proc, stop_event, error_queue, name), daemon=True)
    thread.start()
    monitor_threads.append(thread)


def rl_local_multi_model(
    config: RLConfig,
    model_groups: list[ModelGroup],
    infer_gpu_ids: list[int],
    trainer_gpu_ids: list[int],
    logger,
):
    """Run multi-model training: N inference servers, 1 orchestrator, N trainers."""
    n_models = len(model_groups)

    # Validate GPU counts are divisible by number of model groups
    if len(infer_gpu_ids) % n_models != 0:
        raise ValueError(
            f"num_infer_gpus ({len(infer_gpu_ids)}) must be divisible by "
            f"{n_models} model groups"
        )
    if len(trainer_gpu_ids) % n_models != 0:
        raise ValueError(
            f"num_train_gpus ({len(trainer_gpu_ids)}) must be divisible by "
            f"{n_models} model groups"
        )

    infer_per_model = len(infer_gpu_ids) // n_models
    train_per_model = len(trainer_gpu_ids) // n_models

    start_command = sys.argv
    log_dir = get_log_dir(config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    config_dir = Path(".pydantic_config") / uuid.uuid4().hex
    config_dir.mkdir(parents=True, exist_ok=True)

    # Write orchestrator config (with actor_clients and actor_run_paths populated)
    with open(config_dir / "orchestrator.toml", "wb") as f:
        tomli_w.dump(config.orchestrator.model_dump(exclude_none=True, mode="json"), f)

    # Write per-model-group inference and trainer configs
    for group in model_groups:
        # Inference config
        if config.inference is not None:
            infer_config = copy.deepcopy(config.inference)
            infer_config.model.name = group.model_name
            infer_config.server.port = group.port
            # Each model group gets its share of inference GPUs
            infer_config.parallel.dp = infer_per_model // infer_config.parallel.tp
            with open(config_dir / f"inference_{group.index}.toml", "wb") as f:
                tomli_w.dump(infer_config.model_dump(exclude_none=True, mode="json"), f)

        # Trainer config for this model group
        trainer_config = copy.deepcopy(config.trainer)
        trainer_config.model.name = group.model_name
        trainer_config.output_dir = group.group_dir
        has_lora = config.trainer.model.lora is not None
        if has_lora and len(group.actors) > 1:
            trainer_config.max_concurrent_runs = len(group.actors)
            trainer_config.pack_full_step = True
        else:
            trainer_config.max_concurrent_runs = 1
            trainer_config.pack_full_step = False
        with open(config_dir / f"trainer_{group.index}.toml", "wb") as f:
            tomli_w.dump(trainer_config.model_dump(exclude_none=True, mode="json"), f)

    # Start processes
    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    stop_events: dict[str, Event] = {}

    try:
        # Spawn N inference servers
        if config.inference is not None:
            for group in model_groups:
                gpu_slice = infer_gpu_ids[group.index * infer_per_model:(group.index + 1) * infer_per_model]
                _spawn_inference(
                    config_path=config_dir / f"inference_{group.index}.toml",
                    gpu_ids=gpu_slice,
                    log_file_path=log_dir / f"inference_{group.index}.stdout",
                    name=f"inference_{group.index}",
                    processes=processes,
                    monitor_threads=monitor_threads,
                    error_queue=error_queue,
                    stop_events=stop_events,
                    logger=logger,
                )

        # Spawn orchestrator
        orchestrator_cmd = [
            "uv", "run", "orchestrator", "@",
            (config_dir / "orchestrator.toml").as_posix(),
        ]
        logger.info("Starting orchestrator process")
        with open(log_dir / "orchestrator.stdout", "w") as log_file:
            orchestrator_process = Popen(
                orchestrator_cmd,
                stdout=log_file,
                stderr=log_file,
                env={
                    **os.environ,
                    "LOGURU_FORCE_COLORS": "1",
                    "WANDB_PROGRAM": "uv run rl",
                    "WANDB_ARGS": json.dumps(start_command),
                },
            )
        processes.append(orchestrator_process)
        stop_event = Event()
        stop_events["orchestrator"] = stop_event
        orch_monitor = Thread(
            target=monitor_process,
            args=(orchestrator_process, stop_event, error_queue, "orchestrator"),
            daemon=True,
        )
        orch_monitor.start()
        monitor_threads.append(orch_monitor)

        # Spawn N trainers
        for group in model_groups:
            gpu_slice = trainer_gpu_ids[group.index * train_per_model:(group.index + 1) * train_per_model]
            _spawn_trainer(
                config_path=config_dir / f"trainer_{group.index}.toml",
                gpu_ids=gpu_slice,
                output_dir=group.group_dir,
                log_file_path=log_dir / f"trainer_{group.index}.stdout",
                name=f"trainer_{group.index}",
                start_command=start_command,
                processes=processes,
                monitor_threads=monitor_threads,
                error_queue=error_queue,
                stop_events=stop_events,
                logger=logger,
            )

        logger.success(
            f"[MULTI-MODEL] Startup complete: {n_models} inference servers, "
            f"1 orchestrator, {n_models} trainers"
        )

        # Wait for orchestrator and all trainers to finish
        critical_events = ["orchestrator"] + [f"trainer_{g.index}" for g in model_groups]
        while not all(stop_events.get(name, Event()).is_set() for name in critical_events):
            if error_queue:
                logger.error(f"Error: {error_queue[0]}")
                logger.error("Terminating all processes...")
                cleanup_threads(monitor_threads)
                cleanup_processes(processes)
                sys.exit(1)
            time.sleep(1)

        # Check exit codes
        if orchestrator_process.returncode != 0:
            logger.error(f"Orchestrator failed with exit code {orchestrator_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        logger.success("Multi-model RL training finished!")
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


def rl_local(config: RLConfig):
    # Setup logger
    logger = setup_logger(
        config.log.level or "info",
        log_file=config.output_dir / "logs" / "rl.log" if config.log.file else None,
        json_logging=config.log.json_logging,
    )

    # If dump_config is set, write resolved subconfigs and exit early
    if config.dump_config is not None:
        logger.warning(
            "--dump-config is set. No RL training will be started. Only writing resolved subconfigs to disk."
        )
        write_subconfigs(config, config.dump_config)
        logger.info(f"Dumping resolved subconfigs to {config.dump_config}")
        logger.info(f"  Wrote trainer config to {config.dump_config / 'trainer.toml'}")
        logger.info(f"  Wrote orchestrator config to {config.dump_config / 'orchestrator.toml'}")
        if config.inference is not None:
            logger.info(f"  Wrote inference config to {config.dump_config / 'inference.toml'}")
        if config.teacher_inference is not None:
            logger.info(f"  Wrote teacher inference config to {config.dump_config / 'teacher_inference.toml'}")
        logger.success(f"Config dump complete. Files written to {config.dump_config}")
        logger.warning("To start an RL run, remove --dump-config from your command.")
        return

    # Derive GPU IDs from deployment config
    assert config.deployment.type == "single_node"
    gpu_offset = 0
    num_infer_gpus = config.deployment.num_infer_gpus if config.inference is not None else 0
    infer_gpu_ids = list(range(gpu_offset, gpu_offset + num_infer_gpus))
    gpu_offset += num_infer_gpus
    trainer_gpu_ids = list(range(gpu_offset, gpu_offset + config.deployment.num_train_gpus))
    gpu_offset += config.deployment.num_train_gpus
    num_teacher_gpus = config.deployment.num_teacher_gpus or 0
    teacher_gpu_ids = list(range(gpu_offset, gpu_offset + num_teacher_gpus)) if num_teacher_gpus > 0 else []

    start_command = sys.argv
    logger.info("Starting RL run")
    logger.debug(f"RL start command: {' '.join(start_command)}")

    # Check for existing processes on GPUs
    all_gpu_ids = list(set(infer_gpu_ids + trainer_gpu_ids + teacher_gpu_ids))
    check_gpus_available(all_gpu_ids)

    # Detect training mode from env actors + config
    mode, actor_models = detect_training_mode(config)
    logger.info(f"Training mode: {mode}")

    # Override config for multi-agent modes (can't be done at parse time since env isn't loaded)
    if mode in ("lora", "multi_model"):
        config.orchestrator.output_dir = config.output_dir / "orchestrator"
    if mode == "lora":
        config.trainer.pack_full_step = True
        config.trainer.max_concurrent_runs = len(actor_models)

        # Create per-actor run dirs so trainer finds them immediately at startup
        for actor_id in actor_models:
            run_name = f"run_{actor_id}"
            control_dir = config.output_dir / run_name / "control"
            control_dir.mkdir(parents=True, exist_ok=True)
            actor_orch_config = {
                "model": {"lora": {"name": run_name}},
                "optim": {"lr": config.orchestrator.optim.lr},
            }
            with open(control_dir / "orch.toml", "wb") as f:
                tomli_w.dump(actor_orch_config, f)

    if mode == "multi_model":
        model_groups = setup_multi_model(config, actor_models, logger)
        rl_local_multi_model(config, model_groups, infer_gpu_ids, trainer_gpu_ids, logger)
        return

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

    # Write all resolved subconfigs to disk
    config_dir = Path(".pydantic_config") / uuid.uuid4().hex
    write_subconfigs(config, config_dir)

    # Start processes
    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    stop_events: dict[str, Event] = {}

    try:
        # Optionally, start inference process
        if config.inference:
            inference_cmd = ["uv", "run", "inference", "@", (config_dir / "inference.toml").as_posix()]
            logger.info(f"Starting inference process on GPU(s) {' '.join(map(str, infer_gpu_ids))}")
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
            logger.warning(
                "No inference config specified, skipping starting inference server. Make sure your inference server is running."
            )

        # Optionally, start teacher inference process
        if config.teacher_inference:
            if not teacher_gpu_ids:
                raise ValueError(
                    "teacher_inference is configured but deployment.num_teacher_gpus is not set. "
                    "Either set deployment.num_teacher_gpus to start a teacher inference server, "
                    "or omit teacher_inference and configure orchestrator.teacher_model to use an existing server."
                )

            teacher_inference_cmd = ["uv", "run", "inference", "@", (config_dir / "teacher_inference.toml").as_posix()]
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
            (config_dir / "orchestrator.toml").as_posix(),
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
            (config_dir / "trainer.toml").as_posix(),
        ]
        logger.info(f"Starting trainer process on GPU(s) {' '.join(map(str, trainer_gpu_ids))}")
        logger.debug(f"Training start command: {' '.join(trainer_cmd)}")
        with open(log_dir / "trainer.stdout", "w") as log_file:
            trainer_process = Popen(
                trainer_cmd,
                env={
                    **os.environ,
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


def render_slurm_script(config: RLConfig, config_dir: Path) -> tuple[str, str]:
    """Render the SLURM script template. Returns (script, log_message)."""
    from jinja2 import Environment, FileSystemLoader

    assert config.slurm is not None
    assert config.slurm.template_path is not None

    env = Environment(loader=FileSystemLoader(config.slurm.template_path.parent), keep_trailing_newline=True)
    template = env.get_template(config.slurm.template_path.name)

    log_dir = config.output_dir / "logs"

    if config.deployment.type == "single_node":
        import tomli_w

        config_path = config_dir / "rl.toml"
        config_dict = config.model_dump(exclude={"slurm"}, exclude_none=True, mode="json")
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, "wb") as f:
            tomli_w.dump(config_dict, f)

        script = template.render(
            config_path=config_path,
            job_name=config.slurm.job_name,
            project_dir=config.slurm.project_dir,
            partition=config.slurm.partition,
            output_dir=config.output_dir,
            gpus_per_node=config.deployment.gpus_per_node,
        )
        log_message = (
            f"Logs:\n"
            f"  Trainer:       tail -F {log_dir}/trainer.stdout\n"
            f"  Orchestrator:  tail -F {log_dir}/orchestrator.stdout\n"
            f"  Inference:     tail -F {log_dir}/inference.stdout"
        )
    else:
        script = template.render(
            config_dir=config_dir,
            job_name=config.slurm.job_name,
            project_dir=config.slurm.project_dir,
            partition=config.slurm.partition,
            output_dir=config.output_dir,
            orchestrator_output_dir=config.orchestrator.output_dir,
            num_train_nodes=config.deployment.num_train_nodes,
            num_infer_nodes=config.deployment.num_infer_nodes,
            num_teacher_nodes=config.deployment.num_teacher_nodes,
            gpus_per_node=config.deployment.gpus_per_node,
        )
        slurm_log_dir = config.output_dir / "slurm"
        log_message = (
            f"Logs:\n"
            f"  Trainer:       tail -F {slurm_log_dir}/latest_train_node_rank_0.log\n"
            f"  Orchestrator:  tail -F {slurm_log_dir}/latest_orchestrator.log\n"
            f"  Inference:     tail -F {slurm_log_dir}/latest_infer_node_rank_0.log"
        )

    return script, log_message


def rl_slurm(config: RLConfig):
    assert config.slurm is not None

    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    config_dir = config.output_dir / "configs"

    if config.deployment.type == "multi_node":
        write_subconfigs(config, config_dir)
        logger.info(f"Wrote subconfigs to {config_dir}")

    script, log_message = render_slurm_script(config, config_dir)
    script_path = config.output_dir / "rl.sbatch"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    logger.info(f"Wrote SLURM script to {script_path}")

    if config.slurm.dry_run:
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
    validate_output_dir(config.output_dir, resuming=resuming, clean=clean)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.slurm is not None:
        rl_slurm(config)
    else:
        rl_local(config)


def main():
    rl(parse_argv(RLConfig))


if __name__ == "__main__":
    main()
