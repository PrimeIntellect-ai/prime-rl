import asyncio
import os
import shutil
import time
from pathlib import Path

from prime_rl.utils.logger import get_logger


def get_log_dir(output_dir: Path) -> Path:
    return output_dir / "logs"


def create_attempt_log_dir(run_dir: Path) -> Path:
    """Create ``logs/attempt_<n>`` for this launch attempt and repoint ``logs/latest`` to it.

    Every launch — fresh or resumed — gets its own numbered log directory, so a resume
    never overwrites an earlier attempt's logs. Returns the attempt directory."""
    logs_dir = get_log_dir(run_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    attempts = (
        int(p.name.removeprefix("attempt_"))
        for p in logs_dir.glob("attempt_*")
        if p.name.removeprefix("attempt_").isdigit()
    )
    attempt_dir = logs_dir / f"attempt_{1 + max(attempts, default=0)}"
    attempt_dir.mkdir()
    # Atomically repoint the relative ``latest`` symlink: create a temp link, then rename.
    tmp_link = logs_dir / f".{attempt_dir.name}"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    os.symlink(attempt_dir.name, tmp_link)
    os.replace(tmp_link, logs_dir / "latest")
    return attempt_dir


def latest_log_dir(run_dir: Path) -> Path:
    """The current attempt's log directory, via the ``logs/latest`` symlink."""
    return get_log_dir(run_dir) / "latest"


def format_log_message(
    log_dir: Path,
    trainer: bool = False,
    orchestrator: bool = False,
    evals: bool = False,
    inference: bool = False,
    job_log: bool = False,
    train_env_names: list[str] | None = None,
    eval_env_names: list[str] | None = None,
    num_train_nodes: int = 1,
    num_infer_nodes: int = 0,
) -> str:
    """Format a log message showing where to find all log files."""
    col = 18
    i1 = " " * 2
    i2 = " " * 3
    i3 = " " * 4
    max_name = col - 4

    log_lines: list[str] = []
    if job_log:
        log_lines.append(f"{i1}{'Job:':<{col}}tail -F {log_dir.parent.parent}/job_*.log")
    if trainer:
        log_lines.append(f"{i1}{'Trainer:':<{col}}tail -F {log_dir}/trainer.log")
        if num_train_nodes > 1:
            log_lines.append(f"{i2}{'All nodes:':<{col - 1}}tail -F {log_dir}/trainer/node_*.log")
        log_lines.append(f"{i2}{'All ranks:':<{col - 1}}tail -F {log_dir}/trainer/torchrun/*/*/*/*.log")
    if orchestrator:
        log_lines.append(f"{i1}{'Orchestrator:':<{col}}tail -F {log_dir}/orchestrator.log")
    if evals:
        log_lines.append(f"{i1}{'Evals:':<{col}}tail -F {log_dir}/evals.log")
    if inference:
        log_lines.append(f"{i1}{'Inference:':<{col}}tail -F {log_dir}/inference.log")
        if num_infer_nodes > 1:
            log_lines.append(f"{i2}{'All nodes:':<{col - 1}}tail -F {log_dir}/inference/node_*.log")
    if train_env_names or eval_env_names:
        env_log_dir = log_dir / "envs"
        log_lines.append(f"{i1}{'Envs:':<{col}}tail -F {env_log_dir}/*/*.log")
        if train_env_names:
            log_lines.append(f"{i2}{'Train:':<{col - 1}}tail -F {env_log_dir}/train/*.log")
            for name in train_env_names:
                short = name if len(name) <= max_name else name[: max_name - 3] + "..."
                log_lines.append(f"{i3}{f'{short}:':<{col - 2}}tail -F {env_log_dir}/train/{name}.log")
        if eval_env_names:
            log_lines.append(f"{i2}{'Eval:':<{col - 1}}tail -F {env_log_dir}/eval/*.log")
            for name in eval_env_names:
                short = name if len(name) <= max_name else name[: max_name - 3] + "..."
                log_lines.append(f"{i3}{f'{short}:':<{col - 2}}tail -F {env_log_dir}/eval/{name}.log")
    return "Logs:\n" + "\n".join(log_lines)


def get_config_dir(output_dir: Path) -> Path:
    """Resolved per-component config dumps (JSON). The launch TOML copy lives one
    level up, at `configs/<entrypoint>.toml`."""
    return output_dir / "configs" / "resolved"


def write_launch_toml(run_dir: Path, name: str) -> None:
    """Copy the launch `@` TOML file(s) verbatim to `configs/<name>.toml`."""
    import sys

    argv = sys.argv[1:]
    paths = []
    for i, arg in enumerate(argv):
        # root config references only: `@ file`; a `--flag @ file` / `--flag @file`
        # is a nested reference and belongs under its flag, not in the launch copy
        if arg == "@" and i + 1 < len(argv) and (i == 0 or not argv[i - 1].startswith("--")):
            paths.append(Path(argv[i + 1]))
    texts = [p.read_text() for p in paths if p.suffix == ".toml" and p.is_file()]
    if not texts:
        return
    if len(texts) > 1:
        texts = [f"# @ {p}\n{t}" for p, t in zip(paths, texts)]
    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f"{name}.toml").write_text("\n".join(texts))


def get_ckpt_dir(output_dir: Path) -> Path:
    return output_dir / "checkpoints"


def get_rollout_dir(output_dir: Path) -> Path:
    return output_dir / "rollouts"


def get_eval_dir(output_dir: Path) -> Path:
    return output_dir / "evals"


def get_broadcast_dir(output_dir: Path) -> Path:
    return output_dir / "broadcasts"


def get_step_path(path: Path, step: int) -> Path:
    return path / f"step_{step}"


def get_all_ckpt_steps(ckpt_dir: Path) -> list[int]:
    """Gets all checkpoint steps from the checkpoint directory, sorted in ascending order."""
    step_dirs = list(ckpt_dir.glob("step_*"))
    return sorted([int(step_dir.name.split("_")[-1]) for step_dir in step_dirs])


def resolve_latest_ckpt_step(ckpt_dir: Path) -> int | None:
    """Gets the latest checkpoint step from the checkpoint directory. Returns None if no checkpoints are found."""
    steps = get_all_ckpt_steps(ckpt_dir)
    if len(steps) == 0:
        logger = get_logger()
        logger.warning(f"No checkpoints found in {ckpt_dir}. Starting from scratch.")
        return None
    latest_step = steps[-1]
    logger = get_logger()
    logger.info(f"Found latest checkpoint in {ckpt_dir}: {latest_step}")
    return latest_step


def has_checkpoints(output_dir: Path) -> bool:
    """Check if the output directory contains any checkpoints."""
    ckpt_dir = get_ckpt_dir(output_dir)
    return ckpt_dir.exists() and any(ckpt_dir.iterdir())


# Launcher artifacts that may exist in a run directory before training starts: resolved
# configs and the SLURM script/job log (a submitted job re-invokes the entrypoint inside
# the run directory). Everything else is treated as artifacts of a previous run.
LAUNCHER_ARTIFACTS = ("configs", "rl.sbatch", "sft.sbatch", "job_*.log")


def has_run_artifacts(run_dir: Path) -> bool:
    """Check if the run directory contains artifacts beyond what the launcher pre-writes."""
    if not run_dir.exists():
        return False
    launcher_entries = {entry for pattern in LAUNCHER_ARTIFACTS for entry in run_dir.glob(pattern)}
    return any(entry not in launcher_entries for entry in run_dir.iterdir())


def validate_run_dir(
    run_dir: Path, *, output_dir: Path, resuming: bool, clean: bool, ckpt_output_dir: Path | None = None
) -> None:
    """Validate the run directory before training starts.

    Raises if the run directory was already used by a previous run, unless explicitly
    resuming or opting into cleaning — a second run writing into the same run directory
    would overwrite or interleave with the first run's artifacts.

    When ckpt_output_dir is set, checkpoints live there instead of under
    run_dir, so the guard and clean logic check both locations.
    """
    if resuming:
        return
    if clean:
        if not run_dir.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError(f"clean requires the run directory ({run_dir}) to remain under output_dir ({output_dir})")
        logger = get_logger()
        dirs_to_clean = [run_dir]
        if ckpt_output_dir is not None and ckpt_output_dir != run_dir:
            dirs_to_clean.append(ckpt_output_dir)
        for d in dirs_to_clean:
            if d.exists():
                logger.warning(f"Cleaning existing directory: {d}")
                shutil.rmtree(d)
        return
    blocked = None
    if has_run_artifacts(run_dir):
        blocked = f"Run directory '{run_dir}' already contains artifacts from a previous run."
    elif ckpt_output_dir is not None and ckpt_output_dir != run_dir and has_checkpoints(ckpt_output_dir):
        blocked = f"Checkpoint directory '{ckpt_output_dir}' already contains checkpoints from a previous run."
    if blocked:
        raise FileExistsError(
            f"{blocked} "
            f"To resume the latest step of the previous run, pass --resume (or --resume.step N). "
            f"To delete the existing directory and start fresh, set clean=true or --clean via CLI. "
            f"Otherwise use a unique run name (run.name or --run.name via CLI) or output_dir for this run."
        )


def clean_future_steps(output_dir: Path, resume_step: int) -> None:
    """Remove stale rollouts, broadcasts, and traces past ``resume_step``.

    Pass ``resume_step=-1`` to wipe every step directory (fresh runs).
    """
    dirs = [
        get_rollout_dir(output_dir),
        get_broadcast_dir(output_dir),
    ]

    for directory in dirs:
        steps_to_delete = [step for step in get_all_ckpt_steps(directory) if step > resume_step]
        if not steps_to_delete:
            continue
        get_logger().info(
            f"Deleting {len(steps_to_delete)} step directories in {directory} ({','.join(map(str, steps_to_delete))})"
        )
        for step in steps_to_delete:
            shutil.rmtree(get_step_path(directory, step))


def sync_wait_for_path(path: Path, interval: int = 1, log_interval: int = 10) -> None:
    logger = get_logger()
    wait_time = 0
    logger.debug(f"Waiting for path `{path}`")
    while True:
        if path.exists():
            logger.debug(f"Found path `{path}`")
            break
        if wait_time % log_interval == 0 and wait_time > 0:  # Every log_interval seconds
            logger.debug(f"Waiting for path `{path}` for {wait_time} seconds")
        time.sleep(interval)
        wait_time += interval


async def wait_for_path(path: Path, interval: int = 1, log_interval: int = 10) -> None:
    logger = get_logger()
    wait_time = 0
    logger.debug(f"Waiting for path `{path}`")
    while True:
        if path.exists():
            logger.debug(f"Found path `{path}`")
            break
        if wait_time % log_interval == 0 and wait_time > 0:  # Every log_interval seconds
            logger.debug(f"Waiting for path `{path}` for {wait_time} seconds")
        await asyncio.sleep(interval)
        wait_time += interval
