"""The trainer's side of the in-run eval: after ``train()`` returned (its model, optimizer and data loaders are out of
scope, the process groups destroyed), every rank lets go of the GPUs; the ranks other than each node's
``local_rank 0`` exit 0, and ``local_rank 0`` becomes the node's eval process (``node``).

torchrun waits until all its workers have exited and only fails the node on a non-zero exit, so early clean exits
keep the node (and its pod) up for the eval process. With ``handoff = "exec"`` that process replaces the trainer
process (same pid, so torchrun keeps watching it; the trainer's GPU and host memory are released by the exec).
"""

import gc
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import psutil

from prime_rl.trainer.sft.inrun_eval.plan import build_plan
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger

NODE_MODULE = "prime_rl.trainer.sft.inrun_eval.node"


def stop_children(timeout: float = 30.0) -> None:
    """Terminate this process's descendants (data loader workers, compile workers): an exec would orphan them with
    the trainer's memory, and they would never notice (the pid stays the same)."""
    try:
        children = psutil.Process().children(recursive=True)
    except psutil.Error:
        return
    for child in children:
        try:
            child.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(children, timeout=timeout)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass


def release_gpu() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def close_inherited_fds() -> None:
    """Make every fd above stderr close-on-exec (driver / NCCL sockets opened by C code may lack it); unlike
    closing them, this keeps the process usable should the exec fail."""
    try:
        fds = [int(fd) for fd in os.listdir("/proc/self/fd")]
    except OSError:
        fds = list(range(3, 4096))
    for fd in fds:
        if fd > 2:
            try:
                os.set_inheritable(fd, False)
            except OSError:
                pass


def node_files(config) -> tuple[list[str] | None, list[str]]:
    """The export files this node keeps after the eval (its share of the world-wide upload split) and the small
    files only the master keeps, from the base checkpoint's layout (as ``hf_export.prepare`` splits it)."""
    from prime_rl.trainer.sft.hf_export import BASE_FILES, assign_files, files_by_node

    world = get_world()
    cfg = config.hf_export
    if cfg is None:
        return None, []
    base_dir = Path(cfg.base_dir) if cfg.base_dir is not None else Path(config.model.name)
    try:
        weight_map = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
        sizes = {name: (base_dir / name).stat().st_size for name in sorted(set(weight_map.values()))}
    except Exception as exc:
        get_logger().warning(f"in-run eval: cannot read the export layout from {base_dir} ({exc!r}): no dedupe")
        return None, []
    split = files_by_node(assign_files(sizes, world.world_size), world.local_world_size)
    return split.get(world.rank // world.local_world_size, []), [*BASE_FILES, "README.md", "export_report.json"]


def master_addr() -> str:
    return os.environ.get("MASTER_ADDR") or socket.gethostname()


def credential_check(config) -> tuple[str | None, dict]:
    """The Prime team id the eval uses and what the trainer pod has: only whether the platform's per-run
    ``PRIME_API_KEY`` is set (never its value), and where the team id comes from."""
    from prime_rl.trainer.sft.inrun_eval.harness import credentials

    fallback = (getattr(config, "env_vars", None) or {}).get("PRIME_TEAM_ID")
    creds = credentials(config.inrun_eval.harness, dict(os.environ), fallback)
    return creds["prime_team_id"], creds


def start(config, result) -> None:
    """Never returns on a rank of a node (they exit 0 or exec into the node's eval process)."""
    logger = get_logger()
    world = get_world()
    cfg = config.inrun_eval
    node = world.rank // world.local_world_size
    stop_children()
    release_gpu()
    if world.local_rank != 0:
        logger.info(f"in-run eval: rank {world.rank} hands its GPU to node {node}'s eval process and exits")
        _exit(0)

    plan = None
    try:
        team, creds = credential_check(config)
        logger.info(
            f"in-run eval: node {node}: PRIME_API_KEY present: {'yes' if creds['prime_api_key_present'] else 'NO'}; "
            f"PRIME_TEAM_ID: {creds['prime_team_id'] or 'unset'} ({creds['prime_team_id_source']})"
            + (f"; the eval will fail: {creds['problem']}" if creds["problem"] else "")
        )
        keep, master_only = node_files(config) if cfg.model_dir is None else (None, [])
        cleanup = config.hf_export.cleanup if config.hf_export is not None else None
        plan = build_plan(
            cfg,
            run_dir=config.run_dir,
            exports=result.exports,
            node=node,
            num_nodes=world.num_nodes,
            gpus_per_node=config.deployment.gpus_per_node,
            master_addr=master_addr(),
            node_files=keep,
            master_files=master_only,
            delete_pushed_exports=bool(cleanup and cleanup.pushed_exports),
            prime_team_id=team,
            credentials={k: v for k, v in creds.items() if k != "problem"},
        )
        work = Path(plan.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        plan_path = work / "plan.json"
        plan_path.write_text(plan.to_json())
    except Exception as exc:
        # The exports are done (and pushed): a broken eval setup must not fail the run.
        logger.opt(exception=True).error(f"in-run eval: could not plan the eval, skipping it: {exc!r}")
        _exit(0)

    logger.info(
        f"in-run eval: node {node} hands over to its eval process ({len(plan.checkpoints)} checkpoint(s): "
        f"{[c.name for c in plan.checkpoints]}, aliases {plan.aliases}, skipped {plan.skipped}; "
        f"rounds {plan.rounds}; "
        f"{'no cap' if cfg.budget_hours is None else f'safety cap {cfg.budget_hours} h'})"
    )
    argv = [sys.executable, "-m", NODE_MODULE, str(plan_path)]
    _flush(logger)
    if cfg.handoff == "exec":
        close_inherited_fds()
        try:
            os.execv(sys.executable, argv)
        except OSError as exc:
            logger.error(f"in-run eval: exec failed ({exc!r}), running it as a child instead")
    code = subprocess.call(argv, close_fds=True)
    logger.info(f"in-run eval: node {node}'s eval process exited with {code}")
    _exit(0)


def _flush(logger) -> None:
    try:
        logger.complete()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()


def _exit(code: int) -> None:
    _flush(get_logger())
    os._exit(code)
