import multiprocessing as mp
import os
import queue
import shutil
import signal
import socket
import tempfile
import threading
import uuid
from contextlib import ExitStack
from functools import partial
from multiprocessing.connection import Connection
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1 import set_base_sandbox_labels

from prime_rl.configs.env_worker_pool import EnvWorkerPoolConfig
from prime_rl.env import run_env_worker_pool, set_episode_hook
from prime_rl.env.sample_server import run_sample_server
from prime_rl.env.sandbox_owners import register_sandbox_pool
from prime_rl.env.worker import set_sample_server
from prime_rl.orchestrator.utils import setup_env_worker_pool_logging
from prime_rl.transports.payload import PayloadHook, StoreClient, StoreWriter, payload_host
from prime_rl.utils.config import check_mode, cli
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import set_save_root
from prime_rl.utils.utils import clean_exit

SAMPLE_SERVER_STARTUP_TIMEOUT_S = 120


def setup_worker(
    log_level: str | None,
    json_logging: bool,
    sandbox_labels: list[str],
    store: StoreWriter | StoreClient,
    sample_server: str,
    output_dir: Path,
) -> None:
    setup_env_worker_pool_logging(log_level, json_logging)
    set_base_sandbox_labels(sandbox_labels)
    set_save_root(output_dir)
    # Episode streams never carry training arrays; the hook keeps them in the replica's store.
    set_episode_hook(PayloadHook(store=store))
    set_sample_server(sample_server)


def setup_sample_server(log_level: str, json_logging: bool, output_dir: Path) -> None:
    setup_env_worker_pool_logging(log_level, json_logging)
    # Its store writes a rollout checkpoint's samples (`persist`), only under the run's directory.
    set_save_root(output_dir)


def start_sample_server(config: EnvWorkerPoolConfig, store_dir: Path | None) -> tuple[mp.Process, Connection, str, str]:
    """Spawn the replica's sample server, which holds its sample store (on disk under
    ``store_dir``, else in memory) and scores the samples its env workers keep. Returns the
    process, the pipe end whose closing stops it, the store's URL and the worker's address."""
    context = mp.get_context("spawn")
    death_read, death_write = context.Pipe(duplex=False)
    ready_read, ready_write = context.Pipe(duplex=False)
    process = context.Process(
        target=run_sample_server,
        kwargs=dict(
            store_dir=store_dir,
            hosts=payload_host(config.serve.address),
            scoring=config.serve.scoring.model_dump(),
            setup=partial(setup_sample_server, config.log.level, config.log.json_logging, config.output_dir),
            ready=ready_write,
            death_pipe=death_read,
        ),
        daemon=False,
    )
    process.start()
    death_read.close()
    ready_write.close()
    if not ready_read.poll(SAMPLE_SERVER_STARTUP_TIMEOUT_S):
        process.terminate()
        stop_sample_server(process)
        raise TimeoutError(f"The sample server did not start within {SAMPLE_SERVER_STARTUP_TIMEOUT_S}s")
    try:
        store_url, address = ready_read.recv()
    except EOFError:  # it exited without reporting ready
        process.join()
        raise RuntimeError(f"The sample server exited (code {process.exitcode}) before it started") from None
    return process, death_write, store_url, address


def stop_sample_server(process: mp.Process, timeout: float = 10) -> None:
    """End the sample server once its death pipe is closed or it got SIGTERM. It stops on its
    own then; one still running after ``timeout`` is stuck, and is killed so it never outlives
    the pool and its store."""
    process.join(timeout)
    if process.is_alive():
        get_logger().warning(f"The sample server did not stop within {timeout}s; killing it")
        process.kill()
        process.join()


def stop_pool_with(process: mp.Process, stopping: threading.Event, crashed: threading.Event) -> None:
    """Stop the pool if its sample server dies: its env workers can keep no samples. A crash
    sets ``crashed``, so the pool exits with an error once it has stopped."""
    process.join()
    if stopping.is_set():
        return
    # Exit code 0: it was told to stop (a launcher stopping the whole run signals every process).
    if process.exitcode == 0:
        get_logger().info("The sample server stopped; stopping the env worker pool")
    else:
        get_logger().error(f"The sample server exited (code {process.exitcode}); stopping the env worker pool")
        crashed.set()
    os.kill(os.getpid(), signal.SIGTERM)


def remove_dead_stores(store_dir: Path) -> None:
    """Remove the rollout stores of this node's pools that died without deleting theirs
    (``<host>-<pid>-<random>``)."""
    for path in store_dir.glob(f"{socket.gethostname()}-*"):
        pid = path.name.rsplit("-", 2)[1]
        if not Path("/proc", pid).exists():
            shutil.rmtree(path, ignore_errors=True)


def publish_address(addresses: queue.SimpleQueue, path: Path) -> None:
    """Publish the broker's bound address atomically for waiting clients."""
    address = addresses.get()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".address.tmp")
    tmp.write_text(address)
    tmp.replace(path)


@clean_exit
def run_pool(config: EnvWorkerPoolConfig):
    run_name = os.environ.get("PRL_RUN_NAME")
    sandbox_labels = [run_name] if run_name else []
    addresses: queue.SimpleQueue | None = None
    if config.address_file is not None:
        addresses = queue.SimpleQueue()
        threading.Thread(target=publish_address, args=(addresses, config.address_file), daemon=True).start()
    with ExitStack() as stack:
        attempt_dir = os.environ.get("PRL_ATTEMPT_CONFIG_DIR")
        if (
            attempt_dir
            and config.address_file
            and any(value.runtime.type == "prime" for _, value in config.env if isinstance(value, vf.AgentConfig))
        ):
            sandbox_labels.append(stack.enter_context(register_sandbox_pool(Path(attempt_dir))))
        directory = None
        if config.store_dir is not None:
            # The sample server serves the samples the env workers write, so they outlive
            # any one env worker.
            config.store_dir.mkdir(parents=True, exist_ok=True)
            remove_dead_stores(config.store_dir)
            prefix = f"{socket.gethostname()}-{os.getpid()}-"
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix=prefix, dir=config.store_dir)))
        sample_server, death_pipe, store_url, sample_server_address = start_sample_server(config, directory)
        stopping, crashed = threading.Event(), threading.Event()
        # Unwound last to first: mark the stop, then end the worker.
        stack.callback(stop_sample_server, sample_server)
        stack.callback(death_pipe.close)
        stack.callback(stopping.set)
        threading.Thread(target=stop_pool_with, args=(sample_server, stopping, crashed), daemon=True).start()
        store = StoreWriter(directory, store_url) if directory is not None else StoreClient(store_url)
        # Apply logging, sandbox labels, the payload hook and the sample server's address in
        # the broker and every env worker.
        run_env_worker_pool(
            pool=config.serve.pool,
            address=config.serve.address or "tcp://127.0.0.1:0",
            address_queue=addresses,
            worker_setup=partial(
                setup_worker,
                config.log.level,
                config.log.json_logging,
                sandbox_labels,
                store,
                sample_server_address,
                config.output_dir,
            ),
            config_data=config.env.model_dump(mode="json"),
            max_concurrent=config.serve.max_concurrent,
        )
        # The pool stops on the SIGTERM a crash sends, as on a clean shutdown; only this tells them apart.
        if crashed.is_set():
            raise RuntimeError(f"The sample server crashed (exit code {sample_server.exitcode})")


def main():
    """Main entry-point for the env worker pool. Run using `uv run env-worker-pool`"""
    # verifiers keys run-scoped state (creation limiters) by $VF_RUN_ID. Every launcher,
    # the Python entrypoints and the SLURM templates alike, hands env worker pools $PRL_RUN_ID;
    # a standalone pool is a run of its own.
    os.environ.setdefault("VF_RUN_ID", os.environ.get("PRL_RUN_ID") or uuid.uuid4().hex)
    check_mode("env-worker-pool")
    run_pool(cli(EnvWorkerPoolConfig))


if __name__ == "__main__":
    main()
