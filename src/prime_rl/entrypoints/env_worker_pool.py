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

from verifiers.v1.runtimes import set_base_sandbox_labels

from prime_rl.configs.env_worker_pool import EnvWorkerPoolConfig
from prime_rl.env import run_env_worker_pool, set_episode_hook
from prime_rl.env.algorithm import run_algorithm_worker
from prime_rl.env.worker import set_algorithm_worker
from prime_rl.orchestrator.utils import setup_env_worker_pool_logging
from prime_rl.transports.payload import PayloadHook, StoreClient, StoreWriter, payload_host
from prime_rl.utils.config import check_mode, cli
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import clean_exit

ALGORITHM_WORKER_STARTUP_TIMEOUT_S = 120


def setup_worker(
    log_level: str | None,
    json_logging: bool,
    sandbox_labels: list[str],
    store: StoreWriter | StoreClient,
    algorithm_worker: str,
) -> None:
    setup_env_worker_pool_logging(log_level, json_logging)
    set_base_sandbox_labels(sandbox_labels)
    # Episode streams never carry training arrays; the hook keeps them in the replica's store.
    set_episode_hook(PayloadHook(store=store))
    set_algorithm_worker(algorithm_worker)


def start_algorithm_worker(
    config: EnvWorkerPoolConfig, store_dir: Path | None
) -> tuple[mp.Process, Connection, str, str]:
    """Spawn the replica's algorithm worker, which holds its sample store (on disk under
    ``store_dir``, else in memory) and scores the samples its env workers keep. Returns the
    process, the pipe end whose closing stops it, the store's URL and the worker's address."""
    context = mp.get_context("spawn")
    death_read, death_write = context.Pipe(duplex=False)
    ready_read, ready_write = context.Pipe(duplex=False)
    process = context.Process(
        target=run_algorithm_worker,
        kwargs=dict(
            store_dir=store_dir,
            hosts=payload_host(config.serve.address),
            scoring=config.serve.scoring.model_dump(),
            setup=partial(setup_env_worker_pool_logging, config.log.level, config.log.json_logging),
            ready=ready_write,
            death_pipe=death_read,
        ),
        daemon=False,
    )
    process.start()
    death_read.close()
    ready_write.close()
    if not ready_read.poll(ALGORITHM_WORKER_STARTUP_TIMEOUT_S):
        process.terminate()
        raise TimeoutError(f"The algorithm worker did not start within {ALGORITHM_WORKER_STARTUP_TIMEOUT_S}s")
    store_url, address = ready_read.recv()
    return process, death_write, store_url, address


def stop_pool_with(process: mp.Process, stopping: threading.Event) -> None:
    """Stop the pool if its algorithm worker dies: its env workers can keep no samples."""
    process.join()
    if stopping.is_set():
        return
    # Exit code 0: it was told to stop (a launcher stopping the whole run signals every process).
    log = get_logger().info if process.exitcode == 0 else get_logger().error
    log(f"The algorithm worker exited (code {process.exitcode}); stopping the env worker pool")
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
        directory = None
        if config.store_dir is not None:
            # The algorithm worker serves the samples the env workers write, so they outlive
            # any one env worker.
            config.store_dir.mkdir(parents=True, exist_ok=True)
            remove_dead_stores(config.store_dir)
            prefix = f"{socket.gethostname()}-{os.getpid()}-"
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix=prefix, dir=config.store_dir)))
        algorithm_worker, death_pipe, store_url, algorithm_address = start_algorithm_worker(config, directory)
        stopping = threading.Event()
        # Unwound last to first: mark the stop, then end the worker.
        stack.callback(algorithm_worker.join, 10)
        stack.callback(death_pipe.close)
        stack.callback(stopping.set)
        threading.Thread(target=stop_pool_with, args=(algorithm_worker, stopping), daemon=True).start()
        store = StoreWriter(directory, store_url) if directory is not None else StoreClient(store_url)
        # Apply logging, sandbox labels, the payload hook and the algorithm worker's address in
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
                algorithm_address,
            ),
            config_data=config.env.model_dump(mode="json"),
            max_concurrent=config.serve.max_concurrent,
        )


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
