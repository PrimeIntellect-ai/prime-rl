import os
import queue
import threading
import uuid
from functools import partial
from pathlib import Path

from verifiers.v1.runtimes import set_base_sandbox_labels

from prime_rl.configs.env_worker_pool import EnvWorkerPoolConfig
from prime_rl.env import run_env_worker_pool, set_episode_hook
from prime_rl.orchestrator.utils import setup_env_worker_pool_logging
from prime_rl.transports.payload import PayloadHook
from prime_rl.utils.config import cli
from prime_rl.utils.utils import clean_exit


def setup_worker(
    log_level: str | None, json_logging: bool, sandbox_labels: list[str], bind_address: str | None
) -> None:
    setup_env_worker_pool_logging(log_level, json_logging)
    set_base_sandbox_labels(sandbox_labels)
    # Episode streams never carry training arrays; the hook keeps them for the trainer.
    set_episode_hook(PayloadHook(bind_address))


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
    # Apply logging, sandbox labels, and the payload hook in the broker and every worker.
    run_env_worker_pool(
        pool=config.serve.pool,
        address=config.serve.address or "tcp://127.0.0.1:0",
        address_queue=addresses,
        worker_setup=partial(
            setup_worker,
            config.log.level,
            config.log.json_logging,
            sandbox_labels,
            config.serve.address,
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
    run_pool(cli(EnvWorkerPoolConfig))


if __name__ == "__main__":
    main()
