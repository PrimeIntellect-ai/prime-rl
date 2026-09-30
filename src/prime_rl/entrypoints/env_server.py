import os
import queue
import threading
import uuid
from functools import partial
from pathlib import Path

from verifiers.v1 import pool_serve_kwargs
from verifiers.v1.runtimes import set_base_sandbox_labels
from verifiers.v1.serve import env_config_data, serve_env

from prime_rl.configs.env_server import EnvServerConfig
from prime_rl.orchestrator.utils import setup_env_server_logging
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit


def setup_worker(log_level: str | None, json_logging: bool, sandbox_labels: list[str]) -> None:
    setup_env_server_logging(log_level, json_logging)
    set_base_sandbox_labels(sandbox_labels)


def publish_address(addresses: queue.SimpleQueue, path: Path) -> None:
    """Write the bound address once serve_env reports it (it blocks in the server loop
    afterwards). Atomic, so a waiting client never reads a partial line."""
    address = addresses.get()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".address.tmp")
    tmp.write_text(address)
    tmp.replace(path)


@clean_exit
def run_server(config: EnvServerConfig):
    run_name = os.environ.get("PRL_RUN_NAME")
    sandbox_labels = [run_name] if run_name else []
    addresses: queue.SimpleQueue | None = None
    if config.address_file is not None:
        addresses = queue.SimpleQueue()
        threading.Thread(target=publish_address, args=(addresses, config.address_file), daemon=True).start()
    # ``serve.pool`` (static or elastic) sizes the server. serve_env applies the worker
    # setup in this process and in every spawned worker.
    serve_env(
        **pool_serve_kwargs(config.serve.pool),
        address=config.serve.address or "tcp://127.0.0.1:0",
        address_queue=addresses,
        log_setup=partial(setup_worker, config.log.level, config.log.json_logging, sandbox_labels),
        config_data=env_config_data(config.env),
        max_concurrent=config.serve.max_concurrent,
    )


def main():
    """Main entry-point for the env server. Run using `uv run env-server`"""
    set_proc_title("EnvServer")
    # verifiers keys run-scoped state (creation limiters) by $VF_RUN_ID. Every launcher,
    # the Python entrypoints and the SLURM templates alike, hands env servers $PRL_RUN_ID;
    # a standalone server is a run of its own.
    os.environ.setdefault("VF_RUN_ID", os.environ.get("PRL_RUN_ID") or uuid.uuid4().hex)
    run_server(cli(EnvServerConfig))


if __name__ == "__main__":
    main()
