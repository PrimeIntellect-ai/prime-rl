import os
from functools import partial

from verifiers.v1 import pool_serve_kwargs
from verifiers.v1.runtimes import set_base_sandbox_labels
from verifiers.v1.serve import env_config_data, serve_env

from prime_rl.configs.env_server import EnvServerConfig
from prime_rl.inference.vllm.routed_experts import install_native_routed_experts_normalizer
from prime_rl.orchestrator.utils import setup_env_server_logging
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit


def setup_worker(log_level: str | None, json_logging: bool, sandbox_labels: list[str]) -> None:
    setup_env_server_logging(log_level, json_logging)
    set_base_sandbox_labels(sandbox_labels)
    install_native_routed_experts_normalizer()


@clean_exit
def run_server(config: EnvServerConfig):
    run_name = os.environ.get("PRL_RUN_NAME")
    sandbox_labels = [run_name] if run_name else []
    # ``serve.pool`` (static or elastic) sizes the server. serve_env applies the worker
    # setup in this process and in every spawned worker.
    serve_env(
        **pool_serve_kwargs(config.serve.pool),
        address=config.serve.address,
        log_setup=partial(setup_worker, config.log.level, config.log.json_logging, sandbox_labels),
        config_data=env_config_data(config.env),
        max_concurrent=config.serve.max_concurrent,
    )


def main():
    """Main entry-point for the env server. Run using `uv run env-server`"""
    set_proc_title("EnvServer")
    run_server(cli(EnvServerConfig))


if __name__ == "__main__":
    main()
