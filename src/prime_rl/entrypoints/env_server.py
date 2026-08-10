from functools import partial

from verifiers.v1 import pool_serve_kwargs
from verifiers.v1.serve import env_config_data, serve_env

from prime_rl.configs.env_server import EnvServerConfig
from prime_rl.orchestrator.utils import setup_env_server_logging
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit


@clean_exit
def run_server(config: EnvServerConfig):
    # ``serve.pool`` (static or elastic) sizes the server. serve_env applies the
    # logging setup in this process and in every spawned worker.
    serve_env(
        **pool_serve_kwargs(config.serve.pool),
        address=config.serve.address,
        log_setup=partial(setup_env_server_logging, config.log.level, config.log.json_logging),
        config_data=env_config_data(config.env),
        max_concurrent=config.serve.max_concurrent,
    )


def main():
    """Main entry-point for the env server. Run using `uv run env-server`"""
    set_proc_title("EnvServer")
    run_server(cli(EnvServerConfig))


if __name__ == "__main__":
    main()
