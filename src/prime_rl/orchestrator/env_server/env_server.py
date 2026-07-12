from functools import partial

from verifiers.v1 import pool_serve_kwargs
from verifiers.v1.serve import serve_env

from prime_rl.configs.env_server import EnvServerConfig
from prime_rl.orchestrator.utils import setup_env_server_logging
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit


@clean_exit
def run_server(config: EnvServerConfig):
    env = config.env
    address = env.address or "tcp://127.0.0.1:5000"
    # The topology's ``pool`` sizes the server. ``serve_env`` applies logging setup in
    # this process and every spawned worker.
    serve_env(
        **pool_serve_kwargs(env.pool),
        legacy=False,
        address=address,
        log_setup=partial(setup_env_server_logging, config.log.level, config.log.json_logging),
        config=env,
    )


def main():
    """Main entry-point for env-server. Run using `uv run env-server`"""
    set_proc_title("EnvServer")
    run_server(cli(EnvServerConfig))


if __name__ == "__main__":
    main()
