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
    # The env's ``pool`` (static or elastic) sizes the server; a v0/legacy env runs through
    # the bridge, a v1 env is a native taskset — both serve vf.Trace over the same protocol,
    # so the orchestrator is agnostic. serve_env applies the logging setup in this process
    # and in every spawned worker.
    server_kwargs = (
        {"env_id": env.env_id, "env_args": env.args, "extra_env_kwargs": env.extra_env_kwargs}
        if env.is_legacy
        else {"config": env}
    )
    serve_env(
        **pool_serve_kwargs(env.pool),
        legacy=env.is_legacy,
        address=address,
        log_setup=partial(setup_env_server_logging, config.log.level, config.log.json_logging),
        **server_kwargs,
    )


def main():
    """Main entry-point for env-server. Run using `uv run env-server`"""
    set_proc_title("EnvServer")
    run_server(cli(EnvServerConfig))


if __name__ == "__main__":
    main()
