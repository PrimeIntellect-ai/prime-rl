from functools import partial

from verifiers.v1 import pool_serve_kwargs
from verifiers.v1.serve import env_config_data, serve_env

from prime_rl.configs.env import EnvConfig
from prime_rl.orchestrator.utils import setup_env_logging
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit


@clean_exit
def run_server(config: EnvConfig):
    # ``serve.pool`` (static or elastic) sizes the server; a v0/legacy env runs through
    # the bridge, a v1 env is a native env block — both speak the same serve protocol,
    # so the orchestrator is agnostic. serve_env applies the logging setup in this process
    # and in every spawned worker.
    server_kwargs = (
        {"env_id": config.env_id, "env_args": config.legacy.args, "extra_env_kwargs": config.legacy.extra_env_kwargs}
        if config.is_legacy
        else {"config_data": env_config_data(config.env), "max_concurrent": config.serve.max_concurrent}
    )
    serve_env(
        **pool_serve_kwargs(config.serve.pool),
        legacy=config.is_legacy,
        address=config.serve.address,
        log_setup=partial(setup_env_logging, config.log.level, config.log.json_logging),
        **server_kwargs,
    )


def main():
    """Main entry-point for the env server. Run using `uv run env`"""
    set_proc_title("Env")
    run_server(cli(EnvConfig))


if __name__ == "__main__":
    main()
