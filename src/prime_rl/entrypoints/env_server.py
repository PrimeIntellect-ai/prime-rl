import os
from functools import partial

from verifiers.v1 import pool_serve_kwargs
from verifiers.v1.serve import env_config_data, serve_env

from prime_rl.configs.env_server import EnvServerConfig
from prime_rl.orchestrator.utils import setup_env_server_logging
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.run_assets import IMAGE_OFFLOAD_DIR_ENV, resolve_image_offload_dir
from prime_rl.utils.utils import clean_exit


@clean_exit
def run_server(config: EnvServerConfig):
    # Renderers offload images to the dir named by this env var; resolve it from
    # this server's own config, letting an operator-set env win (multi-node).
    os.environ.setdefault(
        IMAGE_OFFLOAD_DIR_ENV, str(resolve_image_offload_dir(config.output_dir, config.multimodal, os.environ))
    )
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
        log_setup=partial(setup_env_server_logging, config.log.level, config.log.json_logging),
        **server_kwargs,
    )


def main():
    """Main entry-point for the env server. Run using `uv run env-server`"""
    set_proc_title("EnvServer")
    run_server(cli(EnvServerConfig))


if __name__ == "__main__":
    main()
