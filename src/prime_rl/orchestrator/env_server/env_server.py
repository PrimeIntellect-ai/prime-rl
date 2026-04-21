import asyncio

from verifiers.serve import ZMQEnvServer

from prime_rl.configs.env_server import EnvServerConfig
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import get_log_dir
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit, get_env_ids_to_install, install_env


@clean_exit
def run_server(config: EnvServerConfig):
    setup_logger(config.log.level, json_logging=config.log.json_logging)

    # install environment if not already installed
    env_ids_to_install = set()
    env_ids_to_install.update(get_env_ids_to_install([config.env]))
    for env_id in env_ids_to_install:
        install_env(env_id, prerelease=config.env_install_prerelease)

    log_dir = (get_log_dir(config.output_dir) / config.env.resolved_name).as_posix()

    server = ZMQEnvServer(
        env_id=config.env.stripped_id,
        env_args=config.env.args,
        extra_env_kwargs=config.env.extra_env_kwargs,
        log_level=config.log.level,
        log_dir=log_dir,
        json_logging=config.log.json_logging,
        num_workers=config.env.num_workers,
        **{"address": config.env.address} if config.env.address is not None else {},
    )
    asyncio.run(server.run())


def main():
    """Main entry-point for env-server. Run using `uv run env-server`"""
    set_proc_title("EnvServer")
    run_server(cli(EnvServerConfig))


if __name__ == "__main__":
    main()
