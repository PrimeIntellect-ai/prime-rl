import asyncio

from loguru import logger
from verifiers.workers import ZMQEnvServer

from prime_rl.orchestrator.env_server.config import EnvServerConfig
from prime_rl.utils.pathing import get_log_dir
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit, get_env_ids_to_install, install_env


@clean_exit
@logger.catch(reraise=True)
def run_server(config: EnvServerConfig):
    # install environment if not already installed
    env_ids_to_install = set()
    env_ids_to_install.update(get_env_ids_to_install([config.env]))
    for env_id in env_ids_to_install:
        install_env(env_id)

    env_name = config.env.name or config.env.id
    log_file = (get_log_dir(config.output_dir) / "train" / f"{env_name}.log").as_posix()
    ZMQEnvServer.run_server(
        env_id=config.env.id,
        env_args=config.env.args,
        extra_env_kwargs={},
        log_level=config.log.level,
        log_file_level=config.log.vf_level,
        log_file=log_file,
        **{"address": config.env.address} if config.env.address is not None else {},
    )


def main():
    """Main entry-point for env-server. Run using `uv run env-server`"""
    asyncio.run(run_server(parse_argv(EnvServerConfig)))


if __name__ == "__main__":
    main()
