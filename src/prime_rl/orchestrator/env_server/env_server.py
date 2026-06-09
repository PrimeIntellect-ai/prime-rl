import asyncio

from verifiers.nano.serve import EnvServer

from prime_rl.configs.env_server import EnvServerConfig
from prime_rl.orchestrator.utils import intercept_vf_logging
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit


@clean_exit
def run_server(config: EnvServerConfig):
    setup_logger(config.log.level, json_logging=config.log.json_logging)
    # Route vf-nano's stdlib logging (the server's own logs) through our handler.
    intercept_vf_logging(logger="verifiers.nano", level=config.log.level)

    # TODO(vf-nano, experimental): temporary. vf-nano envs are local packages
    # (installed in this venv); no hub install.
    server = EnvServer(config.env, address=config.env.address or "tcp://127.0.0.1:5000")
    asyncio.run(server.run())


def main():
    """Main entry-point for env-server. Run using `uv run env-server`"""
    set_proc_title("EnvServer")
    run_server(cli(EnvServerConfig))


if __name__ == "__main__":
    main()
