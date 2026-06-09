import asyncio

from verifiers.v1.legacy import LegacyEnvServer
from verifiers.v1.serve import EnvServer

from prime_rl.configs.env_server import EnvServerConfig
from prime_rl.orchestrator.utils import intercept_vf_logging
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit


@clean_exit
def run_server(config: EnvServerConfig):
    setup_logger(config.log.level, json_logging=config.log.json_logging)
    # Route v1's stdlib logging (the server's own logs) through our handler.
    intercept_vf_logging(logger="verifiers.v1", level=config.log.level)

    env = config.env
    address = env.address or "tcp://127.0.0.1:5000"
    # A v0/legacy env runs a classic verifiers env through the bridge; a v1 env is a native
    # v1 taskset. Both serve vf.Trace over the same protocol, so the orchestrator is agnostic.
    if env.is_legacy:
        server: EnvServer = LegacyEnvServer(env_id=env.env_id, env_args=env.args, address=address)
    else:
        server = EnvServer(env, address=address)
    asyncio.run(server.run())


def main():
    """Main entry-point for env-server. Run using `uv run env-server`"""
    set_proc_title("EnvServer")
    run_server(cli(EnvServerConfig))


if __name__ == "__main__":
    main()
