import asyncio

import verifiers as vf

from prime_rl.orchestrator.utils import (
    set_semaphore,
)
from prime_rl.synthesize.config import SynthesizeConfig
from prime_rl.utils.client import (
    check_has_model,
    check_health,
    setup_admin_clients,
    setup_clients,
)
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit


@clean_exit
async def synthesize(config: SynthesizeConfig):
    # Initialize the logger
    logger = setup_logger(
        config.log.level, log_file=config.output_dir / "logs" / "synthesize.log" if config.log.file else None
    )
    vf.setup_logging(level=config.log.vf_level.upper())

    logger.info("Starting synthetic data generation")
    logger.info(f"Model: {config.model}")
    logger.info(f"Environments: {', '.join([env.name or env.id for env in config.env])}")
    logger.info(f"Sampling: {config.sampling}")

    # Setup clients
    logger.info(
        f"Initializing OpenAI client (base_url={', '.join(config.client.base_url)}, api_key_var={config.client.api_key_var}, server_type={config.client.server_type}, headers={config.client.headers})"
    )
    clients = setup_clients(config.client)
    admin_clients = setup_admin_clients(config.client)

    # Check health of the client
    logger.info("Waiting for inference pool to be ready")
    await check_health(admin_clients)
    await check_has_model(clients, config.model.name)
    logger.success(f"Inference pool is healthy and serves {config.model.name}")

    # Set global semaphore
    await set_semaphore(config.max_concurrent or -1)

    # Generate synthetic data
    logger.info("Generating synthetic data")

    logger.success("Synthetic data generation finished!")


def main():
    asyncio.run(synthesize(parse_argv(SynthesizeConfig)))


if __name__ == "__main__":
    main()
