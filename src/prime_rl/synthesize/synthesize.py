import asyncio

from prime_rl.synthesize.config import SynthesizeConfig
from prime_rl.synthesize.utils import generate_synthetic_data
from prime_rl.utils.client import (
    check_health,
    maybe_check_has_model,
    setup_admin_clients,
    setup_clients,
)
from prime_rl.utils.logger import intercept_verifiers_logging, setup_logger
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit, get_env_ids_to_install, install_env


@clean_exit
async def synthesize(config: SynthesizeConfig):
    # Initialize the logger
    logger = setup_logger(
        config.log.level,
        log_file=config.output_dir / "logs" / "synthesize.log" if config.log.file else None,
        json_logging=config.log.json_logging,
    )
    intercept_verifiers_logging(level=config.log.vf_level)

    env_names = [env.name or env.id for env in config.env]
    logger.info(f"Starting synthetic data generation for {config.model.name} in environments {', '.join(env_names)}")
    logger.info(f"Using sampling config {config.sampling}")

    # Install environments
    env_ids_to_install = get_env_ids_to_install(config.env)
    for env_id in env_ids_to_install:
        install_env(env_id)

    # Setup clients
    logger.info(
        f"Initializing OpenAI client (base_url={', '.join(config.client.base_url)}, api_key_var={config.client.api_key_var}, headers={config.client.headers})"
    )
    clients = setup_clients(config.client)
    admin_clients = setup_admin_clients(config.client)

    # Check health of the client
    logger.info("Waiting for inference pool to be ready")
    await check_health(admin_clients)
    await maybe_check_has_model(clients, config.model.name, skip_model_check=config.client.skip_model_check)
    logger.success(f"Inference pool is healthy and serves {config.model.name}")

    # Generate synthetic data
    logger.info("Starting synthetic data generation")
    await asyncio.gather(
        *[
            generate_synthetic_data(
                clients=clients,
                env_id=env.id,
                env_name=env.name,
                env_args=env.args,
                num_examples=env.num_examples or config.num_examples,
                rollouts_per_example=env.rollouts_per_example or config.rollouts_per_example,
                reasoning_field=config.reasoning_field,
                skip_first=env.skip_first,
                output_dir=config.output_dir,
                model_config=config.model,
                sampling_config=config.sampling,
                client_config=config.client,
            )
            for env in config.env
        ]
    )

    logger.success("Synthetic data generation finished!")


def main():
    asyncio.run(synthesize(parse_argv(SynthesizeConfig)))


if __name__ == "__main__":
    main()
