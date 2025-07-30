import asyncio

from prime_rl.eval.config import EvalConfig
from prime_rl.eval.utils import run_benchmark
from prime_rl.orchestrator.client import (
    check_has_model,
    check_health,
    reload_weights,
    setup_client,
)
from prime_rl.orchestrator.logger import setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit


@clean_exit
async def eval(config: EvalConfig):
    # Initialize the logger
    logger = setup_logger(config.log)
    logger.info("Starting evaluation")
    logger.info(f"Model: {config.model}")
    logger.info(f"Sampling: {config.sampling}")
    logger.info(f"Benchmarks: {config.benchmarks}")

    # Initialize the monitor
    logger.info(f"Initializing monitor ({config.monitor})")
    monitor = setup_monitor(config.monitor, None, run_config=config)

    # Setup client
    logger.info(f"Initializing OpenAI client ({config.client.host}:{config.client.port})")
    client = setup_client(config.client)

    # Check health of the client
    logger.info("Waiting for inference pool to be ready")
    await check_health(client)
    await check_has_model(client, config.model.name)
    logger.success(f"Inference pool is healthy and serves {config.model.name}")

    # Reset weights to base model to allow reusing inference server across runs
    logger.info("Resetting weights to base model")
    await reload_weights(client)

    # Run benchmarks on base model
    logger.info(f"Running evals on base model {config.model.name}")
    await asyncio.gather(
        *[
            run_benchmark(
                client,
                benchmark,
                config.model,
                config.sampling,
                rollouts_per_prompt=rollouts_per_prompt,
                ckpt_step=0,
                monitor=monitor,
            )
            for benchmark, rollouts_per_prompt in zip(config.benchmarks, config.rollouts_per_prompt)
        ]
    )

    logger.info("Evaluation finished!")


def main():
    asyncio.run(eval(parse_argv(EvalConfig)))


if __name__ == "__main__":
    main()
