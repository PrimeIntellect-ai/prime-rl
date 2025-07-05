import asyncio

from zeroband.eval.config import EvalConfig
from zeroband.eval.utils import run_benchmark
from zeroband.training.orchestrator.client import (
    check_has_model,
    check_health,
    reload_weights,
    reset_weights,
    setup_client,
)
from zeroband.training.orchestrator.logger import setup_logger
from zeroband.training.orchestrator.utils import wait_for_weight_checkpoint
from zeroband.utils.monitor import setup_monitor
from zeroband.utils.pydantic_config import parse_argv
from zeroband.utils.utils import clean_exit


@clean_exit
async def eval(config: EvalConfig):
    # Initialize the logger
    logger = setup_logger(config.log)
    logger.info("Starting evaluation")
    logger.info(f"Model: {config.model}")
    logger.info(f"Sampling: {config.sampling}")
    logger.info(f"Evaluation: {config.eval}")

    # Initialize the monitor
    logger.info(f"Initializing monitor ({config.monitor})")
    setup_monitor(config.monitor, None, run_config=config)

    # Setup client
    logger.info(f"Initializing OpenAI client ({config.client.base_url})")
    client = setup_client(config.client)

    # Check health of the client
    logger.info("Waiting for inference pool to be ready")
    await check_health(client)
    await check_has_model(client, config.model.name)
    logger.success(f"Inference pool is healthy and serves {config.model.name}")

    # Reset weights to base model to allow reusing inference server across runs
    logger.info("Resetting weights to base model")
    await reset_weights(client)

    # Run benchmarks on base model
    logger.info(f"Running evals on base model {config.model.name}")
    for benchmark in config.eval.benchmarks:
        await run_benchmark(
            client,
            benchmark,
            config.model,
            config.sampling,
            step=0,
            use_tqdm=config.use_tqdm,
        )

    # If specified, run online evaluation
    if config.eval.online:
        logger.info(
            f"Running online evaluation on {config.model.name} every {config.eval.online.interval} steps from checkpoint directory {config.eval.online.ckpt_path}"
        )
        step = config.eval.online.interval
        while True:
            # Wait for checkpoint to be available
            wait_for_weight_checkpoint(config.eval.online.ckpt_path, step)
            await reload_weights(client, config.eval.online.ckpt_path, step)

            # Run benchmarks on new checkpoint
            logger.info(f"Running evals for checkpoint step {step}")
            for benchmark in config.eval.benchmarks:
                await run_benchmark(
                    client,
                    benchmark,
                    config.model,
                    config.sampling,
                    step,
                    use_tqdm=config.use_tqdm,
                )

            # Update eval step to next checkpoint step
            step += config.eval.online.interval

            if config.eval.online.max_steps and step > config.eval.online.max_steps:
                logger.info(
                    f"Reached maximum number of steps ({config.eval.online.max_steps}). Stopping online evaluation."
                )
                break

    logger.info("Evaluation finished!")


def main():
    asyncio.run(eval(parse_argv(EvalConfig)))


if __name__ == "__main__":
    main()
