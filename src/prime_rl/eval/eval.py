import asyncio

from prime_rl.eval.config import OfflineEvalConfig
from prime_rl.eval.utils import run_eval
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
async def eval(config: OfflineEvalConfig):
    # Initialize the logger
    logger = setup_logger(config.log)
    logger.info("Starting evaluation")
    logger.info(f"Model: {config.model}")
    logger.info(f"Sampling: {config.sampling}")
    logger.info(f"Eval IDs: {config.ids}")

    # Initialize the monitor
    logger.info(f"Initializing monitor ({config.monitor})")
    monitor = setup_monitor(
        config=config.monitor,
        outputs_dir=None,
        run_config=config,
    )

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
            run_eval(
                client=client,
                eval_id=eval_id,
                env_args=config.args.get(eval_id, {}),
                model_config=config.model,
                sampling_config=config.sampling,
                num_examples=num_examples,
                rollouts_per_example=rollouts_per_example,
                ckpt_step=0,
                monitor=monitor,
            )
            for eval_id, num_examples, rollouts_per_example in zip(
                config.ids, config.num_examples, config.rollouts_per_example
            )
        ]
    )

    logger.info("Evaluation finished!")


def main():
    asyncio.run(eval(parse_argv(OfflineEvalConfig)))


if __name__ == "__main__":
    main()
