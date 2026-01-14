import asyncio

import verifiers as vf

from prime_rl.eval.config import OfflineEvalConfig
from prime_rl.eval.utils import run_evals
from prime_rl.orchestrator.utils import set_semaphore
from prime_rl.utils.client import (
    check_has_model,
    check_health,
    reload_weights,
    setup_admin_clients,
    setup_clients,
    setup_evals_client,
    update_weights,
)
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit, get_env_ids_to_install, get_step_path, install_env


@clean_exit
async def eval(config: OfflineEvalConfig):
    # Initialize the logger
    logger = setup_logger(
        config.log.level, log_file=config.output_dir / "logs" / "eval.log" if config.log.file else None
    )
    vf.setup_logging(level=config.log.vf_level.upper())

    env_names = [env.name or env.id for env in config.env]
    logger.info(f"Starting evals for {config.model.name} in environments {', '.join(env_names)}")
    logger.info(f"Using sampling config {config.sampling}")

    # Install environments
    env_ids_to_install = get_env_ids_to_install(config.env)
    for env_id in env_ids_to_install:
        install_env(env_id)

    # Initialize the monitor
    logger.info(f"Initializing monitor ({config.wandb})")
    setup_monitor(
        config=config.wandb,
        output_dir=None,
        run_config=config,
    )

    # Setup clients
    logger.info(
        f"Initializing OpenAI client (base_url={', '.join(config.client.base_url)}, api_key_var={config.client.api_key_var}, headers={config.client.headers})"
    )
    clients = setup_clients(config.client)
    admin_clients = setup_admin_clients(config.client)
    evals_client = setup_evals_client()

    # Check health of the client
    logger.info("Waiting for inference pool to be ready")
    await check_health(admin_clients)
    await check_has_model(clients, config.model.name)
    logger.success(f"Inference pool is healthy and serves {config.model.name}")

    # Reset weights to base model to allow reusing inference server across runs
    logger.info("Resetting weights to base model")
    await reload_weights(admin_clients)

    # Run benchmarks on base model
    await set_semaphore(config.max_concurrent or -1)
    if config.eval_base:
        logger.info(f"Evaluating model {config.model.name}")
        await run_evals(
            clients=clients,
            eval_config=config,
            model_config=config.model,
            sampling_config=config.sampling,
            evals_client=evals_client,
            reasoning_field=config.reasoning_field,
            output_dir=config.output_dir,
            ckpt_step=0,
            resume_path=config.resume_path,
        )

    # If specified, evaluate all checkpoints found in the weights directory
    if config.weights_dir is not None:
        logger.info(f"Evaluating weight checkpoints in {config.weights_dir}")
        ckpt_steps = sorted([int(step_path.name.split("_")[-1]) for step_path in config.weights_dir.glob("step_*")])
        logger.info(f"Found {len(ckpt_steps)} weight checkpoints (steps: {', '.join(map(str, ckpt_steps))})")

        # Filter the steps to evaluate
        if config.steps is not None:
            ckpt_steps = [step for step in ckpt_steps if step in config.steps]

        logger.info(f"Evaluating {len(ckpt_steps)} weight checkpoints (steps: {', '.join(map(str, ckpt_steps))})")
        for ckpt_step in ckpt_steps[::-1]:
            # Update the weights
            logger.info(f"Evaluating model {config.model.name} at checkpoint {ckpt_step}")
            await update_weights(admin_clients, get_step_path(config.weights_dir, ckpt_step))

            # Run evals on checkpoint
            await run_evals(
                clients=clients,
                eval_config=config,
                model_config=config.model,
                sampling_config=config.sampling,
                evals_client=evals_client,
                reasoning_field=config.reasoning_field,
                output_dir=config.output_dir,
                ckpt_step=ckpt_step,
                resume_path=config.resume_path,
            )

        if config.watcher:
            already_evaluated_ckpt_steps = ckpt_steps

            while True:
                all_ckpt_steps = sorted(
                    [int(step_path.name.split("_")[-1]) for step_path in config.weights_dir.glob("step_*")]
                )
                new_ckpt_steps = [step for step in all_ckpt_steps if step not in already_evaluated_ckpt_steps]
                if len(new_ckpt_steps) > 0:
                    logger.info(f"New checkpoints to evaluate: {', '.join(map(str, new_ckpt_steps))}")
                    for ckpt_step in new_ckpt_steps:
                        logger.info(f"Evaluating model {config.model.name} at checkpoint {ckpt_step}")
                        await update_weights(admin_clients, get_step_path(config.weights_dir, ckpt_step))
                        await run_evals(
                            clients=clients,
                            eval_config=config,
                            model_config=config.model,
                            sampling_config=config.sampling,
                            evals_client=evals_client,
                            reasoning_field=config.reasoning_field,
                            output_dir=config.output_dir,
                            ckpt_step=ckpt_step,
                            resume_path=config.resume_path,
                        )
                        already_evaluated_ckpt_steps.append(ckpt_step)
                else:
                    logger.info("No new checkpoints to evaluate, waiting for 10 seconds")
                    await asyncio.sleep(10)

    logger.success("Eval finished!")


def main():
    asyncio.run(eval(parse_argv(OfflineEvalConfig)))


if __name__ == "__main__":
    main()
