import asyncio
from datetime import datetime

import pandas as pd
import torch
from datasets import DatasetDict, load_from_disk
from verifiers.scripts.eval import eval_environment_async, push_eval_to_prime_hub

from prime_rl.eval.config import OfflineEvalConfig
from prime_rl.eval.utils import (
    compute_pass_at_k,
    make_dataset,
    prepare_sampling_args,
)
from prime_rl.orchestrator.client import (
    check_has_model,
    check_health,
    reload_weights,
    setup_client,
    update_weights,
)
from prime_rl.orchestrator.utils import parse_is_truncated_completions, parse_num_completion_tokens
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.monitor import get_monitor, setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import capitalize, clean_exit, get_eval_dir, get_step_path


async def _run_evals_with_prime_rl_features(
    client,
    config: OfflineEvalConfig,
    ckpt_step: int,
    step: int | None = None,
):
    """
    Run multi-env evaluation using verifiers, then apply prime-rl specific processing.
    
    This wraps verifiers.scripts.eval.eval_environments_parallel and adds:
    - Wandb/monitor logging
    - Disk saving
    - Prime Hub push
    - HF Hub push
    - Pass@k computation
    - Checkpoint-aware metrics
    """
    logger = get_logger()
    monitor = get_monitor()
    
    # Prepare global sampling args
    global_sampling_args = prepare_sampling_args(config.sampling, config.client)
    
    # Prepare per-env sampling args dict
    sampling_args_dict = None
    if config.sampling_args_per_env:
        sampling_args_dict = {}
        for env_id, env_sampling_override in config.sampling_args_per_env.items():
            # Merge global args with per-env overrides
            env_sampling = global_sampling_args.copy() if global_sampling_args else {}
            env_sampling.update(env_sampling_override)
            sampling_args_dict[env_id] = env_sampling
    
    # Use verifiers' parallel multi-env evaluation with per-env support
    logger.info(f"Running parallel evaluation on {len(config.environment_ids)} environments: {', '.join(config.environment_ids)}")
    
    results_dict = {}
    for env_id in config.environment_ids:
        env_model = config.models_per_env.get(env_id, config.model.name) if config.models_per_env else config.model.name
        env_sampling = sampling_args_dict.get(env_id, global_sampling_args) if sampling_args_dict else global_sampling_args
        
        logger.info(f"Evaluating {env_id} with model {env_model}")
        
        env_num_examples = config.num_examples_per_env.get(env_id, config.num_examples)
        env_rollouts = config.rollouts_per_example_per_env.get(env_id, config.rollouts_per_example)
        env_max_concurrent = config.max_concurrent_per_env.get(env_id, config.max_concurrent)
        
        result = await eval_environment_async(
            env=env_id,
            env_args=config.environment_args.get(env_id, {}),
            client=client,
            model=env_model,
            num_examples=env_num_examples,
            rollouts_per_example=env_rollouts,
            max_concurrent=env_max_concurrent,
            sampling_args=env_sampling,
        )
        results_dict[result[0]] = result[1]
    
    # Process each environment's results with prime-rl specific features
    for eval_id, generate_outputs in results_dict.items():
        num_examples = config.num_examples_per_env.get(eval_id, config.num_examples)
        rollouts_per_example = config.rollouts_per_example_per_env.get(eval_id, config.rollouts_per_example)
        env_model = config.models_per_env.get(eval_id, config.model.name) if config.models_per_env else config.model.name
        
        # Extract metrics
        rewards = torch.tensor(generate_outputs.reward).reshape(-1, rollouts_per_example).float()
        responses = [state["responses"] for state in generate_outputs.state]
        completion_lens = torch.tensor(parse_num_completion_tokens(responses)).reshape(-1, rollouts_per_example).float()
        is_truncated = torch.tensor(parse_is_truncated_completions(responses)).reshape(-1, rollouts_per_example).float()
        
        k = rollouts_per_example
        example_ids = [i // rollouts_per_example for i in range(len(generate_outputs.reward))]
        sample_stats = pd.DataFrame({"example_id": example_ids, "reward": rewards.flatten().tolist()})
        unique_rewards = sample_stats.reward.unique()
        could_be_binary = set(unique_rewards).issubset({0.0, 1.0})
        
        if could_be_binary:
            pass_at_k = (
                sample_stats.groupby("example_id")
                .apply(lambda x: compute_pass_at_k(x.reward), include_groups=False)
                .apply(pd.Series)
            )
        else:
            pass_at_k = None
            logger.warning(f"[{eval_id}] Skipping pass@k computation because rewards appear to be non-binary")
        
        # Log statistics
        message = f"Evaluated {eval_id} (Avg@{k}={sample_stats.reward.mean():.4f}"
        if could_be_binary and pass_at_k is not None:
            for pass_rate, pass_rate_score in pd.Series(pass_at_k.mean()).items():
                message += f", {capitalize(str(pass_rate))}: {pass_rate_score:.4f}"
        message += f", Completion Length: {completion_lens.mean():.2f} (±{completion_lens.std():.2f}), Truncated: {is_truncated.mean() * 100:.1f}%)"
        logger.success(message)
        
        # Log statistics to monitor (WandB)
        eval_metrics = {f"avg@{k}": rewards.mean().item()}
        if could_be_binary and pass_at_k is not None:
            eval_metrics.update(pd.Series(pass_at_k.mean()).to_dict())
        
        eval_metrics = {**{f"eval/{eval_id}/{k}": v for k, v in eval_metrics.items()}}
        if step is None:
            step = ckpt_step
        eval_metrics.update({"progress/ckpt_step": ckpt_step, "step": step})
        monitor.log(eval_metrics)
        
        # Log completion length metrics
        eval_completion_len_metrics = {
            f"eval_completion_len/{eval_id}/avg": completion_lens.mean().item(),
            f"eval_completion_len/{eval_id}/max": completion_lens.max().item(),
            f"eval_completion_len/{eval_id}/min": completion_lens.min().item(),
            "progress/ckpt_step": ckpt_step,
            "step": step,
        }
        monitor.log(eval_completion_len_metrics)
        
        # Save to disk if requested
        if config.save_to_disk:
            eval_dir = get_step_path(get_eval_dir(config.output_dir), ckpt_step) / eval_id
            dataset = make_dataset(generate_outputs)
            dataset.save_to_disk(eval_dir)
            logger.info(f"Saved eval results for {eval_id} to disk ({eval_dir})")
        
        # Push to Prime Hub if requested
        if config.push_to_hub:
            hub_metrics = {
                f"avg@{k}": float(rewards.mean().item()),
                "completion_length_avg": float(completion_lens.mean().item()),
                "completion_length_max": float(completion_lens.max().item()),
                "completion_length_min": float(completion_lens.min().item()),
                "completion_length_std": float(completion_lens.std().item()),
                "truncated_rate": float(is_truncated.mean().item()),
                "num_samples": int(rewards.numel()),
                "num_unique_examples": len(set(example_ids)),
            }
            
            if could_be_binary and pass_at_k is not None:
                for pass_rate, pass_rate_score in pd.Series(pass_at_k.mean()).items():
                    hub_metrics[str(pass_rate)] = float(pass_rate_score)
            
            hub_metadata = {
                "checkpoint_step": ckpt_step,
                "step": step if step is not None else ckpt_step,
                "num_examples": num_examples,
                "rollouts_per_example": rollouts_per_example,
                "max_concurrent": config.max_concurrent_per_env.get(eval_id, config.max_concurrent),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "sampling_config": {
                    "temperature": config.sampling.temperature,
                    "max_tokens": config.sampling.max_tokens,
                    "top_p": config.sampling.top_p,
                    "top_k": config.sampling.top_k,
                    "min_p": config.sampling.min_p,
                    "min_tokens": config.sampling.min_tokens,
                    "repetition_penalty": config.sampling.repetition_penalty,
                    "reasoning_effort": config.sampling.reasoning_effort,
                    "seed": config.sampling.seed,
                },
                "client_config": {
                    "base_url": config.client.base_url,
                    "server_type": config.client.server_type,
                    "timeout": config.client.timeout,
                },
                "env_args": config.environment_args.get(eval_id, {}),
                "is_binary_task": could_be_binary,
            }
            
            # Prepare sample-level results
            hub_results = []
            for i in range(len(example_ids)):
                result_entry = {
                    "example_id": int(example_ids[i]),
                    "reward": float(generate_outputs.reward[i]),
                    "task": str(generate_outputs.task[i]) if i < len(generate_outputs.task) else "",
                    "answer": str(generate_outputs.answer[i]) if i < len(generate_outputs.answer) else "",
                }
                if i < len(generate_outputs.info):
                    info = generate_outputs.info[i]
                    if isinstance(info, dict):
                        if "score" in info:
                            result_entry["score"] = float(info["score"])
                        if "correct" in info:
                            result_entry["correct"] = bool(info["correct"])
                hub_results.append(result_entry)
            
            eval_name = f"{env_model}-{eval_id}-step{ckpt_step}"
            push_eval_to_prime_hub(
                eval_name=eval_name,
                model_name=env_model,
                dataset=eval_id,
                metrics=hub_metrics,
                metadata=hub_metadata,
                results=hub_results if hub_results else None,
            )
    
    # HF Hub push for all environments
    if config.save_to_hf is not None:
        logger.info(f"Pushing eval results for {', '.join(config.environment_ids)} to HF Hub")
        eval_dirs = [
            get_step_path(get_eval_dir(config.output_dir), ckpt_step) / eval_id
            for eval_id in config.environment_ids
        ]
        dataset_dict = DatasetDict(
            {path.name.replace("-", "_"): load_from_disk(path) for path in eval_dirs}
        )
        dataset_dict.push_to_hub(config.save_to_hf)
        logger.info(f"Pushed eval results to HF Hub (https://huggingface.co/datasets/{config.save_to_hf})")


@clean_exit
async def eval(config: OfflineEvalConfig):
    # Initialize the logger
    logger = setup_logger(
        config.log.level, log_file=config.output_dir / "logs" / "eval.log" if config.log.file else None
    )
    logger.info("Starting evaluation")
    logger.info(f"Model: {config.model}")
    logger.info(f"Sampling: {config.sampling}")
    logger.info(f"Eval IDs: {config.environment_ids}")

    # Initialize the monitor
    logger.info(f"Initializing monitor ({config.wandb})")
    setup_monitor(
        config=config.wandb,
        output_dir=None,
        run_config=config,
    )

    # Setup client
    logger.info(
        f"Initializing OpenAI client (base_url={config.client.base_url}, api_key_var={config.client.api_key_var}, server_type={config.client.server_type})"
    )
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
    if config.eval_base:
        logger.info(f"Evaluating model {config.model.name}")
        await _run_evals_with_prime_rl_features(
            client=client,
            config=config,
            ckpt_step=0,
            step=None,
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
            await update_weights(client, config.weights_dir, ckpt_step)

            # Run evals on checkpoint
            await _run_evals_with_prime_rl_features(
                client=client,
                config=config,
                ckpt_step=ckpt_step,
                step=None,
            )

    logger.success("Eval finished!")


def main():
    asyncio.run(eval(parse_argv(OfflineEvalConfig)))


if __name__ == "__main__":
    main()
