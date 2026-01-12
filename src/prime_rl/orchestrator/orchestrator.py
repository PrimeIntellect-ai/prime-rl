import asyncio
import random
import time

import tomli_w

from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.ckpt import CheckpointManager, Progress
from prime_rl.orchestrator.patches import monkey_patch_chat_completion_logprobs, monkey_patch_oai_iterable_types
from prime_rl.orchestrator.scheduler import EvalRolloutScheduler, TrainScheduler
from prime_rl.orchestrator.trajectories import branch_rollout, interleave_rollout
from prime_rl.transport import TrainingBatch, TrainingSample, setup_training_batch_sender
from prime_rl.utils.event_loop_lag import EventLoopLagMonitor
from prime_rl.utils.pathing import resolve_latest_ckpt_step

# This monkey patch is necessary to avoid Pydantic validating fields using typing.Iterable (e.g. in multimodal or tool call messages) lazily which leads to tokenization errors, for more info see https://github.com/PrimeIntellect-ai/prime-rl/pull/1249
monkey_patch_oai_iterable_types()


# This monkey patch is necessary to avoid heavy CPU overhead from constructing the OAI ChatCompletion Pydantic model with logprobs, for more info see https://github.com/PrimeIntellect-ai/prime-rl/pull/1189
monkey_patch_chat_completion_logprobs()

# Import environment before any other imports

import pandas as pd
from loguru import logger

from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.orchestrator.utils import (
    compute_teacher_logprobs,
    print_benchmark,
    set_semaphore,
)
from prime_rl.utils.client import (
    check_health,
    init_nccl_broadcast,
    setup_admin_clients,
    setup_clients,
)
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import (
    clean_exit,
    to_col_format,
)
from prime_rl.utils.vf import get_completion_len, get_prompt_len, get_seq_len


class Orchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        from pprint import pprint

        pprint(config.model_dump(exclude_none=True, mode="json"))
        self.logger = setup_logger(
            config.log.level, log_file=config.output_dir / "logs" / "orchestrator.log" if config.log.file else None
        )

        self.progress = Progress()
        self.train_scheduler = TrainScheduler(
            progress=self.progress,
            env_worker_group_config=config.train.envs,
            buffer_config=config.buffer,
            batch_size=config.batch_size,
            rollouts_per_example=config.rollouts_per_example,
            oversampling_factor=config.oversampling_factor,
            model_name=config.model.name,
            lora_name=config.model.lora.name if config.model.lora else None,
            max_async_level=config.max_async_level,
            strict_async_level=config.strict_async_level,
            max_off_policy_steps=config.max_off_policy_steps,
            client_config=config.client,
            output_dir=config.output_dir,
        )
        if config.eval:
            self.eval_scheduler = EvalRolloutScheduler(
                env_worker_group_config=config.eval.envs,
                model_name=config.model.name,
            )

        self.logger.info(f"Initializing admin clients ({config.client})")
        self.admin_clients = setup_admin_clients(config.client)

        # Setup training batch sender for sending training examples to trainer
        self.logger.info(f"Initializing training batch sender ({config.rollout_transport})")
        self.training_batch_sender = setup_training_batch_sender(config.output_dir, config.rollout_transport)

        if config.teacher_model:
            self.logger.info(
                f"Initializing teacher clients for {config.teacher_model.model.name} ({config.teacher_model.client})"
            )
            self.teacher_clients = setup_clients(config.teacher_model.client)
        else:
            self.teacher_clients = None

        self.logger.info(f"Initializing monitor(s) (wandb={config.wandb}, prime={config.prime_monitor})")
        self.monitor = setup_monitor(
            wandb_config=config.wandb,
            prime_config=config.prime_monitor,
            tokenizer_config=config.tokenizer,
            output_dir=config.output_dir,
            run_config=config,
        )
        self.event_loop_lag_monitor = EventLoopLagMonitor()

        if config.heartbeat is not None:
            self.logger.info("Initializing heartbeat")
            self.heart = Heartbeat(config.heartbeat.url)
        else:
            self.heart = None

        if config.ckpt is not None:
            self.logger.info(f"Initializing checkpoint manager ({config.ckpt})")
            self.ckpt_manager = CheckpointManager(config.output_dir, config.ckpt)
        else:
            self.ckpt_manager = None

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    def save_config(self):
        """Save orchestrator config to output directory."""
        config_dir = self.config.output_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "orch.toml", "wb") as f:
            tomli_w.dump(self.config.model_dump(exclude_none=True, mode="json"), f)

    @clean_exit
    @logger.catch(reraise=True)
    async def start(self):
        self.logger.info("Starting orchestrator")

        self.save_config()
        self.event_loop_lag_monitor.start()

        # Set up weight broadcast backend
        self.logger.info(f"Initializing weight broadcast ({self.config.weight_broadcast})")
        if self.config.weight_broadcast.type == "nccl":
            await init_nccl_broadcast(
                self.admin_clients,
                self.config.weight_broadcast.host,
                self.config.weight_broadcast.port,
                self.config.weight_broadcast.timeout,
            )

        # TODO: can we not jsut set model.lora.name as lora name initially?
        # if checkpoint_step is not None and config.model.lora is not None:
        #     scheduler.model_name = config.model.lora.name
        #     for workers in scheduler.workers.values():
        #         for worker in workers:
        #             worker.model_name = config.model.lora.name

        # Check health of the client
        self.logger.info("Waiting for inference pool to be ready")
        await check_health(self.admin_clients)
        # await check_has_model(clients, config.model.name)
        self.logger.success("Inference pool ready")

        # Track last online eval checkpoint step for this process
        # last_eval_step = -1

        if self.ckpt_manager:
            assert self.config.ckpt
            if self.config.ckpt.resume_step == -1:
                checkpoint_step = resolve_latest_ckpt_step(self.ckpt_manager.ckpt_dir)
            else:
                checkpoint_step = self.config.ckpt.resume_step
            assert isinstance(checkpoint_step, int)
            self.ckpt_manager.load(self.progress, self.train_scheduler.rollout_scheduler.buffer, step=checkpoint_step)
            logger.info(f"Resuming training from checkpoint step {checkpoint_step}")
            self.train_scheduler.update_policy_scheduler.ckpt_step = (
                self.progress.step
            )  # always resume from the latest checkpoint
            if self.config.eval and self.config.eval.skip_eval_on_resume:
                last_eval_step = self.progress.step  # will skip eval on first resume step
            else:
                last_eval_step = -1
                logger.info("Skipping online eval on resume")
            await self.train_scheduler.update_policy_scheduler.update_weights(checkpoint_step)
        else:
            logger.info("Training from scratch. Resetting weights to base model")
            if self.config.model.lora is None:
                await self.train_scheduler.update_policy_scheduler.reload_weights()
            last_eval_step = -1

        # Iterate over dataset in batches
        max_steps = self.config.max_steps or int(1e9)
        self.logger.info(f"Starting orchestrator loop (max_steps={max_steps or 'infinite'})")
        is_first_step = True
        await set_semaphore(self.config.max_concurrent or -1)

        self.train_scheduler.start()
        if self.config.eval:
            self.eval_scheduler.start()

        while True:
            # save checkpoint (if we are at an interval step and not at the first or last step)
            is_last_step = self.config.max_steps is not None and self.progress.step == self.config.max_steps - 1
            save_ckpt_time = 0
            if (
                (self.config.ckpt and self.config.ckpt.interval)
                and not (is_first_step or is_last_step)
                and self.progress.step % self.config.ckpt.interval == 0
            ):
                assert self.ckpt_manager is not None
                logger.info(f"Saving checkpoint at step {self.progress.step}")
                save_ckpt_start_time = time.perf_counter()
                self.ckpt_manager.save(
                    self.progress, self.train_scheduler.rollout_scheduler.buffer, step=self.progress.step
                )
                save_ckpt_time = time.perf_counter() - save_ckpt_start_time

            # Break if we have reached the maximum number of steps
            if self.config.max_steps and self.progress.step >= self.config.max_steps:
                break

            self.logger.info(f"Starting step {self.progress.step}")
            step_start_time = time.perf_counter()

            if (
                self.config.eval
                and self.progress.step % self.config.eval.interval == 0
                and self.progress.step > last_eval_step
                and (
                    (self.train_scheduler.update_policy_scheduler.ckpt_step == 0 and self.config.eval.eval_base_model)
                    or self.progress.step > 0
                )
            ):
                logger.info(f"Running evals for step {self.progress.step}")
                self.train_scheduler.schedule_rollouts.clear()  # block scheduling rollouts
                tasks = []
                for env_name in self.eval_scheduler.env_worker_group.env_names:
                    tasks.append(self.eval_scheduler.generate_batch(env_name))
                await asyncio.gather(*tasks)
                self.train_scheduler.schedule_rollouts.set()
                last_eval_step = self.progress.step

            # Schedule generating the training batch
            generate_completions_start_time = time.perf_counter()
            train_task = self.train_scheduler.wait_for_batch()

            # Await train rollouts, process results and write batch to disk to consume by trainer
            train_rollouts = await train_task
            generate_completions_time = time.perf_counter() - generate_completions_start_time

            # Compute advantages
            rewards = [rollout["reward"] for rollout in train_rollouts]
            completion_lens = [get_completion_len(rollout) for rollout in train_rollouts]
            advantages = compute_advantages(
                rewards,
                completion_lens,
                self.config.rollouts_per_example,
                self.config.advantage,
            )

            # Update and sample rollouts from the buffer
            make_train_example = (
                interleave_rollout if self.config.trajectory_strategy == "interleaved" else branch_rollout
            )
            train_examples: list[TrainingSample] = []
            for train_rollout, advantage in zip(train_rollouts, advantages):
                train_example = make_train_example(train_rollout)
                if train_example is not None:
                    for te in train_example:
                        te.advantage = advantage
                        te.reward = train_rollout["reward"]
                    train_examples.extend(train_example)
            self.logger.debug(
                f"Converted {len(train_rollouts)} training rollouts to {len(train_examples)} training examples using {self.config.trajectory_strategy} strategy"
            )

            # Compute teacher logprobs if teacher model is configured
            teacher_logprobs_time = 0
            if self.config.teacher_model is not None:
                assert self.teacher_clients is not None
                self.logger.info(f"Computing teacher logprobs for {len(train_examples)} training examples")
                teacher_logprobs_start_time = time.perf_counter()
                teacher_logprobs_list = await compute_teacher_logprobs(
                    clients=self.teacher_clients,
                    model_name=self.config.teacher_model.model.name,
                    samples=train_examples,
                )
                for train_example, teacher_logprobs in zip(train_examples, teacher_logprobs_list):
                    train_example.teacher_logprobs = teacher_logprobs
                teacher_logprobs_time = time.perf_counter() - teacher_logprobs_start_time
                self.logger.debug(f"Computed teacher logprobs in {teacher_logprobs_time:.2f}s")

            training_batch = TrainingBatch(
                examples=train_examples,
                temperature=self.config.sampling.temperature,
                step=self.progress.step,
            )
            self.training_batch_sender.send(training_batch)

            # Gather metrics in dataframes
            results_df = pd.DataFrame(
                {
                    "example_id": [rollout["example_id"] for rollout in train_rollouts],
                    "task": [rollout["task"] for rollout in train_rollouts],
                    "reward": [rollout["reward"] for rollout in train_rollouts],
                    "is_truncated": [rollout["is_truncated"] for rollout in train_rollouts],
                    "error": [rollout["error"] for rollout in train_rollouts],
                    "completion_len": [get_completion_len(rollout) for rollout in train_rollouts],
                    "prompt_len": [get_prompt_len(rollout) for rollout in train_rollouts],
                    "seq_len": [get_seq_len(rollout) for rollout in train_rollouts],
                    "num_turns": [len(rollout["trajectory"]) for rollout in train_rollouts],
                    "generation_ms": [rollout["timing"]["generation_ms"] for rollout in train_rollouts],
                    "scoring_ms": [rollout["timing"]["scoring_ms"] for rollout in train_rollouts],
                }
            )

            # Gather individual reward function metrics
            metrics_df = pd.DataFrame([rollout["metrics"] for rollout in train_rollouts])

            # Update progress metrics and throughput
            num_tokens = int(results_df.seq_len.sum())
            self.progress.total_tokens += num_tokens
            self.progress.total_samples += self.config.batch_size
            self.progress.total_problems += self.config.batch_size // self.config.rollouts_per_example
            throughput = num_tokens / generate_completions_time

            # Compute solve all and none tensors
            solve_all = (
                results_df.groupby("example_id")
                .apply(lambda x: x.reward.sum() == self.config.rollouts_per_example, include_groups=False)
                .mean()
            )
            solve_none = (
                results_df.groupby("example_id").apply(lambda x: x.reward.sum() == 0, include_groups=False).mean()
            )
            effective_batch_size = 1 - solve_none - solve_all

            step_time = time.perf_counter() - step_start_time
            to_log = {
                # Progress metrics
                "progress/tokens": num_tokens,
                "progress/samples": self.config.batch_size,
                "progress/problems": self.config.batch_size // self.config.rollouts_per_example,
                "progress/total_tokens": self.progress.total_tokens,
                "progress/total_samples": self.progress.total_samples,
                "progress/total_problems": self.progress.total_problems,
                # "progress/ckpt_step": ckpt_step,  # Shared W&B axis
                # Sequence length metrics
                "seq_len/mean": results_df.groupby("example_id").seq_len.mean().mean(),
                "seq_len/max": results_df.groupby("example_id").seq_len.mean().max(),
                "seq_len/min": results_df.groupby("example_id").seq_len.mean().min(),
                "prompt_len/mean": results_df.groupby("example_id").prompt_len.mean().mean(),
                "prompt_len/max": results_df.groupby("example_id").prompt_len.mean().max(),
                "prompt_len/min": results_df.groupby("example_id").prompt_len.mean().min(),
                "completion_len/mean": results_df.groupby("example_id").completion_len.mean().mean(),
                "completion_len/max": results_df.groupby("example_id").completion_len.mean().max(),
                "completion_len/min": results_df.groupby("example_id").completion_len.mean().min(),
                "is_truncated/mean": results_df.groupby("example_id").is_truncated.mean().mean(),
                "is_truncated/max": results_df.groupby("example_id").is_truncated.mean().max(),
                "is_truncated/min": results_df.groupby("example_id").is_truncated.mean().min(),
                # Turn metrics
                "num_turns/mean": results_df.groupby("example_id").num_turns.mean().mean(),
                "num_turns/max": results_df.groupby("example_id").num_turns.mean().max(),
                "num_turns/min": results_df.groupby("example_id").num_turns.mean().min(),
                # Verifier timing metrics
                "generation_ms/mean": results_df.groupby("example_id").generation_ms.mean().mean(),
                "generation_ms/max": results_df.groupby("example_id").generation_ms.mean().max(),
                "generation_ms/min": results_df.groupby("example_id").generation_ms.mean().min(),
                "scoring_ms/mean": results_df.groupby("example_id").scoring_ms.mean().mean(),
                "scoring_ms/max": results_df.groupby("example_id").scoring_ms.mean().max(),
                "scoring_ms/min": results_df.groupby("example_id").scoring_ms.mean().min(),
                # Performance metrics
                "perf/throughput": throughput,
                # Train reward
                "reward/mean": results_df.reward.mean(),
                # Batch metrics
                "batch/solve_none": solve_none,
                "batch/solve_all": solve_all,
                "batch/effective_batch_size": effective_batch_size,
                # Error metrics
                "error/mean": (~results_df.error.isna()).mean(),
                **{
                    f"error/{error}": error_rate
                    for error, error_rate in results_df.error.dropna().value_counts(normalize=True).items()
                },
                # Env metrics
                **{f"metrics/{metric}": metrics_df[metric].mean() for metric in metrics_df.columns},
                # Time metrics
                "time/step": step_time,
                "time/generate_completions": generate_completions_time,
                "time/teacher_logprobs": teacher_logprobs_time,
                "time/save_ckpt": save_ckpt_time,
                # Scheduler metrics
                # **scheduler.get_metrics(),
                # Buffer metrics
                # **train_buffer.get_metrics(),
                # Event loop lag metrics
                # **self.event_loop_lag_monitor.get_metrics(),
                # W&B axis
                "step": self.progress.step,
            }

            # If more than one env, add per-env metrics
            if results_df.task.nunique() > 1:
                per_env_reward = results_df.groupby("task").reward.mean().to_dict()
                to_log.update({f"reward/{env}": reward for env, reward in per_env_reward.items()})

                per_env_ratio = results_df.task.value_counts(normalize=True).to_dict()
                to_log.update({f"batch/{env}": ratio for env, ratio in per_env_ratio.items()})

            # Log metrics to monitor(s)
            self.monitor.log(to_log)

            # Log samples to monitor(s) if enabled
            subset_train_rollouts = random.sample(train_rollouts, min(8, len(train_rollouts)))
            self.monitor.log_samples(subset_train_rollouts, step=self.progress.step)

            # Log distributions (rewards, advantages) if enabled
            self.monitor.log_distributions(
                distributions={
                    "rewards": rewards,
                    "advantages": advantages,
                },
                step=self.progress.step,
            )

            step_message = f"Step {self.progress.step} | Time: {step_time:.2f}s | Reward: {results_df.reward.mean():.4f} | Throughput: {throughput:.1f} tokens/s | Seq. Length: {results_df.groupby('example_id').seq_len.mean().mean():.1f} tokens/sample | Async Level: {self.train_scheduler.update_policy_scheduler.async_level} | Max. Off-Policy Level: {self.train_scheduler.update_policy_scheduler.max_off_policy_level}"
            self.logger.success(step_message)

            # Increment step
            self.progress.step += 1
            is_first_step = False

            self.event_loop_lag_monitor.reset()

            # Send heartbeat if configured
            if self.heart is not None:
                self.heart.beat()

        if self.config.eval:
            assert self.eval_scheduler is not None
            logger.info("Running final evals")
            tasks = []
            for env_name in self.eval_scheduler.env_worker_group.env_names:
                tasks.append(self.eval_scheduler.generate_batch(env_name))
            await asyncio.gather(*tasks)

        # Log final (immutable) samples and distributions to monitor(s)
        self.monitor.log_final_samples()
        self.monitor.save_final_summary()

        # Write final checkpoint
        if self.config.ckpt:
            assert self.ckpt_manager is not None
            logger.info("Writing final checkpoint")
            self.ckpt_manager.save(
                self.progress, self.train_scheduler.rollout_scheduler.buffer, step=self.progress.step
            )

        # Close training batch sender
        self.training_batch_sender.close()

        self.logger.success("Orchestrator finished.")

        # Optionally, print benchmark table
        if self.config.bench:
            print_benchmark(to_col_format(self.monitor.history))


def main():
    """Main entry-point for orchestrator. Run using `uv run orchestrator`"""

    config = parse_argv(OrchestratorConfig)
    asyncio.run(Orchestrator(config).start())


if __name__ == "__main__":
    main()
