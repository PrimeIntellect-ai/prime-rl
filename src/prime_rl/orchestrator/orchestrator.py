import asyncio
import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor

import tomli_w

from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.eval_utils import get_eval_sampling_args
from prime_rl.orchestrator.event_loop_lag import EventLoopLagMonitor
from prime_rl.orchestrator.patches import monkey_patch_chat_completion_logprobs, monkey_patch_oai_iterable_types
from prime_rl.orchestrator.trajectories import build_vlm_image_cache, interleave_rollout
from prime_rl.transport import TrainingBatch, TrainingSample, setup_training_batch_sender
from prime_rl.utils.pathing import get_log_dir

# This monkey patch is necessary to avoid Pydantic validating fields using typing.Iterable (e.g. in multimodal or tool call messages) lazily which leads to tokenization errors, for more info see https://github.com/PrimeIntellect-ai/prime-rl/pull/1249
monkey_patch_oai_iterable_types()


# This monkey patch is necessary to avoid heavy CPU overhead from constructing the OAI ChatCompletion Pydantic model with logprobs, for more info see https://github.com/PrimeIntellect-ai/prime-rl/pull/1189
monkey_patch_chat_completion_logprobs()

# Import environment before any other imports

import pandas as pd
import verifiers as vf
from transformers import AutoProcessor, AutoTokenizer

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.ckpt import Progress, setup_ckpt_manager
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.orchestrator.eval_utils import evaluate_env
from prime_rl.orchestrator.filters import apply_filters, setup_filters
from prime_rl.orchestrator.scheduler import Scheduler
from prime_rl.orchestrator.utils import (
    compute_teacher_logprobs,
    get_sampling_args,
    get_weight_dir,
    print_benchmark,
    set_semaphore,
)
from prime_rl.orchestrator.vf_utils import (
    get_completion_len,
    get_seq_len,
    intercept_vf_logging,
    setup_env_client,
    spawn_env_server,
    wait_for_env_servers,
)
from prime_rl.utils.client import (
    init_nccl_broadcast,
    setup_inference_pool,
)
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.temp_scheduling import compute_temperature
from prime_rl.utils.utils import (
    clean_exit,
    get_env_ids_to_install,
    install_env,
    resolve_latest_ckpt_step,
    strip_env_version,
    to_col_format,
)
from prime_rl.utils.vlm import is_vlm_model


def _should_run_eval(eval_config, ckpt_step: int, last_eval_step: int, is_final_step: bool) -> bool:
    """Return whether eval should run for the given checkpoint step."""
    if eval_config is None:
        return False
    if ckpt_step <= last_eval_step:
        return False
    if not (ckpt_step % eval_config.interval == 0 or is_final_step):
        return False
    if ckpt_step == 0:
        return eval_config.eval_base_model
    return True


def _aggregate_filter_metrics(metrics_df: pd.DataFrame, stop_conditions: list[str | None]) -> dict[str, float]:
    filter_columns = [column for column in metrics_df.columns if column.startswith("filter/")]
    if not filter_columns:
        return {}

    filter_df = metrics_df[filter_columns].fillna(0.0)
    n = len(filter_df)
    filter_metrics: dict[str, float] = {}

    for column in filter_columns:
        filter_metrics[f"{column}_count"] = float(filter_df[column].sum())
        filter_metrics[f"{column}_rate"] = float(filter_df[column].mean()) if n > 0 else 0.0

    if n > 0:
        detected_mask = filter_df.sum(axis=1) > 0
        filter_metrics["filter/total_detected_rate"] = float(detected_mask.mean())
        filter_metrics["filter/total_enforced_rate"] = float(
            sum(condition is not None for condition in stop_conditions) / n
        )
    else:
        filter_metrics["filter/total_detected_rate"] = 0.0
        filter_metrics["filter/total_enforced_rate"] = 0.0

    return filter_metrics


class Orchestrator:
    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.logger = get_logger()

    @clean_exit
    async def run(self) -> None:
        config = self.config
        self.logger = setup_logger(
            config.log.level,
            log_file=config.output_dir / "logs" / "orchestrator.log" if config.log.file else None,
            json_logging=config.log.json_logging,
        )
        intercept_vf_logging(logger="verifiers.workers", level=config.log.vf_level)  # show logs from env clients
        self.logger.info("Starting orchestrator")

        event_loop_lag_monitor = EventLoopLagMonitor()
        event_loop_lag_monitor_task = asyncio.create_task(event_loop_lag_monitor.run())

        # Print warning if running in benchmark mode
        if config.bench:
            self.logger.warning(f"Running in benchmark mode (max_steps={config.max_steps})")

        # Save configs to output directory
        config_dir = config.output_dir / "control"
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "orch.toml", "wb") as f:
            tomli_w.dump(config.model_dump(exclude_none=True, mode="json"), f)

        # Install environments
        env_ids_to_install = set()
        env_ids_to_install.update(get_env_ids_to_install(config.env))
        if config.eval is not None:
            env_ids_to_install.update(get_env_ids_to_install(config.eval.env))

        for env_id in env_ids_to_install:
            install_env(env_id)

        # Setup inference pool (handles both static and elastic modes)
        inference_pool = await setup_inference_pool(config.client, model_name=config.model.name)

        # Setup teacher inference pool if configured
        if config.teacher_model:
            self.logger.info(
                f"Initializing teacher inference pool (base_url={', '.join(config.teacher_model.client.base_url)}, "
                f"model={config.teacher_model.model.name})"
            )
            teacher_inference_pool = await setup_inference_pool(
                config.teacher_model.client, model_name=config.teacher_model.model.name
            )
        else:
            teacher_inference_pool = None

        # Check if this is a vision-language model (used throughout for VLM-specific paths)
        is_vlm = is_vlm_model(config.model.name)

        # Load tokenizer and processor (processor only for VLM models)
        self.logger.info(f"Initializing tokenizer for {config.model.name}")
        tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=config.model.trust_remote_code)

        processor = None
        if is_vlm:
            self.logger.info(f"Loading VLM processor for {config.model.name}")
            processor = AutoProcessor.from_pretrained(
                config.model.name, trust_remote_code=config.model.trust_remote_code, use_fast=True
            )

        # Build rollout filters
        rollout_filters = setup_filters(config.filters, vocab_size=tokenizer.vocab_size)
        if rollout_filters:
            self.logger.info(
                f"Initialized {len(rollout_filters)} rollout filter(s): {[f.name for f in rollout_filters]}"
            )

        # Setup monitor
        self.logger.info(f"Initializing monitor (wandb={config.wandb}, prime_monitor={config.prime_monitor})")
        monitor = setup_monitor(
            wandb_config=config.wandb,
            prime_config=config.prime_monitor,
            output_dir=config.output_dir,
            tokenizer=tokenizer,
            run_config=config,
        )

        # Setup heartbeat (only on rank 0, orchestrator is single process)
        heart = None
        if config.heartbeat is not None:
            self.logger.info("Initializing heartbeat")
            heart = Heartbeat(config.heartbeat.url)

        # Load environment and extract dataset
        self.logger.info(
            f"Loading {len(config.env)} training environment(s) ({', '.join(env.name or env.id for env in config.env)})"
        )
        env_ids = [strip_env_version(env.id) for env in config.env]
        train_env_names = [env.name or env_id for env_id, env in zip(env_ids, config.env)]
        train_env_group = vf.EnvGroup(
            envs=[vf.load_environment(env_id, **env.args) for env_id, env in zip(env_ids, config.env)],
            env_names=train_env_names,
            map_kwargs=dict(writer_batch_size=1),  # set defensively to not error on map operations on large datasets
        )

        train_env_clients, train_env_processes = await self._setup_env_servers(config.env, env_ids, train_env_names, "train")
        for env, env_client in zip(train_env_group.envs, train_env_clients):
            env.env_client = env_client

        if config.eval:
            env_ids = [strip_env_version(env.id) for env in config.eval.env]
            eval_envs = [vf.load_environment(env_id, **env.args) for env_id, env in zip(env_ids, config.eval.env)]
            eval_env_names = [env.name or env_id for env_id, env in zip(env_ids, config.eval.env)]
            eval_sampling_args = get_eval_sampling_args(config.eval.sampling)

            eval_env_clients, eval_env_processes = await self._setup_env_servers(config.eval.env, env_ids, eval_env_names, "eval")
            for eval_env, eval_env_client in zip(eval_envs, eval_env_clients):
                eval_env.env_client = eval_env_client
        else:
            eval_envs: list[vf.Environment] = []
            eval_env_names: list[str] = []
            eval_sampling_args = {}

        # Setup buffer
        self.logger.info(f"Setting up buffer ({config.buffer})")
        train_dataset = train_env_group.get_dataset(seed=config.buffer.seed)
        buffer = Buffer(train_dataset, train_env_group.env_names, config.buffer)

        # Get checkpoint manager
        self.logger.info(f"Initializing checkpoint manager ({config.ckpt})")
        ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)

        checkpoint_step = None
        if config.ckpt and config.ckpt.resume_step is not None and ckpt_manager is not None:
            if config.ckpt.resume_step == -1:
                checkpoint_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
            else:
                checkpoint_step = config.ckpt.resume_step

        scheduler = Scheduler(
            env=train_env_group,
            buffer=buffer,
            inference_pool=inference_pool,
            config=config,
        )

        if checkpoint_step is not None and config.model.lora is not None:
            assert config.model.lora.name is not None
            scheduler.model_name = config.model.lora.name

        # Check health of the inference pool
        self.logger.info("Waiting for inference pool to be ready")
        await inference_pool.wait_for_ready(config.model.name)
        self.logger.success("Inference pool ready")

        # Check health of teacher inference server if configured
        if config.teacher_model and teacher_inference_pool:
            self.logger.info("Waiting for teacher inference pool to be ready")
            await teacher_inference_pool.wait_for_ready(config.teacher_model.model.name)
            self.logger.success("Teacher inference pool ready")

        # Set up weight broadcast backend
        self.logger.info(f"Initializing weight broadcast ({config.weight_broadcast})")
        if config.weight_broadcast.type == "nccl":
            await init_nccl_broadcast(
                inference_pool.admin_clients,
                config.weight_broadcast.host,
                config.weight_broadcast.port,
                config.weight_broadcast.timeout,
            )

        # Setup training batch sender for sending training examples to trainer
        self.logger.info(f"Initializing training batch sender ({config.rollout_transport})")
        training_batch_sender = setup_training_batch_sender(config.output_dir, config.rollout_transport)

        self.last_eval_step = -1
        progress = Progress()
        self.scheduler = scheduler
        self.buffer = buffer
        self.monitor = monitor
        self.progress = progress
        self.tokenizer = tokenizer
        self.event_loop_lag_monitor = event_loop_lag_monitor

        if checkpoint_step is not None and ckpt_manager is not None:
            ckpt_manager.load(progress, buffer, step=checkpoint_step)
            self.logger.info(f"Resuming training from checkpoint step {checkpoint_step}")
            scheduler.ckpt_step = checkpoint_step
            if config.eval and config.eval.skip_eval_on_resume:
                self.last_eval_step = scheduler.ckpt_step
                self.logger.info(f"Skipping online eval on resume (ckpt_step={scheduler.ckpt_step})")

            # In NCCL mode, skip existence check - weights are broadcasted, not stored on disk
            check_exists = config.weight_broadcast.type != "nccl"
            wait_timeout = config.ckpt.wait_for_weights_timeout if config.ckpt else None
            weights_path = get_weight_dir(
                config.output_dir, scheduler.ckpt_step, check_exists=check_exists, wait_timeout=wait_timeout
            )
            lora_name = config.model.lora.name if config.model.lora else None
            await inference_pool.update_weights(weights_path, lora_name=lora_name, step=scheduler.ckpt_step)
        else:
            self.logger.info("Training from scratch")

        # Continuously get group rollouts and send them to the trainer.
        # The trainer decides when to step based on the configured batch budget.
        self.logger.info(f"Starting orchestrator streaming loop (max_steps={config.max_steps or 'infinite'})")
        await set_semaphore(config.max_concurrent or -1)

        # Set initial temperature / sampling args
        initial_ckpt_step = scheduler.ckpt_step
        progress.step = initial_ckpt_step
        temperature = compute_temperature(progress.step, config.sampling, config.max_steps)
        sampling_args = get_sampling_args(config.sampling, temperature=temperature)
        scheduler.set_sampling_args(sampling_args)

        async def maybe_run_eval(ckpt_step: int, is_final_step: bool = False) -> None:
            if not _should_run_eval(config.eval, ckpt_step, self.last_eval_step, is_final_step):
                return

            assert config.eval is not None
            self.last_eval_step = ckpt_step
            self.logger.info(f"Running evals for checkpoint step {ckpt_step}")

            # Pause weight updates and re-scheduling of training rollouts during eval
            # to avoid evaluating across different checkpoints and avoid congestion
            scheduler.checkpoint_ready.clear()

            # For heavy eval workloads, it might be necessary additionally cancel in-flight training rollouts
            if config.eval.cancel_inflight_rollouts_on_eval:
                self.logger.info("Cancelling in-flight training rollouts before starting evals to avoid congestion.")
                scheduler.cancel_inflight_rollouts()

            try:
                await asyncio.gather(
                    *[
                        evaluate_env(
                            env=eval_env,
                            env_name=eval_env_name,
                            get_client=inference_pool.get_next_client,
                            model_name=scheduler.model_name,
                            sampling_args=eval_sampling_args,
                            num_examples=eval_env_config.num_examples or config.eval.num_examples,
                            rollouts_per_example=eval_env_config.rollouts_per_example
                            or config.eval.rollouts_per_example,
                            max_retries=eval_env_config.max_retries,
                            ckpt_step=ckpt_step,
                            step=progress.step,
                        )
                        for eval_env, eval_env_name, eval_env_config in zip(eval_envs, eval_env_names, config.eval.env)
                    ]
                )
            finally:
                scheduler.checkpoint_ready.set()

        initial_is_final_step = bool(config.max_steps and initial_ckpt_step >= config.max_steps)
        await maybe_run_eval(initial_ckpt_step, is_final_step=initial_is_final_step)

        # Start update policy loop
        update_policy_task = asyncio.create_task(scheduler.update_policy_loop())

        # Track consecutive empty batches for retry logic
        empty_batch_retries = 0
        max_empty_batch_retries = 5

        # Persistent ThreadPoolExecutor for parallel rollout processing
        rollout_executor = ThreadPoolExecutor(max_workers=64)

        # Monotonic send counter for TrainingBatch ordering
        send_counter = 0

        # Accumulate rollouts between ckpt_step changes for aggregate logging
        accumulated_rollouts: list[vf.RolloutOutput] = []
        accumulated_advantages: list[float] = []
        accumulated_prefill_lens: list[int] = []
        accumulated_decode_lens: list[int] = []
        step_start_time = time.perf_counter()
        prev_ckpt_step = scheduler.ckpt_step
        evicted_path = config.output_dir / "control" / "evicted.txt"

        while True:
            # Check if this run has been evicted by the trainer
            if evicted_path.exists():
                reason = evicted_path.read_text().strip()
                raise RuntimeError(f"Run evicted by trainer: {reason}")

            # Check if update_policy_task has failed and propagate the exception
            if update_policy_task.done():
                for task in asyncio.all_tasks():
                    task.cancel()
                update_policy_task.result()

            # Capture ckpt_step once for consistency (it's updated by update_policy_loop concurrently)
            ckpt_step = scheduler.ckpt_step

            # On ckpt_step change: log accumulated metrics, run evals, save checkpoint, update temperature
            if ckpt_step > prev_ckpt_step:
                if accumulated_rollouts:
                    progress.step = prev_ckpt_step
                    self._log_accumulated_metrics(
                        accumulated_rollouts,
                        accumulated_advantages,
                        accumulated_prefill_lens,
                        accumulated_decode_lens,
                        prev_ckpt_step,
                        step_start_time,
                    )
                    accumulated_rollouts = []
                    accumulated_advantages = []
                    accumulated_prefill_lens = []
                    accumulated_decode_lens = []
                    step_start_time = time.perf_counter()
                progress.step = ckpt_step

                # Save checkpoint when reaching an interval or final step
                is_final_step = bool(config.max_steps is not None and ckpt_step >= config.max_steps)
                if (
                    ckpt_manager is not None
                    and ckpt_step > 0
                    and (
                        is_final_step
                        or (config.ckpt and config.ckpt.interval is not None and ckpt_step % config.ckpt.interval == 0)
                    )
                ):
                    self.logger.info(f"Saving checkpoint at ckpt_step {ckpt_step}")
                    ckpt_manager.save(progress, buffer, step=ckpt_step)

                await maybe_run_eval(ckpt_step, is_final_step=is_final_step)

                # Update temperature
                temperature = compute_temperature(ckpt_step, config.sampling, config.max_steps)
                sampling_args = get_sampling_args(config.sampling, temperature=temperature)
                scheduler.set_sampling_args(sampling_args)

                prev_ckpt_step = ckpt_step

            # Break if we have reached the maximum number of checkpoint steps
            if config.max_steps is not None and ckpt_step >= config.max_steps:
                break

            # Wait for one completed group rollout and process it
            train_rollouts = await scheduler.next_completed_group()
            if not train_rollouts:
                continue

            # Apply rollout filters (zeros reward/mask for degenerate generations)
            apply_filters(rollout_filters, train_rollouts)

            # Compute advantages
            example_ids = [r["example_id"] for r in train_rollouts]
            rewards = [r["reward"] for r in train_rollouts]
            completion_lens = [get_completion_len(r) for r in train_rollouts]
            advantages = compute_advantages(
                rewards,
                completion_lens,
                config.rollouts_per_example,
                config.advantage,
            )

            # Convert rollouts to training samples
            parallel_preprocess_start = time.perf_counter()
            num_unique_examples = len(set(example_ids))

            # VLM: build image cache for efficient batched preprocessing
            if is_vlm:
                vlm_cache = build_vlm_image_cache(train_rollouts, processor)
                self.logger.info(
                    f"VLM timing: extract={vlm_cache.extract_time:.2f}s, preprocess={vlm_cache.preprocess_time:.2f}s"
                )
            else:
                vlm_cache = None

            # Process rollouts in parallel
            def process_rollout(rollout: vf.RolloutOutput, rollout_idx: int) -> list[TrainingSample] | None:
                return interleave_rollout(rollout, vlm_cache=vlm_cache, cache_key=rollout_idx)

            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(rollout_executor, process_rollout, r, rollout_idx)
                for rollout_idx, r in enumerate(train_rollouts)
            ]
            results = await asyncio.gather(*futures)

            # Collect results and assign advantages
            train_examples: list[TrainingSample] = []
            rollout_prefill_lens: list[int] = []
            rollout_decode_lens: list[int] = []
            for rollout, advantage, samples in zip(train_rollouts, advantages, results):
                rollout_prefill_tokens = 0
                rollout_decode_tokens = 0
                if samples is not None:
                    for sample in samples:
                        sample.advantage = advantage
                        sample.reward = rollout["reward"]
                        sample_decode_tokens = sum(sample.completion_mask)
                        sample_prefill_tokens = (
                            len(sample.prompt_ids) + len(sample.completion_mask) - sample_decode_tokens
                        )
                        rollout_decode_tokens += sample_decode_tokens
                        rollout_prefill_tokens += sample_prefill_tokens
                        train_examples.append(sample)
                rollout_prefill_lens.append(rollout_prefill_tokens)
                rollout_decode_lens.append(rollout_decode_tokens)

            parallel_preprocess_time = time.perf_counter() - parallel_preprocess_start
            self.logger.debug(
                f"Converted {len(train_rollouts)} rollouts ({num_unique_examples} unique examples) "
                f"to {len(train_examples)} training examples in {parallel_preprocess_time:.2f}s"
            )

            # Retry with exponential backoff if batch is empty (e.g., inference temporarily unavailable)
            if len(train_examples) == 0:
                empty_batch_retries += 1
                if empty_batch_retries >= max_empty_batch_retries:
                    raise RuntimeError(
                        f"Step {progress.step} failed after {max_empty_batch_retries} consecutive empty batches"
                    )
                backoff = min(30 * (2 ** (empty_batch_retries - 1)), 300)  # 30s, 60s, 120s, 240s, 300s cap
                self.logger.warning(
                    f"Step {progress.step} produced 0 training samples "
                    f"(attempt {empty_batch_retries}/{max_empty_batch_retries}). Retrying in {backoff}s..."
                )
                await asyncio.sleep(backoff)
                continue

            # Reset retry counter on successful batch
            empty_batch_retries = 0

            if any(advantage != 0.0 for advantage in advantages):
                # Compute teacher logprobs if teacher model is configured
                if config.teacher_model and teacher_inference_pool:
                    teacher_logprobs_list = await compute_teacher_logprobs(
                        clients=teacher_inference_pool.clients,
                        model_name=config.teacher_model.model.name,
                        samples=train_examples,
                    )
                    for train_example, teacher_logprobs in zip(train_examples, teacher_logprobs_list):
                        train_example.teacher_logprobs = teacher_logprobs

                training_batch_sender.send(
                    TrainingBatch(
                        examples=train_examples,
                        step=send_counter,
                    )
                )
                send_counter += 1

            # Accumulate for logging
            accumulated_rollouts.extend(train_rollouts)
            accumulated_advantages.extend(advantages)
            accumulated_prefill_lens.extend(rollout_prefill_lens)
            accumulated_decode_lens.extend(rollout_decode_lens)
            progress.total_tokens += sum(get_seq_len(r) for r in train_rollouts)
            progress.total_samples += len(train_rollouts)
            progress.total_problems += len({r["example_id"] for r in train_rollouts})

            # Send heartbeat if configured
            if heart is not None:
                heart.beat()

        # Flush any remaining accumulated metrics
        if accumulated_rollouts:
            progress.step = prev_ckpt_step
            self._log_accumulated_metrics(
                accumulated_rollouts,
                accumulated_advantages,
                accumulated_prefill_lens,
                accumulated_decode_lens,
                prev_ckpt_step,
                step_start_time,
            )

        # Ensure final-step eval runs once at most.
        await maybe_run_eval(scheduler.ckpt_step, is_final_step=True)

        # Log final (immutable) samples and distributions to monitor(s)
        monitor.log_final_samples()
        monitor.save_final_summary()

        # Write final checkpoint
        if ckpt_manager is not None:
            self.logger.info("Writing final checkpoint")
            ckpt_manager.save(progress, buffer, step=progress.step)

        # Close training batch sender
        training_batch_sender.close()

        # Shutdown rollout executor
        rollout_executor.shutdown(wait=False)

        # Stop scheduler
        scheduler.cancel_inflight_rollouts()
        update_policy_task.cancel()

        # Stop inference pool
        await inference_pool.stop()

        if teacher_inference_pool is not None:
            await teacher_inference_pool.stop()

        # Cancel event loop lag monitor task
        event_loop_lag_monitor_task.cancel()

        # Shutdown env processes
        for process in train_env_processes + eval_env_processes:
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)

        self.logger.success("Orchestrator finished.")

        # Optionally, print benchmark table
        if config.bench:
            print_benchmark(to_col_format(monitor.history))

    async def _setup_env_servers(self, env_configs, env_ids: list[str], env_names: list[str], label: str) -> tuple[list, list[mp.Process]]:
        """Spawn or connect to environment servers, wait for readiness, and return clients."""
        config = self.config
        addresses = []
        env_processes: list[mp.Process] = []
        for env_id, env, env_name in zip(env_ids, env_configs, env_names):
            if env.address is None:
                address, process = spawn_env_server(
                    env_id=env_id,
                    env_args=env.args,
                    extra_env_kwargs=env.extra_env_kwargs,
                    log_level="CRITICAL",
                    log_file=(get_log_dir(config.output_dir) / label / f"{env_name}.log").as_posix(),
                    log_file_level=config.log.vf_level,
                    json_logging=config.log.json_logging,
                )
                env_processes.append(process)
            else:
                address = env.address
            self.logger.info(f"Connecting {label} environment {env_name} to server at {address}")
            addresses.append(address)
        clients = [setup_env_client(address=address, name=name) for name, address in zip(env_names, addresses)]
        self.logger.info(f"Waiting for {label} environment servers to be ready")
        await wait_for_env_servers(clients)
        self.logger.success(f"{label.capitalize()} environment servers ready")
        return clients, env_processes

    def _log_accumulated_metrics(
        self,
        accumulated_rollouts: list[vf.RolloutOutput],
        accumulated_advantages: list[float],
        accumulated_prefill_lens: list[int],
        accumulated_decode_lens: list[int],
        ckpt_step: int,
        step_start_time: float,
    ) -> None:
        """Log aggregated metrics for all rollouts accumulated between ckpt_step changes."""
        config = self.config
        scheduler = self.scheduler
        buffer = self.buffer
        monitor = self.monitor
        progress = self.progress
        event_loop_lag_monitor = self.event_loop_lag_monitor

        results_df = pd.DataFrame(
            {
                "example_id": [r["example_id"] for r in accumulated_rollouts],
                "task": [r["task"] for r in accumulated_rollouts],
                "reward": [r["reward"] for r in accumulated_rollouts],
                "is_truncated": [r["is_truncated"] for r in accumulated_rollouts],
                "error": [r["error"] for r in accumulated_rollouts],
                "seq_len": [get_seq_len(r) for r in accumulated_rollouts],
                "prefill_len": accumulated_prefill_lens,
                "decode_len": accumulated_decode_lens,
                "num_turns": [len(r["trajectory"]) for r in accumulated_rollouts],
                "generation_ms": [r["timing"]["generation_ms"] for r in accumulated_rollouts],
                "scoring_ms": [r["timing"]["scoring_ms"] for r in accumulated_rollouts],
            }
        )
        metrics_df = pd.DataFrame([r["metrics"] for r in accumulated_rollouts])
        stop_conditions = [r.get("stop_condition") for r in accumulated_rollouts]
        filter_metrics = _aggregate_filter_metrics(metrics_df, stop_conditions)
        rollout_metric_columns = [column for column in metrics_df.columns if not column.startswith("filter/")]

        num_tokens = int(results_df.seq_len.sum())
        num_samples = len(accumulated_rollouts)
        num_problems = int(results_df.example_id.nunique())
        step_time = time.perf_counter() - step_start_time
        throughput = num_tokens / step_time if step_time > 0 else 0

        by_example = results_df.groupby("example_id")
        group_reward_sums = [
            sum(r["reward"] for r in accumulated_rollouts[i : i + config.rollouts_per_example])
            for i in range(0, len(accumulated_rollouts), config.rollouts_per_example)
        ]
        num_groups = len(group_reward_sums)
        solve_all = sum(reward_sum == config.rollouts_per_example for reward_sum in group_reward_sums) / num_groups
        solve_none = sum(reward_sum == 0 for reward_sum in group_reward_sums) / num_groups
        effective_batch_size = 1 - solve_none - solve_all

        temperature = compute_temperature(ckpt_step, config.sampling, config.max_steps)

        to_log = {
            "progress/tokens": num_tokens,
            "progress/prefill_tokens": int(sum(accumulated_prefill_lens)),
            "progress/decode_tokens": int(sum(accumulated_decode_lens)),
            "progress/samples": num_samples,
            "progress/problems": num_problems,
            "progress/total_tokens": progress.total_tokens,
            "progress/total_samples": progress.total_samples,
            "progress/total_problems": progress.total_problems,
            "progress/ckpt_step": ckpt_step,
            "seq_len/mean": by_example.seq_len.mean().mean(),
            "seq_len/max": by_example.seq_len.mean().max(),
            "seq_len/min": by_example.seq_len.mean().min(),
            "prefill_len/mean": by_example.prefill_len.mean().mean(),
            "prefill_len/max": by_example.prefill_len.mean().max(),
            "prefill_len/min": by_example.prefill_len.mean().min(),
            "decode_len/mean": by_example.decode_len.mean().mean(),
            "decode_len/max": by_example.decode_len.mean().max(),
            "decode_len/min": by_example.decode_len.mean().min(),
            "is_truncated/mean": by_example.is_truncated.mean().mean(),
            "is_truncated/max": by_example.is_truncated.mean().max(),
            "is_truncated/min": by_example.is_truncated.mean().min(),
            "num_turns/mean": by_example.num_turns.mean().mean(),
            "num_turns/max": by_example.num_turns.mean().max(),
            "num_turns/min": by_example.num_turns.mean().min(),
            "generation_ms/mean": by_example.generation_ms.mean().mean(),
            "generation_ms/max": by_example.generation_ms.mean().max(),
            "generation_ms/min": by_example.generation_ms.mean().min(),
            "scoring_ms/mean": by_example.scoring_ms.mean().mean(),
            "scoring_ms/max": by_example.scoring_ms.mean().max(),
            "scoring_ms/min": by_example.scoring_ms.mean().min(),
            "perf/throughput": throughput,
            "reward/mean": results_df.reward.mean(),
            "sampling/temperature": temperature,
            "batch/solve_none": solve_none,
            "batch/solve_all": solve_all,
            "batch/effective_batch_size": effective_batch_size,
            "error/mean": (~results_df.error.isna()).mean(),
            **{
                f"error/{error}": error_rate
                for error, error_rate in results_df.error.dropna()
                .apply(lambda e: e.get("error") if isinstance(e, dict) else e)
                .value_counts(normalize=True)
                .items()
            },
            **{f"metrics/{metric}": metrics_df[metric].mean() for metric in rollout_metric_columns},
            **filter_metrics,
            "time/step": step_time,
            **scheduler.get_metrics(),
            **buffer.get_metrics(),
            **event_loop_lag_monitor.get_metrics(),
            "step": progress.step,
        }

        if results_df.task.nunique() > 1:
            per_env_reward = results_df.groupby("task").reward.mean().to_dict()
            to_log.update({f"reward/{env}": reward for env, reward in per_env_reward.items()})
            per_env_ratio = results_df.task.value_counts(normalize=True).to_dict()
            to_log.update({f"batch/{env}": ratio for env, ratio in per_env_ratio.items()})

        monitor.log(to_log)

        subset_train_rollouts = random.sample(accumulated_rollouts, min(8, len(accumulated_rollouts)))
        monitor.log_samples(subset_train_rollouts, step=progress.step)

        monitor.log_distributions(
            distributions={
                "rewards": [r["reward"] for r in accumulated_rollouts],
                "advantages": accumulated_advantages,
            },
            step=progress.step,
        )

        self.logger.success(
            f"Step {progress.step} | Time: {step_time:.2f}s | Reward: {results_df.reward.mean():.4f} | "
            f"Throughput: {throughput:.1f} tokens/s | Samples: {num_samples} | "
            f"Max. Off-Policy Level: {scheduler.max_off_policy_level}"
        )

        event_loop_lag_monitor.reset()


def main():
    """Main entry-point for orchestrator. Run using `uv run orchestrator`"""
    asyncio.run(Orchestrator(parse_argv(OrchestratorConfig)).run())


if __name__ == "__main__":
    main()
