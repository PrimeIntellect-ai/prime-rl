"""Async-pipelined RL orchestrator v2.

``Orchestrator`` is the only class that holds the whole picture: it owns the
shared state (policy, progress, ckpt, monitor) and drives the pipeline loop.

Components are deliberately single-purpose and only know about the deps they
need:

- ``RolloutDispatcher`` schedules rollouts and emits ``Rollout``\\ s on its queue.
- ``TrainSink`` ingests train rollouts (tokenize → advantages + pre-filter →
  post-filter), exposes ``pop_batch`` returning a ``TrainBatch``.
- ``EvalSink`` ingests eval rollouts, exposes ``add`` returning an ``EvalBatch``
  on epoch completion + ``build_metrics`` for log time.
- ``MetricsBuilder`` builds the per-step train W&B dict from the popped batch
  and orchestrator-side timings (called inline by the orchestrator).
- ``WeightWatcher`` advances ``Policy`` and notifies the dispatcher.
- ``IntervalLogger`` writes dispatcher gauges + event-loop lag on a time axis.

None of these hold a reference to the orchestrator. The orchestrator wires
them in ``setup()`` and drives them from ``main_loop()``.
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import time

import tomli_w

import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before transitive imports
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.filters import setup_filters
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.patches import (
    monkey_patch_chat_completion_logprobs,
    monkey_patch_oai_iterable_types,
)
from prime_rl.orchestrator.trajectories import offload_images_to_disk
from prime_rl.orchestrator.utils import compute_teacher_logprobs, get_weight_dir, set_default_executor
from prime_rl.orchestrator.vf_utils import get_seq_len, intercept_vf_logging, save_rollouts
from prime_rl.orchestrator_v2.ckpt import setup_ckpt_manager
from prime_rl.orchestrator_v2.dispatcher import RolloutDispatcher
from prime_rl.orchestrator_v2.eval_sink import EvalSink
from prime_rl.orchestrator_v2.log_loop import IntervalLogger
from prime_rl.orchestrator_v2.metrics import MetricsBuilder
from prime_rl.orchestrator_v2.train_sink import TrainSink
from prime_rl.orchestrator_v2.types import EvalBatch, Policy, Progress, Rollout
from prime_rl.orchestrator_v2.watcher import WeightWatcher
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.transport import TrainingBatch, setup_training_batch_sender
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.client import init_nccl_broadcast, setup_inference_pool
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pathing import get_log_dir, get_rollout_dir, get_step_path
from prime_rl.utils.usage_reporter import UsageReporter
from prime_rl.utils.utils import (
    clean_exit,
    get_env_ids_to_install,
    install_env,
    resolve_latest_ckpt_step,
)

monkey_patch_oai_iterable_types()
monkey_patch_chat_completion_logprobs()


# Hard wall-clock budget for the v2 orchestrator's post-training cleanup.
# Mirrors the legacy orchestrator's safety: persist artifacts before this point,
# then force-exit if graceful shutdown wedges (env-server ZMQ recv, vLLM admin
# aclose, etc).
SHUTDOWN_TIMEOUT_S = 300


class Orchestrator:
    """v2 orchestrator. ``await Orchestrator(config).start()`` to run.

    ``stop()`` from outside (or the orchestrator self-stops when the batcher
    has driven ``progress.step`` to ``max_steps``). Cleanup happens at the tail
    of ``start()``.
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
        intercept_vf_logging(logger="verifiers.serve", level="WARN")
        self.logger.info(f"Starting orchestrator v2 ({config.training_mode})")

        if config.bench:
            self.logger.warning(f"Running in benchmark mode (max_steps={config.max_steps})")

        # Cheap synchronous setup. Heavy async work happens in ``setup()``.
        self.progress = Progress()
        self.ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)
        self.policy = Policy(version=0, model_name="")
        self.stopped = asyncio.Event()

        # Populated during ``setup()``. Typing as ``Any`` keeps the class
        # readable; the orchestrator's setup phase fills these in deterministic
        # order so we never see ``None`` past ``setup()``.
        self.tokenizer = None
        self.renderer = None
        self.mm_token_type_ids_mapping: dict[int, int] | None = None
        self.student_inference = None
        self.teacher_inference = None
        self.monitor = None
        self.heart: Heartbeat | None = None
        self.usage_reporter: UsageReporter | None = None
        self.inference_metrics: InferenceMetricsCollector | None = None
        self.train_envs: TrainEnvs | None = None
        self.eval_envs: EvalEnvs | None = None
        self.sender = None
        self.lora_name: str | None = None
        self.resume_step: int | None = None

        # Components — built at the end of ``setup()``.
        self.dispatcher: RolloutDispatcher | None = None
        self.train_sink: TrainSink | None = None
        self.eval_sink: EvalSink | None = None
        self.metrics: MetricsBuilder | None = None
        self.watcher: WeightWatcher | None = None
        self.log_loop: IntervalLogger | None = None

        # Background tasks for components with their own main loop.
        self.component_tasks: list[asyncio.Task] = []

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Async setup: install envs, load models/pools, prepare env workers,
        resume from checkpoint, and construct components."""
        config = self.config
        set_default_executor()

        # Persist the resolved config alongside the run.
        config_dir = config.output_dir / "control"
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "orch.toml", "wb") as f:
            tomli_w.dump(config.model_dump(exclude_none=True, mode="json"), f)

        # Install envs (train + eval).
        env_ids_to_install = set(get_env_ids_to_install(config.train.env))
        if config.eval is not None:
            env_ids_to_install.update(get_env_ids_to_install(config.eval.env))
        for env_id in env_ids_to_install:
            install_env(env_id, prerelease=config.env_install_prerelease)

        self.logger.info(f"Initializing tokenizer ({config.tokenizer})")
        self.tokenizer = setup_tokenizer(config.tokenizer)

        # Student inference pool.
        self.logger.info(
            f"Initializing student inference pool (base_url={', '.join(config.student.client.base_url)}, "
            f"model={config.student.model.name})"
        )
        self.renderer, self.student_inference = await setup_student_inference_pool(
            config=config, tokenizer=self.tokenizer
        )
        self.mm_token_type_ids_mapping = (
            getattr(self.renderer, "mm_token_type_id_map", None) if self.renderer is not None else None
        )
        if self.mm_token_type_ids_mapping == {}:
            self.mm_token_type_ids_mapping = None

        if config.teacher is not None:
            self.logger.info(
                f"Initializing teacher inference pool (base_url={', '.join(config.teacher.client.base_url)}, "
                f"model={config.teacher.model.name})"
            )
            self.teacher_inference = await setup_inference_pool(
                config.teacher.client,
                model_name=config.teacher.model.name,
                train_client_type="openai_chat_completions",
            )

        self.logger.info(f"Initializing monitor (wandb={config.wandb}, prime_monitor={config.prime_monitor})")
        self.monitor = setup_monitor(
            wandb_config=config.wandb,
            prime_config=config.prime_monitor,
            output_dir=config.output_dir,
            tokenizer=self.tokenizer,
            run_config=config,
            keep_full_history=config.bench,
        )

        if config.heartbeat is not None:
            self.heart = Heartbeat(config.heartbeat.url)

        usage_base_url = os.environ.get("PI_USAGE_BASE_URL")
        usage_api_key = os.environ.get("PI_USAGE_API_KEY")
        if usage_base_url and usage_api_key:
            self.usage_reporter = UsageReporter()

        # Filters — train-only by design (eval rollouts are never filtered).
        pre_filters = setup_filters(config.pre_batch_filters, vocab_size=self.tokenizer.vocab_size)
        post_filters = setup_filters(config.post_batch_filters, vocab_size=self.tokenizer.vocab_size)

        # Envs.
        self.logger.info("Loading training environments")
        self.train_envs = TrainEnvs(config.train.env)
        if config.training_mode == "sft":
            for env in self.train_envs:
                env.sampling_args.pop("logprobs", None)
        self.logger.info(f"Loaded {len(self.train_envs)} training environment(s) ({', '.join(self.train_envs.names)})")
        await self.train_envs.start(
            log_dir=get_log_dir(config.output_dir.parent) / "envs" / "train",
            log_level=config.log.vf_level,
            json_logging=config.log.json_logging,
        )
        self.logger.success("Train environment(s) ready")

        if config.eval is not None:
            self.logger.info("Loading eval environment(s)")
            self.eval_envs = EvalEnvs(config.eval.env)
            self.logger.info(f"Loaded {len(self.eval_envs)} eval environment(s) ({', '.join(self.eval_envs.names)})")
            await self.eval_envs.start(
                log_dir=get_log_dir(config.output_dir.parent) / "envs" / "eval",
                log_level=config.log.vf_level,
                json_logging=config.log.json_logging,
            )
            self.logger.success("Eval environment(s) ready")

        # Resume.
        if config.ckpt is not None and config.ckpt.resume_step is not None and self.ckpt_manager is not None:
            if config.ckpt.resume_step == -1:
                self.resume_step = resolve_latest_ckpt_step(self.ckpt_manager.ckpt_dir)
            else:
                self.resume_step = config.ckpt.resume_step

        # Initial policy state — student model name; on resume below we may
        # advance ``policy.version`` and the LoRA-adapter model name.
        self.policy.model_name = self.student_inference.model_name

        # Wait for inference pools to be reachable.
        self.logger.info("Waiting for student inference pool to be ready")
        await self.student_inference.wait_for_ready(config.student.model.name)
        self.logger.success("Student inference pool ready")
        if self.teacher_inference is not None:
            assert config.teacher is not None
            self.logger.info("Waiting for teacher inference pool to be ready")
            await self.teacher_inference.wait_for_ready(config.teacher.model.name)
            self.logger.success("Teacher inference pool ready")

        if config.wandb is not None and config.collect_inference_metrics:
            self.inference_metrics = InferenceMetricsCollector(self.student_inference.admin_clients)
            await self.inference_metrics.start()

        self.logger.info(f"Initializing weight broadcast ({config.weight_broadcast})")
        if config.weight_broadcast.type == "nccl":
            await init_nccl_broadcast(
                self.student_inference.admin_clients,
                config.weight_broadcast.host,
                config.weight_broadcast.port,
                config.weight_broadcast.timeout,
                inference_world_size=config.weight_broadcast.inference_world_size,
                quantize_in_weight_transfer=config.weight_broadcast.quantize_in_weight_transfer,
            )

        self.logger.info(f"Initializing training batch sender ({config.rollout_transport})")
        self.sender = setup_training_batch_sender(config.output_dir, config.rollout_transport)

        self.lora_name = config.student.model.lora.name if config.student.model.lora else None

        # Restore from checkpoint (if resuming).
        if self.resume_step is not None and self.ckpt_manager is not None:
            self.ckpt_manager.load(self.progress, step=self.resume_step)
            self.logger.info(f"Resuming v2 orchestrator from checkpoint step {self.resume_step}")
            check_exists = config.weight_broadcast.type != "nccl"
            wait_timeout = config.ckpt.wait_for_weights_timeout if config.ckpt else None
            weights_path = get_weight_dir(
                config.output_dir, self.progress.step, check_exists=check_exists, wait_timeout=wait_timeout
            )
            await self.student_inference.update_weights(weights_path, lora_name=self.lora_name, step=self.progress.step)
            if self.lora_name is not None:
                self.student_inference.update_model_name(self.lora_name)
                self.policy.model_name = self.lora_name
            self.policy.version = self.progress.step
        else:
            self.logger.info("Training from scratch")

        # ── Construct components — each gets only the deps it needs. ─────
        self.dispatcher = RolloutDispatcher(
            config=config,
            train_envs=self.train_envs,
            eval_envs=self.eval_envs,
            student_inference=self.student_inference,
            teacher_inference=self.teacher_inference,
            policy=self.policy,
            resume_step=self.resume_step,
        )
        self.metrics = MetricsBuilder(config)
        self.train_sink = TrainSink(
            config,
            tokenizer=self.tokenizer,
            renderer=self.renderer,
            mm_token_type_ids_mapping=self.mm_token_type_ids_mapping,
            batch_size=config.batch_size,
            token_batch_size=config.token_batch_size,
            advantage_config=config.advantage,
            pre_filters=pre_filters,
            post_filters=post_filters,
        )
        self.eval_sink = EvalSink(eval_envs=self.eval_envs)
        self.watcher = WeightWatcher(
            config,
            policy=self.policy,
            student_inference=self.student_inference,
            observers=[self.dispatcher],
            lora_name=self.lora_name,
            ckpt_step=self.progress.step,
        )
        self.log_loop = IntervalLogger(
            dispatcher=self.dispatcher,
            policy=self.policy,
            interval=config.experimental.log_loop_interval,
        )

    async def start(self) -> None:
        """Run the orchestrator until shutdown. Drives setup, spawns the
        background tasks, runs the main loop in this task, then cleans up."""
        await self.setup()
        config = self.config
        self.logger.info(f"Starting orchestrator v2 loop (max_steps={config.max_steps or 'infinite'})")
        start_time = time.perf_counter()

        # Spawn background loops (dispatcher schedules, watcher polls, log_loop
        # emits gauges). The pipeline ``main_loop`` runs inline in this task.
        assert self.dispatcher and self.watcher and self.log_loop
        self.component_tasks = [
            asyncio.create_task(self.dispatcher.start(), name="dispatcher"),
            asyncio.create_task(self.watcher.start(), name="watcher"),
            asyncio.create_task(self.log_loop.start(), name="log_loop"),
        ]

        try:
            await self.main_loop()
        finally:
            self.logger.success(f"Orchestrator v2 step loop done in {time.perf_counter() - start_time:.1f}s")
            if self.eval_envs is not None and config.eval is not None:
                self.logger.info("Running final evals")
                await asyncio.gather(
                    *(
                        eval_env.evaluate(
                            model_name=self.student_inference.model_name,
                            get_client=self.student_inference.get_eval_client,
                            step=self.progress.step,
                            cache_salt=str(self.progress.step),
                        )
                        for eval_env in self.eval_envs
                    ),
                    return_exceptions=True,
                )
            assert self.monitor is not None
            self.monitor.save_final_summary()
            if self.ckpt_manager is not None:
                self.logger.info("Writing final v2 checkpoint")
                self.ckpt_manager.save(self.progress, step=self.progress.step)
            await self.shutdown()
            self.logger.success("Orchestrator v2 finished.")
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception as e:
                get_logger().debug(f"malloc_trim(0) failed: {e}")

    async def stop(self) -> None:
        """Signal a graceful shutdown. ``start()`` will observe this on its
        next ``main_loop`` iteration and drive the rest of the teardown."""
        self.stopped.set()

    # ── pipeline ──────────────────────────────────────────────────────────

    async def main_loop(self) -> None:
        """The pipeline driver. Consumes ``Rollout``\\ s from the dispatcher
        (the atomic unit) and routes train vs eval to their respective sinks.

        Both sinks have the same three-level processing structure
        (``process_rollout`` / ``process_group`` / ``process_batch``) and
        share the same batch-boundary signal: ``rollout.is_batch_done``.
        The eval sink relies on the dispatcher to stamp the flag (last
        rollout of an env-epoch); the train sink stamps it itself inside
        ``add()`` when its buffer reaches ``batch_size``.
        """
        assert self.dispatcher and self.train_sink and self.eval_sink
        while not self.stopped.is_set():
            try:
                rollout: Rollout = await asyncio.wait_for(self.dispatcher.out_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if rollout.kind == "eval":
                eval_batch = self.eval_sink.add(rollout)
                if eval_batch is not None:
                    self.log_eval_batch(eval_batch)
                continue

            await self.train_sink.add(rollout)
            while rollout.is_batch_done and not self.stopped.is_set():
                await self.process_one_step()
                # Re-stamp from the sink's polling state: did another batch
                # complete while we were processing this one?
                rollout.is_batch_done = self.train_sink.is_batch_done()

    async def process_one_step(self) -> None:
        """Drive one shipping step. The train sink has done all per-rollout /
        per-group / per-batch processing already; the orchestrator handles the
        I/O concerns (ship to trainer, save rollouts, monitor log, heartbeat,
        usage reporter, ckpt save, step++)."""
        assert self.train_sink and self.metrics and self.dispatcher and self.monitor and self.sender
        config = self.config
        step = self.progress.step

        save_ckpt_time = await self.maybe_save_ckpt(step)

        if config.max_steps is not None and step >= config.max_steps:
            self.logger.success(f"Reached max_steps={config.max_steps}, signaling shutdown")
            self.stopped.set()
            return

        self.logger.info(f"Starting orchestrator step {step}")
        step_start = time.perf_counter()

        # Pop the next ready batch off the train sink. ``pop_batch`` returns
        # the raw cohort + the trainer-bound samples + per-rollout counters;
        # metrics are built at log time below with the final timings.
        batch = self.train_sink.pop_batch()

        if batch.result.n_trainable == 0:
            self.logger.warning(
                f"Step {step}: post-batch filters dropped all {len(batch.rollouts)} rollouts. Trying again."
            )
            return
        if batch.result.n_trainable / len(batch.rollouts) <= 0.1:
            self.logger.warning(
                f"Only {batch.result.n_trainable}/{len(batch.rollouts)} rollouts in the batch are trainable "
                f"({batch.result.n_trainable / len(batch.rollouts):.1%}) — consider reviewing task difficulty / filter config"
            )

        # Persist rollouts to disk (cheap, background thread).
        step_path = get_step_path(get_rollout_dir(config.output_dir), step)
        await asyncio.to_thread(
            save_rollouts, batch.rollouts, step_path / "train_rollouts.jsonl", exclude_keys={"trajectory"}
        )

        # Offload base64 image bytes to disk for memory hygiene (no-op for text-only).
        offload_start = time.perf_counter()
        num_offloaded = offload_images_to_disk(batch.rollouts, config.output_dir)
        if num_offloaded:
            self.logger.info(
                f"Offloaded {num_offloaded} unique images to disk in {time.perf_counter() - offload_start:.2f}s"
            )

        # Teacher logprobs (opd only).
        teacher_logprobs_time = 0.0
        if config.training_mode == "opd" and self.teacher_inference is not None:
            assert config.teacher is not None
            t = time.perf_counter()
            teacher_logprobs_list = await compute_teacher_logprobs(
                clients=self.teacher_inference.train_clients,
                model_name=config.teacher.model.name,
                samples=batch.samples,
            )
            for ex, lp in zip(batch.samples, teacher_logprobs_list):
                ex.teacher_logprobs = lp
            teacher_logprobs_time = time.perf_counter() - t

        await self.wait_barrier(step)

        # Ship to trainer.
        await self.sender.send(TrainingBatch(examples=batch.samples, step=step))

        # Build + ship metrics with the final post-barrier timings.
        step_time = time.perf_counter() - step_start
        metrics = self.metrics.build(
            step=step,
            rollouts=batch.rollouts,
            result=batch.result,
            progress=self.progress,
            dispatcher_gauges=self.dispatcher.gauges(),
            dispatcher_drain=self.dispatcher.drain_metrics(),
            step_time=step_time,
            save_ckpt_time=save_ckpt_time,
            teacher_logprobs_time=teacher_logprobs_time,
            pre_filter_seen=self.train_sink.pre_filter_seen,
            pre_filter_dropped=self.train_sink.pre_filter_dropped,
            pre_filter_dropped_by_name=dict(self.train_sink.pre_filter_dropped_by_name),
        )
        self.monitor.log(metrics, step=step)
        self.monitor.log_samples(batch.rollouts, step=step)
        self.monitor.log_distributions(
            distributions={
                "rewards": [r["reward"] for r in batch.rollouts],
                "advantages": [r["advantage"] for r in batch.rollouts],
            },
            step=step,
        )

        if self.usage_reporter is not None:
            run_id = os.getenv("RUN_ID", "")
            if run_id:
                self.usage_reporter.report_training_usage(
                    run_id=run_id,
                    step=step,
                    tokens=batch.result.num_prefill_tokens + batch.result.num_decode_tokens,
                )
        if self.heart is not None:
            self.heart.beat()

        # Update progress totals.
        num_rollouts = len(batch.rollouts)
        num_unique_examples = len({(r["env_name"], r["example_id"]) for r in batch.rollouts})
        num_tokens = sum(get_seq_len(r) for r in batch.rollouts)
        self.progress.total_tokens += num_tokens
        self.progress.total_samples += num_rollouts
        self.progress.total_problems += num_unique_examples

        reward_mean = float(sum(r["reward"] for r in batch.rollouts) / max(num_rollouts, 1))
        self.logger.success(
            f"Step {step} | Time: {step_time:.2f}s | Reward: {reward_mean:.4f} | "
            f"Seq. Length: {num_tokens / max(num_rollouts, 1):.1f} tokens/sample | "
            f"Trainable: {batch.result.n_trainable}/{num_rollouts} | "
            f"Async Level: {step - self.policy.version} | "
            f"Max. Off-Policy Level: {self.dispatcher.max_off_policy_level}"
        )

        self.train_sink.reset_pre_filter_stats()
        self.progress.step += 1

    def log_eval_batch(self, batch: EvalBatch) -> None:
        """Persist + log one completed eval epoch. Builds the metrics dict
        from the raw rollouts (the ``EvalBatch`` deliberately doesn't carry a
        pre-baked metrics dict — it's a pure view computed here), then handles
        the side effects: save_rollouts, monitor.log_eval_samples, monitor.log."""
        if not batch.rollouts:
            self.logger.warning(f"Eval @ step={batch.step} env={batch.env_name}: no surviving rollouts, skipping log")
            return

        assert self.eval_sink is not None and self.monitor is not None
        metrics = self.eval_sink.build_metrics(batch)

        step_path = get_step_path(get_rollout_dir(self.config.output_dir), batch.step)
        save_rollouts(
            batch.rollouts,
            step_path / f"eval_rollouts_{batch.env_name}.jsonl",
            exclude_keys={"trajectory"},
        )
        self.monitor.log_eval_samples(batch.rollouts, env_name=batch.env_name, step=batch.step)
        self.monitor.log(metrics, step=batch.step)

        n_examples = len({r["example_id"] for r in batch.rollouts})
        reward_mean = metrics.get(f"eval/{batch.env_name}/reward/mean", float("nan"))
        self.logger.success(
            f"Eval @ step={batch.step} env={batch.env_name} | "
            f"Reward: {reward_mean:.4f} | Rollouts: {len(batch.rollouts)} | Examples: {n_examples}"
        )

    async def maybe_save_ckpt(self, step: int) -> float:
        """Save the checkpoint if we're at an interval boundary. Returns the
        elapsed time (0.0 if no save happened)."""
        if self.ckpt_manager is None or self.config.ckpt is None or not self.config.ckpt.interval:
            return 0.0
        if step <= 0:
            return 0.0
        is_last_step = self.config.max_steps is not None and step == self.config.max_steps - 1
        if is_last_step:
            return 0.0
        if step % self.config.ckpt.interval != 0:
            return 0.0
        self.logger.info(f"Saving v2 checkpoint at step {step}")
        t = time.perf_counter()
        await asyncio.to_thread(self.ckpt_manager.save, self.progress, step)
        return time.perf_counter() - t

    async def wait_barrier(self, step: int) -> None:
        """Block until ``policy.version >= step - 1`` so the orchestrator stays
        at most one step ahead of the trainer.

        Cascades backpressure into the dispatcher via the bounded ``out_q``:
        while we wait, the queue fills, the dispatcher stops handing out new
        semaphore permits, and in-flight rollouts drain without new ones being
        scheduled.
        """
        target_lag = 1
        next_warn = 60.0
        t0 = time.perf_counter()
        while True:
            lead = step - self.policy.version
            if lead <= target_lag:
                return
            elapsed = time.perf_counter() - t0
            if elapsed >= next_warn:
                # Just a stall observation — no speculation about cause.
                self.logger.info(
                    f"Orchestrator waiting at async barrier ({int(elapsed)}s): step={step}, "
                    f"policy.version={self.policy.version}, lead={lead} (max_async_level={target_lag})."
                )
                next_warn = elapsed + 60.0
            await asyncio.sleep(0.1)

    # ── shutdown ──────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Bounded best-effort cleanup. Has a global timeout so a wedged peer
        can't keep the process alive forever — training artifacts are already
        persisted before this is reached."""

        async def do_shutdown() -> None:
            if self.sender is not None:
                self.sender.close()
            if self.dispatcher is not None:
                await self.dispatcher.stop()
            if self.watcher is not None:
                await self.watcher.stop()
            if self.log_loop is not None:
                await self.log_loop.stop()
            for task in self.component_tasks:
                await safe_cancel(task)
            self.component_tasks.clear()
            if self.inference_metrics is not None:
                await self.inference_metrics.stop()
            if self.student_inference is not None:
                await self.student_inference.stop()
            if self.teacher_inference is not None:
                await self.teacher_inference.stop()
            if self.train_envs is not None:
                self.train_envs.shutdown()
            if self.eval_envs is not None:
                self.eval_envs.shutdown()
            if self.usage_reporter is not None:
                self.usage_reporter.close()

        task = asyncio.create_task(do_shutdown())
        _, pending = await asyncio.wait({task}, timeout=SHUTDOWN_TIMEOUT_S)
        if pending:
            self.logger.warning(
                f"Orchestrator v2 shutdown did not complete within {SHUTDOWN_TIMEOUT_S}s; "
                "forcing process exit. Training artifacts are already persisted."
            )
            os._exit(0)
        await task


@clean_exit
async def run_orchestrator(config: OrchestratorConfig) -> None:
    """Top-level entrypoint: instantiate ``Orchestrator`` and run it.

    Wrapped in ``@clean_exit`` so wandb is flushed on exit (success or crash);
    keeps that cleanup out of the ``Orchestrator`` class proper.
    """
    await Orchestrator(config).start()


async def setup_student_inference_pool(*, config: OrchestratorConfig, tokenizer):
    """Mirror of ``prime_rl.orchestrator.orchestrator.setup_student_inference_pool``."""
    from renderers.base import create_renderer

    client_config = config.student.client
    model_name = config.student.model.name

    if config.use_renderer:
        renderer = create_renderer(
            tokenizer,
            renderer=config.renderer.name,
            tool_parser=config.renderer.tool_parser,
            reasoning_parser=config.renderer.reasoning_parser,
            preserve_all_thinking=config.renderer.preserve_all_thinking,
            preserve_thinking_between_tool_calls=config.renderer.preserve_thinking_between_tool_calls,
        )
        get_logger().info(f"Initialized {type(renderer).__name__} for {model_name}")
        inference_pool = await setup_inference_pool(
            client_config,
            model_name=model_name,
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_name=config.renderer.name,
            tool_parser=config.renderer.tool_parser,
            reasoning_parser=config.renderer.reasoning_parser,
            renderer_pool_size=config.renderer.pool_size,
            preserve_all_thinking=config.renderer.preserve_all_thinking,
            preserve_thinking_between_tool_calls=config.renderer.preserve_thinking_between_tool_calls,
        )
        get_logger().info("Using direct renderer rollout client")
        return renderer, inference_pool

    get_logger().info("Using MITO (openai_chat_completions) for rollouts")
    inference_pool = await setup_inference_pool(
        client_config,
        model_name=model_name,
        train_client_type="openai_chat_completions",
        eval_client_type="openai_chat_completions",
    )
    return None, inference_pool


def main() -> None:
    """Entrypoint for direct invocation via ``python -m prime_rl.orchestrator_v2.orchestrator``."""
    from prime_rl.utils.config import cli
    from prime_rl.utils.process import set_proc_title

    set_proc_title("OrchestratorV2")
    import uvloop

    uvloop.install()
    asyncio.run(run_orchestrator(cli(OrchestratorConfig)))


if __name__ == "__main__":
    main()
