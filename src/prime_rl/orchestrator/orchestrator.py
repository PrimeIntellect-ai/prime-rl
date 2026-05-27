"""Async-pipelined RL orchestrator.

``Orchestrator`` is the only class that holds the whole picture: it owns the
shared state (policy, progress, ckpt, monitor) and drives the pipeline loop.

Components are deliberately single-purpose and only know about the deps they
need:

- ``RolloutDispatcher`` schedules rollouts and emits ``Rollout``\\ s on its queue.
- ``TrainSink`` ingests train rollouts (tokenize → advantages + pre-filter →
  post-filter), exposes ``pop_batch`` returning a ``TrainBatch``.
- ``EvalSink`` ingests eval rollouts, exposes ``add`` returning a finalized
  ``EvalBatch`` (rollouts + pre-built per-env metrics) on epoch completion.
- ``MetricsBuilder`` builds the per-step train W&B dict from the popped batch
  and orchestrator-side timings (called inline by the orchestrator).
- ``WeightWatcher`` advances ``Policy`` and notifies the dispatcher.
- Each async component owns its own ``PeriodicLogger`` for steady-state
  gauges (dispatcher / watcher / orchestrator-main-loop). They share a
  single log interval and write to console + wandb on the ``_timestamp``
  axis.

None of these hold a reference to the orchestrator. The orchestrator wires
them in ``setup()`` and drives them from ``main_loop()``.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import time

import tomli_w

import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before transitive imports
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.ckpt import setup_ckpt_manager
from prime_rl.orchestrator.dispatcher import DispatcherMode, RolloutDispatcher
from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.event_loop_lag import EventLoopLagMonitor
from prime_rl.orchestrator.filters import setup_filters
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.metrics import MetricsBuilder
from prime_rl.orchestrator.patches import (
    monkey_patch_chat_completion_logprobs,
    monkey_patch_oai_iterable_types,
)
from prime_rl.orchestrator.periodic_logger import PeriodicLogger
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.trajectories import offload_images_to_disk
from prime_rl.orchestrator.types import EvalBatch, Policy, Progress, Rollout, TrainBatch
from prime_rl.orchestrator.utils import compute_teacher_logprobs, get_weight_dir, set_default_executor
from prime_rl.orchestrator.vf_utils import get_seq_len, intercept_vf_logging, save_rollouts
from prime_rl.orchestrator.watcher import WeightWatcher
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.transport import TrainingBatch, setup_training_batch_sender
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.client import init_nccl_broadcast, setup_inference_pool
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import format_time, get_logger, setup_logger
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


# Hard wall-clock budget for the orchestrator's post-training cleanup.
# Persist artifacts before this point, then force-exit if graceful shutdown
# wedges (env-server ZMQ recv, vLLM admin aclose, etc).
SHUTDOWN_TIMEOUT_S = 300


class Orchestrator:
    """``await Orchestrator(config).start()`` to run.

    ``stop()`` from outside (or the orchestrator self-stops when the train
    sink has driven ``progress.step`` to ``max_steps``). Cleanup happens at
    the tail of ``start()``.
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        setup_logger(config.log.level, json_logging=config.log.json_logging)
        # Silence the ``verifiers.*`` namespace by default (parser/rubric
        # mismatches, eval-dataset fallback warnings, etc. from the
        # in-process ``Environment`` instances are noise here), then
        # re-enable ``verifiers.serve`` through our loguru handler so the
        # env-server lifecycle logs still surface with proper formatting.
        logging.getLogger("verifiers").setLevel(logging.CRITICAL + 1)
        intercept_vf_logging(logger="verifiers.serve", level="WARN")
        get_logger().info(f"Starting orchestrator ({config.training_mode})")

        if config.bench:
            get_logger().warning(f"Running in benchmark mode (max_steps={config.max_steps})")

        # Cheap synchronous setup. Heavy async work happens in ``setup()``.
        self.progress = Progress()
        self.ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)
        self.policy = Policy(version=0, model_name="")
        self.stopped = asyncio.Event()
        # Drain mode: after the final train step, stop scheduling new train
        # rollouts but keep consuming until in-flight train + any triggered
        # eval drain. Set in ``ship_train_batch`` when ``max_steps`` is hit.
        self.draining: bool = False
        # Wall-clock timestamp of the previous ``TrainBatch`` arrival from
        # ``TrainSink``. Reset on each new arrival in ``ship_train_batch``
        # so ``step_time`` in the success log is the actual pipeline cycle
        # time (sink-emit to sink-emit) — the slowest component along the
        # path of rollout generation → tokenize → group/batch finalize.
        self.last_batch_at: float | None = None
        # Start time per (env_name, eval_step), stamped by
        # ``maybe_trigger_eval`` and popped by ``log_eval_batch`` so the
        # eval success log can report wall-clock epoch duration.
        self.eval_triggered_at: dict[tuple[str, int], float] = {}

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
        self.periodic_logger: PeriodicLogger | None = None
        self.lag_monitor: EventLoopLagMonitor | None = None
        self.lag_task: asyncio.Task | None = None

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

        get_logger().info(f"Initializing tokenizer ({config.tokenizer})")
        self.tokenizer = setup_tokenizer(config.tokenizer)

        # Student inference pool.
        get_logger().info(
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
            get_logger().info(
                f"Initializing teacher inference pool (base_url={', '.join(config.teacher.client.base_url)}, "
                f"model={config.teacher.model.name})"
            )
            self.teacher_inference = await setup_inference_pool(
                config.teacher.client,
                model_name=config.teacher.model.name,
                train_client_type="openai_chat_completions",
            )

        get_logger().info(f"Initializing monitor (wandb={config.wandb}, prime_monitor={config.prime_monitor})")
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
        pre_filters = setup_filters(config.pre_batch_filters, vocab_size=self.tokenizer.vocab_size, kind="pre-batch")
        post_filters = setup_filters(config.post_batch_filters, vocab_size=self.tokenizer.vocab_size, kind="post-batch")

        # Envs.
        get_logger().info("Loading training environments")
        self.train_envs = TrainEnvs(config.train.env)
        if config.training_mode == "sft":
            for env in self.train_envs:
                env.sampling_args.pop("logprobs", None)
        get_logger().debug(
            f"Loaded {len(self.train_envs)} training environment(s) ({', '.join(self.train_envs.names)})"
        )
        await self.train_envs.start(
            log_dir=get_log_dir(config.output_dir.parent) / "envs" / "train",
            log_level=config.log.vf_level,
            json_logging=config.log.json_logging,
        )
        get_logger().success("Train environment(s) ready")

        if config.eval is not None:
            get_logger().info("Loading eval environment(s)")
            self.eval_envs = EvalEnvs(config.eval.env)
            get_logger().debug(f"Loaded {len(self.eval_envs)} eval environment(s) ({', '.join(self.eval_envs.names)})")
            await self.eval_envs.start(
                log_dir=get_log_dir(config.output_dir.parent) / "envs" / "eval",
                log_level=config.log.vf_level,
                json_logging=config.log.json_logging,
            )
            get_logger().success("Eval environment(s) ready")

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
        get_logger().info("Waiting for student inference pool to be ready")
        await self.student_inference.wait_for_ready(config.student.model.name)
        get_logger().success("Student inference pool ready")
        if self.teacher_inference is not None:
            assert config.teacher is not None
            get_logger().info("Waiting for teacher inference pool to be ready")
            await self.teacher_inference.wait_for_ready(config.teacher.model.name)
            get_logger().success("Teacher inference pool ready")

        if config.wandb is not None and config.collect_inference_metrics:
            self.inference_metrics = InferenceMetricsCollector(
                self.student_inference.admin_clients,
                roles=config.inference_metrics_roles,
            )
            await self.inference_metrics.start()

        get_logger().info(f"Initializing weight broadcast ({config.weight_broadcast})")
        if config.weight_broadcast.type == "nccl":
            await init_nccl_broadcast(
                self.student_inference.admin_clients,
                config.weight_broadcast.host,
                config.weight_broadcast.port,
                config.weight_broadcast.timeout,
                inference_world_size=config.weight_broadcast.inference_world_size,
                quantize_in_weight_transfer=config.weight_broadcast.quantize_in_weight_transfer,
            )

        get_logger().info(f"Initializing training batch sender ({config.rollout_transport})")
        self.sender = setup_training_batch_sender(config.output_dir, config.rollout_transport)

        self.lora_name = config.student.model.lora.name if config.student.model.lora else None

        # Restore from checkpoint (if resuming).
        if self.resume_step is not None and self.ckpt_manager is not None:
            self.ckpt_manager.load(self.progress, step=self.resume_step)
            get_logger().info(f"Resuming orchestrator from checkpoint step {self.resume_step}")
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
            get_logger().info("Training from scratch")

        # ── Construct components — each gets only the deps it needs. ─────
        # Rollouts go to the teacher in sft mode (teacher generates, student
        # is trained via the teacher's outputs), to the student otherwise.
        if config.training_mode == "sft":
            assert self.teacher_inference is not None, "sft mode requires teacher inference"
            rollout_inference = self.teacher_inference
        else:
            rollout_inference = self.student_inference

        # Example sources are orchestrator-owned: the orchestrator triggers
        # eval epochs and the dispatcher just pulls from them.
        self.train_source = TrainSource(self.train_envs, seed=42)
        self.eval_source = EvalSource(self.eval_envs, config.eval)

        assert config.max_inflight_rollouts is not None, "max_inflight_rollouts must be resolved before dispatcher init"
        log_interval = config.experimental.log_interval
        wandb_enabled = config.wandb is not None
        self.dispatcher = RolloutDispatcher(
            train_envs=self.train_envs,
            eval_envs=self.eval_envs,
            train_source=self.train_source,
            eval_source=self.eval_source,
            inference=rollout_inference,
            policy=self.policy,
            max_inflight_rollouts=config.max_inflight_rollouts,
            tasks_per_minute=config.tasks_per_minute,
            max_off_policy_steps=config.max_off_policy_steps,
            training_mode=config.training_mode,
            log_interval=log_interval,
            wandb_enabled=wandb_enabled,
        )
        self.metrics = MetricsBuilder(config)
        self.train_sink = TrainSink(
            config,
            tokenizer=self.tokenizer,
            renderer=self.renderer,
            train_envs=self.train_envs,
            mm_token_type_ids_mapping=self.mm_token_type_ids_mapping,
            batch_size=config.batch_size,
            token_batch_size=config.token_batch_size,
            advantage_config=config.advantage,
            pre_filters=pre_filters,
            post_filters=post_filters,
        )
        self.eval_sink = EvalSink(eval_envs=self.eval_envs) if self.eval_envs is not None else None
        self.watcher = WeightWatcher(
            config,
            policy=self.policy,
            inference=self.student_inference,
            observers=[self.dispatcher],
            lora_name=self.lora_name,
            ckpt_step=self.progress.step,
            log_interval=log_interval,
            wandb_enabled=wandb_enabled,
        )
        # Orchestrator's own periodic logger: tracks event-loop lag (a
        # process-wide concern that fits naturally with the main loop).
        self.lag_monitor = EventLoopLagMonitor()
        self.periodic_logger = PeriodicLogger(
            name="Event Loop",
            collect=self.collect_event_loop_lag,
            metric_keys=list(self.lag_monitor.get_metrics().keys()),
            interval=log_interval,
            wandb_enabled=wandb_enabled,
        )

    async def start(self) -> None:
        """Run the orchestrator until shutdown. Drives setup, spawns the
        background tasks, runs the main loop in this task, then cleans up."""
        await self.setup()
        config = self.config
        get_logger().info(f"Starting orchestrator loop (max_steps={config.max_steps or 'infinite'})")
        start_time = time.perf_counter()

        # Spawn background loops (dispatcher schedules, watcher polls). The
        # pipeline ``main_loop`` runs inline in this task; each component
        # also runs its own ``PeriodicLogger`` for steady-state gauges.
        assert self.dispatcher and self.watcher
        assert self.periodic_logger is not None and self.lag_monitor is not None
        self.lag_task = asyncio.create_task(self.lag_monitor.run(), name="event_loop_lag")
        await self.periodic_logger.start()
        self.component_tasks = [
            asyncio.create_task(self.dispatcher.start(), name="dispatcher"),
            asyncio.create_task(self.watcher.start(), name="watcher"),
        ]

        # Default step-0 base-model eval — fires before any train rollouts
        # unless ``eval.skip_first_step=True`` (or this is a resume).
        self.maybe_trigger_eval(self.progress.step)

        try:
            await self.main_loop()
        finally:
            get_logger().success(f"Orchestrator step loop done in {format_time(time.perf_counter() - start_time)}")
            # No out-of-band final evals: when ``max_steps`` is reached the
            # orchestrator enters drain mode, the dispatcher stops scheduling
            # new train, and any interval-aligned eval at the final step
            # completes through the normal pipeline before ``main_loop`` exits.
            assert self.monitor is not None
            self.monitor.save_final_summary()
            if self.ckpt_manager is not None:
                get_logger().info("Writing final checkpoint")
                self.ckpt_manager.save(self.progress, step=self.progress.step)
            await self.shutdown()
            # Trailing newline gives a visual break between the orchestrator's
            # lifecycle logs and whatever the parent process (e.g. the ``rl``
            # entrypoint) emits next.
            get_logger().success("Orchestrator finished.\n")
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

        Both sinks own their own batch-boundary detection by counting
        arrivals up to ``group_size`` (and ``num_examples * group_size`` for
        eval epochs); both ``add()`` return a finalized batch (or ``None``)
        directly, so the orchestrator just dispatches on the result.
        """
        assert self.dispatcher and self.train_sink
        while not self.stopped.is_set():
            # Drain check: once the orchestrator has signaled draining (final
            # train step done, no more train scheduling), exit as soon as the
            # dispatcher is idle (no in-flight rollouts, no queued eval, queue
            # empty). Any interval-aligned eval at the final step is allowed
            # to complete here — no out-of-band re-evaluation pass needed.
            if self.draining and self.dispatcher.is_idle:
                get_logger().info("Pipeline drained, exiting main loop")
                self.stopped.set()
                break

            try:
                rollout: Rollout = await asyncio.wait_for(self.dispatcher.out_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if rollout.kind == "eval":
                assert self.eval_sink is not None  # eval rollouts only emitted when eval is configured
                eval_batch = self.eval_sink.add(rollout)
                if eval_batch is not None:
                    self.log_eval_batch(eval_batch)
                continue

            train_batch = await self.train_sink.add(rollout)
            # Skip training when draining — any leftover train rollouts that
            # still complete after the final step just get accumulated and
            # discarded; we don't want to do extra trainer steps past max.
            if train_batch is not None and not self.draining and not self.stopped.is_set():
                await self.ship_train_batch(train_batch)

    async def ship_train_batch(self, batch: TrainBatch) -> None:
        """Ship one ``TrainBatch`` out to the trainer + log side-effects.
        Mirrors ``log_eval_batch`` on the eval side: the sink has finished
        all data-transformation work (``TrainSink.process_rollout / group /
        batch``); this method owns only the I/O and lifecycle concerns —
        ckpt save, save_rollouts to disk, teacher logprobs (opd), async
        barrier wait, ``sender.send``, metrics build + ``monitor.log``,
        heartbeat / usage reporter, ``progress.step += 1``, and the eval
        trigger for the freshly-completed step."""
        assert self.train_sink and self.metrics and self.dispatcher and self.monitor and self.sender
        config = self.config
        step = self.progress.step

        # Pipeline cycle time: sink-emit (now) − previous sink-emit. Reset
        # the marker on every arrival so consecutive ``ship_train_batch``
        # calls measure the actual time between batches, not the I/O cost
        # of the orchestrator's ship pipeline (which is overlapped with
        # the dispatcher producing the next batch).
        now = time.perf_counter()
        step_time = (now - self.last_batch_at) if self.last_batch_at is not None else 0.0
        self.last_batch_at = now

        save_ckpt_time = await self.maybe_save_ckpt(step)

        if config.max_steps is not None and step >= config.max_steps:
            self.draining = True
            self.dispatcher.disable_train_scheduling()
            n_cancelled = await self.dispatcher.cancel_inflight_train_rollouts()
            get_logger().info(
                f"Reached max_steps={config.max_steps}, draining pipeline "
                f"(cancelled {n_cancelled} in-flight train rollout(s); "
                f"any triggered eval will complete)"
            )
            return

        get_logger().info(f"Starting orchestrator step {step}")

        if batch.metrics.n_trainable == 0:
            get_logger().warning(
                f"Step {step}: post-batch filters dropped all {len(batch.rollouts)} rollouts. Trying again."
            )
            return
        if batch.metrics.n_trainable / len(batch.rollouts) <= 0.1:
            get_logger().warning(
                f"Only {batch.metrics.n_trainable}/{len(batch.rollouts)} rollouts in the batch are trainable "
                f"({batch.metrics.n_trainable / len(batch.rollouts):.1%}) — consider reviewing task difficulty / filter config"
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
            get_logger().info(
                f"Offloaded {num_offloaded} unique images to disk in {format_time(time.perf_counter() - offload_start)}"
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

        # Build + ship metrics. ``step_time`` is the pipeline cycle time
        # (sink-emit to sink-emit) computed at the top of this method —
        # the meaningful "time per training step" for throughput.
        metrics = self.metrics.build(
            step=step,
            rollouts=batch.rollouts,
            metrics=batch.metrics,
            progress=self.progress,
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
                    tokens=batch.metrics.num_prefill_tokens + batch.metrics.num_decode_tokens,
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

        self.log_train_batch(batch, step=step, step_time=step_time)

        self.train_sink.reset_pre_filter_stats()
        self.progress.step += 1
        # Fire eligible eval epochs for the new training step. The dispatcher
        # picks them up the next time it fills inflight.
        self.maybe_trigger_eval(self.progress.step)

    def maybe_trigger_eval(self, step: int) -> None:
        """Trigger eligible eval epochs and flip ``DispatcherMode`` to
        PREFER_EVAL if anything fired. ``EvalSource.trigger`` handles
        both the startup case (first call → fires every env unless
        ``skip_first_step``) and the steady-state per-interval case
        (subsequent calls → ``step % interval == 0`` per env)."""
        assert self.dispatcher is not None
        fired = self.eval_source.trigger(step)
        if not fired:
            return
        reason = f"eval was triggered for {', '.join(fired)} at step {step}"
        self.dispatcher.switch_mode(DispatcherMode.PREFER_EVAL, reason=reason)
        # Stamp the start time per (env, step) so ``log_eval_batch`` can
        # report wall-clock duration of the epoch in its success log.
        now = time.perf_counter()
        for env_name in fired:
            self.eval_triggered_at[(env_name, step)] = now
        assert self.eval_envs is not None  # non-empty ``fired`` implies eval is configured
        total_rollouts = sum(
            self.eval_envs.get(env_name).config.group_size * len(self.eval_envs.get(env_name).examples)
            for env_name in fired
        )
        get_logger().info(f"Starting evals in {', '.join(fired)} ({total_rollouts} total rollouts)")

    def collect_event_loop_lag(self) -> tuple[str, dict[str, float]]:
        """Format the event-loop-lag periodic line + return the wandb dict.
        Empty payload when no samples have been collected yet."""
        metrics = self.lag_monitor.get_metrics()
        if not metrics:
            return "(no samples yet)", {}
        body = (
            f"min={format_time(metrics['event_loop_lag/min'])} | "
            f"mean={format_time(metrics['event_loop_lag/mean'])} | "
            f"median={format_time(metrics['event_loop_lag/med'])} | "
            f"p90={format_time(metrics['event_loop_lag/p90'])} | "
            f"p99={format_time(metrics['event_loop_lag/p99'])} | "
            f"max={format_time(metrics['event_loop_lag/max'])}"
        )
        return body, metrics

    def log_train_batch(self, batch: TrainBatch, *, step: int, step_time: float) -> None:
        """Emit the per-step ``Train step …`` success line.

        Single-env (the typical case) collapses to one dense line; multi-env
        (``len(train_envs) > 1``) appends an indented ``↳`` line per env
        with the same fields scoped to that env's rollouts. All percentages
        are reported relative to arrivals at the sink (so ``Error``
        accounts for errored rollouts that were group-dropped and never
        reached ``batch.rollouts``).
        """
        n_arrivals_total = sum(batch.metrics.arrivals_by_env.values())
        n_errors_total = sum(batch.metrics.errors_by_env.values())
        n_survivors = len(batch.rollouts)
        n_trainable = batch.metrics.n_trainable
        error_rate = (n_errors_total / n_arrivals_total) if n_arrivals_total else 0.0
        trainable_rate = (n_trainable / n_survivors) if n_survivors else 0.0
        reward_mean = sum(r["reward"] for r in batch.rollouts) / max(n_survivors, 1)
        max_off_policy = max((int(r.get("_off_policy_steps", 0)) for r in batch.rollouts), default=0)

        head = (
            f"Train step {step} | {format_time(step_time)} | Reward {reward_mean:.4f} | "
            f"Error {error_rate:.1%} | Trainable {n_trainable}/{n_survivors} ({trainable_rate:.1%}) | "
            f"Max Off-Policy {max_off_policy}"
        )
        if len(self.train_envs) <= 1:
            get_logger().success(head)
            return

        # Multi-env: one success call with ``\n\t\t``-joined per-env lines
        # so the ``╰─`` indented content visually aligns under the
        # headline (two tabs ≈ 16 cols, matching the ``HH:MM:SS LEVEL ``
        # log prefix width).
        env_names = sorted(set(batch.metrics.arrivals_by_env) | {r["env_name"] for r in batch.rollouts})
        name_width = max(len(n) for n in env_names) if env_names else 0
        lines = [head]
        for env_name in env_names:
            env_rollouts = [r for r in batch.rollouts if r["env_name"] == env_name]
            n_env_arrivals = batch.metrics.arrivals_by_env.get(env_name, 0)
            n_env_errors = batch.metrics.errors_by_env.get(env_name, 0)
            ratio = (n_env_arrivals / n_arrivals_total) if n_arrivals_total else 0.0
            env_error_rate = (n_env_errors / n_env_arrivals) if n_env_arrivals else 0.0
            env_reward = (sum(r["reward"] for r in env_rollouts) / len(env_rollouts)) if env_rollouts else 0.0
            env_max_off_policy = max((int(r.get("_off_policy_steps", 0)) for r in env_rollouts), default=0)
            lines.append(
                f"╰─ {env_name:<{name_width}} | Ratio {ratio:.1%} | Reward {env_reward:.4f} | "
                f"Error {env_error_rate:.1%} | Max Off-Policy {env_max_off_policy}"
            )
        get_logger().success("\n\t\t".join(lines))

    def log_eval_batch(self, batch: EvalBatch) -> None:
        """Persist + log one completed eval epoch. Builds the metrics dict
        from the raw rollouts (the ``EvalBatch`` deliberately doesn't carry a
        pre-baked metrics dict — it's a pure view computed here), then handles
        the side effects: save_rollouts, monitor.log_eval_samples, monitor.log."""
        if not batch.rollouts:
            get_logger().warning(f"Eval @ step={batch.step} env={batch.env_name}: no surviving rollouts, skipping log")
            return

        assert self.monitor is not None
        step_path = get_step_path(get_rollout_dir(self.config.output_dir), batch.step)
        save_rollouts(
            batch.rollouts,
            step_path / f"eval_rollouts_{batch.env_name}.jsonl",
            exclude_keys={"trajectory"},
        )
        self.monitor.log_eval_samples(batch.rollouts, env_name=batch.env_name, step=batch.step)
        self.monitor.log(batch.metrics.to_wandb_dict(env_name=batch.env_name, step=batch.step), step=batch.step)

        n_total = batch.metrics.n_rollouts
        n_valid = n_total - batch.metrics.n_cancelled - batch.metrics.n_errored
        error_rate = ((batch.metrics.n_cancelled + batch.metrics.n_errored) / n_total) if n_total else 0.0
        valid_rate = (n_valid / n_total) if n_total else 0.0
        # ``Max Off-Policy`` here is the worst-case lag across the eval
        # cohort — how many weight updates the eval epoch straddled.
        max_off_policy = max((int(r.get("_off_policy_steps", 0)) for r in batch.rollouts), default=0)
        triggered_at = self.eval_triggered_at.pop((batch.env_name, batch.step), None)
        elapsed = (time.perf_counter() - triggered_at) if triggered_at is not None else 0.0

        get_logger().success(
            f"Eval step {batch.step} ({batch.env_name}) | {format_time(elapsed)} | "
            f"Reward {batch.metrics.reward_mean:.4f} | Error {error_rate:.1%} | "
            f"Valid {n_valid}/{n_total} ({valid_rate:.1%}) | "
            f"Max Off-Policy {max_off_policy}"
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
        get_logger().info(f"Saving checkpoint at step {step}")
        t = time.perf_counter()
        await asyncio.to_thread(self.ckpt_manager.save, self.progress, step)
        return time.perf_counter() - t

    async def wait_barrier(self, step: int) -> None:
        """Block until ``policy.version >= step - 1`` so the orchestrator stays
        at most one step ahead of the trainer.

        Cascades backpressure into the dispatcher via the bounded ``out_q``:
        while we wait, the queue fills, the dispatcher stops handing out new
        inflight permits, and in-flight rollouts drain without new ones being
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
                get_logger().info(
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
            if self.periodic_logger is not None:
                await self.periodic_logger.stop()
            if self.lag_task is not None:
                await safe_cancel(self.lag_task)
                self.lag_task = None
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
            get_logger().warning(
                f"Orchestrator shutdown did not complete within {SHUTDOWN_TIMEOUT_S}s; "
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
    """Build the student inference pool + matching renderer (if configured).

    Returns ``(renderer | None, inference_pool)``. The renderer is None when
    ``config.renderer is None`` (MITO path); otherwise the typed
    ``config.renderer`` discriminated union resolves to a hand-coded /
    default renderer that owns client-side tokenization.
    """
    from renderers.base import create_renderer

    client_config = config.student.client
    model_name = config.student.model.name

    if config.renderer is not None:
        renderer = create_renderer(tokenizer, config.renderer)
        get_logger().info(f"Initialized {type(renderer).__name__} for {model_name}")
        inference_pool = await setup_inference_pool(
            client_config,
            model_name=model_name,
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=config.renderer,
            pool_size=config.pool_size,
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
    """Entrypoint for direct invocation via ``python -m prime_rl.orchestrator.orchestrator``."""
    from prime_rl.utils.config import cli
    from prime_rl.utils.process import set_proc_title

    set_proc_title("Orchestrator")
    import uvloop

    uvloop.install()
    asyncio.run(run_orchestrator(cli(OrchestratorConfig)))


if __name__ == "__main__":
    main()
