"""Async-pipelined RL orchestrator.

``Orchestrator`` owns the shared state (policy, progress, ckpt, monitor)
and drives the pipeline. Components are single-purpose:

- ``Dispatcher`` schedules environment runs and emits completed episodes.
- ``TrainSink`` ingests train rollouts (score → admission → sample compilation)
  and returns a ``TrainBatch`` when the threshold is met.
- ``EvalSink`` ingests eval rollouts and returns an ``EvalBatch`` (the full
  returned cohort) on epoch completion.
- ``TrainEpisodes`` / ``EvalEpisodes`` preserve episode boundaries and build per-step metrics.
- ``WeightWatcher`` advances ``Policy`` and notifies observers.
- ``PeriodicLogger`` polls the components on a shared interval for the
  ``_timestamp``-axis pipeline log.

Components don't reference the orchestrator. The orchestrator wires them
in ``setup()`` and drives them from ``main_loop()``.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import TYPE_CHECKING

import verifiers.v1 as vf
from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from verifiers.v1.runtimes import set_base_sandbox_labels

if TYPE_CHECKING:
    from renderers.base import Renderer
    from transformers.tokenization_utils import PreTrainedTokenizer

    from prime_rl.orchestrator.ckpt import CheckpointManager
    from prime_rl.transports.rollouts.base import MicroBatchSender
    from prime_rl.utils.client import InferencePool
import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before transitive imports
from prime_rl import monitors
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.algo.routing import is_trainable
from prime_rl.orchestrator.ckpt import setup_ckpt_manager
from prime_rl.orchestrator.concurrency import ConcurrencyController
from prime_rl.orchestrator.dispatcher import Dispatcher, DispatcherMetrics, DispatcherMode
from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.metrics import TrainEpisodes
from prime_rl.orchestrator.packing import BatchPacker
from prime_rl.orchestrator.patches import (
    monkey_patch_chat_completion_logprobs,
    monkey_patch_oai_iterable_types,
)
from prime_rl.orchestrator.periodic_logger import PeriodicLogger
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import (
    Cancellation,
    EvalBatch,
    Policy,
    Progress,
    TrainBatch,
)
from prime_rl.orchestrator.utils import (
    episode_group_id,
    episode_staleness,
    eval_work,
    get_weight_dir,
    intercept_vf_logging,
    set_default_executor,
    setup_policy_inference_pool,
    trim_process_memory,
)
from prime_rl.orchestrator.watcher import WeightWatcher
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.transports.rollouts import setup_micro_batch_sender
from prime_rl.transports.weights.nixl.model_express import ModelExpressSession
from prime_rl.utils.async_utils import EventLoopLagMonitor, EventLoopLagStats, safe_cancel
from prime_rl.utils.client import init_nccl_broadcast, init_nixl_broadcast
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import format_time, get_logger, setup_logger
from prime_rl.utils.utils import (
    clean_exit,
    resolve_latest_ckpt_step,
)

monkey_patch_oai_iterable_types()
monkey_patch_chat_completion_logprobs()


# Wall-clock budget for post-training cleanup; force-exit if graceful
# shutdown wedges (env-server ZMQ recv, vLLM admin aclose, etc)
SHUTDOWN_TIMEOUT_S = 300

# Abort after this many consecutive train batches contain no samples.
MAX_CONSECUTIVE_EMPTY_BATCHES = 10

# Maximum batches the orchestrator may run ahead of the trainer. The
# dispatcher is paused via ``update_dispatch_gate`` once this is exceeded;
# resumed when the watcher advances ``policy.version``.
TARGET_LAG = 1

# Default wait for the trainer's startup weight broadcast when no ckpt block
# configures ``wait_for_weights_timeout`` (e.g. a from-scratch run). The
# broadcast is always coming, so wait rather than fail immediately.
STARTUP_WEIGHT_WAIT_TIMEOUT_S = 1200


class Orchestrator:
    # Set in ``__init__``
    config: OrchestratorConfig
    progress: Progress
    policy: Policy
    stopped: asyncio.Event
    draining: bool
    last_batch_at: float | None
    consecutive_empty_batches: int
    eval_triggered_at: dict[tuple[str, int], float]
    ckpt_manager: CheckpointManager
    component_tasks: list[asyncio.Task]

    # Always set by ``setup()``
    tokenizer: PreTrainedTokenizer
    policy_inference: InferencePool
    sender: MicroBatchSender | None
    packer: BatchPacker
    train_envs: TrainEnvs
    train_source: TrainSource
    train_sink: TrainSink
    dispatcher: Dispatcher
    concurrency: ConcurrencyController
    watcher: WeightWatcher
    lag_monitor: EventLoopLagMonitor
    periodic_logger: PeriodicLogger

    # Set by ``setup()`` only when relevant config is present
    renderer: Renderer | None
    mm_token_type_ids_mapping: dict[int, int] | None
    heart: Heartbeat | None
    inference_metrics: InferenceMetricsCollector | None
    eval_envs: EvalEnvs | None
    eval_sink: EvalSink | None
    eval_source: EvalSource | None
    lora_name: str | None
    resume_step: int | None
    lag_task: asyncio.Task | None

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        setup_logger(config.log.level, json_logging=config.log.json_logging)
        # Route the in-process v1 library logging through our handler. The
        # env server runs in a child process, so its logging is separate.
        intercept_vf_logging(logger="verifiers.v1", level="WARN")
        algorithms = sorted({env.algo.type for env in config.train.source if env.algo is not None})
        get_logger().info(f"Starting orchestrator (algorithm: {', '.join(algorithms)})")

        self.progress = Progress()
        self.ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)
        self.policy = Policy(version=0, model_name="")
        self.stopped = asyncio.Event()
        # True after the final train step ships — pipeline winds down without
        # scheduling new train rollouts
        self.draining = False
        # Previous ``TrainBatch`` arrival timestamp; reset every ship so
        # ``step_time`` in the success log is real sink-to-sink cycle time
        self.last_batch_at = None
        # Trigger timestamps so eval success logs can report epoch duration
        self.eval_triggered_at = {}
        self.consecutive_empty_batches = 0
        self.gate_closed_at = None
        # Pulsed by the version hooks so a held ship can re-check ``policy.version``
        self.version_advanced = asyncio.Event()
        self.wait_for_policy_time = 0.0
        self.component_tasks = []

        # Optional attributes — ``setup()`` populates them when the relevant
        # config is present
        self.renderer = None
        self.mm_token_type_ids_mapping = None
        self.heart = None
        self.inference_metrics = None
        self.eval_envs = None
        self.eval_sink = None
        self.eval_source = None
        self.lora_name = None
        self.resume_step = None
        self.lag_task = None
        self.model_express = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Install envs, load models/pools, resume from checkpoint, and
        construct the pipeline components."""
        config = self.config
        set_default_executor()

        get_logger().info(f"Initializing tokenizer ({config.tokenizer})")
        self.tokenizer = setup_tokenizer(config.tokenizer)

        # The one model prime-rl hosts: the live policy. Frozen model
        # references are external endpoints — each env's Algorithm builds its
        # own pools in ``setup()`` below.
        get_logger().info(
            f"Initializing policy inference pool (base_url={config.model.client.base_url}, model={config.model.name})"
        )
        self.renderer, self.policy_inference = await setup_policy_inference_pool(
            config=config, tokenizer=self.tokenizer
        )
        self.mm_token_type_ids_mapping = (
            getattr(self.renderer, "mm_token_type_id_map", None) if self.renderer is not None else None
        )
        if self.mm_token_type_ids_mapping == {}:
            self.mm_token_type_ids_mapping = None

        get_logger().info(f"Initializing monitors ({config.monitors})")
        await monitors.setup(
            wandb=config.monitors.wandb,
            prime=config.monitors.prime,
            file=config.monitors.file,
            output_dir=config.output_dir,
            run_config=config,
            train_env_names=[env.resolved_name for env in config.train.source],
            eval_env_names=[source.resolved_name for source in config.eval.source] if config.eval is not None else [],
        )
        # The launcher-set $PRL_RUN_ID is the run identity; standalone runs mint a local one.
        self.run_id = os.environ.get("PRL_RUN_ID") or uuid.uuid4().hex
        # Base labels for sandboxes created in this process; env-server processes read
        # the same launcher-set env var themselves.
        self.run_name = os.environ.get("PRL_RUN_NAME")
        if self.run_name:
            set_base_sandbox_labels([self.run_name])

        if config.heartbeat is not None:
            self.heart = Heartbeat(config.heartbeat.url)

        get_logger().info("Loading training environments")
        self.train_envs = TrainEnvs(
            config.train.source,
            config.env_addresses,
            policy_pool=self.policy_inference,
            renderer_config=config.renderer,
        )
        get_logger().debug(
            f"Loaded {len(self.train_envs)} training environment(s) ({', '.join(self.train_envs.names)})"
        )
        await self.train_envs.start()
        get_logger().success("Train environment(s) ready")

        if config.eval is not None:
            get_logger().info("Loading eval environment(s)")
            self.eval_envs = EvalEnvs(config.eval.source, config.env_addresses)
            get_logger().debug(f"Loaded {len(self.eval_envs)} eval environment(s) ({', '.join(self.eval_envs.names)})")
            await self.eval_envs.start()
            get_logger().success("Eval environment(s) ready")

        if config.resume is not None:
            if config.resume.dir is not None:
                self.resume_step = config.resume.dir_step
            else:
                self.resume_step = config.resume.step
                if self.resume_step is None:
                    self.resume_step = resolve_latest_ckpt_step(self.ckpt_manager.ckpt_dir)

        # Resume below may bump ``policy.version`` and the LoRA model name
        self.policy.model_name = self.policy_inference.model_name

        get_logger().info("Waiting for policy inference pool to be ready")
        await self.policy_inference.wait_for_ready(config.model.name)
        get_logger().success("Policy inference pool ready")
        # Build + ready pools for each env's frozen sampling source and the
        # algorithm's frozen reference model
        await asyncio.gather(
            *(env.sampler.setup() for env in self.train_envs),
            *(env.algorithm.setup() for env in self.train_envs),
        )

        get_logger().info(f"Initializing weight broadcast ({config.weight_broadcast})")
        if config.weight_broadcast.type == "nccl":
            await init_nccl_broadcast(
                self.policy_inference.admin_clients,
                config.weight_broadcast.host,
                config.weight_broadcast.port,
                config.weight_broadcast.timeout,
                inference_world_size=config.weight_broadcast.inference_world_size,
                quantize_in_weight_transfer=config.weight_broadcast.quantize_in_weight_transfer,
            )
        elif config.weight_broadcast.type == "nixl":
            await init_nixl_broadcast(
                self.policy_inference.admin_clients,
                config.weight_broadcast.host,
                config.weight_broadcast.port,
                config.weight_broadcast.timeout,
                config.weight_broadcast.inference_world_size,
                config.weight_broadcast.session_id,
            )
            self.model_express = ModelExpressSession(
                client=MxClient(server_url=f"{config.weight_broadcast.host}:{config.weight_broadcast.port}"),
                role="orchestrator",
                rank=0,
                session_id=config.weight_broadcast.session_id,
                worker_id="orchestrator",
            )
            self.model_express.publish()
            await asyncio.to_thread(self.model_express.set_status, p2p_pb2.SOURCE_STATUS_INITIALIZING)

        self.lora_name = config.model.lora.name if config.model.lora else None

        self.train_source = TrainSource(self.train_envs)

        if self.resume_step is not None:
            resume = self.config.resume
            resume_path = resume.dir / "orchestrator" if resume is not None and resume.dir is not None else None
            self.ckpt_manager.load(self.progress, self.train_source, step=self.resume_step, path=resume_path)
            # The checkpoint finished step ``resume_step``; resume at the next step. Derive the step
            # from ``resume_step`` (not the loaded progress.step) so it stays coordinated with the
            # trainer even when ``ckpt.skip_progress`` left the counter unrestored.
            self.progress.step = self.resume_step + 1
            get_logger().info(f"Resuming orchestrator from checkpoint step {self.resume_step}")
        else:
            get_logger().info("Training from scratch")

        self.packer = BatchPacker(config)
        get_logger().info(f"Initializing micro batch sender ({config.rollout_transport})")
        self.sender = setup_micro_batch_sender(
            config.output_dir, config.num_train_workers, self.progress.step, config.rollout_transport
        )

        # Sync inference to the incoming policy before the first step, rendezvousing
        # with the trainer's startup broadcast (v{resume_step} on resume, v0 from
        # scratch).
        sync_version = self.resume_step if self.resume_step is not None else 0
        if config.weight_broadcast.type == "nixl":
            weights_path = None
        else:
            check_exists = config.weight_broadcast.type == "filesystem"
            # The trainer's startup broadcast is always coming, so wait for it
            # rather than failing immediately when the directory is not there yet.
            wait_timeout = (config.ckpt.wait_for_weights_timeout if config.ckpt else None) or (
                STARTUP_WEIGHT_WAIT_TIMEOUT_S
            )
            weights_path = get_weight_dir(
                config.output_dir, sync_version, check_exists=check_exists, wait_timeout=wait_timeout
            )
        if self.model_express is not None:
            await asyncio.to_thread(self.model_express.set_status, p2p_pb2.SOURCE_STATUS_READY)
        await self.policy_inference.update_weights(weights_path, lora_name=self.lora_name, step=sync_version)
        if self.model_express is not None:
            await asyncio.to_thread(self.model_express.set_status, p2p_pb2.SOURCE_STATUS_INITIALIZING)
            # Complete the startup rendezvous before the watcher begins its next cycle.
            await asyncio.to_thread(
                self.model_express.wait_for,
                "trainer",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=config.weight_broadcast.timeout,
            )
        if self.lora_name is not None:
            self.policy_inference.update_model_name(self.lora_name)
            self.policy.model_name = self.lora_name
        self.policy.version = sync_version

        self.eval_source: EvalSource | None = (
            EvalSource(
                self.eval_envs,
                config.eval,
                is_resumed=self.resume_step is not None,
            )
            if config.eval is not None and self.eval_envs is not None
            else None
        )

        log_interval = config.log.interval
        wandb_enabled = monitors.get(monitors.WandbMonitor) is not None

        self.concurrency = ConcurrencyController(config.concurrency, fallback_cost=config.seq_len)
        self.dispatcher = Dispatcher(
            train_envs=self.train_envs,
            eval_envs=self.eval_envs,
            train_source=self.train_source,
            eval_source=self.eval_source,
            policy_pool=self.policy_inference,
            policy=self.policy,
            progress=self.progress,
            initial_max_inflight=self.concurrency.max_inflight,
            max_inflight_ceiling=config.concurrency.max_inflight,
            tasks_per_minute=config.tasks_per_minute,
            max_staleness=config.max_staleness,
            run_id=self.run_id,
            run_name=self.run_name,
            on_episode_complete=self.concurrency.record_episode,
        )
        self.concurrency.bind(
            set_limit=self.dispatcher.set_limit,
            get_inflight=lambda: self.dispatcher.current_inflight,
            on_overload=self.dispatcher.cancel_inflight,
        )
        # The collector always polls — it feeds the concurrency controller;
        # W&B mirroring is gated on the registered monitor (the collector logs
        # to the global W&B session, which only exists when init succeeded).
        self.inference_metrics = InferenceMetricsCollector(
            self.policy_inference.admin_clients,
            roles=config.inference_metrics_roles,
            on_load=self.concurrency.observe,
            log_to_wandb=wandb_enabled and config.collect_inference_metrics,
        )
        await self.inference_metrics.start()
        self.train_sink = TrainSink(
            config,
            tokenizer=self.tokenizer,
            train_envs=self.train_envs,
            mm_token_type_ids_mapping=self.mm_token_type_ids_mapping,
            progress=self.progress,
            batch_size=config.batch_size,
            token_batch_size=config.token_batch_size,
            on_result=self.train_source.on_result,
        )

        self.eval_sink = EvalSink(eval_envs=self.eval_envs) if self.eval_envs is not None else None
        self.watcher = WeightWatcher(
            config,
            policy=self.policy,
            inference=self.policy_inference,
            observers=[self.dispatcher, self],
            lora_name=self.lora_name,
            ckpt_step=self.policy.version,
            model_express=self.model_express,
        )
        # Single periodic logger for the whole pipeline. It's the only
        # consumer of ``dispatcher.metrics.drained()`` (which clears on read)
        self.lag_monitor = EventLoopLagMonitor()
        self.periodic_logger = PeriodicLogger(
            name="Pipeline",
            collect=self.collect_pipeline_view,
            metric_keys=[
                *list(self.dispatcher.gauges().keys()),
                *list(self.concurrency.gauges().keys()),
                *DispatcherMetrics.drain_keys(
                    train_envs={e.name for e in self.train_envs},
                    eval_envs={e.name for e in self.eval_envs} if self.eval_envs is not None else set(),
                ),
                *list(self.watcher.gauges().keys()),
                "event_loop_lag/min",
                "event_loop_lag/mean",
                "event_loop_lag/median",
                "event_loop_lag/p90",
                "event_loop_lag/p99",
                "event_loop_lag/max",
                "event_loop_lag/n",
            ],
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
        # pipeline ``main_loop`` runs inline in this task; the single
        # ``PeriodicLogger`` polls dispatcher / watcher / sinks / lag
        # monitor each ``log.interval`` seconds for the pipeline-view log
        self.lag_task = asyncio.create_task(self.lag_monitor.run(), name="event_loop_lag")
        await self.periodic_logger.start()
        self.component_tasks = [
            asyncio.create_task(self.dispatcher.start(), name="dispatcher"),
            asyncio.create_task(self.watcher.start(), name="watcher"),
        ]

        # Base-model eval (policy v0) — fires before any train rollouts, logged at the first
        # step, unless ``eval.skip_first_step=True``. On resume, defaults to assuming a clean
        # exit (evals already completed); set ``eval.retrigger_on_resume=True`` to also re-fire
        # interval-aligned evals at the checkpoint step (e.g. after a crash).
        if config.eval is not None and config.eval.retrigger_on_resume and self.resume_step is not None:
            self.maybe_trigger_eval(self.resume_step)
        self.maybe_trigger_eval(self.progress.step)

        # Anchor step-time clock so the first step measures startup → first batch
        self.last_batch_at = time.perf_counter()

        # ``clean_exit`` stays False if ``main_loop`` raises (signal-driven
        # CancelledError, KeyboardInterrupt, or a real error), so the teardown
        # logs a forced-cleanup warning instead of a clean-exit success.
        clean_exit = False
        try:
            await self.main_loop()
            await self.wait_for_final_broadcast()
            clean_exit = True
        finally:
            elapsed = format_time(time.perf_counter() - start_time)
            if clean_exit:
                get_logger().success(f"Orchestrator step loop done in {elapsed}")
                # The collector logs to the W&B run, so it must stop before
                # finalize marks the run finished
                if self.inference_metrics is not None:
                    await self.inference_metrics.stop()
                # Finalize only on a clean exit — a crashed run must not be marked
                # completed; the platform run's atexit hook marks it failed instead.
                await monitors.finalize()
            else:
                get_logger().warning(f"Orchestrator interrupted after {elapsed} — forcing cleanup (not a clean exit)")
            # ``progress.step`` points at the next (unshipped) step; the last finished step is
            # ``progress.step - 1``. Checkpoint it as ``step_{progress.step - 1}`` (no-op before the
            # first ship).
            if self.config.ckpt is not None and self.progress.step > 1:
                self.progress.step -= 1
                get_logger().info("Writing final checkpoint")
                self.ckpt_manager.save(self.progress, self.train_source, step=self.progress.step)
            await self.stop()
            if clean_exit:
                get_logger().success("Orchestrator finished.")
            else:
                get_logger().warning("Orchestrator cleanup complete (forced).")
            trim_process_memory()

    async def wait_for_final_broadcast(self) -> None:
        """Stay alive for the trainer's last in-memory broadcast (v{max_steps-1};
        nothing samples from v{max_steps}, so it is never sent). An in-memory
        broadcast is a blocking collective — tearing down the watcher before
        the rendezvous would strand the trainer inside it."""
        config = self.config
        if config.weight_broadcast.type not in ("nccl", "nixl") or config.max_steps is None:
            return
        final_version = config.max_steps - 1
        if self.policy.version >= final_version:
            return
        get_logger().info(f"Waiting for the trainer's final broadcast (v{final_version}) before shutdown")

        async def wait() -> None:
            while self.policy.version < final_version:
                self.version_advanced.clear()
                if self.policy.version >= final_version:
                    return
                # A dead watcher can never deliver the broadcast — fail out
                # instead of idling until the timeout.
                self._raise_if_component_stopped()
                try:
                    await asyncio.wait_for(self.version_advanced.wait(), timeout=5)
                except asyncio.TimeoutError:
                    pass

        try:
            await asyncio.wait_for(wait(), timeout=config.weight_broadcast.timeout)
        except asyncio.TimeoutError:
            get_logger().warning(
                f"Trainer did not broadcast v{final_version} within {config.weight_broadcast.timeout}s — "
                "shutting down anyway"
            )

    async def main_loop(self) -> None:
        """Consume completed episodes and group ``Cancellation``\\ s from the
        dispatcher and route them to the train / eval sink. The sinks return a
        finalized batch (or ``None``); we just dispatch on the result."""
        while not self.stopped.is_set():
            self._raise_if_component_stopped()
            if self.draining and self.dispatcher.is_idle:
                get_logger().info("Pipeline drained, exiting main loop")
                self.stopped.set()
                break

            try:
                item = await asyncio.wait_for(self.dispatcher.out_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                self._raise_if_component_stopped()
                continue

            if isinstance(item, Cancellation):
                assert item.kind == "train"  # eval groups are never dropped
                train_batch = await self.train_sink.cancel(item)
                if train_batch is not None and not self.draining and not self.stopped.is_set():
                    await self.finalize_train_batch(train_batch)
                continue
            episode = item

            # Every completed rollout — errored, rejected, or never batched — lands in the
            # ``all`` trace file the moment it arrives, so it survives crashes and drains.
            # Train rollouts belong to the batch window currently collecting (``progress.step``),
            # eval rollouts to the step whose eval triggered them.
            if episode.run is None:
                raise ValueError("Dispatched episode is missing run identity")
            if not isinstance(episode.run, vf.TrainRunInfo):
                raise ValueError("Orchestrated episode is missing training-run provenance")
            kind = episode.run.work.type
            step = episode.run.work.step if kind == "eval" else self.progress.step
            await monitors.log([episode], step, kind, "all")

            if kind == "eval":
                assert self.eval_sink is not None  # eval rollouts only emitted when eval is configured
                eval_batch = self.eval_sink.add(episode)
                if eval_batch is not None:
                    await self.finalize_eval_batch(eval_batch)
                continue

            train_batch = await self.train_sink.add(episode)
            # In drain mode any late-arriving train batch is dropped — we
            # don't want to ship past ``max_steps``
            if train_batch is not None and not self.draining and not self.stopped.is_set():
                await self.finalize_train_batch(train_batch)

    def _raise_if_component_stopped(self) -> None:
        """Propagate unexpected background-component termination to the run."""
        for task in self.component_tasks:
            if not task.done():
                continue
            if task.cancelled():
                raise RuntimeError(f"{task.get_name()} stopped unexpectedly")
            error = task.exception()
            if error is not None:
                raise error
            raise RuntimeError(f"{task.get_name()} exited unexpectedly")

    async def finalize_train_batch(self, batch: TrainBatch) -> None:
        """Ship one ``TrainBatch`` out to the trainer and handle the I/O
        side-effects (ckpt, monitors.log, reference scoring, sender.send,
        metrics, heartbeat, progress, eval trigger). The sink has already
        done all data-transformation work."""
        config = self.config
        step = self.progress.step

        # Sink-to-sink cycle time — the actual time between batches, not
        # including the orchestrator's ship I/O (overlapped with the
        # dispatcher producing the next batch)
        now = time.perf_counter()
        step_time = (now - self.last_batch_at) if self.last_batch_at is not None else 0.0
        self.last_batch_at = now

        # A resume can start past the end (checkpoint written at the final
        # step, or a lowered ``max_steps``): never ship beyond the budget.
        if config.max_steps is not None and step > config.max_steps:
            await self.start_draining(f"Step {step} exceeds max_steps={config.max_steps}")
            return

        if not batch.samples:
            self.consecutive_empty_batches += 1
            get_logger().warning(
                f"Step {step}: empty train batch after {len(batch.episodes)} finalized episodes "
                f"(consecutive empty batches: "
                f"{self.consecutive_empty_batches}/{MAX_CONSECUTIVE_EMPTY_BATCHES})"
            )
            if self.consecutive_empty_batches >= MAX_CONSECUTIVE_EMPTY_BATCHES:
                raise RuntimeError(
                    f"{self.consecutive_empty_batches} consecutive empty train batches — "
                    "check algorithm credit and task difficulty."
                )
            return
        self.consecutive_empty_batches = 0
        effective = batch.cohort.effective
        n_trainable = sum(is_trainable(record.trace) for record in effective.records)
        if effective.num_traces and n_trainable / effective.num_traces <= 0.1:
            get_logger().warning(
                f"Only {n_trainable}/{effective.num_traces} effective traces are trainable "
                f"({n_trainable / effective.num_traces:.1%}) — consider reviewing task difficulty"
            )

        # Ship batch ``step`` only once the trainer has published v{step-1-TARGET_LAG}.
        # Without this, fast envs fill batches from buffered rollouts and the
        # orchestrator races arbitrarily far ahead of the trainer. Always
        # satisfiable: the trainer broadcasts every version except v{max_steps},
        # and ``wait_for_final_broadcast`` keeps the watcher alive through the
        # last rendezvous after the pipeline drains.
        required_version = step - 1 - TARGET_LAG
        if self.policy.version < required_version:
            get_logger().info(
                f"Holding batch {step} until the trainer publishes policy v{required_version} "
                f"(currently v{self.policy.version})"
            )
            hold_start = time.perf_counter()
            while True:
                self.version_advanced.clear()
                if self.policy.version >= required_version:
                    break
                await self.version_advanced.wait()
            self.wait_for_policy_time += time.perf_counter() - hold_start

        # The effective (clean, trained-on) subset is logged at ship time; the full arrival
        # window already streamed into the ``all`` cohort on arrival.
        await monitors.log(effective.vf_episodes, step, "train", "effective")

        pack_start_time = time.perf_counter()
        micro_batch_grid = await asyncio.to_thread(self.packer.pack, batch.samples)
        pack_time = time.perf_counter() - pack_start_time
        await self.sender.send(micro_batch_grid)
        self.progress.step += 1
        self.update_dispatch_gate()
        # Checkpoint the step we just shipped (resume point: continue at step + 1).
        save_ckpt_time = await self.maybe_save_ckpt(step)
        trim_process_memory()

        # Episode metrics over the {agg,<env>} × {all,effective} matrix. ``all`` is the
        # full arrival window; ``effective`` is the exact shipped cohort.
        metrics: dict[str, float] = {}
        for subset, pool in (("all", batch.episodes), ("effective", effective)):
            metrics |= pool.metrics.to_wandb(prefix="train/agg", subset=subset)
            for env_name, env_pool in pool.by_env().items():
                metrics |= env_pool.metrics.to_wandb(prefix=f"train/{env_name}", subset=subset)

        # Progress / timing / env-share accounting (assembled here, not in the metrics
        # objects). ``num_tokens`` is over the full arrival window; the input/output breakdown is over
        # the effective (shipped) subset, summing the same ``vf.Trace`` token properties the metric
        # matrix reports.
        num_tokens = batch.episodes.num_total_tokens
        num_input = sum(record.trace.num_input_tokens for record in effective.records)
        num_output = sum(record.trace.num_output_tokens for record in effective.records)
        num_rollouts = batch.episodes.num_traces
        num_unique_examples = len({episode_group_id(episode) for episode in batch.episodes})
        metrics |= {
            "progress/tokens": num_tokens,
            "progress/input_tokens": num_input,
            "progress/output_tokens": num_output,
            "progress/rollouts": num_rollouts,
            "progress/tasks": num_unique_examples,
            "progress/total_tokens": self.progress.total_tokens,
            "progress/total_rollouts": self.progress.total_samples,
            "progress/total_tasks": self.progress.total_problems,
            "time/step": step_time,
            "time/pack": pack_time,
            "time/save_ckpt": save_ckpt_time,
            "time/wait_for_policy": self.wait_for_policy_time,
            "step": step,
        }
        # Staleness of the shipped cohort, decomposed into its in-flight and
        # in-queue shares; ``dropped`` counts queued traces the sink voided
        # since the last ship.
        staleness = [episode_staleness(episode, step) for episode in effective]
        if staleness:
            totals, in_flight, in_queue = (list(values) for values in zip(*staleness))
            metrics |= {
                "staleness/mean": sum(totals) / len(totals),
                "staleness/max": float(max(totals)),
                "staleness/in_flight/mean": sum(in_flight) / len(in_flight),
                "staleness/in_flight/max": float(max(in_flight)),
                "staleness/in_queue/mean": sum(in_queue) / len(in_queue),
                "staleness/in_queue/max": float(max(in_queue)),
            }
        metrics["staleness/dropped"] = float(self.train_sink.stale_drops)
        self.train_sink.stale_drops = 0
        for env_name, env_pool in batch.episodes.by_env().items():
            metrics[f"batch/{env_name}"] = env_pool.num_traces / batch.episodes.num_traces
        metrics |= self.train_source.metrics()
        await monitors.log(metrics, step=step)
        self.wait_for_policy_time = 0.0

        if self.heart is not None:
            self.heart.beat()

        self.progress.total_tokens += num_tokens
        self.progress.total_samples += num_rollouts
        self.progress.total_problems += num_unique_examples

        self.log_train_batch(batch, step=step, step_time=step_time)

        self.maybe_trigger_eval(self.progress.step)
        # Drain right after shipping the final batch. Waiting for a further
        # batch to fill would burn inference on data that can never train —
        # and with a tight ``max_staleness`` it never fills at all (the
        # versions it would need are never broadcast).
        if config.max_steps is not None and step >= config.max_steps:
            await self.start_draining("Shipped the final batch")
        trim_process_memory()

    async def start_draining(self, reason: str) -> None:
        """Stop scheduling train work and let the pipeline empty; triggered
        eval epochs still run to completion."""
        self.draining = True
        self.dispatcher.disable_train_scheduling()
        n_cancelled = await self.dispatcher.cancel_inflight_train_episodes()
        get_logger().info(
            f"{reason} — draining pipeline (cancelled {n_cancelled} in-flight "
            f"train episode(s); any in-flight evals will complete)"
        )

    def maybe_trigger_eval(self, step: int) -> None:
        """Fire eligible eval epochs and flip to ``PREFER_EVAL`` if anything
        fires. No-op when eval is not configured."""
        if self.eval_source is None:
            return
        fired = self.eval_source.trigger(step)
        if not fired:
            return
        reason = f"eval was triggered at step {step}"
        self.dispatcher.switch_mode(DispatcherMode.PREFER_EVAL, reason=reason)
        now = time.perf_counter()
        for env_name in fired:
            self.eval_triggered_at[(env_name, step)] = now
        assert self.eval_envs is not None
        census = {
            env_name: self.eval_envs.get(env_name).config.group_size * len(self.eval_envs.get(env_name).examples)
            for env_name in fired
        }
        get_logger().info(f"Starting evals in {', '.join(fired)} ({sum(census.values())} total rollouts)")

    def collect_pipeline_view(self) -> tuple[str, dict[str, float]]:
        """Pipeline view for the orchestrator's ``PeriodicLogger``. Returns
        ``(console_body, wandb_payload)``. Per-env ``(env=N, …)``
        breakdowns inline only when there's more than one train / eval env;
        the eval halves drop entirely when nothing is accumulating."""
        disp_gauges = self.dispatcher.gauges()
        disp_drain = self.dispatcher.metrics.drained(
            train_envs={e.name for e in self.train_envs},
            eval_envs={e.name for e in self.eval_envs} if self.eval_envs is not None else set(),
        )
        watcher_gauges = self.watcher.gauges()
        lag_stats = EventLoopLagStats.from_monitor(self.lag_monitor)

        inflight_by_env = self.dispatcher.inflight_by_env
        inflight_train = self.dispatcher.inflight_train_count
        inflight_eval = self.dispatcher.inflight_eval_count
        train_batch, train_target, _train_unit = self.train_sink.batch_progress()
        train_buffered = self.train_sink.buffered_count()
        train_batch_by_env = self.train_sink.pending_batch_by_env()
        eval_batches = self.eval_sink.batch_progress() if self.eval_sink is not None else []
        multi_train = len(self.train_envs) > 1
        multi_eval = self.eval_envs is not None and len(self.eval_envs) > 1

        # Train batch: finalized-group survivors only (0→target). Partial-group
        # arrivals are surfaced as a separate ``(+N buffered)`` addendum
        train_pct = train_batch / train_target if train_target else 0.0
        train_batch_part = f"Train batch {train_batch}/{train_target} ({train_pct:.1%})"
        if multi_train:
            pairs = [(e.name, train_batch_by_env.get(e.name, 0)) for e in self.train_envs]
            train_batch_part += " (" + ", ".join(f"{n}={v}" for n, v in pairs) + ")"
        if train_buffered:
            train_batch_part += f" (+{train_buffered} buffered)"

        eval_batch_part = ""
        for env, _step, eb, exp, _ebuf in eval_batches:
            eval_pct = eb / exp if exp else 0.0
            eval_batch_part += f" | {env} {eb}/{exp} ({eval_pct:.1%})"

        # Unified inflight tail: total, then train/eval split, then per-env
        # (only when more than one env of a kind makes the split ambiguous)
        inflight_part = (
            f"{inflight_train + inflight_eval} inflight episodes (train={inflight_train}, eval={inflight_eval}"
        )
        if multi_train or multi_eval:
            env_pairs = [(e.name, inflight_by_env.get(("train", e.name), 0)) for e in self.train_envs]
            if self.eval_envs is not None:
                env_pairs += [(e.name, inflight_by_env.get(("eval", e.name), 0)) for e in self.eval_envs]
            inflight_part += " | " + ", ".join(f"{n}={v}" for n, v in env_pairs)
        inflight_part += ")"

        body = train_batch_part + eval_batch_part + "; " + inflight_part

        payload: dict[str, float] = {**disp_gauges, **disp_drain, **watcher_gauges, **self.concurrency.gauges()}
        if lag_stats.n > 0:
            payload["event_loop_lag/min"] = lag_stats.min
            payload["event_loop_lag/mean"] = lag_stats.mean
            payload["event_loop_lag/median"] = lag_stats.median
            payload["event_loop_lag/p90"] = lag_stats.p90
            payload["event_loop_lag/p99"] = lag_stats.p99
            payload["event_loop_lag/max"] = lag_stats.max
            payload["event_loop_lag/n"] = float(lag_stats.n)
        return body, payload

    def log_train_batch(self, batch: TrainBatch, *, step: int, step_time: float) -> None:
        """Per-step ``Step …`` success line. Multi-env runs append an indented ``╰─`` line per env.
        Every quality metric (Reward, Trainable, Turns, Branches, Max Staleness, Truncation) is
        computed over exactly the traces shipped to the trainer this step (``batch.cohort``).
        ``Error``, ``Cancelled``, and ``Ratio`` are rates over the step's full arrival window —
        over the shipped set they are 0/0/share-of-shipped by construction, so the window is the
        only scope where they carry signal (and they stay disjoint: a cancellation is a pipeline
        decision, not a rollout failure)."""
        episodes = batch.episodes
        effective = batch.cohort.effective
        eff = effective.metrics
        n_generated = episodes.num_traces
        n_effective = effective.num_traces
        n_trainable = sum(is_trainable(record.trace) for record in effective.records)
        trainable_rate = (n_trainable / n_effective) if n_effective else 0.0
        max_staleness = max((episode_staleness(episode, step)[0] for episode in effective), default=0)

        head = (
            f"Step {step} | {format_time(step_time):>7} | Reward {eff.reward.mean():.4f} | "
            f"Trainable {n_trainable}/{n_effective} ({trainable_rate:.1%}) | "
            f"Turns {eff.num_turns.mean():.1f} | Branches {eff.num_branches.mean():.1f} | "
            f"Max Staleness {max_staleness} | "
            f"Error {episodes.metrics.has_error.mean():.1%} | Cancelled {episodes.metrics.cancelled.mean():.1%} | "
            f"Truncation {eff.is_truncated.mean():.1%}"
        )
        if len(self.train_envs) <= 1:
            get_logger().success(head)
            return

        window_by_env = episodes.by_env()
        shipped_by_env = effective.by_env()
        env_names = sorted(set(window_by_env) | set(shipped_by_env))
        name_width = max((len(name) for name in env_names), default=0)
        lines = [head]
        for env_name in env_names:
            pool = window_by_env.get(env_name, TrainEpisodes())
            env_eff_pool = shipped_by_env.get(env_name, TrainEpisodes())
            env_eff = env_eff_pool.metrics
            ratio = (pool.num_traces / n_generated) if n_generated else 0.0
            lines.append(
                f"╰─ {env_name:<{name_width}} | Ratio {ratio:.1%} | Reward {env_eff.reward.mean():.4f} | "
                f"Turns {env_eff.num_turns.mean():.1f} | Branches {env_eff.num_branches.mean():.1f} | "
                f"Max Staleness {max((episode_staleness(episode, step)[0] for episode in env_eff_pool), default=0)} | "
                f"Error {pool.metrics.has_error.mean():.1%} | Cancelled {pool.metrics.cancelled.mean():.1%} | "
                f"Truncation {env_eff.is_truncated.mean():.1%}"
            )
        get_logger().success("\n\t\t ".join(lines))

    async def finalize_eval_batch(self, batch: EvalBatch) -> None:
        """Persist + log one completed eval epoch through the monitors."""
        if not batch.episodes:
            get_logger().warning(f"Eval @ step={batch.step} env={batch.env_name}: no episodes returned, skipping log")
            return

        # The non-errored subset is logged on epoch completion (multiple eval envs share the
        # step's trace file — each epoch appends its cohort once, and every record carries
        # ``env_name``); the full returned cohort already streamed into ``all`` on arrival.
        await monitors.log(batch.episodes.effective.vf_episodes, batch.step, "eval", "effective")
        policy_spans = [eval_work(episode).policy for episode in batch.episodes]
        if any(span is None for span in policy_spans):
            raise ValueError(f"Eval {batch.env_name} step {batch.step} is missing policy provenance")
        policy_versions = {span.start for span in policy_spans if span is not None}
        policy_version = min(policy_versions)
        if len(policy_versions) > 1:
            get_logger().warning(
                f"Eval {batch.env_name} step {batch.step} had mixed policy versions: {sorted(policy_versions)}"
            )
        # Episode metrics over {all,effective} (eval batches are per-env, so no `agg` axis).
        # ``effective`` = non-errored; pass@k / pass^k only over the effective set.
        episodes = batch.episodes
        effective = episodes.effective
        metrics: dict[str, float] = {}
        for subset, pool in (("all", episodes), ("effective", effective)):
            metrics |= pool.metrics.to_wandb(prefix=f"eval/{batch.env_name}", subset=subset)
        metrics[f"eval/{batch.env_name}/policy_version"] = float(policy_version)
        metrics["step"] = float(batch.step)
        await monitors.log(metrics, step=batch.step)

        # Success line — quality metrics over the effective set, error rate over the full returned
        # cohort. ``Stat.mean()`` is 0.0 for an empty set.
        eff, full = effective.metrics, episodes.metrics
        triggered_at = self.eval_triggered_at.pop((batch.env_name, batch.step), None)
        elapsed = (time.perf_counter() - triggered_at) if triggered_at is not None else 0.0
        get_logger().success(
            f"Evaluated {batch.env_name} (Step {batch.step}) | "
            f"Policy v{policy_version} | {format_time(elapsed):>7} | Reward {eff.reward.mean():.4f} | "
            f"Turns {eff.num_turns.mean():.1f} | Branches {eff.num_branches.mean():.1f} | "
            f"Error {full.has_error.mean():.1%} | Truncation {eff.is_truncated.mean():.1%}"
        )

    async def maybe_save_ckpt(self, step: int) -> float:
        """Checkpoint the step just shipped if it's an interval boundary. Returns
        elapsed time (0.0 when no save happened)."""
        if self.config.ckpt is None or not self.config.ckpt.interval:
            return 0.0
        # The final step's checkpoint is written once in ``start()``'s teardown; skip it here so
        # we don't double-save. This mirrors the trainer (its is_last_step skips the in-loop save).
        if self.config.max_steps is not None and step >= self.config.max_steps:
            return 0.0
        if step % self.config.ckpt.interval != 0:
            return 0.0
        get_logger().info(f"Saving checkpoint at step {step}")
        t = time.perf_counter()
        # Synchronous on purpose: the payload is tiny, and snapshotting on the
        # event loop keeps the dispatcher from mutating TrainSource mid-save
        self.ckpt_manager.save(self.progress, self.train_source, step)
        return time.perf_counter() - t

    def update_dispatch_gate(self) -> None:
        """Pause/resume the dispatcher based on how far the in-flight batch runs
        ahead of ``policy.version``. ``progress.step`` is always the batch being
        collected — advanced right after shipping — so both call sites (ship time
        here, policy update in ``on_new_version``) share one lead formula. Steps
        are 1-indexed while policy versions stay 0-indexed, so the shipped-batch
        count is ``progress.step - 1``."""
        lead = (self.progress.step - 1) - self.policy.version
        gate = self.dispatcher.dispatch_allowed
        was_set = gate.is_set()
        if lead > TARGET_LAG:
            if was_set:
                get_logger().info(
                    "Pausing dispatcher to prevent orchestrator from racing from trainer. Waiting for new policy..."
                )
                self.gate_closed_at = time.perf_counter()
            gate.clear()
        else:
            if not was_set:
                get_logger().info("Resuming dispatcher")
                if self.gate_closed_at is not None:
                    self.wait_for_policy_time += time.perf_counter() - self.gate_closed_at
                    self.gate_closed_at = None
            gate.set()

    async def on_version_pending(self, step: int) -> None:
        """``VersionObserver`` hook, fired at publish confirmation (pre-apply):
        ``policy.version`` already carries the new version, so wake a held ship."""
        if self.model_express is not None:
            await asyncio.to_thread(self.model_express.set_status, p2p_pb2.SOURCE_STATUS_READY)
        self.version_advanced.set()

    async def on_new_version(self, step: int) -> None:
        """``VersionObserver`` hook: the weight update completed;
        re-evaluate the dispatch gate (may resume if the trainer caught up)."""
        if self.model_express is not None:
            await asyncio.to_thread(self.model_express.set_status, p2p_pb2.SOURCE_STATUS_INITIALIZING)
        self.update_dispatch_gate()

    async def stop(self) -> None:
        """Bounded best-effort teardown of all components. Has a global
        timeout so a wedged peer can't keep the process alive forever —
        training artifacts are already persisted before this is reached."""

        async def teardown() -> None:
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
            if getattr(self, "policy_inference", None) is not None:
                await self.policy_inference.stop()
            if self.train_envs is not None:
                for env in self.train_envs:
                    for pool in (*env.sampler.connected_pools, *env.algorithm.connected_pools):
                        await pool.stop()

        task = asyncio.create_task(teardown())
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
    """Top-level entrypoint. Wrapped in ``@clean_exit`` so wandb is flushed
    on exit (success or crash); keeps that out of the class.
    """
    await Orchestrator(config).start()


def main() -> None:
    from prime_rl.utils.config import cli
    from prime_rl.utils.process import set_proc_title

    set_proc_title("Orchestrator")
    import uvloop

    uvloop.install()
    asyncio.run(run_orchestrator(cli(OrchestratorConfig)))


if __name__ == "__main__":
    main()
