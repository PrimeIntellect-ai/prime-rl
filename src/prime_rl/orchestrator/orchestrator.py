"""Async-pipelined RL orchestrator.

``Orchestrator`` owns the shared state (policy, progress, ckpt, monitor)
and drives the pipeline. Components are single-purpose:

- ``RolloutDispatcher`` schedules rollouts; emits ``Rollout`` (train/eval
  discriminated by ``kind``) on its queue.
- ``TrainSink`` ingests train rollouts (tokenize → advantages → filters)
  and returns a ``TrainBatch`` when the threshold is met.
- ``EvalSink`` ingests eval rollouts and returns an ``EvalBatch`` (the full
  returned cohort) on epoch completion.
- ``TrainRollouts`` / ``EvalRollouts`` carry the rollouts and build the per-step W&B metrics
  (``batch.rollouts.metrics`` / ``.effective.metrics``).
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
from typing import TYPE_CHECKING

import tomli_w
import verifiers.v1 as vf

if TYPE_CHECKING:
    from renderers.base import Renderer
    from transformers.tokenization_utils import PreTrainedTokenizer

    from prime_rl.orchestrator.ckpt import CheckpointManager
    from prime_rl.transport.base import TrainingBatchSender
    from prime_rl.utils.client import InferencePool
    from prime_rl.utils.monitor.base import Monitor
import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before transitive imports
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.ckpt import setup_ckpt_manager
from prime_rl.orchestrator.dispatcher import DispatcherMetrics, DispatcherMode, RolloutDispatcher
from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.filters import setup_filters
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.patches import (
    monkey_patch_chat_completion_logprobs,
    monkey_patch_oai_iterable_types,
)
from prime_rl.orchestrator.periodic_logger import PeriodicLogger
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import (
    EvalBatch,
    Policy,
    Progress,
    Rollout,
    TrainBatch,
)
from prime_rl.orchestrator.utils import (
    get_weight_dir,
    intercept_vf_logging,
    save_rollouts,
    set_default_executor,
    setup_policy_inference_pool,
    trim_process_memory,
)
from prime_rl.orchestrator.watcher import WeightWatcher
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.transport import TrainingBatch, setup_training_batch_sender
from prime_rl.utils.async_utils import EventLoopLagMonitor, EventLoopLagStats, safe_cancel
from prime_rl.utils.client import init_nccl_broadcast
from prime_rl.utils.config import to_toml_dict
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import format_time, get_logger, setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pathing import get_log_dir, get_trace_path
from prime_rl.utils.usage_reporter import UsageReporter
from prime_rl.utils.utils import (
    clean_exit,
    resolve_latest_ckpt_step,
)

monkey_patch_oai_iterable_types()
monkey_patch_chat_completion_logprobs()


# Wall-clock budget for post-training cleanup; force-exit if graceful
# shutdown wedges (env-server ZMQ recv, vLLM admin aclose, etc)
SHUTDOWN_TIMEOUT_S = 300

# Abort after this many consecutive train batches drop all rollouts to
# post-batch filters — usually a misconfigured filter or homogeneous-reward
# dataset; fail loudly instead of spinning
MAX_CONSECUTIVE_EMPTY_BATCHES = 10

# Maximum batches the orchestrator may run ahead of the trainer: shipping
# batch N holds until the trainer has published policy v{N-1-TARGET_LAG}
# (``finalize_train_batch``). Dispatch needs no lead gate of its own —
# ``train_needed`` stops it once the batch being collected is covered, and the
# hold stops the batch counter from running ahead.
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
    ckpt_manager: CheckpointManager | None
    component_tasks: list[asyncio.Task]

    # Always set by ``setup()``
    tokenizer: PreTrainedTokenizer
    policy_inference: InferencePool
    monitor: Monitor
    sender: TrainingBatchSender
    train_envs: TrainEnvs
    train_source: TrainSource
    train_sink: TrainSink
    dispatcher: RolloutDispatcher
    watcher: WeightWatcher
    lag_monitor: EventLoopLagMonitor
    periodic_logger: PeriodicLogger

    # Set by ``setup()`` only when relevant config is present
    renderer: Renderer | None
    mm_token_type_ids_mapping: dict[int, int] | None
    heart: Heartbeat | None
    usage_reporter: UsageReporter | None
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
        algorithms = sorted({env.algo.type for env in config.train.env if env.algo is not None})
        get_logger().info(f"Starting orchestrator (algorithm: {', '.join(algorithms)})")

        if config.bench:
            get_logger().warning(f"Running in benchmark mode (max_steps={config.max_steps})")

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
        self.wait_for_policy_time = 0.0
        # Newest trainer-produced version the watcher has seen (pre-apply);
        # ``policy.version`` trails it by the inference weight update
        self.published_version = 0
        # Pulsed by the version hooks so a held ship can re-check the version
        self.version_advanced = asyncio.Event()
        self.component_tasks = []

        # Optional attributes — ``setup()`` populates them when the relevant
        # config is present
        self.renderer = None
        self.mm_token_type_ids_mapping = None
        self.heart = None
        self.usage_reporter = None
        self.inference_metrics = None
        self.eval_envs = None
        self.eval_sink = None
        self.eval_source = None
        self.lora_name = None
        self.resume_step = None
        self.lag_task = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Install envs, load models/pools, resume from checkpoint, and
        construct the pipeline components."""
        config = self.config
        set_default_executor()

        # Persist the resolved config alongside the run
        config_dir = config.output_dir / "control"
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "orch.toml", "wb") as f:
            tomli_w.dump(to_toml_dict(config), f)

        get_logger().info(f"Initializing tokenizer ({config.tokenizer})")
        self.tokenizer = setup_tokenizer(config.tokenizer)

        # The one model prime-rl hosts: the live policy. Frozen model
        # references are external endpoints — each env's Algorithm builds its
        # own pools in ``setup()`` below.
        get_logger().info(
            f"Initializing policy inference pool (base_url={', '.join(config.model.client.base_url)}, "
            f"model={config.model.name})"
        )
        self.renderer, self.policy_inference = await setup_policy_inference_pool(
            config=config, tokenizer=self.tokenizer
        )
        self.mm_token_type_ids_mapping = (
            getattr(self.renderer, "mm_token_type_id_map", None) if self.renderer is not None else None
        )
        if self.mm_token_type_ids_mapping == {}:
            self.mm_token_type_ids_mapping = None

        get_logger().info(f"Initializing monitor (wandb={config.wandb}, prime_monitor={config.prime_monitor})")
        self.monitor = setup_monitor(
            wandb_config=config.wandb,
            prime_config=config.prime_monitor,
            output_dir=config.output_dir,
            tokenizer=self.tokenizer,
            run_config=config,
            keep_full_history=config.bench,
            train_env_names=[env.resolved_name for env in config.train.env],
            eval_env_names=[env.resolved_name for env in config.eval.env] if config.eval is not None else [],
        )

        if config.heartbeat is not None:
            self.heart = Heartbeat(config.heartbeat.url)

        usage_base_url = os.environ.get("PI_USAGE_BASE_URL")
        usage_api_key = os.environ.get("PI_USAGE_API_KEY")
        if usage_base_url and usage_api_key:
            self.usage_reporter = UsageReporter()

        # Filters apply to train rollouts only
        pre_filters = setup_filters(config.pre_batch_filters, vocab_size=self.tokenizer.vocab_size, kind="pre-batch")
        post_filters = setup_filters(config.post_batch_filters, vocab_size=self.tokenizer.vocab_size, kind="post-batch")

        get_logger().info("Loading training environments")
        self.train_envs = TrainEnvs(
            config.train.env, policy_pool=self.policy_inference, renderer_config=config.renderer
        )
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

        if config.ckpt is not None and config.ckpt.resume_step is not None and self.ckpt_manager is not None:
            if config.ckpt.resume_step == -1:
                self.resume_step = resolve_latest_ckpt_step(self.ckpt_manager.ckpt_dir)
            else:
                self.resume_step = config.ckpt.resume_step

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

        if config.wandb is not None and config.collect_inference_metrics:
            self.inference_metrics = InferenceMetricsCollector(
                self.policy_inference.admin_clients,
                roles=config.inference_metrics_roles,
            )
            await self.inference_metrics.start()

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

        get_logger().info(f"Initializing training batch sender ({config.rollout_transport})")
        self.sender = setup_training_batch_sender(config.output_dir, config.rollout_transport)

        self.lora_name = config.model.lora.name if config.model.lora else None

        if self.resume_step is not None and self.ckpt_manager is not None:
            self.ckpt_manager.load(self.progress, step=self.resume_step)
            # The checkpoint finished step ``resume_step``; resume at the next step. Derive the step
            # from ``resume_step`` (not the loaded progress.step) so it stays coordinated with the
            # trainer even when ``ckpt.skip_progress`` left the counter unrestored.
            self.progress.step = self.resume_step + 1
            get_logger().info(f"Resuming orchestrator from checkpoint step {self.resume_step}")
        else:
            get_logger().info("Training from scratch")

        # Sync inference to the incoming policy before the first step when resuming or when using
        # NCCL, which rendezvouses with the trainer's first-step broadcast (train.py). A fresh
        # filesystem run needs no sync (the base model IS policy v0) and must not wait for a
        # startup broadcast: runs registered after the trainer's first step never receive one.
        if self.resume_step is not None or config.weight_broadcast.type == "nccl":
            sync_version = self.resume_step if self.resume_step is not None else 0
            check_exists = config.weight_broadcast.type != "nccl"
            # Without a ckpt block, fall back to a default timeout instead of not waiting at all.
            wait_timeout = config.ckpt.wait_for_weights_timeout if config.ckpt else STARTUP_WEIGHT_WAIT_TIMEOUT_S
            weights_path = get_weight_dir(
                config.output_dir, sync_version, check_exists=check_exists, wait_timeout=wait_timeout
            )
            await self.policy_inference.update_weights(weights_path, lora_name=self.lora_name, step=sync_version)
            if self.lora_name is not None:
                self.policy_inference.update_model_name(self.lora_name)
                self.policy.model_name = self.lora_name
            self.policy.version = sync_version

        self.train_source = TrainSource(self.train_envs, seed=42)
        self.eval_source: EvalSource | None = (
            EvalSource(
                self.eval_envs,
                config.eval,
                is_resumed=self.resume_step is not None,
            )
            if config.eval is not None and self.eval_envs is not None
            else None
        )

        assert config.max_inflight_rollouts is not None, "max_inflight_rollouts must be resolved before dispatcher init"
        log_interval = config.log.interval
        wandb_enabled = config.wandb is not None
        self.dispatcher = RolloutDispatcher(
            train_envs=self.train_envs,
            eval_envs=self.eval_envs,
            train_source=self.train_source,
            eval_source=self.eval_source,
            policy_pool=self.policy_inference,
            policy=self.policy,
            max_inflight_rollouts=config.max_inflight_rollouts,
            tasks_per_minute=config.tasks_per_minute,
            train_needed=self.train_needed,
        )
        self.train_sink = TrainSink(
            config,
            tokenizer=self.tokenizer,
            train_envs=self.train_envs,
            mm_token_type_ids_mapping=self.mm_token_type_ids_mapping,
            batch_size=config.batch_size,
            token_batch_size=config.token_batch_size,
            pre_filters=pre_filters,
            post_filters=post_filters,
        )
        self.eval_sink = EvalSink(eval_envs=self.eval_envs) if self.eval_envs is not None else None
        self.watcher = WeightWatcher(
            config,
            policy=self.policy,
            inference=self.policy_inference,
            observers=[self],
            lora_name=self.lora_name,
            ckpt_step=self.policy.version,
        )
        # Single periodic logger for the whole pipeline. It's the only
        # consumer of ``dispatcher.metrics.drained()`` (which clears on read)
        self.lag_monitor = EventLoopLagMonitor()
        self.periodic_logger = PeriodicLogger(
            name="Pipeline",
            collect=self.collect_pipeline_view,
            metric_keys=[
                *list(self.dispatcher.gauges().keys()),
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
        # step, unless ``eval.skip_first_step=True`` (or this is a resume)
        self.maybe_trigger_eval(self.progress.step)

        # Anchor step-time clock so the first step measures startup → first batch
        self.last_batch_at = time.perf_counter()

        # ``clean_exit`` stays False if ``main_loop`` raises (signal-driven
        # CancelledError, KeyboardInterrupt, or a real error), so the teardown
        # logs a forced-cleanup warning instead of a clean-exit success.
        clean_exit = False
        try:
            await self.main_loop()
            clean_exit = True
        finally:
            elapsed = format_time(time.perf_counter() - start_time)
            if clean_exit:
                get_logger().success(f"Orchestrator step loop done in {elapsed}")
            else:
                get_logger().warning(f"Orchestrator interrupted after {elapsed} — forcing cleanup (not a clean exit)")
            self.monitor.save_final_summary()
            # ``progress.step`` points at the next (unshipped) step; the last finished step is
            # ``progress.step - 1``. Checkpoint it as ``step_{progress.step - 1}`` (no-op before the
            # first ship).
            if self.ckpt_manager is not None and self.progress.step > 1:
                self.progress.step -= 1
                get_logger().info("Writing final checkpoint")
                self.ckpt_manager.save(self.progress, step=self.progress.step)
            await self.stop()
            if clean_exit:
                get_logger().success("Orchestrator finished.")
            else:
                get_logger().warning("Orchestrator cleanup complete (forced).")
            trim_process_memory()

    async def main_loop(self) -> None:
        """Consume episodes (``list[Rollout]``) from the dispatcher and route them
        to the train / eval sink. Both sinks return a finalized batch (or
        ``None``) from ``add()``; we just dispatch on the result."""
        while not self.stopped.is_set():
            if self.draining and self.dispatcher.is_idle:
                get_logger().info("Pipeline drained, exiting main loop")
                self.stopped.set()
                break

            try:
                episode: list[Rollout] = await asyncio.wait_for(self.dispatcher.out_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            # Every completed rollout — errored, filtered, or never batched — lands in the
            # ``all`` trace file the moment it arrives, so it survives crashes and drains.
            # Train rollouts belong to the batch window currently collecting (``progress.step``),
            # eval rollouts to the step whose eval triggered them.
            kind = episode[0].kind
            step = episode[0].eval_step if kind == "eval" else self.progress.step
            assert step is not None
            run: vf.RunInfo = (
                vf.EvalRunInfo(id=self.monitor.run_id, step=step)
                if kind == "eval"
                else vf.TrainRunInfo(id=self.monitor.run_id, step=step)
            )
            for rollout in episode:
                rollout.stamp(
                    run,
                    env_name=rollout.env_name,
                    group_id=str(rollout.group_id),
                    episode_id=rollout.episode_id,
                    policy_version=rollout.policy_version,
                )
            await asyncio.to_thread(
                save_rollouts,
                [rollout.to_record() for rollout in episode],
                get_trace_path(self.config.output_dir, step, kind, "all"),
            )

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

    async def finalize_train_batch(self, batch: TrainBatch) -> None:
        """Ship one ``TrainBatch`` out to the trainer and handle the I/O
        side-effects (ckpt, save_rollouts, reference scoring, sender.send,
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

        if config.max_steps is not None and step > config.max_steps:
            self.draining = True
            self.dispatcher.disable_train_scheduling()
            n_cancelled = await self.dispatcher.cancel_inflight_train_rollouts()
            get_logger().info(
                f"Draining pipeline (cancelled {n_cancelled} in-flight train rollout(s); "
                f"any in-flight evals will complete)"
            )
            return

        if not batch.samples:
            self.consecutive_empty_batches += 1
            get_logger().warning(
                f"Step {step}: empty train batch (0 of {len(batch.rollouts)} generated rollouts shipped — "
                f"all errored or filtered out) "
                f"(consecutive empty batches: {self.consecutive_empty_batches}/{MAX_CONSECUTIVE_EMPTY_BATCHES})"
            )
            if self.consecutive_empty_batches >= MAX_CONSECUTIVE_EMPTY_BATCHES:
                raise RuntimeError(
                    f"{self.consecutive_empty_batches} consecutive empty train batches — "
                    "check filter config (pre_batch_filters / post_batch_filters) or task difficulty."
                )
            return
        self.consecutive_empty_batches = 0
        n_trainable = sum(1 for r in batch.rollouts if r.is_trainable)
        if n_trainable / len(batch.rollouts) <= 0.1:
            get_logger().warning(
                f"Only {n_trainable}/{len(batch.rollouts)} generated rollouts are trainable "
                f"({n_trainable / len(batch.rollouts):.1%}) — consider reviewing task difficulty / filter config"
            )

        # Hold the ship until the trainer has produced policy v{step-1-TARGET_LAG}.
        # Demand only stops *new* rollouts; on tiny fast envs already-covered batches
        # complete faster than the trainer consumes them, letting the orchestrator
        # finish all its steps and tear down while the trainer still has pending NCCL
        # broadcasts — which then hang forever without the watcher to serve the
        # receive handshake. This never waits for an unpublishable version: under
        # NCCL the trainer's last broadcast is v{max_steps-2} (``nccl_broadcast_unused``),
        # exactly what the final batch requires. Bench runs have no trainer
        # publishing versions, so they ship freely.
        #
        # Under NCCL the wait is on the *applied* version: the trainer blocks in the
        # receive handshake until the watcher applies, so releasing on publish could
        # tear the watcher down mid-handshake and strand the trainer — and the apply
        # costs the hold nothing (the trainer can't outrun it). Filesystem broadcast
        # decouples the two: the trainer never blocks, so the *published* version
        # (seen by the watcher pre-apply) suffices — waiting for the apply would
        # serialize every inference weight reload into the ship path for no benefit.
        wait_for_apply = config.weight_broadcast.type == "nccl"
        required_version = step - 1 - TARGET_LAG
        if not config.bench:
            hold_start = time.perf_counter()
            logged = False
            while True:
                self.version_advanced.clear()
                version = self.policy.version if wait_for_apply else max(self.published_version, self.policy.version)
                if version >= required_version:
                    break
                if not logged:
                    logged = True
                    get_logger().info(
                        f"Holding batch {step} until the trainer publishes policy v{required_version} "
                        f"(currently v{version})"
                    )
                await self.version_advanced.wait()
            if logged:
                self.wait_for_policy_time += time.perf_counter() - hold_start

        # Stamp each rollout's consumption lag — batch ``step`` trains on policy
        # v{step-1}, so a rollout generated by v{k} trains (step-1)-k versions
        # off-policy, queue time included. Frozen-sourced rollouts stay 0.
        for r in batch.rollouts:
            if self.train_envs.get(r.env_name).sampler.samples_from_live_policy:
                r.off_policy_steps = (step - 1) - r.policy_version

        # The effective (clean, trained-on) subset lands in the per-step ``effective`` trace file
        # at ship time; the full arrival window already streamed into ``all`` on arrival.
        # to_record drops the per-node training tensors — they're for training, not the rollout
        # record, and can't round-trip json (raw numpy bytes).
        effective = batch.rollouts.effective
        records = [r.to_record() for r in effective]
        await asyncio.to_thread(save_rollouts, records, get_trace_path(config.output_dir, step, "train", "effective"))

        await self.sender.send(TrainingBatch(examples=batch.samples, step=step))
        self.progress.step += 1
        # The batch counter moved, so the staleness cutoff moved with it —
        # sweep buffered survivors that can no longer ship in time. In-flight
        # groups are swept on the next weight update instead (their aborts
        # must land while the engine is stepping, see ``on_version_pending``).
        self.train_sink.drop_stale((self.progress.step - 1) - self.config.max_off_policy_steps)
        # Checkpoint the step we just shipped (resume point: continue at step + 1).
        save_ckpt_time = await self.maybe_save_ckpt(step)
        trim_process_memory()

        # Rollout metrics over the {agg,<env>} × {all,effective} matrix. ``batch.rollouts`` is the
        # full arrival window (errored + filtered included); ``.effective`` is the clean subset.
        metrics: dict[str, float] = {}
        for subset, pool in (("all", batch.rollouts), ("effective", effective)):
            metrics |= pool.metrics.to_wandb(prefix="train/agg", subset=subset)
            for env_name, env_pool in pool.by_env().items():
                metrics |= env_pool.metrics.to_wandb(prefix=f"train/{env_name}", subset=subset)

        # Progress / timing / env-share / pre-filter accounting (assembled here, not in the metrics
        # objects). ``num_tokens`` is over the full arrival window; the input/output breakdown is over
        # the effective (shipped) subset, summing the same ``vf.Trace`` token properties the metric
        # matrix reports.
        num_tokens = sum(r.num_total_tokens for r in batch.rollouts)
        num_input = sum(r.num_input_tokens for r in effective)
        num_output = sum(r.num_output_tokens for r in effective)
        num_rollouts = len(batch.rollouts)
        num_unique_examples = len({r.group_id for r in batch.rollouts})
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
            "time/save_ckpt": save_ckpt_time,
            "time/wait_for_policy": self.wait_for_policy_time,
            "step": step,
        }
        for env_name, env_pool in batch.rollouts.by_env().items():
            metrics[f"batch/{env_name}"] = len(env_pool) / len(batch.rollouts)
        if self.train_sink.pre_filter_seen > 0:
            metrics["pre_filters/all/dropped_rate"] = (
                self.train_sink.pre_filter_dropped / self.train_sink.pre_filter_seen
            )
            for name, count in self.train_sink.pre_filter_dropped_by_name.items():
                metrics[f"pre_filters/all/{name}/rate"] = count / self.train_sink.pre_filter_seen
        self.monitor.log(metrics, step=step)
        self.wait_for_policy_time = 0.0
        self.monitor.log_samples(effective.rollouts, step=step)
        self.monitor.log_distributions(
            distributions={
                "rewards": [r.reward for r in effective],
                "advantages": [a for r in effective if (a := r.scalar_advantage()) is not None],
            },
            step=step,
        )

        if self.usage_reporter is not None:
            run_id = os.getenv("RUN_ID", "")
            if run_id:
                self.usage_reporter.report_training_usage(
                    run_id=run_id,
                    step=step,
                    tokens=num_input + num_output,
                )
        if self.heart is not None:
            self.heart.beat()

        self.progress.total_tokens += num_tokens
        self.progress.total_samples += num_rollouts
        self.progress.total_problems += num_unique_examples

        self.log_train_batch(batch, step=step, step_time=step_time)

        self.train_sink.reset_pre_filter_stats()
        self.maybe_trigger_eval(self.progress.step)
        trim_process_memory()

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
        total_rollouts = sum(
            self.eval_envs.get(env_name).config.group_size * len(self.eval_envs.get(env_name).examples)
            for env_name in fired
        )
        get_logger().info(f"Starting evals in {', '.join(fired)} ({total_rollouts} total rollouts)")

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
            f"{inflight_train + inflight_eval} inflight rollouts (train={inflight_train}, eval={inflight_eval}"
        )
        if multi_train or multi_eval:
            env_pairs = [(e.name, inflight_by_env.get(("train", e.name), 0)) for e in self.train_envs]
            if self.eval_envs is not None:
                env_pairs += [(e.name, inflight_by_env.get(("eval", e.name), 0)) for e in self.eval_envs]
            inflight_part += " | " + ", ".join(f"{n}={v}" for n, v in env_pairs)
        inflight_part += ")"

        body = train_batch_part + eval_batch_part + "; " + inflight_part

        payload: dict[str, float] = {**disp_gauges, **disp_drain, **watcher_gauges}
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
        ``Error`` is the sink-level rate (errored arrivals / total arrivals, over the full window);
        the quality metrics are over the effective (clean, trained-on) subset; ``Trainable`` is
        relative to all generated rollouts."""
        rollouts = batch.rollouts
        effective = rollouts.effective
        eff = effective.metrics
        n_generated = len(rollouts)
        n_trainable = sum(1 for r in rollouts if r.is_trainable)
        trainable_rate = (n_trainable / n_generated) if n_generated else 0.0
        max_off_policy = max((r.off_policy_steps for r in effective), default=0)

        head = (
            f"Step {step} | {format_time(step_time):>7} | Reward {eff.reward.mean():.4f} | "
            f"Trainable {n_trainable}/{n_generated} ({trainable_rate:.1%}) | "
            f"Turns {eff.num_turns.mean():.1f} | Branches {eff.num_branches.mean():.1f} | "
            f"Max Off-Policy {max_off_policy} | "
            f"Error {rollouts.metrics.has_error.mean():.1%} | Truncation {eff.is_truncated.mean():.1%}"
        )
        if len(self.train_envs) <= 1:
            get_logger().success(head)
            return

        by_env = rollouts.by_env()
        name_width = max((len(n) for n in by_env), default=0)
        lines = [head]
        for env_name in sorted(by_env):
            pool = by_env[env_name]
            env_eff_pool = pool.effective
            env_eff = env_eff_pool.metrics
            ratio = (len(pool) / n_generated) if n_generated else 0.0
            lines.append(
                f"╰─ {env_name:<{name_width}} | Ratio {ratio:.1%} | Reward {env_eff.reward.mean():.4f} | "
                f"Turns {env_eff.num_turns.mean():.1f} | Branches {env_eff.num_branches.mean():.1f} | "
                f"Max Off-Policy {max((r.off_policy_steps for r in env_eff_pool), default=0)} | "
                f"Error {pool.metrics.has_error.mean():.1%} | Truncation {env_eff.is_truncated.mean():.1%}"
            )
        get_logger().success("\n\t\t ".join(lines))

    async def finalize_eval_batch(self, batch: EvalBatch) -> None:
        """Persist + log one completed eval epoch (save_rollouts,
        monitor.log_eval_samples, monitor.log)."""
        if not batch.rollouts:
            get_logger().warning(f"Eval @ step={batch.step} env={batch.env_name}: no rollouts returned, skipping log")
            return

        # The non-errored subset lands in the per-step ``effective`` trace file on epoch
        # completion (multiple eval envs share the step file — each epoch appends its cohort
        # once, and every record carries ``env_name``); the full returned cohort already
        # streamed into ``all`` on arrival.
        records = [r.to_record() for r in batch.rollouts.effective]
        await asyncio.to_thread(
            save_rollouts, records, get_trace_path(self.config.output_dir, batch.step, "eval", "effective")
        )
        self.monitor.log_eval_samples(batch.rollouts, env_name=batch.env_name, step=batch.step)
        policy_versions = {r.policy_version for r in batch.rollouts}
        policy_version = min(policy_versions)
        if len(policy_versions) > 1:
            get_logger().warning(
                f"Eval {batch.env_name} step {batch.step} had mixed policy versions: {sorted(policy_versions)}"
            )
        # Rollout metrics over {all,effective} (eval batches are per-env, so no `agg` axis).
        # ``effective`` = non-errored; pass@k / pass^k only over the effective set.
        rollouts = batch.rollouts
        effective = rollouts.effective
        metrics: dict[str, float] = {}
        for subset, pool in (("all", rollouts), ("effective", effective)):
            metrics |= pool.metrics.to_wandb(prefix=f"eval/{batch.env_name}", subset=subset)
        metrics[f"eval/{batch.env_name}/policy_version"] = float(policy_version)
        metrics["step"] = float(batch.step)
        self.monitor.log(metrics, step=batch.step)

        # Success line — reward / turns / truncation over the effective set, error rate + branches
        # over the full returned cohort. ``Stat.mean()`` is 0.0 for an empty set.
        eff, full = effective.metrics, rollouts.metrics
        triggered_at = self.eval_triggered_at.pop((batch.env_name, batch.step), None)
        elapsed = (time.perf_counter() - triggered_at) if triggered_at is not None else 0.0
        get_logger().success(
            f"Evaluated {batch.env_name} (Step {batch.step}) | "
            f"Policy v{policy_version} | {format_time(elapsed):>7} | Reward {eff.reward.mean():.4f} | "
            f"Turns {eff.num_turns.mean():.1f} | Branches {full.num_branches.mean():.1f} | "
            f"Error {full.has_error.mean():.1%} | Truncation {eff.is_truncated.mean():.1%}"
        )

    async def maybe_save_ckpt(self, step: int) -> float:
        """Checkpoint the step just shipped if it's an interval boundary. Returns
        elapsed time (0.0 when no save happened)."""
        if self.ckpt_manager is None or self.config.ckpt is None or not self.config.ckpt.interval:
            return 0.0
        # The final step's checkpoint is written once in ``start()``'s teardown; skip it here so
        # we don't double-save. This mirrors the trainer (its is_last_step skips the in-loop save).
        if self.config.max_steps is not None and step >= self.config.max_steps:
            return 0.0
        if step % self.config.ckpt.interval != 0:
            return 0.0
        get_logger().info(f"Saving checkpoint at step {step}")
        t = time.perf_counter()
        await asyncio.to_thread(self.ckpt_manager.save, self.progress, step)
        return time.perf_counter() - t

    def train_needed(self) -> bool:
        """Whether the dispatcher may start another train rollout — injected as
        its scheduling predicate, re-evaluated per scheduled group. True while
        the batch being collected — plus up to ``TARGET_LAG`` batches of
        lookahead, staleness budget permitting — isn't covered yet by buffered,
        partial-group, and in-transit rollouts. The lookahead keeps dispatch
        continuous in steady state (each arrival frees demand for the next
        start) instead of synchronizing it into one wave per batch, which
        idles inference whenever rollouts spend time outside the engine
        (multi-turn env round-trips) and stampedes the env server. False while
        rollouts started now would already exceed ``max_off_policy_steps``."""
        lag = (self.progress.step - 1) - self.policy.version
        if lag > self.config.max_off_policy_steps:
            return False
        # A rollout started now that lands l batches ahead trains lag+l versions
        # behind; only look ahead as far as the staleness budget allows.
        lookahead = min(TARGET_LAG, self.config.max_off_policy_steps - lag)
        # ``out_q`` backlog counts as coverage: while a ship holds for the
        # trainer, the main loop isn't routing arrivals into the sink, and
        # without this the dispatcher would over-provision through every hold
        # (the surplus then spills into later batches, aging as it goes).
        # Eval rollouts in the backlog undercount train demand briefly —
        # conservative, and it self-corrects as the queue drains.
        # A batch held for ship (already popped from the sink) deliberately does
        # NOT count as coverage: dispatch-ahead during the hold is what keeps
        # inference busy through trainer-bound waits, and its staleness stays
        # bounded by the freshness-scaled lookahead and the drop cutoff.
        in_transit = self.dispatcher.inflight_train_count + self.dispatcher.out_q.qsize()
        return self.train_sink.remaining_rollout_demand(in_transit, lookahead) > 0

    async def on_version_pending(self, step: int) -> None:
        """``VersionObserver`` hook, fired *before* the engines pause for the
        weight update: record the newly published version (waking a held ship
        — filesystem holds release on publish), then drop everything already
        past the staleness cutoff — in-flight groups (their aborts must be
        processed while the engine is still stepping) and buffered survivors
        in the sink. The cutoff is the oldest generation version that can
        still train within ``max_off_policy_steps``, were it to ship in the
        batch being collected (which trains on policy v{progress.step - 1})."""
        self.published_version = max(self.published_version, step)
        self.version_advanced.set()
        cutoff = (self.progress.step - 1) - self.config.max_off_policy_steps
        await self.dispatcher.cancel_stale_train_groups(cutoff)
        self.train_sink.drop_stale(cutoff)

    async def on_new_version(self, step: int) -> None:
        """``VersionObserver`` hook: the watcher just advanced ``policy.version``;
        wake a held ship in ``finalize_train_batch``."""
        self.version_advanced.set()

    async def stop(self) -> None:
        """Bounded best-effort teardown of all components. Has a global
        timeout so a wedged peer can't keep the process alive forever —
        training artifacts are already persisted before this is reached."""

        async def teardown() -> None:
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
            if getattr(self, "policy_inference", None) is not None:
                await self.policy_inference.stop()
            if self.train_envs is not None:
                for env in self.train_envs:
                    for pool in (*env.sampler.connected_pools, *env.algorithm.connected_pools):
                        await pool.stop()
                self.train_envs.shutdown()
            if self.eval_envs is not None:
                self.eval_envs.shutdown()
            if self.usage_reporter is not None:
                self.usage_reporter.close()

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
