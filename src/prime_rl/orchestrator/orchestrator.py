"""Async-pipelined RL orchestrator.

``Orchestrator`` owns the shared state (policy, progress, ckpt, monitor)
and drives the pipeline. Components are single-purpose:

- ``RolloutDispatcher`` schedules rollouts; emits ``TrainRollout`` /
  ``EvalRollout`` on its queue.
- ``TrainSink`` ingests train rollouts (tokenize → advantages → filters)
  and returns a ``TrainBatch`` when the threshold is met.
- ``EvalSink`` ingests eval rollouts and returns an ``EvalBatch`` (with
  per-env metrics) on epoch completion.
- ``MetricsBuilder`` builds the per-step train W&B dict.
- ``WeightWatcher`` advances ``Policy`` and notifies observers.
- ``PeriodicLogger`` polls the components on a shared interval for the
  ``_timestamp``-axis pipeline log.

Components don't reference the orchestrator. The orchestrator wires them
in ``setup()`` and drives them from ``main_loop()``.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import time
from typing import TYPE_CHECKING

import tomli_w

if TYPE_CHECKING:
    from renderers.base import Renderer
    from transformers.tokenization_utils import PreTrainedTokenizer

    from prime_rl.orchestrator.ckpt import CheckpointManager
    from prime_rl.transport.base import TrainingBatchSender
    from prime_rl.utils.client import InferencePool
    from prime_rl.utils.monitor.base import Monitor
from verifiers.clients.renderer_client import get_bridge_metrics, reset_bridge_metrics
from verifiers.utils.async_utils import EventLoopLagMonitor, EventLoopLagStats

import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before transitive imports
from prime_rl.configs.orchestrator import EMAPerMemberAdvantageConfig, OrchestratorConfig
from prime_rl.orchestrator.ckpt import setup_ckpt_manager
from prime_rl.orchestrator.dispatcher import DispatcherMetrics, DispatcherMode, RolloutDispatcher
from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.filters import setup_filters
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.member_generation import uncovered_trainable_members, validate_member_references
from prime_rl.orchestrator.metrics import MetricsBuilder
from prime_rl.orchestrator.multi_agent_advantage import RAEState, validate_advantage_mode
from prime_rl.orchestrator.patches import (
    monkey_patch_chat_completion_logprobs,
    monkey_patch_oai_iterable_types,
)
from prime_rl.orchestrator.periodic_logger import PeriodicLogger
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import (
    EvalBatch,
    EvalRollout,
    FinishedRollout,
    Policy,
    Progress,
    TrainBatch,
    TrainRollout,
    rollouts_for_logging,
)
from prime_rl.orchestrator.utils import (
    append_rollouts,
    compute_teacher_logprobs,
    get_weight_dir,
    intercept_vf_logging,
    save_rollouts,
    set_default_executor,
    setup_student_inference_pool,
)
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


# Wall-clock budget for post-training cleanup; force-exit if graceful
# shutdown wedges (env-server ZMQ recv, vLLM admin aclose, etc)
SHUTDOWN_TIMEOUT_S = 300

# Abort after this many consecutive train batches drop all rollouts to
# post-batch filters — usually a misconfigured filter or homogeneous-reward
# dataset; fail loudly instead of spinning
MAX_CONSECUTIVE_EMPTY_BATCHES = 10

# Maximum batches the orchestrator may run ahead of the trainer. The
# dispatcher is paused via ``update_dispatch_gate`` once this is exceeded;
# resumed when the watcher advances ``policy.version``.
TARGET_LAG = 1


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
    student_inference: InferencePool
    monitor: Monitor
    sender: TrainingBatchSender
    train_envs: TrainEnvs
    train_source: TrainSource
    train_sink: TrainSink
    dispatcher: RolloutDispatcher
    watcher: WeightWatcher
    metrics: MetricsBuilder
    lag_monitor: EventLoopLagMonitor
    periodic_logger: PeriodicLogger

    # Set by ``setup()`` only when relevant config is present
    renderer: Renderer | None
    mm_token_type_ids_mapping: dict[int, int] | None
    teacher_inference: InferencePool | None
    heart: Heartbeat | None
    usage_reporter: UsageReporter | None
    inference_metrics: InferenceMetricsCollector | None
    eval_envs: EvalEnvs | None
    eval_sink: EvalSink | None
    eval_source: EvalSource | None
    lora_name: str | None
    resume_step: int | None
    lag_task: asyncio.Task | None
    rae_state: RAEState | None

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        setup_logger(config.log.level, json_logging=config.log.json_logging)
        # Silence in-process ``verifiers.*`` library noise but keep
        # ``verifiers.serve`` (env-server lifecycle) through our handler
        logging.getLogger("verifiers").setLevel(logging.CRITICAL + 1)
        intercept_vf_logging(logger="verifiers.serve", level="WARN")
        get_logger().info(f"Starting orchestrator ({config.training_mode})")

        if config.bench:
            get_logger().warning(f"Running in benchmark mode (max_steps={config.max_steps})")

        self.progress = Progress()
        self.ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)
        self.policy = Policy(version=0, model_name="")
        self.stopped = asyncio.Event()
        # True after the final train step ships — pipeline winds down without
        # scheduling new train rollouts
        self.draining = False
        self.logged_drain_linger = False
        # Previous ``TrainBatch`` arrival timestamp; reset every ship so
        # ``step_time`` in the success log is real sink-to-sink cycle time
        self.last_batch_at = None
        # Trigger timestamps so eval success logs can report epoch duration
        self.eval_triggered_at = {}
        self.consecutive_empty_batches = 0
        self.component_tasks = []

        # Optional attributes — ``setup()`` populates them when the relevant
        # config is present
        self.renderer = None
        self.mm_token_type_ids_mapping = None
        self.teacher_inference = None
        self.heart = None
        self.usage_reporter = None
        self.inference_metrics = None
        self.eval_envs = None
        self.eval_sink = None
        self.eval_source = None
        self.lora_name = None
        self.resume_step = None
        self.lag_task = None
        # Per-env ``advantage`` is resolved at config validation; one shared RAE
        # state serves every ``ema_per_member`` env (momentum consistency is
        # enforced by ``validate_ema_advantage_momentum``).
        ema_advantage = next(
            (env.advantage for env in config.train.env if isinstance(env.advantage, EMAPerMemberAdvantageConfig)),
            None,
        )
        self.rae_state = RAEState(momentum=ema_advantage.momentum) if ema_advantage is not None else None

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
            tomli_w.dump(config.model_dump(exclude_none=True, mode="json"), f)

        env_ids_to_install = set(get_env_ids_to_install(config.train.env))
        if config.eval is not None:
            env_ids_to_install.update(get_env_ids_to_install(config.eval.env))
        for env_id in env_ids_to_install:
            install_env(env_id, prerelease=config.env_install_prerelease)

        get_logger().info(f"Initializing tokenizer ({config.tokenizer})")
        self.tokenizer = setup_tokenizer(config.tokenizer)

        # Student inference pool
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

        # Filters apply to train rollouts only
        pre_filters = setup_filters(config.pre_batch_filters, vocab_size=self.tokenizer.vocab_size, kind="pre-batch")
        post_filters = setup_filters(config.post_batch_filters, vocab_size=self.tokenizer.vocab_size, kind="post-batch")

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

        self.validate_multi_agent_setup()

        if config.ckpt is not None and config.ckpt.resume_step is not None and self.ckpt_manager is not None:
            if config.ckpt.resume_step == -1:
                self.resume_step = resolve_latest_ckpt_step(self.ckpt_manager.ckpt_dir)
            else:
                self.resume_step = config.ckpt.resume_step

        # Resume below may bump ``policy.version`` and the LoRA model name
        self.policy.model_name = self.student_inference.model_name

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

        if self.resume_step is not None and self.ckpt_manager is not None:
            self.ckpt_manager.load(self.progress, step=self.resume_step, rae_state=self.rae_state)
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

        # SFT generates rollouts via the teacher (the student is trained on
        # the teacher's outputs); RL / OPD generate via the student
        if config.training_mode == "sft":
            assert self.teacher_inference is not None, "sft mode requires teacher inference"
            rollout_inference = self.teacher_inference
        else:
            rollout_inference = self.student_inference

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
            inference=rollout_inference,
            eval_inference=self.student_inference,
            policy=self.policy,
            max_inflight_rollouts=config.max_inflight_rollouts,
            tasks_per_minute=config.tasks_per_minute,
            max_off_policy_steps=config.max_off_policy_steps,
            training_mode=config.training_mode,
            multi_agent=config.multi_agent,
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
            rae_state=self.rae_state,
            pre_filters=pre_filters,
            post_filters=post_filters,
        )
        self.eval_sink = EvalSink(eval_envs=self.eval_envs) if self.eval_envs is not None else None
        self.watcher = WeightWatcher(
            config,
            policy=self.policy,
            inference=self.student_inference,
            observers=[self.dispatcher, self],
            lora_name=self.lora_name,
            ckpt_step=self.progress.step,
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

    def validate_multi_agent_setup(self) -> None:
        """Fail-fast startup census over the loaded envs (config-time pydantic
        validation cannot see the env protocols): every member id referenced in
        ``multi_agent.fixed`` / ``multi_agent.train_one`` must exist in a loaded
        multi-agent env, and each train env's resolved advantage must match its
        multi-agent-ness."""
        config = self.config
        envs = [*self.train_envs, *(self.eval_envs if self.eval_envs is not None else [])]
        members_by_env = {env.name: env.multi_agent_members() for env in envs if env.is_multi_agent}
        validate_member_references(config.multi_agent, members_by_env)
        if config.multi_agent.train_one is not None:
            for env in self.train_envs:
                if not env.is_multi_agent:
                    continue
                uncovered = uncovered_trainable_members(config.multi_agent, env.multi_agent_members())
                if uncovered:
                    get_logger().warning(
                        f"multi_agent.train_one is set, but env {env.name!r} members {uncovered} are neither "
                        "train_one candidates nor fixed targets — they are trainable by default in EVERY "
                        "rollout. Add them to train_one.members or bind them to a fixed target if unintended."
                    )
        for env in self.train_envs:
            validate_advantage_mode(env.name, is_multi_agent=env.is_multi_agent, advantage=env.config.advantage)

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

        # Default step-0 base-model eval — fires before any train rollouts
        # unless ``eval.skip_first_step=True`` (or this is a resume)
        self.maybe_trigger_eval(self.progress.step)

        # Anchor step-time clock so step 0 measures startup → first batch
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
            if self.ckpt_manager is not None:
                get_logger().info("Writing final checkpoint")
                self.ckpt_manager.save(self.progress, step=self.progress.step, rae_state=self.rae_state)
            await self.stop()
            if clean_exit:
                get_logger().success("Orchestrator finished.")
            else:
                get_logger().warning("Orchestrator cleanup complete (forced).")
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception as e:
                get_logger().debug(f"malloc_trim(0) failed: {e}")

    async def main_loop(self) -> None:
        """Consume ``FinishedRollout``\\ s from the dispatcher and route them
        to the train / eval sink. Both sinks return a finalized batch (or
        ``None``) from ``add()``; we just dispatch on the result."""
        while not self.stopped.is_set():
            if self.draining and self.dispatcher.is_idle:
                if self._trainer_broadcast_pending():
                    # The trainer broadcasts every version up to max_steps - 2 and blocks in
                    # _wait_for_nccl_ready until the watcher arms the receive. With off-policy
                    # slack the dispatcher can ship the final batches before those versions
                    # arrive — exiting now would strand the trainer forever. Stay alive until
                    # the watcher has adopted the last expected version.
                    if not self.logged_drain_linger:
                        self.logged_drain_linger = True
                        assert self.config.max_steps is not None
                        get_logger().info(
                            f"Drained, but trainer broadcasts are still expected "
                            f"(v{self.policy.version} < v{self.config.max_steps - 2}) — waiting before exit"
                        )
                    await asyncio.sleep(0.5)
                    continue
                get_logger().info("Pipeline drained, exiting main loop")
                self.stopped.set()
                break

            try:
                rollout: FinishedRollout = await asyncio.wait_for(self.dispatcher.out_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if isinstance(rollout, EvalRollout):
                assert self.eval_sink is not None  # eval rollouts only emitted when eval is configured
                eval_batch = self.eval_sink.add(rollout)
                if eval_batch is not None:
                    await self.finalize_eval_batch(eval_batch)
                continue

            assert isinstance(rollout, TrainRollout)
            await self.maybe_save_failed_train_rollout(rollout)
            train_batch = await self.train_sink.add(rollout)
            # In drain mode any late-arriving train batch is dropped — we
            # don't want to ship past ``max_steps``
            if train_batch is not None and not self.draining and not self.stopped.is_set():
                await self.finalize_train_batch(train_batch)

    async def maybe_save_failed_train_rollout(self, rollout: TrainRollout) -> None:
        """Persist errored train arrivals before group filtering can drop them."""
        if not self.config.dump_failed_train_rollouts or rollout.error is None:
            return
        step_path = get_step_path(get_rollout_dir(self.config.output_dir), self.progress.step)
        dump_trajectory = (
            self.config.dump_trajectory
            if self.config.dump_failed_train_trajectory is None
            else self.config.dump_failed_train_trajectory
        )
        await asyncio.to_thread(
            append_rollouts,
            [rollout.to_dict()],
            step_path / "train_failed_rollouts.jsonl",
            exclude_keys=None if dump_trajectory else {"trajectory"},
        )

    def _trainer_broadcast_pending(self) -> bool:
        """Whether the trainer still has NCCL broadcasts only the watcher can receive."""
        if self.config.weight_broadcast.type != "nccl":
            return False
        if self.config.max_steps is None:
            return False
        return self.policy.version < self.config.max_steps - 2

    async def finalize_train_batch(self, batch: TrainBatch) -> None:
        """Ship one ``TrainBatch`` out to the trainer and handle the I/O
        side-effects (ckpt, save_rollouts, teacher logprobs, sender.send,
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

        save_ckpt_time = await self.maybe_save_ckpt(step)

        if config.max_steps is not None and step >= config.max_steps:
            await self._enter_drain()
            return

        if batch.metrics.n_trainable == 0:
            self.consecutive_empty_batches += 1
            get_logger().warning(
                f"Step {step}: post-batch filters dropped all {len(batch.rollouts)} rollouts "
                f"(consecutive empty batches: {self.consecutive_empty_batches}/{MAX_CONSECUTIVE_EMPTY_BATCHES})"
            )
            if self.consecutive_empty_batches >= MAX_CONSECUTIVE_EMPTY_BATCHES:
                raise RuntimeError(
                    f"{self.consecutive_empty_batches} consecutive zero-trainable batches — "
                    "check filter config (pre_batch_filters / post_batch_filters) or task difficulty."
                )
            return
        self.consecutive_empty_batches = 0
        if batch.metrics.n_trainable / len(batch.rollouts) <= 0.1:
            get_logger().warning(
                f"Only {batch.metrics.n_trainable}/{len(batch.rollouts)} rollouts in the batch are trainable "
                f"({batch.metrics.n_trainable / len(batch.rollouts):.1%}) — consider reviewing task difficulty / filter config"
            )

        # Materialize at the I/O boundary so prime-rl metadata travels with
        # the raw vf payload on disk + in wandb sample tables
        log_rollouts = rollouts_for_logging(batch)
        rollout_dicts = [r.to_dict() for r in log_rollouts]
        step_path = get_step_path(get_rollout_dir(config.output_dir), step)
        await asyncio.to_thread(
            save_rollouts,
            rollout_dicts,
            step_path / "train_rollouts.jsonl",
            exclude_keys=None if config.dump_trajectory else {"trajectory"},
        )

        teacher_logprobs_time = 0.0  # opd only
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

        await self.sender.send(TrainingBatch(examples=batch.samples, step=step))
        self.update_dispatch_gate()

        # Snapshot + reset the renderer client's process-global TITO bridge
        # counters so each step logs a disjoint window
        bridge_metrics = get_bridge_metrics()
        reset_bridge_metrics()
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
            bridge_metrics=bridge_metrics,
        )
        self.monitor.log(metrics, step=step)
        self.monitor.log_samples(rollout_dicts, step=step)
        self.monitor.log_distributions(
            distributions={
                "rewards": [r.reward for r in batch.rollouts],
                "advantages": [r.advantage for r in batch.rollouts if r.advantage is not None],
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

        num_rollouts = len(batch.rollouts)
        num_unique_examples = len({r.group_id for r in batch.rollouts})
        num_tokens = sum(
            r.raw["token_usage"]["final_input_tokens"] + r.raw["token_usage"]["final_output_tokens"]
            for r in batch.rollouts
        )
        self.progress.total_tokens += num_tokens
        self.progress.total_samples += num_rollouts
        self.progress.total_problems += num_unique_examples

        self.log_train_batch(batch, step=step, step_time=step_time)

        self.train_sink.reset_pre_filter_stats()
        self.progress.step += 1
        self.maybe_trigger_eval(self.progress.step)
        if config.max_steps is not None and self.progress.step >= config.max_steps:
            # Drain as soon as the final batch has shipped. Waiting for one more finalized
            # batch to enter drain deadlocks under NCCL weight broadcast: the dispatch gate
            # needs policy version >= max_steps - 1 to assemble that batch, but the trainer
            # never broadcasts its receiver-less final step over NCCL (only the filesystem
            # transport writes it).
            await self._enter_drain()

    async def _enter_drain(self) -> None:
        if self.draining:
            return
        self.draining = True
        self.dispatcher.disable_train_scheduling()
        n_cancelled = await self.dispatcher.cancel_inflight_train_rollouts()
        get_logger().info(
            f"Draining pipeline (cancelled {n_cancelled} in-flight train rollout(s); any in-flight evals will complete)"
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
        """Per-step ``Step …`` success line. Multi-env runs append an
        indented ``╰─`` line per env. ``Error`` is relative to arrivals at
        the sink (errored rollouts may have been group-dropped before
        reaching ``batch.rollouts``)."""
        n_arrivals_total = sum(batch.metrics.arrivals_by_env.values())
        n_errors_total = sum(batch.metrics.errors_by_env.values())
        n_survivors = len(batch.rollouts)
        n_trainable = batch.metrics.n_trainable
        error_rate = (n_errors_total / n_arrivals_total) if n_arrivals_total else 0.0
        trainable_rate = (n_trainable / n_survivors) if n_survivors else 0.0
        reward_mean = sum(r.reward for r in batch.rollouts) / max(n_survivors, 1)
        max_off_policy = max((r.off_policy_steps for r in batch.rollouts), default=0)
        turns_mean = sum(len(r.raw.get("trajectory") or []) for r in batch.rollouts) / max(n_survivors, 1)
        truncation_rate = sum(1 for r in batch.rollouts if r.is_truncated) / max(n_survivors, 1)

        head = (
            f"Step {step} | {format_time(step_time):>7} | Reward {reward_mean:.4f} | "
            f"Trainable {n_trainable}/{n_survivors} ({trainable_rate:.1%}) | "
            f"Turns {turns_mean:.1f} | Max Off-Policy {max_off_policy} | "
            f"Error {error_rate:.1%} | Truncation {truncation_rate:.1%}"
        )
        if len(self.train_envs) <= 1:
            get_logger().success(head)
            return

        env_names = sorted(set(batch.metrics.arrivals_by_env) | {r.env_name for r in batch.rollouts})
        name_width = max(len(n) for n in env_names) if env_names else 0
        lines = [head]
        for env_name in env_names:
            env_rollouts = [r for r in batch.rollouts if r.env_name == env_name]
            n_env_arrivals = batch.metrics.arrivals_by_env.get(env_name, 0)
            n_env_errors = batch.metrics.errors_by_env.get(env_name, 0)
            ratio = (n_env_arrivals / n_arrivals_total) if n_arrivals_total else 0.0
            env_error_rate = (n_env_errors / n_env_arrivals) if n_env_arrivals else 0.0
            env_reward = (sum(r.reward for r in env_rollouts) / len(env_rollouts)) if env_rollouts else 0.0
            env_max_off_policy = max((r.off_policy_steps for r in env_rollouts), default=0)
            env_turns = (
                sum(len(r.raw.get("trajectory") or []) for r in env_rollouts) / len(env_rollouts)
                if env_rollouts
                else 0.0
            )
            env_truncation = sum(1 for r in env_rollouts if r.is_truncated) / len(env_rollouts) if env_rollouts else 0.0
            lines.append(
                f"╰─ {env_name:<{name_width}} | Ratio {ratio:.1%} | Reward {env_reward:.4f} | "
                f"Turns {env_turns:.1f} | Max Off-Policy {env_max_off_policy} | "
                f"Error {env_error_rate:.1%} | Truncation {env_truncation:.1%}"
            )
        get_logger().success("\n\t\t ".join(lines))

    async def finalize_eval_batch(self, batch: EvalBatch) -> None:
        """Persist + log one completed eval epoch (save_rollouts,
        monitor.log_eval_samples, monitor.log)."""
        if not batch.rollouts:
            get_logger().warning(f"Eval @ step={batch.step} env={batch.env_name}: no surviving rollouts, skipping log")
            return

        rollout_dicts = [r.to_dict() for r in batch.rollouts]
        step_path = get_step_path(get_rollout_dir(self.config.output_dir), batch.step)
        await asyncio.to_thread(
            save_rollouts,
            rollout_dicts,
            step_path / f"eval_rollouts_{batch.env_name}.jsonl",
            exclude_keys=None if self.config.dump_trajectory else {"trajectory"},
        )
        self.monitor.log_eval_samples(rollout_dicts, env_name=batch.env_name, step=batch.step)
        self.monitor.log(batch.metrics.to_wandb_dict(env_name=batch.env_name, step=batch.step), step=batch.step)

        n_total = batch.metrics.n_rollouts
        error_rate = ((batch.metrics.n_cancelled + batch.metrics.n_errored) / n_total) if n_total else 0.0
        max_off_policy = max((r.off_policy_steps for r in batch.rollouts), default=0)
        triggered_at = self.eval_triggered_at.pop((batch.env_name, batch.step), None)
        elapsed = (time.perf_counter() - triggered_at) if triggered_at is not None else 0.0

        get_logger().success(
            f"Evaluated {batch.env_name} (Step {batch.step}) | "
            f"{format_time(elapsed):>7} | Reward {batch.metrics.reward_mean:.4f} | "
            f"Turns {batch.metrics.num_turns_mean:.1f} | Max Off-Policy {max_off_policy} | "
            f"Error {error_rate:.1%} | Truncation {batch.metrics.truncation_rate:.1%}"
        )

    async def maybe_save_ckpt(self, step: int) -> float:
        """Save the checkpoint if we're at an interval boundary. Returns
        elapsed time (0.0 when no save happened)."""
        if self.ckpt_manager is None or self.config.ckpt is None or not self.config.ckpt.interval:
            return 0.0
        if step <= 0:
            return 0.0
        # Skip only the drain-entry step (step == max_steps, which never ships):
        # it would double-save with the final checkpoint in ``start()`` (also at
        # progress.step == max_steps). The last *shipped* step (max_steps - 1) is
        # NOT skipped — the trainer saves there (its is_last_step is max_steps),
        # so the orchestrator must too or resume from that interval ckpt breaks.
        near_end = self.config.max_steps is not None and step >= self.config.max_steps
        if near_end:
            return 0.0
        if step % self.config.ckpt.interval != 0:
            return 0.0
        get_logger().info(f"Saving checkpoint at step {step}")
        t = time.perf_counter()
        await asyncio.to_thread(self.ckpt_manager.save, self.progress, step, rae_state=self.rae_state)
        return time.perf_counter() - t

    def update_dispatch_gate(self) -> None:
        """Pause/resume the dispatcher based on how far the orchestrator's
        next batch would run ahead of ``policy.version``. Called from two
        sites: after shipping a batch (step advances) and from
        ``on_new_version`` (policy advances)."""
        lead = (self.progress.step + 1) - self.policy.version
        gate = self.dispatcher.dispatch_allowed
        was_set = gate.is_set()
        if lead > TARGET_LAG:
            if was_set:
                get_logger().info(
                    "Pausing dispatcher to prevent orchestrator from racing from trainer. Waiting for new policy..."
                )
            gate.clear()
        else:
            if not was_set:
                get_logger().info("Resuming dispatcher")
            gate.set()

    async def on_new_version(self, step: int) -> None:
        """``VersionObserver`` hook: the watcher just advanced ``policy.version``;
        re-evaluate the dispatch gate (may resume if the trainer caught up)."""
        self.update_dispatch_gate()

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
