"""Async-pipelined RL orchestrator.

The orchestrator owns no pipeline logic. ``setup()`` builds every component from
its own config, ``wire()`` binds their hooks, and ``start()`` runs their tasks
until the shipper drains the pipeline. The components:

- ``Dispatcher`` schedules environment runs and delivers each result once.
- ``TrainSink`` scores finished train groups and compiles them into samples.
- ``Queue`` buffers compiled groups, sweeps stale ones, and cuts batches.
- ``Shipper`` ships batches to the trainer, owns the step and the lag gate.
- ``Evaluator`` fires eval epochs and reports them.
- ``ConcurrencyController`` moves the in-flight cap off the engines' load.
- ``InferenceMetricsCollector`` polls the engines and feeds the controller.
- ``WeightWatcher`` applies new policy versions and owns the version.
- ``PeriodicLogger`` logs the pipeline view every ``log.interval`` seconds.

Components never reference each other or the orchestrator: every edge is a hook
bound here, so each one constructs and replays standalone in a test.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid

from verifiers.v1.runtimes import set_base_sandbox_labels

import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before transitive imports
from prime_rl import monitors
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.ckpt import setup_ckpt_manager
from prime_rl.orchestrator.clients import AdminPlane, InferenceClient, setup_admin_plane
from prime_rl.orchestrator.concurrency import ConcurrencyController
from prime_rl.orchestrator.dispatcher import Dispatcher, DispatcherMode
from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.evaluator import Evaluator, EvaluatorConfig
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.packing import BatchPacker
from prime_rl.orchestrator.patches import (
    monkey_patch_chat_completion_logprobs,
    monkey_patch_oai_iterable_types,
)
from prime_rl.orchestrator.periodic_logger import PeriodicLogger
from prime_rl.orchestrator.queue import Queue, QueueConfig
from prime_rl.orchestrator.shipper import Shipper
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.utils import intercept_vf_logging, set_default_executor, trim_process_memory
from prime_rl.orchestrator.watcher import WeightWatcher
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.transports.batch import setup_batch_sender
from prime_rl.transports.batch.base import BatchSender
from prime_rl.transports.weights import setup_weight_receiver
from prime_rl.utils.async_utils import EventLoopLagMonitor, EventLoopLagStats, safe_cancel
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import format_time, get_logger, setup_logger
from prime_rl.utils.pathing import get_broadcast_dir, get_ckpt_dir, get_config_dir
from prime_rl.utils.utils import clean_exit, resolve_latest_ckpt_step

monkey_patch_oai_iterable_types()
monkey_patch_chat_completion_logprobs()


# Wall-clock budget for post-training cleanup; force-exit if graceful
# shutdown wedges (env-server ZMQ recv, vLLM admin aclose, etc)
SHUTDOWN_TIMEOUT_S = 300

# Default wait for the trainer's startup weight broadcast when no ckpt block
# configures ``wait_for_weights_timeout`` (e.g. a from-scratch run). The
# broadcast is always coming, so wait rather than fail immediately.
STARTUP_WEIGHT_WAIT_TIMEOUT_S = 1200


def lag_gauges(monitor: EventLoopLagMonitor) -> dict[str, float]:
    stats = EventLoopLagStats.from_monitor(monitor)
    if stats.n == 0:
        return {}
    return {
        "event_loop_lag/min": stats.min,
        "event_loop_lag/mean": stats.mean,
        "event_loop_lag/median": stats.median,
        "event_loop_lag/p90": stats.p90,
        "event_loop_lag/p99": stats.p99,
        "event_loop_lag/max": stats.max,
        "event_loop_lag/n": float(stats.n),
    }


class Orchestrator:
    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        setup_logger(config.log.level, json_logging=config.log.json_logging)
        # Route the in-process v1 library logging through our handler. The
        # env server runs in a child process, so its logging is separate.
        intercept_vf_logging(logger="verifiers.v1", level="WARN")
        get_logger().info("Starting orchestrator")
        self.component_tasks: list[asyncio.Task] = []
        self.lag_task: asyncio.Task | None = None
        # Assigned by ``setup()``; None so a teardown after a partial setup is plain attribute checks
        self.clients: InferenceClient | None = None
        self.admin_plane: AdminPlane | None = None
        self.sender: BatchSender | None = None
        self.dispatcher: Dispatcher | None = None
        self.watcher: WeightWatcher | None = None
        self.inference_metrics: InferenceMetricsCollector | None = None
        self.periodic_logger: PeriodicLogger | None = None
        self.train_envs: TrainEnvs | None = None

    # ── setup ──────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Install envs, load models and pools, resume from checkpoint, build the
        components and wire them."""
        config = self.config
        set_default_executor()

        get_logger().info(f"Initializing tokenizer ({config.tokenizer})")
        t0 = time.perf_counter()
        self.tokenizer = setup_tokenizer(config.tokenizer)
        get_logger().debug(f"Initialized tokenizer in {format_time(time.perf_counter() - t0)}")

        # The one model prime-rl hosts: the live policy. Frozen model references are
        # external endpoints — each env's Algorithm builds its own pools in ``setup()``.
        get_logger().info(f"Initializing policy inference pool ({config.model})")
        self.clients = InferenceClient(
            config.model.client,
            model_name=config.model.name,
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=config.renderer,
        )
        self.admin_plane = setup_admin_plane(config.model.client, config.model.name)

        await monitors.setup(
            producer="orch",
            wandb=config.monitors.wandb,
            prime=config.monitors.prime,
            file=config.monitors.file,
            output_dir=config.output_dir,
            run_config=config,
            train_env_names=[env.resolved_name for env in config.train.source],
            eval_env_names=[source.resolved_name for source in config.eval.source] if config.eval is not None else [],
        )
        # The launcher-set $PRL_RUN_ID is the run identity; standalone runs mint a local one.
        run_id = os.environ.get("PRL_RUN_ID") or uuid.uuid4().hex
        run_name = os.environ.get("PRL_RUN_NAME")
        if run_name:
            set_base_sandbox_labels([run_name])

        config_dir = get_config_dir(config.output_dir)
        self.train_envs = TrainEnvs(
            config.train.source, config.env_addresses, config_dir, clients=self.clients, renderer_config=config.renderer
        )
        self.eval_envs = (
            EvalEnvs(config.eval.source, config.env_addresses, config_dir) if config.eval is not None else None
        )

        self.resume_step: int | None = None
        if config.resume is not None:
            if config.resume.dir is not None:
                self.resume_step = config.resume.dir_step
            else:
                self.resume_step = config.resume.step
                if self.resume_step is None:
                    self.resume_step = resolve_latest_ckpt_step(get_ckpt_dir(config.output_dir))
        get_logger().info(
            f"Resuming from step {self.resume_step}" if self.resume_step is not None else "Starting from scratch"
        )

        # Transports are local setup — initialize them before the env and inference waits.
        packer = BatchPacker(config)
        start_step = self.resume_step + 1 if self.resume_step is not None else 1
        get_logger().info(f"Initializing micro batch sender ({config.rollout_transport})")
        self.sender = setup_batch_sender(
            config.output_dir, config.num_train_workers, start_step, config.rollout_transport
        )

        # Wait phase: envs, then inference, then the trainer's startup broadcast.
        get_logger().info(f"Loading train environments ({', '.join(self.train_envs.names)})")
        t0 = time.perf_counter()
        await self.train_envs.start()
        get_logger().success(f"Train environments ready in {format_time(time.perf_counter() - t0)}")
        if self.eval_envs is not None:
            get_logger().info(f"Loading eval environments ({', '.join(self.eval_envs.names)})")
            t0 = time.perf_counter()
            await self.eval_envs.start()
            get_logger().success(f"Eval environments ready in {format_time(time.perf_counter() - t0)}")

        train_source = TrainSource(self.train_envs)
        ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)
        self.shipper = Shipper(
            max_steps=config.max_steps,
            packer=packer,
            sender=self.sender,
            ckpt_manager=ckpt_manager,
            ckpt_config=config.ckpt,
            train_source=train_source,
            heart=Heartbeat(config.heartbeat.url) if config.heartbeat is not None else None,
        )
        if self.resume_step is not None:
            resume = config.resume
            resume_path = resume.dir / "orchestrator" if resume is not None and resume.dir is not None else None
            # The checkpoint finished ``resume_step``; the step derives from it (not the
            # loaded counter) so it stays coordinated with the trainer even when
            # ``ckpt.skip_progress`` leaves the counter unrestored.
            loaded = ckpt_manager.load(self.resume_step, path=resume_path)
            self.shipper.resume(self.resume_step, loaded[0] if loaded else None)
            if loaded is not None:
                train_source.load_state_dict(loaded[1])
                get_logger().info(f"Resumed curriculum state for {', '.join(loaded[1]['envs'])}")

        get_logger().info("Waiting for policy inference pool to be ready")
        t0 = time.perf_counter()
        await self.admin_plane.wait_for_ready(config.model.name)
        get_logger().success(f"Policy inference pool ready after {format_time(time.perf_counter() - t0)}")
        # Build + ready pools for each env's frozen generation source and the
        # algorithm's frozen reference model
        await asyncio.gather(
            *(env.generation_source.setup() for env in self.train_envs),
            *(env.algorithm.setup() for env in self.train_envs),
        )

        get_logger().info(f"Initializing weight broadcast ({config.weight_broadcast})")
        t0 = time.perf_counter()
        # A LoRA run's adapter is registered under the base model name: the single
        # adapter shadows it, so requests keep addressing one stable name.
        receiver = setup_weight_receiver(
            get_broadcast_dir(config.output_dir),
            config.weight_broadcast,
            admin_plane=self.admin_plane,
            model_name=config.model.name,
        )
        await receiver.initialize()
        get_logger().debug(f"Initialized weight broadcast in {format_time(time.perf_counter() - t0)}")
        self.watcher = WeightWatcher(receiver)

        self.evaluator: Evaluator | None = None
        eval_source: EvalSource | None = None
        if config.eval is not None and self.eval_envs is not None:
            eval_source = EvalSource(
                self.eval_envs,
                intervals=config.eval.intervals,
                skip_first_step=config.eval.skip_first_step,
                is_resumed=self.resume_step is not None,
            )
            self.evaluator = Evaluator(
                EvaluatorConfig(
                    max_steps=config.max_steps,
                    retrigger_on_resume=config.eval.retrigger_on_resume,
                    resume_step=self.resume_step,
                ),
                eval_source=eval_source,
                eval_envs=self.eval_envs,
            )

        self.concurrency = ConcurrencyController(config.concurrency, fallback_cost=config.seq_len)
        self.dispatcher = Dispatcher(
            dispatch_per_minute=config.dispatch_per_minute,
            train_envs=self.train_envs,
            eval_envs=self.eval_envs,
            train_source=train_source,
            eval_source=eval_source,
            policy_clients=self.clients,
            initial_max_inflight=self.concurrency.max_inflight,
            max_inflight_ceiling=config.concurrency.max_inflight,
            max_off_policy_steps=config.max_off_policy_steps,
            run_id=run_id,
            run_name=run_name,
        )
        self.inference_metrics = InferenceMetricsCollector(
            self.admin_plane.clients, roles=config.inference_metrics_roles, log=config.collect_inference_metrics
        )
        self.sink = TrainSink(self.train_envs)
        self.queue = Queue(
            QueueConfig(
                batch_size=config.batch_size,
                token_batch_size=config.token_batch_size,
                max_off_policy_steps=config.max_off_policy_steps,
                constant_trainer_batch_size=config.constant_trainer_batch_size,
                seq_len=config.seq_len,
            )
        )
        self.lag_monitor = EventLoopLagMonitor()
        self.periodic_logger = PeriodicLogger(name="Pipeline", interval=config.log.interval)
        self.wire(train_source)

        # The collector always polls — it feeds the concurrency controller. One awaited
        # scrape so the controller derives (and logs) its initial limit before the loop
        # starts; failures are tolerated.
        await self.inference_metrics.start()
        await self.inference_metrics.probe()

        # Sync inference to the incoming policy before the first step, rendezvousing with
        # the trainer's startup broadcast (v{resume_step} on resume, v0 from scratch).
        sync_version = self.resume_step if self.resume_step is not None else 0
        wait_timeout = (config.ckpt.wait_for_weights_timeout if config.ckpt else None) or STARTUP_WEIGHT_WAIT_TIMEOUT_S
        get_logger().info(f"Syncing inference to the trainer's startup broadcast (v{sync_version})")
        t0 = time.perf_counter()
        await self.watcher.sync_startup(sync_version, timeout=wait_timeout)
        get_logger().debug(f"Synced inference to policy v{sync_version} in {format_time(time.perf_counter() - t0)}")

    def wire(self, train_source: TrainSource) -> None:
        """Every edge between components, in one place."""
        dispatcher, watcher, shipper = self.dispatcher, self.watcher, self.shipper
        assert dispatcher is not None and watcher is not None and self.inference_metrics is not None
        assert self.periodic_logger is not None

        self.concurrency.bind(
            set_limit=dispatcher.set_limit,
            get_inflight=lambda: dispatcher.current_inflight,
            on_overload=dispatcher.cancel_inflight,
        )
        self.inference_metrics.bind(on_load=self.concurrency.observe)
        dispatcher.bind(
            step=shipper.step,
            version=lambda: watcher.version,
            on_train=self.sink.ingest,
            on_eval=self.evaluator.ingest if self.evaluator is not None else None,
            on_episode_complete=self.concurrency.record_episode,
        )
        self.sink.bind(on_group=self.queue.put, admit=train_source.on_result)
        self.queue.bind(step=shipper.step, on_batch=shipper.on_batch)
        shipper.bind(
            version=lambda: watcher.version,
            wait_for_version=watcher.wait_for,
            gate=dispatcher.gate,
            on_drain=dispatcher.drain_train,
        )
        on_new_version = [dispatcher.on_new_version]
        if self.evaluator is not None:
            self.evaluator.bind(
                prefer_eval=lambda reason: dispatcher.switch_mode(DispatcherMode.PREFER_EVAL, reason=reason)
            )
            on_new_version.append(self.evaluator.trigger)
        on_new_version.append(shipper.on_version)
        watcher.bind(on_version_pending=[dispatcher.on_version_pending], on_new_version=on_new_version)

        logger = self.periodic_logger
        logger.register(status=self.queue.status, gauges=self.queue.gauges)
        logger.register(status=self.sink.status)
        if self.evaluator is not None:
            logger.register(status=self.evaluator.status)
        logger.register(status=dispatcher.status, gauges=dispatcher.gauges)
        logger.register(gauges=watcher.gauges)
        logger.register(gauges=self.concurrency.gauges)
        logger.register(gauges=lambda: lag_gauges(self.lag_monitor))

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Set up, run the component tasks until the pipeline drains, then clean up."""
        await self.setup()
        config = self.config
        assert self.dispatcher is not None and self.watcher is not None and self.periodic_logger is not None
        get_logger().info(f"Starting orchestrator loop (max_steps={config.max_steps or 'infinite'})")
        start_time = time.perf_counter()

        self.lag_task = asyncio.create_task(self.lag_monitor.run(), name="event_loop_lag")
        await self.periodic_logger.start()
        self.component_tasks = [
            asyncio.create_task(self.dispatcher.start(), name="dispatcher"),
            asyncio.create_task(self.watcher.start(), name="watcher"),
        ]
        self.shipper.start_clock()

        # ``clean_exit`` stays False if the run raises (signal-driven CancelledError,
        # KeyboardInterrupt, or a real error), so teardown logs a forced-cleanup warning.
        clean_exit = False
        try:
            await self.run()
            # Stay alive for the trainer's last broadcast: every broadcast is a blocking
            # rendezvous, and tearing down the watcher first would strand the trainer.
            if config.max_steps is not None:
                await self.watcher.wait_for(config.max_steps, reason="before shutdown")
            clean_exit = True
        finally:
            elapsed = format_time(time.perf_counter() - start_time)
            # Saved before finalize, which tells the launcher the run is done.
            self.shipper.save_final()
            if clean_exit:
                get_logger().success(f"Orchestrator step loop done in {elapsed}")
                # The background loggers write through the monitors, so they must stop
                # before finalize marks the run finished
                if self.inference_metrics is not None:
                    await self.inference_metrics.stop()
                await self.periodic_logger.stop()
                # Finalize only on a clean exit — a crashed run must not be marked
                # completed; the platform run's atexit hook marks it failed instead.
                await monitors.finalize()
            else:
                get_logger().warning(f"Orchestrator interrupted after {elapsed} — forcing cleanup (not a clean exit)")
            await self.stop()
            if clean_exit:
                get_logger().success("Orchestrator finished")
            else:
                get_logger().warning("Orchestrator cleanup complete (forced)")
            trim_process_memory()

    async def run(self) -> None:
        """Wait until the shipper starts draining and the dispatcher runs dry. A
        component task that dies first ends the run with its error."""
        assert self.dispatcher is not None
        draining = asyncio.create_task(self.shipper.draining.wait(), name="draining")
        try:
            while True:
                done, _ = await asyncio.wait({draining, *self.component_tasks}, return_when=asyncio.FIRST_COMPLETED)
                self._raise_if_component_stopped()
                if draining in done:
                    break
            while not self.dispatcher.is_idle:
                self._raise_if_component_stopped()
                await asyncio.sleep(0.5)
            get_logger().info("Pipeline drained")
        finally:
            await safe_cancel(draining)

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

    async def stop(self) -> None:
        """Bounded best-effort teardown of all components. Has a global timeout so a
        wedged peer can't keep the process alive forever — training artifacts are
        already persisted before this is reached."""

        async def teardown() -> None:
            if self.sender is not None:
                get_logger().debug("Closing micro batch sender")
                self.sender.close()
            if self.dispatcher is not None:
                get_logger().debug("Stopping dispatcher")
                await self.dispatcher.stop()
            if self.watcher is not None:
                get_logger().debug("Stopping weight watcher")
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
                get_logger().debug("Stopping inference metrics collector")
                await self.inference_metrics.stop()
            if self.clients is not None:
                await self.clients.aclose()
            if self.admin_plane is not None:
                await self.admin_plane.aclose()
            if self.train_envs is not None:
                get_logger().debug("Stopping generation source and algorithm clients")
                for env in self.train_envs:
                    for clients in (env.generation_source.connected, env.algorithm.connected):
                        if clients is not None:
                            await clients.aclose()

        get_logger().info("Stopping orchestrator components")
        t0 = time.perf_counter()
        task = asyncio.create_task(teardown())
        _, pending = await asyncio.wait({task}, timeout=SHUTDOWN_TIMEOUT_S)
        if pending:
            get_logger().warning(
                f"Orchestrator shutdown did not complete within {SHUTDOWN_TIMEOUT_S}s; "
                "forcing process exit. Training artifacts are already persisted."
            )
            os._exit(0)
        await task
        get_logger().debug(f"Stopped orchestrator components in {format_time(time.perf_counter() - t0)}")


@clean_exit
async def run_orchestrator(config: OrchestratorConfig) -> None:
    """Top-level entrypoint. Wrapped in ``@clean_exit`` so wandb is flushed
    on exit (success or crash); keeps that out of the class."""
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
