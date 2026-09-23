"""EvalRunner: the eval engine shared by ``eval`` (one epoch against the served weights)
and ``online-eval`` (an epoch per weight broadcast).

Scheduling reuses the orchestrator's components unchanged: an eval-only ``Dispatcher``
admits episodes under the adaptive ``ConcurrencyController``, fed by the
``InferenceMetricsCollector``'s ``/metrics`` polls, and the ``Evaluator`` reports each
epoch. Eval episodes are version-pinned measurements and are never cancelled on load -
a controller cut only blocks admission until the pool drains.

Env servers belong to the launcher (``eval``, ``sft``), like the orchestrator's belong to
``rl``: a source without an explicit ``serve.address`` is found through the address file
its server publishes; one with an address is reached directly."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import verifiers.v1 as vf

from prime_rl.configs.eval import EvalConfig, SFTOnlineEvalConfig
from prime_rl.configs.orchestrator import EvaluatorConfig
from prime_rl.orchestrator import live
from prime_rl.orchestrator.clients import AdminPlane, InferenceClient
from prime_rl.orchestrator.concurrency import ConcurrencyController
from prime_rl.orchestrator.dispatcher import Dispatcher, DispatcherMode
from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.evaluator import Evaluator
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.patches import (
    monkey_patch_chat_completion_logprobs,
    monkey_patch_oai_iterable_types,
)
from prime_rl.orchestrator.periodic_logger import PeriodicLogger
from prime_rl.orchestrator.utils import intercept_vf_logging, set_default_executor
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_config_dir

monkey_patch_oai_iterable_types()
monkey_patch_chat_completion_logprobs()

# How often ``run_epoch`` re-checks for a superseding checkpoint while it waits for episodes.
POLL_INTERVAL_S = 2.0


class EvalRunner:
    def __init__(self, config: EvalConfig | SFTOnlineEvalConfig, *, run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir
        intercept_vf_logging(logger="verifiers.v1", level="WARN")
        self.dispatcher_task: asyncio.Task | None = None

        # Assigned in setup(); None-initialized so stop() can tear down a
        # partially completed setup with plain attribute checks.
        self.clients: InferenceClient | None = None
        self.admin_plane: AdminPlane | None = None
        self.dispatcher: Dispatcher | None = None
        self.inference_metrics: InferenceMetricsCollector | None = None
        self.periodic_logger: PeriodicLogger | None = None

    async def setup(self, *, skip_first_step: bool = False, is_resumed: bool = False) -> None:
        config = self.config
        set_default_executor()

        # The launcher-set $PRL_RUN_ID is the run identity; standalone runs mint a local one.
        run_id = os.environ.get("PRL_RUN_ID") or uuid.uuid4().hex
        run_name = os.environ.get("PRL_RUN_NAME")

        get_logger().info(f"Initializing inference pool (base_url={config.client.base_url}, model={config.model})")
        self.clients = InferenceClient(config.client, model_name=config.model)
        self.admin_plane = AdminPlane(config.client)

        get_logger().info("Loading eval environment(s)")
        self.eval_envs = EvalEnvs(config.source, config.env_addresses, get_config_dir(self.run_dir))
        await self.eval_envs.start()
        get_logger().info(f"Eval environment(s) ready ({', '.join(self.eval_envs.names)})")

        get_logger().info("Waiting for inference pool to be ready")
        await self.admin_plane.wait_for_ready(config.model)
        get_logger().info("Inference pool ready")

        online = isinstance(config, SFTOnlineEvalConfig)
        self.eval_source = EvalSource(
            self.eval_envs,
            intervals=config.intervals if online else None,
            skip_first_step=skip_first_step,
            is_resumed=is_resumed,
        )
        self.evaluator = Evaluator(
            EvaluatorConfig(max_steps=config.max_steps if online else None, upload_epochs=True),
            eval_source=self.eval_source,
            eval_envs=self.eval_envs,
        )

        # Pessimistic per-episode token cost for the controller's starting cap,
        # only used when the engine doesn't report its max context length.
        fallback_cost = max((source.sampling.max_completion_tokens or 0) for source in config.source) or 8192
        self.concurrency = ConcurrencyController(config.concurrency, fallback_cost=fallback_cost)
        self.dispatcher = Dispatcher(
            config.dispatcher,
            train_envs=None,
            eval_envs=self.eval_envs,
            train_source=None,
            eval_source=self.eval_source,
            policy_clients=self.clients,
            initial_max_inflight=self.concurrency.max_inflight,
            max_inflight_ceiling=config.concurrency.max_inflight,
            run_id=run_id,
            run_name=run_name,
        )
        self.inference_metrics = InferenceMetricsCollector(config.inference_metrics, self.admin_plane.clients)
        self.periodic_logger = PeriodicLogger(name="Eval", interval=config.log.interval)
        self.wire()

        # Fail fast when adaptivity has no signal: external API endpoints (e.g. Prime
        # Inference) expose no vLLM /metrics, so without a probe hit the cap would
        # silently sit at min_inflight forever. A pinned band (min_inflight =
        # max_inflight) makes the controller inert and is the supported way to run
        # against such endpoints.
        if not await self.inference_metrics.probe():
            concurrency = config.concurrency
            if concurrency.min_inflight != concurrency.max_inflight:
                urls = ", ".join(str(client.base_url) for client in self.admin_plane.clients)
                raise ValueError(
                    f"No engine metrics at {urls} - adaptive concurrency has no load signal. "
                    "The endpoint does not expose vLLM /metrics (e.g. an external inference API); "
                    "pin the concurrency with `-c N` (concurrency.min_inflight = max_inflight = N)."
                )
            get_logger().info(f"No engine metrics - running with concurrency pinned at {concurrency.min_inflight}")
        await self.inference_metrics.start()

    def wire(self) -> None:
        dispatcher, evaluator = self.dispatcher, self.evaluator
        assert dispatcher is not None and self.inference_metrics is not None and self.periodic_logger is not None
        # No ``on_overload``: eval episodes are measurements and are never cancelled —
        # a cut only blocks admission until the pool drains.
        self.concurrency.bind(set_limit=dispatcher.set_limit, get_inflight=lambda: dispatcher.current_inflight)
        self.inference_metrics.bind(on_load=self.concurrency.observe)
        dispatcher.bind(on_eval=evaluator.ingest, on_episode_complete=self.concurrency.record_episode)
        evaluator.bind(prefer_eval=lambda reason: dispatcher.switch_mode(DispatcherMode.PREFER_EVAL, reason=reason))
        self.periodic_logger.register(status=evaluator.status)
        self.periodic_logger.register(status=self.status, gauges=dispatcher.gauges)
        self.periodic_logger.register(gauges=self.concurrency.gauges)

    def status(self) -> str:
        assert self.dispatcher is not None
        stages = live.stage_counts(list(self.dispatcher.inflight.values()))
        return (
            f"{self.dispatcher.inflight_eval_count} inflight episodes "
            f"(cap {self.dispatcher.max_inflight}, signal {self.concurrency.signal})"
            + (f" - {stages}" if stages else "")
        )

    async def start(self) -> None:
        assert self.dispatcher is not None and self.periodic_logger is not None
        self.dispatcher_task = asyncio.create_task(self.dispatcher.start(), name="dispatcher")
        await self.periodic_logger.start()

    async def run_epoch(
        self,
        step: int,
        *,
        force: bool = False,
        restored: Sequence[vf.Episode] = (),
        superseding_step: Callable[[], int | None] | None = None,
    ) -> list[str]:
        """Fire the envs due at ``step``, run their epochs to completion and return the
        fired names. ``restored`` episodes of this epoch landed before a resume and rejoin
        it first; when ``superseding_step`` returns a newer checkpoint, the unfinished
        episodes of this epoch are cancelled so the caller can move on to it."""
        assert self.dispatcher is not None
        fired = await self.evaluator.trigger(step, force=force)
        # Landed episodes rejoin their epoch whether or not anything is still owed.
        if restored:
            get_logger().info(f"{len(restored)} restored episodes rejoin the epoch")
        for episode in restored:
            await self.evaluator.restore(episode)
        if not fired:
            return []

        cancellation_task: asyncio.Task[int] | None = None
        newer_step: int | None = None
        while self.evaluator.is_pending(step, fired):
            self._raise_if_dispatcher_stopped()
            if (
                cancellation_task is None
                and superseding_step is not None
                and (newer_step := superseding_step()) is not None
            ):
                get_logger().warning(
                    f"Checkpoint {newer_step} is ready - cancelling unfinished eval episodes for step {step}"
                )
                cancellation_task = asyncio.create_task(
                    self.dispatcher.cancel_eval_step(step), name=f"cancel-eval-step-{step}"
                )
            self.evaluator.changed.clear()
            try:
                await asyncio.wait_for(self.evaluator.changed.wait(), timeout=POLL_INTERVAL_S)
            except asyncio.TimeoutError:
                if cancellation_task is not None and cancellation_task.done():
                    cancellation_task.result()

        if cancellation_task is not None:
            cancelled = await cancellation_task
            get_logger().warning(
                f"Cancelled {cancelled} unfinished eval episodes for step {step}; advancing to checkpoint {newer_step}"
            )
        return fired

    def _raise_if_dispatcher_stopped(self) -> None:
        task = self.dispatcher_task
        if task is not None and task.done():
            if task.cancelled():
                raise RuntimeError("dispatcher stopped unexpectedly")
            if task.exception() is not None:
                raise task.exception()
            raise RuntimeError("dispatcher exited unexpectedly")

    async def drain(self) -> None:
        """Stop the background loggers so nothing logs after the monitors finalize."""
        assert self.periodic_logger is not None and self.inference_metrics is not None
        await self.periodic_logger.stop()
        await self.inference_metrics.stop()

    async def stop(self) -> None:
        """Best-effort teardown; tolerates a partially completed ``setup()``."""
        if self.periodic_logger is not None:
            await self.periodic_logger.stop()
        if self.inference_metrics is not None:
            await self.inference_metrics.stop()
        if self.dispatcher is not None:
            await self.dispatcher.stop()
        if self.clients is not None:
            await self.clients.aclose()
        if self.admin_plane is not None:
            await self.admin_plane.aclose()
