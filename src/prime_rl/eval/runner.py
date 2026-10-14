"""EvalRunner: the eval engine shared by ``eval`` (one epoch against the served weights)
and ``online-eval`` (an epoch per weight broadcast).

Scheduling reuses the orchestrator pipeline unchanged: an eval-only ``Dispatcher``
admits episodes under the adaptive ``ConcurrencyController``, fed by the
``InferenceMetricsCollector``'s ``/metrics`` polls. Eval episodes are version-pinned
measurements and are never cancelled on load - a controller cut only blocks admission
until the pool drains.

Env servers belong to the launcher (``eval``, ``sft``), like the orchestrator's belong to
``rl``: a source without an explicit ``serve.address`` is found through the address file
its server publishes; one with an address is reached directly.

Every task the dispatcher runs reports back to its env's taskset exactly once, from
``run_epoch`` on the event loop as the dispatcher's result arrives: ``observe(task,
episode)`` with the final episode (after the env's own retries and ``finalize``), or
``on_failed(task, error)`` when the request produced no episode (the env server died,
the request raised). A streaming taskset runs one rollout per task (``group_size`` 1,
``EvalEnv.open_feed``), so its feedback is per task, as the ``Taskset`` hooks promise;
a bounded taskset hears once per rollout. A streaming taskset may also ``cancel(key)``
a task it handed out: the runner polls ``cancelled()`` while the epoch runs, the
dispatcher stops the task's rollout (aborting its env-server run), and reports it
through ``on_cancelled(task)`` instead; one that landed first is observed as usual. An
exception from any hook stops the run. Rollouts that never finish get none:
those cancelled on shutdown or superseded by a newer checkpoint (an online eval). A
relaunched streaming taskset re-yields its unfinished work. Episodes restored on
``--resume`` were reported by the attempt that ran them."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import verifiers.v1 as vf

from prime_rl import monitors
from prime_rl.configs.eval import EvalConfig, SFTOnlineEvalConfig
from prime_rl.eval.control import ControlServer
from prime_rl.orchestrator import live
from prime_rl.orchestrator.annotations import stamp_arrival, stamp_batch
from prime_rl.orchestrator.clients import AdminPlane, InferenceClient
from prime_rl.orchestrator.concurrency import ConcurrencyController
from prime_rl.orchestrator.dispatcher import Dispatcher, DispatcherMode
from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.metrics import dispatch_failure_metrics
from prime_rl.orchestrator.periodic_logger import PeriodicLogger
from prime_rl.orchestrator.types import DispatchFailure, EvalBatch, GroupCancellation, Policy
from prime_rl.orchestrator.utils import (
    episode_env_name,
    eval_work,
    intercept_vf_logging,
    set_default_executor,
)
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_config_dir

# How often ``run_epoch`` re-checks for a superseding checkpoint while it waits for episodes.
POLL_INTERVAL_S = 2.0


class EvalRunner:
    def __init__(self, config: EvalConfig | SFTOnlineEvalConfig, *, run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir
        intercept_vf_logging(logger="verifiers.v1", level="WARN")

        self.eval_triggered_at: dict[tuple[str, int], float] = {}
        self.dispatcher_task: asyncio.Task | None = None
        self.cancel_tasks: set[asyncio.Task] = set()
        """Taskset-requested cancels in progress (``request_cancels``). Run off the
        epoch loop: a dropped group's marker goes onto ``out_q``, which only that loop
        drains."""
        self.reported: set[tuple[str, str]] = set()
        """Eval tasks ``(env, key)`` reported to their taskset since the last cancel
        poll: a cancel asked before the report and polled after it is moot."""

        # Assigned in setup(); None-initialized so stop() can tear down a
        # partially completed setup with plain attribute checks.
        self.clients: InferenceClient | None = None
        self.admin_plane: AdminPlane | None = None
        self.dispatcher: Dispatcher | None = None
        self.inference_metrics: InferenceMetricsCollector | None = None
        self.periodic_logger: PeriodicLogger | None = None
        self.heart: Heartbeat | None = None
        self.control: ControlServer | None = None

    async def setup(self, *, skip_first_step: bool = False, is_resumed: bool = False) -> None:
        config = self.config
        set_default_executor()

        # The heartbeat is beaten only by landed episodes — the first episode is its
        # first beat. No startup ping on purpose: the run-start to first-episode gap
        # (pool boot, env servers, the first episode's full duration) would otherwise
        # sit inside the ping stream as an abnormally long silence and flip a healthy
        # run stale. Until then the heartbeat has simply never been pinged.
        if config.heartbeat is not None:
            self.heart = Heartbeat(config.heartbeat)

        # The launcher-set $PRL_RUN_ID is the run identity; standalone runs mint a local one.
        self.run_id = os.environ.get("PRL_RUN_ID") or uuid.uuid4().hex
        self.run_name = os.environ.get("PRL_RUN_NAME")

        get_logger().info(f"Initializing inference pool (base_url={config.client.base_url}, model={config.model})")
        self.clients = InferenceClient(config.client, model_name=config.model)
        self.admin_plane = AdminPlane(config.client)

        get_logger().info("Loading eval environment(s)")
        # online evals repeat per checkpoint: only a standalone eval streams
        self.eval_envs = EvalEnvs(
            config.source,
            config.env_addresses,
            get_config_dir(self.run_dir),
            streams=not isinstance(config, SFTOnlineEvalConfig),
        )
        await self.eval_envs.start()
        get_logger().info(f"Eval environment(s) ready ({', '.join(self.eval_envs.names)})")

        get_logger().info("Waiting for inference pool to be ready")
        await self.admin_plane.wait_for_ready(config.model)
        get_logger().info("Inference pool ready")

        intervals = config.intervals if isinstance(config, SFTOnlineEvalConfig) else None
        self.eval_source = EvalSource(
            self.eval_envs, intervals=intervals, skip_first_step=skip_first_step, is_resumed=is_resumed
        )
        self.eval_sink = EvalSink(eval_envs=self.eval_envs)
        self.policy = Policy(version=0, model_name=config.model)

        # Pessimistic per-episode token cost for the controller's starting cap,
        # only used when the engine doesn't report its max context length.
        fallback_cost = max((source.sampling.max_completion_tokens or 0) for source in config.source) or 8192
        self.concurrency = ConcurrencyController(config.concurrency, fallback_cost=fallback_cost)
        self.dispatcher = Dispatcher(
            train_envs=None,
            eval_envs=self.eval_envs,
            train_source=None,
            eval_source=self.eval_source,
            policy_clients=self.clients,
            policy=self.policy,
            policy_weights_change=isinstance(config, SFTOnlineEvalConfig),
            progress=None,
            initial_max_inflight=self.concurrency.max_inflight,
            max_inflight_ceiling=config.concurrency.max_inflight,
            tasks_per_minute=config.tasks_per_minute,
            max_off_policy_steps=0,
            run_id=self.run_id,
            run_name=self.run_name,
            on_episode_complete=self.concurrency.record_episode,
            record_eval_tasks=True,
        )
        # No ``on_overload``: eval episodes are measurements and are never
        # cancelled — a cut only blocks admission until the pool drains.
        self.concurrency.bind(
            set_limit=self.dispatcher.set_limit,
            get_inflight=lambda: self.dispatcher.current_inflight,
        )
        # The collector always polls — it feeds the concurrency controller;
        # metrics fan out to every registered monitor.
        self.inference_metrics = InferenceMetricsCollector(
            self.admin_plane.clients,
            on_load=self.concurrency.observe,
        )
        # Fail fast when adaptivity has no signal: external API endpoints
        # (e.g. Prime Inference) expose no vLLM /metrics, so without a probe
        # hit the cap would silently sit at min_inflight forever. A pinned
        # band (min_inflight = max_inflight) makes the controller inert and
        # is the supported way to run against such endpoints.
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

        self.periodic_logger = PeriodicLogger(
            name="Eval",
            collect=self.collect_pipeline_view,
            interval=config.log.interval,
        )

    async def start(self) -> None:
        self.dispatcher_task = asyncio.create_task(self.dispatcher.start(), name="dispatcher")
        await self.periodic_logger.start()
        self.control = ControlServer(self)
        await self.control.start()

    async def run_epoch(
        self,
        fired: list[str],
        step: int,
        *,
        restored: Sequence[vf.Episode] = (),
        superseding_step: Callable[[], int | None] | None = None,
    ) -> None:
        """Run the epoch ``EvalSource.trigger`` queued for ``step`` in the fired envs and
        finalize each env's batch as it completes. ``restored`` episodes of this epoch
        landed before a resume and rejoin it first; when ``superseding_step`` returns a
        newer checkpoint, the unfinished episodes of this epoch are cancelled so the
        caller can move on to it."""
        for env_name in fired:
            if (expected := self.eval_sink.batch_size_for(env_name)) is not None:
                await monitors.log_eval_plan(env_name, step, expected)

        now = time.perf_counter()
        for env_name in fired:
            self.eval_triggered_at[(env_name, step)] = now
        total_rollouts = sum(
            request.rollouts or 0
            for request in self.eval_source.queue
            if request.step == step and request.env_name in fired
        )
        restored_part = f", {len(restored)} restored" if restored else ""
        get_logger().info(
            f"Starting evals in {', '.join(fired)} at step {step} ({total_rollouts} total rollouts{restored_part})"
        )
        self.dispatcher.switch_mode(DispatcherMode.PREFER_EVAL, reason=f"eval was triggered at step {step}")

        pending = {env_name for env_name in fired if self.eval_sink.batch_size_for(env_name) != 0}
        for episode in restored:
            await self.land(episode, pending)
        cancellation_task: asyncio.Task[int] | None = None
        newer_step: int | None = None

        while pending:
            self.request_cancels()
            for window in self.eval_sink.due_windows():
                await self.finalize_eval_batch(window)
            for env_name in [name for name in pending if self.stream_done(name)]:
                await self.finalize_eval_batch(self.eval_sink.process_batch((env_name, step)))
                pending.discard(env_name)
            if not pending:
                break
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

            try:
                item = await asyncio.wait_for(self.dispatcher.out_q.get(), timeout=POLL_INTERVAL_S)
            except asyncio.TimeoutError:
                if cancellation_task is not None and cancellation_task.done():
                    cancellation_task.result()
                self.raise_if_dispatcher_stopped()
                continue

            if isinstance(item, GroupCancellation):
                if item.reason == "requested":
                    self.report_cancelled(item)
                eval_batch = self.eval_sink.cancel(item)
            elif isinstance(item, DispatchFailure):
                self.report_failed(item)
                eval_batch = self.eval_sink.fail(item)
            else:
                stamp_arrival([item], "eval", eval_work(item).step)
                if self.heart is not None:
                    self.heart.beat()
                self.observe(item)
                await self.land(item, pending)
                continue
            if eval_batch is not None:
                await self.finalize_eval_batch(eval_batch)
                if eval_batch.final:
                    pending.discard(eval_batch.env_name)

        # every requested cancel's marker has landed; let its aborted runs unwind
        await asyncio.gather(*self.cancel_tasks)
        self.cancel_tasks.clear()
        if cancellation_task is not None:
            cancelled = await cancellation_task
            get_logger().warning(
                f"Cancelled {cancelled} unfinished eval episodes for step {step}; advancing to checkpoint {newer_step}"
            )

    def request_cancels(self) -> None:
        """Start the cancels the streaming tasksets asked for since the last poll, and
        raise the error of one that failed. A taskset cancels only tasks not yet
        reported to it, so a key reported since the last poll needs nothing."""
        for task in [task for task in self.cancel_tasks if task.done()]:
            self.cancel_tasks.discard(task)
            task.result()
        for env in self.eval_envs:
            if env.feed is None:
                continue
            for key in env.taskset.cancelled():
                self.cancel_task(env.name, key)
        self.reported.clear()

    def cancel_task(self, env_name: str, key: str) -> None:
        """Stop eval task ``key`` of ``env_name``: its open group is dropped (in-flight
        rollout aborted) off the epoch loop, or, with no group yet, the task is dropped
        unrun when it comes off the source. Either way the unlanded rollout reaches the
        taskset's ``on_cancelled``. An open group wins over a report since the last
        poll: that report was of an earlier run of the same key."""
        taskset = type(self.eval_envs.get(env_name).taskset).__name__
        if self.dispatcher.eval_groups(env_name, key):
            get_logger().info(f"{taskset} cancelled task {key} in {env_name}")
            self.cancel_tasks.add(
                asyncio.create_task(self.dispatcher.cancel_eval_task(env_name, key), name=f"cancel-{key}")
            )
            return
        if (env_name, key) in self.reported:
            return  # it landed (or failed) before the cancel was polled: nothing left to stop
        get_logger().info(f"{taskset} cancelled task {key} in {env_name}")
        # not off the source yet: dropped unrun when it comes off
        self.dispatcher.requested_cancels.add((env_name, key))

    def raise_if_dispatcher_stopped(self) -> None:
        """A dead dispatcher delivers no more results: fail the epoch rather than wait."""
        task = self.dispatcher_task
        if task is None or not task.done():
            return
        if not task.cancelled() and (error := task.exception()) is not None:
            raise error
        raise RuntimeError("The dispatcher stopped while an eval epoch was still running")

    def observe(self, episode: vf.Episode) -> None:
        """Hand a dispatched rollout's final episode to its taskset, before the episode
        is persisted or counted. A ``--resume`` keeps only episodes it finds persisted,
        so every kept episode was observed by the attempt that ran it."""
        task = self.dispatcher.eval_tasks.pop(episode.id)
        env_name = episode_env_name(episode)
        self.mark_reported(env_name, task.key)
        taskset = self.eval_envs.get(env_name).taskset
        try:
            taskset.observe(task, episode)
        except Exception:
            get_logger().error(f"{type(taskset).__name__}.observe failed for task {task.key} in {env_name}")
            raise

    def report_failed(self, failure: DispatchFailure) -> None:
        """Tell the taskset a dispatched rollout produced no episode."""
        assert failure.task is not None and failure.exception is not None  # the dispatcher attaches them
        self.mark_reported(failure.env_name, failure.task_key)
        taskset = self.eval_envs.get(failure.env_name).taskset
        try:
            taskset.on_failed(failure.task, failure.exception)
        except Exception:
            get_logger().error(
                f"{type(taskset).__name__}.on_failed failed for task {failure.task_key} in {failure.env_name}"
            )
            raise

    def mark_reported(self, env_name: str, key: str) -> None:
        self.reported.add((env_name, key))
        self.dispatcher.requested_cancels.discard((env_name, key))

    def report_cancelled(self, cancellation: GroupCancellation) -> None:
        """Tell the taskset each rollout of a task it cancelled stopped unlanded."""
        assert cancellation.task is not None  # a requested cancel carries its task
        self.mark_reported(cancellation.env_name, cancellation.task.key)
        taskset = self.eval_envs.get(cancellation.env_name).taskset
        for _ in range(cancellation.count):
            try:
                taskset.on_cancelled(cancellation.task)
            except Exception:
                get_logger().error(
                    f"{type(taskset).__name__}.on_cancelled failed for task {cancellation.task.key} "
                    f"in {cancellation.env_name}"
                )
                raise

    def stream_done(self, env_name: str) -> bool:
        """Whether a streaming env's taskset ended and every rollout it dispatched landed.
        A cancel in progress may still owe its marker."""
        return (
            self.eval_envs.get(env_name).feed is not None
            and not self.eval_source.streaming(env_name)
            and self.dispatcher.eval_env_settled(env_name)
            and all(task.done() for task in self.cancel_tasks)
        )

    async def land(self, episode: vf.Episode, pending: set[str]) -> None:
        """One episode of the epoch, arrived (and observed) or restored: through the
        monitors and into its env's batch, which is finalized once the epoch is complete."""
        step = eval_work(episode).step
        await monitors.log([episode], step, "eval", "all")
        eval_batch = self.eval_sink.add(episode)
        if eval_batch is not None:
            await self.finalize_eval_batch(eval_batch)
            if eval_batch.final:
                pending.discard(eval_batch.env_name)

    async def finalize_eval_batch(self, batch: EvalBatch) -> None:
        """Persist + log one completed eval epoch through the monitors, mirroring the
        orchestrator: effective episodes plus the ``eval/{env}/...`` metric dict. A
        streaming epoch logs window by window (``EvalSink``), each window's metrics
        over its own attempts, so a crash loses at most the open window."""
        if not batch.episodes and not batch.failures and not batch.cancelled:
            if batch.window:  # a stream's empty last window: its earlier windows still close the epoch
                await monitors.log_eval_epoch(batch.env_name, batch.step, [], final=True)
                self.eval_triggered_at.pop((batch.env_name, batch.step), None)
                get_logger().success(f"Evaluated {batch.env_name} (Step {batch.step}) in {batch.window} windows")
                return
            if batch.final:
                get_logger().warning(
                    f"Eval @ step={batch.step} env={batch.env_name}: no attempts returned, skipping log"
                )
            return

        if batch.episodes.effective:
            await monitors.log(batch.episodes.effective.vf_episodes, batch.step, "eval", "effective")
            await monitors.log_annotations(stamp_batch(batch.episodes.effective.vf_episodes, batch.step))
        await monitors.log_eval_epoch(batch.env_name, batch.step, batch.episodes.vf_episodes, final=batch.final)

        episodes = batch.episodes
        effective = episodes.effective
        metrics: dict[str, float] = {}
        for subset, pool in (("all", episodes), ("effective", effective)):
            metrics |= pool.metrics.to_wandb(prefix=f"eval/{batch.env_name}", subset=subset)
        total_attempts = len(episodes) + len(batch.failures) + batch.cancelled
        metrics |= dispatch_failure_metrics(
            batch.failures,
            prefix=f"eval/{batch.env_name}/all",
            total_attempts=total_attempts,
        )
        if batch.cancelled:
            metrics[f"eval/{batch.env_name}/all/cancelled/count"] = float(batch.cancelled)
            metrics[f"eval/{batch.env_name}/all/cancelled/mean"] = batch.cancelled / total_attempts
        metrics[f"eval/{batch.env_name}/policy_version"] = float(batch.step)
        if batch.window is not None:
            metrics[f"eval/{batch.env_name}/window"] = float(batch.window)
        metrics["step"] = float(batch.step)
        await monitors.log(metrics, step=batch.step)

        eff, full = effective.metrics, episodes.metrics
        key = (batch.env_name, batch.step)
        triggered_at = self.eval_triggered_at.pop(key, None) if batch.final else self.eval_triggered_at.get(key)
        elapsed = (time.perf_counter() - triggered_at) if triggered_at is not None else 0.0
        if not batch.final:
            get_logger().info(
                f"Evaluated {batch.env_name} (Step {batch.step}) window {batch.window} | "
                f"{format_time(elapsed):>7} | Reward {eff.reward.mean():.4f} | "
                f"Error {full.has_error.mean():.1%} | Attempts {total_attempts}"
                + (f" | Cancelled {batch.cancelled}" if batch.cancelled else "")
            )
            return
        if batch.cancelled:
            get_logger().warning(
                f"Partially evaluated {batch.env_name} (Step {batch.step}) | "
                f"{format_time(elapsed):>7} | Reward {eff.reward.mean():.4f} | "
                f"Error {full.has_error.mean():.1%} | "
                f"Completed {len(episodes)}/{total_attempts} | Cancelled {batch.cancelled}/{total_attempts}"
            )
            return
        get_logger().success(
            f"Evaluated {batch.env_name} (Step {batch.step}) | "
            f"{format_time(elapsed):>7} | Reward {eff.reward.mean():.4f} | "
            f"Turns {eff.num_turns.mean():.1f} | Branches {eff.num_branches.mean():.1f} | "
            f"Error {full.has_error.mean():.1%} | Truncation {eff.is_truncated.mean():.1%} | "
            f"Timeout {full.is_timeout.mean():.1%}"
        )

    def collect_pipeline_view(self) -> tuple[str, dict[str, float]]:
        """Pipeline view for the ``PeriodicLogger``: per-env epoch progress plus the
        in-flight pool against the controller's current cap."""
        disp_gauges = self.dispatcher.gauges()
        disp_drain = self.dispatcher.metrics.drained(train_envs=set(), eval_envs={env.name for env in self.eval_envs})

        parts = []
        for env_name, _step, arrived, expected in sorted(self.eval_sink.batch_progress()):
            # a stream's size is known only once it ends
            parts.append(
                f"{env_name} {arrived}/{expected} ({arrived / expected:.1%})" if expected else f"{env_name} {arrived}"
            )
        progress_part = " | ".join(parts) if parts else "Idle"

        stages = live.stage_counts(list(self.dispatcher.inflight.values()))
        body = (
            f"{progress_part}; {self.dispatcher.inflight_eval_count} inflight episodes "
            f"(cap {self.dispatcher.max_inflight}, signal {self.concurrency.signal})"
            + (f" - {stages}" if stages else "")
        )
        payload = {**disp_gauges, **disp_drain, **self.concurrency.gauges()}
        return body, payload

    async def drain(self) -> None:
        """Stop the background loggers so nothing logs after the monitors finalize."""
        await self.periodic_logger.stop()
        await self.inference_metrics.stop()

    async def stop(self) -> None:
        """Best-effort teardown; tolerates a partially completed ``setup()``."""
        if self.control is not None:
            await self.control.stop()
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
