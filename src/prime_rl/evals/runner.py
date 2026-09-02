"""EvalRunner: the eval engine shared by ``evals`` (one epoch against the served weights)
and ``online-evals`` (an epoch per weight broadcast).

Scheduling reuses the orchestrator pipeline unchanged: an eval-only ``Dispatcher``
admits episodes under the adaptive ``ConcurrencyController``, fed by the
``InferenceMetricsCollector``'s ``/metrics`` polls. Eval episodes are version-pinned
measurements and are never cancelled on load - a controller cut only blocks admission
until the pool drains.

Env servers: sources without an explicit ``serve.address`` get an env server spawned
by this process at their derived address; sources with one are externally managed
(e.g. spawned by the ``sft`` launcher, which stamps the derived addresses into the
online config)."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from subprocess import Popen

from prime_rl import monitors
from prime_rl.configs.evals import EvalsConfig, OnlineEvalsConfig
from prime_rl.orchestrator.annotations import stamp_arrival, stamp_batch
from prime_rl.orchestrator.clients import AdminClients, InferenceClient
from prime_rl.orchestrator.concurrency import ConcurrencyController
from prime_rl.orchestrator.dispatcher import Dispatcher, DispatcherMetrics, DispatcherMode
from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.metrics import dispatch_failure_metrics
from prime_rl.orchestrator.patches import (
    monkey_patch_chat_completion_logprobs,
    monkey_patch_oai_iterable_types,
)
from prime_rl.orchestrator.periodic_logger import PeriodicLogger
from prime_rl.orchestrator.types import DispatchFailure, EvalBatch, GroupCancellation, Policy
from prime_rl.orchestrator.utils import (
    episode_group_id,
    eval_work,
    intercept_vf_logging,
    set_default_executor,
)
from prime_rl.utils.config import dump_resolved_config
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_config_dir
from prime_rl.utils.process import DEFAULT_COMMON_ENV_VARS, cleanup_processes

monkey_patch_oai_iterable_types()
monkey_patch_chat_completion_logprobs()

# How often ``run_epoch`` re-checks for a superseding checkpoint while it waits for episodes.
POLL_INTERVAL_S = 2.0


class EvalRunner:
    def __init__(self, config: EvalsConfig | OnlineEvalsConfig, *, run_dir: Path, log_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir
        self.log_dir = log_dir
        intercept_vf_logging(logger="verifiers.v1", level="WARN")

        self.eval_triggered_at: dict[tuple[str, int], float] = {}
        self.env_server_procs: list[Popen] = []
        self.dispatcher_task: asyncio.Task | None = None

        # Assigned in setup(); None-initialized so stop() can tear down a
        # partially completed setup with plain attribute checks.
        self.clients: InferenceClient | None = None
        self.admin_clients: AdminClients | None = None
        self.dispatcher: Dispatcher | None = None
        self.inference_metrics: InferenceMetricsCollector | None = None
        self.periodic_logger: PeriodicLogger | None = None

    async def setup(self, *, skip_first_step: bool = False, is_resumed: bool = False) -> None:
        config = self.config
        set_default_executor()

        # The launcher-set $PRL_RUN_ID is the run identity; standalone runs mint a local one.
        self.run_id = os.environ.get("PRL_RUN_ID") or uuid.uuid4().hex
        self.run_name = os.environ.get("PRL_RUN_NAME")
        wandb_enabled = monitors.get(monitors.WandbMonitor) is not None

        get_logger().info(f"Initializing inference pool (base_url={config.client.base_url}, model={config.model})")
        self.clients = InferenceClient(config.client, model_name=config.model)
        self.admin_clients = AdminClients(config.client)

        self.spawn_env_servers()

        get_logger().info("Loading eval environment(s)")
        self.eval_envs = EvalEnvs(config.source, config.env_addresses)
        await self.eval_envs.start()
        get_logger().success(f"Eval environment(s) ready ({', '.join(self.eval_envs.names)})")

        get_logger().info("Waiting for inference pool to be ready")
        await self.admin_clients.wait_for_ready(config.model)
        get_logger().success("Inference pool ready")

        self.eval_source = EvalSource(self.eval_envs, skip_first_step=skip_first_step, is_resumed=is_resumed)
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
            progress=None,
            initial_max_inflight=self.concurrency.max_inflight,
            max_inflight_ceiling=config.concurrency.max_inflight,
            tasks_per_minute=None,
            max_off_policy_steps=0,
            run_id=self.run_id,
            run_name=self.run_name,
            on_episode_complete=self.concurrency.record_episode,
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
            self.admin_clients.clients,
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
                urls = ", ".join(str(client.base_url) for client in self.admin_clients.clients)
                raise ValueError(
                    f"No engine metrics at {urls} - adaptive concurrency has no load signal. "
                    "The endpoint does not expose vLLM /metrics (e.g. an external inference API); "
                    "pin the concurrency with `-c N` (concurrency.min_inflight = max_inflight = N)."
                )
            get_logger().warning(f"No engine metrics - running with concurrency pinned at {concurrency.min_inflight}")
        await self.inference_metrics.start()

        self.periodic_logger = PeriodicLogger(
            name="Evals",
            collect=self.collect_pipeline_view,
            metric_keys=[
                *list(self.dispatcher.gauges().keys()),
                *list(self.concurrency.gauges().keys()),
                *DispatcherMetrics.drain_keys(train_envs=set(), eval_envs={env.name for env in self.eval_envs}),
            ],
            interval=config.log.interval,
            wandb_enabled=wandb_enabled,
        )

    def spawn_env_servers(self) -> None:
        """Spawn one env server per source without an explicit ``serve.address``,
        at the source's derived address."""
        config = self.config
        addresses = config.env_addresses
        config_dir = get_config_dir(self.run_dir) / "envs" / "eval"
        log_dir = self.log_dir / "envs" / "eval"
        for source in config.source:
            if source.serve.address is not None:
                continue
            name = source.resolved_name
            address = addresses[("eval", name)]
            source_dict = dump_resolved_config(source)
            server_config = {
                "env": source_dict["env"],
                "serve": {**(source_dict.get("serve") or {}), "address": address},
                "log": {"level": config.log.vf_level, "json_logging": config.log.json_logging},
            }
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / f"{name}.json"
            config_path.write_text(json.dumps(server_config, indent=2))
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{name}.log"
            get_logger().info(f"Starting env server {name} at {address} (logs: {log_path})")
            with open(log_path, "w") as log_file:
                process = Popen(
                    ["env-server", "@", config_path.as_posix()],
                    env={**os.environ, **DEFAULT_COMMON_ENV_VARS},
                    stdout=log_file,
                    stderr=log_file,
                )
            self.env_server_procs.append(process)

    async def start(self) -> None:
        self.dispatcher_task = asyncio.create_task(self.dispatcher.start(), name="dispatcher")
        await self.periodic_logger.start()

    async def run_epoch(
        self,
        fired: list[str],
        step: int,
        *,
        on_group_completed: Callable[[int], None] | None = None,
        superseding_step: Callable[[], int | None] | None = None,
    ) -> None:
        """Run the epoch ``EvalSource.trigger`` queued for ``step`` in the fired envs and
        finalize each env's batch as it completes. ``on_group_completed`` receives the
        source index of every finished group (offline cursor tracking); when
        ``superseding_step`` returns a newer checkpoint, the unfinished episodes of this
        epoch are cancelled so the caller can move on to it."""
        for env_name in fired:
            task_count = self.eval_source.triggered_task_count(env_name, step)
            self.eval_sink.set_batch_size(env_name, step, task_count * self.eval_sink.group_size_for(env_name))

        now = time.perf_counter()
        for env_name in fired:
            self.eval_triggered_at[(env_name, step)] = now
        total_rollouts = sum(
            self.eval_envs.get(request.env_name).config.group_size
            for request in self.eval_source.queue
            if request.step == step and request.env_name in fired
        )
        get_logger().info(f"Starting evals in {', '.join(fired)} at step {step} ({total_rollouts} total rollouts)")
        self.dispatcher.switch_mode(DispatcherMode.PREFER_EVAL, reason=f"eval was triggered at step {step}")

        pending = {env_name for env_name in fired if self.eval_sink.batch_size_for(env_name, step) > 0}
        cancellation_task: asyncio.Task[int] | None = None
        newer_step: int | None = None

        while pending:
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
                if superseding_step is not None:
                    item = await asyncio.wait_for(self.dispatcher.out_q.get(), timeout=POLL_INTERVAL_S)
                else:
                    item = await self.dispatcher.out_q.get()
            except asyncio.TimeoutError:
                if cancellation_task is not None and cancellation_task.done():
                    cancellation_task.result()
                continue

            if isinstance(item, GroupCancellation):
                eval_batch = self.eval_sink.cancel(item)
                group_completed = False
                group_id = item.group_id
            elif isinstance(item, DispatchFailure):
                eval_batch = self.eval_sink.fail(item)
                group_completed = not self.eval_sink.has_pending_group(item.group_id)
                group_id = item.group_id
            else:
                item_step = eval_work(item).step
                stamp_arrival([item], "eval", item_step)
                await monitors.log([item], item_step, "eval", "all")
                eval_batch = self.eval_sink.add(item)
                group_id = episode_group_id(item)
                group_completed = not self.eval_sink.has_pending_group(group_id)
            if eval_batch is not None:
                await self.finalize_eval_batch(eval_batch)
                pending.discard(eval_batch.env_name)
            if group_completed:
                source_index = self.dispatcher.pop_source_index(group_id)
                if on_group_completed is None:
                    continue
                if source_index is None:
                    raise RuntimeError(f"Eval group {group_id} is missing its source cursor")
                on_group_completed(source_index)

        if cancellation_task is not None:
            cancelled = await cancellation_task
            get_logger().warning(
                f"Cancelled {cancelled} unfinished eval episodes for step {step}; advancing to checkpoint {newer_step}"
            )

    async def finalize_eval_batch(self, batch: EvalBatch) -> None:
        """Persist + log one completed eval epoch through the monitors, mirroring the
        orchestrator: effective episodes plus the ``eval/{env}/...`` metric dict."""
        if not batch.episodes and not batch.failures and not batch.cancelled:
            get_logger().warning(f"Eval @ step={batch.step} env={batch.env_name}: no attempts returned, skipping log")
            return

        if batch.episodes.effective:
            await monitors.log(batch.episodes.effective.vf_episodes, batch.step, "eval", "effective")
            await monitors.log_annotations(stamp_batch(batch.episodes.effective.vf_episodes, batch.step))
        await monitors.log_eval_epoch(batch.env_name, batch.step, batch.episodes.vf_episodes)

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
        metrics["step"] = float(batch.step)
        await monitors.log(metrics, step=batch.step)

        eff, full = effective.metrics, episodes.metrics
        triggered_at = self.eval_triggered_at.pop((batch.env_name, batch.step), None)
        elapsed = (time.perf_counter() - triggered_at) if triggered_at is not None else 0.0
        if batch.cancelled:
            get_logger().warning(
                f"Partially evaluated {batch.env_name} (Step {batch.step}) | "
                f"{format_time(elapsed):>7} | Reward {eff.reward.mean():.4f} | "
                f"Completed {len(episodes)}/{total_attempts} | Cancelled {batch.cancelled}/{total_attempts}"
            )
            return
        get_logger().success(
            f"Evaluated {batch.env_name} (Step {batch.step}) | "
            f"{format_time(elapsed):>7} | Reward {eff.reward.mean():.4f} | "
            f"Turns {eff.num_turns.mean():.1f} | Branches {eff.num_branches.mean():.1f} | "
            f"Error {full.has_error.mean():.1%} | Truncation {eff.is_truncated.mean():.1%}"
        )

    def collect_pipeline_view(self) -> tuple[str, dict[str, float]]:
        """Pipeline view for the ``PeriodicLogger``: per-env epoch progress plus the
        in-flight pool against the controller's current cap."""
        disp_gauges = self.dispatcher.gauges()
        disp_drain = self.dispatcher.metrics.drained(train_envs=set(), eval_envs={env.name for env in self.eval_envs})

        parts = []
        for env_name, _step, arrived, expected, buffered in sorted(self.eval_sink.batch_progress()):
            part = f"{env_name} {arrived}/{expected} ({arrived / expected:.1%})" if expected else env_name
            if buffered:
                part += f" (+{buffered} buffered)"
            parts.append(part)
        progress_part = " | ".join(parts) if parts else "Idle"

        body = (
            f"{progress_part}; {self.dispatcher.inflight_eval_count} inflight episodes "
            f"(cap {self.dispatcher.max_inflight}, signal {self.concurrency.signal})"
        )
        payload = {**disp_gauges, **disp_drain, **self.concurrency.gauges()}
        return body, payload

    async def drain(self) -> None:
        """Stop the background loggers so nothing logs after the monitors finalize."""
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
        if self.admin_clients is not None:
            await self.admin_clients.aclose()
        cleanup_processes(self.env_server_procs)
