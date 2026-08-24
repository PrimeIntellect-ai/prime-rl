"""Evals: multi-env evals against a live inference server.

Standalone (no ``[online]``), it runs one epoch of every configured eval
source against the weights the inference server currently serves, then exits.
With ``[online]``, it watches a broadcasts directory for offered weight
broadcasts through a ``WeightReceiver`` (announced by their
``.sender_ready`` marker), moves the inference server onto each of them, and
runs the configured evals against the updated weights, sequentially per
broadcast so every epoch measures exactly one policy version. Every offered
broadcast must be received, even when no eval is due — the trainer blocks
inside the handshake until the receiver acknowledges.

Scheduling reuses the orchestrator pipeline unchanged: an eval-only
``Dispatcher`` admits episodes under the adaptive ``ConcurrencyController``,
fed by the ``InferenceMetricsCollector``'s ``/metrics`` polls. Eval episodes
are version-pinned measurements, so an overload cut only blocks admission and
the in-flight pool drains through natural completions.

Env servers: sources without an explicit ``serve.address`` get an env server
spawned by the evals process at their derived address; sources with one are
externally managed (e.g. spawned by the ``sft`` launcher, which stamps the
derived addresses into this config)."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from subprocess import Popen

from prime_rl import monitors
from prime_rl.configs.evals import EvalsConfig
from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig
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
from prime_rl.orchestrator.utils import eval_work, intercept_vf_logging, set_default_executor
from prime_rl.transports.weights import WeightReceiver, setup_weight_receiver
from prime_rl.utils.config import dump_resolved_config
from prime_rl.utils.logger import format_time, get_logger, setup_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_config_dir, get_log_dir
from prime_rl.utils.process import DEFAULT_COMMON_ENV_VARS, cleanup_processes
from prime_rl.utils.utils import clean_exit

monkey_patch_oai_iterable_types()
monkey_patch_chat_completion_logprobs()

# How often to re-scan the broadcasts directory for new weight broadcasts.
POLL_INTERVAL_S = 2.0

# Budget for the trainer's startup broadcast: it is always coming, but only
# after the trainer has finished loading the model.
STARTUP_BROADCAST_TIMEOUT_S = 1200


class Evals:
    def __init__(self, config: EvalsConfig) -> None:
        self.config = config
        setup_logger(config.log.level, json_logging=config.log.json_logging)
        intercept_vf_logging(logger="verifiers.v1", level="WARN")
        mode = f"online (broadcasts_dir={config.online.broadcasts_dir})" if config.online is not None else "standalone"
        get_logger().info(f"Starting evals ({mode})")

        # The last weight-checkpoint step already handled (evaluated or skipped).
        self.last_step = (config.online.resume_step if config.online is not None else None) or 0
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

    async def setup(self) -> None:
        config = self.config
        set_default_executor()

        get_logger().info(f"Initializing monitors ({config.monitors})")
        await monitors.setup(
            wandb=config.monitors.wandb,
            file=config.monitors.file,
            output_dir=config.output_dir,
            run_config=config,
            eval_env_names=[source.resolved_name for source in config.eval.source],
            overview_flavor="sft",
        )
        # The launcher-set $PRL_RUN_ID is the run identity; standalone runs mint a local one.
        self.run_id = os.environ.get("PRL_RUN_ID") or uuid.uuid4().hex
        self.run_name = os.environ.get("PRL_RUN_NAME")
        wandb_enabled = monitors.get(monitors.WandbMonitor) is not None

        get_logger().info(f"Initializing inference pool (base_url={config.eval.client.base_url}, model={config.model})")
        self.clients = InferenceClient(config.eval.client, model_name=config.model)
        self.admin_clients = AdminClients(config.eval.client)

        self.spawn_env_servers()

        get_logger().info("Loading eval environment(s)")
        self.eval_envs = EvalEnvs(config.eval.source, config.eval.env_addresses)
        await self.eval_envs.start()
        get_logger().success(f"Eval environment(s) ready ({', '.join(self.eval_envs.names)})")

        get_logger().info("Waiting for inference pool to be ready")
        await self.admin_clients.wait_for_ready(config.model)
        get_logger().success("Inference pool ready")

        self.receiver: WeightReceiver | None = None
        if config.online is not None:
            assert config.online.broadcasts_dir is not None
            # A hand-written online config may omit the transport; broadcasts
            # are then plain filesystem checkpoints.
            weight_broadcast = config.weight_broadcast or FileSystemWeightBroadcastConfig()
            get_logger().info(f"Initializing weight broadcast ({weight_broadcast})")
            self.receiver = setup_weight_receiver(
                config.online.broadcasts_dir,
                weight_broadcast,
                admin_clients=self.admin_clients.clients,
                model_name=config.model,
            )
            await self.receiver.initialize()

        is_resumed = config.online is not None and config.online.resume_step is not None
        self.eval_source = EvalSource(self.eval_envs, config.eval, is_resumed=is_resumed)
        self.eval_sink = EvalSink(eval_envs=self.eval_envs)
        self.policy = Policy(version=0, model_name=config.model)

        # Pessimistic per-episode token cost for the controller's starting cap,
        # only used when the engine doesn't report its max context length.
        fallback_cost = max((source.sampling.max_completion_tokens or 0) for source in config.eval.source) or 8192
        self.concurrency = ConcurrencyController(config.eval.concurrency, fallback_cost=fallback_cost)
        self.dispatcher = Dispatcher(
            train_envs=None,
            eval_envs=self.eval_envs,
            train_source=None,
            eval_source=self.eval_source,
            policy_clients=self.clients,
            policy=self.policy,
            progress=None,
            initial_max_inflight=self.concurrency.max_inflight,
            max_inflight_ceiling=config.eval.concurrency.max_inflight,
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
        # W&B mirroring is gated on the registered monitor.
        self.inference_metrics = InferenceMetricsCollector(
            self.admin_clients.clients,
            on_load=self.concurrency.observe,
            log_to_wandb=wandb_enabled,
        )
        # Fail fast when adaptivity has no signal: external API endpoints
        # (e.g. Prime Inference) expose no vLLM /metrics, so without a probe
        # hit the cap would silently sit at min_inflight forever. A pinned
        # band (min_inflight = max_inflight) makes the controller inert and
        # is the supported way to run against such endpoints.
        if not await self.inference_metrics.probe():
            concurrency = config.eval.concurrency
            if concurrency.min_inflight != concurrency.max_inflight:
                urls = ", ".join(str(client.base_url) for client in self.admin_clients.clients)
                raise ValueError(
                    f"No engine metrics at {urls} - adaptive concurrency has no load signal. "
                    "The endpoint does not expose vLLM /metrics (e.g. an external inference API); "
                    "pin the concurrency by setting eval.concurrency.min_inflight = max_inflight."
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
        addresses = config.eval.env_addresses
        config_dir = get_config_dir(config.output_dir) / "envs" / "eval"
        log_dir = get_log_dir(config.output_dir) / "envs" / "eval"
        for source in config.eval.source:
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

    async def run(self) -> None:
        await self.setup()
        self.dispatcher_task = asyncio.create_task(self.dispatcher.start(), name="dispatcher")
        await self.periodic_logger.start()

        if self.config.online is None:
            await self.maybe_run_evals(step=0)
        else:
            await self.watch()

        # The periodic logger and the collector log to the W&B run, so they
        # must stop before finalize marks the run finished.
        await self.periodic_logger.stop()
        await self.inference_metrics.stop()
        get_logger().success("Evals finished!")

    async def watch(self) -> None:
        """Online mode: evaluate each eligible weight broadcast as it appears."""
        config = self.config
        assert config.online is not None
        online = config.online

        # Rendezvous with the trainer's startup broadcast (v0 fresh, the
        # checkpoint step on resume) — always, for every transport: an
        # in-memory trainer blocks inside its startup broadcast until this
        # receive, and for filesystem it guarantees the served weights match
        # the trainer's incoming policy.
        assert self.receiver is not None
        startup_step = online.resume_step or 0
        await self.receiver.sync_startup(startup_step, timeout=STARTUP_BROADCAST_TIMEOUT_S)
        self.policy.version = startup_step

        if online.resume_step is None:
            # The first trigger fires every env (policy v0) unless ``skip_first_step``.
            await self.maybe_run_evals(step=0)
        elif config.eval.retrigger_on_resume:
            # Re-fire evals at the resume step (e.g. after a crash that lost in-flight
            # evals); the startup rendezvous above already loaded its weights. The
            # final broadcast force-fires every env, exactly like the watch loop below.
            is_final = online.max_steps is not None and online.resume_step >= online.max_steps
            await self.maybe_run_evals(step=online.resume_step, force=is_final)

        get_logger().info(f"Watching {online.broadcasts_dir} for new weight broadcasts (max_steps={online.max_steps})")
        while True:
            assert online.broadcasts_dir is not None  # resolved by the config validator
            steps = get_all_ckpt_steps(online.broadcasts_dir)
            published = {step: self.receiver.is_published(step) for step in steps}
            newest_published = max((step for step in steps if published[step]), default=None)
            # Also walk eval-due steps that are no longer on disk: broadcast cleaning
            # (the trainer keeps only the newest broadcast) can delete a step before
            # this scan sees it, and a vanished step would otherwise be skipped
            # without a trace.
            for step in sorted(set(steps) | self.deleted_due_steps(steps, newest_published)):
                if step <= self.last_step:
                    continue
                if step not in published:
                    get_logger().warning(
                        f"Weight broadcast for eval step {step} was deleted before it could be "
                        "evaluated (broadcast cleaning outpaced the evals process) - skipping its evals"
                    )
                    self.last_step = max(self.last_step, step)
                    continue
                if not published[step]:
                    # The trainer writes broadcasts in ascending order, so a marker-less
                    # step below a published one is an abandoned partial write (e.g. a
                    # crash mid-save), not one in progress — skip it instead of wedging.
                    if newest_published is None or newest_published < step:
                        break  # still being written — later steps can't be ready either
                    get_logger().warning(
                        f"Weight broadcast step {step} is not marked published but newer "
                        "broadcasts are - treating it as abandoned and skipping its evals"
                    )
                    self.last_step = max(self.last_step, step)
                    continue
                is_final = online.max_steps is not None and step >= online.max_steps
                await self.maybe_run_evals(step=step, reload_weights=True, force=is_final)
            if online.max_steps is not None and self.last_step >= online.max_steps:
                break
            await asyncio.sleep(POLL_INTERVAL_S)

    def deleted_due_steps(self, steps: list[int], newest_published: int | None) -> set[int]:
        """Eval-due steps up to the newest published broadcast that are missing from
        the broadcasts dir — the trainer wrote them (it broadcasts at every due step),
        so their absence means broadcast cleaning removed them before they were
        evaluated."""
        if newest_published is None:
            return set()
        due = {
            step
            for interval in self.eval_source.intervals.values()
            for step in range(interval, newest_published + 1, interval)
        }
        return due - set(steps)

    async def maybe_run_evals(self, step: int, *, reload_weights: bool = False, force: bool = False) -> None:
        """Fire eligible envs for one checkpoint step and run the full epoch(s),
        reloading the inference weights first. No-op when no env is due — except
        that a live transport's broadcast must always be received (the trainer
        is blocked inside it), eval or no eval."""
        if reload_weights:
            assert self.receiver is not None
            broadcast_dir = self.receiver.step_dir(step)
            if not self.receiver.is_published(step):
                get_logger().warning(f"No published weight broadcast for step {step} ({broadcast_dir}) - skipping eval")
                self.last_step = max(self.last_step, step)
                return

        fired = self.eval_source.trigger(step, force=force)
        self.last_step = max(self.last_step, step)

        if reload_weights:
            # Every offered version must be received: the trainer blocks inside
            # the handshake, so a failed receive fails the run loudly.
            get_logger().info(f"Updating inference weights to broadcast step {step} ({broadcast_dir})")
            await self.receiver.receive(step)

        if not fired:
            return

        now = time.perf_counter()
        for env_name in fired:
            self.eval_triggered_at[(env_name, step)] = now
        total_rollouts = sum(
            self.eval_envs.get(env_name).config.group_size * len(self.eval_envs.get(env_name).examples)
            for env_name in fired
        )

        # The dispatcher only schedules eval in PREFER_EVAL, so nothing dispatches
        # between the trigger above and the weight reload completing.
        self.policy.version = step
        get_logger().info(f"Starting evals in {', '.join(fired)} at step {step} ({total_rollouts} total rollouts)")
        self.dispatcher.switch_mode(DispatcherMode.PREFER_EVAL, reason=f"eval was triggered at step {step}")
        await self.consume_epoch(fired)

    async def consume_epoch(self, fired: list[str]) -> None:
        """Consume dispatcher episodes until every fired env's epoch finalizes,
        routing them through the sink and monitors."""
        # An env with no examples emits no episodes, so its epoch can never finalize.
        pending = {env_name for env_name in fired if self.eval_sink.batch_size_for(env_name) > 0}
        while pending:
            item = await self.dispatcher.out_q.get()
            if isinstance(item, GroupCancellation):
                raise RuntimeError("Eval dispatcher emitted a group cancellation")
            if isinstance(item, DispatchFailure):
                eval_batch = self.eval_sink.fail(item)
            else:
                step = eval_work(item).step
                await monitors.log([item], step, "eval", "all")
                eval_batch = self.eval_sink.add(item)
            if eval_batch is not None:
                await self.finalize_eval_batch(eval_batch)
                pending.discard(eval_batch.env_name)

    async def finalize_eval_batch(self, batch: EvalBatch) -> None:
        """Persist + log one completed eval epoch through the monitors, mirroring the
        orchestrator: effective episodes plus the ``eval/{env}/...`` metric dict."""
        if not batch.episodes and not batch.failures:
            get_logger().warning(f"Eval @ step={batch.step} env={batch.env_name}: no attempts returned, skipping log")
            return

        if batch.episodes.effective:
            await monitors.log(batch.episodes.effective.vf_episodes, batch.step, "eval", "effective")

        episodes = batch.episodes
        effective = episodes.effective
        metrics: dict[str, float] = {}
        for subset, pool in (("all", episodes), ("effective", effective)):
            metrics |= pool.metrics.to_wandb(prefix=f"eval/{batch.env_name}", subset=subset)
        total_attempts = len(episodes) + len(batch.failures)
        metrics |= dispatch_failure_metrics(
            batch.failures,
            prefix=f"eval/{batch.env_name}/all",
            total_attempts=total_attempts,
        )
        metrics[f"eval/{batch.env_name}/policy_version"] = float(batch.step)
        metrics["step"] = float(batch.step)
        await monitors.log(metrics, step=batch.step)

        eff, full = effective.metrics, episodes.metrics
        triggered_at = self.eval_triggered_at.pop((batch.env_name, batch.step), None)
        elapsed = (time.perf_counter() - triggered_at) if triggered_at is not None else 0.0
        dispatch_failure_rate = len(batch.failures) / total_attempts
        get_logger().success(
            f"Evaluated {batch.env_name} (Step {batch.step}) | "
            f"{format_time(elapsed):>7} | Reward {eff.reward.mean():.4f} | "
            f"Turns {eff.num_turns.mean():.1f} | Branches {eff.num_branches.mean():.1f} | "
            f"Error {full.has_error.mean():.1%} | Dispatch Failure {dispatch_failure_rate:.1%} | "
            f"Truncation {eff.is_truncated.mean():.1%}"
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


@clean_exit
async def run_evals(config: EvalsConfig) -> None:
    evals = Evals(config)
    try:
        await evals.run()
        # Finalize only on a clean exit — a crashed evals must not mark the run completed.
        await monitors.finalize()
    finally:
        await evals.stop()


def main() -> None:
    from prime_rl.utils.config import cli
    from prime_rl.utils.process import set_proc_title

    set_proc_title("Evals")
    asyncio.run(run_evals(cli(EvalsConfig)))


if __name__ == "__main__":
    main()
