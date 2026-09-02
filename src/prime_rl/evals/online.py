"""Online evals: evaluate the trainer's weight broadcasts as they appear.

The process watches a broadcasts directory for offered weight broadcasts through a
``WeightReceiver`` (announced by their ``.sender_ready`` marker), moves the inference
server onto each of them, and runs the due eval sources against the updated weights,
sequentially per broadcast so every epoch measures exactly one policy version. Every
offered broadcast must be received, even when no eval is due — the trainer blocks inside
the handshake until the receiver acknowledges. By default, a newer checkpoint cancels
unfinished episodes from the prior version; ``cancel_on_new_checkpoint = false`` drains
every epoch instead."""

from __future__ import annotations

import asyncio
from pathlib import Path

from prime_rl import monitors
from prime_rl.configs.evals import OnlineEvalsConfig
from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig
from prime_rl.evals.runner import POLL_INTERVAL_S, EvalRunner
from prime_rl.transports.weights import WeightReceiver, setup_weight_receiver
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps
from prime_rl.utils.utils import clean_exit

# Budget for the trainer's startup broadcast: it is always coming, but only
# after the trainer has finished loading the model.
STARTUP_BROADCAST_TIMEOUT_S = 1200


class OnlineEvals:
    def __init__(self, config: OnlineEvalsConfig, log_dir: Path) -> None:
        self.config = config
        self.runner = EvalRunner(config, run_dir=config.output_dir, log_dir=log_dir)
        # The last weight-broadcast step already handled (evaluated or skipped).
        self.last_step = config.resume_step or 0
        self.receiver: WeightReceiver | None = None

    async def run(self) -> None:
        config = self.config
        get_logger().info(f"Initializing monitors ({config.monitors})")
        await monitors.setup(
            producer="online-evals",
            wandb=config.monitors.wandb,
            file=config.monitors.file,
            output_dir=config.output_dir,
            run_config=config,
            eval_env_names=[source.resolved_name for source in config.source],
            overview_flavor="sft",
        )
        await self.runner.setup(skip_first_step=config.skip_first_step, is_resumed=config.resume_step is not None)

        assert config.broadcasts_dir is not None  # resolved by the config validator
        # A hand-written config may omit the transport; broadcasts are then plain
        # filesystem checkpoints.
        weight_broadcast = config.weight_broadcast or FileSystemWeightBroadcastConfig()
        get_logger().info(f"Initializing weight broadcast ({weight_broadcast})")
        self.receiver = setup_weight_receiver(
            config.broadcasts_dir,
            weight_broadcast,
            admin_clients=self.runner.admin_clients.clients,
            model_name=config.model,
        )
        await self.receiver.initialize()

        await self.runner.start()
        await self.watch()
        await self.runner.drain()
        get_logger().success("Online evals finished!")

    async def watch(self) -> None:
        """Evaluate each eligible weight broadcast as it appears."""
        config = self.config
        assert self.receiver is not None
        assert config.broadcasts_dir is not None

        # Rendezvous with the trainer's startup broadcast (v0 fresh, the checkpoint step
        # on resume) — always, for every transport: an in-memory trainer blocks inside
        # its startup broadcast until this receive, and for filesystem it guarantees the
        # served weights match the trainer's incoming policy.
        startup_step = config.resume_step or 0
        await self.receiver.sync_startup(startup_step, timeout=STARTUP_BROADCAST_TIMEOUT_S)
        self.runner.policy.version = startup_step

        if config.resume_step is None:
            # The first trigger fires every env (policy v0) unless ``skip_first_step``.
            await self.maybe_run_evals(step=0)
        elif config.retrigger_on_resume:
            # Re-fire evals at the resume step (e.g. after a crash that lost in-flight
            # evals); the startup rendezvous above already loaded its weights. The
            # final broadcast force-fires every env, exactly like the watch loop below.
            is_final = config.max_steps is not None and config.resume_step >= config.max_steps
            await self.maybe_run_evals(step=config.resume_step, force=is_final)

        get_logger().info(f"Watching {config.broadcasts_dir} for new weight broadcasts (max_steps={config.max_steps})")
        while True:
            steps = get_all_ckpt_steps(config.broadcasts_dir)
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
                is_final = config.max_steps is not None and step >= config.max_steps
                await self.maybe_run_evals(step=step, reload_weights=True, force=is_final)
            if config.max_steps is not None and self.last_step >= config.max_steps:
                break
            await asyncio.sleep(POLL_INTERVAL_S)

    def next_published_step(self, step: int) -> int | None:
        """Return the first newer checkpoint offered by the trainer."""
        assert self.config.broadcasts_dir is not None
        assert self.receiver is not None
        return next(
            (
                candidate
                for candidate in get_all_ckpt_steps(self.config.broadcasts_dir)
                if candidate > step and self.receiver.is_published(candidate)
            ),
            None,
        )

    def deleted_due_steps(self, steps: list[int], newest_published: int | None) -> set[int]:
        """Eval-due steps up to the newest published broadcast that are missing from
        the broadcasts dir — the trainer wrote them (it broadcasts at every due step),
        so their absence means broadcast cleaning removed them before they were
        evaluated."""
        if newest_published is None:
            return set()
        due = {
            step
            for interval in self.runner.eval_source.intervals.values()
            for step in range(interval, newest_published + 1, interval)
        }
        return due - set(steps)

    async def maybe_run_evals(self, step: int, *, reload_weights: bool = False, force: bool = False) -> None:
        """Fire eligible envs for one checkpoint step and run the full epoch(s),
        reloading the inference weights first. No-op when no env is due — except
        that a live transport's broadcast must always be received (the trainer
        is blocked inside it), eval or no eval."""
        runner = self.runner
        if reload_weights:
            assert self.receiver is not None
            broadcast_dir = self.receiver.step_dir(step)
            if not self.receiver.is_published(step):
                get_logger().warning(f"No published weight broadcast for step {step} ({broadcast_dir}) - skipping eval")
                self.last_step = max(self.last_step, step)
                return

        # Trigger before the reload: the dispatcher only schedules eval in PREFER_EVAL,
        # so nothing dispatches until ``run_epoch`` switches modes below.
        fired = runner.eval_source.trigger(step, force=force)
        self.last_step = max(self.last_step, step)

        if reload_weights:
            # Every offered version must be received: the trainer blocks inside
            # the handshake, so a failed receive fails the run loudly.
            get_logger().info(f"Updating inference weights to broadcast step {step} ({broadcast_dir})")
            await runner.dispatcher.on_version_pending(step)
            await self.receiver.receive(step)
            runner.policy.version = step
            await runner.dispatcher.on_new_version(step)
        else:
            runner.policy.version = step

        if not fired:
            return
        superseding_step = (lambda: self.next_published_step(step)) if self.config.cancel_on_new_checkpoint else None
        await runner.run_epoch(fired, step, superseding_step=superseding_step)


@clean_exit
async def run_online_evals(config: OnlineEvalsConfig, log_dir: Path) -> None:
    evals = OnlineEvals(config, log_dir)
    try:
        await evals.run()
        # Finalize only on a clean exit — a crashed run must not mark the run completed.
        await monitors.finalize()
    finally:
        await evals.runner.stop()
