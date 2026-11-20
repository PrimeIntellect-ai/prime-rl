"""The orchestrator of an ``sft`` or ``pretrain`` run: it owns the run's schedule.

The launcher starts it next to the trainer. At the start of every step the trainer asks it for the step's plan
(``prime_rl.orchestrator.schedule``): whether to validate the weights (inline, on the trainer's GPUs) and whether to
broadcast them for online evals. With ``[orchestrator.eval]`` it runs the online evals: it moves the inference server
onto every broadcast it planned, in order, and runs the due eval sources against the updated weights, sequentially
per broadcast so every epoch measures exactly one policy version. Every planned broadcast must be received, even when
no eval is due - the trainer blocks inside the handshake until the receiver acknowledges. By default, a newer
broadcast cancels unfinished episodes of the prior version; ``cancel_on_new_checkpoint = false`` drains every epoch
instead."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from prime_rl import monitors
from prime_rl.configs.supervised import SupervisedOrchestratorProcessConfig
from prime_rl.eval.runner import EvalRunner
from prime_rl.orchestrator.schedule import plan_step
from prime_rl.transports.steps import StepPlan, StepPlanServer, StepRequest
from prime_rl.transports.weights import WeightReceiver, setup_weight_receiver
from prime_rl.utils.config import cli, dump_resolved_config
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import get_broadcast_dir, prepare_attempt_dirs
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit

# Budget for the trainer's startup broadcast: it is always coming, but only
# after the trainer has finished loading the model.
STARTUP_BROADCAST_TIMEOUT_S = 1200


class SupervisedOrchestrator:
    def __init__(self, config: SupervisedOrchestratorProcessConfig) -> None:
        self.config = config
        self.runner: EvalRunner | None = None
        if config.eval is not None:
            self.runner = EvalRunner(config.eval, model=config.model, log=config.log, run_dir=config.output_dir)
        self.receiver: WeightReceiver | None = None
        self.start_step: int | None = None
        # Planned broadcasts, each with whether it is the final one; None once the trainer is done.
        self.broadcasts: asyncio.Queue[tuple[int, bool] | None] = asyncio.Queue()

    def plan(self, request: StepRequest) -> StepPlan:
        if self.start_step is None:
            self.start_step = request.step
        plan = plan_step(request, self.start_step, self.config.val, self.config.eval)
        get_logger().debug(f"Trainer step {request.step}: {plan}")
        if plan.broadcast:
            self.broadcasts.put_nowait((request.step, request.stop == "finish"))
        if request.stop is not None:
            self.broadcasts.put_nowait(None)
        return plan

    async def run(self, server: StepPlanServer) -> None:
        if self.runner is None:
            await server.serve(self.plan)
            return
        await self.setup_evals()
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(server.serve(self.plan))
            tasks.create_task(self.evaluate())
        await self.runner.drain()
        get_logger().success("Online evals finished!")

    async def setup_evals(self) -> None:
        config, runner = self.config, self.runner
        assert config.eval is not None and runner is not None
        get_logger().info(f"Initializing monitors ({config.monitors})")
        await monitors.setup(
            producer="orchestrator",
            wandb=config.monitors.wandb,
            prime=config.monitors.prime,
            file=config.monitors.file,
            output_dir=config.output_dir,
            run_config=config,
            eval_env_names=[source.resolved_name for source in config.eval.source],
            overview_flavor="sft",
        )
        await runner.setup(skip_first_step=config.eval.skip_first_step, is_resumed=config.resume_step is not None)
        get_logger().info(f"Initializing weight broadcast ({config.weight_broadcast})")
        self.receiver = setup_weight_receiver(
            get_broadcast_dir(config.output_dir),
            config.weight_broadcast,
            admin_plane=runner.admin_plane,
            model_name=config.model,
        )
        await self.receiver.initialize()
        await runner.start()

    async def evaluate(self) -> None:
        """Evaluate the startup weights, then every planned broadcast until the trainer's last step."""
        config, runner, receiver = self.config, self.runner, self.receiver
        assert config.eval is not None and runner is not None and receiver is not None

        # Rendezvous with the trainer's startup broadcast (v0 fresh, the checkpoint step on resume) - always, for
        # every transport: an in-memory trainer blocks inside its startup broadcast until this receive, and for
        # filesystem it guarantees the served weights match the trainer's incoming policy.
        startup_step = config.resume_step or 0
        await receiver.sync_startup(startup_step, timeout=STARTUP_BROADCAST_TIMEOUT_S)
        runner.policy.version = startup_step
        if config.resume_step is None:
            # The first trigger fires every env (policy v0) unless ``skip_first_step``.
            await self.run_eval(0)
        elif config.eval.retrigger_on_resume:
            # Re-fire evals at the resume step (e.g. after a crash that lost in-flight evals).
            await self.run_eval(config.resume_step)

        while (planned := await self.broadcasts.get()) is not None:
            step, final = planned
            await receiver.wait_published(step)
            await self.run_eval(step, receive=True, force=final)

    async def run_eval(self, step: int, *, receive: bool = False, force: bool = False) -> None:
        """Fire the envs due at ``step`` (every env with ``force``, at the final step), move the inference server
        onto the step's broadcast if ``receive``, and run the epoch."""
        runner = self.runner
        # Trigger before the receive: the dispatcher only schedules eval in PREFER_EVAL,
        # so nothing dispatches until ``run_epoch`` switches modes below.
        fired, _ = runner.eval_source.trigger(step, force=force)
        if receive:
            get_logger().info(f"Updating inference weights to broadcast step {step}")
            await runner.dispatcher.on_version_pending(step)
            await self.receiver.receive(step)
            runner.policy.version = step
            await runner.dispatcher.on_new_version(step)
        if not fired:
            return
        superseding_step = (lambda: self.superseding_step(step)) if self.config.eval.cancel_on_new_checkpoint else None
        await runner.run_epoch(fired, step, superseding_step=superseding_step)

    def superseding_step(self, step: int) -> int | None:
        """A newer broadcast, once the trainer offers it. It offers one at a time, so this is the next planned one."""
        newer = self.receiver.next_version(step)
        return newer if newer > step else None


@clean_exit
async def run_orchestrator(config: SupervisedOrchestratorProcessConfig, config_dir: Path) -> None:
    # Bound before anything slow: the trainer waits for its address.
    server = StepPlanServer(config_dir)
    orchestrator = SupervisedOrchestrator(config)
    try:
        await orchestrator.run(server)
        # Finalize only on a clean exit - a crashed run must not mark the run completed.
        await monitors.finalize()
    finally:
        server.close()
        if orchestrator.runner is not None:
            await orchestrator.runner.stop()


def main():
    set_proc_title("Orchestrator")
    config = cli(SupervisedOrchestratorProcessConfig)
    config_dir, log_dir = prepare_attempt_dirs(config.output_dir)
    os.environ["PRL_ATTEMPT_CONFIG_DIR"] = str(config_dir)
    os.environ["PRL_ATTEMPT_LOG_DIR"] = str(log_dir)
    (config_dir / "orchestrator.json").write_text(json.dumps(dump_resolved_config(config), indent=2))
    setup_logger(config.log.level, json_logging=config.log.json_logging)
    asyncio.run(run_orchestrator(config, config_dir))


if __name__ == "__main__":
    main()
