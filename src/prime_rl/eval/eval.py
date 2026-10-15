"""Evals: one epoch of every configured eval source against the weights the inference
server currently serves.

Every episode streams through the monitors as it arrives; a finished epoch also goes to
the platform when ``monitors.prime`` is set. An interrupted run resumes with
``--resume`` from its trace stream: the landed episodes rejoin the epoch and only the
rollouts still owed run (``prime_rl.eval.resume``)."""

from __future__ import annotations

import verifiers.v1 as vf

from prime_rl import monitors
from prime_rl.configs.eval import EvalConfig
from prime_rl.eval import resume
from prime_rl.eval.runner import EvalRunner
from prime_rl.utils.config import dump_resolved_config
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import clean_exit


class Eval:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.runner = EvalRunner(config, run_dir=config.run_dir)

    async def run(self) -> None:
        config = self.config
        landed: list[vf.WireEpisode] = []
        if config.resume:
            # read and set aside before the monitors start: the resumed attempt writes a fresh stream
            landed = resume.take_landed(config.run_dir, config)
        get_logger().info(f"Initializing monitors ({config.monitors})")
        await monitors.setup(
            producer="eval",
            wandb=config.monitors.wandb,
            prime=config.monitors.prime,
            file=config.monitors.file,
            output_dir=config.run_dir,
            run_config=config,
            eval_env_names=[source.resolved_name for source in config.source],
            overview_flavor="eval",
        )
        resume.stamp_config(config.run_dir, dump_resolved_config(config))
        await self.runner.setup()
        fired, restored = self.runner.eval_source.trigger(0, completed=landed)
        if config.resume:
            get_logger().info(
                f"Resuming from the trace stream: {len(restored)} episodes restored, "
                f"{sum(request.rollouts or 0 for request in self.runner.eval_source.queue)} rollouts owed"
            )

        await self.runner.start()
        await self.runner.run_epoch(fired, 0, restored=restored)
        await self.runner.drain()


@clean_exit
async def run_eval(config: EvalConfig) -> None:
    evaluation = Eval(config)
    try:
        await evaluation.run()
        # Finalize only on a clean exit — a crashed run must not mark itself completed.
        await monitors.finalize()
    finally:
        await evaluation.runner.stop()
