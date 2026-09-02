"""Evals: one epoch of every configured eval source against the weights the inference
server currently serves.

The progress cursor is checkpointed as task groups complete (``[ckpt]``, on by default),
so an interrupted run resumes with ``--resume`` and skips the completed prefix. Every
episode streams through the monitors as it arrives; a finished epoch also goes to the
platform when ``monitors.prime`` is set."""

from __future__ import annotations

from pathlib import Path

from prime_rl import monitors
from prime_rl.configs.evals import EvalsConfig
from prime_rl.evals.ckpt import CheckpointManager
from prime_rl.evals.runner import EvalRunner
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import clean_exit


class Evals:
    def __init__(self, config: EvalsConfig, log_dir: Path) -> None:
        self.config = config
        self.runner = EvalRunner(config, run_dir=config.run_dir, log_dir=log_dir)
        self.ckpt_manager = CheckpointManager(config.run_dir)
        self.last_saved_cursor = 0

    async def run(self) -> None:
        config = self.config
        get_logger().info(f"Initializing monitors ({config.monitors})")
        await monitors.setup(
            producer="evals",
            wandb=config.monitors.wandb,
            prime=config.monitors.prime,
            file=config.monitors.file,
            output_dir=config.run_dir,
            run_config=config,
            eval_env_names=[source.resolved_name for source in config.source],
            overview_flavor="eval",
        )
        await self.runner.setup()
        eval_source = self.runner.eval_source
        if config.resume is not None:
            if config.resume.dir is not None:
                self.ckpt_manager.load(config.resume.dir_step, eval_source, path=config.resume.dir / "evals")
            else:
                self.ckpt_manager.load(config.resume.step or self.ckpt_manager.latest_step(), eval_source)
            self.last_saved_cursor = eval_source.cursor
            get_logger().info(f"Resuming evals from task cursor {eval_source.cursor}")

        await self.runner.start()
        fired = eval_source.trigger(0)
        if fired:
            await self.runner.run_epoch(fired, 0, on_group_completed=self.on_group_completed)
        else:
            get_logger().info("Nothing to evaluate - every task group is already completed")
        self.save_checkpoint(force=True)
        await self.runner.drain()
        get_logger().success("Evals finished!")

    def on_group_completed(self, source_index: int) -> None:
        if self.runner.eval_source.mark_completed(source_index):
            self.save_checkpoint()

    def save_checkpoint(self, *, force: bool = False) -> None:
        if self.config.ckpt is None:
            return
        cursor = self.runner.eval_source.cursor
        if cursor <= 0 or cursor == self.last_saved_cursor:
            return
        if not force and cursor - self.last_saved_cursor < self.config.ckpt.interval:
            return
        self.ckpt_manager.save(self.runner.eval_source, keep_last=self.config.ckpt.keep_last)
        self.last_saved_cursor = cursor


@clean_exit
async def run_evals(config: EvalsConfig, log_dir: Path) -> None:
    evals = Evals(config, log_dir)
    try:
        await evals.run()
        # Finalize only on a clean exit — a crashed run must not mark itself completed.
        await monitors.finalize()
    finally:
        await evals.runner.stop()
