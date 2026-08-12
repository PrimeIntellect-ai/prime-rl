"""Online evaluator: disk-checkpoint-driven evals against a live inference server.

The trainer writes HF weight checkpoints to ``weights/step_{n}`` (``STABLE``
marker on completion) — that is its signal that a new policy is ready. The
evaluator watches the directory, tells the inference server to reload each
eligible checkpoint from disk (``/update_weights``, no NCCL rendezvous), and
runs the configured evals against the updated weights, sequentially per
checkpoint so every epoch measures exactly one policy version.

Reuses the orchestrator's eval components — ``EvalEnvs`` / ``EvalSource`` /
``EvalSink`` / ``EvalRollouts`` — and logs metrics and traces the same way
(``eval/{env}/...`` metrics, ``rollouts/step_{n}/eval/...`` trace files).
"""

from __future__ import annotations

import asyncio
import time
import uuid

import verifiers.v1 as vf

from prime_rl.configs.evaluator import EvaluatorConfig
from prime_rl.orchestrator.envs import EvalEnv, EvalEnvs
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.patches import (
    monkey_patch_chat_completion_logprobs,
    monkey_patch_oai_iterable_types,
)
from prime_rl.orchestrator.types import EvalBatch, Rollout
from prime_rl.orchestrator.utils import intercept_vf_logging, save_rollouts, set_default_executor
from prime_rl.utils.client import InferencePool
from prime_rl.utils.logger import format_time, get_logger, setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pathing import get_all_ckpt_steps, get_step_path, get_trace_path
from prime_rl.utils.utils import clean_exit

monkey_patch_oai_iterable_types()
monkey_patch_chat_completion_logprobs()

# How often to re-scan the weights directory for new checkpoints.
POLL_INTERVAL_S = 2.0


class Evaluator:
    def __init__(self, config: EvaluatorConfig) -> None:
        self.config = config
        setup_logger(config.log.level, json_logging=config.log.json_logging)
        intercept_vf_logging(logger="verifiers.v1", level="WARN")
        get_logger().info(f"Starting evaluator (weights_dir={config.weights_dir})")

        # The last weight-checkpoint step already handled (evaluated or skipped).
        self.last_step = config.resume_step or 0
        self.eval_triggered_at: dict[tuple[str, int], float] = {}

    async def setup(self) -> None:
        config = self.config
        set_default_executor()

        get_logger().info(f"Initializing monitor ({config.wandb})")
        self.monitor = setup_monitor(
            wandb_config=config.wandb,
            file_config=config.file_monitor,
            output_dir=config.output_dir,
            run_config=config,
            eval_env_names=[source.resolved_name for source in config.eval.source],
        )
        self.run_id = self.monitor.run_id or uuid.uuid4().hex

        get_logger().info(
            f"Initializing inference pool (base_url={config.eval.client.base_url}, model={config.model.name})"
        )
        self.pool = InferencePool(config.eval.client, model_name=config.model.name)

        get_logger().info("Loading eval environment(s)")
        self.eval_envs = EvalEnvs(config.eval.source, config.eval.env_addresses)
        await self.eval_envs.start()
        get_logger().success(f"Eval environment(s) ready ({', '.join(self.eval_envs.names)})")

        get_logger().info("Waiting for inference pool to be ready")
        await self.pool.wait_for_ready(config.model.name)
        get_logger().success("Inference pool ready")

        self.eval_source = EvalSource(self.eval_envs, config.eval, is_resumed=config.resume_step is not None)
        self.eval_sink = EvalSink(eval_envs=self.eval_envs)

    async def run(self) -> None:
        await self.setup()
        config = self.config

        if config.resume_step is None:
            # Base-model eval: the inference server starts with the untrained weights,
            # so no reload is needed. The first trigger fires every env (policy v0)
            # unless ``skip_first_step``.
            await self.maybe_run_evals(step=0)
        elif config.eval.retrigger_on_resume:
            # Re-fire interval-aligned evals at the resume step (e.g. after a crash
            # that lost in-flight evals). Requires the resume step's weights on disk.
            await self.maybe_run_evals(step=config.resume_step, reload_weights=True)

        get_logger().info(f"Watching {config.weights_dir} for new weight checkpoints (max_steps={config.max_steps})")
        while True:
            assert config.weights_dir is not None  # resolved by the config validator
            for step in get_all_ckpt_steps(config.weights_dir):
                if step <= self.last_step:
                    continue
                if not (get_step_path(config.weights_dir, step) / "STABLE").exists():
                    break  # still being written — later steps can't be ready either
                is_final = config.max_steps is not None and step >= config.max_steps
                await self.maybe_run_evals(step=step, reload_weights=True, force=is_final)
            if config.max_steps is not None and self.last_step >= config.max_steps:
                break
            await asyncio.sleep(POLL_INTERVAL_S)

        get_logger().success("Evaluator finished!")

    async def maybe_run_evals(self, step: int, *, reload_weights: bool = False, force: bool = False) -> None:
        """Fire eligible envs for one checkpoint step and run the full epoch(s),
        reloading the inference weights first. No-op when no env is due."""
        assert self.config.weights_dir is not None  # resolved by the config validator
        weight_dir = get_step_path(self.config.weights_dir, step)
        if reload_weights and not (weight_dir / "STABLE").exists():
            get_logger().warning(f"No stable weight checkpoint for step {step} ({weight_dir}) - skipping eval")
            self.last_step = max(self.last_step, step)
            return

        fired = self.eval_source.trigger(step, force=force)
        self.last_step = max(self.last_step, step)
        if not fired:
            return

        now = time.perf_counter()
        for env_name in fired:
            self.eval_triggered_at[(env_name, step)] = now
        total_rollouts = sum(
            self.eval_envs.get(env_name).config.group_size * len(self.eval_envs.get(env_name).examples)
            for env_name in fired
        )

        if reload_weights:
            get_logger().info(f"Updating inference weights to checkpoint step {step} ({weight_dir})")
            await self.pool.update_weights(weight_dir, step=step)

        get_logger().info(f"Starting evals in {', '.join(fired)} at step {step} ({total_rollouts} total rollouts)")
        await self.run_evals(step)

    async def run_evals(self, step: int) -> None:
        """Drain the eval queue for one step under bounded concurrency, routing
        finished episodes through the sink and finalizing each env's epoch."""
        semaphore = asyncio.Semaphore(self.config.eval.max_inflight_episodes)
        tasks: list[asyncio.Task] = []
        while (example := self.eval_source.next_example()) is not None:
            env = self.eval_envs.get(example["env_name"])
            group_id = uuid.uuid4()
            for _ in range(env.config.group_size):
                tasks.append(asyncio.create_task(self.run_episode(env, example, group_id, step, semaphore)))

        for future in asyncio.as_completed(tasks):
            episode = await future
            run = vf.EvalRunInfo(id=self.run_id, step=step)
            for rollout in episode:
                rollout.record_run(
                    run,
                    env_name=rollout.env_name,
                    group_id=str(rollout.group_id),
                    episode_id=rollout.episode_id,
                    policy_version=rollout.policy_version,
                )
            await asyncio.to_thread(
                save_rollouts,
                [rollout.to_record() for rollout in episode],
                get_trace_path(self.config.output_dir, step, "eval", "all"),
            )
            eval_batch = self.eval_sink.add(episode)
            if eval_batch is not None:
                await self.finalize_eval_batch(eval_batch)

    async def run_episode(
        self,
        env: EvalEnv,
        example: dict,
        group_id: uuid.UUID,
        step: int,
        semaphore: asyncio.Semaphore,
    ) -> list[Rollout]:
        """Run one episode against the env server; failures synthesize an
        error-marker rollout so the sink's count-based finalization still fires."""
        async with semaphore:
            try:
                rollouts = await env.run(
                    client=self.pool.eval_client,
                    model_name=self.pool.model_name,
                    cache_salt=str(step),
                    task_data=example["task"].data.model_dump(mode="json"),
                )
            except Exception as exc:
                get_logger().warning(f"Rollout task failed in group {group_id} ({env.name}): {exc!r}")
                rollout = Rollout(
                    task=vf.TraceTask(type="Task", data=vf.TaskData(idx=example["task"].data.idx, prompt=None)),
                    agent=vf.AgentInfo(config=vf.AgentConfig()),
                )
                rollout.record_error(exc)
                rollouts = [rollout]

        for rollout in rollouts:
            if not rollout.has_error and rollout.num_turns == 0:
                rollout.errors.append(
                    vf.Error(type="EmptyTrajectory", message="Rollout returned with no trajectory steps")
                )
                rollout.ok = False
                get_logger().warning(f"Empty trajectory in group {group_id} ({env.name})")
            rollout.kind = "eval"
            rollout.env_name = env.name
            rollout.group_id = group_id
            # For SFT-driven evals the policy version *is* the checkpoint step.
            rollout.policy_version = step
            rollout.eval_step = step
        return rollouts

    async def finalize_eval_batch(self, batch: EvalBatch) -> None:
        """Persist + log one completed eval epoch, mirroring the orchestrator:
        effective traces, ``monitor.log_eval_samples``, ``eval/{env}/...`` metrics."""
        if not batch.rollouts:
            get_logger().warning(f"Eval @ step={batch.step} env={batch.env_name}: no rollouts returned, skipping log")
            return

        records = [r.to_record() for r in batch.rollouts.effective]
        await asyncio.to_thread(
            save_rollouts, records, get_trace_path(self.config.output_dir, batch.step, "eval", "effective")
        )
        self.monitor.log_eval_samples(batch.rollouts, env_name=batch.env_name, step=batch.step)

        rollouts = batch.rollouts
        effective = rollouts.effective
        metrics: dict[str, float] = {}
        for subset, pool in (("all", rollouts), ("effective", effective)):
            metrics |= pool.metrics.to_wandb(prefix=f"eval/{batch.env_name}", subset=subset)
        metrics[f"eval/{batch.env_name}/policy_version"] = float(batch.step)
        metrics["step"] = float(batch.step)
        self.monitor.log(metrics, step=batch.step)

        eff, full = effective.metrics, rollouts.metrics
        triggered_at = self.eval_triggered_at.pop((batch.env_name, batch.step), None)
        elapsed = (time.perf_counter() - triggered_at) if triggered_at is not None else 0.0
        get_logger().success(
            f"Evaluated {batch.env_name} (Step {batch.step}) | "
            f"{format_time(elapsed):>7} | Reward {eff.reward.mean():.4f} | "
            f"Turns {eff.num_turns.mean():.1f} | Branches {eff.num_branches.mean():.1f} | "
            f"Error {full.has_error.mean():.1%} | Truncation {eff.is_truncated.mean():.1%}"
        )

    async def stop(self) -> None:
        """Best-effort teardown; tolerates a partially completed ``setup()``."""
        monitor = getattr(self, "monitor", None)
        if monitor is not None:
            monitor.save_final_summary()
        pool = getattr(self, "pool", None)
        if pool is not None:
            await pool.stop()


@clean_exit
async def run_evaluator(config: EvaluatorConfig) -> None:
    evaluator = Evaluator(config)
    try:
        await evaluator.run()
    finally:
        await evaluator.stop()


def main() -> None:
    from prime_rl.utils.config import cli
    from prime_rl.utils.process import set_proc_title

    set_proc_title("Evaluator")
    asyncio.run(run_evaluator(cli(EvaluatorConfig)))


if __name__ == "__main__":
    main()
