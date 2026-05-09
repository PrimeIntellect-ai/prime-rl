import asyncio
import time
from collections import defaultdict
from typing import Any

import verifiers as vf

from prime_rl.configs.orchestrator import (
    BatchingConfig,
    OrchestratorConfig,
    SamplesBatching,
    StepBatching,
    TokensBatching,
)
from prime_rl.orchestrator.ckpt import CkptManager, OrchState
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters, setup_filters
from prime_rl.orchestrator.group import EvalGroup, GRPOGroup, Policy, Trajectory
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.trajectories import interleave_rollout, pretokenize_rollout_trajectory
from prime_rl.orchestrator.vf_utils import get_completion_len
from prime_rl.transport import TrainingBatch, TrainingSample, setup_training_batch_sender
from prime_rl.transport.base import TrainingBatchSender
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import get_monitor


class Done(Exception):
    """Raised by the batcher when max_steps has been reached. Caught by run()."""


# Per-rollout policy version stamped at receipt so PostProcessor can compute
# lag metrics after BatchingStrategy unpacks Trajectories into flat lists.
_POLICY_VERSION_KEY = "_policy_version"


def _split(rollouts: list[vf.RolloutOutput]) -> tuple[list[vf.RolloutOutput], list[vf.RolloutOutput]]:
    trainable: list[vf.RolloutOutput] = []
    filtered: list[vf.RolloutOutput] = []
    for r in rollouts:
        (filtered if r.get("is_filtered") else trainable).append(r)
    return trainable, filtered


class BatchingStrategy:
    def add(self, rollouts: list[vf.RolloutOutput]) -> None: ...
    def has_batch(self) -> bool: ...
    def pop(self) -> tuple[list[vf.RolloutOutput], list[vf.RolloutOutput]]: ...


class StepStrategy:
    """Ship the first `size` rollouts produced, pre-filter."""

    def __init__(self, size: int):
        self.size = size
        self._buf: list[vf.RolloutOutput] = []

    def add(self, rollouts: list[vf.RolloutOutput]) -> None:
        self._buf.extend(rollouts)

    def has_batch(self) -> bool:
        return len(self._buf) >= self.size

    def pop(self) -> tuple[list[vf.RolloutOutput], list[vf.RolloutOutput]]:
        cohort, self._buf = self._buf[: self.size], self._buf[self.size :]
        return _split(cohort)


class SamplesStrategy:
    """Ship when `size` trainable rollouts (post-filter) accumulate."""

    def __init__(self, size: int):
        self.size = size
        self._buf: list[vf.RolloutOutput] = []

    def add(self, rollouts: list[vf.RolloutOutput]) -> None:
        self._buf.extend(rollouts)

    def has_batch(self) -> bool:
        return sum(1 for r in self._buf if not r.get("is_filtered")) >= self.size

    def pop(self) -> tuple[list[vf.RolloutOutput], list[vf.RolloutOutput]]:
        trainable: list[vf.RolloutOutput] = []
        filtered: list[vf.RolloutOutput] = []
        cut = 0
        for i, r in enumerate(self._buf):
            if r.get("is_filtered"):
                filtered.append(r)
            else:
                trainable.append(r)
                if len(trainable) == self.size:
                    cut = i + 1
                    break
        self._buf = self._buf[cut:]
        return trainable, filtered


class TokensStrategy:
    """Ship when trainable completion tokens (post-filter) reach `size`."""

    def __init__(self, size: int):
        self.size = size
        self._buf: list[vf.RolloutOutput] = []

    def add(self, rollouts: list[vf.RolloutOutput]) -> None:
        self._buf.extend(rollouts)

    def has_batch(self) -> bool:
        return sum(get_completion_len(r) for r in self._buf if not r.get("is_filtered")) >= self.size

    def pop(self) -> tuple[list[vf.RolloutOutput], list[vf.RolloutOutput]]:
        trainable: list[vf.RolloutOutput] = []
        filtered: list[vf.RolloutOutput] = []
        tokens = 0
        cut = 0
        for i, r in enumerate(self._buf):
            if r.get("is_filtered"):
                filtered.append(r)
            else:
                trainable.append(r)
                tokens += get_completion_len(r)
                if tokens >= self.size:
                    cut = i + 1
                    break
        self._buf = self._buf[cut:]
        return trainable, filtered


def build_strategy(config: BatchingConfig) -> BatchingStrategy:
    if isinstance(config, StepBatching):
        return StepStrategy(config.size)
    if isinstance(config, SamplesBatching):
        return SamplesStrategy(config.size)
    if isinstance(config, TokensBatching):
        return TokensStrategy(config.size)
    raise ValueError(f"Unknown batching config: {config!r}")


def _key(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


_TIMING_FIELDS = ("total", "setup", "generation", "model", "env", "scoring", "overhead")


def _rollout_timing(r: vf.RolloutOutput) -> dict[str, float]:
    t = r.get("timing")
    if not t:
        return {}
    out: dict[str, float] = {}
    for k in _TIMING_FIELDS:
        v = t.get(k)
        if isinstance(v, dict):
            out[k] = float(v.get("duration", 0.0))
        elif v is not None:
            out[k] = float(v)
    return out


def _rollout_metrics(prefix: str, rollouts: list[vf.RolloutOutput]) -> dict:
    if not rollouts:
        return {}
    rewards = [r.get("reward", 0.0) for r in rollouts]
    lens = [get_completion_len(r) for r in rollouts]
    m: dict = {
        _key(prefix, "reward/mean"): sum(rewards) / len(rewards),
        _key(prefix, "seq_len/mean"): sum(lens) / len(lens),
        _key(prefix, "pass_rate"): sum(1 for r in rewards if r > 0) / len(rewards),
        _key(prefix, "n_rollouts"): len(rollouts),
    }
    if "is_filtered" in rollouts[0]:
        n_filt = sum(1 for r in rollouts if r.get("is_filtered"))
        m[_key(prefix, "filters/drop_rate")] = n_filt / len(rollouts)
    if "filters" in rollouts[0]:
        for name in rollouts[0]["filters"]:
            hits = sum(1 for r in rollouts if r["filters"].get(name))
            m[_key(prefix, f"filters/{name}/rate")] = hits / len(rollouts)
    timings = [_rollout_timing(r) for r in rollouts]
    timings = [t for t in timings if t]
    if timings:
        for field_name in _TIMING_FIELDS:
            vals = [t[field_name] for t in timings if field_name in t]
            if vals:
                m[_key(prefix, f"timing/{field_name}/mean")] = sum(vals) / len(vals)
    return m


def _eval_metrics(prefix: str, trajs: list[Trajectory]) -> dict:
    """Eval-only per-example stats: pass_at_k, avg_at_k, n_examples."""
    non_empty = [t for t in trajs if t.rollouts]
    if not non_empty:
        return {}
    passed = sum(1 for t in non_empty if any(r.get("reward", 0.0) > 0 for r in t.rollouts))
    per_ex_means = [sum(r.get("reward", 0.0) for r in t.rollouts) / len(t.rollouts) for t in non_empty]
    return {
        _key(prefix, "pass_at_k"): passed / len(non_empty),
        _key(prefix, "avg_at_k"): sum(per_ex_means) / len(per_ex_means),
        _key(prefix, "n_examples"): len(non_empty),
    }


class PostProcessor:
    """Converts rollouts -> TrainingSamples, sends the batch, emits per-step logs/metrics."""

    def __init__(self, tokenizer, sender: TrainingBatchSender, policy: Policy):
        self.tokenizer = tokenizer
        self.sender = sender
        self.policy = policy
        self.logger = get_logger()
        self._last_step_t = time.perf_counter()

    async def process(
        self,
        trainable: list[vf.RolloutOutput],
        filtered: list[vf.RolloutOutput],
        step: int,
    ) -> None:
        t0 = time.perf_counter()
        samples = await asyncio.to_thread(self._convert, trainable)
        convert_time = time.perf_counter() - t0
        if not samples:
            self.logger.warning(
                f"Step {step}: shipping empty batch ({len(filtered)} filtered, 0 trainable). "
                f"Trainer may fail on this step — consider relaxing filters or batch_size."
            )

        t1 = time.perf_counter()
        batch = TrainingBatch(examples=samples, step=step)
        await asyncio.to_thread(self.sender.send, batch)
        send_time = time.perf_counter() - t1

        now = time.perf_counter()
        step_time = now - self._last_step_t
        self._last_step_t = now

        self._log(trainable, filtered, samples, step, step_time, convert_time, send_time)

    def _convert(self, rollouts: list[vf.RolloutOutput]) -> list[TrainingSample]:
        samples: list[TrainingSample] = []
        for r in rollouts:
            pretokenize_rollout_trajectory(r, self.tokenizer)
            out = interleave_rollout(r)
            if out is None:
                continue
            for s in out:
                s.advantage = r.get("advantage")
                s.reward = r.get("reward")
            samples.extend(out)
        return samples

    def _max_off_policy(self, rollouts: list[vf.RolloutOutput]) -> int:
        versions = [r.get(_POLICY_VERSION_KEY) for r in rollouts]
        versions = [v for v in versions if v is not None]
        if not versions:
            return 0
        return max(self.policy.version - v for v in versions)

    def _log(
        self,
        trainable: list[vf.RolloutOutput],
        filtered: list[vf.RolloutOutput],
        samples: list[TrainingSample],
        step: int,
        step_time: float,
        convert_time: float,
        send_time: float,
    ) -> None:
        cohort = trainable + filtered
        advs_t = [r.get("advantage") or 0.0 for r in trainable]
        adv_abs = sum(abs(a) for a in advs_t) / len(trainable) if trainable else 0.0
        async_level = step - self.policy.version
        max_off_policy_level = self._max_off_policy(cohort)

        metrics: dict = {
            **_rollout_metrics("", trainable),
            **_rollout_metrics("rollouts", cohort),
            "advantage/abs_mean": adv_abs,
            "batch_size": len(samples),
            "policy_version": self.policy.version,
            "scheduler/async_level": async_level,
            "scheduler/max_off_policy_level": max_off_policy_level,
            "time/step": step_time,
            "time/convert": convert_time,
            "time/ship": send_time,
        }
        get_monitor().log(metrics, step=step)

        reward_str = f"{metrics['reward/mean']:.4f}" if "reward/mean" in metrics else "n/a"
        seqlen_str = f"{metrics['seq_len/mean']:.1f} tokens/sample" if "seq_len/mean" in metrics else "n/a"
        self.logger.success(
            f"Step {step} | "
            f"Time: {step_time:.2f}s | "
            f"Reward: {reward_str} | "
            f"Seq. Length: {seqlen_str} | "
            f"Async Level: {async_level} | "
            f"Max. Off-Policy Level: {max_off_policy_level} | "
            f"Filtered: {len(filtered)}/{len(cohort)}"
        )


class TrainBatcher:
    """Wires the stages: annotate (filters) → accumulate (BatchingStrategy) →
    post-process (PostProcessor). Trajectories arrive pre-scored; eval ones
    are routed into an aggregator keyed by eval_step."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        in_q: asyncio.Queue[Trajectory],
        tokenizer: Any,
        sender: TrainingBatchSender,
        policy: Policy,
        eval_group: EvalGroup | None = None,
        train_groups: list[GRPOGroup] | None = None,
        ckpt_manager: CkptManager | None = None,
        inference_metrics: InferenceMetricsCollector | None = None,
    ):
        self.config = config
        self.in_q = in_q
        self.policy = policy
        self.strategy = build_strategy(config.batch_size)
        self.filters: list[RolloutFilter] = setup_filters(config.filters, vocab_size=tokenizer.vocab_size)
        self.post = PostProcessor(tokenizer, sender, policy)
        self.heartbeat = Heartbeat(config.heartbeat.url) if config.heartbeat else None
        self.eval_group = eval_group
        self.train_groups = train_groups or []
        self.ckpt_manager = ckpt_manager
        self.inference_metrics = inference_metrics
        self.step = 0
        self._eval_buf: dict[int, list[Trajectory]] = defaultdict(list)
        self.logger = get_logger()

    async def _wait_barrier(self) -> None:
        # Don't ship more than max_async_level of the latest policy version.
        # Stalling here cascades backpressure through the queue into the
        # run_groups semaphore.
        target = self.config.max_async_level
        strict = self.config.strict_async_level
        t0 = time.perf_counter()
        next_warn = 30.0
        while True:
            lead = self.step - self.policy.version
            if strict:
                if lead == target:
                    return
            elif lead <= target:
                return
            elapsed = time.perf_counter() - t0
            if elapsed >= next_warn:
                self.logger.warning(
                    f"Batcher stalled at barrier for {int(elapsed)}s: step={self.step}, "
                    f"policy_version={self.policy.version}, lead={lead} "
                    f"(max_async_level={target}). Trainer may be stuck or down."
                )
                next_warn = elapsed + 60.0
            await asyncio.sleep(0.1)

    def _stamp_version(self, traj: Trajectory) -> None:
        for r in traj.rollouts:
            r[_POLICY_VERSION_KEY] = traj.policy_version

    def _handle_eval(self, traj: Trajectory) -> None:
        if traj.eval_step is None or self.eval_group is None:
            return
        if self.filters and traj.rollouts:
            apply_filters(self.filters, traj.rollouts)
        buf = self._eval_buf[traj.eval_step]
        buf.append(traj)
        expected = self.eval_group.expected_eval_count(traj.eval_step)
        if expected is None or len(buf) < expected:
            return
        self._flush_eval(traj.eval_step, self._eval_buf.pop(traj.eval_step))

    def _flush_eval(self, step: int, trajs: list[Trajectory]) -> None:
        per_env: dict[str, list[Trajectory]] = defaultdict(list)
        for t in trajs:
            per_env[t.env_id].append(t)
        metrics: dict = {}
        for env_id, env_trajs in per_env.items():
            flat = [r for t in env_trajs for r in t.rollouts]
            metrics.update(_rollout_metrics(f"eval/{env_id}", flat))
            metrics.update(_eval_metrics(f"eval/{env_id}", env_trajs))
        all_rollouts = [r for t in trajs for r in t.rollouts]
        metrics.update(_rollout_metrics("eval", all_rollouts))
        metrics.update(_eval_metrics("eval", trajs))
        if "eval/reward/mean" not in metrics:
            self.logger.warning(f"Eval @ step {step}: all {len(trajs)} groups timed out, skipping log")
            return
        n_timed_out = sum(1 for t in trajs if not t.rollouts)
        if n_timed_out:
            metrics["eval/timed_out"] = n_timed_out
        get_monitor().log(metrics, step=step)
        envs_str = ", ".join(
            f"{e}=r:{metrics[f'eval/{e}/reward/mean']:.3f}/p@k:{metrics[f'eval/{e}/pass_at_k']:.3f}"
            for e in per_env
            if f"eval/{e}/reward/mean" in metrics
        )
        suffix = f" | Timed out: {n_timed_out}" if n_timed_out else ""
        self.logger.success(
            f"Eval @ step {step} | Reward: {metrics['eval/reward/mean']:.4f} | "
            f"Pass@k: {metrics['eval/pass_at_k']:.3f} | "
            f"N: {metrics['eval/n_rollouts']} rollouts / {metrics['eval/n_examples']} examples | "
            f"Envs: {envs_str}{suffix}"
        )

    async def _maybe_save_ckpt(self) -> None:
        ckpt_interval = self.config.ckpt.interval if self.config.ckpt else None
        if not self.ckpt_manager or not ckpt_interval:
            return
        if self.step <= 0 or self.step % ckpt_interval != 0:
            return
        if self.config.max_steps is not None and self.step >= self.config.max_steps:
            return
        last_eval = self.eval_group.last_eval_step if self.eval_group is not None else 0
        group_states = {g.name: g.state_dict() for g in self.train_groups}
        state = OrchState(step=self.step, last_eval_step=last_eval, group_states=group_states)
        await asyncio.to_thread(self.ckpt_manager.save, state, self.step)

    async def run(self) -> None:
        max_steps = self.config.max_steps
        while True:
            traj = await self.in_q.get()
            self._stamp_version(traj)
            if traj.kind == "eval":
                self._handle_eval(traj)
                continue
            apply_filters(self.filters, traj.rollouts)
            self.strategy.add(traj.rollouts)
            while self.strategy.has_batch():
                await self._wait_barrier()
                trainable, filtered = self.strategy.pop()
                await self.post.process(trainable, filtered, self.step)
                group_metrics: dict = {}
                for g in self.train_groups:
                    group_metrics.update(g.metrics())
                if group_metrics:
                    get_monitor().log(group_metrics, step=self.step)
                if self.inference_metrics is not None:
                    inf_metrics = await self.inference_metrics.collect()
                    if inf_metrics:
                        get_monitor().log(inf_metrics, step=self.step)
                if self.heartbeat is not None:
                    self.heartbeat.beat()
                self.step += 1
                await self._maybe_save_ckpt()
                if max_steps is not None and self.step >= max_steps:
                    raise Done()


def setup_batcher(
    config: OrchestratorConfig,
    *,
    in_q: asyncio.Queue[Trajectory],
    tokenizer: Any,
    policy: Policy,
    eval_group: EvalGroup | None = None,
    train_groups: list[GRPOGroup] | None = None,
    ckpt_manager: CkptManager | None = None,
    inference_metrics: InferenceMetricsCollector | None = None,
) -> TrainBatcher:
    """Thin wrapper around `TrainBatcher` that builds the transport sender."""
    sender = setup_training_batch_sender(config.output_dir, config.rollout_transport)
    return TrainBatcher(
        config,
        in_q=in_q,
        tokenizer=tokenizer,
        sender=sender,
        policy=policy,
        eval_group=eval_group,
        train_groups=train_groups,
        ckpt_manager=ckpt_manager,
        inference_metrics=inference_metrics,
    )
