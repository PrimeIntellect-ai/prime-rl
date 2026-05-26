"""TrainBatcher: drains the dispatcher's queue, builds batches, ships to trainer.

Pipeline per ``Trajectory``:

- ``kind == "eval"``: apply ``post_batch_filters`` for annotation (never drop),
  then bucket by ``eval_step`` and flush per-env eval metrics once the dispatcher's
  expected count for that step has come back.
- ``kind == "train"``: apply ``pre_batch_filters`` (annotate + drop where
  ``enforce=True``) — surviving rollouts go into ``batch_buf``. Once
  ``len(batch_buf) >= batch_size`` (or the token equivalent), wait the async
  barrier (``policy.version >= step - 1``), then ``post_batch_filters`` →
  tokenize → ship → log → increment step.

Signals ``orch.stopped`` when ``progress.step >= max_steps`` so the orchestrator
can drive an orderly shutdown without using exceptions for control flow.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import pandas as pd
import verifiers as vf

from prime_rl.orchestrator.filters import apply_filters
from prime_rl.orchestrator.trajectories import (
    backfill_rollout_tokens,
    interleave_rollout,
    offload_images_to_disk,
)
from prime_rl.orchestrator.utils import compute_teacher_logprobs
from prime_rl.orchestrator.vf_utils import get_seq_len, save_rollouts
from prime_rl.orchestrator_v2.dispatcher import Trajectory
from prime_rl.transport import TrainingBatch, TrainingSample
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_rollout_dir, get_step_path

if TYPE_CHECKING:
    from prime_rl.orchestrator_v2.orchestrator import Orchestrator


class TrainBatcher:
    """Single async task. ``await batcher.start()`` runs until ``stop()`` is called
    (or the batcher signals ``orch.stopped`` after reaching ``max_steps``).

    Takes the parent ``Orchestrator`` instance for shared state — keeps the
    constructor minimal instead of threading 15 individual wiring args through.
    """

    def __init__(self, orch: "Orchestrator") -> None:
        self.orch = orch
        self.config = orch.config
        self.logger = get_logger()

        # Rollout-mode batching (``batch_size``) is the only mode that has a
        # straight rollout count target. Token-mode batching ships when token
        # accumulation crosses a threshold.
        self.batch_size: int | None = self.config.batch_size
        self.token_batch_size: int | None = self.config.token_batch_size

        self.batch_buf: list[vf.RolloutOutput] = []

        # Eval aggregation: bucket trajectories by ``eval_step``; flush when
        # the dispatcher reports the expected count has come back. Mirrors the
        # legacy ``EvalEnv.evaluate`` per-env aggregation but driven by the
        # dispatcher's queue instead of an ``asyncio.gather`` over all examples.
        self.eval_buf: dict[int, list[Trajectory]] = defaultdict(list)
        self.eval_received: dict[int, int] = defaultdict(int)

        # Aggregated pre-filter detection counters reset each batch.
        self.pre_filter_drops_total = 0
        self.pre_filter_drops_by_name: dict[str, int] = defaultdict(int)
        self.pre_filter_rollouts_seen = 0

        # Per-batch step timing — measured between consecutive ships.
        self.last_step_time = time.perf_counter()

        # Cached last batch metrics for IntervalLogger to surface progress
        # gauges in between ship events.
        self.last_batch_step: int | None = None
        self.last_batch_reward: float | None = None
        self.last_batch_size_shipped: int | None = None

        self.stopped = asyncio.Event()
        self.task: asyncio.Task | None = None

    # ── convenience accessors so the body reads naturally ─────────────────

    @property
    def dispatcher(self):
        return self.orch.dispatcher

    @property
    def policy(self):
        return self.orch.policy

    @property
    def progress(self):
        return self.orch.progress

    @property
    def monitor(self):
        return self.orch.monitor

    @property
    def sender(self):
        return self.orch.sender

    @property
    def tokenizer(self):
        return self.orch.tokenizer

    @property
    def renderer(self):
        return self.orch.renderer

    @property
    def mm_token_type_ids_mapping(self):
        return self.orch.mm_token_type_ids_mapping

    @property
    def pre_filters(self):
        return self.orch.pre_filters

    @property
    def post_filters(self):
        return self.orch.post_filters

    @property
    def teacher_inference(self):
        return self.orch.teacher_inference

    @property
    def ckpt_manager(self):
        return self.orch.ckpt_manager

    @property
    def heart(self):
        return self.orch.heart

    @property
    def usage_reporter(self):
        return self.orch.usage_reporter

    # ── public lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Main consumer loop. Runs until ``stop()`` or ``orch.stopped`` fires."""
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set() and not self.orch.stopped.is_set():
                try:
                    traj = await asyncio.wait_for(self.dispatcher.out_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if traj.kind == "eval":
                    self.handle_eval(traj)
                else:
                    await self.handle_train(traj)
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        self.stopped.set()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    # ── train path ────────────────────────────────────────────────────────

    async def handle_train(self, traj: Trajectory) -> None:
        """Apply ``pre_batch_filters`` (annotate+drop) and add survivors to ``batch_buf``."""
        # ``apply_filters`` mutates each rollout in place. Stamp ``policy_version``
        # so post-batch metrics can reflect per-rollout off-policy lag.
        for r in traj.rollouts:
            r["_policy_version"] = traj.policy_version

        if self.pre_filters:
            apply_filters(self.pre_filters, traj.rollouts)
        survivors: list[vf.RolloutOutput] = []
        for r in traj.rollouts:
            self.pre_filter_rollouts_seen += 1
            if r.get("is_filtered"):
                self.pre_filter_drops_total += 1
                for name, hit in (r.get("filters") or {}).items():
                    if hit:
                        self.pre_filter_drops_by_name[name] += 1
                continue
            # Clear filter annotations so post_batch_filters can re-annotate
            # cleanly with their own per-filter detection dict.
            r["filters"] = {}
            r["is_filtered"] = False
            survivors.append(r)

        if not survivors:
            return
        self.batch_buf.extend(survivors)
        while self.batch_ready():
            await self.ship_batch()
            if self.orch.stopped.is_set():
                return

    def batch_ready(self) -> bool:
        """True when ``batch_buf`` has enough material to ship."""
        if self.batch_size is not None:
            return len(self.batch_buf) >= self.batch_size
        if self.token_batch_size is not None:
            return sum(get_seq_len(r) for r in self.batch_buf) >= self.token_batch_size
        return False

    def pop_batch(self) -> list[vf.RolloutOutput]:
        if self.batch_size is not None:
            batch = self.batch_buf[: self.batch_size]
            self.batch_buf = self.batch_buf[self.batch_size :]
            return batch
        # token mode — slice until we hit the budget
        assert self.token_batch_size is not None
        cut = 0
        running = 0
        for i, r in enumerate(self.batch_buf):
            running += get_seq_len(r)
            cut = i + 1
            if running >= self.token_batch_size:
                break
        batch = self.batch_buf[:cut]
        self.batch_buf = self.batch_buf[cut:]
        return batch

    async def wait_barrier(self) -> None:
        """Block until ``policy.version >= step - 1`` so the orchestrator stays
        at most one step ahead of the trainer.

        Cascades backpressure into the dispatcher via the bounded ``out_q``:
        while we wait here, the queue fills, the dispatcher can't push, the
        semaphore stops handing out new permits, and in-flight rollouts drain
        without new ones being scheduled."""
        target_lag = 1
        next_warn = 60.0
        t0 = time.perf_counter()
        while True:
            lead = self.progress.step - self.policy.version
            if lead <= target_lag:
                return
            elapsed = time.perf_counter() - t0
            if elapsed >= next_warn:
                # Just a stall observation — no speculation about cause.
                # Most of the time inference is just faster than the trainer.
                self.logger.info(
                    f"Batcher waiting at async barrier ({int(elapsed)}s): step={self.progress.step}, "
                    f"policy.version={self.policy.version}, lead={lead} (max_async_level={target_lag})."
                )
                next_warn = elapsed + 60.0
            await asyncio.sleep(0.1)

    async def ship_batch(self) -> None:
        step = self.progress.step

        # Save checkpoint (matches legacy: at interval, not on first/last step).
        is_last_step = self.config.max_steps is not None and step == self.config.max_steps - 1
        save_ckpt_time = 0.0
        if (
            self.ckpt_manager is not None
            and self.config.ckpt
            and self.config.ckpt.interval
            and step > 0
            and not is_last_step
            and step % self.config.ckpt.interval == 0
        ):
            self.logger.info(f"Saving v2 checkpoint at step {step}")
            t = time.perf_counter()
            await asyncio.to_thread(self.ckpt_manager.save, self.progress, step)
            save_ckpt_time = time.perf_counter() - t

        # Stop signal if max_steps reached — set the shared event so the
        # orchestrator drives the shutdown. No exception-based control flow.
        if self.config.max_steps is not None and step >= self.config.max_steps:
            self.logger.success(f"Reached max_steps={self.config.max_steps}, signaling shutdown")
            self.orch.stopped.set()
            return

        self.logger.info(f"Starting orchestrator step {step}")
        step_start = time.perf_counter()

        # Pop ``batch_size`` rollouts off the buffer.
        train_rollouts = self.pop_batch()
        num_rollouts = len(train_rollouts)
        num_unique_examples = len({(r["env_name"], r["example_id"]) for r in train_rollouts})

        # Post-batch filter annotation. Mirrors legacy semantics: filtered
        # rollouts stay in the batch for metric aggregation but are excluded
        # from the trainer-bound ``train_examples`` list.
        if self.post_filters:
            await asyncio.to_thread(apply_filters, self.post_filters, train_rollouts)
        else:
            for r in train_rollouts:
                r.setdefault("filters", {})
                r.setdefault("is_filtered", False)

        n_trainable = sum(1 for r in train_rollouts if not r.get("is_filtered"))
        trainable_ratio = n_trainable / num_rollouts if num_rollouts else 0.0
        if n_trainable == 0:
            # Zero trainable signal — drop this batch entirely and loop. The
            # dispatcher keeps feeding new rollouts; eventually one survives.
            self.logger.warning(f"Step {step}: post-batch filters dropped all {num_rollouts} rollouts. Trying again.")
            return
        if trainable_ratio <= 0.1:
            self.logger.warning(
                f"Only {n_trainable}/{num_rollouts} rollouts in the batch are trainable "
                f"({trainable_ratio:.1%}) — consider reviewing task difficulty / filter config"
            )

        # Now block until the trainer is no more than one step behind.
        await self.wait_barrier()

        # Persist rollouts to disk (fire-and-forget background thread).
        step_path = get_step_path(get_rollout_dir(self.config.output_dir), step)
        await asyncio.to_thread(
            save_rollouts, train_rollouts, step_path / "train_rollouts.jsonl", exclude_keys={"trajectory"}
        )

        # Offload base64 image bytes to disk for memory hygiene (no-op for text-only rollouts).
        offload_start = time.perf_counter()
        num_offloaded = offload_images_to_disk(train_rollouts, self.config.output_dir)
        if num_offloaded:
            self.logger.info(
                f"Offloaded {num_offloaded} unique images to disk in {time.perf_counter() - offload_start:.2f}s"
            )

        # Tokenize → TrainingSample list.
        parallel_preprocess_start = time.perf_counter()
        needs_backfill = any(step["tokens"] is None for rollout in train_rollouts for step in rollout["trajectory"])
        if needs_backfill:
            self.logger.info(
                "Backfilling tokens for rollout trajectories (expected for "
                "training_mode=sft against an external teacher API)"
            )
            await asyncio.gather(
                *(
                    asyncio.to_thread(backfill_rollout_tokens, rollout, self.tokenizer, renderer=self.renderer)
                    for rollout in train_rollouts
                )
            )
        interleaved = await asyncio.gather(
            *(
                asyncio.to_thread(interleave_rollout, r, mm_token_type_ids_mapping=self.mm_token_type_ids_mapping)
                for r in train_rollouts
            )
        )

        train_examples: list[TrainingSample] = []
        rollout_prefill_lens: list[int] = []
        rollout_decode_lens: list[int] = []
        samples_per_rollout: list[int] = []
        num_prefill_tokens = 0
        num_decode_tokens = 0
        for rollout, samples in zip(train_rollouts, interleaved):
            prefill = 0
            decode = 0
            if samples is None:
                samples = []
            samples_per_rollout.append(len(samples))
            for sample in samples:
                sample.advantage = rollout.get("advantage")
                sample.reward = rollout.get("reward")
                sample.env_name = rollout.get("env_name")
                sample.training_mode = self.config.training_mode
                sample_decode = sum(sample.completion_mask)
                sample_prefill = len(sample.prompt_ids) + len(sample.completion_mask) - sample_decode
                decode += sample_decode
                prefill += sample_prefill
                if not rollout.get("is_filtered"):
                    train_examples.append(sample)
            rollout_prefill_lens.append(prefill)
            rollout_decode_lens.append(decode)
            num_prefill_tokens += prefill
            num_decode_tokens += decode
        parallel_preprocess_time = time.perf_counter() - parallel_preprocess_start

        # Teacher logprobs (opd only).
        teacher_logprobs_time = 0.0
        if self.config.training_mode == "opd" and self.teacher_inference is not None:
            assert self.config.teacher is not None
            t = time.perf_counter()
            teacher_logprobs_list = await compute_teacher_logprobs(
                clients=self.teacher_inference.train_clients,
                model_name=self.config.teacher.model.name,
                samples=train_examples,
            )
            for ex, lp in zip(train_examples, teacher_logprobs_list):
                ex.teacher_logprobs = lp
            teacher_logprobs_time = time.perf_counter() - t

        # Ship. ``send`` is async (its filesystem implementation does the
        # encode + write in a worker thread internally; wrapping in to_thread
        # would just leave the coroutine unawaited).
        batch = TrainingBatch(examples=train_examples, step=step)
        await self.sender.send(batch)

        # Update progress.
        num_tokens = sum(get_seq_len(r) for r in train_rollouts)
        self.progress.total_tokens += num_tokens
        self.progress.total_samples += num_rollouts
        self.progress.total_problems += num_unique_examples

        # Per-step metrics + W&B log.
        step_time = time.perf_counter() - step_start
        to_log = self.build_metrics(
            step=step,
            train_rollouts=train_rollouts,
            rollout_prefill_lens=rollout_prefill_lens,
            rollout_decode_lens=rollout_decode_lens,
            samples_per_rollout=samples_per_rollout,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_tokens=num_tokens,
            num_rollouts=num_rollouts,
            num_unique_examples=num_unique_examples,
            step_time=step_time,
            parallel_preprocess_time=parallel_preprocess_time,
            teacher_logprobs_time=teacher_logprobs_time,
            save_ckpt_time=save_ckpt_time,
        )
        self.monitor.log(to_log, step=step)
        self.monitor.log_samples(train_rollouts, step=step)
        self.monitor.log_distributions(
            distributions={
                "rewards": [r["reward"] for r in train_rollouts],
                "advantages": [r["advantage"] for r in train_rollouts],
            },
            step=step,
        )

        if self.usage_reporter is not None:
            run_id = os.getenv("RUN_ID", "")
            if run_id:
                self.usage_reporter.report_training_usage(
                    run_id=run_id,
                    step=step,
                    tokens=num_prefill_tokens + num_decode_tokens,
                )

        if self.heart is not None:
            self.heart.beat()

        # Cache for IntervalLogger.
        self.last_batch_step = step
        self.last_batch_reward = float(sum(r["reward"] for r in train_rollouts) / max(num_rollouts, 1))
        self.last_batch_size_shipped = len(train_examples)

        # Reset per-batch pre-filter counters.
        self.pre_filter_drops_total = 0
        self.pre_filter_drops_by_name.clear()
        self.pre_filter_rollouts_seen = 0

        reward_mean = self.last_batch_reward
        message = (
            f"Step {step} | Time: {step_time:.2f}s | Reward: {reward_mean:.4f} | "
            f"Seq. Length: {num_tokens / max(num_rollouts, 1):.1f} tokens/sample | "
            f"Trainable: {n_trainable}/{num_rollouts} | "
            f"Async Level: {step - self.policy.version} | "
            f"Max. Off-Policy Level: {self.dispatcher.max_off_policy_level}"
        )
        self.logger.success(message)

        self.progress.step += 1

    def build_metrics(
        self,
        *,
        step: int,
        train_rollouts: list[vf.RolloutOutput],
        rollout_prefill_lens: list[int],
        rollout_decode_lens: list[int],
        samples_per_rollout: list[int],
        num_prefill_tokens: int,
        num_decode_tokens: int,
        num_tokens: int,
        num_rollouts: int,
        num_unique_examples: int,
        step_time: float,
        parallel_preprocess_time: float,
        teacher_logprobs_time: float,
        save_ckpt_time: float,
    ) -> dict[str, Any]:
        """Assemble the per-step W&B dict. Mirrors the legacy orchestrator's
        metric names byte-for-byte so existing dashboards / alerts keep working.

        Per-rollout pandas aggregations are verbatim from the legacy
        orchestrator; only the source of `scheduler.get_metrics()` and
        `buffer.get_metrics()` is swapped for `dispatcher.{gauges, drain_metrics}`.
        """
        results_df = pd.DataFrame(
            {
                "example_id": [r["example_id"] for r in train_rollouts],
                "env_name": [r["env_name"] for r in train_rollouts],
                "reward": [r["reward"] for r in train_rollouts],
                "is_truncated": [r["is_truncated"] for r in train_rollouts],
                "is_filtered": [r.get("is_filtered", False) for r in train_rollouts],
                "stop_condition": [r.get("stop_condition") for r in train_rollouts],
                "seq_len": [get_seq_len(r) for r in train_rollouts],
                "prefill_len": rollout_prefill_lens,
                "decode_len": rollout_decode_lens,
                "samples_per_rollout": samples_per_rollout,
                "num_turns": [len(r["trajectory"]) for r in train_rollouts],
            }
        )
        metrics_df = pd.DataFrame([(r.get("metrics") or {}) for r in train_rollouts])
        filter_df = pd.DataFrame([(r.get("filters") or {}) for r in train_rollouts])
        timing_df = self.timing_df(train_rollouts)

        def compute_solve_rates(df):
            reward_per_problem = df.groupby(["env_name", "example_id"]).reward.sum()
            solve_none = (reward_per_problem == 0).mean()
            solve_all = (reward_per_problem == self.config.group_size).mean()
            return solve_none, solve_all, 1 - solve_none - solve_all

        by_example = results_df.groupby(["env_name", "example_id"])
        solve_none, solve_all, effective_batch_size = compute_solve_rates(results_df)

        to_log: dict[str, Any] = {
            "progress/tokens": num_tokens,
            "progress/prefill_tokens": num_prefill_tokens,
            "progress/decode_tokens": num_decode_tokens,
            "progress/samples": num_rollouts,
            "progress/problems": num_unique_examples,
            "progress/total_tokens": self.progress.total_tokens,
            "progress/total_samples": self.progress.total_samples,
            "progress/total_problems": self.progress.total_problems,
            "seq_len/all/mean": by_example.seq_len.mean().mean(),
            "seq_len/all/max": by_example.seq_len.mean().max(),
            "seq_len/all/min": by_example.seq_len.mean().min(),
            "prefill_len/all/mean": by_example.prefill_len.mean().mean(),
            "prefill_len/all/max": by_example.prefill_len.mean().max(),
            "prefill_len/all/min": by_example.prefill_len.mean().min(),
            "decode_len/all/mean": by_example.decode_len.mean().mean(),
            "decode_len/all/max": by_example.decode_len.mean().max(),
            "decode_len/all/min": by_example.decode_len.mean().min(),
            "is_truncated/all/mean": by_example.is_truncated.mean().mean(),
            "is_truncated/all/max": by_example.is_truncated.mean().max(),
            "stop_condition/all/generation_truncated": (
                results_df.is_truncated & (results_df.stop_condition != "prompt_too_long")
            ).mean(),
            **{
                f"stop_condition/all/{sc}": rate
                for sc, rate in results_df.stop_condition.dropna().value_counts(normalize=True).items()
            },
            "samples_per_rollout/all/mean": by_example.samples_per_rollout.mean().mean(),
            "samples_per_rollout/all/max": by_example.samples_per_rollout.mean().max(),
            "samples_per_rollout/all/min": by_example.samples_per_rollout.mean().min(),
            "num_turns/all/mean": by_example.num_turns.mean().mean(),
            "num_turns/all/max": by_example.num_turns.mean().max(),
            "num_turns/all/min": by_example.num_turns.mean().min(),
            **{
                f"timing/all/{key}/{stat}": getattr(
                    timing_df[key].groupby([results_df.env_name, results_df.example_id]).mean(),
                    stat,
                )()
                for key in timing_df.columns
                for stat in ("mean", "max", "min")
            },
            "reward/all/mean": by_example.reward.mean().mean(),
            "reward/all/max": by_example.reward.mean().max(),
            "reward/all/min": by_example.reward.mean().min(),
            "solve_none/all": solve_none,
            "solve_all/all": solve_all,
            "effective_batch_size/all": effective_batch_size,
            **{f"batch/{env}": r for env, r in results_df.env_name.value_counts(normalize=True).items()},
            "time/step": step_time,
            "time/teacher_logprobs": teacher_logprobs_time,
            "time/save_ckpt": save_ckpt_time,
            "time/parallel_preprocess": parallel_preprocess_time,
            "filters/all/is_filtered": results_df.is_filtered.astype(float).mean(),
            **{f"filters/all/{name}": filter_df[name].astype(float).mean() for name in filter_df.columns},
            "step": step,
        }

        # Per-env metrics
        per_env_columns = [
            "seq_len",
            "prefill_len",
            "decode_len",
            "is_truncated",
            "samples_per_rollout",
            "num_turns",
        ]
        for env, env_df in results_df.groupby("env_name"):
            env_by_example = env_df.groupby("example_id")
            for col in per_env_columns:
                to_log[f"{col}/{env}/mean"] = env_by_example[col].mean().mean()
                to_log[f"{col}/{env}/max"] = env_by_example[col].mean().max()
                if col != "is_truncated":
                    to_log[f"{col}/{env}/min"] = env_by_example[col].mean().min()
            env_timing_df = timing_df.loc[env_df.index]
            for key in timing_df.columns:
                per_example = env_timing_df.groupby(env_df["example_id"])[key].mean()
                to_log[f"timing/{env}/{key}/mean"] = per_example.mean()
                to_log[f"timing/{env}/{key}/max"] = per_example.max()
                to_log[f"timing/{env}/{key}/min"] = per_example.min()
            to_log[f"reward/{env}/mean"] = env_by_example.reward.mean().mean()
            to_log[f"reward/{env}/max"] = env_by_example.reward.mean().max()
            to_log[f"reward/{env}/min"] = env_by_example.reward.mean().min()
            sn, sa, eb = compute_solve_rates(env_df)
            to_log[f"solve_none/{env}"] = sn
            to_log[f"solve_all/{env}"] = sa
            to_log[f"effective_batch_size/{env}"] = eb
            to_log[f"stop_condition/{env}/generation_truncated"] = (
                env_df.is_truncated & (env_df.stop_condition != "prompt_too_long")
            ).mean()
            for sc, rate in env_df.stop_condition.dropna().value_counts(normalize=True).items():
                to_log[f"stop_condition/{env}/{sc}"] = rate
            env_metrics_df = metrics_df.loc[env_df.index] if not metrics_df.empty else metrics_df
            for metric in metrics_df.columns:
                to_log[f"metrics/{env}/{metric}"] = env_metrics_df.groupby(env_df["example_id"])[metric].mean().mean()
            to_log[f"filters/{env}/is_filtered"] = env_df.is_filtered.astype(float).mean()
            env_filter_df = filter_df.loc[env_df.index] if not filter_df.empty else filter_df
            for name in filter_df.columns:
                to_log[f"filters/{env}/{name}"] = env_filter_df[name].astype(float).mean()

        # Dispatcher gauges + drained counters (replaces legacy scheduler+buffer metrics blocks).
        to_log.update(self.dispatcher.gauges())
        to_log.update(self.dispatcher.drain_metrics())

        # Pre-batch filter detection rates (per batch).
        if self.pre_filter_rollouts_seen > 0:
            to_log["pre_filters/all/dropped_rate"] = self.pre_filter_drops_total / self.pre_filter_rollouts_seen
            for name, count in self.pre_filter_drops_by_name.items():
                to_log[f"pre_filters/all/{name}/rate"] = count / self.pre_filter_rollouts_seen

        return to_log

    @staticmethod
    def timing_df(train_rollouts: list[vf.RolloutOutput]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "total": r["timing"]["total"],
                    "setup": r["timing"]["setup"]["duration"],
                    "generation": r["timing"]["generation"]["duration"],
                    "model": r["timing"]["model"]["duration"],
                    "env": r["timing"]["env"]["duration"],
                    "scoring": r["timing"]["scoring"]["duration"],
                    "overhead": r["timing"]["overhead"],
                }
                for r in train_rollouts
            ]
        )

    # ── eval path ─────────────────────────────────────────────────────────

    def handle_eval(self, traj: Trajectory) -> None:
        """Bucket eval trajectories by ``eval_step`` and flush per-env metrics
        once the dispatcher's expected count comes back."""
        assert traj.eval_step is not None, "eval Trajectory missing eval_step"

        if self.post_filters:
            apply_filters(self.post_filters, traj.rollouts)

        self.eval_buf[traj.eval_step].append(traj)
        self.eval_received[traj.eval_step] += len(traj.rollouts)

        expected = self.dispatcher.expected_eval_counts.get(traj.eval_step)
        if expected is None or self.eval_received[traj.eval_step] < expected:
            return

        trajs = self.eval_buf.pop(traj.eval_step)
        received = self.eval_received.pop(traj.eval_step)
        envs_fired = self.dispatcher.eval_step_envs.pop(traj.eval_step, set())
        self.dispatcher.expected_eval_counts.pop(traj.eval_step, None)
        self.progress.last_eval_step = traj.eval_step

        self.flush_eval(traj.eval_step, trajs, expected, received, envs_fired)

    def flush_eval(
        self,
        eval_step: int,
        trajs: list[Trajectory],
        expected: int,
        received: int,
        envs_fired: set[str],
    ) -> None:
        per_env: dict[str, list[Trajectory]] = defaultdict(list)
        for t in trajs:
            per_env[t.env_name].append(t)

        to_log: dict[str, Any] = {"step": eval_step}
        all_rewards: list[float] = []
        all_lens: list[int] = []

        from prime_rl.orchestrator.eval_utils import compute_pass_at_k

        for env_name, env_trajs in per_env.items():
            rollouts = [r for t in env_trajs for r in t.rollouts]
            if not rollouts:
                continue
            rewards = [r.get("reward", 0.0) for r in rollouts]
            lens = [get_seq_len(r) for r in rollouts]
            all_rewards.extend(rewards)
            all_lens.extend(lens)

            no_response_rate = sum(1 for r in rollouts if not r.get("completion")) / len(rollouts)
            truncation_rate = sum(1 for r in rollouts if r.get("is_truncated")) / len(rollouts)
            prefix = f"eval/{env_name}"
            group_size = 1
            if self.dispatcher.eval_envs is not None:
                try:
                    group_size = self.dispatcher.eval_envs.get(env_name).config.group_size
                except KeyError:
                    pass
            to_log[f"{prefix}/avg@{group_size}"] = float(sum(rewards) / len(rewards))
            to_log[f"{prefix}/reward/mean"] = float(sum(rewards) / len(rewards))
            to_log[f"{prefix}/completion_len/mean"] = float(sum(lens) / len(lens))
            to_log[f"{prefix}/completion_len/max"] = float(max(lens))
            to_log[f"{prefix}/completion_len/min"] = float(min(lens))
            to_log[f"{prefix}/is_truncated/mean"] = float(truncation_rate)
            to_log[f"{prefix}/no_response/mean"] = float(no_response_rate)
            to_log[f"{prefix}/n_rollouts"] = float(len(rollouts))
            to_log[f"{prefix}/n_examples"] = float(len(env_trajs))

            unique_rewards = {float(r) for r in rewards}
            could_be_binary = unique_rewards.issubset({0.0, 1.0})
            if could_be_binary:
                per_example_rewards = [[float(r.get("reward", 0.0)) for r in t.rollouts] for t in env_trajs]
                pass_at_k_per_example = [compute_pass_at_k(rs) for rs in per_example_rewards]
                if pass_at_k_per_example:
                    keys = set().union(*(d.keys() for d in pass_at_k_per_example))
                    for k in keys:
                        values = [d.get(k, 0.0) for d in pass_at_k_per_example]
                        to_log[f"{prefix}/{k}"] = float(sum(values) / len(values))

            step_path = get_step_path(get_rollout_dir(self.config.output_dir), eval_step)
            asyncio.create_task(
                asyncio.to_thread(
                    save_rollouts,
                    rollouts,
                    step_path / "eval_rollouts.jsonl",
                    exclude_keys={"trajectory"},
                )
            )

            self.monitor.log_eval_samples(rollouts, env_name=env_name, step=eval_step)

        if all_rewards:
            to_log["eval/reward/mean"] = float(sum(all_rewards) / len(all_rewards))
            to_log["eval/completion_len/mean"] = float(sum(all_lens) / len(all_lens))
            to_log["eval/n_rollouts"] = float(len(all_rewards))

        envs_str = ",".join(sorted(envs_fired))
        coverage = f"{received}/{expected}"
        self.logger.success(
            f"Eval @ step={eval_step} | Envs: {envs_str} | "
            f"Reward: {to_log.get('eval/reward/mean', float('nan')):.4f} | "
            f"Rollouts: {coverage}"
        )
        self.monitor.log(to_log, step=eval_step)
