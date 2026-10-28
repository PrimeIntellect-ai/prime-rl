"""Checkpoint manager for orchestrator progress and train-source state. Layout:
``<output_dir>/checkpoints/step_N/orchestrator/progress.pt``."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Collection, Iterable
from dataclasses import asdict
from pathlib import Path

import torch

from prime_rl.configs.orchestrator import CheckpointConfig
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import EvalBatch, Progress
from prime_rl.orchestrator.utils import eval_work
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import (
    checkpoint_lifecycle_lock,
    complete_joint_ckpt_step,
    get_ckpt_dir,
    get_resume_ckpt_generations,
    get_retention_path,
    get_step_path,
    invalidate_joint_ckpt_step,
    preflight_checkpoint_storage,
    prune_joint_ckpts,
    read_checkpoint_generation,
)


class CheckpointManager:
    def __init__(self, output_dir: Path, config: CheckpointConfig) -> None:
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self.generation = read_checkpoint_generation(self.ckpt_dir)["generation"]
        self.prune_failures: set[int] = set()

    def get_ckpt_path(self, step: int) -> Path:
        return get_step_path(self.ckpt_dir, step) / "orchestrator"

    def save(self, progress: Progress, train_source: TrainSource, step: int) -> None:
        self.save_state(build_state(progress, train_source), step)

    def save_state(self, state: dict, step: int) -> None:
        """Save a ``build_state`` snapshot as the checkpoint of ``step``."""
        ckpt_path = self.get_ckpt_path(step)
        preflight_checkpoint_storage(self.ckpt_dir)
        invalidate_joint_ckpt_step(self.ckpt_dir, step, component="orchestrator")
        (ckpt_path / "progress.pt").unlink(missing_ok=True)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        # Save to a temporary file and do an atomic rename, to avoid corrupting the last
        # file if the process gets killed while writing
        fd, tmp_name = tempfile.mkstemp(dir=ckpt_path, prefix="progress.pt.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(state, f)
            os.replace(tmp_name, ckpt_path / "progress.pt")
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        get_logger().debug(
            f"Orchestrator checkpoint saved to {ckpt_path} in {format_time(time.perf_counter() - start)}"
        )
        complete_joint_ckpt_step(self.ckpt_dir, step, component="orchestrator", generation=self.generation)

    def maybe_clean(self) -> None:
        prune_joint_ckpts(
            self.ckpt_dir,
            keep_last=self.config.keep_last,
            keep_interval=self.config.keep_interval,
            generation=self.generation,
            failed_steps=self.prune_failures,
        )

    def load(self, progress: Progress, train_source: TrainSource, step: int, path: Path | None = None) -> None:
        """``path`` overrides where the checkpoint is read from (an external run's
        ``step_<N>/orchestrator``)."""
        ckpt_path = path if path is not None else self.get_ckpt_path(step)
        state_file = ckpt_path / "progress.pt"
        if not state_file.exists():
            raise FileNotFoundError(f"Orchestrator checkpoint not found at {state_file}")
        get_logger().debug(f"Loading checkpoint from {state_file}")
        start = time.perf_counter()
        with open(state_file, "rb") as f:
            state = torch.load(f, weights_only=False)
        saved: Progress = state["progress"]
        for key, value in asdict(saved).items():
            if hasattr(progress, key):
                setattr(progress, key, value)
        train_source.load_state_dict(state["train_source"])
        for name in state["train_source"]["envs"]:
            if name in train_source.curricula:
                get_logger().info(f"Resumed curriculum state for env {name}")
        get_logger().debug(f"Orchestrator checkpoint loaded in {format_time(time.perf_counter() - start)}")


def eval_epoch_score(batch: EvalBatch, *, expected_tasks: int, group_size: int) -> float | None:
    """Task-macro continuous reward, or None unless the full epoch is BEST-eligible."""
    if batch.failures or batch.cancelled or len(batch.episodes) != expected_tasks * group_size:
        return None
    rewards_by_task: dict[str, list[float]] = defaultdict(list)
    for episode in batch.episodes:
        work = eval_work(episode)
        policy = work.policy
        if (
            work.step != batch.step
            or policy is None
            or policy.start != batch.step
            or policy.end != batch.step
            or not episode.ok
            or any(trace.has_error for trace in episode.traces)
        ):
            return None
        rewards = [
            reward.value for trace in episode.traces if (reward := trace.rewards.get("continuous_reward")) is not None
        ]
        if len(rewards) != 1 or not math.isfinite(reward := float(rewards[0])):
            return None
        rewards_by_task[episode.task.key].append(reward)
    if len(rewards_by_task) != expected_tasks or any(len(values) != group_size for values in rewards_by_task.values()):
        return None
    return sum(sum(values) / group_size for values in rewards_by_task.values()) / expected_tasks


def select_best_eval(
    valid_scores: dict[tuple[str, int], float],
    *,
    joint_steps: Collection[int],
    required_envs: Collection[str],
) -> tuple[int, float] | None:
    """Select the highest-scoring fully evaluated joint step; earliest wins an exact tie."""
    if not required_envs:
        return None
    scores_by_step: dict[int, dict[str, float]] = defaultdict(dict)
    for (env_name, step), score in valid_scores.items():
        if step in joint_steps and env_name in required_envs and math.isfinite(score):
            scores_by_step[step][env_name] = score
    candidates = [
        (step, sum(scores.values()) / len(required_envs))
        for step, scores in scores_by_step.items()
        if set(scores) == set(required_envs)
    ]
    if not candidates:
        return None
    step, score = min(candidates, key=lambda item: (-item[1], item[0]))
    return step, score


def eval_plan_fingerprint(eval_env) -> str:
    """Version dedupe/BEST by resolved eval behavior and the exact ordered task population."""
    payload = {
        "config": eval_env.config.model_dump(mode="json"),
        "tasks": [(task.key, task.hash) for task in eval_env.examples],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class CheckpointRetention:
    """Atomic generation/plan-versioned eval journal and BEST/pending pins."""

    def __init__(self, ckpt_dir: Path, *, eval_envs: Iterable[str]) -> None:
        self.ckpt_dir = ckpt_dir
        self.eval_envs = frozenset(eval_envs)
        self.generation = read_checkpoint_generation(ckpt_dir)["generation"]
        self.fingerprints: dict[str, str] = {}
        self.path = get_retention_path(ckpt_dir)
        self.state = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": 2, "best_step": None, "best_score": None, "pending_eval": {}, "valid_eval": {}}
        state = json.loads(self.path.read_text())
        if not isinstance(state.get("pending_eval"), dict) or not isinstance(state.get("valid_eval"), dict):
            raise ValueError(f"Invalid checkpoint retention journal at {self.path}")
        return state

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.path.parent, prefix=".retention.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.state, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    @property
    def best_step(self) -> int | None:
        return self.state.get("best_step")

    def configure_eval(self, fingerprints: dict[str, str]) -> None:
        self.fingerprints = fingerprints
        self.reconcile()

    def _matches_context(self, env_name: str, record) -> bool:
        return (
            isinstance(record, dict)
            and isinstance(record.get("generation"), str)
            and bool(record["generation"])
            and record.get("fingerprint") == self.fingerprints.get(env_name)
            and isinstance(record.get("score"), (int, float))
            and math.isfinite(record["score"])
        )

    def _matches_checkpoint(self, env_name: str, step: int, record, joint_generations: dict[int, str]) -> bool:
        return self._matches_context(env_name, record) and joint_generations.get(step) == record.get("generation")

    def valid_eval_envs(self, step: int) -> set[str]:
        joint_generations = get_resume_ckpt_generations(self.ckpt_dir)
        return {
            name
            for name, record in self.state["valid_eval"].get(str(step), {}).items()
            if self._matches_checkpoint(name, step, record, joint_generations)
        }

    def mark_eval_pending(self, step: int, env_names: Iterable[str]) -> None:
        with checkpoint_lifecycle_lock(self.ckpt_dir):
            self.state = self._load()
            pending = self.state["pending_eval"].setdefault(str(step), [])
            pending.extend(name for name in env_names if name not in pending)
            self._write()

    def complete_eval_epoch(self, batch: EvalBatch, *, score: float | None) -> None:
        with checkpoint_lifecycle_lock(self.ckpt_dir):
            self.state = self._load()
            step_key = str(batch.step)
            joint_generations = get_resume_ckpt_generations(self.ckpt_dir)
            if score is not None:
                if batch.env_name not in self.fingerprints:
                    raise RuntimeError(f"Eval plan fingerprint is not configured for {batch.env_name}")
                self.state["valid_eval"].setdefault(step_key, {})[batch.env_name] = {
                    "fingerprint": self.fingerprints[batch.env_name],
                    # An eval of a resume ancestor belongs to that checkpoint's generation.
                    "generation": joint_generations.get(batch.step, self.generation),
                    "score": score,
                }
            pending = self.state["pending_eval"].get(step_key, [])
            joint = batch.step in joint_generations
            if batch.env_name in pending and (score is None or joint):
                pending.remove(batch.env_name)
                if not pending:
                    self.state["pending_eval"].pop(step_key, None)
            self._reconcile_state()
            self._write()

    def reconcile(self) -> None:
        with checkpoint_lifecycle_lock(self.ckpt_dir):
            self.state = self._load()
            before = json.dumps(self.state, sort_keys=True)
            self._reconcile_state()
            if json.dumps(self.state, sort_keys=True) != before:
                self._write()

    def _reconcile_state(self) -> None:
        joint_generations = get_resume_ckpt_generations(self.ckpt_dir)
        joint_steps = set(joint_generations)
        valid_scores: dict[tuple[str, int], float] = {}
        for step_key, epochs in self.state["valid_eval"].items():
            step = int(step_key)
            for env_name, record in epochs.items():
                if self._matches_checkpoint(env_name, step, record, joint_generations):
                    valid_scores[(env_name, step)] = float(record["score"])
                    if env_name in self.state["pending_eval"].get(step_key, []):
                        self.state["pending_eval"][step_key].remove(env_name)
            if not self.state["pending_eval"].get(step_key):
                self.state["pending_eval"].pop(step_key, None)
        best = select_best_eval(valid_scores, joint_steps=joint_steps, required_envs=self.eval_envs)
        self.state["best_step"], self.state["best_score"] = best if best is not None else (None, None)


def should_export_final(ckpt: CheckpointConfig | None, final_step: int | None) -> bool:
    """Final HF export is strictly opt-in and requires a completed final checkpoint."""
    return final_step is not None and ckpt is not None and ckpt.export


def build_state(progress: Progress, train_source: TrainSource) -> dict:
    """A snapshot of the orchestrator's checkpointed state, independent of later changes."""
    return copy.deepcopy({"progress": progress, "train_source": train_source.state_dict()})


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CheckpointManager:
    """The checkpoint manager always exists: ``resume`` decides whether it loads,
    ``ckpt`` whether it saves (a resume without ``ckpt`` loads but saves nothing)."""
    return CheckpointManager(output_dir, config or CheckpointConfig())
