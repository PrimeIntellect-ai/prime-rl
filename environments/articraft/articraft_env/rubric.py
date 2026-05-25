"""Articraft rubric — compiler QC signals as reward.

Three-dimensional weighted reward:
  final_reward = 0.7 * check_fraction + 0.2 * build_success + 0.1 * compile_attempted

``compute_reward()`` maps CompileSignalBundle to a continuous [0, 1] value with
10+ distinct levels, providing dense gradient signal for GRPO.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import verifiers as vf

from agent.models import CompileSignalBundle

from .artifact_manager import ArticraftArtifactManager
from .schema import require_rollout

logger = logging.getLogger(__name__)

COMPILER_QC_CHECKS = frozenset({"isolated_part", "real_overlap"})

DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "check_fraction": 0.7,
    "build_success": 0.2,
    "compile_attempted": 0.1,
}


def compute_reward(
    bundle: CompileSignalBundle | None,
    *,
    num_task_test_checks: int = 0,
    turns_used: int = 0,
    max_turns: int = 50,
) -> float:
    """Map a CompileSignalBundle to a continuous reward in [0, 1].

    Hierarchy (low → high):
      never compiled            → 0.00
      SyntaxError               → 0.05
      RuntimeError              → 0.10
      structural failure        → 0.15
      build ok + QC failures    → 0.30 – 0.80  (linear in passed/total)
      all QC + warnings         → 0.80 – 0.90
      all QC + no warnings      → 0.90 – 1.00  (efficiency bonus)
    """
    if bundle is None:
        return 0.0

    blocking = [s for s in bundle.signals if s.blocking]
    build_failures = [s for s in blocking if s.group == "build"]

    if build_failures:
        kinds = {s.kind for s in build_failures}
        if "syntax_error" in kinds:
            return 0.05
        if "model_valid" not in kinds:
            return 0.10
        return 0.15

    total_qc = len(COMPILER_QC_CHECKS) + num_task_test_checks
    qc_failures = [s for s in blocking if s.group == "qc"]
    qc_score = (total_qc - len(qc_failures)) / total_qc if total_qc > 0 else 1.0

    if qc_score < 1.0:
        return 0.3 + 0.5 * qc_score

    warnings = [s for s in bundle.signals if s.severity == "warning"]
    if warnings:
        return 0.8 + 0.1 * max(0.0, 1.0 - len(warnings) / 5)

    efficiency = max(0.0, 1.0 - turns_used / max_turns) if max_turns > 0 else 1.0
    return 0.9 + 0.1 * efficiency


class ArticraftRubric(vf.Rubric):
    """Compiler QC reward + rule-based bonuses."""

    def __init__(
        self,
        artifact_manager: ArticraftArtifactManager,
        reward_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.artifact_manager = artifact_manager

        w = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        self.add_reward_func(self.check_fraction_reward, weight=w["check_fraction"])
        self.add_reward_func(self.build_success_bonus, weight=w["build_success"])
        self.add_reward_func(self.compile_attempted_bonus, weight=w["compile_attempted"])

        self.add_metric(self.blocking_failure_count)
        self.add_metric(self.warning_count)
        self.add_metric(self.turns_used)
        self.add_metric(self.compile_latency_ms)
        self.add_metric(self.trajectory_token_estimate)

    # ---- reward functions ----

    async def check_fraction_reward(self, state: vf.State, **kwargs: Any) -> float:
        rollout = require_rollout(state)
        bundle_dict = rollout.last_compile_attempt_dict
        if bundle_dict is None:
            return 0.0
        bundle = CompileSignalBundle.from_dict(bundle_dict)
        return compute_reward(
            bundle,
            turns_used=len(rollout.turns),
            max_turns=rollout.max_turns,
        )

    async def build_success_bonus(self, state: vf.State, **kwargs: Any) -> float:
        rollout = require_rollout(state)
        bundle_dict = rollout.last_compile_attempt_dict
        if bundle_dict is None:
            return 0.0
        bundle = CompileSignalBundle.from_dict(bundle_dict)
        build_fails = [
            s for s in bundle.signals if s.blocking and s.group == "build"
        ]
        return 0.0 if build_fails else 1.0

    async def compile_attempted_bonus(self, state: vf.State, **kwargs: Any) -> float:
        rollout = require_rollout(state)
        return 1.0 if any(t.compile_attempted for t in rollout.turns) else 0.0

    # ---- metrics ----

    async def blocking_failure_count(self, state: vf.State, **kwargs: Any) -> float:
        rollout = require_rollout(state)
        bd = rollout.last_compile_attempt_dict
        if bd is None:
            return 0.0
        bundle = CompileSignalBundle.from_dict(bd)
        return float(sum(1 for s in bundle.signals if s.blocking))

    async def warning_count(self, state: vf.State, **kwargs: Any) -> float:
        rollout = require_rollout(state)
        bd = rollout.last_compile_attempt_dict
        if bd is None:
            return 0.0
        bundle = CompileSignalBundle.from_dict(bd)
        return float(sum(1 for s in bundle.signals if s.severity == "warning"))

    async def turns_used(self, state: vf.State, **kwargs: Any) -> float:
        rollout = require_rollout(state)
        return float(len(rollout.turns))

    async def compile_latency_ms(self, state: vf.State, **kwargs: Any) -> float:
        rollout = require_rollout(state)
        return rollout.last_compile_latency_ms or 0.0

    async def trajectory_token_estimate(self, state: vf.State, **kwargs: Any) -> float:
        """Rough token estimate for context pressure tracking."""
        trajectory = state.get("trajectory", [])
        text = json.dumps(trajectory, default=str)
        return len(text) / 4.0

    # ---- cleanup ----

    @vf.cleanup
    async def write_artifacts_handler(self, state: vf.State) -> None:
        """Write trajectory artifacts and apply retention policy.

        Runs *after* ``score_rollout``, so ``rollout.final_reward`` and
        ``state["metrics"]`` are already populated.
        """
        try:
            rollout = require_rollout(state)
        except RuntimeError:
            return

        mgr = self.artifact_manager
        try:
            mgr.save_trajectory(rollout, metrics=state.get("metrics"))
        except Exception:
            logger.exception(
                "save_trajectory failed for work_dir=%s", rollout.work_dir
            )
        mgr.cleanup_rollout(rollout)
        mgr.prune_old_rollouts(rollout)
