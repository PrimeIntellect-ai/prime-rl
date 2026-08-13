"""Generation-pathology detections — rollout-level sample hygiene.

Each detection is a rollout-granularity predicate over the rollout's own
branches (token ids + logprobs), evaluated once at group finalization — the
pipeline's single decision point. Detection results are always recorded; when
``enforce=True`` a detected rollout ships no training samples and never enters
the batch (the batch backfills from fresh groups). Its reward is kept as-is so
it still counts toward the group baseline — the failure signal is real even
when the tokens are poisoned.

Task-level admission (which tasks to roll out, which finalized groups count)
is the task sampler's job; detections guard the one thing selection cannot:
the policy melting down mid-sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from prime_rl.configs.orchestrator import DetectionConfig
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


@dataclass
class DetectionResult:
    detected: bool


class RolloutDetection(Protocol):
    name: str
    enforce: bool

    def check(self, rollout: Rollout) -> DetectionResult: ...


@dataclass
class GibberishDetection:
    """Flags rollouts containing rare tokens generated at high entropy.

    A token is flagged when both:
      - id(token) > token_id_threshold  (rare BPE token)
      - logprob(token) < -log(vocab_size) - logprob_offset  (high entropy)

    References:
      Section 5.2, https://arxiv.org/abs/2510.02387
    """

    name: str
    token_id_threshold: int
    logprob_threshold: float
    enforce: bool = False

    def check(self, rollout: Rollout) -> DetectionResult:
        for branch in rollout.branches:
            # branch.{token_ids,logprobs,sampled_mask} are flat and mutually aligned; the raw
            # node arrays are not (node.logprobs covers only the sampled suffix, not the
            # generation-prompt scaffold that token_ids/mask also span).
            for token_id, logprob, sampled in zip(branch.token_ids, branch.logprobs, branch.sampled_mask):
                if not sampled:
                    continue
                if token_id > self.token_id_threshold and logprob < self.logprob_threshold:
                    return DetectionResult(detected=True)
        return DetectionResult(detected=False)


@dataclass
class RepetitionDetection:
    """Flags rollouts with pathological repetition loops.

    Counts consecutive tokens where logprob > log(prob_threshold), indicating
    the model is generating with very high confidence. When the streak reaches
    the window size, the rollout is flagged.

    References:
      Section 3.2, https://arxiv.org/abs/2506.13585
    """

    name: str
    window: int
    logprob_threshold: float
    enforce: bool = False

    def check(self, rollout: Rollout) -> DetectionResult:
        for branch in rollout.branches:
            # Aligned branch streams (see GibberishDetection), and reset the streak per branch:
            # flat rollout.nodes interleaves distinct root->leaf paths (compaction/subagents),
            # so a per-node walk would run a streak across a branch boundary.
            consecutive = 0
            for logprob, sampled in zip(branch.logprobs, branch.sampled_mask):
                if not sampled:
                    continue
                if logprob > self.logprob_threshold:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive >= self.window:
                    return DetectionResult(detected=True)
        return DetectionResult(detected=False)


def setup_detection(config: DetectionConfig, vocab_size: int) -> RolloutDetection:
    """Create a RolloutDetection from a detection config."""
    if config.type == "gibberish":
        return GibberishDetection(
            name="gibberish",
            token_id_threshold=config.token_id_threshold,
            logprob_threshold=-math.log(vocab_size) - config.logprob_offset,
            enforce=config.enforce,
        )
    elif config.type == "repetition":
        return RepetitionDetection(
            name="repetition",
            window=config.window,
            logprob_threshold=math.log(config.prob_threshold),
            enforce=config.enforce,
        )
    raise ValueError(f"Unknown detection type: {config.type}")


def setup_detections(configs: list[DetectionConfig], vocab_size: int) -> list[RolloutDetection]:
    """Create RolloutDetections from a list of detection configs."""
    detections = [setup_detection(config, vocab_size) for config in configs]
    if detections:
        get_logger().info(f"Configured {len(detections)} rollout detection(s):")
        for config, detection in zip(configs, detections):
            mode = "Enforcing" if detection.enforce else "Monitoring"
            params = ", ".join(f"{k}={v}" for k, v in config.model_dump().items())
            get_logger().info(f"  {mode} {detection.name} detection ({params})")
    return detections


def run_detections(detections: list[RolloutDetection], rollout: Rollout) -> None:
    """Stamp one rollout in place with per-detection results + the exclusion
    verdict: ``rollout.detections`` records per-name bools; ``is_excluded`` is
    True iff an enforcing detection fired. First match wins (no double
    counting). Reward and trajectory tokens are left untouched so the rollout
    still contributes to baseline calculations and metric aggregation."""
    rollout.detections = {d.name: False for d in detections}
    rollout.is_excluded = False
    for detection in detections:
        if detection.check(rollout).detected:
            rollout.detections[detection.name] = True
            if detection.enforce:
                rollout.is_excluded = True
            break
