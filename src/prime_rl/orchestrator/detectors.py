"""Degeneracy detectors: per-trace measurements of pathological generation.

A detector answers one question about one trace's tokens — is it gibberish, is it stuck in a
repetition loop — and nothing else. Every configured detector runs on every trace as soon as it is
tokenized, so its rate is a trace metric like reward or truncation, reported per agent whether or
not anything acts on it.

What to *do* about a detection is a separate decision, made once when the batch is assembled
(``prime_rl.orchestrator.train_sink``). Keeping the two apart is what lets every detector measure
every trace: a policy that stops at the first hit would leave the rest unmeasured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from prime_rl.configs.orchestrator import DetectorsConfig
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout


class Detector(Protocol):
    name: str

    def detect(self, rollout: TrainRollout) -> bool: ...


@dataclass
class GibberishDetector:
    """Rare tokens generated at high entropy.

    A token counts when both:
      - id(token) > token_id_threshold  (rare BPE token)
      - logprob(token) < -log(vocab_size) - logprob_offset  (high entropy)

    References:
      Section 5.2, https://arxiv.org/abs/2510.02387
    """

    name: str
    token_id_threshold: int
    logprob_threshold: float

    def detect(self, rollout: TrainRollout) -> bool:
        for branch in rollout.branches:
            # branch.{token_ids,logprobs,sampled_mask} are flat and mutually aligned; the raw
            # node arrays are not (node.logprobs covers only the sampled suffix, not the
            # generation-prompt scaffold that token_ids/mask also span).
            for token_id, logprob, sampled in zip(branch.token_ids, branch.logprobs, branch.sampled_mask):
                if sampled and token_id > self.token_id_threshold and logprob < self.logprob_threshold:
                    return True
        return False


@dataclass
class RepetitionDetector:
    """A repetition loop: a long stretch of very-high-confidence tokens.

    Counts consecutive tokens with logprob > log(prob_threshold); a streak reaching ``window``
    is a detection.

    References:
      Section 3.2, https://arxiv.org/abs/2506.13585
    """

    name: str
    window: int
    logprob_threshold: float

    def detect(self, rollout: TrainRollout) -> bool:
        for branch in rollout.branches:
            # Aligned branch streams (see GibberishDetector), and reset the streak per branch:
            # flat rollout.nodes interleaves distinct root->leaf paths (compaction/subagents),
            # so a per-node walk would run a streak across a branch boundary.
            consecutive = 0
            for logprob, sampled in zip(branch.logprobs, branch.sampled_mask):
                if not sampled:
                    continue
                consecutive = consecutive + 1 if logprob > self.logprob_threshold else 0
                if consecutive >= self.window:
                    return True
        return False


def setup_detectors(config: DetectorsConfig, vocab_size: int) -> list[Detector]:
    detectors: list[Detector] = []
    if config.gibberish is not None:
        detectors.append(
            GibberishDetector(
                name="gibberish",
                token_id_threshold=config.gibberish.token_id_threshold,
                logprob_threshold=-math.log(vocab_size) - config.gibberish.logprob_offset,
            )
        )
    if config.repetition is not None:
        detectors.append(
            RepetitionDetector(
                name="repetition",
                window=config.repetition.window,
                logprob_threshold=math.log(config.repetition.prob_threshold),
            )
        )
    if detectors:
        get_logger().info(f"Measuring {len(detectors)} degeneracy detector(s): {', '.join(d.name for d in detectors)}")
        if config.drop:
            get_logger().info(f"Dropping detected rollouts: {', '.join(sorted(config.drop))}")
    return detectors


def detect(detectors: list[Detector], rollout: TrainRollout) -> None:
    """Measure every detector on one trace, writing the verdicts to ``rollout.detections``."""
    rollout.detections = {d.name: d.detect(rollout) for d in detectors}


def drop_reasons(rollout: TrainRollout, *, drop_detections: list[str], drop_zero_advantage: bool) -> list[str]:
    """Why this rollout should not be trained on, if anything.

    A detection only drops when the run asked it to — measuring is always on, acting is opt-in.
    Zero credit drops on its own: a scored rollout whose every token is worth nothing produces no
    gradient, so the forward pass is wasted. A rollout that was never scored is *not* zero-credit —
    opd/opsd train through reference KL and assign no advantages at all, so they must survive."""
    reasons = [name for name in drop_detections if rollout.detections.get(name)]
    if drop_zero_advantage and rollout.advantages is not None and not rollout.is_trainable:
        reasons.append("zero_advantage")
    return reasons
