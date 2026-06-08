"""Train-rollout detectors for degenerate generations.

Detectors annotate rollouts for logging and metrics only. They never remove
rollouts from training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout


GIBBERISH_TOKEN_ID_FRACTION = 0.75
GIBBERISH_LOGPROB_OFFSET = 2.0
REPETITION_WINDOW = 3_000
REPETITION_PROB_THRESHOLD = 0.99


@dataclass
class DetectionResult:
    detected: bool
    detection_index: int | None = None


class RolloutDetector(Protocol):
    name: str

    def check(self, rollout: "TrainRollout") -> DetectionResult: ...


@dataclass
class GibberishDetector:
    """Detects rollouts containing rare tokens generated at high entropy.

    A token is detected when both:
      - id(token) > token_id_threshold  (rare BPE token)
      - logprob(token) < -log(vocab_size) - logprob_offset  (high entropy)

    References:
      Section 5.2, https://arxiv.org/abs/2510.02387
    """

    name: str
    token_id_threshold: int
    logprob_threshold: float

    def check(self, rollout: "TrainRollout") -> DetectionResult:
        global_idx = 0
        for step in rollout.raw["trajectory"]:
            tokens = step["tokens"]
            if tokens is None:
                continue
            for token_id, logprob in zip(tokens["completion_ids"], tokens["completion_logprobs"]):
                if token_id > self.token_id_threshold and logprob < self.logprob_threshold:
                    return DetectionResult(detected=True, detection_index=global_idx)
                global_idx += 1
        return DetectionResult(detected=False)


@dataclass
class RepetitionDetector:
    """Detects rollouts with pathological repetition loops.

    Counts consecutive tokens where logprob > log(prob_threshold), indicating
    the model is generating with very high confidence. When the streak reaches
    the window size, the rollout is detected.

    References:
      Section 3.2, https://arxiv.org/abs/2506.13585
    """

    name: str
    window: int
    logprob_threshold: float

    def check(self, rollout: "TrainRollout") -> DetectionResult:
        consecutive = 0
        global_idx = 0
        for step in rollout.raw["trajectory"]:
            tokens = step["tokens"]
            if tokens is None:
                continue
            for logprob in tokens["completion_logprobs"]:
                if logprob > self.logprob_threshold:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive >= self.window:
                    return DetectionResult(detected=True, detection_index=global_idx)
                global_idx += 1
        return DetectionResult(detected=False)


def setup_detectors(vocab_size: int) -> list[RolloutDetector]:
    """Create the built-in train-rollout detectors."""
    gibberish_token_id_threshold = int(vocab_size * GIBBERISH_TOKEN_ID_FRACTION)
    detectors: list[RolloutDetector] = [
        GibberishDetector(
            name="gibberish",
            token_id_threshold=gibberish_token_id_threshold,
            logprob_threshold=-math.log(vocab_size) - GIBBERISH_LOGPROB_OFFSET,
        ),
        RepetitionDetector(
            name="repetition",
            window=REPETITION_WINDOW,
            logprob_threshold=math.log(REPETITION_PROB_THRESHOLD),
        ),
    ]
    get_logger().info(
        "Monitoring rollout detectors: "
        f"gibberish(token_id_threshold={gibberish_token_id_threshold}, "
        f"logprob_offset={GIBBERISH_LOGPROB_OFFSET}), "
        f"repetition(window={REPETITION_WINDOW}, prob_threshold={REPETITION_PROB_THRESHOLD})"
    )
    return detectors


def apply_detectors(detectors: list[RolloutDetector], rollouts: list["TrainRollout"]) -> None:  # noqa: F821
    """Annotate rollouts in place with per-detector detection results."""
    for rollout in rollouts:
        rollout.detections = {detector.name: False for detector in detectors}

    if not detectors:
        return

    for rollout in rollouts:
        for detector in detectors:
            result = detector.check(rollout)
            if result.detected:
                rollout.detections[detector.name] = True
