"""Orchestrator-side rollout filters for detecting degenerate generations.

Filters run after rollouts complete, inspecting token IDs and logprobs to
detect gibberish or repetition. Detected rollouts get reward zeroed and
completion mask cleared so they don't contribute to training.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import verifiers as vf
from loguru import logger


@dataclass
class FilterResult:
    detected: bool
    detection_index: int | None = None


class RolloutFilter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def check(self, rollout: vf.RolloutOutput) -> FilterResult: ...


class GibberishFilter(RolloutFilter):
    """Flags rollouts containing rare tokens generated at high entropy.

    A token is flagged when both:
      - id(token) > token_id_threshold  (rare BPE token)
      - logprob(token) < -log(vocab_size) - logprob_offset  (high entropy)

    References:
      Section 5.2, https://arxiv.org/abs/2510.02387
    """

    def __init__(self, token_id_threshold: int, logprob_threshold: float):
        self.token_id_threshold = token_id_threshold
        self.logprob_threshold = logprob_threshold

    @property
    def name(self) -> str:
        return "gibberish"

    @classmethod
    def from_config(cls, config, vocab_size: int) -> "GibberishFilter":
        logprob_threshold = -math.log(vocab_size) - config.logprob_offset
        return cls(
            token_id_threshold=config.token_id_threshold,
            logprob_threshold=logprob_threshold,
        )

    def check(self, rollout: vf.RolloutOutput) -> FilterResult:
        global_idx = 0
        for step in rollout["trajectory"]:
            tokens = step["tokens"]
            if tokens is None:
                continue
            for token_id, logprob in zip(tokens["completion_ids"], tokens["completion_logprobs"]):
                if token_id > self.token_id_threshold and logprob < self.logprob_threshold:
                    return FilterResult(detected=True, detection_index=global_idx)
                global_idx += 1
        return FilterResult(detected=False)


class RepetitionFilter(RolloutFilter):
    """Flags rollouts with pathological repetition loops.

    Counts consecutive tokens where logprob > log(prob_threshold), indicating
    the model is generating with very high confidence. When the streak reaches
    the window size, the rollout is flagged.

    References:
      Section 3.2, https://arxiv.org/abs/2506.13585
    """

    def __init__(self, window: int, logprob_threshold: float):
        self.window = window
        self.logprob_threshold = logprob_threshold

    @property
    def name(self) -> str:
        return "repetition"

    @classmethod
    def from_config(cls, config) -> "RepetitionFilter":
        return cls(
            window=config.window,
            logprob_threshold=math.log(config.prob_threshold),
        )

    def check(self, rollout: vf.RolloutOutput) -> FilterResult:
        consecutive = 0
        global_idx = 0
        for step in rollout["trajectory"]:
            tokens = step["tokens"]
            if tokens is None:
                continue
            for logprob in tokens["completion_logprobs"]:
                if logprob > self.logprob_threshold:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive >= self.window:
                    return FilterResult(detected=True, detection_index=global_idx)
                global_idx += 1
        return FilterResult(detected=False)


def apply_filters(
    filters: list[RolloutFilter],
    rollouts: list[vf.RolloutOutput],
) -> dict[str, float]:
    """Apply filters to rollouts. Detected rollouts get reward zeroed and mask cleared.

    First matching filter wins per rollout (no double-counting).

    Returns aggregate metrics dict for logging.
    """
    if not filters:
        return {}

    counts: dict[str, int] = {f.name: 0 for f in filters}
    total_filtered = 0

    for rollout in rollouts:
        for filt in filters:
            result = filt.check(rollout)
            if result.detected:
                rollout["reward"] = 0.0
                for step in rollout["trajectory"]:
                    tokens = step["tokens"]
                    if tokens is not None:
                        tokens["completion_mask"] = [0] * len(tokens["completion_mask"])
                rollout["stop_condition"] = filt.name
                if rollout.get("metrics") is None:
                    rollout["metrics"] = {}
                rollout["metrics"][f"filter/{filt.name}"] = 1.0
                counts[filt.name] += 1
                total_filtered += 1
                break

    n = len(rollouts)
    metrics: dict[str, float] = {}
    for f in filters:
        metrics[f"filter/{f.name}_count"] = float(counts[f.name])
        metrics[f"filter/{f.name}_rate"] = counts[f.name] / n if n > 0 else 0.0
    metrics["filter/total_filtered_rate"] = total_filtered / n if n > 0 else 0.0

    if total_filtered > 0:
        logger.info(
            f"Filtered {total_filtered}/{n} rollouts "
            f"({', '.join(f'{name}={c}' for name, c in counts.items() if c > 0)})"
        )

    return metrics
