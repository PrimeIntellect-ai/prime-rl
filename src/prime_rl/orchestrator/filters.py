"""Orchestrator-side rollout filters for detecting degenerate generations.

Filters run after rollouts complete, inspecting token IDs and logprobs to
detect gibberish or repetition. Detection metrics are always tracked.
When enforce=True, detected rollouts also get reward zeroed and completion
mask cleared so they don't contribute to training.
"""

import math
from dataclasses import dataclass
from typing import Protocol

import verifiers as vf
from loguru import logger

from prime_rl.orchestrator.config import FilterConfigType, GibberishFilterConfig, RepetitionFilterConfig


@dataclass
class FilterResult:
    detected: bool
    detection_index: int | None = None


class RolloutFilter(Protocol):
    name: str
    enforce: bool

    def check(self, rollout: vf.RolloutOutput) -> FilterResult: ...


@dataclass
class GibberishFilter:
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


@dataclass
class RepetitionFilter:
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


def setup_filter(config: FilterConfigType, vocab_size: int) -> RolloutFilter:
    """Create a RolloutFilter from a filter config."""
    if isinstance(config, GibberishFilterConfig):
        vs = config.vocab_size or vocab_size
        return GibberishFilter(
            name="gibberish",
            token_id_threshold=config.token_id_threshold,
            logprob_threshold=-math.log(vs) - config.logprob_offset,
            enforce=config.enforce,
        )
    elif isinstance(config, RepetitionFilterConfig):
        return RepetitionFilter(
            name="repetition",
            window=config.window,
            logprob_threshold=math.log(config.prob_threshold),
            enforce=config.enforce,
        )
    raise ValueError(f"Unknown filter config type: {type(config)}")


def setup_filters(configs: list[FilterConfigType], vocab_size: int) -> list[RolloutFilter]:
    """Create RolloutFilters from a list of filter configs."""
    return [setup_filter(config, vocab_size) for config in configs]


def apply_filters(
    filters: list[RolloutFilter],
    rollouts: list[vf.RolloutOutput],
) -> dict[str, float]:
    """Apply filters to rollouts. Detection metrics are always tracked.

    When a filter has enforce=True, detected rollouts also get reward zeroed,
    completion mask cleared, and stop_condition set.

    First matching filter wins per rollout (no double-counting).

    Returns aggregate metrics dict for logging.
    """
    if not filters:
        return {}

    counts: dict[str, int] = {f.name: 0 for f in filters}
    total_detected = 0
    total_enforced = 0

    for rollout in rollouts:
        for filt in filters:
            result = filt.check(rollout)
            if result.detected:
                counts[filt.name] += 1
                total_detected += 1

                if rollout.get("metrics") is None:
                    rollout["metrics"] = {}
                rollout["metrics"][f"filter/{filt.name}"] = 1.0

                if filt.enforce:
                    rollout["reward"] = 0.0
                    for step in rollout["trajectory"]:
                        tokens = step["tokens"]
                        if tokens is not None:
                            tokens["completion_mask"] = [0] * len(tokens["completion_mask"])
                    rollout["stop_condition"] = filt.name
                    total_enforced += 1

                break

    n = len(rollouts)
    metrics: dict[str, float] = {}
    for f in filters:
        metrics[f"filter/{f.name}_count"] = float(counts[f.name])
        metrics[f"filter/{f.name}_rate"] = counts[f.name] / n if n > 0 else 0.0
    metrics["filter/total_detected_rate"] = total_detected / n if n > 0 else 0.0
    metrics["filter/total_enforced_rate"] = total_enforced / n if n > 0 else 0.0

    if total_detected > 0:
        enforced_msg = f", enforced {total_enforced}" if total_enforced > 0 else ""
        logger.info(
            f"Detected {total_detected}/{n} rollouts "
            f"({', '.join(f'{name}={c}' for name, c in counts.items() if c > 0)})" + enforced_msg
        )

    return metrics
