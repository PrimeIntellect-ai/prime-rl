"""Hardcoded rollout checks between scoring and training.

Gibberish and repetition detection runs on every trainable rollout and is
tracked in metrics only — a detection never drops a rollout. Zero-advantage
rollouts carry no learning signal (unless the env's algorithm says otherwise)
and are dropped before they enter the training batch.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout

# Gibberish: rare tokens generated at high entropy (Section 5.2,
# https://arxiv.org/abs/2510.02387). A token is flagged when its id exceeds
# the threshold (rare BPE token, sorted by merge order) and its logprob is
# below ``-log(vocab_size) - offset`` (high entropy).
GIBBERISH_TOKEN_ID_THRESHOLD = 100_000
GIBBERISH_LOGPROB_OFFSET = 2.0

# Repetition: pathological high-confidence loops (Section 3.2,
# https://arxiv.org/abs/2506.13585). Flagged when ``WINDOW`` consecutive
# tokens are each sampled with probability above ``PROB_THRESHOLD``.
REPETITION_WINDOW = 3_000
REPETITION_PROB_THRESHOLD = 0.99


def gibberish_logprob_threshold(vocab_size: int) -> float:
    return -math.log(vocab_size) - GIBBERISH_LOGPROB_OFFSET


def detect_gibberish(rollout: Rollout, logprob_threshold: float) -> bool:
    for branch in rollout.branches:
        # branch.{token_ids,logprobs,sampled_mask} are flat and mutually aligned; the raw
        # node arrays are not (node.logprobs covers only the sampled suffix, not the
        # generation-prompt scaffold that token_ids/mask also span).
        for token_id, logprob, sampled in zip(branch.token_ids, branch.logprobs, branch.sampled_mask):
            if not sampled:
                continue
            if token_id > GIBBERISH_TOKEN_ID_THRESHOLD and logprob < logprob_threshold:
                return True
    return False


def detect_repetition(rollout: Rollout) -> bool:
    logprob_threshold = math.log(REPETITION_PROB_THRESHOLD)
    for branch in rollout.branches:
        # Aligned branch streams (see detect_gibberish), and reset the streak per branch:
        # flat rollout.nodes interleaves distinct root->leaf paths (compaction/subagents),
        # so a per-node walk would run a streak across a branch boundary.
        consecutive = 0
        for logprob, sampled in zip(branch.logprobs, branch.sampled_mask):
            if not sampled:
                continue
            if logprob > logprob_threshold:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= REPETITION_WINDOW:
                return True
    return False


def has_zero_advantage(rollout: Rollout) -> bool:
    """True when the advantage stream is present but all zero (e.g. all
    rollouts in a GRPO group earned the same reward, so the centered advantage
    collapses). Algorithms that assign no advantage (opd/opsd) never match."""
    return rollout.advantages is not None and all(a == 0.0 for a in rollout.advantages)
