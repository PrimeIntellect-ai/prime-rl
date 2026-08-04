"""Degeneracy measurements: what a trace's tokens say about how it was generated.

Each one asks a single question — is it gibberish, is it stuck in a repetition loop — of every
trace, unconditionally, as soon as it is tokenized. They are metrics, reported per agent beside
reward and truncation, so they say something whether or not a run acts on them.

Acting on one is a separate decision, made once when the batch is assembled
(``prime_rl.orchestrator.train_sink``). Keeping the two apart is what lets every trace be measured
for all of them: a policy that stopped at the first hit would leave the rest unmeasured.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout

TOKEN_ID_THRESHOLD = 100_000
"""Token IDs above this are candidates for gibberish. BPE tokens are sorted by merge order."""

LOGPROB_OFFSET = 2.0
"""Offset from the uniform-distribution logprob: `-log(vocab_size) - LOGPROB_OFFSET`."""

REPETITION_WINDOW = 3_000
"""Consecutive high-probability sampled tokens that count as a repetition loop."""

REPETITION_PROB = 0.99
"""Tokens sampled above this probability count toward the window."""


def is_gibberish(rollout: TrainRollout, vocab_size: int) -> bool:
    """Whether the trace generated a rare token at high entropy — a rare BPE id sampled as if
    the model had no idea (Section 5.2, https://arxiv.org/abs/2510.02387)."""
    threshold = -math.log(vocab_size) - LOGPROB_OFFSET
    for branch in rollout.branches:
        # branch.{token_ids,logprobs,sampled_mask} are flat and mutually aligned; the raw node
        # arrays are not (node.logprobs covers only the sampled suffix, not the generation-prompt
        # scaffold that token_ids/mask also span).
        for token_id, logprob, sampled in zip(branch.token_ids, branch.logprobs, branch.sampled_mask):
            if sampled and token_id > TOKEN_ID_THRESHOLD and logprob < threshold:
                return True
    return False


def is_repetitive(rollout: TrainRollout) -> bool:
    """Whether the trace held very high confidence for a long stretch — the signature of a
    repetition loop (Section 3.2, https://arxiv.org/abs/2506.13585)."""
    threshold = math.log(REPETITION_PROB)
    for branch in rollout.branches:
        # Aligned branch streams (see `is_gibberish`), and reset the streak per branch: flat
        # rollout.nodes interleaves distinct root->leaf paths (compaction/subagents), so a per-node
        # walk would run a streak across a branch boundary.
        consecutive = 0
        for logprob, sampled in zip(branch.logprobs, branch.sampled_mask):
            if not sampled:
                continue
            consecutive = consecutive + 1 if logprob > threshold else 0
            if consecutive >= REPETITION_WINDOW:
                return True
    return False


def measure(rollout: TrainRollout, vocab_size: int) -> None:
    """Record every measurement on the trace. All of them, every time — a rate is only a rate if
    nothing decided in advance which traces to look at."""
    rollout.degeneracy = {
        "gibberish": is_gibberish(rollout, vocab_size),
        "repetition": is_repetitive(rollout),
    }


def drop_reasons(rollout: TrainRollout, *, drop: list[str], drop_zero_advantage: bool) -> list[str]:
    """Why this rollout should not be trained on, if anything.

    A measurement only drops when the run asked it to — measuring is unconditional, acting is not.
    Zero credit drops on its own: a scored rollout whose every token is worth nothing produces no
    gradient, so the forward pass is wasted. A rollout that was never scored is *not* zero-credit —
    opd/opsd train through reference KL and assign no advantages at all, so they must survive."""
    reasons = [name for name in drop if rollout.degeneracy.get(name)]
    if drop_zero_advantage and rollout.advantages is not None and not rollout.is_trainable:
        reasons.append("zero_advantage")
    return reasons
