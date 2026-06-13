from __future__ import annotations

import asyncio
import math
import random
import statistics
from collections import defaultdict
from itertools import cycle
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import AdvantageConfig, RLCSDAdvantageConfig
from prime_rl.orchestrator.algo.advantage import apply_advantage_fn, broadcast
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.utils import compute_prefill_logprobs
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import TrainRollout
    from prime_rl.utils.client import InferencePool

_ADV_EPS = 1e-6


def _std_norm_advantage_fn(rollouts: list[TrainRollout]) -> list[list[float]]:
    """Std-normalized group-relative advantage (the paper's Eq. 8):
    ``(r - mean) / (std + eps)``, broadcast over completion tokens."""
    rewards = [r.reward for r in rollouts]
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    return broadcast(rollouts, [(r - mean) / (std + _ADV_EPS) for r in rewards])


def _hint_pools(
    group: list[TrainRollout], correct_threshold: float, min_contrast_gap: float
) -> tuple[list[TrainRollout], list[TrainRollout]]:
    """Partition one group into hint pools. Positives are verified correct
    (``reward >= correct_threshold``); negatives must be clearly wrong
    (``reward < correct_threshold - min_contrast_gap``). Rollouts in the band
    between are neither — they never serve as hints, so near-threshold noise
    stops producing contrast as the group tightens."""
    correct = [r for r in group if (r.reward or 0.0) >= correct_threshold]
    wrong = [r for r in group if (r.reward or 0.0) < correct_threshold - min_contrast_gap]
    return correct, wrong


def _contrastive_signal(pos_logprobs: list[float], neg_logprobs: list[list[float]]) -> list[float]:
    """Per-token contrast ``e_ctr`` (Eq. 7): the teacher's logprob under the
    correct hint minus the log of the *mean probability* over the K incorrect
    hints (log-mean-exp, not mean logprob)."""
    k = len(neg_logprobs)
    signal = []
    for t, pos in enumerate(pos_logprobs):
        neg_t = [neg[t] for neg in neg_logprobs]
        peak = max(neg_t)
        log_mean_neg = peak + math.log(sum(math.exp(v - peak) for v in neg_t) / k)
        signal.append(pos - log_mean_neg)
    return signal


def _modulated_token_advantages(
    signal: list[float],
    advantages: list[float],
    completion_mask: list[bool],
    *,
    lam: float,
    tau: float,
    delta: float,
    eta: float,
) -> list[float] | None:
    """Two-path token advantages (Eqs. 9-15): squash the contrast through
    ``lam·tanh(·/tau)``, modulate the per-token base advantages (uniform
    under group-norm assignment) at tokens above the ``delta`` mask with a
    sign-preserving clamp, and fold the paper's independent path
    normalization into the magnitudes — the clipped surrogate is positively
    homogeneous in the advantage, so per-rollout weights ``L/|U|`` and
    ``eta·L/|M|`` reproduce the two-path objective without touching the
    trainer. Returns ``None`` when no token is trainable."""
    modulation = [lam * math.tanh(e / tau) for e in signal]
    trainable = [t for t, trains in enumerate(completion_mask) if trains]
    if not trainable:
        return None
    modulated = {t for t in trainable if abs(modulation[t]) > delta}
    num_total = len(trainable)
    num_modulated = len(modulated)
    num_plain = num_total - num_modulated

    token_advantages = [0.0] * len(signal)
    for t in trainable:
        base = advantages[t]
        if t in modulated:
            shifted = base + modulation[t]
            clamped = max(0.0, shifted) if base >= 0 else min(0.0, shifted)
            token_advantages[t] = eta * clamped * (num_total / num_modulated)
        else:
            token_advantages[t] = base * (num_total / num_plain)
    return token_advantages


class RLCSDAlgorithm(Algorithm):
    """RLCSD (arXiv:2606.11709): GRPO anchored by the verifier, with a
    contrastive self-distillation signal modulating the advantage magnitude
    at high-signal tokens.

    At group time, std-normalized group-relative credit (broadcast per
    token). At ship time, each surviving rollout's tokens are prefill-scored
    under the teacher conditioned on a correct sibling rollout and on K
    incorrect siblings (byte-identical hint template, so the
    privilege-induced style shift cancels in the subtraction); the squashed
    contrast modulates the base advantages with a sign-preserving clamp and
    overwrites the sample's advantage stream on the ``rl`` component.
    Rollouts whose group offers no contrast (no correct or no incorrect
    sibling) keep their plain group-norm stream."""

    action_loss_type = "rl"
    model_role = "teacher"

    def __init__(self, advantage: AdvantageConfig, policy_pool: InferencePool, renderer: Renderer | None):
        super().__init__(advantage, policy_pool, renderer)
        assert isinstance(advantage, RLCSDAdvantageConfig)
        assert renderer is not None, "rlcsd requires the renderer (validated at config time)"
        self.num_negative_hints = advantage.num_negative_hints
        self.tau = advantage.tau
        self.lam = advantage.lam
        self.delta = advantage.delta
        self.eta = advantage.eta
        self.correct_threshold = advantage.correct_threshold
        self.min_contrast_gap = advantage.min_contrast_gap
        self.template = advantage.template
        self.max_concurrent = advantage.max_concurrent
        self.teacher = advantage.model
        self.teacher_pool: InferencePool | None = None  # connected in setup()

    async def setup(self) -> None:
        self.teacher_pool = await self.connect(self.teacher)

    def assign_advantages(self, rollouts: list[TrainRollout]) -> None:
        apply_advantage_fn(rollouts, _std_norm_advantage_fn)

    async def query_references(self, rollouts: list[TrainRollout]) -> None:
        pool = self.teacher_pool
        assert pool is not None, "teacher pool not connected — Algorithm.setup() must run first"
        semaphore = asyncio.Semaphore(self.max_concurrent)
        clients = cycle(pool.train_clients)

        groups: dict[object, list[TrainRollout]] = defaultdict(list)
        for rollout in rollouts:
            groups[rollout.group_id].append(rollout)

        tasks = []
        for group in groups.values():
            correct, wrong = _hint_pools(group, self.correct_threshold, self.min_contrast_gap)
            for rollout in group:
                # Hints come from siblings only — conditioning the teacher on
                # the rollout itself shifts it toward degenerate over-confidence.
                pos_pool = [s for s in correct if s is not rollout]
                neg_pool = [s for s in wrong if s is not rollout]
                if not pos_pool or not neg_pool:
                    continue  # no contrast available — the rollout keeps its plain group-norm stream
                tasks.append(self._score_rollout(rollout, pos_pool, neg_pool, semaphore, pool, next(clients)))
        # Contrast needs a correct AND an incorrect sibling; early in training
        # (or with a miscalibrated correct_threshold) most groups offer none
        # and the batch silently trains as plain GRPO — make that visible.
        get_logger().debug(f"rlcsd: contrast available for {len(tasks)}/{len(rollouts)} rollouts")
        if tasks:
            await asyncio.gather(*tasks)

    async def _score_rollout(
        self,
        rollout: TrainRollout,
        pos_pool: list[TrainRollout],
        neg_pool: list[TrainRollout],
        semaphore: asyncio.Semaphore,
        pool: InferencePool,
        client,
    ) -> None:
        assert len(rollout.samples) == 1  # single-step trajectory → one sample
        sample = rollout.samples[0]
        completion_ids = list(sample.completion_ids)

        pos_hint = random.choice(pos_pool)
        neg_hints = random.sample(neg_pool, min(self.num_negative_hints, len(neg_pool)))

        async def hinted_logprobs(hint: TrainRollout) -> list[float]:
            prefix_ids = self._hinted_prefix_ids(rollout, hint)
            async with semaphore:
                full = await compute_prefill_logprobs(client, pool.model_name, prefix_ids + completion_ids)
            return full[-len(completion_ids) :]

        results = await asyncio.gather(hinted_logprobs(pos_hint), *(hinted_logprobs(h) for h in neg_hints))
        signal = _contrastive_signal(results[0], list(results[1:]))

        # The assigned stream is completion-aligned (single sample, asserted above).
        advantages = rollout.advantages if rollout.advantages is not None else [0.0] * len(completion_ids)
        token_advantages = _modulated_token_advantages(
            signal,
            advantages,
            list(sample.completion_mask),
            lam=self.lam,
            tau=self.tau,
            delta=self.delta,
            eta=self.eta,
        )
        if token_advantages is not None:
            sample.advantages = [0.0] * len(sample.prompt_ids) + token_advantages

    def _hint_text(self, rollout: TrainRollout) -> str:
        """A sibling rollout's full completion text — the reference solution
        the teacher is conditioned on."""
        trajectory = rollout.raw.get("trajectory") or []
        if len(trajectory) != 1:
            raise ValueError(
                f"rlcsd supports single-step trajectories only; "
                f"env '{rollout.env_name}' produced {len(trajectory)} steps."
            )
        parts = [m.get("content") for m in trajectory[0]["completion"] if isinstance(m.get("content"), str)]
        return "\n".join(parts)

    def _hinted_prefix_ids(self, rollout: TrainRollout, hint: TrainRollout) -> list[int]:
        """Rebuild the rollout's first-turn prompt with the hint woven into
        the last user message, rendered through the policy's renderer — the
        same messages → token ids path the policy's own prompts take."""
        trajectory = rollout.raw.get("trajectory") or []
        if len(trajectory) != 1:
            raise ValueError(
                f"rlcsd supports single-step trajectories only; "
                f"env '{rollout.env_name}' produced {len(trajectory)} steps."
            )
        messages = [dict(m) for m in trajectory[0]["prompt"]]
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not user_indices:
            raise ValueError(f"rlcsd found no user message to condition (env '{rollout.env_name}').")
        last_user = messages[user_indices[-1]]
        question = last_user.get("content")
        if not isinstance(question, str):
            raise ValueError("rlcsd supports text-only prompts (user content must be a string).")
        last_user["content"] = self.template.format(question=question, hint=self._hint_text(hint))
        assert self.renderer is not None
        return self.renderer.render_ids(messages, add_generation_prompt=True)
