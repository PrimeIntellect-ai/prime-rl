from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import verifiers as vf
from jaxtyping import Float
from torch import Tensor

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout
    from prime_rl.transport import TrainingSample

from prime_rl.configs.orchestrator import (
    AdvantageConfig,
    CustomAdvantageConfig,
    LengthPenaltyConfig,
    TokensLengthPenaltyConfig,
    TurnsLengthPenaltyConfig,
)
from prime_rl.orchestrator.utils import get_model_completion_len, get_tool_response_len
from prime_rl.utils.utils import import_object


@dataclass
class AdvantageInputs:
    """Inputs for advantage computation of a single group (one example × N rollouts)."""

    rollouts: list[vf.RolloutOutput]


@dataclass
class AdvantageOutputs:
    """Outputs from advantage computation of a single group."""

    advantages: list[float]


AdvantageFn = Callable[..., AdvantageOutputs]
"""Type for an advantage function.

Expected signature:
    def my_advantage(inputs: AdvantageInputs, **kwargs) -> AdvantageOutputs:
        ...

The function receives a single group and returns a list of advantages with one
entry per rollout. `assign_advantages` calls it on one already-grouped cohort.
"""


def default_advantage_fn(
    inputs: AdvantageInputs,
    length_penalty: LengthPenaltyConfig | None = None,
) -> AdvantageOutputs:
    """Default GRPO advantage for a single group: reward minus per-group baseline.

    `length_penalty` enables correctness-gated efficiency shaping over a per-rollout
    cost: tokens (weighted completion + tool-response) or trajectory turn count.
    """
    rewards = torch.tensor([r["reward"] for r in inputs.rollouts], dtype=torch.float32)

    if isinstance(length_penalty, TokensLengthPenaltyConfig):
        w_c = length_penalty.completion_weight
        w_t = length_penalty.tool_response_weight
        costs = torch.tensor(
            [w_c * get_model_completion_len(r) + w_t * get_tool_response_len(r) for r in inputs.rollouts],
            dtype=rewards.dtype,
        )
        return AdvantageOutputs(advantages=_efficiency_shaping(rewards, costs).tolist())
    if isinstance(length_penalty, TurnsLengthPenaltyConfig):
        costs = torch.tensor([len(r["trajectory"]) for r in inputs.rollouts], dtype=rewards.dtype)
        return AdvantageOutputs(advantages=_efficiency_shaping(rewards, costs).tolist())

    return AdvantageOutputs(advantages=(rewards - rewards.mean()).tolist())


def _efficiency_shaping(
    rewards: Float[Tensor, "group_size"],
    costs: Float[Tensor, "group_size"],
) -> Float[Tensor, "group_size"]:
    """Correctness-gated efficiency shaping with bounded advantages.

    Shapes rewards with a bounded efficiency bonus before standard GRPO subtraction,
    preserving zero-mean advantages within the group. `costs` is a per-rollout cost
    (e.g., completion length in tokens or number of turns).

    Correct rollouts get reward amplified by up to 2x based on relative efficiency.
    Incorrect rollouts are untouched. Lower-cost correct rollouts get higher advantage.
    """
    max_reward = rewards.max()
    correct_mask = rewards >= max_reward
    num_correct = correct_mask.sum()

    # No shaping when max reward is 0 — no correct rollouts to differentiate
    if max_reward <= 0:
        return rewards - rewards.mean()

    # Mean cost of correct rollouts
    mean_correct_cost = (costs * correct_mask).sum() / num_correct.clamp(min=1)

    # Bounded efficiency bonus: [0, 1], positive for below-average cost, zero for above.
    # When mean_correct_cost is 0 (e.g. tool-only shaping with no harness metric, or
    # all-zero turn counts), no rollouts can be differentiated — fall back to no bonus.
    if mean_correct_cost <= 0:
        return rewards - rewards.mean()

    bonus = (1 - costs / mean_correct_cost).clamp(0, 1)

    # Shape rewards: correct rollouts amplified by up to 2x, incorrect untouched
    shaped_rewards = rewards * (1 + bonus * correct_mask)
    return shaped_rewards - shaped_rewards.mean()


def setup_advantage_fn(config: AdvantageConfig) -> AdvantageFn:
    """Setup advantage function from config."""
    if isinstance(config, CustomAdvantageConfig):
        custom_fn = import_object(config.import_path)
        kwargs = config.kwargs

        def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
            return custom_fn(inputs, **kwargs)

        return advantage_fn

    def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
        return default_advantage_fn(inputs, length_penalty=config.length_penalty)

    return advantage_fn


def assign_advantages(
    rollouts: list["TrainRollout"],  # noqa: F821 (forward ref)
    advantage_fn: AdvantageFn | None,
) -> None:
    """Compute and assign advantages for one finished group of rollouts
    (``TrainSink.process_group`` hands in a single group's surviving rollouts).
    ``advantage_fn=None`` is the trivial case (advantage = reward); a custom
    ``advantage_fn`` receives the raw ``vf.RolloutOutput``\\ s via
    ``AdvantageInputs.rollouts``.
    """
    if advantage_fn is None:
        for rollout in rollouts:
            rollout.advantage = rollout.reward
        return
    result = advantage_fn(AdvantageInputs(rollouts=[r.raw for r in rollouts]))
    for rollout, advantage in zip(rollouts, result.advantages):
        rollout.advantage = advantage


# --------------------------------------------------------------------------------------------------
# Layer 2 — the per-term, per-token advantage (the loss-term ``advantage`` axis). One float per token
# (``0`` = masked); built orchestrator-side from ``RenderHints`` + the group's rewards. ``grpo`` reuses
# Layer 1 above (reward -> per-rollout scalar); ``echo``/``sft`` read attribution only.
# --------------------------------------------------------------------------------------------------


@dataclass
class RenderHints:
    """All per-token render info for one training unit (rollout/sample), aligned to
    ``prompt_ids + completion_ids``.

    Built orchestrator-side from the rollout's tokens + ``prompt_attribution``; passed to the term
    ``advantage_fn`` (and, via the shipped slice, to trainer-side cores/hooks). ``rollout`` is the raw
    renderer output for anything not pre-parsed (orchestrator-side only).
    """

    token_id: list[int]
    role: list[str | None]  # per token; None = unattributed
    tool_name: list[str | None]  # per token; set when role == "tool"
    is_sampled: list[bool]  # the sampled completion tokens (today's loss_mask)
    inference_logprob: list[float]  # sampled logprob; 0.0 on prompt tokens
    reward: float | None = None
    rollout: vf.RolloutOutput | None = None


TermAdvantageFn = Callable[..., list[list[float]]]
"""A loss term's advantage axis. Signature: ``fn(group: list[RenderHints], **kwargs) -> list[list[float]]``
— one inner list per unit, one float per token; ``0`` masks the token."""


def grpo_advantage(
    group: list[RenderHints],
    *,
    tau: float = 1.0,
    length_penalty: LengthPenaltyConfig | None = None,
) -> list[list[float]]:
    """Per-token GRPO advantage: the per-rollout scalar (reward minus group baseline, optional length
    penalty — Layer 1) broadcast over each unit's sampled tokens x ``tau``, ``0`` elsewhere."""
    scalars = default_advantage_fn(
        AdvantageInputs(rollouts=[h.rollout for h in group]), length_penalty=length_penalty
    ).advantages
    return [[scalar * tau if sampled else 0.0 for sampled in h.is_sampled] for h, scalar in zip(group, scalars)]


def echo_advantage(
    group: list[RenderHints],
    *,
    roles: list[str],
    tool_names: set[str] | None = None,
    alpha: float = 1.0,
) -> list[list[float]]:
    """Per-token echo signal: ``alpha`` on tokens whose role matches (and, for ``tool``, whose
    ``tool_name`` is allowed), ``0`` elsewhere."""
    role_set = set(roles)
    return [
        [
            alpha if (role in role_set and (role != "tool" or tool_names is None or tool_name in tool_names)) else 0.0
            for role, tool_name in zip(h.role, h.tool_name)
        ]
        for h in group
    ]


def sft_advantage(group: list[RenderHints], *, alpha: float = 1.0) -> list[list[float]]:
    """Per-token SFT signal: ``alpha`` on the sampled tokens, ``0`` elsewhere."""
    return [[alpha if sampled else 0.0 for sampled in h.is_sampled] for h in group]


def build_render_hints(sample: TrainingSample, rollout: vf.RolloutOutput | None = None) -> RenderHints:
    """Build a sample's ``RenderHints`` from the finished (interleaved) ``TrainingSample``.

    The sample-derivable fields are exact: ``token_id``, ``is_sampled`` (prompt + completion masks),
    ``inference_logprob``. Sampled (model-generated) tokens get role ``"assistant"``; prompt-side roles
    (system/user/tool) are not attributed here yet — that needs the per-step interleave alignment and
    is added when prompt-role advantage_fns are wired.
    """
    n_prompt = len(sample.prompt_ids)
    token_id = list(sample.prompt_ids) + list(sample.completion_ids)
    is_sampled = [False] * n_prompt + [bool(m) for m in sample.completion_mask]
    inference_logprob = [0.0] * n_prompt + list(sample.completion_logprobs)
    return RenderHints(
        token_id=token_id,
        role=["assistant" if sampled else None for sampled in is_sampled],
        tool_name=[None] * len(token_id),
        is_sampled=is_sampled,
        inference_logprob=inference_logprob,
        reward=rollout.get("reward") if rollout is not None else None,
        rollout=rollout,
    )
