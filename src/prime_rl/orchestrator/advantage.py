from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import torch
from jaxtyping import Float, Int
from torch import Tensor

from prime_rl.configs.orchestrator import AdvantageConfig, CustomAdvantageConfig
from prime_rl.utils.utils import import_object


@dataclass
class AdvantageInputs:
    """Inputs for advantage computation."""

    rewards: Float[Tensor, "num_problems rollouts_per_example"]
    completion_lengths: Int[Tensor, "num_problems rollouts_per_example"]


@dataclass
class AdvantageOutputs:
    """Outputs from advantage computation."""

    advantages: Float[Tensor, "num_problems rollouts_per_example"]


AdvantageFn = Callable[..., AdvantageOutputs]
"""Type for an advantage function.

Expected signature:
    def my_advantage(inputs: AdvantageInputs, **kwargs) -> AdvantageOutputs:
        ...
"""


def default_advantage_fn(inputs: AdvantageInputs, length_weighted_mean: bool = False) -> AdvantageOutputs:
    """Default GRPO advantage: reward minus per-problem baseline."""
    if length_weighted_mean:
        baseline = (inputs.rewards * inputs.completion_lengths).sum(
            dim=1, keepdim=True
        ) / inputs.completion_lengths.sum(dim=1, keepdim=True)
    else:
        baseline = inputs.rewards.mean(dim=1, keepdim=True)

    return AdvantageOutputs(advantages=inputs.rewards - baseline)


def setup_advantage_fn(config: AdvantageConfig) -> AdvantageFn:
    """Setup advantage function from config."""
    if isinstance(config, CustomAdvantageConfig):
        custom_fn = import_object(config.import_path)
        kwargs = config.kwargs

        def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
            return custom_fn(inputs, **kwargs)

        return advantage_fn

    def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
        return default_advantage_fn(inputs, length_weighted_mean=config.length_weighted_mean)

    return advantage_fn


def compute_advantages(
    rewards: list[float],
    completion_lengths: list[int],
    samples_per_problem: int,
    advantage_config: AdvantageConfig | None,
) -> list[float]:
    """
    Computes advantages from a flattened list of rewards, grouped by problem.

    Args:
        rewards: Flattened list of rewards where first `samples_per_problem` rewards are for the first problem
        completion_lengths: List of completion lengths for each reward
        samples_per_problem: Number of samples (and thus, rewards) per problem
        advantage_config: Configuration for advantage computation (DefaultAdvantageConfig or CustomAdvantageConfig)
    """
    if not advantage_config:
        return rewards

    advantage_fn = setup_advantage_fn(advantage_config)

    inputs = AdvantageInputs(
        rewards=torch.tensor(rewards).view(-1, samples_per_problem),
        completion_lengths=torch.tensor(completion_lengths).view(-1, samples_per_problem),
    )

    result = advantage_fn(inputs)
    return result.advantages.flatten().tolist()


def compute_per_agent_advantages(rollouts: list[dict]) -> None:
    """Compute per-agent GRPO advantages for multi-agent rollouts.

    For multi-agent environments, each trajectory step is tagged with an agent_id
    and has a per-agent reward. Standard GRPO computes advantages from the rollout-
    level mean reward, which is invariant when agent payoffs sum to a constant.

    This function computes advantages per agent: for each (example, agent) pair,
    the baseline is the mean of that agent's rewards across rollouts of the same
    example. Advantages are written directly to trajectory steps so they flow
    through interleave_rollout -> TrainingSample.advantage.

    No-ops if rollouts don't contain per-agent trajectory steps.
    """
    if not rollouts:
        return

    # Quick check: do rollouts have per-agent trajectory steps?
    has_agents = False
    for r in rollouts[:3]:
        for step in r.get("trajectory", []):
            if step.get("extras", {}).get("agent_id"):
                has_agents = True
                break
        if has_agents:
            break
    if not has_agents:
        return

    # Group rollouts by example_id
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rollouts:
        groups[r["example_id"]].append(r)

    for group in groups.values():
        # Collect per-agent rewards: agent_id -> list of (step, reward)
        agent_entries: dict[str, list[tuple[dict, float]]] = defaultdict(list)
        for r in group:
            for step in r.get("trajectory", []):
                agent_id = step.get("extras", {}).get("agent_id")
                reward = step.get("reward")
                if agent_id is not None and reward is not None:
                    agent_entries[agent_id].append((step, reward))

        # Compute per-agent baseline and set per-step advantages
        for agent_id, entries in agent_entries.items():
            baseline = sum(reward for _, reward in entries) / len(entries)
            for step, reward in entries:
                step["advantage"] = reward - baseline
