import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass

from datasets import Dataset

from prime_rl.orchestrator.config import (
    DataBufferConfig,
    OnlineDifficultyBufferConfig,
    PriorityPoolBufferConfig,
    SimpleBufferConfig,
)
from prime_rl.utils.logger import get_logger


@dataclass
class Rollout:
    problem_id: int
    prompt_tokens: list[int]
    prompt_mask: list[int]
    completion_tokens: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    reward: float
    advantage: float


def make_rollouts(
    problem_ids: list[int],
    prompt_tokens: list[list[int]],
    prompt_masks: list[list[int]],
    completion_tokens: list[list[int]],
    completion_masks: list[list[int]],
    completion_logprobs: list[list[float]],
    rewards: list[list[float]],
    advantages: list[list[float]],
) -> list[Rollout]:
    assert (
        len(problem_ids)
        == len(prompt_tokens)
        == len(prompt_masks)
        == len(completion_tokens)
        == len(completion_masks)
        == len(completion_logprobs)
        == len(rewards)
        == len(advantages)
    ), (
        f"The number of problem_ids, prompt_tokens, prompt_masks, completion_tokens, completion_masks, completion_logprobs, rewards, and advantages must be equal, but got ({len(problem_ids)=}, {len(prompt_tokens)=}, {len(prompt_masks)=}, {len(completion_tokens)=}, {len(completion_masks)=}, {len(completion_logprobs)=}, {len(rewards)=}, {len(advantages)=})"
    )
    return [
        Rollout(
            problem_id=problem_id,
            prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask,
            completion_tokens=completion_tokens,
            completion_mask=completion_mask,
            completion_logprobs=completion_logprobs,
            reward=reward,
            advantage=advantage,
        )
        for problem_id, prompt_tokens, prompt_mask, completion_tokens, completion_mask, completion_logprobs, reward, advantage in zip(
            problem_ids,
            prompt_tokens,
            prompt_masks,
            completion_tokens,
            completion_masks,
            completion_logprobs,
            rewards,
            advantages,
        )
    ]


class Buffer(ABC):
    """
    Abstract base class for buffers. A buffer is a stateful class storing raw
    dataset samples and completed rollouts. Crucially, any instance of this
    class defines a strategy for sampling from the dataset and the rollouts.
    """

    def __init__(self, dataset: Dataset, buffer_config: DataBufferConfig):
        self.dataset = dataset
        self.config = buffer_config
        self.logger = get_logger()

        # Initialize prompt and rollout buffers
        self.problem_ids = list(range(len(dataset)))
        self.problem_buffer: dict[int, dict] = {
            problem_id: problem for problem_id, problem in zip(self.problem_ids, dataset)
        }
        self.rollout_buffer: dict[int, list[Rollout]] = {}
        self.metadata: dict[int, dict] = {problem_id: {} for problem_id in self.problem_ids}

    @abstractmethod
    def sample_problems(self, n: int) -> tuple[list[int], list[dict]]:
        """
        Samples `n` problems from the dataset. Returns a list of problem IDs
        and a list of dictionaries representing the problems. The dictionary keys
        correspond to the fields of the dataset used for initializing the pool.

        Args:
            n: The number of problems to sample.

        Returns:
            A tuple of two lists. The first list contains the problem IDs of the
            sampled problems. The second list contains the problems themselves.
        """
        pass

    @abstractmethod
    def update(self, rollouts: list[Rollout]):
        """
        Updates the buffer state with the completed rollouts. Should store
        rollouts in the rollout buffer and update metadata about problems
        relevant for sampling.

        Args:
            rollouts: A list of rollouts to update the pool with.
        """
        pass

    @abstractmethod
    def sample_rollouts(self, n: int) -> list[Rollout]:
        """
        Samples rollouts for `n` problems from the rollout buffer which are
        ready for training. Thus, the length of the list returned is equal to
        `n` * `rollouts_per_prompt`.  Logs a warning if there are less than `n`
        samples available.

        Args:
            n: The number of problems to return rollouts for.

        Returns:
            A list of rollouts that are ready to be used by the trainer.
        """
        pass


class SimpleBuffer(Buffer):
    """
    Simple buffer that samples problems from the dataset in chronological order
    and immediately returns all generated rollouts to the trainer.
    """

    def __init__(self, dataset: Dataset, buffer_config: SimpleBufferConfig):
        super().__init__(dataset, buffer_config)

    def sample_problems(self, n: int) -> tuple[list[int], list[dict]]:
        # Get indices to sample
        assert len(self.problem_ids) >= n, (
            f"There should be at least {n} problems in the buffer, but found only {len(self.problem_ids)}"
        )
        sampled_problem_ids = random.sample(self.problem_ids, n)
        assert len(sampled_problem_ids) == n
        self.logger.debug(f"Sampled {n} problems ({sampled_problem_ids=})")

        # Get problems from indices
        sampled_problems = [self.problem_buffer[problem_id] for problem_id in sampled_problem_ids]

        return sampled_problem_ids, sampled_problems

    def update(self, rollouts: list[Rollout]):
        # Group rollouts by problem_id
        rollouts_by_problem_id = defaultdict(list)
        for rollout in rollouts:
            rollouts_by_problem_id[rollout.problem_id].append(rollout)

        # Add grouped rollouts to the buffer
        self.rollout_buffer.update(rollouts_by_problem_id)

    def sample_rollouts(self, n: int) -> list[Rollout]:
        # Take the first n problems from the rollout buffer
        available_problem_ids = list(self.rollout_buffer.keys())
        assert len(available_problem_ids) == n, (
            "The number of available problems should always be equal to the requested number of problems"
        )
        sampled_problem_ids = available_problem_ids[:n]
        assert len(sampled_problem_ids) == n, (
            "The number of sampled problems should always be equal to the requested number of problems"
        )

        # Build (flattened) list of rollouts
        sampled_rollouts: list[Rollout] = []
        for problem_id in sampled_problem_ids:
            sampled_rollout = self.rollout_buffer.pop(problem_id)
            sampled_rollouts.extend(sampled_rollout)

        return sampled_rollouts


class PriorityPoolBuffer(Buffer):
    """
    The priority pool buffer ensures that a specified fraction of problems are
    sampled from a "low" and "high" priority pool. Updates priority information
    based on the rollout rewards and advantages. Releases all rollouts to the
    trainer.
    """

    def __init__(self, dataset: Dataset, buffer_config: PriorityPoolBufferConfig):
        super().__init__(dataset, buffer_config)

        # Add priority information to metadata
        if self.config.priority_field is not None:
            assert self.config.priority_field in self.dataset.column_names
            priorities = self.dataset[self.config.priority_field]
            self.logger.info(f"Found priority field {self.config.priority_field} in dataset")
        else:
            self.logger.info("No priority field specified, initalizing all problems as `high` priority")
            priorities = ["high"] * len(self.problem_ids)

        assert len(priorities) == len(self.problem_ids)
        assert all(priority in ["low", "high", "discarded"] for priority in priorities)
        for problem_id, priority in zip(self.problem_ids, priorities):
            self.metadata[problem_id].update({"priority": priority})

    def sample_problems(self, n: int) -> tuple[list[int], list[dict]]:
        # Compute number of low and high priority problems to sample
        n_low = int(n * self.config.low_priority_fraction)
        n_high = n - n_low
        self.logger.debug(f"{n_low=}, {n_high=}")

        # Get low and high priority problem
        low_priority_problem_ids = [
            problem_id for problem_id, metadata in self.metadata.items() if metadata["priority"] == "low"
        ]
        high_priority_problem_ids = [
            problem_id for problem_id, metadata in self.metadata.items() if metadata["priority"] == "high"
        ]
        self.logger.debug(
            f"Found {len(low_priority_problem_ids)} low priority and {len(high_priority_problem_ids)} high priority samples"
        )

        # Sample low priority problems
        # Cannot sample more than the number of low priority problems available
        n_low_sampled = min(n_low, len(low_priority_problem_ids))
        sampled_low_priority_problem_ids = random.sample(low_priority_problem_ids, n_low_sampled)
        assert len(sampled_low_priority_problem_ids) == n_low_sampled
        if n_low_sampled < n_low:
            n_high = n - n_low

        # Sample the rest from the high priority samples
        assert len(high_priority_problem_ids) >= n_high
        sampled_high_priority_problem_ids = random.sample(high_priority_problem_ids, n_high)
        assert len(sampled_high_priority_problem_ids) == n_high

        sampled_problem_ids = sampled_low_priority_problem_ids + sampled_high_priority_problem_ids
        assert len(sampled_problem_ids) == n
        self.logger.debug(
            f"Sampled {n} problems (low_priority={len(sampled_low_priority_problem_ids)}, high_priority={len(sampled_high_priority_problem_ids)}, {sampled_problem_ids=})"
        )

        # Sample problems
        sampled_problems = [self.problem_buffer[problem_id] for problem_id in sampled_problem_ids]

        return sampled_problem_ids, sampled_problems

    def update(self, rollouts: list[Rollout]):
        # Group rollouts by problem_id
        rollouts_by_problem_id = defaultdict(list)
        for rollout in rollouts:
            rollouts_by_problem_id[rollout.problem_id].append(rollout)

        # Add grouped rollouts to the buffer
        self.rollout_buffer.update(rollouts_by_problem_id)

        # Update metadata with priority information
        stats = Counter()
        for problem_id, rollouts in rollouts_by_problem_id.items():
            rewards = [rollout.reward for rollout in rollouts]
            advantages = [rollout.advantage for rollout in rollouts]
            if all(abs(a) < 1e-6 for a in advantages) and all(r > 0.5 for r in rewards):
                priority = "discarded"
            elif all(abs(a) < 1e-6 for a in advantages) and all(r < 0.5 for r in rewards):
                priority = "low"
            else:
                priority = "high"
            stats[priority] += 1
            self.metadata[problem_id].update({"priority": priority})
        self.logger.info(f"Updated priority information: {stats=}")

    def sample_rollouts(self, n: int) -> list[Rollout]:
        # Take the first n rollouts from the rollout buffer
        available_problem_ids = list(self.rollout_buffer.keys())
        assert len(available_problem_ids) == n, (
            "The number of available problems should always be equal to the requested number of problems"
        )
        sampled_problem_ids = available_problem_ids[:n]
        assert len(sampled_problem_ids) == n, (
            "The number of sampled problems should always be equal to the requested number of problems"
        )

        # Build (flattened) list of rollouts
        sampled_rollouts: list[Rollout] = []
        for problem_id in sampled_problem_ids:
            sampled_rollout = self.rollout_buffer.pop(problem_id)
            sampled_rollouts.extend(sampled_rollout)

        return sampled_rollouts


class OnlineDifficultyBuffer(Buffer):
    """
    The online difficulty buffer ensures that any sampled rollouts are within
    some configurable difficulty range. This means it may not return the
    specified number of rollouts. It is the orchestrator's task to sample more.
    An oversampling factor can be specified to increase the chance that at least
    n problems are within the difficulty range.
    """

    def __init__(self, dataset: Dataset, buffer_config: OnlineDifficultyBufferConfig):
        super().__init__(dataset, buffer_config)

    def sample_problems(self, n: int) -> tuple[list[int], list[dict]]:
        # Multiply by oversampling factor
        n = int(self.config.oversampling_factor * n)

        # Get indices to sample
        assert len(self.problem_ids) >= n, (
            f"There should be at least {n} problems in the buffer, but found only {len(self.problem_ids)}"
        )
        sampled_problem_ids = random.sample(self.problem_ids, n)
        self.logger.debug(f"Sampled {n} problems ({sampled_problem_ids=})")

        # Sample problems
        sampled_problems = [self.problem_buffer[problem_id] for problem_id in sampled_problem_ids]

        return sampled_problem_ids, sampled_problems

    def update(self, rollouts: list[Rollout]):
        # Group rollouts by problem_id
        rollouts_by_problem_id = defaultdict(list)
        for rollout in rollouts:
            rollouts_by_problem_id[rollout.problem_id].append(rollout)

        # Do not keep rollouts from older weight checkpoints
        # TODO: Can we lift this constraint? Maybe, instead of clearing we mark
        # the out of range samples as discarded and remove them from the list of
        # problem IDs that can be sampled
        self.rollout_buffer.clear()

        # Add grouped rollouts to the buffer
        self.rollout_buffer.update(rollouts_by_problem_id)

        # Update metadata with difficulty information
        for problem_id, rollouts in rollouts_by_problem_id.items():
            reward = sum(rollout.reward for rollout in rollouts) / len(rollouts)
            self.metadata[problem_id].update({"reward": reward})

    def sample_rollouts(self, n: int) -> list[Rollout]:
        available_problem_ids = list(self.rollout_buffer.keys())
        sampled_problem_ids, sampled_rollouts = [], []
        num_too_easy, num_too_hard = 0, 0
        # Take the first n rollouts within the difficulty range
        for problem_id in available_problem_ids:
            reward = self.metadata[problem_id]["reward"]
            if self.config.min_reward is not None and reward < self.config.min_reward:
                num_too_hard += 1
                continue
            if self.config.max_reward is not None and reward > self.config.max_reward:
                num_too_easy += 1
                continue
            sampled_rollout = self.rollout_buffer.pop(problem_id)
            sampled_rollouts.extend(sampled_rollout)
            sampled_problem_ids.append(problem_id)

        assert all(
            [
                self.config.min_reward or -1e9 <= self.metadata[problem_id]["reward"] <= self.config.max_reward or 1e9
                for problem_id in sampled_problem_ids
            ]
        )
        self.logger.info(
            f"Sampled {len(sampled_problem_ids)} problems ({sampled_problem_ids=}) within difficulty range [{self.config.min_reward=}, {self.config.max_reward=}]"
        )

        if len(sampled_problem_ids) < n:
            self.logger.warning(
                f"Only {len(sampled_problem_ids)} (<{n}) problems with rollouts available ({num_too_easy=}, {num_too_hard=})"
            )

        return sampled_rollouts


def setup_buffer(dataset: Dataset, buffer_config: DataBufferConfig) -> Buffer:
    if buffer_config.type == "simple":
        return SimpleBuffer(dataset, buffer_config)
    elif buffer_config.type == "priority_pool":
        return PriorityPoolBuffer(dataset, buffer_config)
    elif buffer_config.type == "online_difficulty":
        return OnlineDifficultyBuffer(dataset, buffer_config)
