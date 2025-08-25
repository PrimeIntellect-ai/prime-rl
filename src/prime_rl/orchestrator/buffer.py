import json
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import Dataset

from prime_rl.orchestrator.config import (
    DataBufferConfigType,
    DifficultyPoolBufferConfig,
    OnlineDifficultyBufferConfig,
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
    is_truncated: bool
    reward: float
    advantage: float


def make_rollouts(
    problem_ids: list[int],
    prompt_tokens: list[list[int]],
    prompt_masks: list[list[int]],
    completion_tokens: list[list[int]],
    completion_masks: list[list[int]],
    completion_logprobs: list[list[float]],
    is_truncated: list[bool],
    rewards: list[float],
    advantages: list[float],
) -> list[Rollout]:
    assert (
        len(problem_ids)
        == len(prompt_tokens)
        == len(prompt_masks)
        == len(completion_tokens)
        == len(completion_masks)
        == len(completion_logprobs)
        == len(is_truncated)
        == len(rewards)
        == len(advantages)
    ), (
        f"The number of problem_ids, prompt_tokens, prompt_masks, completion_tokens, completion_masks, completion_logprobs, is_truncated, rewards, and advantages must be equal, but got ({len(problem_ids)=}, {len(prompt_tokens)=}, {len(prompt_masks)=}, {len(completion_tokens)=}, {len(completion_masks)=}, {len(completion_logprobs)=}, {len(is_truncated)=}, {len(rewards)=}, {len(advantages)=})"
    )
    return [
        Rollout(
            problem_id=problem_id,
            prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask,
            completion_tokens=completion_tokens,
            completion_mask=completion_mask,
            completion_logprobs=completion_logprobs,
            is_truncated=is_truncated,
            reward=reward,
            advantage=advantage,
        )
        for problem_id, prompt_tokens, prompt_mask, completion_tokens, completion_mask, completion_logprobs, is_truncated, reward, advantage in zip(
            problem_ids,
            prompt_tokens,
            prompt_masks,
            completion_tokens,
            completion_masks,
            completion_logprobs,
            is_truncated,
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

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.logger = get_logger()

        # Initialize prompt and rollout buffers
        self.problem_ids = list(range(len(dataset)))
        self.problem_buffer: dict[int, dict] = {
            problem_id: problem for problem_id, problem in zip(self.problem_ids, dataset)
        }
        self.rollout_buffer: dict[int, list[Rollout]] = {}
        self.metadata: dict[int, dict] = {problem_id: {} for problem_id in self.problem_ids}

    def save(self, path: Path):
        """Saves the buffer state to a single HF dataset."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Strip stale columns if present before proceding.
        dataset = self.dataset.remove_columns(
            [c for c in ("metadata", "rollouts") if c in self.dataset.column_names]
        )

        # Serialize metadata and rollouts
        dataset = dataset.add_column(
            "metadata", [json.dumps(self.metadata.get(i, {})) for i in range(len(dataset))]
        )
        dataset = dataset.add_column(
            "rollouts",
            [json.dumps([asdict(r) for r in self.rollout_buffer.get(i, [])]) for i in range(len(dataset))],
        )

        dataset.save_to_disk(path)

    def load(self, path: Path):
        """Loads the buffer state from a single HF dataset."""
        try:
            dataset = Dataset.load_from_disk(path)

            # Deserialization of metadata and rollouts
            self.metadata = {}
            if "metadata" in dataset.column_names:
                for i, m in enumerate(dataset["metadata"]):
                    if not m:
                        self.metadata[i] = {}
                        continue
                    try:
                        self.metadata[i] = json.loads(m)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse metadata row {i}: {e}")
                        self.metadata[i] = {}
            else:
                self.metadata = {i: {} for i in range(len(dataset))}

            self.rollout_buffer = {}
            if "rollouts" in dataset.column_names:
                for i, serialized in enumerate(dataset["rollouts"]):
                    if not serialized:
                        continue
                    try:
                        rollout_dicts = json.loads(serialized)
                        if rollout_dicts:
                            self.rollout_buffer[i] = [Rollout(**r) for r in rollout_dicts]
                    except (json.JSONDecodeError, TypeError) as e:
                        self.logger.warning(f"Failed to parse rollouts row {i}: {e}")

            self.dataset = dataset.remove_columns(
                [c for c in ("metadata", "rollouts") if c in dataset.column_names]
            )

            self.problem_ids = list(range(len(self.dataset)))
            self.problem_buffer = {
                problem_id: problem
                for problem_id, problem in zip(self.problem_ids, self.dataset)
            }

        except Exception as e:
            self.logger.error(f"Failed to load buffer from {path}: {e}")
            raise

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
        `n` * `rollouts_per_example`.  Logs a warning if there are less than `n`
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
        super().__init__(dataset)
        self.config = buffer_config

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


class DifficultyPoolBuffer(Buffer):
    """
    The difficulty pool buffer ensures that a specified fraction of problems are
    sampled from an "easy", "normal" and "hard" difficulty pool. Updates
    difficulty information based on the rollout rewards and advantages. Releases
    all rollouts to the trainer.
    """

    def __init__(self, dataset: Dataset, buffer_config: DifficultyPoolBufferConfig):
        super().__init__(dataset)

        # Add difficulty information to metadata
        self.config = buffer_config
        if self.config.difficulty_field is not None:
            assert self.config.difficulty_field in self.dataset.column_names
            difficulties = self.dataset[self.config.difficulty_field]
            self.logger.info(f"Found difficulty field {self.config.difficulty_field} in dataset")
        else:
            self.logger.warning("No difficulty field specified, initalizing all problems as `normal` difficulty")
            difficulties = ["normal"] * len(self.problem_ids)

        assert len(difficulties) == len(self.problem_ids)
        assert all(difficulty in ["easy", "normal", "hard"] for difficulty in difficulties)
        for problem_id, difficulty in zip(self.problem_ids, difficulties):
            self.metadata[problem_id].update({"difficulty": difficulty})

    def sample_problems(self, n: int) -> tuple[list[int], list[dict]]:
        # Compute number of easy, normal and hard problems to sample
        n_easy = int(n * self.config.easy_fraction)
        n_hard = int(n * self.config.hard_fraction)
        n_normal = n - n_easy - n_hard
        self.logger.debug(f"Sampling {n_easy=}, {n_normal=}, {n_hard=}")

        # Get low and high priority problem
        easy_problem_ids = [
            problem_id for problem_id, metadata in self.metadata.items() if metadata["difficulty"] == "easy"
        ]
        normal_problem_ids = [
            problem_id for problem_id, metadata in self.metadata.items() if metadata["difficulty"] == "normal"
        ]
        hard_problem_ids = [
            problem_id for problem_id, metadata in self.metadata.items() if metadata["difficulty"] == "hard"
        ]
        self.logger.debug(
            f"Found {len(easy_problem_ids)} easy, {len(normal_problem_ids)} normal and {len(hard_problem_ids)} hard problems"
        )

        # Sample easy problems
        # Cannot sample more than the number of low priority problems available
        n_easy_sampled = min(n_easy, len(easy_problem_ids))
        sampled_easy_problem_ids = random.sample(easy_problem_ids, n_easy_sampled)
        assert len(sampled_easy_problem_ids) == n_easy_sampled
        if n_easy_sampled < n_easy:
            self.logger.warning(
                f"Only {n_easy_sampled} easy problems available, sampling {n_easy - n_easy_sampled} normal problems more"
            )
            n_normal += n_easy - n_easy_sampled

        # Sample hard problems
        n_hard_sampled = min(n_hard, len(hard_problem_ids))
        sampled_hard_problem_ids = random.sample(hard_problem_ids, n_hard_sampled)
        assert len(sampled_hard_problem_ids) == n_hard_sampled
        if n_hard_sampled < n_hard:
            self.logger.warning(
                f"Only {n_hard_sampled} hard problems available, sampling {n_hard - n_hard_sampled} normal problems more"
            )
            n_normal += n_hard - n_hard_sampled

        # Sample normal problems
        # TODO: This is not entirely safe - runs may crash once all samples become easy and not enough samples are in normal pool
        assert len(normal_problem_ids) >= n_normal
        n_normal_sampled = min(n_normal, len(normal_problem_ids))
        sampled_normal_problem_ids = random.sample(normal_problem_ids, n_normal_sampled)
        assert len(sampled_normal_problem_ids) == n_normal_sampled

        sampled_problem_ids = sampled_easy_problem_ids + sampled_normal_problem_ids + sampled_hard_problem_ids
        assert len(sampled_problem_ids) == n
        self.logger.debug(
            f"Sampled {n} problems (easy={len(sampled_easy_problem_ids)}, normal={len(sampled_normal_problem_ids)}, hard={len(sampled_hard_problem_ids)}, {sampled_problem_ids=})"
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
            reward = sum([rollout.reward for rollout in rollouts]) / len(rollouts)
            # TODO(Justus): Should we also have rules based on advantages here?
            # TODO(Justus): Should we move samples between pools based on average reward or all(r > threshold for r in rewards)?
            if reward > self.config.easy_border:
                new_difficulty = "easy"
            elif reward < self.config.hard_border:
                new_difficulty = "hard"
            else:
                new_difficulty = "normal"
            old_difficulty = self.metadata[problem_id]["difficulty"]
            stats[(old_difficulty, new_difficulty)] += 1
            self.metadata[problem_id].update({"difficulty": new_difficulty})
        stats_str = ", ".join([f"{v} problems moved from `{k[0]}` to `{k[1]}`" for k, v in stats.items()])
        self.logger.debug(f"Updated difficulty information ({stats_str})")

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
        super().__init__(dataset)
        self.config = buffer_config

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
        self.logger.debug(
            f"Sampled {len(sampled_problem_ids)} rollouts ({sampled_problem_ids=}) within difficulty range [{self.config.min_reward=}, {self.config.max_reward=}]"
        )

        if len(sampled_problem_ids) < n:
            self.logger.warning(
                f"Only {len(sampled_problem_ids)} (<{n}) valid problems with rollouts available ({num_too_easy=}, {num_too_hard=})"
            )

        return sampled_rollouts


def setup_buffer(dataset: Dataset, buffer_config: DataBufferConfigType) -> Buffer:
    if buffer_config.type == "simple":
        return SimpleBuffer(dataset, buffer_config)
    elif buffer_config.type == "difficulty-pool":
        return DifficultyPoolBuffer(dataset, buffer_config)
    elif buffer_config.type == "online-difficulty":
        return OnlineDifficultyBuffer(dataset, buffer_config)