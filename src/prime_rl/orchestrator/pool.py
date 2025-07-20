from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from datasets import Dataset

from prime_rl.orchestrator.config import (
    DataPoolConfig,
    DefaultPoolConfig,
    OnlineDifficultyPoolConfig,
    PriorityPoolConfig,
)
from prime_rl.utils.logger import get_logger


@dataclass
class Rollout:
    problem_id: int
    prompt_tokens: List[int]
    prompt_mask: List[int]
    completion_tokens: List[int]
    completion_mask: List[int]
    completion_logprobs: List[float]
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


class Pool(ABC):
    """
    Abstract base class for data pools. This abstraction is used to sample
    problems from a dataset, store rollouts, and release rollouts for the
    trainer according to different sampling strategies.
    """

    def __init__(self, dataset: Dataset, pool_config: DataPoolConfig):
        self.dataset = dataset
        self.config = pool_config
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
        and a list of dictionaries representing the problem. The dictionary keys
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
        Updates the pool state with the completed rollouts. Should store
        rollouts in the rollout buffer and update metadata about problems
        relevant for sampling.

        Args:
            rollouts: A list of rollouts to update the pool with.
        """
        pass

    @abstractmethod
    def sample_rollouts(self, n: int) -> list[Rollout]:
        """
        Samples rollouts for `n` problems from the rollout buffer. Thus, the
        length of the list returned is equal to `n` * `rollouts_per_prompt`. Logs a warning
        if there are less than `n` samples available.

        Args:
            n: The number of problems to return rollouts for.

        Returns:
            A list of rollouts that are ready to be used by the trainer.
        """
        pass


class DefaultPool(Pool):
    """
    Simple pool that samples problems in a round-robin fashion and
    immediately returns all rollouts to the trainer.
    """

    def __init__(self, dataset: Dataset, pool_config: DefaultPoolConfig):
        super().__init__(dataset, pool_config)

        # Initialize metadata to include epoch information
        self.epoch = 1
        self.epoch_step = 0
        for problem_id in self.problem_ids:
            self.metadata[problem_id].update({"epoch": 1})

    def sample_problems(self, n: int) -> tuple[list[int], list[dict]]:
        # Get indices to sample
        start_idx = self.epoch_step * n
        sampled_problem_ids = range(start_idx, start_idx + n)
        self.logger.debug(f"Sampling {n} problems ({sampled_problem_ids=})")

        # Sample problems
        sampled_problems = [self.problem_buffer[problem_id] for problem_id in sampled_problem_ids]

        # Update metadata
        self.metadata.update({problem_id: {"epoch": self.epoch + 1} for problem_id in sampled_problem_ids})
        self.epoch_step += 1

        # If all prompts have been sampled, increment epoch and reset step
        if all(self.metadata[problem_id]["epoch"] == self.epoch + 1 for problem_id in self.problem_ids):
            self.logger.info(f"Epoch {self.epoch} complete. Starting epoch {self.epoch + 1} and resetting epoch step.")
            self.epoch += 1
            self.epoch_step = 0

        return sampled_problem_ids, sampled_problems

    def update(self, rollouts: list[Rollout]):
        # Group rollouts by problem_id
        rollouts_by_problem_id = defaultdict(list)
        for rollout in rollouts:
            rollouts_by_problem_id[rollout.problem_id].append(rollout)

        # Add grouped rollouts to the buffer
        self.rollout_buffer.update(rollouts_by_problem_id)

    def sample_rollouts(self, n: int) -> list[Rollout]:
        # Take the first n rollouts from the rollout buffer
        available_problem_ids = list(self.rollout_buffer.keys())
        assert len(available_problem_ids) == n, (
            "The number of available problems should always be equal to the requested number of problems"
        )
        sampled_problem_ids = available_problem_ids[:n]

        if len(sampled_problem_ids) < n:
            self.logger.warning(f"Only {len(sampled_problem_ids)} (<{n}) problems with rollouts available.")

        sampled_rollouts: list[Rollout] = []
        for problem_id in sampled_problem_ids:
            sampled_rollout = self.rollout_buffer.pop(problem_id)
            sampled_rollouts.extend(sampled_rollout)

        return sampled_rollouts


class PriorityPool(Pool):
    def __init__(self, dataset: Dataset, pool_config: PriorityPoolConfig):
        pass


class OnlineDifficultyPool(Pool):
    def __init__(self, dataset: Dataset, pool_config: OnlineDifficultyPoolConfig):
        pass


def setup_pool(dataset: Dataset, pool_config: DataPoolConfig) -> Pool:
    if pool_config.strategy == "default":
        return DefaultPool(dataset, pool_config)
    elif pool_config.strategy == "priority":
        return PriorityPool(dataset, pool_config)
    elif pool_config.strategy == "online_difficulty":
        return OnlineDifficultyPool(dataset, pool_config)


# class DataPool:
#     def __init__(self, dataset: Dataset, pool_config: DataPoolConfig):
#         uuids = [str(uuid.uuid4()) for _ in dataset]
#         self.dataset = dataset.add_column("prime_rl_data_uid", uuids, new_fingerprint="added_uid")
#         self.sample_info = {}
#         self.data_loading_config = data_loading_config
#         self.buffer = []
#
#         for i, uid in enumerate(uuids):
#             self.sample_info[uid] = {"dataset_index": i, "already_sampled_current_epoch": False, "num_sampled": 0}
#
#             if not data_loading_config.difficulty_prioritization_strategy.enabled:
#                 continue
#
#             if data_loading_config.difficulty_prioritization_strategy.priority_data_field:
#                 priority = self.dataset[i][data_loading_config.difficulty_prioritization_strategy.priority_data_field]
#                 assert priority in ["low", "high", "discarded"], (
#                     f"sampling priority pool value most be 'low' or 'high', found {priority} in column {data_loading_config.difficulty_prioritization_strategy.priority_data_field}"
#                 )
#
#                 self.sample_info[uid]["priority"] = priority
#
#             else:
#                 self.sample_info[uid]["priority"] = "high"
#
#     def maybe_postprocess(
#         self,
#         per_problem_uids: list[str],
#         per_problem_rewards: list[list[float]],
#         per_problem_advantages: list[list[float]],
#     ):
#         if self.data_loading_config.difficulty_prioritization_strategy.enabled:
#             self._postprocess_difficulty_prioritization(per_problem_uids, per_problem_rewards, per_problem_advantages)
#
#     def sample_batch(self, n: int):
#         if self.data_loading_config.difficulty_prioritization_strategy.enabled:
#             sampled_uids = self._sample_batch_priority_aware(n)
#         else:
#             sampled_uids = self._sample_batch_normal(n)
#
#         dataset_indices = []
#         for uid in sampled_uids:
#             dataset_indices.append(self.sample_info[uid]["dataset_index"])
#
#         self._mark_as_sampled(sampled_uids)
#
#         return self.dataset.select(dataset_indices)
#
#     def add_samples(self, samples: list[dict], priorities: list[Literal["low", "high"]] = []):
#         for sample in samples:
#             assert "prompt" in sample, "Each sample must have a 'prompt' field"
#             assert "task" in sample, "Each sample must have a 'task' field"
#             assert "info" in sample or "answer" in sample, "Each sample must have either 'info' or 'answer' field"
#
#         # Generate UIDs for new samples
#         uuids = [str(uuid.uuid4()) for _ in samples]
#
#         # Add samples to dataset
#         for i, (sample, uid) in enumerate(zip(samples, uuids)):
#             # Add sample to dataset
#             sample["prime_rl_data_uid"] = uid
#             self.dataset = self.dataset.add_item(sample, new_fingerprint="")
#
#             # Initialize sample info
#             self.sample_info[uid] = {
#                 "dataset_index": len(self.dataset) - 1,
#                 "already_sampled_current_epoch": False,
#                 "num_sampled": 0,
#             }
#
#             # Set priority if provided or if difficulty prioritization is enabled
#             if self.data_loading_config.difficulty_prioritization_strategy.enabled:
#                 if priorities:
#                     self.sample_info[uid]["priority"] = priorities[i]
#                 else:
#                     self.sample_info[uid]["priority"] = "high"
#
#     def empty_buffer(self):
#         self.buffer = []
#
#     def add_to_buffer(self, generated_samples):
#         self.buffer.extend(generated_samples)
#
#     def get_buffered_samples(self):
#         buffered_samples = self.buffer
#         self.buffer = []
#         return buffered_samples
#
#     def _postprocess_difficulty_prioritization(
#         self,
#         per_problem_uids: list[str],
#         per_problem_rewards: list[list[float]],
#         per_problem_advantages: list[list[float]],
#     ):
#         EPSILON = 1e-6
#
#         # we sometimes have format rewards, so it is a little weird to ask for 1 or 0
#         for uid, rewards, advantages in zip(per_problem_uids, per_problem_rewards, per_problem_advantages):
#             if all(abs(a) < EPSILON for a in advantages) and all(r > 0.5 for r in rewards):
#                 self.sample_info[uid]["priority"] = "discarded"
#
#             elif all(abs(a) < EPSILON for a in advantages) and all(r < 0.5 for r in rewards):
#                 self.sample_info[uid]["priority"] = "low"
#
#             else:
#                 self.sample_info[uid]["priority"] = "high"
#
#     def _sample_batch_priority_aware(self, n: int):
#         uids = list(self.sample_info.keys())
#
#         n_low = int(n * self.data_loading_config.difficulty_prioritization_strategy.low_priority_batch_fraction)
#         n_high = n - n_low
#
#         high_priority_uids = [uid for uid in uids if self.sample_info[uid]["priority"] == "high"]
#         high_priority_uids_not_sampled = [
#             uid for uid in high_priority_uids if not self.sample_info[uid]["already_sampled_current_epoch"]
#         ]
#
#         low_priority_uids = [uid for uid in uids if self.sample_info[uid]["priority"] == "low"]
#         low_priority_uids_not_sampled = [
#             uid for uid in uids if not self.sample_info[uid]["already_sampled_current_epoch"]
#         ]
#
#         if len(low_priority_uids) < n_low:
#             n_high = n_high + n_low - len(low_priority_uids)
#
#         sampled_high_priority_uids = self._sample_epoch_aware(
#             n_high, high_priority_uids, high_priority_uids_not_sampled
#         )
#         sampled_low_priority_uids = self._sample_epoch_aware(n_low, low_priority_uids, low_priority_uids_not_sampled)
#
#         return sampled_high_priority_uids + sampled_low_priority_uids
#
#     def _sample_epoch_aware(self, n, all_uids, all_uids_not_yet_sampled):
#         if len(all_uids_not_yet_sampled) == 0:
#             self._reset_already_sampled(all_uids)
#             sampled_uids = self._sample_n_elements_from_list(all_uids, n)
#
#         # in this case we keep the remaining samples from this epoch and sample the rest from the new epoch
#         elif n > len(all_uids_not_yet_sampled):
#             sampled_uids = all_uids_not_yet_sampled
#             self._reset_already_sampled(all_uids)
#             n_remaining_high = n - len(all_uids_not_yet_sampled)
#             sampled_uids.extend(self._sample_n_elements_from_list(all_uids, n_remaining_high))
#
#         else:
#             sampled_uids = self._sample_n_elements_from_list(all_uids_not_yet_sampled, n)
#
#         return sampled_uids
#
#     def _sample_n_elements_from_list(self, elements: list, n: int):
#         indices = np.random.choice(len(elements), size=n, replace=True)
#
#         sampled_elements = []
#         for idx in indices:
#             sampled_elements.append(elements[idx])
#
#         return sampled_elements
#
#     def _sample_batch_normal(self, n: int):
#         uids = list(self.sample_info.keys())
#         not_sampled_uids = [uid for uid in uids if not self.sample_info[uid]["already_sampled_current_epoch"]]
#         sampled_uids = self._sample_epoch_aware(n, uids, not_sampled_uids)
#
#         return sampled_uids
#
#     def _mark_as_sampled(self, uids: list[str]):
#         for uid in uids:
#             self.sample_info[uid]["already_sampled_current_epoch"] = True
#             self.sample_info[uid]["num_sampled"] += 1
#
#     def _reset_already_sampled(self, uids: list[str]):
#         for uid in uids:
#             self.sample_info[uid]["already_sampled_current_epoch"] = False
#
