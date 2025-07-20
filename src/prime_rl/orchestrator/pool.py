import uuid
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
from datasets import Dataset

from prime_rl.orchestrator.config import DataLoadingConfig


@dataclass
class GeneratedSample:
    prompt_tokens: List[int]
    completion_tokens: List[int]
    completion_logprobs: List[float]
    prompt_masks: List[int]
    completion_masks: List[int]
    reward: List[float]
    advantages: List[float]


class DataPool:
    def __init__(self, dataset: Dataset, data_loading_config: DataLoadingConfig):
        uuids = [str(uuid.uuid4()) for _ in dataset]
        self.dataset = dataset.add_column("prime_rl_data_uid", uuids, new_fingerprint="added_uid")
        self.sample_info = {}
        self.data_loading_config = data_loading_config
        self.buffer = []

        for i, uid in enumerate(uuids):
            self.sample_info[uid] = {"dataset_index": i, "already_sampled_current_epoch": False, "num_sampled": 0}

            if not data_loading_config.difficulty_prioritization_strategy.enabled:
                continue

            if data_loading_config.difficulty_prioritization_strategy.priority_data_field:
                priority = self.dataset[i][data_loading_config.difficulty_prioritization_strategy.priority_data_field]
                assert priority in ["low", "high", "discarded"], (
                    f"sampling priority pool value most be 'low' or 'high', found {priority} in column {data_loading_config.difficulty_prioritization_strategy.priority_data_field}"
                )

                self.sample_info[uid]["priority"] = priority

            else:
                self.sample_info[uid]["priority"] = "high"

    def maybe_postprocess(
        self,
        per_problem_uids: list[str],
        per_problem_rewards: list[list[float]],
        per_problem_advantages: list[list[float]],
    ):
        if self.data_loading_config.difficulty_prioritization_strategy.enabled:
            self._postprocess_difficulty_prioritization(per_problem_uids, per_problem_rewards, per_problem_advantages)

    def sample_batch(self, n: int):
        if self.data_loading_config.difficulty_prioritization_strategy.enabled:
            sampled_uids = self._sample_batch_priority_aware(n)
        else:
            sampled_uids = self._sample_batch_normal(n)

        dataset_indices = []
        for uid in sampled_uids:
            dataset_indices.append(self.sample_info[uid]["dataset_index"])

        self._mark_as_sampled(sampled_uids)

        return self.dataset.select(dataset_indices)

    def add_samples(self, samples: list[dict], priorities: list[Literal["low", "high"]] = []):
        for sample in samples:
            assert "prompt" in sample, "Each sample must have a 'prompt' field"
            assert "task" in sample, "Each sample must have a 'task' field"
            assert "info" in sample or "answer" in sample, "Each sample must have either 'info' or 'answer' field"

        # Generate UIDs for new samples
        uuids = [str(uuid.uuid4()) for _ in samples]

        # Add samples to dataset
        for i, (sample, uid) in enumerate(zip(samples, uuids)):
            # Add sample to dataset
            sample["prime_rl_data_uid"] = uid
            self.dataset = self.dataset.add_item(sample, new_fingerprint="")

            # Initialize sample info
            self.sample_info[uid] = {
                "dataset_index": len(self.dataset) - 1,
                "already_sampled_current_epoch": False,
                "num_sampled": 0,
            }

            # Set priority if provided or if difficulty prioritization is enabled
            if self.data_loading_config.difficulty_prioritization_strategy.enabled:
                if priorities:
                    self.sample_info[uid]["priority"] = priorities[i]
                else:
                    self.sample_info[uid]["priority"] = "high"

    def empty_buffer(self):
        self.buffer = []

    def add_to_buffer(self, generated_samples):
        self.buffer.extend(generated_samples)

    def get_buffered_samples(self):
        buffered_samples = self.buffer
        self.buffer = []
        return buffered_samples

    def _postprocess_difficulty_prioritization(
        self,
        per_problem_uids: list[str],
        per_problem_rewards: list[list[float]],
        per_problem_advantages: list[list[float]],
    ):
        EPSILON = 1e-6

        # we sometimes have format rewards, so it is a little weird to ask for 1 or 0
        for uid, rewards, advantages in zip(per_problem_uids, per_problem_rewards, per_problem_advantages):
            if all(abs(a) < EPSILON for a in advantages) and all(r > 0.5 for r in rewards):
                self.sample_info[uid]["priority"] = "discarded"

            elif all(abs(a) < EPSILON for a in advantages) and all(r < 0.5 for r in rewards):
                self.sample_info[uid]["priority"] = "low"

            else:
                self.sample_info[uid]["priority"] = "high"

    def _sample_batch_priority_aware(self, n: int):
        uids = list(self.sample_info.keys())

        n_low = int(n * self.data_loading_config.difficulty_prioritization_strategy.low_priority_batch_fraction)
        n_high = n - n_low

        high_priority_uids = [uid for uid in uids if self.sample_info[uid]["priority"] == "high"]
        high_priority_uids_not_sampled = [
            uid for uid in high_priority_uids if not self.sample_info[uid]["already_sampled_current_epoch"]
        ]

        low_priority_uids = [uid for uid in uids if self.sample_info[uid]["priority"] == "low"]
        low_priority_uids_not_sampled = [
            uid for uid in uids if not self.sample_info[uid]["already_sampled_current_epoch"]
        ]

        if len(low_priority_uids) < n_low:
            n_high = n_high + n_low - len(low_priority_uids)

        sampled_high_priority_uids = self._sample_epoch_aware(
            n_high, high_priority_uids, high_priority_uids_not_sampled
        )
        sampled_low_priority_uids = self._sample_epoch_aware(n_low, low_priority_uids, low_priority_uids_not_sampled)

        return sampled_high_priority_uids + sampled_low_priority_uids

    def _sample_epoch_aware(self, n, all_uids, all_uids_not_yet_sampled):
        if len(all_uids_not_yet_sampled) == 0:
            self._reset_already_sampled(all_uids)
            sampled_uids = self._sample_n_elements_from_list(all_uids, n)

        # in this case we keep the remaining samples from this epoch and sample the rest from the new epoch
        elif n > len(all_uids_not_yet_sampled):
            sampled_uids = all_uids_not_yet_sampled
            self._reset_already_sampled(all_uids)
            n_remaining_high = n - len(all_uids_not_yet_sampled)
            sampled_uids.extend(self._sample_n_elements_from_list(all_uids, n_remaining_high))

        else:
            sampled_uids = self._sample_n_elements_from_list(all_uids_not_yet_sampled, n)

        return sampled_uids

    def _sample_n_elements_from_list(self, elements: list, n: int):
        indices = np.random.choice(len(elements), size=n, replace=True)

        sampled_elements = []
        for idx in indices:
            sampled_elements.append(elements[idx])

        return sampled_elements

    def _sample_batch_normal(self, n: int):
        uids = list(self.sample_info.keys())
        not_sampled_uids = [uid for uid in uids if not self.sample_info[uid]["already_sampled_current_epoch"]]
        sampled_uids = self._sample_epoch_aware(n, uids, not_sampled_uids)

        return sampled_uids

    def _mark_as_sampled(self, uids: list[str]):
        for uid in uids:
            self.sample_info[uid]["already_sampled_current_epoch"] = True
            self.sample_info[uid]["num_sampled"] += 1

    def _reset_already_sampled(self, uids: list[str]):
        for uid in uids:
            self.sample_info[uid]["already_sampled_current_epoch"] = False
