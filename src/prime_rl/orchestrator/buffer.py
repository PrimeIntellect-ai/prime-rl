import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import verifiers as vf
from datasets import Dataset, load_from_disk

from prime_rl.orchestrator.config import BufferConfig
from prime_rl.utils.utils import mean_normalize


class Buffer:
    """A buffer for storing rollouts and metadata."""

    POOLS = ["easy", "normal", "hard"]

    def __init__(self, dataset: Dataset, buffer_config: BufferConfig):
        self.config = buffer_config
        if self.config.seed is not None:
            random.seed(self.config.seed)

        # Initialize buffer state
        assert "example_id" in dataset.column_names, "The dataset must contain a `example_id` column."
        assert isinstance(dataset["example_id"][0], int), "The `example_id` column must be of type int."
        assert len(set(dataset["example_id"])) == len(dataset), "The `example_id` column must be unique."
        self.dataset = dataset
        self.problem_ids = dataset["example_id"]
        self.problem_buffer = {pid: dict(problem) for pid, problem in zip(self.problem_ids, dataset)}
        self.rollout_buffer: list[vf.State] = []
        self.metadata = {pid: {"difficulty": "normal"} for pid in self.problem_ids}

        # The number of problems/rollouts sampled from each pool at the current step (will reset with every call to get_metrics)
        self.num_sampled_problems_per_pool = defaultdict(int)  # Will reset every step
        self.num_sampled_rollouts_per_pool = defaultdict(int)  # Will reset every step
        self.num_filtered_rollouts_per_difficulty = defaultdict(int)  # Will reset every step
        self.num_rollouts = 0  # Will reset every step

    def save(self, path: Path) -> None:
        """Saves metadata and rollouts as separate HF datasets."""
        path.mkdir(parents=True, exist_ok=True)

        metadata_path = path / "metadata"
        metadata_data = [{"problem_id": pid, **self.metadata[pid]} for pid in self.problem_ids]
        Dataset.from_list(metadata_data).save_to_disk(metadata_path)

        rollouts_path = path / "rollouts"
        if self.rollout_buffer:
            Dataset.from_list(list(map(dict, self.rollout_buffer))).save_to_disk(rollouts_path)
        elif rollouts_path.exists():
            shutil.rmtree(rollouts_path)

    def load(self, path: Path) -> None:
        """Loads metadata and rollouts from separate HF datasets. Uses the existing dataset stored in the buffer."""
        # Load metadata
        metadata_path = path / "metadata"
        if not metadata_path.exists():
            raise ValueError(f"Metadata dataset not found at {metadata_path}")
        metadata_dataset = cast(Dataset, load_from_disk(metadata_path))
        problem_ids = metadata_dataset["problem_id"]
        metadata_dataset = metadata_dataset.remove_columns("problem_id")
        self.metadata = {
            problem_id: {"difficulty": "normal", **cast(dict, metadata)}
            for problem_id, metadata in zip(problem_ids, metadata_dataset)
        }

        # Load rollouts
        rollouts_path = path / "rollouts"
        if rollouts_path.exists():
            raise NotImplementedError("Loading rollouts from disk is not implemented yet")
            # rollouts_dataset = load_from_disk(rollouts_path)
            # self.rollout_buffer = [**cast(dict, row)) for row in rollouts_dataset]

    def sample_problems(self, n: int) -> list[dict]:
        """Samples `n` problems from the dataset using difficulty pools."""
        n_easy = int(n * self.config.easy_fraction)
        n_hard = int(n * self.config.hard_fraction)
        n_normal = n - n_easy - n_hard

        by_difficulty = defaultdict(list)
        for problem_id, metadata in self.metadata.items():
            by_difficulty[metadata["difficulty"]].append(problem_id)

        def sample_pool(pool_ids: list[int], target: int, pool_name: str) -> tuple[list[int], int]:
            sampled_count = min(target, len(pool_ids))
            sampled_ids = random.sample(pool_ids, sampled_count) if sampled_count > 0 else []
            self.num_sampled_problems_per_pool[pool_name] += sampled_count
            return sampled_ids, target - sampled_count

        sampled_easy, easy_deficit = sample_pool(by_difficulty["easy"], n_easy, "easy")
        sampled_hard, hard_deficit = sample_pool(by_difficulty["hard"], n_hard, "hard")
        sampled_normal, _ = sample_pool(by_difficulty["normal"], n_normal + easy_deficit + hard_deficit, "normal")

        sampled_ids = sampled_easy + sampled_normal + sampled_hard
        return [self.problem_buffer[pid] for pid in sampled_ids]

    def update(self, rollouts: list[vf.State]):
        """Updates the buffer state with completed rollouts."""
        rollouts_by_example = defaultdict(list)
        for rollout in rollouts:
            problem_id = rollout["example_id"]
            rollouts_by_example[problem_id].append(rollout)

        for problem_id, example_rollouts in rollouts_by_example.items():
            avg_reward = sum(rollout["reward"] for rollout in example_rollouts) / len(example_rollouts)
            if self.config.easy_threshold is not None and avg_reward >= self.config.easy_threshold:
                new_difficulty = "easy"
            elif self.config.hard_threshold is not None and avg_reward <= self.config.hard_threshold:
                new_difficulty = "hard"
            else:
                new_difficulty = "normal"

            self.metadata[problem_id]["difficulty"] = new_difficulty
            self.num_sampled_rollouts_per_pool[new_difficulty] += 1

            self.num_rollouts += len(example_rollouts)
            if self.config.online_difficulty_filtering:
                if avg_reward == 0.0:
                    self.num_filtered_rollouts_per_difficulty["hard"] += len(example_rollouts)
                    continue
                elif avg_reward == 1.0:
                    self.num_filtered_rollouts_per_difficulty["easy"] += len(example_rollouts)
                    continue
            self.rollout_buffer.extend(example_rollouts)

    def sample_rollouts(self, n: int) -> list[vf.State]:
        """Samples the latest `n` rollouts from the buffer."""
        n = min(n, len(self.rollout_buffer))
        sampled = self.rollout_buffer[-n:]
        self.rollout_buffer = self.rollout_buffer[:-n]
        return sampled

    def get_metrics(self) -> dict[str, float]:
        metrics = {}

        # Add ratio of problems sampled from each pool this step
        problem_pool_counts = [self.num_sampled_problems_per_pool.get(pool, 0.0) for pool in self.POOLS]
        problem_pool_ratio = mean_normalize(problem_pool_counts)
        prefix = "buffer/sampled_problems"
        metrics.update({f"{prefix}/{pool}": value for pool, value in zip(self.POOLS, problem_pool_ratio)})

        # Add ratio of rollouts sampled from each pool this step
        rollout_pool_counts = [self.num_sampled_rollouts_per_pool.get(pool, 0.0) for pool in self.POOLS]
        rollout_pool_ratio = mean_normalize(rollout_pool_counts)
        prefix = "buffer/sampled_rollouts"
        metrics.update({f"{prefix}/{pool}": value for pool, value in zip(self.POOLS, rollout_pool_ratio)})

        # Add ratio of rollouts filtered out this step
        easy_filtered_ratio = (
            self.num_filtered_rollouts_per_difficulty["easy"] / self.num_rollouts if self.num_rollouts > 0 else 0.0
        )
        hard_filtered_ratio = (
            self.num_filtered_rollouts_per_difficulty["hard"] / self.num_rollouts if self.num_rollouts > 0 else 0.0
        )
        prefix = "buffer/filtered_rollouts"
        metrics.update(
            {
                f"{prefix}/easy": easy_filtered_ratio,
                f"{prefix}/hard": hard_filtered_ratio,
            }
        )

        # Add overall ratio of problems over pools
        pool_counter = Counter(m.get("difficulty", "normal") for m in self.metadata.values())
        pool_counts = [pool_counter.get(pool, 0.0) for pool in self.POOLS]
        pool_ratio = mean_normalize(pool_counts)
        metrics.update({f"buffer/pool/{pool}": value for pool, value in zip(self.POOLS, pool_ratio)})

        # Reset per-step metrics
        self.num_sampled_problems_per_pool = defaultdict(int)
        self.num_sampled_rollouts_per_pool = defaultdict(int)
        self.num_filtered_rollouts_per_difficulty = defaultdict(int)
        self.num_rollouts = 0
        return metrics
