import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk

from prime_rl.orchestrator.config import BufferConfig
from prime_rl.utils.utils import mean_normalize
from prime_rl.utils.vf import Rollout


class Buffer:
    """A buffer for storing rollouts and metadata."""

    DIFFICULTY_LEVELS = ["easy", "normal", "hard"]

    def __init__(self, dataset: Dataset, buffer_config: BufferConfig, buffer_path: Path | None = None):
        self.config = buffer_config
        if self.config.seed is not None:
            random.seed(self.config.seed)

        self._init_buffer(dataset, buffer_path)
        # The number of problems/ rollouts sampled from each pool at the current step (will reset with every call to get_metrics)
        self.num_sampled_problems_per_pool = defaultdict(int)
        self.num_sampled_rollouts_per_pool = defaultdict(int)

    def _init_buffer(self, dataset: Dataset, buffer_path: Path | None = None) -> None:
        """Initializes the buffer state from datasets."""
        # Use example_id column from verifiers
        assert "example_id" in dataset.column_names, "The dataset must contain a `example_id` column."
        assert isinstance(dataset["example_id"][0], int), "The `example_id` column must be of type int."
        assert len(set(dataset["example_id"])) == len(dataset), "The `example_id` column must be unique."
        self.problem_ids = dataset["example_id"]

        if not buffer_path:
            self.rollout_buffer: list[Rollout] = []
            self.metadata = {pid: {"difficulty": "normal"} for pid in self.problem_ids}
        else:
            metadata_path = buffer_path.parent / "metadata"
            if not metadata_path.exists():
                raise ValueError(f"Metadata dataset not found at {metadata_path}")
            metadata_dataset = load_from_disk(metadata_path)
            loaded_metadata = {
                row["problem_id"]: {k: v for k, v in row.items() if k != "problem_id"} for row in metadata_dataset
            }

            self.metadata = {}
            for pid in self.problem_ids:
                if pid in loaded_metadata:
                    self.metadata[pid] = loaded_metadata[pid]
                else:
                    self.metadata[pid] = {"difficulty": "normal"}

            rollouts_path = buffer_path.parent / "rollouts"
            if rollouts_path.exists():
                rollouts_dataset = load_from_disk(rollouts_path)
                self.rollout_buffer = [dict(row) for row in rollouts_dataset]
            else:
                self.rollout_buffer = []

        self.dataset = dataset
        self.problem_buffer = {pid: dict(problem) for pid, problem in zip(self.problem_ids, dataset)}

    def save(self, path: Path) -> None:
        """Saves metadata and rollouts as separate HF datasets."""
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata_path = path.parent / "metadata"
        metadata_data = [{"problem_id": pid, **self.metadata[pid]} for pid in self.problem_ids]
        Dataset.from_list(metadata_data).save_to_disk(metadata_path)

        rollouts_path = path.parent / "rollouts"
        if self.rollout_buffer:
            Dataset.from_list(self.rollout_buffer).save_to_disk(rollouts_path)
        elif rollouts_path.exists():
            shutil.rmtree(rollouts_path)

    def load(self, path: Path) -> None:
        """Loads metadata and rollouts from separate HF datasets. Uses the existing dataset stored in the buffer."""
        self._init_buffer(self.dataset, buffer_path=path)

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

    def update(self, rollouts: list[Rollout]):
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

            if (self.config.filter_min_threshold is not None and avg_reward <= self.config.filter_min_threshold) or (
                self.config.filter_max_threshold is not None and avg_reward >= self.config.filter_max_threshold
            ):
                continue

            self.rollout_buffer.extend(example_rollouts)

    def sample_rollouts(self, n: int) -> list[Rollout]:
        """Samples the latest `n` rollouts from the buffer."""
        n = min(n, len(self.rollout_buffer))
        sampled = self.rollout_buffer[-n:]
        self.rollout_buffer = self.rollout_buffer[:-n]
        return sampled

    def get_metrics(self) -> dict[str, float]:
        metrics = {}

        # Add ratio of problems sampled from each pool this step
        problem_pool_counts = [self.num_sampled_problems_per_pool.get(pool, 0.0) for pool in self.DIFFICULTY_LEVELS]
        problem_pool_ratio = mean_normalize(problem_pool_counts)
        prefix = "buffer/sampled_problems"
        metrics.update({f"{prefix}/{key}": value for key, value in zip(self.DIFFICULTY_LEVELS, problem_pool_ratio)})

        # Add ratio of rollouts sampled from each difficulty level this step
        rollout_pool_counts = [self.num_sampled_rollouts_per_pool.get(pool, 0.0) for pool in self.DIFFICULTY_LEVELS]
        prefix = "buffer/sampled_rollouts"
        rollout_pool_ratio = mean_normalize(rollout_pool_counts)
        metrics.update({f"{prefix}/{key}": value for key, value in zip(self.DIFFICULTY_LEVELS, rollout_pool_ratio)})

        pool_counts = Counter(m.get("difficulty", "normal") for m in self.metadata.values())
        pool_ratio = mean_normalize(list(pool_counts.values()))
        metrics.update({f"buffer/pool/{key}": value for key, value in zip(pool_counts.keys(), pool_ratio)})

        # Reset per-step metrics
        self.num_sampled_problems_per_pool = defaultdict(int)
        self.num_sampled_rollouts_per_pool = defaultdict(int)

        return metrics
