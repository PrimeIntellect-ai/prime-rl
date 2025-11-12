import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk

from prime_rl.orchestrator.config import BufferConfig
from prime_rl.utils.vf import Rollout


class Buffer:
    """A buffer for storing rollouts and metadata."""

    def __init__(self, dataset: Dataset, buffer_config: BufferConfig, buffer_path: Path | None = None):
        self.config = buffer_config
        if self.config.seed is not None:
            random.seed(self.config.seed)

        self._init_buffer(dataset, buffer_path)
        self.problem_metrics = defaultdict(int)
        self.rollout_metrics = defaultdict(int)

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
            loaded_metadata = {row["problem_id"]: {k: v for k, v in row.items() if k != "problem_id"} for row in metadata_dataset}
            
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
            self.problem_metrics[pool_name] += sampled_count
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
            has_zero_advantage = all(rollout["advantage"] == 0.0 for rollout in example_rollouts)
            
            if self.config.easy_threshold is not None and avg_reward >= self.config.easy_threshold:
                new_difficulty = "easy"
            elif self.config.hard_threshold is not None and avg_reward <= self.config.hard_threshold:
                new_difficulty = "hard"
            else:
                new_difficulty = "normal"

            self.metadata[problem_id]["difficulty"] = new_difficulty
            self.rollout_metrics[new_difficulty] += len(example_rollouts)
            
            if (has_zero_advantage
                or (self.config.filter_min_threshold is not None and avg_reward <= self.config.filter_min_threshold)
                or (self.config.filter_max_threshold is not None and avg_reward >= self.config.filter_max_threshold)):
                continue

            self.rollout_buffer.extend(example_rollouts)

    def sample_rollouts(self, n: int) -> list[Rollout]:
        """Samples the latest `n` rollouts from the buffer."""
        n = min(n, len(self.rollout_buffer))
        sampled = self.rollout_buffer[-n:]
        self.rollout_buffer = self.rollout_buffer[:-n]
        return sampled

    def _get_normalized_metrics(self, metrics: dict[str, int], prefix: str) -> dict[str, float]:
        """Helper method to normalize metrics and format them for logging."""
        count_total = sum(metrics.values())
        return {
            f"{prefix}/{key}": count / count_total if count_total > 0 else 0
            for key, count in metrics.items()
        }

    def get_metrics(self) -> dict[str, float]:
        """Returns normalized metrics for problems, rollouts, and data distribution."""
        metrics = {
            **self._get_normalized_metrics(self.problem_metrics, "problem_metrics"),
            **self._get_normalized_metrics(self.rollout_metrics, "rollout_metrics"),
        }
        
        difficulty_counts = Counter(md.get("difficulty", "normal") for md in self.metadata.values())
        total = sum(difficulty_counts.values())
        for difficulty in ["easy", "normal", "hard"]:
            metrics[f"data_metrics/{difficulty}"] = difficulty_counts[difficulty] / total if total > 0 else 0.0
        
        return metrics
    