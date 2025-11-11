import random
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk

from prime_rl.orchestrator.config import BufferConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.vf import Rollout


class Buffer:
    """
    Buffer that samples problems from difficulty pools (easy/normal/hard).
    
    Rollouts are stored individually in a flat list. Difficulty pools are updated based on
    rollout advantage and reward:
    - If advantage == 0: reward 1.0 → easy pool, reward 0.0 → hard pool (rollout not stored)
    - If advantage != 0: normal pool (rollout stored in buffer)
    
    Sampling returns the latest n rollouts from the buffer.
    """

    def __init__(self, dataset: Dataset, buffer_config: BufferConfig):
        self.logger = get_logger()
        self.config = buffer_config
        if self.config.seed is not None:
            random.seed(self.config.seed)

        # Initialize buffer
        self._init_buffer(dataset, self.config.from_scratch)
        self.problem_metrics = defaultdict(int)
        self.rollout_metrics = defaultdict(int)

    def _init_buffer(self, dataset: Dataset, from_scratch: bool, dataset_path: Path | None = None) -> None:
        """Initializes the buffer state from datasets."""
        if "id" not in dataset.column_names:
            dataset = dataset.add_column("id", list(range(len(dataset))), new_fingerprint=str(uuid.uuid4()))
        self.problem_ids = dataset["id"]

        if from_scratch:
            self.rollout_buffer: list[Rollout] = []
            self.metadata = {pid: {"difficulty": "normal"} for pid in self.problem_ids}
        else:
            if not dataset_path:
                raise ValueError("dataset_path is required when loading from checkpoint")
            
            metadata_path = dataset_path.parent / "metadata"
            if not metadata_path.exists():
                raise ValueError(f"Metadata dataset not found at {metadata_path}")
            metadata_dataset = load_from_disk(metadata_path)
            self.metadata = {row["problem_id"]: {k: v for k, v in row.items() if k != "problem_id"} for row in metadata_dataset}
            
            rollouts_path = dataset_path.parent / "rollouts"
            if rollouts_path.exists():
                rollouts_dataset = load_from_disk(rollouts_path)
                self.rollout_buffer = [Rollout(**row) for row in rollouts_dataset]
            else:
                self.rollout_buffer = []
            
            for pid, md in self.metadata.items():
                if md.get("difficulty") not in ["easy", "normal", "hard"]:
                    raise ValueError(f"Invalid difficulty {md.get('difficulty')} for problem {pid}")

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

    def load(self, path: Path) -> None:
        """Loads metadata and rollouts from separate HF datasets. Uses the existing dataset stored in the buffer."""
        self._init_buffer(self.dataset, from_scratch=False, dataset_path=path)

    def sample_problems(self, n: int) -> list[dict]:
        """Samples `n` problems from the dataset using difficulty pools."""
        n_easy = int(n * self.config.easy_fraction)
        n_hard = int(n * self.config.hard_fraction)
        n_normal = n - n_easy - n_hard

        # Group problem IDs by difficulty
        by_difficulty = defaultdict(list)
        for problem_id, metadata in self.metadata.items():
            by_difficulty[metadata["difficulty"]].append(problem_id)

        # Sample from each pool, falling back to normal if insufficient
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
        for rollout in rollouts:
            problem_id = rollout["example_id"]
            if rollout["advantage"] == 0:
                new_difficulty = "easy" if rollout["reward"] == 1.0 else "hard"
            else:
                self.rollout_buffer.append(rollout)
                new_difficulty = "normal"
            self.metadata[problem_id]["difficulty"] = new_difficulty
            self.rollout_metrics[new_difficulty] += 1

    def sample_rollouts(self, n: int) -> list[Rollout]:
        """Samples the latest `n` rollouts from the buffer."""
        if not self.rollout_buffer or n <= 0:
            return []
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
        
        # Calculate data distribution metrics from metadata
        difficulty_counts = Counter(md.get("difficulty", "normal") for md in self.metadata.values())
        total = sum(difficulty_counts.values())
        for difficulty in ["easy", "normal", "hard"]:
            metrics[f"data_metrics/{difficulty}"] = difficulty_counts[difficulty] / total if total > 0 else 0.0
        
        return metrics
    