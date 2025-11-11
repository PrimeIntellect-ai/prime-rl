import random
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk

from prime_rl.orchestrator.config import BufferConfig
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
        self.config = buffer_config
        if self.config.seed is not None:
            random.seed(self.config.seed)

        # Initialize buffer
        self._init_buffer(dataset, self.config.dataset_path)
        self.problem_metrics = defaultdict(int)
        self.rollout_metrics = defaultdict(int)

    def _init_buffer(self, dataset: Dataset, dataset_path: Path | None = None) -> None:
        """Initializes the buffer state from datasets."""
        if "id" not in dataset.column_names:
            dataset = dataset.add_column("id", list(range(len(dataset))), new_fingerprint=str(uuid.uuid4()))
        self.problem_ids = dataset["id"]

        if not dataset_path:
            self.rollout_buffer: list[Rollout] = []
            self.replay_buffer: list[Rollout] = []
            self.metadata = {pid: {"difficulty": "normal"} for pid in self.problem_ids}
        else:
            if self.config.refresh_metadata:
                self.metadata = {pid: {"difficulty": "normal"} for pid in self.problem_ids}
            else:
                metadata_path = dataset_path.parent / "metadata"
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
            
            if self.config.refresh_rollout_buffer:
                self.rollout_buffer = []
            else:
                rollouts_path = dataset_path.parent / "rollouts"
                if rollouts_path.exists():
                    rollouts_dataset = load_from_disk(rollouts_path)
                    self.rollout_buffer = [Rollout(**row) for row in rollouts_dataset]
                else:
                    self.rollout_buffer = []
            
            replay_path = dataset_path.parent / "replay"
            if self.config.use_replay_buffer and not self.config.refresh_rollout_buffer and replay_path.exists():
                replay_dataset = load_from_disk(replay_path)
                self.replay_buffer = [Rollout(**row) for row in replay_dataset]
            else:
                self.replay_buffer = []

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
        
        replay_path = path.parent / "replay"
        if self.replay_buffer:
            Dataset.from_list(self.replay_buffer).save_to_disk(replay_path)

    def load(self, path: Path) -> None:
        """Loads metadata and rollouts from separate HF datasets. Uses the existing dataset stored in the buffer."""
        self._init_buffer(self.dataset, dataset_path=path)

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
            if problem_id not in self.metadata:
                continue
            if rollout["advantage"] == 0:
                new_difficulty = "easy" if rollout["reward"] == 1.0 else "hard"
            else:
                self.rollout_buffer.append(rollout)
                new_difficulty = "normal"
            self.metadata[problem_id]["difficulty"] = new_difficulty
            self.rollout_metrics[new_difficulty] += 1

    def sample_rollouts(self, n: int) -> list[Rollout]:
        """Samples `n` rollouts from the rollout and replay buffers (if enabled)."""
        sampled = []
        sampled_from_rollout = []

        if self.config.take_all_rollouts:
            n_from_rollout = len(self.rollout_buffer)
        else:
            n_from_rollout = min(n, len(self.rollout_buffer))

        if n_from_rollout > 0:
            sampled_from_rollout = self.rollout_buffer[-n_from_rollout:]
            self.rollout_buffer = self.rollout_buffer[:-n_from_rollout]
            sampled.extend(sampled_from_rollout)
    
        n_remaining = n - len(sampled)
        if self.config.use_replay_buffer and n_remaining > 0 and len(self.replay_buffer) > 0:
            n_from_replay = min(n_remaining, len(self.replay_buffer))
            selected_items = self.replay_buffer[-n_from_replay:]
            self.replay_buffer = self.replay_buffer[:-n_from_replay]
            sampled.extend(selected_items)
            self.replay_buffer = selected_items + self.replay_buffer
            
        if self.config.use_replay_buffer:
            self.replay_buffer.extend(sampled_from_rollout)
            
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
    