import json
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
    Buffer that samples problems from difficulty pools (easy/normal/hard) and filters
    rollouts by reward range.
    """

    def __init__(self, dataset: Dataset, buffer_config: BufferConfig):
        self.logger = get_logger()
        self.config = buffer_config
        if self.config.seed is not None:
            random.seed(self.config.seed)

        # Initialize buffer
        self._init_buffer(dataset, self.config.from_scratch)
        self.problem_metrics = {"easy": 0, "normal": 0, "hard": 0}
        self.rollout_metrics = {"too_easy": 0, "rollouts_sampled": 0, "too_hard": 0}

    def _init_buffer(self, dataset: Dataset, from_scratch: bool) -> None:
        """Initializes the buffer state from a dataset."""
        # Store problem IDs
        if "id" not in dataset.column_names:
            dataset = dataset.add_column("id", list(range(len(dataset))), new_fingerprint=str(uuid.uuid4()))
        self.problem_ids = dataset["id"]

        if from_scratch:
            self.rollout_buffer: dict[int, list[Rollout]] = {}
            self.metadata = {pid: {"difficulty": "normal"} for pid in self.problem_ids}
        else:
            if not all(col in dataset.column_names for col in ("metadata", "rollouts")):
                raise ValueError("Dataset must contain `metadata` and `rollouts` columns when `from_scratch=False`")
            self.metadata = {pid: json.loads(md) for pid, md in zip(self.problem_ids, dataset["metadata"])}
            self.rollout_buffer = {}
            for pid, rollouts_json in zip(self.problem_ids, dataset["rollouts"]):
                rollouts_parsed = json.loads(rollouts_json)
                if rollouts_parsed:
                    self.rollout_buffer[pid] = [Rollout(**r) for r in rollouts_parsed]
            dataset = dataset.remove_columns(["metadata", "rollouts"])
            for pid, md in self.metadata.items():
                if md.get("difficulty") not in ["easy", "normal", "hard"]:
                    raise ValueError(f"Invalid difficulty {md.get('difficulty')} for problem {pid}")

        self.dataset = dataset
        self.problem_buffer = {pid: dict(problem) for pid, problem in zip(self.problem_ids, dataset)}

    def save(self, path: Path) -> None:
        """Saves the buffer state to a single HF dataset."""
        dataset = self.dataset.remove_columns([c for c in ("metadata", "rollouts") if c in self.dataset.column_names])
        rollout_buffer = {pid: self.rollout_buffer.get(pid, []) for pid in self.problem_ids}
        dataset = dataset.add_column(
            "metadata", [json.dumps(self.metadata[pid]) for pid in self.problem_ids], new_fingerprint="metadata-ckpt"
        )
        dataset = dataset.add_column(
            "rollouts", [json.dumps(rollout_buffer[pid]) for pid in self.problem_ids], new_fingerprint="rollouts-ckpt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(path)

    def load(self, path: Path) -> None:
        """Loads the buffer state from a single HF dataset."""
        self.dataset = load_from_disk(path)
        self._init_buffer(self.dataset, from_scratch=False)

    def sample_problems(self, n: int) -> tuple[list[dict], dict[str, int]]:
        """Samples `n` problems from the dataset using difficulty pools. Returns problems and stats."""
        n_easy = int(n * self.config.easy_fraction)
        n_hard = int(n * self.config.hard_fraction)
        n_normal = n - n_easy - n_hard

        # Group problem IDs by difficulty
        by_difficulty = {"easy": [], "normal": [], "hard": []}
        for problem_id, metadata in self.metadata.items():
            by_difficulty[metadata["difficulty"]].append(problem_id)

        # Sample from each pool, falling back to normal if insufficient
        def sample_from_pool(pool_ids: list[int], target: int, pool_name: str) -> tuple[list[int], int]:
            sampled = min(target, len(pool_ids))
            if sampled < target:
                self.logger.warning(
                    f"Only {sampled} {pool_name} problem(s) available, sampling {target - sampled} normal problem(s) more"
                )
            return (random.sample(pool_ids, sampled) if sampled > 0 else [], target - sampled)

        sampled_easy, deficit = sample_from_pool(by_difficulty["easy"], n_easy, "easy")
        n_normal += deficit
        sampled_hard, deficit = sample_from_pool(by_difficulty["hard"], n_hard, "hard")
        n_normal += deficit
        sampled_normal, _ = sample_from_pool(by_difficulty["normal"], n_normal, "normal")

        sampled_problem_ids = sampled_easy + sampled_normal + sampled_hard
        self.problem_metrics["easy"] += len(sampled_easy)
        self.problem_metrics["normal"] += len(sampled_normal)
        self.problem_metrics["hard"] += len(sampled_hard)
        return [self.problem_buffer[pid] for pid in sampled_problem_ids]

    def update(self, rollouts: list[Rollout]):
        """Updates the buffer state with completed rollouts."""
        rollouts_by_problem_id = defaultdict(list)
        for rollout in rollouts:
            rollouts_by_problem_id[rollout["example_id"]].append(rollout)

        self.rollout_buffer = rollouts_by_problem_id

        stats = Counter()
        for problem_id, problem_rollouts in rollouts_by_problem_id.items():
            reward = sum(r["reward"] for r in problem_rollouts) / len(problem_rollouts)
            new_difficulty = (
                "easy" if reward > self.config.easy_border else "hard" if reward < self.config.hard_border else "normal"
            )
            old_difficulty = self.metadata[problem_id].get("difficulty", "normal")
            if old_difficulty != new_difficulty:
                stats[(old_difficulty, new_difficulty)] += 1
            self.metadata[problem_id].update({"reward": reward, "difficulty": new_difficulty})

        if stats:
            self.logger.debug(", ".join([f"{v} problem(s) moved from `{k[0]}` to `{k[1]}`" for k, v in stats.items()]))

    def sample_rollouts(self, n: int) -> tuple[list[Rollout], dict[str, int]]:
        """Samples rollouts for `n` problems within the configured reward range. Returns rollouts and stats."""
        sampled_rollouts, num_sampled = [], 0
        num_too_hard, num_too_easy = 0, 0

        for problem_id, rollouts in list(self.rollout_buffer.items()):
            if num_sampled >= n:
                break
            reward = self.metadata[problem_id]["reward"]
            if (self.config.min_reward is not None and reward <= self.config.min_reward) or (
                self.config.max_reward is not None and reward >= self.config.max_reward
            ):
                if self.config.min_reward is not None and reward <= self.config.min_reward:
                    num_too_hard += 1
                else:
                    num_too_easy += 1
                continue
            sampled_rollouts.extend(self.rollout_buffer.pop(problem_id))
            num_sampled += 1

        self.rollout_metrics["too_hard"] += num_too_hard
        self.rollout_metrics["too_easy"] += num_too_easy
        self.rollout_metrics["rollouts_sampled"] += num_sampled
        return sampled_rollouts

    def _get_normalized_metrics(self, metrics: dict[str, int], prefix: str) -> dict[str, float]:
        """Helper method to normalize metrics and format them for logging."""
        count_total = sum(metrics.values())
        return {
            f"{prefix}/{key}": count / count_total if count_total > 0 else 0
            for key, count in metrics.items()
        }

    def get_metrics(self) -> dict[str, float]:
        return {
            **self._get_normalized_metrics(self.problem_metrics, "problem_metrics"),
            **self._get_normalized_metrics(self.rollout_metrics, "rollout_metrics"),
            **self._get_normalized_metrics(self.metadata, "metadata_metrics"),
        }
    