import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import cast

import verifiers as vf
from datasets import Dataset, load_from_disk

from prime_rl.orchestrator.config import BufferConfig
from prime_rl.utils.utils import mean_normalize
from prime_rl.utils.vf import from_serializable_state, to_serializable_state


class Buffer:
    """A buffer for storing rollouts and metadata."""

    POOLS = ["easy", "normal", "hard"]

    def __init__(self, dataset: Dataset, buffer_config: BufferConfig, env_names: list[str]):
        self.config = buffer_config
        if self.config.seed is not None:
            random.seed(self.config.seed)

        assert "example_id" in dataset.column_names, "The dataset must contain a `example_id` column."
        assert len(dataset) > 0, "The dataset must contain at least one problem."
        assert isinstance(dataset["example_id"][0], int), "The `example_id` column must be of type int."
        assert len(set(dataset["example_id"])) == len(dataset), "The `example_id` column must be unique."
        self.dataset = dataset

        self.env_names = env_names
        
        if self.config.env_ratios is not None:
            assert len(self.config.env_ratios) == len(self.env_names), (
                f"env_ratios length ({len(self.config.env_ratios)}) must match "
                f"number of environments ({len(self.env_names)})"
            )
            assert all(ratio >= 0 for ratio in self.config.env_ratios), "All env_ratios must be non-negative."
            assert not all(ratio == 0 for ratio in self.config.env_ratios), "At least one env_ratio must be non-zero."
            self.env_probabilities = mean_normalize(self.config.env_ratios) # Convert ratios to probabilities
        else:
            # Count problems per environment for uniform sampling across problems
            problem_counts = {env: 0 for env in self.env_names}
            for problem in dataset:
                env_name = problem["task"]
                problem_counts[env_name] += 1
            counts = [problem_counts[env] for env in self.env_names]
            self.env_probabilities = mean_normalize(counts)

        self._init_problem_pools()

        for problem in dataset:
            problem_dict = dict(problem)
            example_id = problem_dict["example_id"]
            env_name = problem_dict["task"]
            self.problem_buffer[env_name][example_id] = problem_dict

        self.rollout_buffer: list[vf.State] = []
        self._reset_step_metrics()

    def _init_problem_pools(self) -> None:
        self.problem_buffer: dict[str, dict[int, dict]] = {env: {} for env in self.env_names}
        self.easy_problems: dict[int, dict] = {}
        self.hard_problems: dict[int, dict] = {}

    def _reset_step_metrics(self):
        """Reset per-step metrics (called after get_metrics)."""
        zero_per_pool = lambda: {p: 0 for p in self.POOLS}
        self.num_problems_per_pool = {env: zero_per_pool() for env in self.env_names}
        self.num_rollouts_per_pool = {env: zero_per_pool() for env in self.env_names}

    def save(self, path: Path) -> None:
        """Saves pool assignments and rollouts."""
        path.mkdir(parents=True, exist_ok=True)

        easy_pairs = [[eid, p["task"]] for eid, p in self.easy_problems.items()]
        hard_pairs = [[eid, p["task"]] for eid, p in self.hard_problems.items()]

        (path / "easy.json").write_text(json.dumps(easy_pairs))
        (path / "hard.json").write_text(json.dumps(hard_pairs))

        rollouts_path = path / "rollouts"
        if self.rollout_buffer:
            serializable_rollouts = [to_serializable_state(rollout) for rollout in self.rollout_buffer]
            Dataset.from_list(list(map(dict, serializable_rollouts))).save_to_disk(rollouts_path)
        elif rollouts_path.exists():
            shutil.rmtree(rollouts_path)

    def _load_pool_ids(self, path: Path) -> set[tuple[int, str]]:
        """Load (example_id, task) pairs from JSON. Returns empty set if not found."""
        if not path.exists():
            return set()
        data = json.loads(path.read_text())
        return {(eid, task) for eid, task in data}

    def load(self, path: Path) -> None:
        """Loads pool assignments and rollouts."""
        easy_ids = self._load_pool_ids(path / "easy.json")
        hard_ids = self._load_pool_ids(path / "hard.json")

        self._init_problem_pools()

        for problem in self.dataset:
            problem_dict = dict(problem)
            example_id = problem_dict["example_id"]
            env_name = problem_dict["task"]
            key = (example_id, env_name)

            if key in easy_ids:
                self.easy_problems[example_id] = problem_dict
            elif key in hard_ids:
                self.hard_problems[example_id] = problem_dict
            else:
                self.problem_buffer[env_name][example_id] = problem_dict

        # Load rollouts, filtering out removed environments and problems
        rollouts_path = path / "rollouts"
        self.rollout_buffer = []
        if rollouts_path.exists():
            rollouts_dataset = load_from_disk(rollouts_path)
            valid_example_ids = set(self.dataset["example_id"])
            env_names_set = set(self.env_names)
            for row in rollouts_dataset:
                state = from_serializable_state(cast(dict, row))
                if state["task"] in env_names_set and state["example_id"] in valid_example_ids:
                    self.rollout_buffer.append(state)

        self.convert_difficulty_pools()

    def convert_difficulty_pools(self) -> None:
        """Converts a fraction of easy and hard problems to normal based on config."""
        self._convert_pool_to_normal(self.easy_problems, self.config.easy_fraction)
        self._convert_pool_to_normal(self.hard_problems, self.config.hard_fraction)

    def _convert_pool_to_normal(self, source_pool: dict[int, dict], fraction: float) -> None:
        """Moves a fraction of problems from the given pool back to normal."""
        if fraction <= 0.0 or not source_pool:
            return
        num_to_convert = round(len(source_pool) * fraction)
        if num_to_convert <= 0:
            return
        for pid in random.sample(list(source_pool), num_to_convert):
            problem = source_pool.pop(pid)
            env_name = problem["task"]
            self.problem_buffer[env_name][pid] = problem

    def sample_problems(self, n: int) -> list[dict]:
        """Samples `n` problems from the buffer."""
        non_empty = [env for env in self.env_names if self.problem_buffer[env]]
        if not non_empty:
            return []

        weights = [self.env_probabilities[self.env_names.index(env)] for env in non_empty]
        total_weight = sum(weights)
        if total_weight == 0:
            # All remaining environments correspond to zero-ratio entries, fall back to uniform sampling.
            normalized_weights = [1.0] * len(non_empty)
        else:
            normalized_weights = [w / total_weight for w in weights]
        sampled = []
        for env_name in random.choices(non_empty, weights=normalized_weights, k=n):
            sampled.append(random.choice(list(self.problem_buffer[env_name].values())))
        return sampled

    def update(self, rollouts: list[vf.State]):
        """Updates the buffer state with completed rollouts."""
        rollouts_by_example = defaultdict(list)
        for rollout in rollouts:
            rollouts_by_example[rollout["example_id"]].append(rollout)

        for problem_id, example_rollouts in rollouts_by_example.items():
            avg_reward = sum(r["reward"] for r in example_rollouts) / len(example_rollouts)
            env_name = example_rollouts[0]["task"]

            if self.config.easy_threshold is not None and avg_reward >= self.config.easy_threshold:
                pool = "easy"
            elif self.config.hard_threshold is not None and avg_reward <= self.config.hard_threshold:
                pool = "hard"
            else:
                pool = "normal"

            if pool != "normal" and problem_id in self.problem_buffer[env_name]:
                problem = self.problem_buffer[env_name].pop(problem_id)
                target_pool = self.easy_problems if pool == "easy" else self.hard_problems
                target_pool[problem_id] = problem
            self.num_problems_per_pool[env_name][pool] += 1
            if self.config.online_difficulty_filtering:
                if avg_reward == 0.0:
                    self.num_rollouts_per_pool[env_name]["hard"] += len(example_rollouts)
                    continue
                elif avg_reward == 1.0:
                    self.num_rollouts_per_pool[env_name]["easy"] += len(example_rollouts)
                    continue
            self.num_rollouts_per_pool[env_name]["normal"] += len(example_rollouts)
            self.rollout_buffer.extend(example_rollouts)

    def sample_rollouts(self, n: int) -> list[vf.State]:
        """Samples the latest `n` rollouts from the buffer."""
        n = min(n, len(self.rollout_buffer))
        sampled = self.rollout_buffer[-n:]
        self.rollout_buffer = self.rollout_buffer[:-n]
        return sampled

    def get_metrics(self) -> dict[str, float]:
        metrics = {}

        for env_name in self.env_names:
            env_suffix = f"/{env_name}"
            problems = self.num_problems_per_pool[env_name]
            rollouts = self.num_rollouts_per_pool[env_name]
            num_problems = sum(problems.values())
            num_rollouts = sum(rollouts.values())

            for pool in ["easy", "hard"]:
                if num_problems > 0:
                    metrics[f"buffer/evicted_problems/{pool}{env_suffix}"] = problems[pool] / num_problems
                if num_rollouts > 0:
                    metrics[f"buffer/filtered_rollouts/{pool}{env_suffix}"] = rollouts[pool] / num_rollouts

        total_normal = sum(len(self.problem_buffer[env]) for env in self.env_names)
        pool_counts = [len(self.easy_problems), total_normal, len(self.hard_problems)]
        for pool, ratio in zip(self.POOLS, mean_normalize(pool_counts)):
            metrics[f"buffer/pool/{pool}"] = ratio

        self._reset_step_metrics()
        return metrics
