import json
import random
import shutil
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Literal, cast

import verifiers as vf
from datasets import Dataset, load_from_disk

from prime_rl.orchestrator.config import BufferConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import format_num, mean, mean_normalize
from prime_rl.utils.vf import from_serializable_state, to_serializable_state


class Buffer:
    """A buffer for storing rollouts and metadata."""

    POOLS = ["easy", "normal", "hard"]

    def __init__(
        self, env_group: vf.EnvGroup, buffer_config: BufferConfig, dataset_type: Literal["train", "val"] = "train"
    ):
        self.env_group = env_group
        self.config = buffer_config
        self.dataset_type = dataset_type
        self.logger = get_logger()

        if self.config.seed is not None:
            random.seed(self.config.seed)

        if self.dataset_type == "train":
            self.dataset = env_group.get_dataset(seed=self.config.seed)
        elif self.dataset_type == "val":
            self.dataset = env_group.get_eval_dataset(seed=self.config.seed)
        else:
            raise ValueError(f"Invalid dataset type: {self.dataset_type}")

        self.env_names = env_group.env_names

        # Basic assertions
        assert "example_id" in self.dataset.column_names, "The dataset must contain a `example_id` column."
        assert "prompt" in self.dataset.column_names, "The dataset must contain a `prompt` column."
        assert "task" in self.dataset.column_names, "The dataset must contain a `task` column."
        assert len(self.dataset) > 0, "The dataset must contain at least one problem."
        assert isinstance(self.dataset["example_id"][0], int), "The `example_id` column must be of type int."
        assert len(set(self.dataset["example_id"])) == len(self.dataset), "The `example_id` column must be unique."
        assert sorted(set(self.dataset["task"])) == self.env_names, (
            "The `task` column must contain all environment names."
        )

        # Initialize example buffer (env_name -> (example_id -> example))
        self.example_buffer: dict[str, dict[int, dict]] = defaultdict(dict)
        for example in map(partial(cast, dict), self.dataset):
            self.example_buffer[example["task"]][example["example_id"]] = example
        assert len(self.example_buffer) == len(self.env_names)
        self.logger.debug(
            f"Initialized {dataset_type} buffer with {format_num(len(self.dataset), precision=0)} example(s) in {len(self.env_names)} environment(s)"
        )

        if self.config.env_ratios is not None:
            # Convert ratios to probabilities
            env_ratio = mean_normalize(self.config.env_ratios)
            self.env_probs = {env_name: ratio for env_name, ratio in zip(self.env_names, env_ratio)}
            self.logger.debug(
                f"Sampling {dataset_type} buffer according to provided environment ratios: {', '.join(f'{k}={100 * v:.1f}%' for k, v in self.env_probs.items())}"
            )
        else:
            # Count problems per environment for uniform sampling across problems
            env_counts = [len(self.example_buffer[env_name]) for env_name in self.env_names]
            env_ratio = mean_normalize(env_counts)
            self.env_probs = {env_name: ratio for env_name, ratio in zip(self.env_names, env_ratio)}
            self.logger.debug(
                f"Sampling {dataset_type} buffer according to natural environment distribution: {', '.join(f'{k}={100 * v:.1f}%' for k, v in self.env_probs.items())}"
            )

        # Initialize buffers for easy/ hard examples (example_id -> example)
        self.easy_examples: dict[int, dict] = {}
        self.hard_examples: dict[int, dict] = {}

        # Initialize rollout buffer (flat list of rollouts)
        self.rollout_buffer: list[vf.State] = []

        self.reset_step_metrics()

    def save(self, path: Path) -> None:
        """Saves pool assignments and rollouts."""
        path.mkdir(parents=True, exist_ok=True)

        easy_pairs = [[eid, p["task"]] for eid, p in self.easy_examples.items()]
        hard_pairs = [[eid, p["task"]] for eid, p in self.hard_examples.items()]

        (path / "easy.json").write_text(json.dumps(easy_pairs))
        (path / "hard.json").write_text(json.dumps(hard_pairs))

        rollouts_path = path / "rollouts"
        if self.rollout_buffer:
            serializable_rollouts = [to_serializable_state(rollout) for rollout in self.rollout_buffer]
            Dataset.from_list(list(map(dict, serializable_rollouts))).save_to_disk(rollouts_path)
        elif rollouts_path.exists():
            shutil.rmtree(rollouts_path)

    def load(self, path: Path) -> None:
        """Loads pool assignments and rollouts."""

        def load_pool_ids(path: Path) -> set[tuple[int, str]]:
            """Load (example_id, task) pairs from JSON. Returns empty set if not found."""
            if not path.exists():
                return set()
            data = json.loads(path.read_text())
            return {(eid, task) for eid, task in data}

        easy_ids = load_pool_ids(path / "easy.json")
        hard_ids = load_pool_ids(path / "hard.json")

        # self.init_empty_buffers()

        for problem in self.dataset:
            problem_dict = dict(problem)
            example_id = problem_dict["example_id"]
            env_name = problem_dict["task"]
            key = (example_id, env_name)

            if key in easy_ids:
                self.easy_examples[example_id] = problem_dict
            elif key in hard_ids:
                self.hard_examples[example_id] = problem_dict
            else:
                self.example_buffer[env_name][example_id] = problem_dict

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

        def convert_pool_to_normal(source_pool: dict[int, dict], fraction: float) -> None:
            """Moves a fraction of problems from the given pool back to normal."""
            if fraction <= 0.0 or not source_pool:
                return
            num_to_convert = round(len(source_pool) * fraction)
            if num_to_convert <= 0:
                return
            for pid in random.sample(list(source_pool), num_to_convert):
                problem = source_pool.pop(pid)
                env_name = problem["task"]
                self.example_buffer[env_name][pid] = problem

        convert_pool_to_normal(self.easy_examples, self.config.easy_fraction)
        convert_pool_to_normal(self.hard_examples, self.config.hard_fraction)

    def sample_problems(self, n: int) -> list[dict]:
        """Samples n problems from the buffer, respecting env ratios."""

        non_empty_envs = [env for env, examples in self.example_buffer.items() if examples]

        if not non_empty_envs:
            self.logger.warning("No environments with problems. Returning no problems.")
            return []

        non_empty_env_probs = [self.env_probs[env] for env in non_empty_envs]
        sampled_examples = []
        for sampled_env in random.choices(non_empty_envs, weights=non_empty_env_probs, k=n):
            sampled_example = random.choice(list(self.example_buffer[sampled_env].values()))
            sampled_examples.append(sampled_example)

        return sampled_examples

    def update(self, rollouts: list[vf.State]):
        """Updates the buffer state with completed rollouts."""

        rollouts_by_example = defaultdict(list)
        for rollout in rollouts:
            rollouts_by_example[rollout["example_id"]].append(rollout)

        for example_id, example_rollouts in rollouts_by_example.items():
            avg_reward = mean([r["reward"] for r in example_rollouts])
            env_name = example_rollouts[0]["task"]

            if self.config.easy_threshold is not None and avg_reward >= self.config.easy_threshold:
                pool = "easy"
            elif self.config.hard_threshold is not None and avg_reward <= self.config.hard_threshold:
                pool = "hard"
            else:
                pool = "normal"

            if pool != "normal" and example_id in self.example_buffer[env_name]:
                example = self.example_buffer[env_name].pop(example_id)
                target_pool = self.easy_examples if pool == "easy" else self.hard_examples
                target_pool[example_id] = example

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
        """Samples the latest n rollouts from the buffer."""
        n = min(n, len(self.rollout_buffer))
        sampled_rollouts = self.rollout_buffer[-n:]
        self.rollout_buffer = self.rollout_buffer[:-n]
        return sampled_rollouts

    def reset_step_metrics(self) -> None:
        """Reset per-step metrics (called after get_metrics)."""
        zero_per_pool = lambda: {p: 0 for p in self.POOLS}
        self.num_problems_per_pool = {env: zero_per_pool() for env in self.env_names}
        self.num_rollouts_per_pool = {env: zero_per_pool() for env in self.env_names}

    def get_metrics(self) -> dict[str, float]:
        """Returns the buffer metrics for the current step."""

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

        total_normal = sum(len(self.example_buffer[env]) for env in self.env_names)
        pool_counts = [len(self.easy_examples), total_normal, len(self.hard_examples)]
        for pool, ratio in zip(self.POOLS, mean_normalize(pool_counts)):
            metrics[f"buffer/pool/{pool}"] = ratio

        self.reset_step_metrics()

        return metrics
