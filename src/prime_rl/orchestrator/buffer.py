import hashlib
import json
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Literal, cast

import verifiers as vf
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
        assert len(self.dataset) > 0, "The dataset must contain at least one example."
        assert isinstance(self.dataset["example_id"][0], int), "The `example_id` column must be of type int."
        assert len(set(self.dataset["example_id"])) == len(self.dataset), "The `example_id` column must be unique."
        assert set(self.dataset["task"]) == set(self.env_names), "The `task` column must contain all environment names."

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
                f"Sampling {dataset_type} buffer according to provided environment ratios ({', '.join(f'{k}={v:.2f}' for k, v in self.env_probs.items())})"
            )
        else:
            # Count examples per environment to sample according to natural env distribution
            env_counts = [len(self.example_buffer[env_name]) for env_name in self.env_names]
            env_ratio = mean_normalize(env_counts)
            self.env_probs = {env_name: ratio for env_name, ratio in zip(self.env_names, env_ratio)}
            self.logger.debug(
                f"Sampling {dataset_type} buffer according to natural environment distribution ({', '.join(f'{k}={v:.2f}' for k, v in self.env_probs.items())})"
            )

        # Initialize buffers for easy/ hard examples
        self.easy_examples: list[dict] = []
        self.hard_examples: list[dict] = []

        # Initialize rollout buffer (flat list of rollouts)
        self.rollout_buffer: list[vf.State] = []

        self.reset_step_metrics()

    def get_example_hash(self, example: dict) -> str:
        """Returns a hash of the example based on hash keys."""
        hash_keys = [key for key in self.config.hash_keys if key in example]
        assert hash_keys, "No hashable keys found in example."
        return hashlib.sha256(json.dumps([example[key] for key in hash_keys]).encode()).hexdigest()

    def save(self, path: Path) -> None:
        """Saves pool assignments and rollout buffer."""
        path.mkdir(parents=True, exist_ok=True)

        def write_jsonl(lst: list[dict], path: Path) -> None:
            with open(path, "w") as f:
                for item in lst:
                    f.write(json.dumps(item) + "\n")

        write_jsonl(self.easy_examples, path / "easy_examples.jsonl")
        write_jsonl(self.hard_examples, path / "hard_examples.jsonl")

        serializable_rollouts = [to_serializable_state(rollout) for rollout in self.rollout_buffer]
        write_jsonl(serializable_rollouts, path / "rollout_buffer.jsonl")

    def load(self, path: Path) -> None:
        """Loads pool assignments and rollouts."""

        def read_jsonl(path: Path) -> list[dict]:
            with open(path, "r") as f:
                return [json.loads(line) for line in f]

        saved_easy_examples = read_jsonl(path / "easy_examples.jsonl")
        saved_hard_examples = read_jsonl(path / "hard_examples.jsonl")
        saved_rollout_buffer = [
            from_serializable_state(rollout) for rollout in read_jsonl(path / "rollout_buffer.jsonl")
        ]

        if any(saved_easy_examples) or any(saved_hard_examples) or any(saved_rollout_buffer):
            # Build hash lookup for example buffer (env -> (example_hash -> example_id))
            example_hash_lookup = defaultdict(dict)
            all_hashes = set()
            for env in self.example_buffer:
                for example_id, example in self.example_buffer[env].items():
                    example_hash = self.get_example_hash(example)
                    if example_hash in all_hashes:
                        self.logger.warning(
                            f"Duplicate example hash found based on hash_keys={self.config.hash_keys}. Overwriting with latest example. This may cause unexpected behavior when resuming the buffer."
                        )
                    example_hash_lookup[env][example_hash] = example_id
                    all_hashes.add(example_hash)

            def move_saved_pool(saved_examples: list[dict], target_pool: list[dict]) -> int:
                """Moves saved examples to the target pool from example buffer based on hash lookup."""
                num_moved = 0
                for example in saved_examples:
                    example_hash = self.get_example_hash(example)
                    for env in example_hash_lookup:
                        if example_hash in example_hash_lookup[env]:
                            example_id = example_hash_lookup[env][example_hash]
                            example = self.example_buffer[env].pop(example_id, None)
                            if example is not None:
                                target_pool.append(example)
                                num_moved += 1
                                break
                return num_moved

            if any(saved_easy_examples):
                num_moved = move_saved_pool(saved_easy_examples, self.easy_examples)
                self.logger.debug(
                    f"Loaded {num_moved}/{len(saved_easy_examples)} example(s) to easy pool from checkpoint."
                )
                if num_moved != len(saved_easy_examples):
                    num_not_moved = len(saved_easy_examples) - num_moved
                    self.logger.warning(
                        f"Could not move {num_not_moved} example(s) from checkpoint to easy pool. This usually means you resumed with an env mix that does not contain all previous examples."
                    )

            if any(saved_hard_examples):
                num_moved = move_saved_pool(saved_hard_examples, self.hard_examples)
                self.logger.debug(
                    f"Moved {num_moved}/{len(saved_hard_examples)} example(s) to hard pool from checkpoint."
                )
                if num_moved != len(saved_hard_examples):
                    num_not_moved = len(saved_hard_examples) - num_moved
                    self.logger.warning(
                        f"Could not move {num_not_moved} example(s) from checkpoint to hard pool. This usually means you resumed with an env mix that does not contain all previous examples."
                    )

            if any(saved_rollout_buffer):
                # Extend rollout buffer, but only include rollouts for which the example still exists in the example buffer
                valid_saved_rollouts = [
                    rollout for rollout in saved_rollout_buffer if rollout["task"] in self.env_names
                ]
                self.rollout_buffer.extend(valid_saved_rollouts)
                self.logger.debug(f"Loaded {len(valid_saved_rollouts)} rollout(s) from checkpoint.")

            # Load rollouts, filtering out removed environments and problems
            def convert_examples_to_normal(examples: list[dict], fraction: float) -> int:
                """Moves a fraction of examples from the given pool back to normal."""
                if fraction <= 0.0 or not examples:
                    return 0
                num_moved = round(len(examples) * fraction)
                if num_moved <= 0:
                    return 0
                for _ in range(num_moved):
                    example = random.choice(examples)
                    env_name = example["task"]
                    example_id = example["example_id"]
                    examples.remove(example)
                    self.example_buffer[env_name][example_id] = example
                return num_moved

            num_easy_examples = len(self.easy_examples)
            num_moved = convert_examples_to_normal(self.easy_examples, self.config.easy_fraction)
            self.logger.debug(f"Converted {num_moved}/{num_easy_examples} example(s) back to normal from easy pool.")
            num_hard_examples = len(self.hard_examples)
            num_moved = convert_examples_to_normal(self.hard_examples, self.config.hard_fraction)
            self.logger.debug(f"Converted {num_moved}/{num_hard_examples} example(s) back to normal from hard pool.")
        else:
            self.logger.debug("No easy/ hard examples or rollouts found in checkpoint")

    def sample_examples(self, n: int) -> list[dict]:
        """Samples n examples from the buffer, respecting env ratios."""

        non_empty_envs = [env for env, examples in self.example_buffer.items() if examples]

        if not non_empty_envs:
            raise ValueError("No environments left with examples.")

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
                target_pool.append(example)

            self.num_examples_per_step[env_name][pool] += 1
            if self.config.online_difficulty_filtering:
                if avg_reward == 0.0:
                    self.num_rollouts_per_step[env_name]["hard"] += len(example_rollouts)
                    continue
                elif avg_reward == 1.0:
                    self.num_rollouts_per_step[env_name]["easy"] += len(example_rollouts)
                    continue

            self.num_rollouts_per_step[env_name]["normal"] += len(example_rollouts)
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
        # num examples per env per step per pool (env_name -> (pool -> num_examples))
        self.num_examples_per_step = {env: zero_per_pool() for env in self.env_names}
        # num rollouts per env per step per pool (env_name -> (pool -> num_rollouts))
        self.num_rollouts_per_step = {env: zero_per_pool() for env in self.env_names}

    def get_metrics(self) -> dict[str, float]:
        """Returns the buffer metrics for the current step."""

        metrics = {}

        # sum over envs (e.g. log globally)
        num_examples_per_step_per_pool = {
            pool: sum(self.num_examples_per_step[env][pool] for env in self.env_names) for pool in self.POOLS
        }
        num_rollouts_per_step_per_pool = {
            pool: sum(self.num_rollouts_per_step[env][pool] for env in self.env_names) for pool in self.POOLS
        }
        num_examples_per_step = sum(num_examples_per_step_per_pool.values())
        num_rollouts_per_step = sum(num_rollouts_per_step_per_pool.values())

        for pool in ["easy", "hard"]:
            if num_examples_per_step:
                metrics[f"evicted_examples/{pool}"] = num_examples_per_step_per_pool[pool] / num_examples_per_step
            if num_rollouts_per_step:
                metrics[f"filtered_rollouts/{pool}"] = num_rollouts_per_step_per_pool[pool] / num_rollouts_per_step

        total_normal = sum(len(self.example_buffer[env]) for env in self.env_names)
        pool_counts = [len(self.easy_examples), total_normal, len(self.hard_examples)]
        pool_ratios = mean_normalize(pool_counts)
        for pool, pool_ratio in zip(self.POOLS, pool_ratios):
            metrics[f"pool/{pool}"] = pool_ratio

        self.reset_step_metrics()

        return metrics
