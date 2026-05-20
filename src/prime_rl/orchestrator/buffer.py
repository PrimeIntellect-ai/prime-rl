from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import verifiers as vf
from verifiers.utils.save_utils import make_serializable

from prime_rl.configs.orchestrator import BufferConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import format_num, mean, mean_normalize

if TYPE_CHECKING:
    from prime_rl.orchestrator.envs import TrainEnv, TrainEnvs


POOLS = ["easy", "normal", "hard"]


class _EnvBuffer:
    """Manages examples and difficulty pools for a single env."""

    def __init__(self, env: TrainEnv, config: BufferConfig):
        self.env_name = env.name
        self.config = config

        dataset = env.get_dataset(seed=config.seed)
        if "example_id" not in dataset.column_names:
            dataset = dataset.map(lambda ex, idx: {**ex, "example_id": idx}, with_indices=True)

        assert len(dataset) > 0, f"Dataset for {env.name} must contain at least one example."
        assert "example_id" in dataset.column_names, f"Dataset for {env.name} must contain an `example_id` column."
        assert "prompt" in dataset.column_names, f"Dataset for {env.name} must contain a `prompt` column."

        self.examples: dict[int | str, dict] = {}
        for example in map(partial(cast, dict), dataset):
            example["env_name"] = env.name
            self.examples[example["example_id"]] = example

        self.easy_examples: list[dict] = []
        self.hard_examples: list[dict] = []

        self.reset_step_metrics()

    @property
    def num_normal(self) -> int:
        return len(self.examples)

    @property
    def num_total(self) -> int:
        return self.num_normal + len(self.easy_examples) + len(self.hard_examples)

    def sample_example(self) -> dict:
        key = random.choice(tuple(self.examples))
        return self.examples[key]

    def get_example_hash(self, example: dict) -> str:
        hash_keys = [key for key in self.config.hash_keys if key in example]
        assert hash_keys, "No hashable keys found in example."
        return hashlib.sha256(json.dumps([example[key] for key in hash_keys]).encode()).hexdigest()

    def update_pools(self, example_id: int | str, avg_reward: float) -> str:
        """Assign example to pool based on reward. Returns pool name."""
        if self.config.easy_threshold is not None and avg_reward >= self.config.easy_threshold:
            pool = "easy"
        elif self.config.hard_threshold is not None and avg_reward <= self.config.hard_threshold:
            pool = "hard"
        else:
            pool = "normal"

        if pool != "normal" and example_id in self.examples:
            example = self.examples.pop(example_id)
            target = self.easy_examples if pool == "easy" else self.hard_examples
            target.append(example)

        self.num_examples_per_step[pool] += 1
        return pool

    def reset_step_metrics(self) -> None:
        zero = lambda: {p: 0 for p in POOLS}
        self.num_examples_per_step = zero()
        self.num_rollouts_per_step = zero()
        self.num_zero_std_rollouts_per_step = 0

    def get_metrics(self) -> dict[str, float]:
        metrics = {}
        num_examples = sum(self.num_examples_per_step.values())
        num_rollouts = sum(self.num_rollouts_per_step.values()) + self.num_zero_std_rollouts_per_step

        for pool in ["easy", "hard"]:
            if num_examples:
                metrics[f"evicted_examples/{self.env_name}/{pool}"] = self.num_examples_per_step[pool] / num_examples
            if num_rollouts:
                metrics[f"filtered_rollouts/{self.env_name}/{pool}"] = self.num_rollouts_per_step[pool] / num_rollouts
        if num_rollouts:
            metrics[f"filtered_rollouts/{self.env_name}/zero_std"] = (
                self.num_zero_std_rollouts_per_step / num_rollouts
            )

        pool_counts = [len(self.easy_examples), self.num_normal, len(self.hard_examples)]
        pool_ratios = mean_normalize(pool_counts)
        for pool, ratio in zip(POOLS, pool_ratios):
            metrics[f"pool/{self.env_name}/{pool}"] = ratio

        self.reset_step_metrics()
        return metrics


class Buffer:
    """Manages multiple Buffers with env-ratio-aware sampling."""

    def __init__(self, envs: TrainEnvs, config: BufferConfig):
        self.config = config
        self.logger = get_logger()

        if config.seed is not None:
            random.seed(config.seed)

        self.env_buffers: dict[str, _EnvBuffer] = {}
        for env in envs:
            self.env_buffers[env.name] = _EnvBuffer(env, config)
        self.env_names = envs.names

        total = sum(eb.num_total for eb in self.env_buffers.values())
        self.logger.debug(
            f"Initialized buffer with {format_num(total, precision=0)} example(s) "
            f"in {len(self.env_names)} environment(s)"
        )

        env_ratios = [env.config.ratio for env in envs]
        if any(r is not None for r in env_ratios):
            env_ratio = mean_normalize(env_ratios)
            self.env_probs = dict(zip(self.env_names, env_ratio))
            self.logger.debug(
                f"Sampling buffer according to provided environment ratios "
                f"({', '.join(f'{k}={v:.2f}' for k, v in self.env_probs.items())})"
            )
        else:
            env_counts = [self.env_buffers[name].num_normal for name in self.env_names]
            env_ratio = mean_normalize(env_counts)
            self.env_probs = dict(zip(self.env_names, env_ratio))
            self.logger.debug(
                f"Sampling buffer according to natural environment distribution "
                f"({', '.join(f'{k}={v:.2f}' for k, v in self.env_probs.items())})"
            )

        self.rollout_buffer: list[vf.RolloutOutput] = []

    def sample_examples(self, n: int) -> list[dict]:
        """Samples n examples across envs, respecting env ratios."""
        non_empty = [name for name, eb in self.env_buffers.items() if eb.examples]
        if not non_empty:
            raise ValueError("No environments left with examples.")

        weights = [self.env_probs[name] for name in non_empty]
        return [self.env_buffers[name].sample_example() for name in random.choices(non_empty, weights=weights, k=n)]

    def update(self, rollouts: list[vf.RolloutOutput]):
        """Updates buffer state with completed rollouts."""
        rollouts_by_example = defaultdict(list)
        for rollout in rollouts:
            rollouts_by_example[(rollout["env_name"], rollout["example_id"])].append(rollout)

        for (env_name, example_id), example_rollouts in rollouts_by_example.items():
            eb = self.env_buffers[env_name]
            rewards = [r["reward"] for r in example_rollouts]
            avg_reward = mean(rewards)
            pool = eb.update_pools(example_id, avg_reward)

            if self.config.online_difficulty_filtering:
                if len(set(rewards)) == 1:
                    eb.num_zero_std_rollouts_per_step += len(example_rollouts)
                    continue
                if pool != "normal":
                    eb.num_rollouts_per_step[pool] += len(example_rollouts)
                    continue

            eb.num_rollouts_per_step["normal"] += len(example_rollouts)
            self.rollout_buffer.extend(example_rollouts)

    def sample_rollouts(self, n: int) -> list[vf.RolloutOutput]:
        """Samples the latest n rollouts from the buffer."""
        n = min(n, len(self.rollout_buffer))
        sampled = self.rollout_buffer[-n:]
        self.rollout_buffer = self.rollout_buffer[:-n]
        return sampled

    def save(self, path: Path) -> None:
        """Saves pool assignments and rollout buffer."""
        path.mkdir(parents=True, exist_ok=True)

        def write_jsonl(lst: list, filepath: Path) -> None:
            with open(filepath, "w") as f:
                for item in lst:
                    f.write(json.dumps(item, default=make_serializable) + "\n")

        all_easy = [ex for eb in self.env_buffers.values() for ex in eb.easy_examples]
        all_hard = [ex for eb in self.env_buffers.values() for ex in eb.hard_examples]
        write_jsonl(all_easy, path / "easy_examples.jsonl")
        write_jsonl(all_hard, path / "hard_examples.jsonl")
        write_jsonl(self.rollout_buffer, path / "rollout_buffer.jsonl")

    def load(self, path: Path) -> None:
        """Loads pool assignments and rollouts from checkpoint."""

        def read_jsonl(filepath: Path) -> list[dict]:
            with open(filepath, "r") as f:
                return [json.loads(line) for line in f]

        saved_easy = read_jsonl(path / "easy_examples.jsonl")
        saved_hard = read_jsonl(path / "hard_examples.jsonl")
        saved_rollouts = cast(list[vf.RolloutOutput], read_jsonl(path / "rollout_buffer.jsonl"))

        if not any(saved_easy) and not any(saved_hard) and not any(saved_rollouts):
            self.logger.debug("No easy/ hard examples or rollouts found in checkpoint")
            return

        # Build hash lookup across all env buffers: env -> (hash -> example_id)
        hash_lookup: dict[str, dict[str, int]] = defaultdict(dict)
        all_hashes: set[str] = set()
        for env_name, eb in self.env_buffers.items():
            for example_id, example in eb.examples.items():
                h = eb.get_example_hash(example)
                if h in all_hashes:
                    self.logger.warning(
                        f"Duplicate example hash found based on hash_keys={self.config.hash_keys}. "
                        "Overwriting with latest example. This may cause unexpected behavior when resuming the buffer."
                    )
                hash_lookup[env_name][h] = example_id
                all_hashes.add(h)

        def move_saved_pool(saved_examples: list[dict], pool_name: str) -> int:
            num_moved = 0
            for example in saved_examples:
                # Use any env buffer to compute hash (hash_keys are config-level)
                first_eb = next(iter(self.env_buffers.values()))
                h = first_eb.get_example_hash(example)
                for env_name, env_hashes in hash_lookup.items():
                    if h in env_hashes:
                        example_id = env_hashes[h]
                        eb = self.env_buffers[env_name]
                        matched = eb.examples.pop(example_id, None)
                        if matched is not None:
                            target = eb.easy_examples if pool_name == "easy" else eb.hard_examples
                            target.append(matched)
                            num_moved += 1
                            break
            return num_moved

        if any(saved_easy):
            num_moved = move_saved_pool(saved_easy, "easy")
            self.logger.debug(f"Loaded {num_moved}/{len(saved_easy)} example(s) to easy pool from checkpoint.")
            if num_moved != len(saved_easy):
                self.logger.warning(
                    f"Could not move {len(saved_easy) - num_moved} example(s) from checkpoint to easy pool. "
                    "This usually means you resumed with an env mix that does not contain all previous examples."
                )

        if any(saved_hard):
            num_moved = move_saved_pool(saved_hard, "hard")
            self.logger.debug(f"Moved {num_moved}/{len(saved_hard)} example(s) to hard pool from checkpoint.")
            if num_moved != len(saved_hard):
                self.logger.warning(
                    f"Could not move {len(saved_hard) - num_moved} example(s) from checkpoint to hard pool. "
                    "This usually means you resumed with an env mix that does not contain all previous examples."
                )

        if any(saved_rollouts):
            valid = [r for r in saved_rollouts if r.get("env_name") in self.env_names]
            self.rollout_buffer.extend(valid)
            self.logger.debug(f"Loaded {len(valid)} rollout(s) from checkpoint.")

        def convert_to_normal(eb: _EnvBuffer, pool: list[dict], fraction: float) -> int:
            if fraction <= 0.0 or not pool:
                return 0
            num_to_move = round(len(pool) * fraction)
            if num_to_move <= 0:
                return 0
            for _ in range(num_to_move):
                example = random.choice(pool)
                pool.remove(example)
                eb.examples[example["example_id"]] = example
            return num_to_move

        for eb in self.env_buffers.values():
            n_easy = len(eb.easy_examples)
            moved = convert_to_normal(eb, eb.easy_examples, self.config.easy_fraction)
            self.logger.debug(f"Converted {moved}/{n_easy} example(s) back to normal from easy pool ({eb.env_name}).")
            n_hard = len(eb.hard_examples)
            moved = convert_to_normal(eb, eb.hard_examples, self.config.hard_fraction)
            self.logger.debug(f"Converted {moved}/{n_hard} example(s) back to normal from hard pool ({eb.env_name}).")

    def get_metrics(self) -> dict[str, float]:
        metrics = {}

        # Aggregate cross-env totals
        total_examples_per_pool = {p: 0 for p in POOLS}
        total_rollouts_per_pool = {p: 0 for p in POOLS}
        total_zero_std_rollouts = 0
        for eb in self.env_buffers.values():
            for p in POOLS:
                total_examples_per_pool[p] += eb.num_examples_per_step[p]
                total_rollouts_per_pool[p] += eb.num_rollouts_per_step[p]
            total_zero_std_rollouts += eb.num_zero_std_rollouts_per_step

        total_examples = sum(total_examples_per_pool.values())
        total_rollouts = sum(total_rollouts_per_pool.values()) + total_zero_std_rollouts

        for pool in ["easy", "hard"]:
            if total_examples:
                metrics[f"evicted_examples/{pool}"] = total_examples_per_pool[pool] / total_examples
            if total_rollouts:
                metrics[f"filtered_rollouts/{pool}"] = total_rollouts_per_pool[pool] / total_rollouts
        if total_rollouts:
            metrics["filtered_rollouts/zero_std"] = total_zero_std_rollouts / total_rollouts

        total_normal = sum(eb.num_normal for eb in self.env_buffers.values())
        total_easy = sum(len(eb.easy_examples) for eb in self.env_buffers.values())
        total_hard = sum(len(eb.hard_examples) for eb in self.env_buffers.values())
        pool_ratios = mean_normalize([total_easy, total_normal, total_hard])
        for pool, ratio in zip(POOLS, pool_ratios):
            metrics[f"pool/{pool}"] = ratio

        # Per-env metrics
        for eb in self.env_buffers.values():
            metrics.update(eb.get_metrics())

        return metrics
