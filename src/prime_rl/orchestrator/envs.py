from __future__ import annotations

import asyncio
import atexit
import multiprocessing as mp
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Generic, TypeVar

import pandas as pd
import verifiers as vf
from verifiers.serve import ZMQEnvClient, ZMQEnvServer
from verifiers.utils.serve_utils import get_free_port

from prime_rl.configs.orchestrator import EnvConfig, EvalEnvConfig, TrainEnvConfig
from prime_rl.orchestrator.eval_utils import compute_pass_at_k
from prime_rl.orchestrator.vf_utils import get_completion_len
from prime_rl.utils.logger import ProgressTracker, get_logger
from prime_rl.utils.monitor import get_monitor
from prime_rl.utils.utils import capitalize

REQUIRED_STATE_COLUMNS = ["trajectory", "sampling_args"]


class Env:
    """Wraps a vf.Environment - only exposes features used in PRIME-RL."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.sampling_args: dict = {}

        get_logger().info(f"Initializing {config.resolved_name} ({config})")
        self._env: vf.Environment = vf.load_environment(config.stripped_id, **config.args)
        self._env_client: ZMQEnvClient | None = None
        self._env_server_process: BaseProcess | None = None

    @property
    def name(self) -> str:
        return self.config.resolved_name

    @property
    def env(self) -> vf.Environment:
        return self._env

    @property
    def env_client(self) -> ZMQEnvClient:
        if not self._env_client:
            raise RuntimeError(
                f"Env {self.name} has no env client connected. Call connect() first to connect to an env server."
            )
        return self._env_client

    @property
    def requires_group_scoring(self) -> bool:
        return any(self.env.rubric._is_group_func(func) for func in self.env.rubric._get_reward_funcs())

    async def start(
        self,
        log_dir: Path,
        log_level: str | None = None,
        json_logging: bool = False,
    ) -> None:
        """Spawn an env server (if needed) and connect to it."""
        if self.config.address is None:
            address = self._spawn(log_dir=log_dir, log_level=log_level, json_logging=json_logging)
        else:
            address = self.config.address
        get_logger().debug(f"Connecting {self.name} to env server {address}")
        self._env_client = ZMQEnvClient(address=address, name=self.name)
        await self.env_client.wait_for_server_startup()

    def _spawn(
        self,
        log_dir: Path,
        log_level: str | None = None,
        json_logging: bool = False,
    ) -> str:
        assert isinstance(self.config.num_workers, int), (
            f"num_workers must be resolved before spawn, got {self.config.num_workers!r}"
        )
        num_workers = self.config.num_workers
        address = f"tcp://127.0.0.1:{get_free_port()}"
        get_logger().debug(f"Spawning env server {self.name} ({address=}, {num_workers=})")
        process = mp.get_context("spawn").Process(
            target=ZMQEnvServer.run_server,
            args=(
                self.config.stripped_id,
                self.config.args,
                self.config.extra_env_kwargs,
                log_level,
                (log_dir / self.name).as_posix(),
            ),
            kwargs=dict(
                address=address,
                json_logging=json_logging,
                console_logging=False,
                num_workers=num_workers,
            ),
            daemon=False,
        )
        process.start()
        self._env_server_process = process
        return address

    def _sampling_args_with_salt(self, cache_salt: str) -> dict:
        sampling_args = {**self.sampling_args}
        extra_body = {**sampling_args.get("extra_body", {}), "cache_salt": cache_salt}
        sampling_args["extra_body"] = extra_body
        return sampling_args

    async def run_rollout(
        self,
        client: vf.ClientConfig,
        example: dict,
        model_name: str,
        cache_salt: str,
    ) -> vf.RolloutOutput:
        """Run a single rollout for an example."""
        return await self.env.run_rollout(
            vf.RolloutInput(**example),
            client=client,
            model=model_name,
            sampling_args=self._sampling_args_with_salt(cache_salt),
            max_retries=self.config.max_retries,
            state_columns=REQUIRED_STATE_COLUMNS,
            env_client=self.env_client,
        )

    async def run_group(
        self,
        client: vf.ClientConfig,
        example: dict,
        model_name: str,
        rollouts_per_example: int,
        cache_salt: str,
    ) -> list[vf.RolloutOutput]:
        """Run a group of rollouts for an example. Required for group-scoring envs."""
        return await self.env.run_group(
            [vf.RolloutInput(**example) for _ in range(rollouts_per_example)],
            client=client,
            model=model_name,
            sampling_args=self._sampling_args_with_salt(cache_salt),
            max_retries=self.config.max_retries,
            state_columns=REQUIRED_STATE_COLUMNS,
            env_client=self.env_client,
        )

    def shutdown(self) -> None:
        if self._env_server_process is None:
            return
        self._env_server_process.terminate()
        self._env_server_process = None


class TrainEnv(Env):
    config: TrainEnvConfig

    def __init__(self, config: TrainEnvConfig):
        super().__init__(config)
        self.sampling_args = config.sampling.to_sampling_args()

    def get_dataset(self, seed: int | None = None):
        return self.env.get_dataset(seed=seed)


class EvalEnv(Env):
    config: EvalEnvConfig

    def __init__(self, config: EvalEnvConfig):
        super().__init__(config)
        self.sampling_args = config.sampling.to_sampling_args()
        self.examples = self.env.get_eval_dataset(n=config.num_examples).to_list()

    async def evaluate(
        self,
        model_name: str,
        get_client: Callable[[], Awaitable[vf.ClientConfig]],
        ckpt_step: int,
        step: int,
        cache_salt: str,
    ) -> list[vf.RolloutOutput]:
        num_examples = len(self.examples)
        rollouts_per_example = self.config.rollouts_per_example
        get_logger().info(f"Evaluating {self.name} ({num_examples=}, {rollouts_per_example=})")
        total_rollouts = num_examples * rollouts_per_example
        pbar = ProgressTracker(total=total_rollouts, desc=f"Evaluating {self.name}")
        eval_start = time.perf_counter()

        if self.requires_group_scoring:

            async def run_with_progress(example: dict) -> list[vf.RolloutOutput] | None:
                """Run rollouts_per_example rollouts as a scored group for one example."""
                try:
                    client = await get_client()
                    outputs = await self.run_group(
                        client=client,
                        example=example,
                        model_name=model_name,
                        rollouts_per_example=rollouts_per_example,
                        cache_salt=cache_salt,
                    )
                    pbar.update(rollouts_per_example)
                    return outputs
                except Exception as e:
                    get_logger().warning(f"Group failed: {e}")
                    pbar.update(rollouts_per_example)
                    return None

            coros = [run_with_progress(example) for example in self.examples]

        else:

            async def run_with_progress(example: dict) -> list[vf.RolloutOutput] | None:
                """Run a single rollout for one example."""
                try:
                    client = await get_client()
                    output = await self.run_rollout(
                        client=client, example=example, model_name=model_name, cache_salt=cache_salt
                    )
                    pbar.update(1)
                    return [output]
                except Exception as e:
                    get_logger().warning(f"Rollout failed: {e}")
                    pbar.update(1)
                    return None

            coros = [run_with_progress(example) for example in self.examples for _ in range(rollouts_per_example)]

        try:
            results = await asyncio.gather(*coros)
        finally:
            pbar.close()

        successful_outputs = [o for group in results if group is not None for o in group]
        failed_count = total_rollouts - len(successful_outputs)
        eval_time = time.perf_counter() - eval_start

        if failed_count:
            get_logger().warning(
                f"{failed_count}/{total_rollouts} ({failed_count / total_rollouts * 100:.1f}%) rollouts failed"
            )

        if not successful_outputs:
            get_logger().warning(f"All rollouts failed for {self.name}, skipping logging metrics")
            get_monitor().log(
                {
                    f"eval/{self.name}/failed_rollouts": failed_count / total_rollouts,
                    "progress/ckpt_step": ckpt_step,
                    "step": step,
                },
                step=step,
            )
            return []

        # Log metrics
        monitor = get_monitor()

        rows = [
            {
                "example_id": o["example_id"],
                "reward": o["reward"],
                "completion_len": get_completion_len(o),
                "is_truncated": o["is_truncated"],
                "has_error": o.get("error") is not None,
                "no_response": not o.get("completion"),
            }
            for o in successful_outputs
        ]
        results_df = pd.DataFrame(rows)

        unique_rewards = results_df.reward.dropna().unique()
        could_be_binary = set(unique_rewards).issubset({0.0, 1.0})
        if could_be_binary:
            pass_at_k = (
                results_df.groupby("example_id")
                .apply(lambda x: compute_pass_at_k(x.reward.dropna()), include_groups=False)
                .apply(pd.Series)
            )
        else:
            pass_at_k = None
            get_logger().warning("Skipping computing pass@k rates because the task rewards appear to be non-binary")

        message = (
            f"Evaluated {self.name} in {eval_time:.2f}s (Avg@{rollouts_per_example}={results_df.reward.mean():.4f}"
        )
        if could_be_binary:
            assert pass_at_k is not None
            for pass_rate, pass_rate_score in pd.Series(pass_at_k.mean()).items():
                message += f", {capitalize(str(pass_rate))}: {pass_rate_score:.4f}"

        message += (
            f", No-response: {results_df.no_response.mean() * 100:.1f}%"
            f", Completion Length: {results_df.completion_len.mean():.2f} (±{results_df.completion_len.std():.2f}, ∈[{results_df.completion_len.min():.2f}, {results_df.completion_len.max():.2f}])"
            f", Truncated: {results_df.is_truncated.mean() * 100:.1f}%)"
        )
        get_logger().success(message)

        eval_metrics = {
            f"avg@{rollouts_per_example}": float(results_df.reward.mean()),
            "no_response/mean": float(results_df.no_response.mean()),
            "no_response/count": int(results_df.no_response.sum()),
            "completion_len/mean": results_df.completion_len.mean().item(),
            "completion_len/max": results_df.completion_len.max().item(),
            "completion_len/min": results_df.completion_len.min().item(),
            "is_truncated/mean": results_df.is_truncated.mean().item(),
            "failed_rollouts": failed_count / total_rollouts,
            "time": eval_time,
        }
        if could_be_binary:
            assert pass_at_k is not None
            eval_metrics.update(pd.Series(pass_at_k.mean()).to_dict())
        eval_metrics = {f"eval/{self.name}/{key}": v for key, v in eval_metrics.items()}
        eval_metrics.update({"progress/ckpt_step": ckpt_step, "step": step})
        monitor.log(eval_metrics, step=step)
        monitor.log_eval_samples(successful_outputs, env_name=self.name, step=step)

        return successful_outputs


EnvT = TypeVar("EnvT", bound=Env)


class Envs(Generic[EnvT]):
    """Base container for a set of Env instances."""

    _envs: dict[str, EnvT]

    @property
    def names(self) -> list[str]:
        return list(self._envs.keys())

    @property
    def configs(self) -> list[EnvConfig]:
        return [env.config for env in self._envs.values()]

    def get(self, name: str) -> EnvT:
        return self._envs[name]

    def __iter__(self) -> Iterator[EnvT]:
        return iter(self._envs.values())

    def __len__(self) -> int:
        return len(self._envs)

    async def start(
        self,
        log_dir: Path,
        log_level: str | None = None,
        json_logging: bool = False,
    ) -> None:
        """Spawn env servers (where needed) and connect env clients one at a time.

        Serialized to avoid a TOCTOU port race: get_free_port() only holds the port
        until it returns, so parallel spawns can hand the same port to two children.
        """
        for env in self:
            await env.start(log_dir=log_dir, log_level=log_level, json_logging=json_logging)
        atexit.register(self.shutdown)

    def shutdown(self) -> None:
        """Terminate all spawned env server processes in parallel."""
        processes = [env._env_server_process for env in self if env._env_server_process is not None]
        if not processes:
            return
        logger = get_logger()
        logger.info(f"Shutting down {len(processes)} env server(s), waiting for sandbox cleanup...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=25)
            if p.is_alive():
                logger.warning(f"Env server {p.pid} did not exit after 25s, force killing")
                p.kill()
                p.join(timeout=5)
        for env in self:
            env._env_server_process = None


class TrainEnvs(Envs[TrainEnv]):
    """Collection of training environments."""

    def __init__(self, configs: Sequence[TrainEnvConfig]):
        self._envs: dict[str, TrainEnv] = {}
        for config in configs:
            env = TrainEnv(config)
            self._envs[env.name] = env


class EvalEnvs(Envs[EvalEnv]):
    """Collection of evaluation environments."""

    def __init__(self, configs: Sequence[EvalEnvConfig]):
        self._envs: dict[str, EvalEnv] = {}
        for config in configs:
            env = EvalEnv(config)
            self._envs[env.name] = env
