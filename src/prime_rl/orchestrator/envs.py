from __future__ import annotations

import asyncio
import atexit
import multiprocessing as mp
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Sequence
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Generic, TypeVar

import pandas as pd
import verifiers as vf
from verifiers.serve import ZMQEnvClient, ZMQEnvServer
from verifiers.utils.serve_utils import get_free_port

from prime_rl.configs.multi_agent import MultiAgentConfig
from prime_rl.configs.orchestrator import EMAPerMemberAdvantageConfig, EnvConfig, EvalEnvConfig, TrainEnvConfig
from prime_rl.orchestrator.advantage import AdvantageFn, setup_advantage_fn
from prime_rl.orchestrator.eval_utils import compute_pass_at_k
from prime_rl.orchestrator.member_generation import (
    DISPATCH_ID_FIELD,
    compile_member_generation_plan,
)
from prime_rl.utils.logger import ProgressTracker, get_logger
from prime_rl.utils.monitor import get_monitor
from prime_rl.utils.utils import capitalize

REQUIRED_STATE_COLUMNS = ["trajectory"]

# Request fields the orchestrator injects into learner sampling args for its own
# training loop. Fixed (non-learner) members are never trained, so these fields
# are at best wasted and at worst rejected (400) by external API endpoints:
# - "logprobs": inference logprobs feed the trainer's importance ratios.
# - "return_token_ids": completion token ids feed trainer token alignment.
# - "top_k" / "min_p": full-distribution sampling sentinels (-1 / 0.0) injected
#   by resolve_env_config so learner logprobs match the trainer's softmax;
#   vLLM-only params. Per-target values belong in the fixed target's sampling.
# - "cache_salt": per-dispatch KV-cache isolation for the learner.
LEARNER_ONLY_SAMPLING_FIELDS = frozenset({"logprobs"})
LEARNER_ONLY_EXTRA_BODY_FIELDS = frozenset({"cache_salt", "return_token_ids", "top_k", "min_p"})


class Env:
    """Wraps a vf.Environment - only exposes features used in PRIME-RL."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.sampling_args: dict = {}

        get_logger().debug(f"Initializing {config.resolved_name} ({config})")
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

    @property
    def is_multi_agent(self) -> bool:
        return bool(getattr(self.env, "is_multi_agent", False))

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

    def _sampling_args_with_salt(self, cache_salt: str | None) -> dict:
        sampling_args = {**self.sampling_args}
        if cache_salt is None:
            return sampling_args
        extra_body = {**sampling_args.get("extra_body", {}), "cache_salt": cache_salt}
        sampling_args["extra_body"] = extra_body
        return sampling_args

    def _fixed_member_sampling_args(self) -> dict:
        """Sampling args for fixed (non-learner) member targets: inherit the
        learner's portable sampling defaults but strip learner-only request
        fields (see LEARNER_ONLY_*_FIELDS), and copy the nested ``extra_body``
        so downstream mutation cannot alias the learner's extra_body. A fixed
        target's own explicit ``sampling`` config still overrides."""
        sampling_args = {k: v for k, v in self.sampling_args.items() if k not in LEARNER_ONLY_SAMPLING_FIELDS}
        extra_body = sampling_args.pop("extra_body", None)
        if extra_body is not None:
            extra_body = {k: v for k, v in extra_body.items() if k not in LEARNER_ONLY_EXTRA_BODY_FIELDS}
            if extra_body:
                sampling_args["extra_body"] = extra_body
        return sampling_args

    def multi_agent_members(self) -> list[str]:
        members = getattr(self.env, "members", None)
        if members is None:
            members = getattr(self.env.rubric, "members", None)
        if not members:
            raise ValueError(f"Multi-agent config is enabled for env {self.name}, but no member ids were found.")
        return list(members)

    def compile_generation(
        self,
        config: MultiAgentConfig,
        *,
        client: vf.ClientConfig,
        model_name: str,
        cache_salt: str | None,
        dispatch_id: str,
        group_id: str,
    ) -> vf.MemberGenerationPlan | None:
        if not config.enabled:
            return None
        return compile_member_generation_plan(
            config,
            member_ids=self.multi_agent_members(),
            default_client=client,
            default_model=model_name,
            learner_sampling_args=self._sampling_args_with_salt(cache_salt),
            fixed_sampling_args=self._fixed_member_sampling_args(),
            dispatch_id=dispatch_id,
            group_id=group_id,
        )

    @property
    def state_columns(self) -> list[str]:
        """Required columns plus any extras configured on the env, deduped (required first)."""
        merged: list[str] = []
        for col in (*REQUIRED_STATE_COLUMNS, *self.config.state_columns):
            if col not in merged:
                merged.append(col)
        return merged

    async def run_rollout(
        self,
        client: vf.ClientConfig,
        example: dict,
        model_name: str,
        cache_salt: str | None,
        generation: vf.MemberGenerationPlan | None = None,
        dispatch_id: str | None = None,
    ) -> vf.RolloutOutput:
        """Run a single rollout for an example."""
        output = await self.env.run_rollout(
            vf.RolloutInput(**example),
            client=client,
            model=model_name,
            sampling_args=self._sampling_args_with_salt(cache_salt),
            max_retries=self.config.max_retries,
            state_columns=self.state_columns,
            env_client=self.env_client,
            generation=generation,
        )
        if dispatch_id is not None:
            output[DISPATCH_ID_FIELD] = dispatch_id
        return output

    async def run_group(
        self,
        client: vf.ClientConfig,
        example: dict,
        model_name: str,
        group_size: int,
        cache_salt: str | None,
        generation: vf.GenerationPlan | None = None,
        dispatch_ids: list[str] | None = None,
    ) -> list[vf.RolloutOutput]:
        """Run a group of rollouts for an example. Required for group-scoring envs."""
        outputs = await self.env.run_group(
            [vf.RolloutInput(**example) for _ in range(group_size)],
            client=client,
            model=model_name,
            sampling_args=self._sampling_args_with_salt(cache_salt),
            max_retries=self.config.max_retries,
            state_columns=self.state_columns,
            env_client=self.env_client,
            generation=generation,
        )
        if dispatch_ids is not None:
            if len(dispatch_ids) != len(outputs):
                raise ValueError(f"dispatch_ids length {len(dispatch_ids)} does not match output length {len(outputs)}")
            for output, dispatch_id in zip(outputs, dispatch_ids):
                output[DISPATCH_ID_FIELD] = dispatch_id
        return outputs

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
        # Built once — custom advantage funcs do an ``import_object`` we don't
        # want to pay per group. ``None`` = reward-only path. ``ema_per_member``
        # has no group-level fn: the train sink computes RAE advantages instead.
        self.advantage_fn: AdvantageFn | None = (
            setup_advantage_fn(config.advantage)
            if config.advantage is not None and not isinstance(config.advantage, EMAPerMemberAdvantageConfig)
            else None
        )

    def get_dataset(self, seed: int | None = None):
        return self.env.get_dataset(seed=seed)


class EvalEnv(Env):
    config: EvalEnvConfig

    def __init__(self, config: EvalEnvConfig):
        super().__init__(config)
        self.sampling_args = config.sampling.to_sampling_args()
        self.examples = self.env.get_eval_dataset(n=config.num_examples, seed=config.seed).to_list()

    async def evaluate(
        self,
        model_name: str,
        get_client: Callable[[], Awaitable[vf.ClientConfig]],
        ckpt_step: int,
        step: int,
        cache_salt: str,
        multi_agent: MultiAgentConfig | None = None,
        eval_clients: Sequence[vf.ClientConfig] | None = None,
    ) -> list[vf.RolloutOutput]:
        num_examples = len(self.examples)
        group_size = self.config.group_size
        get_logger().info(f"Evaluating {self.name} ({num_examples=}, {group_size=})")
        total_rollouts = num_examples * group_size
        pbar = ProgressTracker(total=total_rollouts, desc=f"Evaluating {self.name}")
        eval_start = time.perf_counter()
        active_multi_agent = (
            multi_agent if multi_agent is not None and multi_agent.enabled and self.is_multi_agent else None
        )
        max_concurrent_rollouts_per_client = self.config.max_concurrent_rollouts_per_client
        eval_client_pool = list(eval_clients or [])
        client_load: Counter[tuple[str, str | None]] = Counter()

        def client_identity(client: vf.ClientConfig) -> tuple[str, str | None]:
            return (client.api_base_url, client.extra_headers.get("X-data-parallel-rank"))

        def acquire_eval_client(cost: int) -> vf.ClientConfig:
            client = min(eval_client_pool, key=lambda c: client_load[client_identity(c)])
            client_load[client_identity(client)] += cost
            return client

        async def acquire_client(cost: int) -> vf.ClientConfig:
            if max_concurrent_rollouts_per_client is None:
                return await get_client()
            return acquire_eval_client(cost)

        def release_client(client: vf.ClientConfig, cost: int) -> None:
            if max_concurrent_rollouts_per_client is None:
                return
            identity = client_identity(client)
            client_load[identity] -= cost
            if client_load[identity] <= 0:
                del client_load[identity]

        if self.requires_group_scoring:

            async def run_with_progress(example: dict) -> list[vf.RolloutOutput] | None:
                """Run group_size rollouts as a scored group for one example."""
                client = await acquire_client(group_size)
                try:
                    group_key = f"eval:{step}:{self.name}:{example['example_id']}"
                    dispatch_ids = [f"{group_key}:{idx}" for idx in range(group_size)]
                    generation = (
                        [
                            self.compile_generation(
                                active_multi_agent,
                                client=client,
                                model_name=model_name,
                                cache_salt=cache_salt,
                                dispatch_id=dispatch_id,
                                group_id=group_key,
                            )
                            for dispatch_id in dispatch_ids
                        ]
                        if active_multi_agent is not None
                        else None
                    )
                    outputs = await self.run_group(
                        client=client,
                        example=example,
                        model_name=model_name,
                        group_size=group_size,
                        cache_salt=cache_salt,
                        generation=generation,
                        dispatch_ids=dispatch_ids if active_multi_agent is not None else None,
                    )
                    pbar.update(group_size)
                    return outputs
                except Exception as e:
                    get_logger().warning(f"Group failed: {e}")
                    pbar.update(group_size)
                    return None
                finally:
                    release_client(client, group_size)

            jobs = self.examples
            cost_per_job = group_size

        else:

            async def run_with_progress(job: tuple[dict, int]) -> list[vf.RolloutOutput] | None:
                """Run a single rollout for one example."""
                example, rollout_idx = job
                client = await acquire_client(1)
                try:
                    group_key = f"eval:{step}:{self.name}:{example['example_id']}"
                    dispatch_id = f"{group_key}:{rollout_idx}"
                    generation = (
                        self.compile_generation(
                            active_multi_agent,
                            client=client,
                            model_name=model_name,
                            cache_salt=cache_salt,
                            dispatch_id=dispatch_id,
                            group_id=group_key,
                        )
                        if active_multi_agent is not None
                        else None
                    )
                    output = await self.run_rollout(
                        client=client,
                        example=example,
                        model_name=model_name,
                        cache_salt=cache_salt,
                        generation=generation,
                        dispatch_id=dispatch_id if active_multi_agent is not None else None,
                    )
                    pbar.update(1)
                    return [output]
                except Exception as e:
                    get_logger().warning(f"Rollout failed: {e}")
                    pbar.update(1)
                    return None
                finally:
                    release_client(client, 1)

            jobs = [(example, rollout_idx) for example in self.examples for rollout_idx in range(group_size)]
            cost_per_job = 1

        try:
            if max_concurrent_rollouts_per_client is None:
                results = await asyncio.gather(*(run_with_progress(job) for job in jobs))
            else:
                if not eval_client_pool:
                    raise RuntimeError("Eval dynamic refill requires at least one eval client")

                eval_jobs: asyncio.Queue[dict | tuple[dict, int] | None] = asyncio.Queue()
                worker_count = min(
                    len(jobs),
                    max(1, (len(eval_client_pool) * max_concurrent_rollouts_per_client) // cost_per_job),
                )
                for job in jobs:
                    eval_jobs.put_nowait(job)
                for _ in range(worker_count):
                    eval_jobs.put_nowait(None)

                async def worker() -> list[list[vf.RolloutOutput] | None]:
                    worker_results: list[list[vf.RolloutOutput] | None] = []
                    while True:
                        job = await eval_jobs.get()
                        try:
                            if job is None:
                                return worker_results
                            worker_results.append(await run_with_progress(job))
                        finally:
                            eval_jobs.task_done()

                get_logger().info(
                    f"Eval dynamic refill enabled for {self.name}: "
                    f"{worker_count} workers across {len(eval_client_pool)} inference clients"
                )
                worker_results = await asyncio.gather(*(worker() for _ in range(worker_count)))
                results = [result for worker_result in worker_results for result in worker_result]
        finally:
            pbar.close()

        successful_outputs = [o for group in results if group is not None for o in group]
        for output in successful_outputs:
            output["env_name"] = self.name
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
                "completion_len": o["token_usage"]["final_output_tokens"],
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

        message = f"Evaluated {self.name} in {eval_time:.2f}s (Avg@{group_size}={results_df.reward.mean():.4f}"
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
            f"avg@{group_size}": float(results_df.reward.mean()),
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
        eval_metrics["progress/ckpt_step"] = ckpt_step
        eval_metrics["step"] = step
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
    def multi_agent_names(self) -> set[str]:
        return {env.name for env in self if env.is_multi_agent}

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
        logger.debug(f"Shutting down {len(processes)} env server(s)")
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
