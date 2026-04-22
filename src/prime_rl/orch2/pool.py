from dataclasses import dataclass
from typing import Iterator

import verifiers as vf

from prime_rl.configs.orchestrator import OrchestratorConfig, TrainEnvConfig


@dataclass
class Pool:
    id: str
    dataset: Iterator[dict]
    env: "EnvWorker"
    concurrency: int
    rollouts_per_group: int
    max_off_policy: int


class EnvWorker:
    def __init__(
        self,
        env: vf.Environment,
        client: vf.ClientConfig,
        model: str,
        sampling_args: dict,
    ):
        self.env = env
        self.client = client
        self.model = model
        self.sampling_args = sampling_args

    async def rollout(self, example: dict, policy_version: int) -> vf.RolloutOutput:
        del policy_version  # tracked on the Group, not the rollout
        return await self.env.run_rollout(
            vf.RolloutInput(**example),
            client=self.client,
            model=self.model,
            sampling_args=self.sampling_args,
            state_columns=["trajectory", "sampling_args"],
        )


def cycle_forever(ds) -> Iterator[dict]:
    i = 0
    while True:
        for row in ds:
            r = dict(row)
            r["example_id"] = i
            i += 1
            yield r


def build_pool(env_cfg: TrainEnvConfig, cfg: OrchestratorConfig, concurrency: int) -> Pool:
    env = vf.load_environment(env_cfg.stripped_id, **env_cfg.args)
    client = vf.ClientConfig(
        client_type="openai_chat_completions",
        api_base_url=cfg.client.base_url[0],
        api_key_var=cfg.client.api_key_var,
        timeout=cfg.client.timeout,
        connect_timeout=cfg.client.connect_timeout,
    )
    worker = EnvWorker(
        env=env,
        client=client,
        model=cfg.model.name,
        sampling_args=env_cfg.sampling.to_sampling_args(),
    )
    return Pool(
        id=env_cfg.resolved_name,
        dataset=cycle_forever(env.get_dataset()),
        env=worker,
        concurrency=concurrency,
        rollouts_per_group=cfg.rollouts_per_example,
        max_off_policy=cfg.max_off_policy_steps,
    )
