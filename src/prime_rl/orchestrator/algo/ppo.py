"""PPO credit assignment using a separate token-value service."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import httpx
import verifiers.v1 as vf

from prime_rl.configs.algorithm import PPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.algo.gae import skip_observation_gae
from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.transports.batch import TrainingSample

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


class PPOAlgorithm(Algorithm):
    def __init__(self, config: PPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.config = config
        self.value_client = httpx.AsyncClient(base_url=config.value_url, timeout=3600)

    async def setup(self) -> None:
        deadline = time.monotonic() + 1800
        while True:
            try:
                response = await self.value_client.get("/health")
                response.raise_for_status()
                return
            except httpx.ConnectError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Value service did not start at {self.config.value_url}")
                await asyncio.sleep(1)

    async def aclose(self) -> None:
        await self.value_client.aclose()

    async def score_samples_by_sample(self, trace: vf.Trace, samples: list[TrainingSample]) -> None:
        for _, branch, sample, scored_length in self._scored_branches([(trace, samples)]):
            response = await self.value_client.post("/score", json={"token_ids": sample.token_ids[:scored_length]})
            response.raise_for_status()
            score = response.json()
            self._apply_score(trace, branch, sample, score["values"], score["bootstrap_value"], scored_length)

    async def score_samples(self, trace_samples: list[tuple[vf.Trace, list[TrainingSample]]]) -> None:
        scored = self._scored_branches(trace_samples)
        if not scored:
            return
        response = await self.value_client.post(
            "/score",
            json={"batched_token_ids": [sample.token_ids[:length] for _, _, sample, length in scored]},
        )
        response.raise_for_status()
        result = response.json()
        values_batch = result["values"]
        bootstraps = result["bootstrap_values"]
        if len(values_batch) != len(scored) or len(bootstraps) != len(scored):
            raise ValueError("Value service returned the wrong number of samples")
        for (trace, branch, sample, length), values, bootstrap in zip(scored, values_batch, bootstraps, strict=True):
            self._apply_score(trace, branch, sample, values, bootstrap, length)

    def _scored_branches(
        self, trace_samples: list[tuple[vf.Trace, list[TrainingSample]]]
    ) -> list[tuple[vf.Trace, vf.Branch, TrainingSample, int]]:
        scored = []
        for trace, samples in trace_samples:
            branches = list(iter_trainable_branches(trace))
            if len(branches) != len(samples):
                raise ValueError("PPO branch/sample count mismatch")
            for (branch, _), sample in zip(branches, samples, strict=True):
                scored_length = min(len(sample.token_ids), self.config.value_seq_len or len(sample.token_ids))
                scored.append((trace, branch, sample, scored_length))
        return scored

    def _apply_score(
        self,
        trace: vf.Trace,
        branch: vf.Branch,
        sample: TrainingSample,
        values: list[float],
        bootstrap_value: float,
        scored_length: int,
    ) -> None:
        if len(values) != scored_length:
            raise ValueError("Value service returned a misaligned token stream")
        scored_mask = sample.mask[:scored_length]
        n_actions = sum(scored_mask)
        policy_lambda = self.config.policy_lambda
        if self.config.length_adaptive_alpha is not None and n_actions:
            policy_lambda = max(0.0, 1 - 1 / (self.config.length_adaptive_alpha * n_actions))
        advantages, targets = skip_observation_gae(
            values,
            scored_mask,
            trace.reward if scored_length == len(sample.token_ids) else 0.0,
            gamma=self.config.gamma,
            policy_lambda=policy_lambda,
            value_lambda=self.config.value_lambda,
            bootstrap_value=bootstrap_value if scored_length < len(sample.token_ids) else 0.0,
        )
        unscored = len(sample.token_ids) - scored_length
        sample.old_values = values + [0.0] * unscored
        sample.advantages = advantages + [0.0] * unscored
        sample.value_targets = targets + [0.0] * unscored
        sample.value_mask = list(scored_mask) + [False] * unscored

        offset = 0
        for node in branch.nodes:
            end = offset + len(node.token_ids)
            node_advantages = [
                advantage
                for advantage, sampled in zip(sample.advantages[offset:end], sample.mask[offset:end], strict=True)
                if sampled
            ]
            if node_advantages:
                node.advantages = node_advantages
            offset = end
        sample.mask = sample.value_mask.copy()
