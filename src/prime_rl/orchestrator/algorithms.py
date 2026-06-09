"""Orchestrator-side algorithm strategies.

The config side (``prime_rl.configs.algorithm``) defines *what* an algorithm
is — a preset of sampling, scoring (advantage + token scorer), and loss
routing. This module implements the runtime pieces the orchestrator executes:

- **Token scorers** — async per-sample scoring that attaches per-token data by
  querying the teacher pool (bounded concurrency). Runs at batch-ship time via
  :func:`score_train_batch`.
- **Loss routing stamping** — :func:`stamp_loss_routing` translates an env's
  loss routing config into the per-token wire fields the trainer executes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from itertools import cycle
from typing import TYPE_CHECKING, Protocol

import verifiers as vf

from prime_rl.configs.algorithm import (
    DemoTeacherLogprobsConfig,
    LossRoutingConfig,
    TeacherLogprobsConfig,
    TokenScorerConfig,
)
from prime_rl.orchestrator.utils import compute_prefill_logprobs
from prime_rl.transport import TrainingSample
from prime_rl.transport.types import LOSS_CORE_CE, LOSS_CORE_RL, LOSS_CORE_TEACHER_KL

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

    from prime_rl.orchestrator.types import TrainRollout

ACTION_LOSS_CORES = {"rl": LOSS_CORE_RL, "ce": LOSS_CORE_CE, "teacher_kl": LOSS_CORE_TEACHER_KL}


def stamp_loss_routing(sample: TrainingSample, loss: LossRoutingConfig) -> None:
    """Stamp the env's loss routing onto one sample's wire fields.

    Action tokens (the trainable completion tokens) get the configured action
    core. When the algorithm trains on observations, env-provided tokens
    (tagged by ``interleave_rollout`` in ``completion_obs_mask``) flip from
    masked-out to trainable on the CE core with ``observation_weight``.
    ``completion_obs_mask`` is orchestrator-internal and cleared here so it
    never ships.
    """
    action_core = ACTION_LOSS_CORES[loss.action]
    sample.loss_core = action_core
    obs_mask = sample.completion_obs_mask
    sample.completion_obs_mask = None
    if loss.observation == "none" or obs_mask is None or not any(obs_mask):
        return

    prompt_len = len(sample.prompt_ids)
    seq_len = prompt_len + len(sample.completion_ids)
    cores = [action_core] * seq_len
    weights = [1.0] * seq_len
    completion_mask = list(sample.completion_mask)
    for i, is_obs in enumerate(obs_mask):
        if is_obs:
            cores[prompt_len + i] = LOSS_CORE_CE
            weights[prompt_len + i] = loss.observation_weight
            completion_mask[i] = True
    sample.completion_mask = completion_mask
    sample.token_loss_cores = cores
    sample.token_loss_weights = weights


@dataclass
class ScoringContext:
    """Resources a token scorer may use: the teacher pool's train clients
    (cycled for load balancing) and the student tokenizer."""

    teacher_clients: list[vf.ClientConfig]
    teacher_model_name: str
    tokenizer: PreTrainedTokenizer


class TokenScorer(Protocol):
    async def score(self, rollouts: list[TrainRollout], ctx: ScoringContext) -> None: ...


class TeacherLogprobsScorer:
    """Fill ``TrainingSample.teacher_logprobs`` by scoring each sample's own
    context under the teacher (on-policy distillation)."""

    def __init__(self, config: TeacherLogprobsConfig):
        self.config = config

    async def score(self, rollouts: list[TrainRollout], ctx: ScoringContext) -> None:
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        samples = [sample for rollout in rollouts for sample in rollout.samples]

        async def score_sample(client: vf.ClientConfig, sample: TrainingSample) -> None:
            async with semaphore:
                token_ids = list(sample.prompt_ids) + list(sample.completion_ids)
                sample.teacher_logprobs = await compute_prefill_logprobs(client, ctx.teacher_model_name, token_ids)

        await asyncio.gather(
            *[score_sample(client, sample) for client, sample in zip(cycle(ctx.teacher_clients), samples)]
        )


class DemoTeacherLogprobsScorer:
    """Fill ``TrainingSample.teacher_logprobs`` by scoring each sample's
    completion under the teacher conditioned on an expert demonstration (SDFT).

    The teacher prefix is rebuilt from the rollout's first-turn prompt
    messages with the demonstration woven into the last user message; the
    returned completion logprobs are aligned back onto the sample (student
    prompt positions are 0.0 and stay outside the loss mask).
    """

    def __init__(self, config: DemoTeacherLogprobsConfig):
        self.config = config

    def _teacher_prefix_ids(self, rollout: TrainRollout, ctx: ScoringContext) -> list[int]:
        trajectory = rollout.raw.get("trajectory") or []
        if len(trajectory) != 1:
            raise ValueError(
                f"demo_teacher_logprobs supports single-step trajectories only; "
                f"env '{rollout.env_name}' produced {len(trajectory)} steps."
            )
        info = rollout.raw.get("info") or {}
        demonstration = info.get(self.config.demo_key) if isinstance(info, dict) else None
        if demonstration is None:
            raise ValueError(
                f"demo_teacher_logprobs requires '{self.config.demo_key}' in the example's info dict "
                f"(env '{rollout.env_name}', example {rollout.example_id})."
            )

        messages = [dict(m) for m in trajectory[0]["prompt"]]
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not user_indices:
            raise ValueError(f"demo_teacher_logprobs found no user message to condition (env '{rollout.env_name}').")
        last_user = messages[user_indices[-1]]
        question = last_user.get("content")
        if not isinstance(question, str):
            raise ValueError("demo_teacher_logprobs supports text-only prompts (user content must be a string).")
        last_user["content"] = self.config.template.format(question=question, demonstration=demonstration)

        return ctx.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    async def score(self, rollouts: list[TrainRollout], ctx: ScoringContext) -> None:
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def score_rollout(client: vf.ClientConfig, rollout: TrainRollout) -> None:
            prefix_ids = self._teacher_prefix_ids(rollout, ctx)
            assert len(rollout.samples) == 1  # single-step trajectory → one sample
            sample = rollout.samples[0]
            async with semaphore:
                full_logprobs = await compute_prefill_logprobs(
                    client, ctx.teacher_model_name, prefix_ids + list(sample.completion_ids)
                )
            completion_logprobs = full_logprobs[-len(sample.completion_ids) :]
            sample.teacher_logprobs = [0.0] * len(sample.prompt_ids) + completion_logprobs

        await asyncio.gather(
            *[score_rollout(client, rollout) for client, rollout in zip(cycle(ctx.teacher_clients), rollouts)]
        )


def setup_token_scorer(config: TokenScorerConfig | None) -> TokenScorer | None:
    if config is None:
        return None
    if isinstance(config, TeacherLogprobsConfig):
        return TeacherLogprobsScorer(config)
    return DemoTeacherLogprobsScorer(config)


async def score_train_batch(
    rollouts: list[TrainRollout],
    scorers: dict[str, TokenScorer],
    ctx: ScoringContext,
) -> None:
    """Run each env's token scorer over its unfiltered rollouts, concurrently
    across envs. Per-env concurrency is bounded by the scorer's own config."""
    tasks = []
    for env_name, scorer in scorers.items():
        env_rollouts = [r for r in rollouts if r.env_name == env_name and not r.is_filtered]
        if env_rollouts:
            tasks.append(scorer.score(env_rollouts, ctx))
    if tasks:
        await asyncio.gather(*tasks)
