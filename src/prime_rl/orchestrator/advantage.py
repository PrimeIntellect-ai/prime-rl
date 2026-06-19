from __future__ import annotations

import inspect
import os
from collections.abc import Awaitable, Callable
from typing import Literal, cast

import httpx
import verifiers.v1 as vf
from openai import AsyncOpenAI
from renderers import create_renderer_pool
from vllm.entrypoints.serve.disagg.protocol import GenerateResponse

from prime_rl.utils.utils import import_object

AdvantageLoss = Literal["rl", "ce"]
AdvantageScope = Literal["rollout", "group"]
TraceAdvantageFn = Callable[[list[vf.Trace]], list[vf.Trace] | Awaitable[list[vf.Trace]]]


@vf.advantage(loss="rl", scope="group")
async def grpo(traces: list[vf.Trace]) -> list[vf.Trace]:
    rewards = [trace.reward for trace in traces]
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    for trace, reward in zip(traces, rewards, strict=True):
        advantage = reward - mean
        for branch in trace.branches:
            branch.advantages = [advantage if sampled else 0.0 for sampled in branch.sampled_mask]
            branch.mask = list(branch.sampled_mask)
    return traces


@vf.advantage(loss="rl", scope="group")
async def max_rl(traces: list[vf.Trace]) -> list[vf.Trace]:
    rewards = [trace.reward for trace in traces]
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    for trace, reward in zip(traces, rewards, strict=True):
        advantage = 0.0 if mean <= 0 else (reward - mean) / mean
        for branch in trace.branches:
            branch.advantages = [advantage if sampled else 0.0 for sampled in branch.sampled_mask]
            branch.mask = list(branch.sampled_mask)
    return traces


@vf.advantage(loss="rl", scope="rollout")
async def reward(traces: list[vf.Trace]) -> list[vf.Trace]:
    for trace in traces:
        for branch in trace.branches:
            branch.advantages = [trace.reward if sampled else 0.0 for sampled in branch.sampled_mask]
            branch.mask = list(branch.sampled_mask)
    return traces


@vf.advantage(loss="ce", scope="rollout")
async def sft(traces: list[vf.Trace]) -> list[vf.Trace]:
    for trace in traces:
        for branch in trace.branches:
            branch.advantages = [1.0 if sampled else 0.0 for sampled in branch.sampled_mask]
            branch.mask = list(branch.sampled_mask)
    return traces


@vf.advantage(loss="ce", scope="rollout")
async def echo(traces: list[vf.Trace]) -> list[vf.Trace]:
    selected_roles = {"system", "user", "tool"}
    for trace in traces:
        for branch in trace.branches:
            advantages: list[float] = []
            mask: list[bool] = []
            for node in branch.nodes:
                role = getattr(node.message, "role", "")
                selected = role in selected_roles and not node.sampled
                advantages.extend([1.0 if selected else 0.0] * len(node.token_ids))
                mask.extend([selected] * len(node.token_ids))
            branch.advantages = advantages
            branch.mask = mask
    return traces


@vf.advantage(loss="rl", scope="group")
async def opd(traces: list[vf.Trace]) -> list[vf.Trace]:
    for trace in traces:
        model_config = trace.models.get("teacher")
        if model_config is None:
            raise ValueError("opd requires trace.models['teacher'].")
        if not isinstance(model_config, vf.TrainClientConfig):
            raise ValueError("opd requires trace.models['teacher'] to be a train client.")
        if model_config.model is None:
            raise ValueError("opd requires trace.models['teacher'].model.")
        client = AsyncOpenAI(
            base_url=model_config.base_url,
            api_key=os.environ.get(model_config.api_key_var, "EMPTY"),
            default_headers=model_config.headers or None,
        )
        base = str(client.base_url).rstrip("/").removesuffix("/v1")
        for branch in trace.branches:
            response = await client.post(
                f"{base}/inference/v1/generate",
                cast_to=httpx.Response,
                body={
                    "model": model_config.model,
                    "token_ids": list(branch.token_ids),
                    "sampling_params": {
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "prompt_logprobs": 1,
                    },
                },
            )
            parsed = GenerateResponse.model_validate_json(response.content)
            model_logprobs: list[float] = []
            for entry in parsed.prompt_logprobs or []:
                if not entry:
                    model_logprobs.append(0.0)
                    continue
                first = next(iter(entry.values()))
                lp = first.logprob if hasattr(first, "logprob") else first.get("logprob")
                model_logprobs.append(float(lp) if lp is not None else 0.0)
            if len(model_logprobs) != len(branch.token_ids):
                raise ValueError(
                    f"opd prefill returned {len(model_logprobs)} logprobs for {len(branch.token_ids)} tokens."
                )
            branch.advantages = [
                model_lp - actor_lp if sampled else 0.0
                for model_lp, actor_lp, sampled in zip(
                    model_logprobs, branch.logprobs, branch.sampled_mask, strict=True
                )
            ]
            branch.mask = list(branch.sampled_mask)
    return traces


@vf.advantage(loss="rl", scope="group")
async def opsd(traces: list[vf.Trace]) -> list[vf.Trace]:
    template = (
        "{question}\n\n"
        "Here is an example of an expert response:\n"
        "<demonstration>\n{demonstration}\n</demonstration>\n\n"
        "Answer with a response of your own."
    )
    for trace in traces:
        model_config = trace.models.get("policy")
        if model_config is None:
            raise ValueError("opsd requires trace.models['policy'].")
        if not isinstance(model_config, vf.TrainClientConfig):
            raise ValueError("opsd requires trace.models['policy'] to be a train client.")
        if model_config.model is None:
            raise ValueError("opsd requires trace.models['policy'].model.")
        client = AsyncOpenAI(
            base_url=model_config.base_url,
            api_key=os.environ.get(model_config.api_key_var, "EMPTY"),
            default_headers=model_config.headers or None,
        )
        base = str(client.base_url).rstrip("/").removesuffix("/v1")
        demonstration = trace.info.get("demonstration", trace.info.get("answer"))
        if demonstration is None:
            demonstration = getattr(trace.task, "demonstration", None)
        if demonstration is None:
            demonstration = getattr(trace.task, "answer", None)
        if not isinstance(demonstration, str):
            raise ValueError(
                "opsd requires a string demonstration on trace.info['demonstration'], "
                "trace.info['answer'], trace.task.demonstration, or trace.task.answer."
            )
        renderer = create_renderer_pool(
            model_config.renderer_model_name or model_config.model,
            model_config.renderer,
            size=model_config.pool_size,
        )
        for branch in trace.branches:
            sampled_nodes = [node for node in branch.nodes if node.sampled]
            if len(sampled_nodes) != 1:
                raise ValueError(f"opsd requires exactly one sampled message per branch, got {len(sampled_nodes)}.")
            sampled_index = next(idx for idx, node in enumerate(branch.nodes) if node.sampled)
            prompt_nodes = branch.nodes[:sampled_index]
            messages = [node.message.model_dump(exclude_none=True) for node in prompt_nodes]
            user_indices = [idx for idx, message in enumerate(messages) if message.get("role") == "user"]
            if not user_indices:
                raise ValueError("opsd found no user message to condition.")
            last_user = messages[user_indices[-1]]
            question = last_user.get("content")
            if not isinstance(question, str):
                raise ValueError("opsd supports text-only user prompts.")
            last_user["content"] = template.format(question=question, demonstration=demonstration)
            prefix_ids = renderer.render_ids(messages, add_generation_prompt=True)
            sampled_token_ids = [
                token_id for token_id, sampled in zip(branch.token_ids, branch.sampled_mask, strict=True) if sampled
            ]
            response = await client.post(
                f"{base}/inference/v1/generate",
                cast_to=httpx.Response,
                body={
                    "model": model_config.model,
                    "token_ids": list(prefix_ids) + sampled_token_ids,
                    "sampling_params": {
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "prompt_logprobs": 1,
                    },
                },
            )
            parsed = GenerateResponse.model_validate_json(response.content)
            model_logprobs: list[float] = []
            for entry in parsed.prompt_logprobs or []:
                if not entry:
                    model_logprobs.append(0.0)
                    continue
                first = next(iter(entry.values()))
                lp = first.logprob if hasattr(first, "logprob") else first.get("logprob")
                model_logprobs.append(float(lp) if lp is not None else 0.0)
            if len(model_logprobs) != len(prefix_ids) + len(sampled_token_ids):
                raise ValueError(
                    f"opsd prefill returned {len(model_logprobs)} logprobs for "
                    f"{len(prefix_ids) + len(sampled_token_ids)} tokens."
                )
            completion_logprobs = model_logprobs[-len(sampled_token_ids) :] if sampled_token_ids else []
            completion_index = 0
            advantages: list[float] = []
            for actor_lp, sampled in zip(branch.logprobs, branch.sampled_mask, strict=True):
                if sampled:
                    advantages.append(completion_logprobs[completion_index] - actor_lp)
                    completion_index += 1
                else:
                    advantages.append(0.0)
            branch.advantages = advantages
            branch.mask = list(branch.sampled_mask)
    return traces


BUILTIN_ADVANTAGES: dict[str, TraceAdvantageFn] = {
    grpo.__name__: grpo,
    max_rl.__name__: max_rl,
    reward.__name__: reward,
    sft.__name__: sft,
    echo.__name__: echo,
    opd.__name__: opd,
    opsd.__name__: opsd,
}


def load_advantage(name: str) -> TraceAdvantageFn:
    fn = BUILTIN_ADVANTAGES.get(name)
    if fn is None:
        fn = cast(TraceAdvantageFn, import_object(name))
    if not callable(fn) or not getattr(fn, "advantage", False):
        raise ValueError(f"Advantage {name!r} must be decorated with @vf.advantage.")
    return fn


def advantage_loss(fn: TraceAdvantageFn) -> AdvantageLoss:
    return cast(AdvantageLoss, getattr(fn, "advantage_loss"))


def advantage_scope(fn: TraceAdvantageFn) -> AdvantageScope:
    return cast(AdvantageScope, getattr(fn, "advantage_scope"))


async def run_advantage(fn: TraceAdvantageFn, traces: list[vf.Trace]) -> list[vf.Trace]:
    result = fn(traces)
    if inspect.isawaitable(result):
        return await result
    return result
