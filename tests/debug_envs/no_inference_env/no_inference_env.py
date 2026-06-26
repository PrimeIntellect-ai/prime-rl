from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import numpy as np
import pybase64
import verifiers as vf
from datasets import Dataset
from verifiers.clients import Client
from verifiers.types import RolloutInput, SamplingArgs, State, TimeSpan, TrajectoryStep


@dataclass(frozen=True)
class TokenPlan:
    prompt_len: int
    completion_lens: list[int]
    env_delta_lens: list[int]


def _constant_reward(**kwargs) -> float:
    return 1.0


def _spread(total: int, parts: int) -> list[int]:
    if parts <= 0:
        return []
    q, r = divmod(total, parts)
    return [q + (1 if i < r else 0) for i in range(parts)]


def _token_plan(seq_len: int, turns: int, prompt_len: int, completion_fraction: float) -> TokenPlan:
    if turns < 1:
        raise ValueError("turns must be >= 1")
    if seq_len <= prompt_len + turns:
        raise ValueError("seq_len is too small for the requested turns and prompt length")
    body = seq_len - prompt_len
    completion_total = max(turns, int(body * completion_fraction))
    completion_total = min(completion_total, body)
    env_total = body - completion_total
    # Only turns with tool calls receive env (tool-result) tokens.
    # has_tool_call = turn_idx % 3 != 1, so count tool-call turns in 0..turns-2.
    # When there are no tool-call turns (e.g. turns=1), fold env tokens into
    # completions so the seq_len invariant still holds.
    tool_call_turns = sum(1 for t in range(max(turns - 1, 0)) if t % 3 != 1)
    if tool_call_turns == 0:
        completion_total += env_total
        env_total = 0
    env_deltas = _spread(env_total, tool_call_turns)
    env_delta_lens = [env_deltas.pop(0) if (t % 3 != 1 and env_deltas) else 0 for t in range(max(turns - 1, 0))]
    return TokenPlan(
        prompt_len=prompt_len,
        completion_lens=_spread(completion_total, turns),
        env_delta_lens=env_delta_lens,
    )


def _ids(start: int, count: int, *, vocab_size: int) -> list[int]:
    # Keep IDs below the default gibberish threshold while still looking nontrivial.
    usable = min(vocab_size - 100, 90_000)
    return [100 + ((start + i) % usable) for i in range(count)]


def _non_negative_float(name: str, value: float) -> float:
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _routed_payload(length: int, layers: int, topk: int, n_experts: int, *, start: int, salt: int):
    if length <= 0:
        return None
    pattern = np.arange(layers * topk, dtype=np.uint16).reshape(layers, topk)
    pattern = ((pattern + salt) % min(n_experts, 256)).astype(np.uint8)
    routed = np.empty((length, layers, topk), dtype=np.uint8)
    routed[:] = pattern
    routed += np.arange(length, dtype=np.uint8).reshape(length, 1, 1)
    return {
        "data": pybase64.b64encode(memoryview(np.ascontiguousarray(routed))).decode("ascii"),
        "shape": [length, layers, topk],
        "start": start,
    }


class NoInferenceEnv(vf.Environment):
    def __init__(
        self,
        *,
        turns: int = 30,
        seq_len: int = 30_000,
        prompt_len: int = 128,
        completion_fraction: float = 0.70,
        include_routed_experts: bool = True,
        # Defaults mirror the GLM-5 routed-expert replay shape.
        routed_layers: int = 78,
        routed_topk: int = 8,
        n_routed_experts: int = 256,
        num_examples: int = 16,
        vocab_size: int = 154_880,
        rollout_delay_mean_seconds: float = 0.0,
        rollout_delay_std_seconds: float = 0.0,
        **kwargs,
    ) -> None:
        self.turns = turns
        self.seq_len = seq_len
        self.prompt_len = prompt_len
        self.completion_fraction = completion_fraction
        self.include_routed_experts = include_routed_experts
        self.routed_layers = routed_layers
        self.routed_topk = routed_topk
        self.n_routed_experts = n_routed_experts
        self.vocab_size = vocab_size
        self.plan = _token_plan(seq_len, turns, prompt_len, completion_fraction)
        self.rollout_delay_mean_seconds = _non_negative_float("rollout_delay_mean_seconds", rollout_delay_mean_seconds)
        self.rollout_delay_std_seconds = _non_negative_float("rollout_delay_std_seconds", rollout_delay_std_seconds)
        self.delay_rng = np.random.default_rng(0)

        rows = [
            {
                "question": f"Return the deterministic no-inference trajectory for example {i}.",
                "answer": "ok",
                "info": {
                    "no_inference_env": True,
                    "seq_len": seq_len,
                    "turns": turns,
                    "include_routed_experts": include_routed_experts,
                    "routed_layers": routed_layers,
                    "routed_topk": routed_topk,
                    "n_routed_experts": n_routed_experts,
                    "rollout_delay_mean_seconds": self.rollout_delay_mean_seconds,
                    "rollout_delay_std_seconds": self.rollout_delay_std_seconds,
                    "glm5_num_hidden_layers": 78,
                    "glm5_sparse_moe_layers": 75,
                },
            }
            for i in range(num_examples)
        ]

        tool_defs = [
            vf.Tool(
                name="lookup_record",
                description="Look up a deterministic debug record.",
                parameters={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            ),
            vf.Tool(
                name="write_note",
                description="Write a deterministic debug note.",
                parameters={
                    "type": "object",
                    "properties": {"note": {"type": "string"}},
                    "required": ["note"],
                },
            ),
        ]

        super().__init__(
            dataset=Dataset.from_list(rows),
            rubric=vf.Rubric(funcs=[_constant_reward]),
            tool_defs=tool_defs,
            score_rollouts=True,
            **kwargs,
        )

    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        timing = state["timing"]
        start_time = time.time()
        timing.generation.start = start_time
        timing.setup.start = start_time
        timing.setup.end = start_time

        delay_seconds = self._sample_rollout_delay_seconds()
        info = state.get("info")
        if isinstance(info, dict):
            info["sampled_rollout_delay_seconds"] = delay_seconds
        if delay_seconds > 0:
            delay_start = time.time()
            await asyncio.sleep(delay_seconds)
            delay_end = time.time()
            timing.env.spans.append(TimeSpan(start=delay_start, end=delay_end))

        build_start = time.time()
        trajectory, completion, final_input_tokens, final_output_tokens = self._build_trajectory(
            state["trajectory_id"],
            model=model,
            example_id=int(state["example_id"]),
        )
        build_end = time.time()
        timing.model.spans.append(TimeSpan(start=build_start, end=build_end))
        state["trajectory"] = trajectory
        state["completion"] = completion
        state["is_completed"] = True
        state["is_truncated"] = False
        state["stop_condition"] = "no_inference_complete"
        state["token_usage"] = {
            "input_tokens": float(sum(len(step["tokens"]["prompt_ids"]) for step in trajectory)),
            "output_tokens": float(sum(len(step["tokens"]["completion_ids"]) for step in trajectory)),
            "final_input_tokens": float(final_input_tokens),
            "final_output_tokens": float(final_output_tokens),
        }
        end_time = time.time()
        timing.generation.end = end_time
        return state

    def _sample_rollout_delay_seconds(self) -> float:
        mean = self.rollout_delay_mean_seconds
        std = self.rollout_delay_std_seconds
        if mean == 0.0 and std == 0.0:
            return 0.0
        if std == 0.0:
            return mean
        return max(0.0, float(self.delay_rng.normal(mean, std)))

    def _build_trajectory(self, trajectory_id: str, *, model: str, example_id: int):
        prompt_tokens = _ids(example_id * 17, self.plan.prompt_len, vocab_size=self.vocab_size)
        current_prompt_ids = list(prompt_tokens)
        current_messages = [
            {
                "role": "system",
                "content": "You are executing a deterministic no-inference memory rollout.",
            },
            {
                "role": "user",
                "content": f"Example {example_id}: produce the fixed heavy trajectory.",
            },
        ]

        trajectory: list[TrajectoryStep] = []
        completion_messages: list[dict] = []
        token_cursor = self.plan.prompt_len

        for turn_idx, completion_len in enumerate(self.plan.completion_lens):
            completion_ids = _ids(token_cursor, completion_len, vocab_size=self.vocab_size)
            token_cursor += completion_len

            has_tool_call = turn_idx % 3 != 1
            tool_call_id = f"call_no_inference_{example_id}_{turn_idx}"
            assistant_msg = {
                "role": "assistant",
                "content": f"Turn {turn_idx}: deterministic reasoning chunk.",
                "reasoning_content": (
                    f"Reasoning turn {turn_idx}: inspect synthetic state, decide whether to call a tool, "
                    "then continue the fixed replay."
                ),
            }
            if has_tool_call:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tool_call_id,
                        "name": "lookup_record" if turn_idx % 2 == 0 else "write_note",
                        "arguments": f'{{"key": "example-{example_id}-turn-{turn_idx}"}}',
                    }
                ]

            routed = None
            routed_start = 0
            if self.include_routed_experts:
                if turn_idx == 0:
                    routed_len = len(current_prompt_ids) + completion_len - 1
                else:
                    prefix_len = len(trajectory[-1]["tokens"]["prompt_ids"]) + len(
                        trajectory[-1]["tokens"]["completion_ids"]
                    )
                    routed_start = prefix_len - 1
                    routed_len = len(current_prompt_ids) + completion_len - prefix_len
                routed = _routed_payload(
                    routed_len,
                    self.routed_layers,
                    self.routed_topk,
                    self.n_routed_experts,
                    start=routed_start,
                    salt=example_id + turn_idx,
                )

            tokens = {
                "prompt_ids": list(current_prompt_ids),
                "prompt_mask": [0] * len(current_prompt_ids),
                "completion_ids": completion_ids,
                "completion_mask": [1] * completion_len,
                "completion_logprobs": [-0.25] * completion_len,
                "overlong_prompt": False,
                "is_truncated": False,
                "routed_experts": routed,
            }

            response = vf.Response(
                id=f"no-inference-{example_id}-{turn_idx}",
                created=int(time.time()),
                model=model,
                usage=vf.Usage(
                    prompt_tokens=len(current_prompt_ids),
                    reasoning_tokens=completion_len // 3,
                    completion_tokens=completion_len,
                    total_tokens=len(current_prompt_ids) + completion_len,
                ),
                message=vf.ResponseMessage(
                    role="assistant",
                    content=assistant_msg["content"],
                    reasoning_content=assistant_msg["reasoning_content"],
                    tool_calls=assistant_msg.get("tool_calls"),
                    finish_reason="tool_calls" if has_tool_call else "stop",
                    is_truncated=False,
                    tokens=None,
                ),
            )

            trajectory.append(
                TrajectoryStep(
                    prompt=list(current_messages),
                    completion=[assistant_msg],
                    response=response,
                    tokens=tokens,
                    reward=None,
                    advantage=None,
                    is_truncated=False,
                    trajectory_id=trajectory_id,
                    extras={"turn_idx": turn_idx, "no_inference_env": True},
                )
            )
            completion_messages.append(assistant_msg)
            current_messages = [*current_messages, assistant_msg]
            current_prompt_ids = [*current_prompt_ids, *completion_ids]

            if has_tool_call and turn_idx < len(self.plan.env_delta_lens):
                env_len = self.plan.env_delta_lens[turn_idx]
                env_ids = _ids(token_cursor, env_len, vocab_size=self.vocab_size)
                token_cursor += env_len
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Tool result for turn {turn_idx}: " + ("debug payload " * 8),
                }
                current_messages = [*current_messages, tool_msg]
                completion_messages.append(tool_msg)
                current_prompt_ids = [*current_prompt_ids, *env_ids]

        final_input_tokens = len(trajectory[-1]["tokens"]["prompt_ids"])
        final_output_tokens = len(trajectory[-1]["tokens"]["completion_ids"])
        assert len(prompt_tokens) + sum(self.plan.completion_lens) + sum(self.plan.env_delta_lens) == self.seq_len
        return (
            trajectory,
            completion_messages,
            final_input_tokens,
            final_output_tokens,
        )


def load_environment(**kwargs) -> vf.Environment:
    return NoInferenceEnv(**kwargs)
