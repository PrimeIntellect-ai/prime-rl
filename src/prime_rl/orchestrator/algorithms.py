from __future__ import annotations

import asyncio

import verifiers.v1 as vf


class GRPO(vf.Algorithm[vf.AlgorithmConfig]):
    async def advantage(self, traces: list[vf.Trace]) -> list[vf.Trace]:
        rewards = [trace.reward for trace in traces]
        baseline = sum(rewards) / len(rewards) if rewards else 0.0
        for trace, reward in zip(traces, rewards, strict=True):
            scalar = reward - baseline
            for branch in trace.branches:
                sampled_mask = [sampled and not trace.has_error for sampled in branch.sampled_mask]
                branch.advantages = [float(scalar) if sampled else 0.0 for sampled in sampled_mask]
                branch.mask = sampled_mask
        return traces


class MaxRL(vf.Algorithm[vf.AlgorithmConfig]):
    async def advantage(self, traces: list[vf.Trace]) -> list[vf.Trace]:
        rewards = [trace.reward for trace in traces]
        mean = sum(rewards) / len(rewards) if rewards else 0.0
        for trace, reward in zip(traces, rewards, strict=True):
            scalar = 0.0 if mean <= 0 else (reward - mean) / mean
            for branch in trace.branches:
                sampled_mask = [sampled and not trace.has_error for sampled in branch.sampled_mask]
                branch.advantages = [float(scalar) if sampled else 0.0 for sampled in sampled_mask]
                branch.mask = sampled_mask
        return traces


class RL(vf.Algorithm[vf.AlgorithmConfig]):
    async def advantage(self, traces: list[vf.Trace]) -> list[vf.Trace]:
        for trace in traces:
            for branch in trace.branches:
                sampled_mask = [sampled and not trace.has_error for sampled in branch.sampled_mask]
                branch.advantages = [float(trace.reward) if sampled else 0.0 for sampled in sampled_mask]
                branch.mask = sampled_mask
        return traces


class SFT(vf.Algorithm[vf.AlgorithmConfig]):
    def loss(self) -> str:
        return "ce"

    async def advantage(self, traces: list[vf.Trace]) -> list[vf.Trace]:
        for trace in traces:
            for branch in trace.branches:
                sampled_mask = [sampled and not trace.has_error for sampled in branch.sampled_mask]
                branch.advantages = [1.0 if sampled else 0.0 for sampled in sampled_mask]
                branch.mask = sampled_mask
        return traces


class Echo(vf.Algorithm[vf.AlgorithmConfig]):
    def loss(self) -> str:
        return "ce"

    async def advantage(self, traces: list[vf.Trace]) -> list[vf.Trace]:
        for trace in traces:
            for branch in trace.branches:
                values: list[float] = []
                mask: list[bool] = []
                for node in branch.nodes:
                    role = getattr(node.message, "role", None)
                    for sampled in node.mask:
                        use_token = not sampled and role == "tool" and not trace.has_error
                        values.append(0.1 if use_token else 0.0)
                        mask.append(use_token)
                branch.advantages = values
                branch.mask = mask
        return traces


class OPD(vf.Algorithm[vf.AlgorithmConfig]):
    def __init__(self, config: vf.AlgorithmConfig) -> None:
        super().__init__(config)
        self.runtime: vf.ModelRuntime | None = None

    async def setup(self, models: dict[str, vf.ModelRuntime]) -> None:
        model = "reference"
        if self.config.model_extra is not None:
            value = self.config.model_extra.get("model")
            if isinstance(value, str):
                model = value
            elif value is not None:
                raise ValueError("opd model must be a string model key")
        runtime = models.get(model)
        if runtime is None:
            raise ValueError(f"opd requires models[{model!r}]")
        if not isinstance(runtime.client, vf.TrainClient):
            raise ValueError(f"opd requires models[{model!r}].client to be token-capable")
        self.runtime = runtime

    async def advantage(self, traces: list[vf.Trace]) -> list[vf.Trace]:
        if self.runtime is None:
            raise RuntimeError("opd.setup() must run before opd.advantage()")
        calls: list[asyncio.Task[list[float]]] = []
        branch_refs: list[tuple[vf.Trace, int]] = []
        for trace in traces:
            for branch in trace.branches:
                calls.append(
                    asyncio.create_task(self.runtime.client.prefill_logprobs(self.runtime.model, branch.token_ids))
                )
                branch_refs.append((trace, branch.index))

        scored = await asyncio.gather(*calls) if calls else []
        for (trace, branch_index), reference_logprobs in zip(branch_refs, scored, strict=True):
            branch = trace.branches[branch_index]
            sampled_mask = [sampled and not trace.has_error for sampled in branch.sampled_mask]
            branch.advantages = [
                float(reference_logprob - rollout_logprob) if sampled else 0.0
                for reference_logprob, rollout_logprob, sampled in zip(
                    reference_logprobs, branch.logprobs, sampled_mask, strict=True
                )
            ]
            branch.mask = sampled_mask
        return traces


class OPSD(vf.Algorithm[vf.AlgorithmConfig]):
    def __init__(self, config: vf.AlgorithmConfig) -> None:
        super().__init__(config)
        self.runtime: vf.ModelRuntime | None = None

    async def setup(self, models: dict[str, vf.ModelRuntime]) -> None:
        model = "reference"
        if self.config.model_extra is not None:
            value = self.config.model_extra.get("model")
            if isinstance(value, str):
                model = value
            elif value is not None:
                raise ValueError("opsd model must be a string model key")
        runtime = models.get(model)
        if runtime is None:
            raise ValueError(f"opsd requires models[{model!r}]")
        if not isinstance(runtime.client, vf.TrainClient):
            raise ValueError(f"opsd requires models[{model!r}].client to be token-capable")
        if runtime.renderer is None:
            raise ValueError(f"opsd requires models[{model!r}].renderer")
        self.runtime = runtime

    async def advantage(self, traces: list[vf.Trace]) -> list[vf.Trace]:
        if self.runtime is None:
            raise RuntimeError("opsd.setup() must run before opsd.advantage()")
        calls: list[asyncio.Task[list[float]]] = []
        branch_refs: list[tuple[vf.Trace, int, int]] = []
        for trace in traces:
            if len(trace.branches) != 1:
                raise ValueError(f"opsd supports single-branch traces only; got {len(trace.branches)}")
            branch = trace.branches[0]
            sampled_nodes = [node_index for node_index, node in enumerate(branch.nodes) if any(node.mask)]
            if len(sampled_nodes) != 1:
                raise ValueError(f"opsd supports one sampled model turn; got {len(sampled_nodes)}")
            demonstration = trace.info.get("demonstration")
            if demonstration is None and hasattr(trace.task, "demonstration"):
                demonstration = getattr(trace.task, "demonstration")
            if demonstration is None and hasattr(trace.task, "answer"):
                demonstration = getattr(trace.task, "answer")
            if demonstration is None:
                raise ValueError("opsd requires trace.info['demonstration'], task.demonstration, or task.answer")
            sampled_node_index = sampled_nodes[0]
            messages: list[dict[str, object]] = [
                node.message.model_dump(mode="json") for node in branch.nodes[:sampled_node_index]
            ]
            user_indices = [index for index, message in enumerate(messages) if message.get("role") == "user"]
            if not user_indices:
                raise ValueError("opsd found no user message to condition")
            last_user = messages[user_indices[-1]]
            question = last_user.get("content")
            if not isinstance(question, str):
                raise ValueError("opsd supports text-only user prompts")
            last_user["content"] = (
                f"{question}\n\n"
                "Here is an example of an expert response:\n"
                f"<demonstration>\n{demonstration}\n</demonstration>\n\n"
                "Answer with a response of your own."
            )
            prefix_ids = await asyncio.to_thread(self.runtime.renderer.render_ids, messages, add_generation_prompt=True)
            sampled_token_ids = [
                token_id for token_id, sampled in zip(branch.token_ids, branch.sampled_mask, strict=True) if sampled
            ]
            calls.append(
                asyncio.create_task(
                    self.runtime.client.prefill_logprobs(self.runtime.model, prefix_ids + sampled_token_ids)
                )
            )
            branch_refs.append((trace, branch.index, len(sampled_token_ids)))

        scored = await asyncio.gather(*calls) if calls else []
        for (trace, branch_index, sampled_count), full_reference_logprobs in zip(branch_refs, scored, strict=True):
            branch = trace.branches[branch_index]
            reference_completion_logprobs = full_reference_logprobs[-sampled_count:] if sampled_count else []
            sampled_mask = [sampled and not trace.has_error for sampled in branch.sampled_mask]
            values: list[float] = []
            completion_offset = 0
            for sampled, rollout_logprob in zip(sampled_mask, branch.logprobs, strict=True):
                if sampled:
                    values.append(float(reference_completion_logprobs[completion_offset] - rollout_logprob))
                    completion_offset += 1
                else:
                    values.append(0.0)
            branch.advantages = values
            branch.mask = sampled_mask
        return traces


BUILTIN_ALGORITHMS: dict[str, type[vf.Algorithm]] = {
    "grpo": GRPO,
    "max_rl": MaxRL,
    "rl": RL,
    "sft": SFT,
    "echo": Echo,
    "opd": OPD,
    "opsd": OPSD,
}
