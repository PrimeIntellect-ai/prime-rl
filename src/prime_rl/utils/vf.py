import asyncio
from collections import defaultdict
from itertools import cycle
from typing import TypedDict

import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI


def merge_metadata(generate_metadata_list: list[vf.GenerateMetadata]) -> vf.GenerateMetadata:
    """Merge multiple GenerateMetadata into a single GenerateMetadata."""
    num_examples = len(generate_metadata_list)  # Assumes one generate metadata per example
    time_ms = max(metadata.time_ms for metadata in generate_metadata_list)
    avg_reward = sum(metadata.avg_reward for metadata in generate_metadata_list) / num_examples
    avg_metrics = {
        key: sum(metadata.avg_metrics[key] for metadata in generate_metadata_list) / num_examples
        for key in generate_metadata_list[0].avg_metrics
    }
    state_columns = []
    for metadata in generate_metadata_list:
        state_columns.extend(metadata.state_columns)
    return vf.GenerateMetadata(
        env_id=generate_metadata_list[0].env_id,
        env_args=generate_metadata_list[0].env_args,
        model=generate_metadata_list[0].model,
        base_url=generate_metadata_list[0].base_url,
        num_examples=num_examples,
        rollouts_per_example=generate_metadata_list[0].rollouts_per_example,
        sampling_args=generate_metadata_list[0].sampling_args,
        date=generate_metadata_list[0].date,
        time_ms=time_ms,
        avg_reward=avg_reward,
        avg_metrics=avg_metrics,
        state_columns=state_columns,
        path_to_save=generate_metadata_list[0].path_to_save,
    )


def merge_outputs(generate_outputs_list: list[vf.GenerateOutputs]) -> vf.GenerateOutputs:
    """Merge multiple GenerateOutputs into a single GenerateOutputs."""
    example_id, prompt, completion, answer, state, reward, info, task, metrics = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        defaultdict(list),
    )
    for generate_output in generate_outputs_list:
        example_id.extend(generate_output.example_id)
        prompt.extend(generate_output.prompt)
        completion.extend(generate_output.completion)
        answer.extend(generate_output.answer)
        state.extend(generate_output.state)
        reward.extend(generate_output.reward)
        info.extend(generate_output.info)
        task.extend(generate_output.task)
        for key, value in generate_output.metrics.items():
            metrics[key].extend(value)
    metadata = merge_metadata([generate_output.metadata for generate_output in generate_outputs_list])
    return vf.GenerateOutputs(
        prompt=prompt,
        completion=completion,
        answer=answer,
        state=state,
        reward=reward,
        info=info,
        task=task,
        metrics=metrics,
        metadata=metadata,
        example_id=example_id,
    )


async def generate_group(
    client: AsyncOpenAI,
    env: vf.Environment,
    model_name: str,
    problem: dict,
    rollouts_per_example: int,
    sampling_args: dict,
    use_tqdm: bool = False,
    max_concurrent: int | None = None,
) -> vf.GenerateOutputs:
    """Asynchronously generate and score rollouts for one problem."""
    generate_kwargs = {
        "inputs": Dataset.from_list([problem] * rollouts_per_example),
        "client": client,
        "model": model_name,
        "sampling_args": sampling_args,
        "use_tqdm": use_tqdm,
    }
    if max_concurrent is not None:
        generate_kwargs["max_concurrent"] = max_concurrent
    return await env.generate(**generate_kwargs)


async def generate_batch(
    clients: list[AsyncOpenAI],
    env: vf.Environment,
    model_name: str,
    problems: list[dict],
    rollouts_per_example: int,
    sampling_args: dict,
    pbar_description: str = "Generating rollouts",
    max_concurrent: int | None = None,
) -> vf.GenerateOutputs:
    """Asynchronously generate and score rollouts for a list of problems."""
    from tqdm import tqdm

    pbar = tqdm(total=len(problems) * rollouts_per_example, desc=pbar_description)

    async def generate_group_with_progress(client, problem):
        """Generate rollouts for one problem and update progress."""
        result = await generate_group(
            client,
            env,
            model_name,
            problem,
            rollouts_per_example,
            sampling_args,
            use_tqdm=False,
            max_concurrent=max_concurrent,
        )
        pbar.update(rollouts_per_example)
        return result

    try:
        generate_outputs_list: list[vf.GenerateOutputs] = await asyncio.gather(
            *[generate_group_with_progress(client, problem) for client, problem in zip(cycle(clients), problems)]
        )
    finally:
        pbar.close()

    return merge_outputs(generate_outputs_list)


# Non-batched version of vf.ProcessedOutputs
# Also includes advantage and example_id field
class Rollout(TypedDict):
    example_id: int
    task: str  # Typically the env name
    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    reward: float
    advantage: float
    is_truncated: bool
    metrics: dict[str, float]


def make_rollouts(
    generate_outputs: vf.GenerateOutputs,
    processed_outputs: vf.ProcessedOutputs,
    advantages: list[float],
    all_is_truncated: list[bool],
) -> list[Rollout]:
    """Processs vf.ProcessedOutputs to a list of rollouts."""
    rollouts = []
    for i, (
        example_id,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
        completion_logprobs,
        reward,
        advantage,
        is_truncated,
        task,
    ) in enumerate(
        zip(
            generate_outputs.example_id,
            processed_outputs.prompt_ids,
            processed_outputs.prompt_mask,
            processed_outputs.completion_ids,
            processed_outputs.completion_mask,
            processed_outputs.completion_logprobs,
            processed_outputs.rewards,
            advantages,
            all_is_truncated,
            generate_outputs.task,
        )
    ):
        metrics = {k: v[i] for k, v in generate_outputs.metrics.items()}
        rollouts.append(
            Rollout(
                example_id=example_id,
                task=task,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                completion_logprobs=completion_logprobs,
                reward=reward,
                metrics=metrics,
                advantage=advantage,
                is_truncated=is_truncated,
            )
        )

    return rollouts
