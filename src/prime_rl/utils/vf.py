import asyncio
from collections import defaultdict
from itertools import cycle

from datasets import Dataset
from openai import AsyncOpenAI
from verifiers import Environment
from verifiers.types import GenerateMetadata, GenerateOutputs


def merge_metadata(generate_metadata_list: list[GenerateMetadata]) -> GenerateMetadata:
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
    return GenerateMetadata(
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


def merge_outputs(generate_outputs_list: list[GenerateOutputs]) -> GenerateOutputs:
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
    return GenerateOutputs(
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
    env: Environment,
    model_name: str,
    problem: dict,
    rollouts_per_example: int,
    sampling_args: dict,
    semaphore: asyncio.Semaphore | None,
    use_tqdm: bool = False,
) -> GenerateOutputs:
    """Asynchronously generate and score rollouts for one problem."""
    return await env.generate(
        inputs=Dataset.from_list([problem] * rollouts_per_example),
        client=client,
        model=model_name,
        sampling_args=sampling_args,
        semaphore=semaphore,
        use_tqdm=use_tqdm,
    )


async def generate_batch(
    clients: list[AsyncOpenAI],
    env: Environment,
    model_name: str,
    problems: list[dict],
    rollouts_per_example: int,
    sampling_args: dict,
    semaphore: asyncio.Semaphore | None,
    use_tqdm: bool = True,
) -> GenerateOutputs:
    """Asynchronously generate and score rollouts for a list of problems."""
    from tqdm import tqdm

    if use_tqdm:
        pbar = tqdm(total=len(problems) * rollouts_per_example, desc="Generating rollouts")
    else:
        pbar = None

    async def generate_group_with_progress(client, problem):
        """Generate rollouts for one problem and update progress."""
        result = await generate_group(
            client, env, model_name, problem, rollouts_per_example, sampling_args, semaphore, use_tqdm=False
        )
        if pbar:
            pbar.update(rollouts_per_example)
        return result

    try:
        generate_outputs_list: list[GenerateOutputs] = await asyncio.gather(
            *[generate_group_with_progress(client, problem) for client, problem in zip(cycle(clients), problems)]
        )
    finally:
        if pbar:
            pbar.close()

    return merge_outputs(generate_outputs_list)
