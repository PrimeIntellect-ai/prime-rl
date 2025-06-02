import pyarrow as pa
from vllm import RequestOutput

from zeroband.inference.rewards import RequestRewards
from zeroband.utils.parquet import pa_schema


def get_parquet_table(
    request_outputs: list[RequestOutput],
    request_rewards: list[RequestRewards],
    prompts: list[str],
    proofs: list[bytes],
    step: int,
    target_lengths: list[int],
) -> pa.Table:
    # Iterator over proofs
    proof_iter = iter(proofs)

    # Create flattened list of records for PyArrow table
    records = []
    for request_output, request_rewards, prompt, target_length in zip(request_outputs, request_rewards, prompts, target_lengths):
        assert request_output.request_id == request_rewards.request_id
        for output, reward in zip(request_output.outputs, request_rewards.rewards):
            assert output.index == reward.completion_id
            records.append(
                {
                    "input_tokens": request_output.prompt_token_ids,
                    "output_tokens": output.token_ids,
                    "prompt": prompt,
                    "completion": output.text,
                    "advantages": reward.advantage,
                    "rewards": reward.reward,
                    "task_rewards": reward.task_reward,
                    "length_penalties": reward.length_penalty,
                    "proofs": next(proof_iter) if len(output.token_ids) > 1 else b"",
                    "step": step,
                    "target_lengths": target_length,
                    "task_type": request_rewards.task_type,
                }
            )

    return pa.Table.from_pylist(records, schema=pa_schema)
