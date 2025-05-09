import pyarrow as pa
from vllm import RequestOutput
from vllm.sequence import SampleLogprobs
from zeroband.utils.parquet import pa_schema


def get_own_logprobs(sample_logprobs: SampleLogprobs) -> float:
    logprobs = []

    for logprob in sample_logprobs:
        assert isinstance(logprob, dict), "Logprobs should be a dict"
        assert len(logprob) == 1, "Logprobs should be a dict with 1 key"

        _token_id, logprob_p = list(logprob.items())[0]
        logprobs.append(logprob_p.logprob)

    return logprobs


def get_parquet_table(
    generated_tokens: list[RequestOutput],
    grouped_advantages: dict[int, list[float]],
    grouped_rewards: dict[int, list[float]],
    grouped_task_rewards: dict[int, list[float]],
    grouped_length_penalties: dict[int, list[float]],
    proofs: list[bytes],
    step: int,
    target_lengths: list[int],
) -> pa.Table:
    input_tokens_list = []
    output_tokens_list = []
    input_logprobs_list = []
    output_logprobs_list = []
    advantages_list = []
    rewards_list = []
    task_rewards_list = []
    length_penalty_list = []
    proofs_list = []
    steps_list = []
    target_lengths_list = []

    proof_iter = iter(proofs)

    for request, target_len in zip(generated_tokens, target_lengths):
        request_id = request.request_id
        advantages = grouped_advantages[request_id]
        rewards = grouped_rewards[request_id]
        task_rewards = grouped_task_rewards[request_id]
        length_penalties = grouped_length_penalties[request_id]
        for adv, reward, task_reward, length_penalty, output in zip(advantages, rewards, task_rewards, length_penalties, request.outputs):
            input_tokens_list.append(request.prompt_token_ids)
            output_tokens_list.append(output.token_ids)
            input_logprobs_list.append([0] * len(request.prompt_token_ids))  # putting 0 for now as not needed in the grpo loss
            output_logprobs_list.append(get_own_logprobs(output.logprobs))
            advantages_list.append(adv)
            rewards_list.append(reward)
            task_rewards_list.append(task_reward)
            length_penalty_list.append(length_penalty)
            proofs_list.append(next(proof_iter) if len(output.token_ids) > 1 else b"")
            steps_list.append(step)
            target_lengths_list.append(target_len)

    arrays = [
        pa.array(input_tokens_list, type=pa.list_(pa.int32())),
        pa.array(output_tokens_list, type=pa.list_(pa.int32())),
        pa.array(input_logprobs_list, type=pa.list_(pa.float32())),
        pa.array(output_logprobs_list, type=pa.list_(pa.float32())),
        pa.array(advantages_list, type=pa.float32()),
        pa.array(rewards_list, type=pa.float32()),
        pa.array(task_rewards_list, type=pa.float32()),
        pa.array(length_penalty_list, type=pa.float32()),
        pa.array(proofs_list, type=pa.binary()),
        pa.array(steps_list, type=pa.int32()),
        pa.array(target_lengths_list, type=pa.int32()),
    ]
    return pa.Table.from_arrays(arrays, schema=pa_schema)
