import pyarrow as pa
from vllm import RequestOutput
from vllm.sequence import SampleLogprobs

from zeroband.inference.rewards import RequestRewards
from zeroband.utils.parquet import pa_schema


def extract_logprobs(sample_logprobs: SampleLogprobs | None) -> list[float] | None:
    """Extract logprobs from vllm output. Returns None if logprobs are not available."""
    if sample_logprobs is None:
        return None

    logprobs = []
    for logprob in sample_logprobs:
        assert isinstance(logprob, dict), "Logprobs should be a dict"
        assert len(logprob) == 1, "Logprobs should be a dict with 1 key"

        _token_id, logprob_p = list(logprob.items())[0]
        logprobs.append(logprob_p.logprob)

    return logprobs


def get_parquet_table(
    request_outputs: list[RequestOutput],
    request_rewards: list[RequestRewards],
    prompts: list[str],
    proofs: list[bytes],
    step: int,
    target_lengths: list[int],
    enable_logprobs: bool,
) -> pa.Table:
    # Iterator over proofs
    proof_iter = iter(proofs)

    # Create flattened list of records for PyArrow table
    records = []
    for request_output, request_rewards, prompt, target_length in zip(request_outputs, request_rewards, prompts, target_lengths):
        assert request_output.request_id == request_rewards.request_id
        for output, reward in zip(request_output.outputs, request_rewards.rewards):
            assert output.index == reward.completion_id

            # Extract logprobs if enabled and available
            output_logprobs = extract_logprobs(output.logprobs) if enable_logprobs else None
            # For input logprobs, we typically don't have them from vllm, so set to None or zeros
            input_logprobs = [0.0] * len(request_output.prompt_token_ids) if output_logprobs is not None else None

            records.append(
                {
                    "input_tokens": request_output.prompt_token_ids,
                    "output_tokens": output.token_ids,
                    "input_logprobs": input_logprobs,
                    "output_logprobs": output_logprobs,
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
