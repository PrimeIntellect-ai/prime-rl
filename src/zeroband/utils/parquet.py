import pyarrow as pa

pa_schema = pa.schema(
    [
        ("input_tokens", pa.list_(pa.int32())),
        ("output_tokens", pa.list_(pa.int32())),
        ("input_logprobs", pa.list_(pa.float32())),  # Optional - can be null (set to list of zeros)
        ("output_logprobs", pa.list_(pa.float32())),  # Optional - can be null (set to list of zeros)
        ("input_mask", pa.list_(pa.bool_())),
        ("output_mask", pa.list_(pa.bool_())),
        ("prompt", pa.string()),
        ("problem_id", pa.string()),
        ("completion", pa.string()),
        ("advantages", pa.float32()),
        ("rewards", pa.float32()),
        ("task_rewards", pa.float32()),
        ("length_penalties", pa.float32()),
        ("proofs", pa.binary()),
        ("step", pa.int32()),
        ("target_lengths", pa.int32()),
        ("task_type", pa.string()),
        ("seed", pa.int64()),  # Optional - can be null
        ("temperature", pa.float32()),
    ]
)
