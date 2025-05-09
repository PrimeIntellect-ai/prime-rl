import pyarrow as pa

pa_schema = pa.schema(
    [
        ("input_tokens", pa.list_(pa.int32())),
        ("output_tokens", pa.list_(pa.int32())),
        ("input_logprobs", pa.list_(pa.float32())),
        ("output_logprobs", pa.list_(pa.float32())),
        ("advantages", pa.float32()),
        ("rewards", pa.float32()),
        ("task_rewards", pa.float32()),
        ("length_penalties", pa.float32()),
        ("proofs", pa.binary()),
        ("step", pa.int32()),
        ("target_lengths", pa.int32()),
    ]
)
