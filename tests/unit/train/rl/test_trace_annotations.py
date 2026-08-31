import json
import math

import torch

from prime_rl.trainer.rl.trace_annotations import TraceAnnotationWriter


def test_writer_records_one_update_per_annotated_sequence(tmp_path):
    # Two packed sequences; the second carries two tokens of trailing padding
    # (env_name "" and loss-masked False), plus a NaN logprob in the first.
    logprobs = torch.tensor([[-9.0, -0.1, float("nan"), -0.3, -9.0, -0.5, -0.6, -0.7, 0.0, 0.0]])
    entropies = torch.tensor([[9.0, 0.1, 0.2, 0.3, 9.0, 0.5, 0.6, 0.7, 0.0, 0.0]])
    micro_batch = {
        "trace_ids": ["trace-a", "trace-b"],
        "branch_indices": [0, 1],
        "sequence_lengths": [4, 6],
        "loss_mask": torch.tensor([[False, True, True, False, False, True, True, False, False, False]]),
        "env_names": ["env"] * 8 + ["", ""],
    }

    writer = TraceAnnotationWriter(tmp_path, rank=0)
    writer.export(3, micro_batch, {"logprobs": logprobs, "entropy": entropies})

    step_dir = tmp_path / "trace_annotations" / "step_3"
    records = [json.loads(line) for line in (step_dir / "rank_0.jsonl").read_text().splitlines()]
    assert [record["trace_id"] for record in records] == ["trace-a", "trace-b"]

    branch_a = records[0]["branches"][0]
    assert branch_a["index"] == 0
    # Index 0 crosses the packing boundary; the NaN becomes null.
    assert branch_a["trainer_logprobs"][0] is None
    assert branch_a["trainer_logprobs"][2] is None
    assert branch_a["trainer_logprobs"][1] == -0.10000000149011612
    assert branch_a["entropies"][0] is None
    assert len(branch_a["entropies"]) == 4

    branch_b = records[1]["branches"][0]
    assert branch_b["index"] == 1
    # Trailing padding is trimmed off the last sequence.
    assert len(branch_b["trainer_logprobs"]) == 4

    # A sequence without identity or without loss-masked tokens writes nothing.
    writer.export(
        3,
        {
            "trace_ids": ["", "trace-c"],
            "branch_indices": [-1, 0],
            "sequence_lengths": [2, 2],
            "loss_mask": torch.tensor([[True, True, False, False]]),
            "env_names": ["env"] * 4,
        },
        {"logprobs": torch.zeros(1, 4), "entropy": torch.zeros(1, 4)},
    )
    assert len((step_dir / "rank_0.jsonl").read_text().splitlines()) == 2

    assert not (step_dir / "STABLE").exists()
    writer.mark_stable()
    assert (step_dir / "STABLE").exists()

    assert all(
        value is None or math.isfinite(value)
        for record in records
        for branch in record["branches"]
        for stream in (branch["trainer_logprobs"], branch["entropies"])
        for value in stream
    )
