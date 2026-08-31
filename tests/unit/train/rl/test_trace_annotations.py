import json
import math
from types import SimpleNamespace

import torch

from prime_rl.trainer.rl.trace_annotations import TraceAnnotationWriter


def make_writer(tmp_path) -> TraceAnnotationWriter:
    parallel_dims = SimpleNamespace(cp_enabled=False)
    world = SimpleNamespace(rank=0, world_size=1)
    return TraceAnnotationWriter(tmp_path, parallel_dims, world)


def test_writer_appends_updates_next_to_arrival_files(tmp_path):
    # Two packed sequences from different arrival steps; the second carries two tokens
    # of trailing padding (env_name "" and loss-masked False), plus a NaN logprob in the
    # first sequence.
    logprobs = torch.tensor([[-9.0, -0.1, float("nan"), -0.3, -9.0, -0.5, -0.6, -0.7, 0.0, 0.0]])
    entropies = torch.tensor([[9.0, 0.1, 0.2, 0.3, 9.0, 0.5, 0.6, 0.7, 0.0, 0.0]])
    micro_batch = {
        "trace_ids": ["trace-a", "trace-b"],
        "branch_indices": [0, 1],
        "logged_at_steps": [2, 3],
        "sequence_lengths": [4, 6],
        "loss_mask": torch.tensor([[False, True, True, False, False, True, True, False, False, False]]),
        "env_names": ["env"] * 8 + ["", ""],
    }

    writer = make_writer(tmp_path)
    writer.export(5, micro_batch, {"logprobs": logprobs, "entropy": entropies})
    writer.flush()

    def records(step: int) -> list[dict]:
        path = tmp_path / "traces" / f"step_{step}" / "annotations" / "trainer.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    (record_a,) = records(2)
    assert record_a["trace_id"] == "trace-a"
    assert record_a["info"] == {"train": {"trained_at_step": 5}}
    branch_a = record_a["branches"][0]
    assert branch_a["index"] == 0
    # Index 0 crosses the packing boundary; the NaN becomes null.
    assert branch_a["trainer_logprobs"][0] is None
    assert branch_a["trainer_logprobs"][2] is None
    assert branch_a["trainer_logprobs"][1] == -0.10000000149011612
    assert branch_a["entropies"][0] is None
    assert len(branch_a["entropies"]) == 4

    (record_b,) = records(3)
    branch_b = record_b["branches"][0]
    assert branch_b["index"] == 1
    # Trailing padding is trimmed off the last sequence.
    assert len(branch_b["trainer_logprobs"]) == 4

    assert all(
        value is None or math.isfinite(value)
        for record in (record_a, record_b)
        for branch in record["branches"]
        for stream in (branch["trainer_logprobs"], branch["entropies"])
        for value in stream
    )


def test_writer_skips_unidentified_and_untrained_sequences(tmp_path):
    writer = make_writer(tmp_path)
    writer.export(
        1,
        {
            "trace_ids": ["", "trace-c"],
            "branch_indices": [-1, 0],
            "logged_at_steps": [-1, 1],
            "sequence_lengths": [2, 2],
            "loss_mask": torch.tensor([[True, True, False, False]]),
            "env_names": ["env"] * 4,
        },
        {"logprobs": torch.zeros(1, 4), "entropy": torch.zeros(1, 4)},
    )
    writer.flush()
    assert not (tmp_path / "traces").exists()
