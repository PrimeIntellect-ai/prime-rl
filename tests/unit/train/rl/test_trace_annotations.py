import math
from types import SimpleNamespace

import pytest
import torch

from prime_rl.trainer.rl.trace_annotations import TraceAnnotationWriter


@pytest.fixture
def writer(monkeypatch) -> TraceAnnotationWriter:
    """A rank-0 writer whose flush captures the records instead of logging them."""
    writer = TraceAnnotationWriter(SimpleNamespace(cp_enabled=False), SimpleNamespace(rank=0, world_size=1))
    logged: list[dict] = []

    async def log_annotations(updates):
        logged.extend(updates)

    monkeypatch.setattr("prime_rl.trainer.rl.trace_annotations.monitors.log_annotations", log_annotations)
    writer.logged = logged
    return writer


def test_writer_records_one_update_per_trained_sequence(writer):
    # Two packed sequences; the second carries two tokens of trailing padding (env_name
    # "" and loss-masked False), plus a NaN logprob in the first.
    logprobs = torch.tensor([[-9.0, -0.1, float("nan"), -0.3, -9.0, -0.5, -0.6, -0.7, 0.0, 0.0]])
    entropies = torch.tensor([[9.0, 0.1, 0.2, 0.3, 9.0, 0.5, 0.6, 0.7, 0.0, 0.0]])
    writer.export(
        {
            "trace_ids": ["trace-a", "trace-b"],
            "branch_indices": [0, 1],
            "sequence_lengths": [4, 6],
            "loss_mask": torch.tensor([[False, True, True, False, False, True, True, False, False, False]]),
            "env_names": ["env"] * 8 + ["", ""],
        },
        {"logprobs": logprobs, "entropy": entropies},
    )
    writer.flush()

    record_a, record_b = writer.logged
    assert [record["trace_id"] for record in writer.logged] == ["trace-a", "trace-b"]

    branch_a = record_a["branches"][0]
    assert branch_a["index"] == 0
    # Index 0 crosses the packing boundary; the NaN becomes null.
    assert branch_a["trainer_logprobs"][0] is None
    assert branch_a["trainer_logprobs"][2] is None
    assert branch_a["trainer_logprobs"][1] == -0.10000000149011612
    assert branch_a["entropies"][0] is None
    assert len(branch_a["entropies"]) == 4

    # Trailing padding is trimmed off the last sequence.
    assert record_b["branches"][0]["index"] == 1
    assert len(record_b["branches"][0]["trainer_logprobs"]) == 4

    assert all(
        value is None or math.isfinite(value)
        for record in writer.logged
        for branch in record["branches"]
        for stream in (branch["trainer_logprobs"], branch["entropies"])
        for value in stream
    )


def test_writer_skips_unidentified_and_untrained_sequences(writer):
    writer.export(
        {
            "trace_ids": ["", "trace-c"],
            "branch_indices": [-1, 0],
            "sequence_lengths": [2, 2],
            "loss_mask": torch.tensor([[True, True, False, False]]),
            "env_names": ["env"] * 4,
        },
        {"logprobs": torch.zeros(1, 4), "entropy": torch.zeros(1, 4)},
    )
    writer.flush()
    assert writer.logged == []
