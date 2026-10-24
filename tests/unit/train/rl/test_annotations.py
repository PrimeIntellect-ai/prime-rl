from types import SimpleNamespace

import orjson
import torch

from prime_rl.trainer.rl.annotations import AnnotationWriter


def test_export_serializes_masked_positions_as_null():
    writer = AnnotationWriter(SimpleNamespace(cp_enabled=False), SimpleNamespace(), float_decimals=2)
    micro_batch = {
        "trace_ids": ["a", "b"],
        "branch_indices": [0, 1],
        # The second sample carries two tokens of trailing padding
        "sequence_lengths": [3, 4],
        "padding": 2,
        "loss_mask": torch.tensor([[False, True, True, False, True, False, False]]),
        "env_names": ["env", "env"],
    }
    model_output = {
        "logprobs": torch.tensor([[-0.111, -0.222, -0.333, -0.444, float("-inf"), -0.666, -0.777]]),
        "entropy": torch.tensor([[1.111, 2.222, 3.333, 4.444, 5.557, 6.666, 7.777]]),
    }

    writer.export(micro_batch, model_output)

    records = [orjson.loads(orjson.dumps(record, option=orjson.OPT_SERIALIZE_NUMPY)) for record in writer._pending]
    assert [(record["trace_id"], record["branches"][0]["index"]) for record in records] == [("a", 0), ("b", 1)]
    assert records[0]["branches"][0]["trainer_logprobs"] == [None, -0.22, -0.33]
    assert records[0]["branches"][0]["entropies"] == [None, 2.22, 3.33]
    assert records[1]["branches"][0]["trainer_logprobs"] == [None, None]
    assert records[1]["branches"][0]["entropies"] == [None, 5.56]
