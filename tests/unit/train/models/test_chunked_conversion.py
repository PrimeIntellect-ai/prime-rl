from dataclasses import dataclass

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.conversion_ops import ChunkedConvOp
from prime_rl.trainer.models.qwen3_8_flash_next.converting_qwen3_8_flash_next import NGramEmbeddingWeights
from prime_rl.utils.weights import get_chunked_conversion_ops


@dataclass
class PairChunks(ChunkedConvOp):
    runtime_weight: str

    @property
    def packet_count(self):
        return 2

    def hf_to_prime(self, sd):
        if "checkpoint.part0" in sd:
            sd[self.runtime_weight] = torch.cat([sd.pop("checkpoint.part0"), sd.pop("checkpoint.part1")])

    def iter_prime_to_hf(self, weight, resolve_rows):
        middle = (weight.shape[0] + 1) // 2
        yield {"checkpoint.part0": resolve_rows(weight, 0, middle)}
        yield {"checkpoint.part1": resolve_rows(weight, middle, weight.shape[0])}


class ChunkedTestModel(PreTrainedModelPrimeRL):
    def __init__(self):
        super().__init__(PretrainedConfig())
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(3, 5, bias=False)])

    @classmethod
    def conversion_chain(cls, config):
        return [PairChunks("model.layers.0.weight")]

    @classmethod
    def is_prime_state_dict(cls, sd):
        return "model.layers.0.weight" in sd


@pytest.mark.parametrize("rows", [0, 1, 3, 257])
@pytest.mark.parametrize("parts", [1, 3, 128])
def test_ngram_stream_matches_checkpoint_conversion(rows, parts):
    op = NGramEmbeddingWeights("checkpoint.embedding", "runtime.weight", parts)
    weight = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    intervals = []

    def resolve_rows(value, start, stop):
        intervals.append((start, stop))
        return value[start:stop].clone()

    packets = list(op.iter_prime_to_hf(weight, resolve_rows))
    assert len(packets) == op.packet_count == parts
    assert all(0 <= start <= stop <= rows for start, stop in intervals)
    actual = {name: value for packet in packets for name, value in packet.items()}
    shard_size = (rows + parts - 1) // parts
    for index in range(parts):
        torch.testing.assert_close(
            actual[f"checkpoint.embedding.shard_{index}.weight"],
            weight[index * shard_size : (index + 1) * shard_size],
            rtol=0,
            atol=0,
        )
    full = {op.runtime_weight: weight}
    op.prime_to_hf(full)
    assert full.keys() == actual.keys()
    op.hf_to_prime(full)
    torch.testing.assert_close(full[op.runtime_weight], weight, rtol=0, atol=0)


def test_chunked_conversion_is_not_ngram_specific():
    model = ChunkedTestModel()
    state = model.state_dict()
    operations = get_chunked_conversion_ops(model, state)
    op = operations["model.layers.0.weight"]
    assert isinstance(op, PairChunks)
    assert sum(operation.packet_count for operation in operations.values()) == 2
    expected = state[op.runtime_weight].clone()
    op.prime_to_hf(state)
    assert set(state) == {"checkpoint.part0", "checkpoint.part1"}
    op.hf_to_prime(state)
    torch.testing.assert_close(state[op.runtime_weight], expected)
    assert get_chunked_conversion_ops(model, {}) == {}
    assert get_chunked_conversion_ops(nn.Linear(3, 5), {}) == {}
    absent = {}
    op.prime_to_hf(absent)
    assert absent == {}


def test_chunked_conversion_resolves_compiled_weight_names():
    model = ChunkedTestModel()
    model.model.layers[0] = torch.compile(model.model.layers[0], backend="eager")
    operations = get_chunked_conversion_ops(model, model.state_dict())
    assert list(operations) == ["model.layers.0._orig_mod.weight"]
    assert operations["model.layers.0._orig_mod.weight"].runtime_weight == "model.layers.0.weight"
