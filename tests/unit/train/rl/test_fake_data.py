import pytest
import torch
from pydantic import ValidationError

from prime_rl.configs.trainer import FakeDataLoaderConfig
from prime_rl.trainer.rl.data import FakeDataLoader


def test_doc_len_packs_fixed_length_samples():
    config = FakeDataLoaderConfig(generate_samples=True, doc_len=300, batch_size=1)
    loader = FakeDataLoader(config, seq_len=1000, dp_world_size=1)

    micro_batches = loader.get_batch()

    assert len(micro_batches) == 1
    micro_batch = micro_batches[0]
    expected_lengths = [300, 300, 300, 100]
    assert micro_batch["sequence_lengths"] == expected_lengths
    assert micro_batch["seq_lens"].tolist() == expected_lengths
    assert micro_batch["input_ids"].shape == (1, 1000)
    assert micro_batch["position_ids"].shape == (1, 1000)
    expected_position_ids = torch.cat([torch.arange(n) for n in expected_lengths])
    assert torch.equal(micro_batch["position_ids"][0], expected_position_ids)
    assert micro_batch["input_ids"].max() > 100


def test_doc_len_requires_generate_samples():
    with pytest.raises(ValidationError, match="data.fake.doc_len requires data.fake.generate_samples"):
        FakeDataLoaderConfig(doc_len=4)
