import numpy as np
from transformers import Glm4MoeConfig, NemotronHConfig, Qwen3MoeConfig

from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from prime_rl.transports.payload import PayloadSegment, clip_segments, id_bits, ragged_bytes, read_ragged, routed_layers


def test_packed_mask_ids_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    bits = id_bits(151936)
    assert bits == 18
    counts = rng.integers(0, 6, size=50)
    values = rng.integers(0, 151936, size=int(counts.sum()))
    path = tmp_path / "payload.bin"
    path.write_bytes(b"\0" * 13 + ragged_bytes(counts, values, 13, bits))
    whole = PayloadSegment("sampling_mask", str(path), 13, 100, 50, "uint32", [])
    segments = clip_segments([whole], 100, 117) + clip_segments([whole], 117, 150)
    expected = np.full((50, counts.max()), -1)
    for row, (start, count) in enumerate(zip(np.cumsum(counts) - counts, counts)):
        expected[row, :count] = values[start : start + count]
    np.testing.assert_array_equal(read_ragged(segments, "sampling_mask", 100, 150, -1, bits), expected)
    np.testing.assert_array_equal(read_ragged(segments, "sampling_mask", 111, 131, -1, bits), expected[11:31])


def test_routed_layers():
    assert routed_layers(Qwen3MoeConfig(num_hidden_layers=4)) == [0, 1, 2, 3]
    assert routed_layers(Glm4MoeConfig(num_hidden_layers=4, first_k_dense_replace=1)) == [1, 2, 3]
    assert routed_layers(DeepseekV4Config(num_hidden_layers=6, num_hash_layers=3)) == [3, 4, 5]
    assert routed_layers(NemotronHConfig(hybrid_override_pattern="MEM*E")) == [1, 4]
