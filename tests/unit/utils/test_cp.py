from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prime_rl.utils.cp import assert_cp_style_supports_model, shard_for_cp, shard_position_ids_for_cp


def test_shard_position_ids_for_cp_chunks_3d_mrope_positions_on_sequence_dim():
    position_ids = torch.arange(24).view(3, 1, 8)

    rank0 = shard_position_ids_for_cp(position_ids, cp_rank=0, cp_world_size=2)
    rank1 = shard_position_ids_for_cp(position_ids, cp_rank=1, cp_world_size=2)

    torch.testing.assert_close(rank0, position_ids[:, :, :4])
    torch.testing.assert_close(rank1, position_ids[:, :, 4:])


def test_shard_for_cp_rejects_non_divisible_sequence_dim():
    with pytest.raises(ValueError, match="divisible by cp size"):
        shard_for_cp(torch.arange(5).view(1, 5), cp_rank=0, cp_world_size=2)


def test_ulysses_guard_requires_attention_heads_divisible_by_cp():
    model = nn.Module()
    model.config = SimpleNamespace(num_attention_heads=3, num_key_value_heads=2)

    with pytest.raises(ValueError, match="num_attention_heads"):
        assert_cp_style_supports_model("ulysses", model, cp_world_size=2)


def test_ulysses_guard_requires_kv_heads_divisible_by_cp():
    model = nn.Module()
    model.config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=3)

    with pytest.raises(ValueError, match="num_key_value_heads"):
        assert_cp_style_supports_model("ulysses", model, cp_world_size=2)
