import pytest

from prime_rl.inference.vllm.eplb import local_expert_destinations, remap_expert_weight_loader


@pytest.mark.parametrize(
    "physical_to_logical,global_to_local,expected",
    [
        ([2, 0, 3, 1], [0, 1, -1, -1], [[1], [], [0], []]),
        ([2, 0, 1, 3, 2, 0], [-1, 0, -1, 1, 2, 3], [[1, 5], [], [4], [3]]),
    ],
)
def test_reload_experts_follow_current_placement(physical_to_logical, global_to_local, expected):
    destinations = local_expert_destinations(physical_to_logical, global_to_local, 4)
    assert destinations == expected
    weights = [None] * sum(local >= 0 for local in global_to_local)
    writes = [0] * len(weights)

    def loader(param, loaded_weight, weight_name, shard_id, expert_id, return_success):
        local_id = global_to_local[expert_id]
        assert local_id >= 0
        param[local_id] = loaded_weight
        writes[local_id] += 1
        return True

    reload_weight = remap_expert_weight_loader(loader, destinations)
    for checkpoint_id in range(len(physical_to_logical)):
        logical_id = checkpoint_id % 4
        loaded = reload_weight(weights, logical_id, "weight", "w1", checkpoint_id, return_success=True)
        assert loaded == (checkpoint_id < 4 and bool(destinations[logical_id]))

    assert writes == [1] * len(weights)
    for physical_id, local_id in enumerate(global_to_local):
        if local_id >= 0:
            assert weights[local_id] == physical_to_logical[physical_id]


def test_reload_rejects_unmapped_local_experts():
    with pytest.raises(ValueError, match="invalid logical expert"):
        local_expert_destinations([0, -1], [0, 1], 1)
