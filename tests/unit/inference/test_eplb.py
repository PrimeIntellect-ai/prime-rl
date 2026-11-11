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


def _eplb_state(physical_to_logical, expert_map, num_routed_experts=4):
    from types import SimpleNamespace

    import torch

    layer = SimpleNamespace(routed_experts=SimpleNamespace(_expert_map=torch.tensor(expert_map)))
    state = SimpleNamespace(
        physical_to_logical_map=torch.tensor([physical_to_logical]),
        model=SimpleNamespace(moe_layers=[layer], num_routed_experts=num_routed_experts),
    )
    drained = []
    return SimpleNamespace(drain_async=lambda: drained.append(True), model_states={"m": state}), layer, drained


def test_current_expert_placement_drains_and_maps_local_slots():
    from types import SimpleNamespace

    from prime_rl.inference.vllm.eplb import current_expert_placement, moved_slots

    # 4 logical experts over 6 physical slots; this rank holds slots 0-2. Expert 3 sits in slot 0 now, expert 1 twice.
    eplb_state, layer, drained = _eplb_state([3, 1, 1, 0, 2, 0], [0, 1, 2, -1, -1, -1])
    fingerprint, layers = current_expert_placement(eplb_state, SimpleNamespace(compute_hash=lambda: "m"))
    assert drained and fingerprint == ((3, 1, 1, 0, 2, 0),)
    ((experts, destinations),) = layers
    assert experts is layer.routed_experts
    assert destinations == [[], [1, 2], [], [0]]
    assert moved_slots(fingerprint, 4) == 5


def test_nixl_plan_is_traced_again_when_eplb_moves_experts():
    from types import SimpleNamespace

    from prime_rl.inference.vllm.worker.nixl import NIXLWeightUpdateWorker, WeightTransferPlan

    placements = iter([((0, 1, 2, 3),), ((0, 1, 2, 3),), ((3, 1, 2, 0),)])
    traced, released = [], []
    worker = object.__new__(NIXLWeightUpdateWorker)
    worker.weight_transfer_plan = None
    worker.weight_transfer_timeout = 1
    worker.device = "cpu"
    worker.current_expert_placement = lambda: (next(placements), [])
    worker.model_express = SimpleNamespace(
        rank=0, wait_for=lambda *a, **k: ["trainer"], fetch=lambda ref: SimpleNamespace(nixl_metadata=b"")
    )
    worker.trace_weight_loads = lambda table, destinations: traced.append(table) or []
    worker.build_transfer_plan = lambda table, copies: WeightTransferPlan({}, 1, [], [SimpleNamespace(pulls=[])] * 2)
    worker.release_transfer_plan = lambda plan: released.append(plan)
    import prime_rl.inference.vllm.worker.nixl as nixl_worker

    decode = nixl_worker.TrainerTensorTable.decode
    nixl_worker.TrainerTensorTable.decode = staticmethod(lambda metadata: "table")
    sync = nixl_worker.torch.cuda.synchronize
    nixl_worker.torch.cuda.synchronize = lambda device=None: None
    try:
        first = worker.initialize_transfer()
        worker.group_generations = [5, 5]  # publications pulled so far
        assert worker.initialize_transfer() is first and len(traced) == 1  # same placement: same plan
        second = worker.initialize_transfer()  # EPLB moved experts
    finally:
        nixl_worker.TrainerTensorTable.decode = decode
        nixl_worker.torch.cuda.synchronize = sync
    assert second is not first and released == [first] and len(traced) == 2
    assert second.placement == ((3, 1, 2, 0),)
    assert worker.group_generations == [5, 5]
