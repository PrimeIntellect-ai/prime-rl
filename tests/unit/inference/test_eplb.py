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


def _eplb_state(physical_to_logical, expert_map, num_routed_experts=4):
    from types import SimpleNamespace

    import torch

    layer = SimpleNamespace(routed_experts=SimpleNamespace(_expert_map=torch.tensor(expert_map)))
    state = SimpleNamespace(
        physical_to_logical_map=torch.tensor([physical_to_logical]),
        model=SimpleNamespace(moe_layers=[layer], num_routed_experts=num_routed_experts),
    )
    return SimpleNamespace(model_states={"m": state}), layer


def test_current_expert_placement_maps_local_slots():
    from types import SimpleNamespace

    from prime_rl.inference.vllm.eplb import current_expert_placement

    # 4 logical experts over 6 physical slots; this rank holds slots 0-2. Expert 3 sits in slot 0 now, expert 1 twice.
    eplb_state, layer = _eplb_state([3, 1, 1, 0, 2, 0], [0, 1, 2, -1, -1, -1])
    fingerprint, layers = current_expert_placement(eplb_state, SimpleNamespace(compute_hash=lambda: "m"))
    assert fingerprint == ((3, 1, 1, 0, 2, 0),)
    ((experts, destinations),) = layers
    assert experts is layer.routed_experts
    assert destinations == [[], [1, 2], [], [0]]


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


def _policy():
    from vllm.distributed.eplb.policy.default import DefaultEplbPolicy

    return DefaultEplbPolicy


def test_trainer_placement_follows_the_counts_alone():
    import torch

    from prime_rl.inference.vllm.eplb import trainer_placement

    # 8 experts over 4 ranks with 4 replica slots; expert 5 takes most tokens in layer 0, expert 2 in layer 1
    load = torch.tensor([[10, 10, 10, 10, 10, 400, 10, 10], [10, 10, 300, 10, 10, 10, 10, 10]])
    placement = trainer_placement(_policy(), load, 12, 1, 1, 4)
    assert placement.shape == (2, 12)
    for layer, hot in ((0, 5), (1, 2)):
        replicas = torch.bincount(placement[layer], minlength=8)
        assert (replicas >= 1).all() and replicas[hot] == replicas.max() > 1
    # Same counts, same placement: every rank of an EP group, and a restarted engine, agree
    assert torch.equal(placement, trainer_placement(_policy(), load.clone(), 12, 1, 1, 4))
    # A node count that does not divide the ranks falls back to the global policy instead of failing
    assert trainer_placement(_policy(), load, 12, 1, 3, 4).shape == (2, 12)


def test_trainer_placement_installs_maps_without_moving_weights(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch
    import vllm.distributed.parallel_state as parallel_state

    from prime_rl.inference.vllm.eplb import EXPERT_LOAD_FILE, install_trainer_placement

    initial = torch.arange(12) % 8
    weights = torch.randn(12, 4)
    before = weights.clone()
    state = SimpleNamespace(
        physical_to_logical_map=initial.unsqueeze(0).clone(),
        logical_to_physical_map=torch.full((1, 8, 12), -1),
        logical_replica_count=torch.zeros(1, 8, dtype=torch.long),
        model=SimpleNamespace(
            num_moe_layers=1, num_logical_experts=8, num_physical_experts=12, num_expert_groups=1, weights=weights
        ),
    )
    eplb_state = SimpleNamespace(model_states={"m": state}, policy=_policy())
    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: SimpleNamespace(world_size=4))
    # vLLM pins host copies when CUDA is present; this host may have none
    import vllm.distributed.eplb.eplb_state as eplb_module

    monkeypatch.setattr(eplb_module, "PIN_MEMORY", False)
    config = SimpleNamespace(compute_hash=lambda: "m")

    # No counts published with this update: nothing changes
    assert install_trainer_placement(eplb_state, config, str(tmp_path), lambda: 1) is None
    torch.save(torch.tensor([[10, 10, 10, 10, 10, 400, 10, 10]]), tmp_path / EXPERT_LOAD_FILE)
    moved = install_trainer_placement(eplb_state, config, str(tmp_path), lambda: 1)
    new = state.physical_to_logical_map[0]
    assert moved == int((new != initial).sum()) > 0
    # The routing maps follow the new placement; the expert weights are left to the update to write
    replicas = torch.bincount(new, minlength=8)
    assert torch.equal(state.logical_replica_count[0], replicas)
    for logical in range(8):
        slots = state.logical_to_physical_map[0, logical]
        assert sorted(slots[slots >= 0].tolist()) == (new == logical).nonzero().flatten().tolist()
    assert torch.equal(weights, before)

    torch.save(torch.zeros(2, 8, dtype=torch.long), tmp_path / EXPERT_LOAD_FILE)
    import pytest

    with pytest.raises(ValueError, match="shape"):
        install_trainer_placement(eplb_state, config, str(tmp_path), lambda: 1)


def test_engine_weights_count_what_the_engines_computed():
    from types import SimpleNamespace as NS

    import numpy as np

    from prime_rl.orchestrator.trajectories import engine_weights

    # prompt (4) -> reply (2) -> tool output (3) -> reply (1)
    nodes = [NS(token_ids=[0] * n) for n in (4, 2, 3, 1)]

    def call(node, cached):
        return NS(node=node, usage=NS(cached_input_tokens=cached))

    trace = NS(nodes=nodes)
    # Turn 1 prefills the prompt and generates; turn 2 reads the first 6 tokens from the cache
    branch = NS(nodes=nodes, calls=[call(1, 0), call(3, 6)])
    weights = engine_weights(trace, branch, set())
    np.testing.assert_array_equal(weights, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # Turn 2 found only the first 2 cached (an eviction): tokens 2-5 are prefilled a second time
    branch.calls = [call(1, 0), call(3, 2)]
    np.testing.assert_array_equal(engine_weights(trace, branch, set()), [1, 1, 2, 2, 2, 2, 1, 1, 1, 1])
    # A call an earlier branch already counted counts there only
    np.testing.assert_array_equal(engine_weights(trace, branch, {1}), [0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    # No reported cache reads: unknown, so every token counts once (None)
    branch.calls = [NS(node=1, usage=NS(cached_input_tokens=None))]
    assert engine_weights(trace, branch, set()) is None


def test_moe_engine_load_weighs_each_routed_token():
    import torch

    from prime_rl.trainer.models.layers.moe import record_moe_engine_load

    load = torch.zeros(4)
    # Three tokens, top-2: the second token was served from the cache (weight 0), the third prefilled twice
    record_moe_engine_load(load, torch.tensor([[0, 1], [1, 2], [3, 1]]), torch.tensor([1.0, 0.0, 2.0]))
    assert load.tolist() == [1.0, 3.0, 0.0, 2.0]
