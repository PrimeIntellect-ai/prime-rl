import pytest

from prime_rl.trainer.pipeline import (
    action_pool_sizes,
    dualpipev_order,
    globally_ordered_actions,
    local_stage_ids,
    one_f_one_b_order,
    pipeline_actions,
    stage_layer_units,
)


def stage_ops(order, kind, stage):
    return [mb for k, s, mb in order if k == kind and s == stage]


@pytest.mark.parametrize("pp,num_micro_batches,warmup_step", [(2, 4, 1), (4, 9, 1), (4, 9, 2), (16, 40, 2)])
def test_one_f_one_b_order(pp, num_micro_batches, warmup_step):
    for rank in range(pp):
        order = one_f_one_b_order(pp, num_micro_batches, rank, warmup_step)
        # Every micro-batch once each way, in order: both ends of an edge post its transfers alike.
        assert stage_ops(order, "F", rank) == list(range(num_micro_batches))
        assert stage_ops(order, "B", rank) == list(range(num_micro_batches))
        in_flight = peak = 0
        for kind, _, _ in order:
            in_flight += 1 if kind == "F" else -1
            peak = max(peak, in_flight)
        assert peak == min(num_micro_batches, warmup_step * (pp - 1 - rank) + 1)


@pytest.mark.parametrize("pp,num_micro_batches", [(2, 4), (4, 8), (4, 11), (8, 33)])
def test_dualpipev_order(pp, num_micro_batches):
    for rank in range(pp):
        order = dualpipev_order(pp, num_micro_batches, rank)
        for stage in (rank, 2 * pp - 1 - rank):
            assert stage_ops(order, "F", stage) == list(range(num_micro_batches))
            assert stage_ops(order, "B", stage) == list(range(num_micro_batches))
        # No rank keeps more than 2 * pp + 1 half-size chunks (pp + 1/2 micro-batches) in flight.
        in_flight = peak = 0
        for kind, _, _ in order:
            in_flight += 1 if kind == "F" else -1
            peak = max(peak, in_flight)
        assert peak == 2 * pp + 1
    with pytest.raises(ValueError):
        dualpipev_order(4, 7, 0)


def test_action_pool_sizes_cover_early_receives():
    owners = {s: s for s in range(4)}
    actions = pipeline_actions(one_f_one_b_order(4, 12, 1), owners, 4, lookahead=1)
    # Stage 1 keeps 3 micro-batches between forward and backward, plus one posted early, plus a spare.
    assert action_pool_sizes(actions)[1] == 5


@pytest.mark.parametrize("schedule,pp", [("1F1B", 4), ("1F1B-2", 6), ("DualPipeV", 4), ("DualPipeV", 6)])
def test_globally_ordered_actions(schedule, pp):
    num_micro_batches = 3 * pp
    stages_per_rank = 2 if schedule == "DualPipeV" else 1
    name = "DualPipeV" if schedule == "DualPipeV" else "Async1F1B"
    owners = {s: r for r in range(pp) for s in local_stage_ids(r, pp, name, stages_per_rank)}
    if schedule == "DualPipeV":
        orders = [dualpipev_order(pp, num_micro_batches, r) for r in range(pp)]
    else:
        orders = [one_f_one_b_order(pp, num_micro_batches, r, 2 if schedule == "1F1B-2" else 1) for r in range(pp)]
    num_stages = pp * stages_per_rank
    per_rank = [globally_ordered_actions(orders, owners, num_stages, r, lookahead=1) for r in range(pp)]

    def transfer(action, op):
        kind, stage, mb = op
        step = 1 if kind == "F" else -1
        src, dst = (stage, stage + step) if action == "send" else (stage - step, stage)
        return (kind, src, dst, mb)

    for r, actions in enumerate(per_rank):
        assert [op for action, op in actions if action == "run"] == orders[r]
        ran = set()
        for action, op in actions:
            if action == "run":
                ran.add(op)
            elif action == "send":
                assert op in ran
            else:
                assert op not in ran
    # Two neighbours post the transfers between them in the same order.
    for r in range(pp - 1):

        def between(actions):
            ts = [transfer(a, op) for a, op in actions if a != "run"]
            return [t for t in ts if {owners[t[1]], owners[t[2]]} == {r, r + 1}]

        assert between(per_rank[r]) == between(per_rank[r + 1])


def test_stage_layer_units():
    assert [stage_layer_units(5, 2, s) for s in range(2)] == [range(0, 6), range(6, 10)]
    # Unit 2i is layer i's attention block, 2i + 1 its MoE block; a stage may hold neither.
    halves = [0.5, 1.5, 0, 2]
    assert [stage_layer_units(4, 4, s, halves) for s in range(4)] == [
        range(0, 1),
        range(1, 4),
        range(4, 4),
        range(4, 8),
    ]
    with pytest.raises(ValueError):
        stage_layer_units(2, 2, 0, [0.7, 1.3])
    with pytest.raises(ValueError):
        stage_layer_units(2, 2, 0, [1, 2])


def test_one_f_one_b_order_mixed_warmup_steps():
    warmups = []
    for rank in range(4):
        order = one_f_one_b_order(4, 10, rank, [1, 2, 2])
        assert stage_ops(order, "F", rank) == stage_ops(order, "B", rank) == list(range(10))
        warmups.append(next(i for i, (kind, _, _) in enumerate(order) if kind == "B"))
    assert warmups == [6, 5, 3, 1]
    with pytest.raises(ValueError):
        one_f_one_b_order(4, 10, 0, [1, 0, 2])
