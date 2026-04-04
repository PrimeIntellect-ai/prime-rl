import torch
import torch.distributed as dist

from prime_rl.trainer.rl.stats import aggregate_dp_count, get_local_batch_stats


def test_get_local_batch_stats_uses_actual_packed_tokens():
    micro_batches = [
        {
            "input_ids": torch.ones((1, 3), dtype=torch.long),
            "loss_mask": torch.tensor([[True, False, True]]),
            "sample_count": 2,
        },
        {
            "input_ids": torch.ones((1, 5), dtype=torch.long),
            "loss_mask": torch.tensor([[True, True, True, False, False]]),
            "sample_count": 1,
        },
    ]

    stats = get_local_batch_stats(micro_batches)

    assert stats.num_micro_batches == 2
    assert stats.num_tokens == 8
    assert stats.num_loss_tokens == 5
    assert stats.num_samples == 3
    assert stats.max_micro_batch_tokens == 5


def test_aggregate_dp_count_skips_collective_for_single_rank(monkeypatch):
    called = False

    def fake_all_reduce(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    num_tokens = aggregate_dp_count(
        17,
        dp_world_size=1,
        dp_group=None,
        device=torch.device("cpu"),
    )

    assert num_tokens == 17
    assert not called


def test_aggregate_dp_count_sums_across_dp_group(monkeypatch):
    calls = []

    def fake_all_reduce(tensor, op, group):
        calls.append((int(tensor.item()), op, group))
        tensor.add_(13)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    num_tokens = aggregate_dp_count(
        17,
        dp_world_size=2,
        dp_group="dp-group",
        device=torch.device("cpu"),
    )

    assert num_tokens == 30
    assert calls == [(17, dist.ReduceOp.SUM, "dp-group")]
