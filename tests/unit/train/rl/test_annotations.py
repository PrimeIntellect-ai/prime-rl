import os
import socket
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from prime_rl import monitors
from prime_rl.trainer.rl.annotations import AnnotationWriter, gather_streams


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _spawn(fn, world_size: int, *args) -> None:
    mp.spawn(_init_and_run, args=(world_size, _free_port(), fn, args), nprocs=world_size, join=True)


def _init_and_run(rank: int, world_size: int, port: int, fn, args) -> None:
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _streams(rank: int, lengths: list[int]) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(rank)
    streams = []
    for index, length in enumerate(lengths):
        logprobs = -rng.random(length, dtype=np.float32)
        entropies = rng.random(length, dtype=np.float32)
        logprobs[::3] = np.nan
        entropies[0] = np.nan
        streams.append((f"trace-{rank}-{index}", index % 3, logprobs, entropies))
    return streams


def _assert_same_streams(actual, expected) -> None:
    assert [(t, b, len(lp)) for t, b, lp, _ in actual] == [(t, b, len(lp)) for t, b, lp, _ in expected]
    for (_, _, logprobs, entropies), (_, _, want_logprobs, want_entropies) in zip(actual, expected):
        assert logprobs.dtype == entropies.dtype == np.float32
        np.testing.assert_array_equal(logprobs, want_logprobs)
        np.testing.assert_array_equal(entropies, want_entropies)


def _gather_streams_worker(rank: int, world_size: int, lengths_by_rank: list[list[int]]) -> None:
    gathered = gather_streams(_streams(rank, lengths_by_rank[rank]), None)
    if rank != 0:
        assert gathered is None
        return
    expected = [stream for member in range(world_size) for stream in _streams(member, lengths_by_rank[member])]
    _assert_same_streams(gathered, expected)


@pytest.mark.parametrize(
    "lengths_by_rank",
    [
        [[5, 1, 9], [], [7], [3, 3]],  # uneven members, one with nothing
        [[], [], [], []],  # nothing to gather at all
        [[], [4], [], []],  # only a non-receiving member has data
    ],
)
def test_gather_streams_returns_every_member_in_rank_order(lengths_by_rank):
    _spawn(_gather_streams_worker, len(lengths_by_rank), lengths_by_rank)


def _micro_batch(rank: int, step: int) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(100 * rank + step)
    sequence_lengths = [6, 4, 7]
    total = sum(sequence_lengths)
    loss_mask = torch.rand(total, generator=generator) > 0.3
    loss_mask[sum(sequence_lengths[:1]) : sum(sequence_lengths[:2])] = False  # a sample with nothing to train
    micro_batch = {
        "trace_ids": [f"r{rank}-s{step}-a", f"r{rank}-s{step}-b", "" if step else f"r{rank}-s{step}-c"],
        "branch_indices": [0, 1, 2],
        "sequence_lengths": sequence_lengths,
        "loss_mask": loss_mask,
        "padding": 2,
    }
    model_output = {
        "logprobs": -torch.rand(total, generator=generator),
        "entropy": torch.rand(total, generator=generator),
    }
    return micro_batch, model_output


def _fake_parallel_dims(cp: int, rank: int, leaders_group: dist.ProcessGroup | None):
    cp_mesh = SimpleNamespace(get_local_rank=lambda: rank % cp)
    dp_mesh = SimpleNamespace(get_group=lambda: leaders_group)
    return SimpleNamespace(
        cp_enabled=cp > 1,
        pp_enabled=False,
        world_mesh={"cp": cp_mesh},
        get_mesh=lambda name: {"dp": dp_mesh}[name],
    )


def _exported_streams(leader: int, cp: int, world: SimpleNamespace):
    """What one CP leader's writer holds after the test's micro batches."""
    writer = AnnotationWriter(_fake_parallel_dims(cp, leader, None), world, float_decimals=3)
    for step in range(2):
        writer.export(*_micro_batch(leader, step))
    return writer._pending


def _flush_worker(rank: int, world_size: int, cp: int) -> None:
    leaders = list(range(0, world_size, cp))
    leaders_group = dist.new_group(leaders)  # collective: every rank creates it
    world = SimpleNamespace(rank=rank, world_size=world_size)
    logged: list[list[dict[str, Any]]] = []

    async def capture(records):
        logged.append(records)

    monitors.log_annotations = capture

    writer = AnnotationWriter(_fake_parallel_dims(cp, rank, leaders_group), world, float_decimals=3)
    # CP peers see the same micro batches as their leader.
    for step in range(2):
        writer.export(*_micro_batch(rank - rank % cp, step))
    # Non-leaders take no part in the gather: they return before any collective.
    writer.flush()
    if rank != 0:
        assert logged == []
        return

    (records,) = logged
    expected = [stream for leader in leaders for stream in _exported_streams(leader, cp, world)]
    assert len(records) == len(expected) > 0
    for record, (trace_id, branch_index, logprobs, entropies) in zip(records, expected):
        assert record["trace_id"] == trace_id
        (branch,) = record["branches"]
        assert branch["index"] == branch_index
        np.testing.assert_array_equal(branch["trainer_logprobs"], logprobs)
        np.testing.assert_array_equal(branch["entropies"], entropies)


@pytest.mark.parametrize("world_size,cp", [(4, 2), (4, 1), (6, 3)])
def test_flush_logs_every_cp_leaders_streams_once_on_rank_0(world_size, cp):
    _spawn(_flush_worker, world_size, cp)
