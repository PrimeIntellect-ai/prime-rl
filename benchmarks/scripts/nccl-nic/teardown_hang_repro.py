"""Exchange tensors between every pair of ranks with one P2P pattern, then time ``destroy_process_group``.

With NCCL >= 2.31, torch releases up to 2.15 can hang in ``destroy_process_group`` when ranks hold
different sets of communicators in one process group (pytorch/pytorch#201536). The patterns:

- ``lazy-p2p``: no ``device_id``; unbatched isend/irecv, one two-rank communicator per peer.
- ``eager-p2p``: ``device_id`` set, as the prime-rl trainer does; unbatched isend/irecv.
- ``batched-p2p``: no ``device_id``; ``batch_isend_irecv``.
- ``eager-subgroup-p2p``: ``device_id`` set; unbatched isend/irecv on a ``new_group`` over all ranks.

Usage:
    torchrun --standalone --nproc-per-node 8 benchmarks/scripts/nccl-nic/teardown_hang_repro.py lazy-p2p
"""

import argparse
import itertools
import os
import time

import torch
import torch.distributed as dist

PATTERNS = ("lazy-p2p", "eager-p2p", "batched-p2p", "eager-subgroup-p2p")


def exchange(peer: int, send_first: bool, group: dist.ProcessGroup | None, batched: bool) -> None:
    sent = torch.full((1 << 20,), float(dist.get_rank()), device="cuda")
    received = torch.empty_like(sent)
    if batched:
        ops = [dist.P2POp(dist.isend, sent, peer, group), dist.P2POp(dist.irecv, received, peer, group)]
        for work in dist.batch_isend_irecv(ops):
            work.wait()
    elif send_first:
        dist.isend(sent, peer, group).wait()
        dist.irecv(received, peer, group).wait()
    else:
        dist.irecv(received, peer, group).wait()
        dist.isend(sent, peer, group).wait()
    assert received[0].item() == peer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pattern", choices=PATTERNS)
    pattern = parser.parse_args().pattern

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    eager = pattern.startswith("eager")
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank) if eager else None)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    group = dist.new_group(list(range(world_size))) if pattern == "eager-subgroup-p2p" else None

    for low, high in itertools.combinations(range(world_size), 2):
        if rank in (low, high):
            exchange(high if rank == low else low, rank == low, group, pattern == "batched-p2p")

    total = torch.ones(1, device="cuda")
    dist.all_reduce(total)
    torch.cuda.synchronize()
    assert total.item() == world_size

    print(f"rank {rank}: {pattern} entering destroy_process_group", flush=True)
    start = time.perf_counter()
    dist.destroy_process_group()
    print(f"rank {rank}: {pattern} destroy_process_group took {time.perf_counter() - start:.2f} s", flush=True)


if __name__ == "__main__":
    main()
