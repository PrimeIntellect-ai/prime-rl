"""Hosted trainer-only diagnostic: validate and time inter-node collectives."""

from datetime import datetime, timedelta, timezone
import json
import os
import socket
import time

import torch
import torch.distributed as dist


def run():
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET,ENV"
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5), device_id=torch.device("cuda", local_rank))
    world = dist.get_world_size()

    def report(event, **fields):
        print(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                          "event": event, "rank": rank, "host": socket.gethostname(), **fields}), flush=True)

    report("collectives_start", world_size=world, torch=torch.__version__,
           cuda=torch.version.cuda, nccl=list(torch.cuda.nccl.version()),
           gpu=torch.cuda.get_device_name())
    assert world == 16, f"Expected two 8-GPU nodes, got {world} ranks"
    for shard_mib in (16, 64, 256):
        # BF16 parameter all-gather and FP32 gradient reduce-scatter, as in this trainer.
        gather_count = shard_mib * 1024 * 1024 // 2
        reduce_count = shard_mib * 1024 * 1024 // 4
        source = torch.full((gather_count,), rank + 1, device="cuda", dtype=torch.bfloat16)
        gathered = torch.empty(world * gather_count, device="cuda", dtype=torch.bfloat16)
        gradients = torch.full((world * reduce_count,), rank + 1, device="cuda", dtype=torch.float32)
        reduced = torch.empty(reduce_count, device="cuda", dtype=torch.float32)
        expected = world * (world + 1) // 2
        timings = []
        for iteration in range(30):
            dist.barrier()
            start = time.perf_counter()
            dist.all_gather_into_tensor(gathered, source)
            dist.reduce_scatter_tensor(reduced, gradients)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if iteration >= 5:
                timings.append(elapsed)
            if iteration in (0, 29):
                expected_gather = torch.arange(1, world + 1, device="cuda", dtype=torch.bfloat16)[:, None]
                assert torch.all(gathered.view(world, -1) == expected_gather).item(), "All-gather corruption"
                assert torch.all(reduced == expected).item(), "Reduce-scatter corruption"
        report("collectives_size_passed", shard_mib=shard_mib, iterations=30,
               mean_pair_seconds=sum(timings) / len(timings),
               peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3)
        del source, gathered, gradients, reduced
    dist.barrier()
    report("collectives_passed")
    dist.destroy_process_group()
