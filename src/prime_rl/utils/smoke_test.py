"""Pre-flight check of a node's GPUs before a SLURM job starts: an all-reduce across the
local GPUs. Run with ``torchrun --standalone --nproc-per-node=<gpus> -m prime_rl.utils.smoke_test``."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    x = torch.ones(1 << 20, device=device)
    dist.all_reduce(x)
    assert x.eq(dist.get_world_size()).all().item(), "all-reduce returned a wrong result"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
