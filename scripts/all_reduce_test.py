# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "setuptools",
# ]
# ///

import os
import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.distributed as dist
import argparse  # Import argparse for CLI arguments


import torch.utils.benchmark as benchmark


# Function to initialize the distributed process group
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def main(model_size, n_iters):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    nnodes = world_size // local_world_size

    mat = torch.rand(1, model_size).cuda()


    if local_rank == 0:
        print(f"\n ======== Benchmark all reduce between {world_size} gpus over {nnodes} nodes =========\n")

    t0 = benchmark.Timer(stmt="dist.all_reduce(mat)", setup="from __main__ import dist", globals={"mat": mat})
    measured_time = t0.timeit(n_iters).mean

    bandwidth = model_size * 4 / 1e9 / measured_time

    if local_rank == 0:
        print(f"Average time per iteration: {measured_time:.2f} seconds, Average bandwidth: {bandwidth:.2f} GB/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run distributed all-reduce test.")
    parser.add_argument("--model_size", type=int, default=int(1e9), help="Model size for the test matrix.")
    parser.add_argument("--n_iters", type=int, default=100, help="Number of iterations to time.")

    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    ddp_setup()

    main(args.model_size, args.n_iters)  # Pass model_size, n_iter from CLI
    destroy_process_group()