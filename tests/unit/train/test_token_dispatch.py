"""Expert-parallel token dispatch on CPU (gloo)."""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from prime_rl.trainer.distributed.token_dispatcher import TorchTokenDispatcher

NUM_EXPERTS, TOP_K, DIM = 8, 3, 4


def _worker(rank: int, world_size: int, store_path: str, result_path: str) -> None:
    dist.init_process_group("gloo", store=dist.FileStore(store_path, world_size), rank=rank, world_size=world_size)
    results = []
    # Spread routing, then an empty rank 1, so that some experts get no tokens.
    for num_tokens in (5 + 3 * rank, 0 if rank == 1 else 2):
        for alignment in (1, 4):
            generator = torch.Generator().manual_seed(rank)
            x = torch.randn(num_tokens, DIM, generator=generator)
            scores = torch.rand(num_tokens, TOP_K, generator=generator)
            selected = torch.rand(num_tokens, NUM_EXPERTS, generator=generator).argsort(dim=1)[:, :TOP_K]

            def experts(routed: torch.Tensor, counts: torch.Tensor, *host_counts: torch.Tensor) -> torch.Tensor:
                results.append((host_counts, counts))
                return routed

            dispatcher = TorchTokenDispatcher(NUM_EXPERTS, TOP_K, alignment, dist.group.WORLD)
            dispatcher.run(x, scores, selected, experts, score_before_experts=False)
    torch.save(results, f"{result_path}.{rank}")
    dist.destroy_process_group()


def test_dispatch_passes_the_counts_per_expert_on_the_host():
    """The experts get the counts per local expert, as laid out in their input, from the dispatch's one host copy."""
    world_size = 2
    with tempfile.TemporaryDirectory() as tmp:
        store_path, result_path = os.path.join(tmp, "store"), os.path.join(tmp, "result")
        mp.spawn(_worker, args=(world_size, store_path, result_path), nprocs=world_size)
        for rank in range(world_size):
            for host_counts, counts in torch.load(f"{result_path}.{rank}"):
                assert len(host_counts) == 1 and host_counts[0].device.type == "cpu"
                assert host_counts[0].tolist() == counts.tolist()
