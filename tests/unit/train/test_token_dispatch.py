"""Expert-parallel token dispatchers on CPU (gloo): one copy per rank vs one copy per expert."""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from prime_rl.trainer.distributed.token_dispatcher import PerRankTorchTokenDispatcher, TorchTokenDispatcher

NUM_EXPERTS, TOP_K, DIM = 16, 3, 6


def _scaled_square_experts(first_expert: int):
    """Expert e squares its input and scales it by the global expert number + 1 (rows after the last group give 0)."""

    def experts(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        expert = torch.searchsorted(torch.cumsum(counts.long(), dim=0), torch.arange(x.shape[0]), right=True)
        scale = torch.where(expert < len(counts), first_expert + 1 + expert, 0).to(x.dtype)
        return x.square() * scale.unsqueeze(1)

    return experts


def _routing(rank: int, num_tokens: int, local_experts_only: bool):
    """Random tokens, scores and distinct experts; with local_experts_only every token stays on its rank's experts."""
    generator = torch.Generator().manual_seed(rank)
    x = torch.randn(num_tokens, DIM, generator=generator)
    scores = torch.rand(num_tokens, TOP_K, generator=generator)
    choices, offset = NUM_EXPERTS, 0
    if local_experts_only:
        choices = NUM_EXPERTS // dist.get_world_size()
        offset = rank * choices
    selected = [torch.randperm(choices, generator=generator)[:TOP_K] + offset for _ in range(num_tokens)]
    selected = torch.stack(selected) if selected else torch.zeros(0, TOP_K, dtype=torch.int64)
    return x, scores, selected


def _run(dispatcher, x, scores, selected, score_before_experts, first_expert, grad_seed, compiled=False):
    x = x.clone().requires_grad_()
    scores = scores.clone().requires_grad_()
    experts = _scaled_square_experts(first_expert)

    def forward(x: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return dispatcher.run(x, scores, selected, experts, score_before_experts=score_before_experts)

    if compiled:  # model.compile.fullgraph compiles blocks with fullgraph=True: the dispatch must not break the graph
        torch._dynamo.reset()
        forward = torch.compile(forward, fullgraph=True, backend="aot_eager")
    output = forward(x, scores)
    weight = torch.randn(output.shape, generator=torch.Generator().manual_seed(grad_seed))
    (output * weight).sum().backward()
    return output.detach(), x.grad, scores.grad


def _worker(rank: int, world_size: int, store_path: str, result_path: str) -> None:
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    torch._dynamo.config.capture_scalar_outputs = True  # as apply_compile sets it
    group = dist.group.WORLD
    first_expert = rank * (NUM_EXPERTS // world_size)
    results = []
    for score_before_experts in (False, True):
        for alignment in (1, 4):
            routings = ((5 + 3 * rank, False), (0 if rank == 1 else 4, False), (6, True))
            for index, (num_tokens, local_only) in enumerate(routings):
                x, scores, selected = _routing(rank, num_tokens, local_only)
                per_expert = TorchTokenDispatcher(NUM_EXPERTS, TOP_K, alignment, group)
                args = (x, scores, selected, score_before_experts, first_expert, rank + 10)
                expected = _run(per_expert, *args)
                # Compiling takes most of the time: on two ranks, the compiled dispatch runs every routing (an empty
                # rank, local experts only) once, and both scoring orders, with one alignment.
                compile_too = world_size == 2 and alignment == 4 and (not score_before_experts or index == 0)
                per_rank = PerRankTorchTokenDispatcher(NUM_EXPERTS, TOP_K, alignment, group)
                for compiled in (False, True) if compile_too else (False,):
                    results.append((_run(per_rank, *args, compiled), expected))
    torch.save(results, f"{result_path}.{rank}")
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_per_rank_dispatch_matches_per_expert_dispatch(world_size):
    with tempfile.TemporaryDirectory() as tmp:
        store_path, result_path = os.path.join(tmp, "store"), os.path.join(tmp, "result")
        mp.spawn(_worker, args=(world_size, store_path, result_path), nprocs=world_size)
        for rank in range(world_size):
            for actual, expected in torch.load(f"{result_path}.{rank}"):
                for a, e in zip(actual, expected, strict=True):
                    torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-6)
