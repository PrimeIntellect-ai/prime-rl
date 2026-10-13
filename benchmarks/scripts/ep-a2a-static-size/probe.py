"""Compile a checkpointed toy MoE block that dispatches through TorchTokenDispatcher and report graph breaks.

usage: TORCH_LOGS=recompiles,graph_breaks uv run --no-sync torchrun --nproc-per-node=2 probe.py [--fullgraph]
"""

import argparse

import torch
import torch.distributed as dist
import torch.nn as nn

from prime_rl.configs.trainer import ActivationCheckpointConfig
from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper
from prime_rl.trainer.distributed.token_dispatcher import TorchTokenDispatcher

NUM_EXPERTS = 8
TOP_K = 2
HIDDEN = 64


class ToyMoEBlock(nn.Module):
    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.router = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        self.expert_weight = nn.Parameter(torch.randn(HIDDEN, HIDDEN) / HIDDEN**0.5)
        self.dispatcher = TorchTokenDispatcher(NUM_EXPERTS, TOP_K, token_group_alignment=16, group=group)

    def experts(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.expert_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.softmax(self.router(x).float(), dim=-1)
        top_scores, selected_experts = torch.topk(scores, TOP_K, dim=-1)
        return x + self.dispatcher.run(x, top_scores, selected_experts, self.experts, score_before_experts=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--iters", type=int, default=6)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(0)

    block = ToyMoEBlock(dist.group.WORLD).cuda()
    block = get_activation_checkpoint_wrapper(ActivationCheckpointConfig())(block)
    torch._dynamo.config.capture_scalar_outputs = True
    block.compile(fullgraph=args.fullgraph)

    generator = torch.Generator().manual_seed(1000 + rank)
    for step in range(args.iters):
        num_tokens = 96 + 32 * ((step + rank) % 3)
        x = torch.randn(num_tokens, HIDDEN, generator=generator).cuda().requires_grad_()
        out = block(x)
        out.square().mean().backward()
        print(f"rank={rank} step={step} num_tokens={num_tokens} out_sum={out.sum().item():.4f}", flush=True)

    counters = torch._dynamo.utils.counters
    print(f"rank={rank} graph_breaks={dict(counters['graph_break'])}", flush=True)
    print(f"rank={rank} frames={dict(counters['frames'])}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
