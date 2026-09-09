"""Compare native full-offload updates with and without gradient page reclamation."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from prime_rl.configs.trainer import AdamWConfig, OptimizerInBackwardOffloadConfig
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.mismatch import parameter_change_counts


class ProbeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(256, 256, bias=False)
        self.router = nn.Linear(256, 128, bias=False)

    def forward(self, x):
        return self.router(self.first(x).relu().float())


def run(reclaim: bool, inputs: list[torch.Tensor]):
    torch.manual_seed(42)
    model = ProbeModel().cuda()
    fully_shard(model.first, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
    fully_shard(model.router)
    fully_shard(model)
    named = list(model.named_parameters())
    policy = {
        id(param): (torch.float32, torch.float32) if "router" in name else (torch.bfloat16, torch.bfloat16)
        for name, param in named
    }
    optimizer, manager = setup_optimizer(
        AdamWConfig(lr=1e-3, max_norm=None),
        named,
        ParallelDims(1, 1, 1, 1, 1, 1),
        full_offload_config=OptimizerInBackwardOffloadConfig(release_gradient_pages=reclaim, numa_bind=False),
        model=model,
        full_offload_dtype_policy=policy,
    )
    snapshots = []
    for _ in range(3):
        router_before = model.router.weight.to_local().detach().clone()
        optimizer.zero_grad()
        manager.begin_step(gradient_scale=0.5, overlap_optimizer=True)
        for index, x in enumerate(inputs):
            manager.begin_backward(final_backward=index == len(inputs) - 1)
            model(x).square().sum().backward()
            manager.finish_backward()
        optimizer.step()
        changes, nonfinite = parameter_change_counts(router_before, model.router.weight.to_local()).tolist()
        assert changes > 0 and nonfinite == 0
        snapshots.append(
            [
                value.clone()
                for group in optimizer.param_groups
                for param in group["params"]
                for value in (param.detach(), optimizer.state[param]["exp_avg"], optimizer.state[param]["exp_avg_sq"])
            ]
        )
    manager.close()
    return snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    with tempfile.TemporaryDirectory(prefix="mismatch-offload-") as directory:
        dist.init_process_group("nccl", init_method=f"file://{directory}/store", rank=0, world_size=1)
        torch.manual_seed(7)
        inputs = [torch.randn(17, 256, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        retained, reclaimed = run(False, inputs), run(True, inputs)
        for old_step, new_step in zip(retained, reclaimed):
            for old, new in zip(old_step, new_step):
                assert old.dtype == new.dtype == torch.float32
                assert torch.isfinite(old).all() and torch.isfinite(new).all()
                assert torch.equal(old.view(torch.int32), new.view(torch.int32))
        result = {
            "steps": 3,
            "microbatches_per_step": 2,
            "masters_and_adam_moments_bitwise_equal": True,
            "router_update_probe_detected_all_steps": True,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
