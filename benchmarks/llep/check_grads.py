"""Forward and backward exactness of LLEP on Mega MoE against Mega MoE and an fp32 autograd reference.

torchrun --nproc_per_node=8 benchmarks/llep/check_grads.py --tokens-per-rank 4096
"""

import argparse
import os
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent))
from common import clamped_swiglu, make_routing  # noqa: E402

from prime_rl.configs.trainer import LLEPConfig  # noqa: E402
from prime_rl.trainer.models.layers.activations import ClampedSilu  # noqa: E402
from prime_rl.trainer.models.layers.llep_mega_moe import LLEPMegaMoEExpertCompute  # noqa: E402
from prime_rl.trainer.models.layers.mega_moe import MegaMoEExpertCompute, reserve_sms_for_comm  # noqa: E402


def reference(x, scores, topk, gate_up, down, limit, dy, first_expert, group):
    """fp32 naive EP: every rank applies its own experts to all ranks' tokens; returns this rank's
    (y, dx, dscores) and the local experts' (dw1, dw2)."""
    rank = group.rank()
    xs, ss, ks, dys = (_gather(t, group) for t in (x.float(), scores.float(), topk, dy.float()))
    xs.requires_grad_()
    ss.requires_grad_()
    w1 = gate_up.float().requires_grad_()
    w2 = down.float().requires_grad_()
    out = torch.zeros_like(xs)
    for e in range(gate_up.shape[0]):
        token, slot = (ks == first_expert + e).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        h = clamped_swiglu(xs[token] @ w1[e].t(), limit) @ w2[e].t()
        out = out.index_add(0, token, h * ss[token, slot].unsqueeze(1))
    out.backward(dys)
    parts = [out.detach(), xs.grad, ss.grad]
    for t in parts:
        dist.all_reduce(t, group=group)
    n = x.shape[0]
    y, dx, ds = (t[rank * n : (rank + 1) * n] for t in parts)
    return y, dx, ds, w1.grad, w2.grad


def _gather(t, group):
    out = [torch.empty_like(t) for _ in range(group.size())]
    dist.all_gather(out, t.contiguous(), group=group)
    return torch.cat(out)


def rel(a, b):
    err = torch.tensor(((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item(), device="cuda")
    dist.all_reduce(err, op=dist.ReduceOp.MAX)
    return err.item()


def all_equal(a, b):
    same = torch.tensor(float(torch.equal(a, b)), device="cuda")
    dist.all_reduce(same, op=dist.ReduceOp.MIN)
    return bool(same.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-rank", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--limit", type=float, default=10.0)
    parser.add_argument("--scenarios", default="balanced,hot:50:4,hot:95:16,hot:95:1,zipf:1.2")
    parser.add_argument("--repush", action="store_true", help="mark LLEP's slots stale between forward and backward")
    parser.add_argument("--chunk-tokens", type=int, default=None, help="LLEP tokens per rank per kernel call")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("NCCL_MAX_CTAS", "8")
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    group = dist.group.WORLD
    rank, world = dist.get_rank(), dist.get_world_size()
    per_rank = args.num_experts // world

    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    local_gate_up = (
        torch.randn(per_rank, 2 * args.intermediate, args.hidden, device="cuda", generator=gen) * 0.02
    ).bfloat16()
    local_down = (torch.randn(per_rank, args.hidden, args.intermediate, device="cuda", generator=gen) * 0.02).bfloat16()
    gen.manual_seed(99 + rank)
    x0 = torch.randn(args.tokens_per_rank, args.hidden, device="cuda", generator=gen).bfloat16()
    dy = torch.randn(args.tokens_per_rank, args.hidden, device="cuda", generator=gen).bfloat16()

    num_sms = reserve_sms_for_comm(16, 2)
    computes = {
        "mega": MegaMoEExpertCompute(
            args.num_experts, args.top_k, args.hidden, args.intermediate, group, args.tokens_per_rank, num_sms
        ),
        "llep_mega": LLEPMegaMoEExpertCompute(
            args.num_experts,
            args.top_k,
            args.hidden,
            args.intermediate,
            group,
            args.tokens_per_rank,
            num_sms,
            LLEPConfig(chunk_tokens=args.chunk_tokens),
        ),
    }
    for scenario in args.scenarios.split(","):
        scores0, topk = make_routing(scenario, args.tokens_per_rank, args.num_experts, args.top_k, rank, world)
        yr, dxr, dsr, ref_dw1, ref_dw2 = reference(
            x0, scores0, topk, local_gate_up, local_down, args.limit, dy, rank * per_rank, group
        )
        results = {}
        for name, compute in computes.items():
            gate_up = local_gate_up.clone().requires_grad_()
            down = local_down.clone().requires_grad_()
            experts = types.SimpleNamespace(
                gate_up_proj=gate_up, gate_proj=None, up_proj=None, down_proj=down, activation=ClampedSilu(args.limit)
            )
            x = x0.clone().requires_grad_()
            scores = scores0.clone().requires_grad_()
            y = compute(experts, x, scores, topk)
            if args.repush and name == "llep_mega":
                from prime_rl.trainer.models.layers.llep_mega_moe import _STATES

                _STATES[compute.state_key].slots.plan_id = None
            y.backward(dy)
            results[name] = (y.detach(), x.grad, scores.grad, gate_up.grad, down.grad)
        for name, (y, dx, ds, dw1, dw2) in results.items():
            line = (
                f"{scenario:10s} {name:10s} y={rel(y, yr):.4f} dx={rel(dx, dxr):.4f} "
                f"dscores={rel(ds, dsr):.4f} dw1={rel(dw1, ref_dw1):.4f} dw2={rel(dw2, ref_dw2):.4f}"
            )
            if rank == 0:
                print(line, flush=True)
        mega, llep = results["mega"], results["llep_mega"]
        flags = [all_equal(a, b) for a, b in zip(mega, llep)]
        if rank == 0:
            print(
                f"{scenario:10s} llep_mega == mega bitwise: y={flags[0]} dx={flags[1]} dscores={flags[2]} "
                f"dw1={flags[3]} dw2={flags[4]}; llep vs mega dw1={rel(llep[3], mega[3]):.2e} dw2={rel(llep[4], mega[4]):.2e}",
                flush=True,
            )
        else:
            rel(llep[3], mega[3]), rel(llep[4], mega[4])
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
