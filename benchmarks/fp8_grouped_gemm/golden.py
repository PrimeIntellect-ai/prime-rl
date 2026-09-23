"""Capture or check byte-level goldens of the DeepGEMM FP8 grouped GEMM: op outputs and the weight casts.

Inputs are regenerated from fixed seeds and their digest is stored, so a check refuses to compare against
goldens drawn from different inputs. Tokens are laid out at the op's own dispatcher alignment, and only
the real token rows, the weight gradient and the weight casts are compared, none of which depend on that
alignment. So goldens captured before a change of alignment still apply after it.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import deep_gemm
import torch

from prime_rl.trainer.distributed.token_dispatcher import permute_for_grouped_gemm
from prime_rl.trainer.models.kernels.fp8_utils import (
    GROUP_ALIGNMENT,
    grouped_per_block_cast_to_fp8_triton,
    ue8m0_for_device,
)
from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm
from prime_rl.trainer.models.layers.grouped_gemm import DeepGemmFP8GroupedGemm


def _ragged(num_experts: int, mean: int, hot: int, head: list[int], tail: list[int]) -> list[int]:
    """`head`, one expert at `hot` rows, then experts sharing the rest of `num_experts * mean` evenly, then `tail`."""
    rest = num_experts - len(head) - len(tail) - 1
    remaining = num_experts * mean - hot - sum(head) - sum(tail)
    spread = [remaining // rest + (1 if i < remaining % rest else 0) for i in range(rest)]
    return [*head, hot, *spread, *tail]


@dataclass(frozen=True)
class Case:
    name: str
    counts: list[int]
    k: int
    n: int
    dispatch: str
    seed: int


CASES = [
    Case("gate_up-balanced-1536", [1536] * 32, 4096, 4096, "dispatcher", 1),
    Case("down-ragged-512", _ragged(32, 512, 7 * 512, [0, 1, 127, 128, 129, 5], [0]), 2048, 4096, "dispatcher", 2),
    Case("gate_up-ragged-1536", _ragged(32, 1536, 6 * 1536, [0, 3, 64], []), 4096, 4096, "dispatcher", 3),
    Case("small-sub_alignment", [1, 127, 5, 64, 0, 33, 100, 7], 512, 256, "dispatcher", 4),
    Case("small-manual-empty", [0, 896, 3, 60, 1, 40, 24, 0], 512, 256, "manual", 5),
]
RAW_TAIL_ROWS = 128
WEIGHT_STD = 0.02


def _digest(t: torch.Tensor) -> dict:
    t = t.detach().contiguous()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sha256": hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
        "sum": t.float().sum(dtype=torch.float64).item(),
        "absmax": t.float().abs().max().item() if t.numel() else 0.0,
    }


def _inputs(case: Case):
    """Tokens laid out at the op's alignment: "dispatcher" via `permute_for_grouped_gemm`, "manual" by hand,
    which keeps an empty expert a truly empty group."""
    alignment = DeepGemmFP8GroupedGemm().token_group_alignment
    torch.manual_seed(case.seed)
    with torch.device("cuda"):
        tokens = torch.randn(sum(case.counts), case.k, dtype=torch.bfloat16)
        weight = (torch.randn(len(case.counts), case.k, case.n) * WEIGHT_STD).to(torch.bfloat16)
        real_probe = torch.randn(sum(case.counts), case.n, dtype=torch.bfloat16)
    if case.dispatch == "manual":
        padded_counts = [-(-count // alignment) * alignment for count in case.counts]
        x = tokens.new_zeros(sum(padded_counts) + RAW_TAIL_ROWS, case.k)
        real_rows = torch.zeros(x.shape[0], dtype=torch.bool, device="cuda")
        src = dst = 0
        for count, padded_count in zip(case.counts, padded_counts):
            x[dst : dst + count] = tokens[src : src + count]
            real_rows[dst : dst + count] = True
            src += count
            dst += padded_count
        offs = torch.tensor(padded_counts, device="cuda").cumsum(0).to(torch.int32)
    else:
        x, padded_counts, state = permute_for_grouped_gemm(
            tokens,
            torch.tensor(case.counts, dtype=torch.int64, device="cuda"),
            experts_per_rank=len(case.counts),
            num_ranks=1,
            alignment=alignment,
        )
        offs = torch.cumsum(padded_counts, dim=0, dtype=torch.int32)
        real_rows = state.permuted_indices != -1
    assert torch.equal(x[real_rows], tokens)
    probe = torch.zeros(x.shape[0], case.n, device="cuda", dtype=torch.bfloat16)
    probe[real_rows] = real_probe
    return tokens, weight, real_probe, x, offs, probe, real_rows


def _record(case: Case) -> dict:
    tokens, weight, real_probe, x, offs, probe, real_rows = _inputs(case)
    x_leaf = x.clone().requires_grad_(True)
    weight_leaf = weight.clone().requires_grad_(True)
    out = grouped_fp8_gemm(x_leaf, weight_leaf, offs)
    (out * probe).sum().backward()

    tensors = {"out": out[real_rows], "grad_x": x_leaf.grad[real_rows], "grad_weight": weight_leaf.grad}
    use_ue8m0 = ue8m0_for_device(x.device)
    for name, tensor in (("weight_fwd", weight.transpose(1, 2)), ("weight_dgrad", weight)):
        fp8, scales = grouped_per_block_cast_to_fp8_triton(tensor, use_ue8m0, GROUP_ALIGNMENT)
        tensors[f"{name}_fp8"] = fp8
        tensors[f"{name}_scales"] = scales
    inputs = (("tokens", tokens), ("weight", weight), ("probe", real_probe))
    return {
        "inputs": {name: _digest(t) for name, t in inputs},
        "tensors": {name: _digest(t) for name, t in tensors.items()},
    }


def _device_header() -> dict:
    return {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "deep_gemm": getattr(deep_gemm, "__version__", "unknown"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["capture", "check"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    path = args.directory / "goldens.json"
    if args.mode == "capture":
        args.directory.mkdir(parents=True, exist_ok=True)
        goldens = {"header": _device_header(), "cases": {case.name: _record(case) for case in CASES}}
        path.write_text(json.dumps(goldens, indent=1))
        print(f"captured {len(CASES)} cases to {path}")
        return

    goldens = json.loads(path.read_text())
    failures = []
    for case in CASES:
        golden, current = goldens["cases"][case.name], _record(case)
        for name, digest in golden["inputs"].items():
            if current["inputs"][name]["sha256"] != digest["sha256"]:
                raise SystemExit(f"{case.name}: input {name} differs from the captured one, goldens do not apply")
        for name, digest in golden["tensors"].items():
            now = current["tensors"][name]
            status = "ok" if now["sha256"] == digest["sha256"] else "DIFFERS"
            if status != "ok":
                failures.append(f"{case.name}/{name}")
            print(f"{case.name:28s} {name:24s} {status:8s} sum {digest['sum']:.6e} -> {now['sum']:.6e}")
    if failures:
        raise SystemExit(f"{len(failures)} tensors differ: {failures}")
    print("all tensors bit-identical to the goldens")


if __name__ == "__main__":
    main()
