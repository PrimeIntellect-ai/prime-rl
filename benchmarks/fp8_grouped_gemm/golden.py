"""Capture or check byte-level goldens of the DeepGEMM FP8 grouped GEMM: op outputs and every intermediate cast.

Inputs are regenerated from fixed seeds and their digest is stored, so a check refuses to compare against
goldens drawn from different inputs. Only bytes the base implementation defines are compared: rows past
`offs[-1]` and the per-token casts' padding rows are left uninitialized by design.
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
    build_grouped_layout,
    grouped_per_block_cast_to_fp8_triton,
    grouped_per_channel_cast_to_fp8_rowmajor_triton,
    grouped_per_channel_cast_to_fp8_sm90_kmajor_triton,
    grouped_per_token_cast_to_fp8_triton,
    ue8m0_for_device,
)
from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm


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
    Case("gate_up-balanced-1536", [1536] * 32, 4096, 4096, "align8", 1),
    Case("down-ragged-512", _ragged(32, 512, 7 * 512, [0, 1, 127, 128, 129, 5], [0]), 2048, 4096, "align8", 2),
    Case("gate_up-ragged-1536", _ragged(32, 1536, 6 * 1536, [0, 3, 64], []), 4096, 4096, "align8", 3),
    Case(
        "down-ragged-512-align128", _ragged(32, 512, 7 * 512, [0, 1, 127, 128, 129, 5], [0]), 2048, 4096, "align128", 2
    ),
    Case("small-sub_alignment", [1, 127, 5, 64, 0, 33, 100, 7], 512, 256, "align8", 4),
    Case("small-raw-empty", [0, 896, 3, 60, 1, 40, 24, 0], 512, 256, "raw", 5),
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
    torch.manual_seed(case.seed)
    with torch.device("cuda"):
        tokens = torch.randn(sum(case.counts), case.k, dtype=torch.bfloat16)
        weight = (torch.randn(len(case.counts), case.k, case.n) * WEIGHT_STD).to(torch.bfloat16)
    if case.dispatch == "raw":
        x = torch.cat((tokens, tokens.new_zeros(RAW_TAIL_ROWS, case.k)))
        offs = torch.tensor(case.counts, device="cuda").cumsum(0).to(torch.int32)
        real_rows = torch.arange(x.shape[0], device="cuda") < tokens.shape[0]
    else:
        x, padded_counts, state = permute_for_grouped_gemm(
            tokens,
            torch.tensor(case.counts, dtype=torch.int64, device="cuda"),
            experts_per_rank=len(case.counts),
            num_ranks=1,
            alignment={"align8": 8, "align128": 128}[case.dispatch],
        )
        offs = torch.cumsum(padded_counts, dim=0, dtype=torch.int32)
        real_rows = state.permuted_indices != -1
    probe = torch.randn(x.shape[0], case.n, device="cuda", dtype=torch.bfloat16) * real_rows.unsqueeze(1)
    return x, weight, offs, probe


def _intermediate_casts(x, weight, offs, probe) -> dict[str, torch.Tensor]:
    """The casts the base wrapper feeds DeepGEMM, restricted to the rows and scales it writes."""
    layout = build_grouped_layout(offs, total_m=x.size(0))
    _, padded_total_m, grouped_layout, block_to_group, ks_tensor, starts, actual_ms, block_starts = layout
    use_ue8m0 = ue8m0_for_device(x.device)
    token_args = (padded_total_m, block_to_group, starts, actual_ms, block_starts, use_ue8m0, GROUP_ALIGNMENT)
    channel_args = (padded_total_m, block_to_group, starts, actual_ms, ks_tensor, block_starts)
    written = grouped_layout != -1

    casts = {"grouped_layout": grouped_layout, "block_to_group": block_to_group, "ks": ks_tensor}
    for name, tensor in (("x", x), ("dy", probe)):
        fp8, scales = grouped_per_token_cast_to_fp8_triton(tensor, *token_args)
        casts[f"{name}_token_fp8"] = fp8[written]
        casts[f"{name}_token_scales"] = scales[written]
    for name, tensor in (("weight_fwd", weight.transpose(1, 2)), ("weight_dgrad", weight)):
        fp8, scales = grouped_per_block_cast_to_fp8_triton(tensor, use_ue8m0, GROUP_ALIGNMENT)
        casts[f"{name}_fp8"] = fp8
        casts[f"{name}_scales"] = scales
    for name, tensor in (("x", x), ("dy", probe)):
        if torch.cuda.get_device_capability(x.device)[0] >= 10:
            fp8, scales = grouped_per_channel_cast_to_fp8_rowmajor_triton(tensor, *channel_args, True, GROUP_ALIGNMENT)
        else:
            fp8, scales = grouped_per_channel_cast_to_fp8_sm90_kmajor_triton(
                tensor, *channel_args, False, GROUP_ALIGNMENT
            )
        casts[f"{name}_channel_fp8"] = fp8
        casts[f"{name}_channel_scales"] = scales
    return casts


def _record(case: Case) -> dict:
    x, weight, offs, probe = _inputs(case)
    used_rows = int(offs[-1])
    x_leaf = x.clone().requires_grad_(True)
    weight_leaf = weight.clone().requires_grad_(True)
    out = grouped_fp8_gemm(x_leaf, weight_leaf, offs)
    (out * probe).sum().backward()

    tensors = {
        "out": out[:used_rows],
        "grad_x": x_leaf.grad[:used_rows],
        "grad_weight": weight_leaf.grad,
        **_intermediate_casts(x, weight, offs, probe),
    }
    return {
        "inputs": {name: _digest(t) for name, t in (("x", x), ("weight", weight), ("offs", offs), ("probe", probe))},
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
