"""The Mega MoE `mxfp8_qat` weights against the MXFP8 QAT recipe's: both rounding modes, fused and separate gate/up.

Mega MoE quantizes the experts in its own (interleaved) layout; decoded back to the natural layout they must equal,
bit for bit, the weights `mxfp8_qat.py` computes with (`_rounded_weight` of each expert weight, with the same step
seed, weight salt and global expert offset).

python benchmarks/llep/check_qat_weights.py
"""

import torch

from prime_rl.trainer.models.fp8_rounding import FP8RoundingState, _stream_id
from prime_rl.trainer.models.layers.mega_moe import mxfp8_qat_forward_weights, mxfp8_rounded_weights
from prime_rl.trainer.models.layers.mxfp8_qat import _rounded_weight


def main() -> None:
    torch.manual_seed(0)
    experts, hidden, inter, ep_rank = 32, 4096, 2048, 5
    offset = ep_rank * experts
    gate = torch.randn(experts, inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    up = torch.randn(experts, inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    down = torch.randn(experts, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.02
    state = FP8RoundingState(47, torch.empty((), device="cuda", dtype=torch.int64))
    state.set_step(123)
    salt = {
        name: torch.tensor(_stream_id(f"model.layers.3.mlp.experts.{name}"), device="cuda", dtype=torch.int64)
        for name in ("gate_proj", "up_proj", "gate_up_proj", "down_proj")
    }

    def recipe(weight, name, stochastic):
        # mxfp8_qat_grouped_mm rounds `weight.transpose(1, 2)` of the transposed expert weight, i.e. the expert
        # weight itself, at offset `fp8_expert_offset * N * K`
        if not stochastic:
            return _rounded_weight(weight)
        return _rounded_weight(weight, state.seed, salt[name], offset * weight.shape[1] * weight.shape[2])

    for stochastic in (False, True):
        for fused in (False, True):
            gate_up = torch.cat([gate, up], dim=1)
            names = ["gate_up_proj"] if fused else ["gate_proj", "up_proj"]
            salts = torch.stack([salt[name] for name in names + ["down_proj"]])
            rounding = (state.seed, salts, offset) if stochastic else (None, None, 0)
            mega_gate_up, mega_down = mxfp8_rounded_weights(mxfp8_qat_forward_weights(gate_up, down, *rounding))
            if fused:
                ref_gate_up = recipe(gate_up, "gate_up_proj", stochastic)
            else:
                ref_gate_up = torch.cat([recipe(gate, "gate_proj", stochastic), recipe(up, "up_proj", stochastic)], 1)
            ref_down = recipe(down, "down_proj", stochastic)
            print(
                f"stochastic={stochastic} fused_gate_up={fused}: gate_up bitwise={torch.equal(mega_gate_up, ref_gate_up)} "
                f"down bitwise={torch.equal(mega_down, ref_down)} "
                f"(rounded != nearest: {(mega_down != recipe(down, 'down_proj', False)).float().mean().item():.3f})"
            )


if __name__ == "__main__":
    main()
