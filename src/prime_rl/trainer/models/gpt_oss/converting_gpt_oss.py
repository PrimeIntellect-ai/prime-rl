"""HF <-> PrimeRL weight conversion for GPT-OSS."""

from torch import Tensor

from prime_rl.trainer.models.conversion_ops import ConvOp, Rename


class GptOssExperts(ConvOp):
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def hf_to_prime(self, state_dict: dict[str, Tensor]) -> None:
        gate_up_name = f"{self.prefix}.gate_up_proj"
        if gate_up_name not in state_dict:
            return

        gate_up = state_dict.pop(gate_up_name)
        gate_up_bias = state_dict.pop(f"{self.prefix}.gate_up_proj_bias")
        state_dict[f"{self.prefix}.gate_proj"] = gate_up[..., ::2].transpose(-2, -1)
        state_dict[f"{self.prefix}.up_proj"] = gate_up[..., 1::2].transpose(-2, -1)
        state_dict[f"{self.prefix}.gate_proj_bias"] = gate_up_bias[..., ::2]
        state_dict[f"{self.prefix}.up_proj_bias"] = gate_up_bias[..., 1::2]
        state_dict[f"{self.prefix}.down_proj"] = state_dict[f"{self.prefix}.down_proj"].transpose(-2, -1)

    def prime_to_hf(self, state_dict: dict[str, Tensor]) -> None:
        gate_name = f"{self.prefix}.gate_proj"
        if gate_name not in state_dict:
            return

        gate = state_dict.pop(gate_name).transpose(-2, -1)
        up = state_dict.pop(f"{self.prefix}.up_proj").transpose(-2, -1)
        gate_bias = state_dict.pop(f"{self.prefix}.gate_proj_bias")
        up_bias = state_dict.pop(f"{self.prefix}.up_proj_bias")
        state_dict[f"{self.prefix}.gate_up_proj"] = gate.new_empty((*gate.shape[:-1], 2 * gate.shape[-1]))
        state_dict[f"{self.prefix}.gate_up_proj"][..., ::2] = gate
        state_dict[f"{self.prefix}.gate_up_proj"][..., 1::2] = up
        state_dict[f"{self.prefix}.gate_up_proj_bias"] = gate_bias.new_empty(
            (*gate_bias.shape[:-1], 2 * gate_bias.shape[-1])
        )
        state_dict[f"{self.prefix}.gate_up_proj_bias"][..., ::2] = gate_bias
        state_dict[f"{self.prefix}.gate_up_proj_bias"][..., 1::2] = up_bias
        state_dict[f"{self.prefix}.down_proj"] = state_dict[f"{self.prefix}.down_proj"].transpose(-2, -1)


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(name.endswith("mlp.experts.gate_up_proj") for name in state_dict)


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any("mlp.experts.gate_proj" in name for name in state_dict)


def conversion_chain(config) -> list[ConvOp]:
    operations: list[ConvOp] = []
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}.mlp"
        operations.extend(
            [
                Rename(f"{prefix}.router.weight", f"{prefix}.router.gate.weight"),
                Rename(f"{prefix}.router.bias", f"{prefix}.router.gate.bias"),
                GptOssExperts(f"{prefix}.experts"),
            ]
        )
    return operations
