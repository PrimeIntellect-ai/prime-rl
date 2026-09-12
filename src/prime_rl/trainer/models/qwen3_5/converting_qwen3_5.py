from torch import Tensor

from prime_rl.trainer.models.conversion_ops import ConvOp, Drop, Rename, SplitConcat
from prime_rl.trainer.models.qwen3_5.configuration_qwen3_5 import Qwen3_5MoeTextConfig


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(name == "mtp.fc.weight" or name.endswith("mlp.experts.gate_up_proj") for name in state_dict)


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    if any(name.endswith("mlp.experts.gate_proj") for name in state_dict):
        return True
    has_qwen_layers = any(name.endswith("linear_attn.A_log") for name in state_dict)
    has_fused_experts = any(name.endswith("mlp.experts.gate_up_proj") for name in state_dict)
    return has_qwen_layers and not has_fused_experts


def conversion_chain(config) -> list[ConvOp]:
    operations: list[ConvOp] = [Drop("mtp.", is_prefix=True)]
    text_config = getattr(config, "text_config", config)
    if not isinstance(text_config, Qwen3_5MoeTextConfig):
        return operations

    model_prefix = "model.language_model" if hasattr(config, "vision_config") else "model"
    for layer_index in range(text_config.num_hidden_layers):
        prefix = f"{model_prefix}.layers.{layer_index}.mlp"
        operations.extend(
            [
                Rename(f"{prefix}.gate.weight", f"{prefix}.router.gate.weight"),
                Rename(
                    f"{prefix}.shared_expert_gate.weight",
                    f"{prefix}.shared_expert.output_gate.weight",
                ),
                SplitConcat(
                    combined=f"{prefix}.experts.gate_up_proj",
                    parts=[
                        (f"{prefix}.experts.gate_proj", None),
                        (f"{prefix}.experts.up_proj", None),
                    ],
                    dim=1,
                ),
            ]
        )
    return operations


__all__ = ["conversion_chain", "is_hf_state_dict", "is_prime_state_dict"]
