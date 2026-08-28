from torch import Tensor

from prime_rl.trainer.models.conversion_ops import ConvOp, Drop, PrefixRename, Rename, Stack
from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return "backbone.embeddings.weight" in state_dict or any(".mixer." in name for name in state_dict)


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return "model.embed_tokens.weight" in state_dict or any(
        namespace in name for name in state_dict for namespace in (".mamba.", ".self_attn.", ".mlp.router.")
    )


def conversion_chain(config: NemotronHConfig) -> list[ConvOp]:
    ops: list[ConvOp] = [
        PrefixRename("backbone.", "model."),
        Drop("mtp.", is_prefix=True),
        Rename("model.embeddings.weight", "model.embed_tokens.weight"),
        Rename("model.norm_f.weight", "model.norm.weight"),
    ]

    for layer_idx, layer_type in enumerate(config.layer_types):
        prefix = f"model.layers.{layer_idx}"
        if layer_type == "mamba":
            ops.append(PrefixRename(f"{prefix}.mixer.", f"{prefix}.mamba."))
        elif layer_type == "attention":
            ops.append(PrefixRename(f"{prefix}.mixer.", f"{prefix}.self_attn."))
        elif layer_type == "moe":
            ops.extend(
                [
                    Rename(f"{prefix}.mixer.gate.weight", f"{prefix}.mlp.router.gate.weight"),
                    Rename(
                        f"{prefix}.mixer.gate.e_score_correction_bias",
                        f"{prefix}.mlp.router.selection_bias",
                    ),
                    Stack(
                        stacked=f"{prefix}.mlp.experts.up_proj",
                        item=f"{prefix}.mixer.experts.{{e}}.up_proj.weight",
                    ),
                    Stack(
                        stacked=f"{prefix}.mlp.experts.down_proj",
                        item=f"{prefix}.mixer.experts.{{e}}.down_proj.weight",
                    ),
                    PrefixRename(f"{prefix}.mixer.shared_experts.", f"{prefix}.mlp.shared_expert."),
                    PrefixRename(f"{prefix}.mixer.fc1_latent_proj.", f"{prefix}.mlp.fc1_latent_proj."),
                    PrefixRename(f"{prefix}.mixer.fc2_latent_proj.", f"{prefix}.mlp.fc2_latent_proj."),
                ]
            )
        else:
            raise ValueError(f"Unsupported Nemotron-H layer type: {layer_type}")
    return ops


__all__ = ["conversion_chain", "is_hf_state_dict", "is_prime_state_dict"]
