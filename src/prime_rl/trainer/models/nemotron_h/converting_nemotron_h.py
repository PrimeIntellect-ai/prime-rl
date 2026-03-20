"""Weight conversion between HuggingFace and PrimeRL formats for NemotronH.

HF NemotronH uses a unified `mixer` attribute for all layer types:
  - Mamba layers: layers.{i}.mixer.{in_proj, conv1d, ...}
  - Attention layers: layers.{i}.mixer.{q_proj, k_proj, v_proj, o_proj}
  - MoE layers: layers.{i}.mixer.{gate, experts, shared_experts, fc1_latent_proj, fc2_latent_proj}

PrimeRL separates these into distinct namespaces:
  - Mamba layers: layers.{i}.mamba.*
  - Attention layers: layers.{i}.self_attn.*
  - MoE layers: layers.{i}.mlp.{router, experts, shared_expert, fc1_latent_proj, fc2_latent_proj}

Global renames:
  - HF: model.embeddings.weight <-> PrimeRL: model.embed_tokens.weight
  - HF: model.norm_f.weight <-> PrimeRL: model.norm.weight
"""

import torch
from torch import Tensor


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    return max(int(k.split(".")[2]) for k in state_dict if k.startswith("model.layers.")) + 1


def _rename_keys(state_dict: dict[str, Tensor], old_prefix: str, new_prefix: str):
    """Rename all keys matching old_prefix to new_prefix in-place."""
    keys_to_rename = [k for k in state_dict if k.startswith(old_prefix)]
    for key in keys_to_rename:
        new_key = new_prefix + key[len(old_prefix) :]
        state_dict[new_key] = state_dict.pop(key)


def convert_hf_layer_to_prime(state_dict: dict[str, Tensor], layer_idx: int, layer_type: str):
    """Convert a single layer from HF to PrimeRL format in-place."""
    prefix = f"model.layers.{layer_idx}."

    if layer_type == "moe":
        _convert_hf_moe_layer_to_prime(state_dict, prefix)
    elif layer_type == "attention":
        _convert_hf_attention_layer_to_prime(state_dict, prefix)
    elif layer_type == "mamba":
        _rename_keys(state_dict, f"{prefix}mixer.", f"{prefix}mamba.")


def _convert_hf_moe_layer_to_prime(state_dict: dict[str, Tensor], prefix: str):
    """Convert MoE layer: mixer.gate -> mlp.router, mixer.experts -> mlp.experts, etc."""
    mixer = f"{prefix}mixer."
    mlp = f"{prefix}mlp."

    # Router: gate.weight -> router.gate (nn.Parameter), gate.e_score_correction_bias -> router.e_score_correction_bias
    if f"{mixer}gate.weight" in state_dict:
        state_dict[f"{mlp}router.gate"] = state_dict.pop(f"{mixer}gate.weight")
    if f"{mixer}gate.e_score_correction_bias" in state_dict:
        state_dict[f"{mlp}router.e_score_correction_bias"] = state_dict.pop(f"{mixer}gate.e_score_correction_bias")

    # Experts: up_proj -> w1, down_proj -> w2 (already 3D tensors in HF)
    if f"{mixer}experts.up_proj" in state_dict:
        state_dict[f"{mlp}experts.w1"] = state_dict.pop(f"{mixer}experts.up_proj")
    if f"{mixer}experts.down_proj" in state_dict:
        state_dict[f"{mlp}experts.w2"] = state_dict.pop(f"{mixer}experts.down_proj")
    # Dummy w3 required by @expert_parallel decorator compatibility
    device = state_dict[f"{mlp}experts.w1"].device if f"{mlp}experts.w1" in state_dict else "cpu"
    state_dict[f"{mlp}experts.w3"] = torch.empty(0, device=device)

    # Shared expert
    _rename_keys(state_dict, f"{mixer}shared_experts.", f"{mlp}shared_expert.")

    # Latent projections
    _rename_keys(state_dict, f"{mixer}fc1_latent_proj.", f"{mlp}fc1_latent_proj.")
    _rename_keys(state_dict, f"{mixer}fc2_latent_proj.", f"{mlp}fc2_latent_proj.")


def _convert_hf_attention_layer_to_prime(state_dict: dict[str, Tensor], prefix: str):
    """Convert attention layer: mixer.{q,k,v,o}_proj -> self_attn.{q,k,v,o}_proj."""
    _rename_keys(state_dict, f"{prefix}mixer.", f"{prefix}self_attn.")


def convert_prime_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int, layer_type: str):
    """Convert a single layer from PrimeRL to HF format in-place."""
    prefix = f"model.layers.{layer_idx}."

    if layer_type == "moe":
        _convert_prime_moe_layer_to_hf(state_dict, prefix)
    elif layer_type == "attention":
        _rename_keys(state_dict, f"{prefix}self_attn.", f"{prefix}mixer.")
    elif layer_type == "mamba":
        _rename_keys(state_dict, f"{prefix}mamba.", f"{prefix}mixer.")


def _convert_prime_moe_layer_to_hf(state_dict: dict[str, Tensor], prefix: str):
    """Convert MoE layer back to HF format."""
    mlp = f"{prefix}mlp."
    mixer = f"{prefix}mixer."

    # Router
    if f"{mlp}router.gate" in state_dict:
        state_dict[f"{mixer}gate.weight"] = state_dict.pop(f"{mlp}router.gate")
    if f"{mlp}router.e_score_correction_bias" in state_dict:
        state_dict[f"{mixer}gate.e_score_correction_bias"] = state_dict.pop(f"{mlp}router.e_score_correction_bias")

    # Experts
    if f"{mlp}experts.w1" in state_dict:
        state_dict[f"{mixer}experts.up_proj"] = state_dict.pop(f"{mlp}experts.w1")
    if f"{mlp}experts.w2" in state_dict:
        state_dict[f"{mixer}experts.down_proj"] = state_dict.pop(f"{mlp}experts.w2")
    # Remove dummy w3 (not present in HF format)
    state_dict.pop(f"{mlp}experts.w3", None)

    # Shared expert
    _rename_keys(state_dict, f"{mlp}shared_expert.", f"{mixer}shared_experts.")

    # Latent projections
    _rename_keys(state_dict, f"{mlp}fc1_latent_proj.", f"{mixer}fc1_latent_proj.")
    _rename_keys(state_dict, f"{mlp}fc2_latent_proj.", f"{mixer}fc2_latent_proj.")


def convert_hf_to_prime(state_dict: dict[str, Tensor], layers_block_type: list[str]):
    """Convert full model from HF to PrimeRL format in-place."""
    # Global renames
    if "model.embeddings.weight" in state_dict:
        state_dict["model.embed_tokens.weight"] = state_dict.pop("model.embeddings.weight")
    if "model.norm_f.weight" in state_dict:
        state_dict["model.norm.weight"] = state_dict.pop("model.norm_f.weight")

    for i, layer_type in enumerate(layers_block_type):
        convert_hf_layer_to_prime(state_dict, i, layer_type)


def convert_prime_to_hf(state_dict: dict[str, Tensor], layers_block_type: list[str]):
    """Convert full model from PrimeRL to HF format in-place."""
    # Global renames
    if "model.embed_tokens.weight" in state_dict:
        state_dict["model.embeddings.weight"] = state_dict.pop("model.embed_tokens.weight")
    if "model.norm.weight" in state_dict:
        state_dict["model.norm_f.weight"] = state_dict.pop("model.norm.weight")

    for i, layer_type in enumerate(layers_block_type):
        convert_prime_layer_to_hf(state_dict, i, layer_type)
