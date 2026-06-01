"""Weight conversion utilities for DeepSeek V3 between HuggingFace and PrimeRL formats."""

from torch import Tensor
import re
import torch

from prime_rl.utils.logger import get_logger


def get_layer_layers(state_dict: dict[str, Tensor], layer_idx: int) -> int:
    layer_keys = [
        lname for lname in state_dict.keys() if f"model.layers.{layer_idx}." in lname
    ]
    return layer_keys


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    pattern = r"model.layers.(\d+)"
    return (
        max(
            int(re.findall(pattern, lname)[0])
            for lname in state_dict.keys()
            if re.search(pattern, lname)
        )
        + 1
    )


def check_layer_has_experts(state_dict: dict[str, Tensor], layer_idx: int) -> int:
    layer_keys = get_layer_layers(state_dict, layer_idx)
    pattern = r"mlp.experts"
    expert_layers = [True for lname in layer_keys if re.search(pattern, lname)]
    return len(expert_layers) > 0


def get_num_experts(state_dict: dict[str, Tensor], layer_idx: int) -> int:
    layer_keys = get_layer_layers(state_dict, layer_idx)
    pattern = r"mlp.experts.(\d+)"
    expert_idx = [
        int(re.findall(pattern, lname)[0])
        for lname in layer_keys
        if re.search(pattern, lname)
    ]
    return max(expert_idx) + 1 if len(expert_idx) > 0 else 0


def keys_converter(layer_idx: int):
    """mapping between HF and PrimeRL keys."""

    # example
    hf_key = f"model.layer.{layer_idx}.hf_name_test"
    prime_key = f"model.layer.{layer_idx}.prime_name_test"
    yield hf_key, prime_key

    ## No experts
    hf_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
    prime_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
    yield hf_key, prime_key

    hf_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
    prime_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
    yield hf_key, prime_key

    hf_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
    prime_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
    yield hf_key, prime_key

    ## MOE Router
    hf_key = f"model.layers.{layer_idx}.mlp.gate.weight"
    prime_key = f"model.layers.{layer_idx}.mlp.router.gate.weight"
    yield hf_key, prime_key

    hf_key = f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
    prime_key = f"model.layers.{layer_idx}.mlp.expert_bias"
    yield hf_key, prime_key

    ## Shared experts
    hf_key = f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"
    prime_key = f"model.layers.{layer_idx}.mlp.shared_expert.w1"
    yield hf_key, prime_key

    hf_key = f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"
    prime_key = f"model.layers.{layer_idx}.mlp.shared_expert.w2"
    yield hf_key, prime_key

    hf_key = f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"
    prime_key = f"model.layers.{layer_idx}.mlp.shared_expert.w3"
    yield hf_key, prime_key


def convert_hf_to_prime(state_dict: dict[str, Tensor]) -> None:
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        # replace keys using mapping
        for hf_key, prime_key in keys_converter(i):
            if hf_key in state_dict:
                state_dict[prime_key] = state_dict.pop(hf_key)

        has_experts = check_layer_has_experts(state_dict, i)
        if not has_experts:
            continue

        ## Experts concatinated
        layer_name = f"model.layers.{i}.mlp.experts.gate_up_proj"
        if layer_name in state_dict:
            w_gate, w_up = state_dict.pop(layer_name).chunk(2, dim=1)
            state_dict[f"model.layers.{i}.mlp.experts.w1"] = w_gate  # gate
            state_dict[f"model.layers.{i}.mlp.experts.w3"] = w_up  # up
            hf_key = f"model.layers.{i}.mlp.experts.down_proj"
            state_dict[f"model.layers.{i}.mlp.experts.w2"] = state_dict.pop(
                hf_key
            )  # up

        num_experts = get_num_experts(state_dict, i)
        if num_experts > 0:
            # safetensors format might have experts splitted
            dim, moe_dim = state_dict[
                f"model.layers.{i}.mlp.experts.{i}.down_proj.weight"
            ].shape
            dtype = state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].dtype

            w1 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)  # Gate
            w2 = torch.empty((num_experts, dim, moe_dim), dtype=dtype)  # Down
            w3 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)  # Up

            for j in range(num_experts):
                w1[j].copy_(
                    state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"]
                )
                w2[j].copy_(
                    state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"]
                )
                w3[j].copy_(
                    state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"]
                )

                del state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"]
                del state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"]
                del state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"]

            state_dict[f"model.layers.{i}.mlp.experts.w1"] = w1  # gate
            state_dict[f"model.layers.{i}.mlp.experts.w2"] = w2  # down
            state_dict[f"model.layers.{i}.mlp.experts.w3"] = w3  # up

    pass


def convert_prime_to_hf(state_dict: dict[str, Tensor]) -> None:

    concat_experts = True
    num_layers = get_max_layer_num(state_dict)

    for i in range(num_layers):
        # replace keys using mapping
        for hf_key, prime_key in keys_converter(i):
            if prime_key in state_dict:
                state_dict[hf_key] = state_dict.pop(prime_key)

        # check layer has experts
        w1_key = f"model.layers.{i}.mlp.experts.w1"
        has_experts = w1_key in state_dict
        if not has_experts:
            continue

        w1 = state_dict.pop(f"model.layers.{i}.mlp.experts.w1")
        w2 = state_dict.pop(f"model.layers.{i}.mlp.experts.w2")
        w3 = state_dict.pop(f"model.layers.{i}.mlp.experts.w3")

        num_experts = w1.shape[0]

        # bias is registred in HF Deepseek, but not used in the calculations
        state_dict[f"model.layers.{i}.mlp.gate.e_score_correction_bias"] = torch.zeros(
            num_experts
        )

        if concat_experts:
            # safetensors format when experts concatinated
            hf_key = f"model.layers.{i}.mlp.experts.down_proj"
            state_dict[hf_key] = w2

            hf_key = f"model.layers.{i}.mlp.experts.gate_up_proj"
            state_dict[hf_key] = torch.cat((w1, w3), dim=1)

        else:
            # safetensors format might have experts splitted
            for j in range(num_experts):
                state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"] = w1[j]
                state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"] = w2[j]
                state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"] = w3[j]


def convert_hf_layer_to_prime(state_dict: dict[str, Tensor], layer_idx: int) -> None:
    raise NotImplementedError()


def convert_prime_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int) -> None:
    raise NotImplementedError()
